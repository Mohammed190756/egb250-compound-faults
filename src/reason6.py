"""Physics reasoning layer applied to the main models.
Symptoms (window level, record = median of 9 windows), computed from the linear crank-angle profiles and envelope spectra,
for real recordings and, through the same combining rule (M-b), for combined rows. Robust z-scores against recordings whose
linked components are healthy; explaining away of the B1 signature; weighted abduction over the 164 labels.
Weights selected on the validation set (real + combined rows, seed-0 tuning-phase posteriors)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from data6 import *
ORDS = np.load((WORK) + '/dataset.npz', allow_pickle=True)['orders']
SYM = ['dP2', 'dP3', 'puls1', 'puls2', 'puls3', 'env_BPFO', 'env_BPFI', 'env_FTF']
PNEU = SYM[:5]
LINK = {'dP2': ['1S_IVb', '1S_DV', '2S_IV', '2S_DV'], 'dP3': ['2S_DV'], 'puls1': ['1S_IVb', '1S_DV'], 'puls2': ['1S_IVb', '1S_DV'], 'puls3': ['2S_DV'],
        'env_BPFO': ['B1'], 'env_BPFI': ['B1'], 'env_FTF': ['B1']}
RULES = {('dP2', 'up'): ['2S_IV', '2S_DV'], ('dP2', 'down'): ['1S_IVb', '1S_DV'], ('dP3', 'down'): ['2S_DV'], ('puls3', 'up'): ['2S_DV'],
         ('puls1', 'up'): ['1S_IVb'], ('puls1', 'down'): ['1S_DV'], ('puls2', 'up'): ['1S_DV'], ('puls2', 'down'): ['1S_IVb'],
         ('env_BPFO', 'up'): ['B1'], ('env_BPFI', 'up'): ['B1'], ('env_FTF', 'up'): ['B1']}
SIGMA = {c: sorted({k for (k, d), cs in RULES.items() if c in cs}) for c in COMPS}   # symptoms that can indicate component c
def symptoms(P, E, ch):
    """P: (n, 9, 13, 72) linear profiles; E: (n, 9, 2, 300) linear envelope spectra (A3, A4). Returns (n, 8) record medians."""
    i = {c: ch.index(c) for c in ('Pr1', 'Pr2', 'Pr3', 'Pr4')}; m = lambda c: P[:, :, i[c]].mean(-1); s = lambda c: P[:, :, i[c]].std(-1)
    band = lambda o: E[:, :, 0, (ORDS > o - 0.1) & (ORDS < o + 0.1)].max(-1)
    W = np.stack([m('Pr2') / m('Pr4'), m('Pr3') / m('Pr4'), np.log(s('Pr1')), np.log(s('Pr2')), np.log(s('Pr3')),
                  np.log(band(7.15)), np.log(band(9.85)), np.log(band(0.42))], -1)
    return np.nanmedian(W, axis=1)
def sym_sets(stat_mask, rows_real, items):
    """Symptoms of real recordings rows_real and of combined rows items (M-b), with the builder of stat_mask."""
    B = Builder(stat_mask); vr = symptoms(B.P[rows_real], B.E[rows_real], B.ch)
    if not items: return B, vr, np.zeros((0, len(SYM)))
    I = np.array([a for a, _, _ in items]); J = np.array([b for _, b, _ in items]); Pm, Em = B.merge_many(I, J, 'b')
    return B, vr, symptoms(Pm, Em, B.ch)
class Ref:
    """Centres, scales and B1 signatures from single-fault and healthy recordings of a training pool."""
    def __init__(s, B, pool_mask):
        idx = np.flatnonzero(pool_mask & (meta.role != 'combo').values); V = symptoms(B.P[idx], B.E[idx], B.ch); y = Y[idx]
        s.c = np.zeros(len(SYM)); s.s = np.ones(len(SYM))
        for k, n in enumerate(SYM):
            ok = np.all(y[:, [COMPS.index(c) for c in LINK[n]]] == 0, 1); v = V[ok, k]; s.c[k] = np.median(v); s.s[k] = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9
        b1 = COMPS.index('B1'); only = np.all(np.delete(y, b1, 1) == 0, 1)
        s.sig = {m: np.median(V[only & (y[:, b1] == m)], 0) for m in range(1, NMODE[b1])}
        s.V, s.y = V, y
    def z(s, v, b1_mode):
        v = v.copy()
        for n, m in enumerate(b1_mode):   # explaining away: remove the known B1 signature from the pneumatic symptoms
            if m > 0: v[n, :5] = v[n, :5] - (s.sig[m][:5] - s.c[:5])
        return (v - s.c) / s.s
HYP = LABEL_Y   # the 164 admissible hypotheses
def counts(Z, tau):
    """N_u and N_s for every record and hypothesis (n, 164)."""
    q = np.where(Z > tau, 1, np.where(Z < -tau, -1, 0)); n = len(Z); Nu = np.zeros((n, len(HYP))); Ns = np.zeros((n, len(HYP)))
    faulty = HYP > 0
    for k, nm in enumerate(SYM):
        for d, sgn in (('up', 1), ('down', -1)):
            cs = RULES.get((nm, d), [])
            ab = q[:, k] == sgn
            if not ab.any(): continue
            expl = np.any(faulty[:, [COMPS.index(c) for c in cs]], 1) if cs else np.zeros(len(HYP), bool)
            Nu += ab[:, None] & ~expl[None, :]
    for j, c in enumerate(COMPS):
        ks = [SYM.index(k) for k in SIGMA[c]]; silent = np.all(q[:, ks] == 0, 1)
        Ns += silent[:, None] & faulty[None, :, j]
    return Nu, Ns
def loglik(prs):
    return sum(np.log(np.maximum(prs[c][:, HYP[:, c]], 1e-9)) for c in range(5))
def diagnose(prs, Z, w, tau=None, cache=None):
    L = loglik(prs)
    if w is None: return HYP[L.argmax(1)]
    wu, ws, wc, tau = w; Nu, Ns = cache[tau] if cache else counts(Z, tau)
    S = L - wu * Nu - ws * Ns - wc * (HYP > 0).sum(1)[None]
    return HYP[S.argmax(1)]
GRID = [(wu, ws, wc, tau) for tau in (2, 3, 4) for wu in (0.5, 1, 2, 4, 8) for ws in (0, 0.5, 1, 2) for wc in (0, 0.5, 1, 2)]
def score(P, T, G):
    ex = (P == T).all(1); return float(np.mean([ex[G == g].mean() for g in np.unique(G)]))
if __name__ == '__main__':
    R0 = (RES) + '/'; od = R0 + 'reason/'; os.makedirs(od, exist_ok=True)
    MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']
    # validation (tuning phase): stats from TRS, rows = real VAL + combined R13-R15
    vitems = pair_items((13, 14, 15)); vr = np.flatnonzero(VAL)
    Bt, sv, sp = sym_sets(TRS, vr, vitems); refT = Ref(Bt, TRS); Vsym = np.concatenate([sv, sp])
    VY = np.concatenate([Y[vr], np.array([label_vec(it[2]) for it in vitems])]); VG = np.array([0 if (y == 0).all() else 1 for y in Y[vr]] + [2] * len(vitems))
    # test: stats from POOL
    tidx = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); Bf, st, _ = sym_sets(POOL, tidx, []); refF = Ref(Bf, POOL); TY = Y[tidx]
    TF = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120)
    json.dump(dict(signatures={f'{name_of(y)}': list(np.round(refF.z(v[None], [0])[0], 3)) for v, y in zip(refF.V, refF.y)}), open(od + 'signatures_raw.json', 'w'))
    rows, sel = [], {}
    for m in MODELS:
        R = pickle.load(open(R0 + f'final_{m}.pkl', 'rb'))
        pv = R[0]['val']; b1v = pv[4].argmax(1); Zv = refT.z(Vsym, b1v); cache = {t: counts(Zv, t) for t in (2, 3, 4)}
        base = score(diagnose(pv, Zv, None), VY, VG); res = [(score(diagnose(pv, Zv, w, cache=cache), VY, VG), -sum(w[:3]), w) for w in GRID]
        best = max(res, key=lambda r: (r[0], r[1])); sel[m] = dict(w=best[2], val_with_R=best[0], val_without=base)
        print(m, 'val', round(base, 3), '->', round(best[0], 3), best[2], flush=True)
        for s in sorted(R):
            pt = R[s]['test']; Zt = refF.z(st, pt[4].argmax(1))
            for lab, w in (('model', None), ('+ R', best[2])):
                P = diagnose(pt, Zt, w)
                for f in ('valve-bearing', 'valve-valve'):
                    k = TF == f; rows.append(dict(model=m, seed=s, system=lab, family=f, label=float((P[k] == TY[k]).all(1).mean()), loc=float(((P[k] > 0) == (TY[k] > 0)).all(1).mean())))
                pickle.dump(P, open(od + f'pred_{m}_{s}_{lab.replace(" ", "").replace("+", "R")}.pkl', 'wb'))
    D = pd.DataFrame(rows); D.to_csv(od + 'reason_eval.csv', index=False); json.dump(sel, open(od + 'selected.json', 'w'), indent=1)
    # symptom signatures of single faults (z-scores, POOL reference, no explaining away) for the figure
    sig = {}
    for lab in ['healthy'] + CONDS:
        k = np.array([name_of(y) == lab for y in refF.y]); sig[lab] = list(np.median((refF.V[k] - refF.c) / refF.s, 0))
    json.dump(sig, open(od + 'signatures.json', 'w'), indent=1)
    pd.set_option('display.width', 200); print(D.groupby(['family', 'model', 'system'])[['label', 'loc']].mean().round(3).unstack('system').to_string())
