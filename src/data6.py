"""Data: actual single-fault recordings combined into all 144 admissible two-fault conditions.
Merge rule (physical superposition, linear domain, per window and crank-angle bin):
  pressure channels      P_c = P_A + P_B - H
  energy channels        R_c = sqrt(max(R_A^2 + R_B^2 - H2, min(R_A^2, R_B^2)))  (independent sources add energy;
                         never below the quieter of the two real recordings: avoids impossible negative energy)
  envelope spectra       E_c = sqrt(max(E_A^2 + E_B^2 - H2, min(E_A^2, E_B^2)))
H, H2: mean healthy profile / energy (Bearing P1 windows allowed by the statistics mask). Pairs matched by repetition.
Label column: 164 names written like the test data."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
from data4 import meta, POOL, VAL, TRS, HEALTHY, FAM, Y, COMPS, NMODE, align, SETUP
P2 = (WORK)
PCH = ['Pr1', 'Pr2', 'Pr3', 'Pr4']
def name_of(y):
    s = '+'.join(f'{c}_{int(m)}' for c, m in zip(COMPS, y) if m > 0); return s or 'healthy'
SINGLE = POOL & (meta.role == 'single').values
CONDS = sorted({name_of(Y[i]) for i in np.flatnonzero(SINGLE)}, key=lambda s: (COMPS.index(s.rsplit('_', 1)[0]), s))
COMP_OF = {c: c.rsplit('_', 1)[0] for c in CONDS}
PAIRS = [tuple(sorted(p, key=lambda s: COMPS.index(COMP_OF[s]))) for p in itertools.combinations(CONDS, 2) if COMP_OF[p[0]] != COMP_OF[p[1]]]
LABELS = ['healthy'] + CONDS + [f'{a}+{b}' for a, b in PAIRS]
def label_vec(lab):
    y = np.zeros(5, int)
    if lab != 'healthy':
        for part in lab.split('+'): c, m = part.rsplit('_', 1); y[COMPS.index(c)] = int(m)
    return y
LABEL_Y = np.array([label_vec(l) for l in LABELS])
REC = {(name_of(Y[i]), int(meta.R[i])): i for i in np.flatnonzero(SINGLE)}
def pair_items(reps):
    return [(REC[(a, r)], REC[(b, r)], f'{a}+{b}') for a, b in PAIRS for r in reps if (a, r) in REC and (b, r) in REC]
_RAW = {}
def raw(stat_mask):
    key = int(stat_mask.sum())
    if key not in _RAW:
        Pa, shift, ch = align(HEALTHY & stat_mask)
        env = np.load(f'{P2}/dataset.npz', allow_pickle=True)['env'][:, :, 2:4, 0, :].astype(np.float64)
        _RAW[key] = (Pa.astype(np.float64), env, ch)
    return _RAW[key]
class Builder:
    def __init__(s, stat_mask):
        s.P, s.E, s.ch = raw(stat_mask); s.ip = [s.ch.index(c) for c in PCH]; s.idn = [i for i in range(13) if i not in s.ip]
        h = np.flatnonzero(HEALTHY & stat_mask)
        if SETUP == 'C2':   # refer every valve-campaign recording to its own healthy session, then shift it to the Bearing P1 reference
            s.P, s.E = s.P.copy(), s.E.copy(); camp = meta.campaign.values
            hv = np.flatnonzero(HEALTHY & stat_mask & (camp == 'Valve')); hb = np.flatnonzero(HEALTHY & stat_mask & (camp == 'Bearing')); v = camp == 'Valve'
            ip, idn = s.ip, s.idn; Pv, Pb = np.nanmean(s.P[hv], axis=(0, 1)), np.nanmean(s.P[hb], axis=(0, 1))
            Qv, Qb = np.nanmean(s.P[hv] ** 2, axis=(0, 1)), np.nanmean(s.P[hb] ** 2, axis=(0, 1)); Ev, Eb = np.nanmean(s.E[hv] ** 2, axis=(0, 1)), np.nanmean(s.E[hb] ** 2, axis=(0, 1))
            X = s.P[v]; X[:, :, ip] += Pb[ip] - Pv[ip]; X[:, :, idn] = np.sqrt(np.maximum(X[:, :, idn] ** 2 + Qb[idn] - Qv[idn], 1e-3 * Qb[idn])); s.P[v] = X
            s.E[v] = np.sqrt(np.maximum(s.E[v] ** 2 + Eb - Ev, 1e-3 * Eb))
        s.Hp = np.nanmean(s.P[h], axis=(0, 1)); s.H2 = np.nanmean(s.P[h] ** 2, axis=(0, 1)); s.HE2 = np.nanmean(s.E[h] ** 2, axis=(0, 1))
        X, E = s.transform(s.P[h], s.E[h]); s.mx = np.nanmean(X, axis=(0, 1)); s.sx = np.nanstd(X, axis=(0, 1, 3), keepdims=True)[0] + 1e-6
        s.me = np.nanmean(E, axis=(0, 1)); s.se = np.nanstd(E, axis=(0, 1, 3), keepdims=True)[0] + 1e-6
    def merge(s, i, j):
        Pa, Pb = s.P[i], s.P[j]; out = np.empty_like(Pa)
        out[:, s.ip] = Pa[:, s.ip] + Pb[:, s.ip] - s.Hp[s.ip]
        a2, b2 = Pa[:, s.idn] ** 2, Pb[:, s.idn] ** 2
        out[:, s.idn] = np.sqrt(np.maximum(a2 + b2 - s.H2[s.idn], np.minimum(a2, b2)))
        ea, eb = s.E[i] ** 2, s.E[j] ** 2
        Ec = np.sqrt(np.maximum(ea + eb - s.HE2, np.minimum(ea, eb))); return out, Ec
    def transform(s, P, E):
        X = P.copy(); X[..., s.idn, :] = np.log(np.maximum(X[..., s.idn, :], 1e-6)); return X, np.log(np.maximum(E, 1e-9))
    def norm(s, X, E):
        X = np.nan_to_num((X - s.mx) / s.sx).clip(-30, 30).astype(np.float32); E = np.nan_to_num((E - s.me) / s.se).clip(-30, 30).astype(np.float32); return X, E
    def real(s, idx):
        X, E = s.transform(s.P[idx], s.E[idx]); return s.norm(X, E)
    def merge_many(s, I, J, rule='b', rng=None):
        """Combine single-fault rows I and J (linear domain). rule b: superposition; c: dominant fault; d: b with severities U(0.5,1)."""
        Pa, Pb = s.P[I], s.P[J]; Ea, Eb = s.E[I], s.E[J]; out = np.empty_like(Pa); ip, idn = s.ip, s.idn
        if rule == 'd': aA = rng.uniform(0.5, 1, (len(I), 1, 1, 1)); aB = rng.uniform(0.5, 1, (len(I), 1, 1, 1))
        else: aA = aB = 1.0
        dA, dB = Pa[:, :, ip] - s.Hp[ip], Pb[:, :, ip] - s.Hp[ip]
        a2, b2 = Pa[:, :, idn] ** 2 - s.H2[idn], Pb[:, :, idn] ** 2 - s.H2[idn]
        ea, eb = Ea ** 2 - s.HE2, Eb ** 2 - s.HE2
        if rule == 'c':
            out[:, :, ip] = s.Hp[ip] + np.where(np.abs(dA) >= np.abs(dB), dA, dB)
            out[:, :, idn] = np.sqrt(np.maximum(s.H2[idn] + np.where(np.abs(a2) >= np.abs(b2), a2, b2), 0))
            Ec = np.sqrt(np.maximum(s.HE2 + np.where(np.abs(ea) >= np.abs(eb), ea, eb), 0)); return out, Ec
        out[:, :, ip] = s.Hp[ip] + aA * dA + aB * dB
        sa, sb = s.H2[idn] + aA * a2, s.H2[idn] + aB * b2          # scaled single energies (always >= 0 for alpha <= 1)
        out[:, :, idn] = np.sqrt(np.maximum(s.H2[idn] + aA * a2 + aB * b2, np.minimum(sa, sb)))
        ta, tb = s.HE2 + aA * ea, s.HE2 + aB * eb
        Ec = np.sqrt(np.maximum(s.HE2 + aA * ea + aB * eb, np.minimum(ta, tb))); return out, Ec
    def pairs(s, items, rule='b', rng=None):
        I = np.array([it[0] for it in items]); J = np.array([it[1] for it in items])
        if rule == 'a':   # deviation sum in the log domain: T(A) + T(B) - mean T(H)
            XA, EA = s.transform(s.P[I], s.E[I]); XB, EB = s.transform(s.P[J], s.E[J]); return s.norm(XA + XB - s.mx, EA + EB - s.me)
        P, E = s.merge_many(I, J, rule, rng); X, E = s.transform(P, E); return s.norm(X, E)
ORD = np.load(f'{P2}/dataset.npz', allow_pickle=True)['orders']
EOI = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 7.15, 9.85, 0.42, 8.85]
def summary(X, E):
    F = [X.mean(-1), X.std(-1), X.min(-1), X.max(-1)]
    D = np.abs(np.fft.rfft(X, axis=-1))[..., 1:7] / X.shape[-1]
    F = np.concatenate([f[..., None] for f in F] + [D], -1).reshape(*X.shape[:-2], -1)
    G = np.stack([E[..., (ORD > o - 0.1) & (ORD < o + 0.1)].max(-1) for o in EOI], -1).reshape(*E.shape[:-2], -1)
    return np.concatenate([F, G], -1).astype(np.float32)
def build(stat_mask, real_mask, reps, rule='b', seed=0):
    B = Builder(stat_mask); ri = np.flatnonzero(real_mask); items = pair_items(reps)
    Xr, Er = B.real(ri); Xp, Ep = B.pairs(items, rule, np.random.default_rng(seed))
    lab = [name_of(Y[i]) for i in ri] + [it[2] for it in items]
    grp = [0 if l == 'healthy' else 1 for l in lab[:len(ri)]] + [2] * len(items)
    return dict(X=np.concatenate([Xr, Xp]), E=np.concatenate([Er, Ep]), label=np.array(lab), y=np.array([label_vec(l) for l in lab]), group=np.array(grp), _B=B, _items=items, _n0=len(ri))
def test_set(stat_mask):
    B = Builder(stat_mask); idx = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); X, E = B.real(idx)
    return dict(X=X, E=E, idx=idx, label=np.array([name_of(Y[i]) for i in idx]), y=Y[idx], family=np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120), group=np.full(300, 2))
if __name__ == '__main__':
    print(len(CONDS), len(PAIRS), len(LABELS), len(pair_items(range(1, 13))), len(pair_items((13, 14, 15))), len(pair_items(range(1, 16))))
