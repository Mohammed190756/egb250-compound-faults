"""Physics reasoning layer as Bayesian evidence combination (applied after the learning models).
Symptoms v (8 per recording) as in reason6. Generative model for hypothesis H (one of the 164 labels):
  mu(H) = c + sum_{(comp, m) in H} delta(comp, m),  delta = median symptoms of single fault (comp, m) - healthy centre c
  v_k | H ~ Laplace(mu_k(H), b_k),  b_k^2 = s_rep,k^2 + s_ses,k^2
  s_rep: pooled within-condition repetition scale (1.4826 MAD) of single faults; s_ses: half the difference between the two healthy sessions
Score  S(H) = sum_c log pbar_c(m_c^H) + lam * sum_k log Laplace(v_k; mu_k(H), b_k);  lam = 1 fixed a priori; lam tuned on validation reported as a variant.
Also: physics-rule consistency of single-fault symptoms, bearing shift vs session size, help/harm counts with exact McNemar tests."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, pickle, numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, (SRC))
from data6 import *
from reason6 import symptoms, sym_sets, SYM, RULES, LINK
R0 = (RES) + '/'; OD = R0 + 'bayes/'; os.makedirs(OD, exist_ok=True)
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']
HYP = LABEL_Y
def mad(x, axis=0): return 1.4826 * np.median(np.abs(x - np.median(x, axis)), axis)
class Sig:
    def __init__(s, B, pool_mask):
        idx = np.flatnonzero(pool_mask & (meta.role != 'combo').values); V = symptoms(B.P[idx], B.E[idx], B.ch); s.V = V; s.lab = np.array([name_of(Y[i]) for i in idx])
        s.c = np.median(V[s.lab == 'healthy'], 0); s.delta = {l: np.median(V[s.lab == l], 0) - s.c for l in CONDS}
        res = np.concatenate([V[s.lab == l] - np.median(V[s.lab == l], 0) for l in ['healthy'] + CONDS]); s.s_rep = mad(res)
        hv = np.flatnonzero(((meta.campaign == 'Valve') & (meta.role == 'healthy')).values); hb = np.flatnonzero(((meta.campaign == 'Bearing') & (meta.role == 'healthy')).values)
        Vv = symptoms(B.P[hv], B.E[hv], B.ch); Vb = symptoms(B.P[hb], B.E[hb], B.ch); s.s_ses = np.abs(np.median(Vv, 0) - np.median(Vb, 0)) / 2
        s.b = np.sqrt(s.s_rep ** 2 + s.s_ses ** 2) + 1e-9
        s.MU = np.stack([s.c + sum((s.delta[p] for p in (l.split('+') if l != 'healthy' else [])), np.zeros(len(SYM))) for l in LABELS])
    def loglik(s, v):   # (n, 164)
        return -(np.abs(v[:, None, :] - s.MU[None]) / s.b).sum(-1) - np.log(2 * s.b).sum()
def model_ll(prs): return sum(np.log(np.maximum(prs[c][:, HYP[:, c]], 1e-9)) for c in range(5))
def decide(prs, L, lam): return HYP[(model_ll(prs) + lam * L).argmax(1)]
def score(P, T, G): ex = (P == T).all(1); return float(np.mean([ex[G == g].mean() for g in np.unique(G)]))
if __name__ == '__main__':
    vitems = pair_items((13, 14, 15)); vr = np.flatnonzero(VAL)
    Bt, sv, sp = sym_sets(TRS, vr, vitems); St = Sig(Bt, TRS); Lv = St.loglik(np.concatenate([sv, sp]))
    VY = np.concatenate([Y[vr], np.array([label_vec(it[2]) for it in vitems])]); VG = np.array([0 if (y == 0).all() else 1 for y in Y[vr]] + [2] * len(vitems))
    tidx = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); Bf, st, _ = sym_sets(POOL, tidx, []); Sf = Sig(Bf, POOL); Lt = Sf.loglik(st); TY = Y[tidx]
    TF = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120); COND = (meta.campaign + '_P' + meta.P.astype(str)).values[tidx]
    LAMS = [0, 0.03, 0.1, 0.3, 1, 3]
    rows, sel, hh, cond_rows = [], {}, [], []
    for m in MODELS:
        Rm = pickle.load(open(R0 + f'final_{m}.pkl', 'rb')); pv = Rm[0]['val']
        vs = {lam: score(decide(pv, Lv, lam), VY, VG) for lam in LAMS}; lt = max(LAMS, key=lambda l: (vs[l], -l)); sel[m] = dict(val=vs, tuned=lt)
        for s in sorted(Rm):
            pt = Rm[s]['test']; P0 = decide(pt, Lt, 0)
            for lam in LAMS:
                P = decide(pt, Lt, lam)
                for f in ('valve-bearing', 'valve-valve'):
                    k = TF == f; rows.append(dict(model=m, seed=s, lam=lam, family=f, label=float((P[k] == TY[k]).all(1).mean()), loc=float(((P[k] > 0) == (TY[k] > 0)).all(1).mean())))
                if lam == 1:
                    for f in ('valve-bearing', 'valve-valve'):
                        k = TF == f
                        for met, fn in (('label', lambda Q: (Q == TY).all(1)), ('loc', lambda Q: ((Q > 0) == (TY > 0)).all(1))):
                            a, b = fn(P0)[k], fn(P)[k]; hh.append(dict(model=m, seed=s, family=f, metric=met, helped=int((~a & b).sum()), harmed=int((a & ~b).sum())))
                    for c in np.unique(COND):
                        k = COND == c; cond_rows.append(dict(model=m, seed=s, cond=c, exact0=float((P0[k] == TY[k]).all(1).mean()), exact1=float((P[k] == TY[k]).all(1).mean()),
                                                         loc0=float(((P0[k] > 0) == (TY[k] > 0)).all(1).mean()), loc1=float(((P[k] > 0) == (TY[k] > 0)).all(1).mean())))
                    pickle.dump(P, open(OD + f'pred_{m}_{s}.pkl', 'wb'))
    D = pd.DataFrame(rows); D.to_csv(OD + 'bayes_eval.csv', index=False); H = pd.DataFrame(hh); H.to_csv(OD + 'help_harm.csv', index=False)
    pd.DataFrame(cond_rows).to_csv(OD + 'bayes_conditions.csv', index=False)
    hs = H.groupby(['family', 'metric', 'model'])[['helped', 'harmed']].sum()
    hs['p'] = [binomtest(int(a), int(a + b), 0.5).pvalue if a + b > 0 else 1.0 for a, b in zip(hs.helped, hs.harmed)]; hs.to_csv(OD + 'help_harm_sum.csv')
    json.dump(dict(selected=sel, b=list(Sf.b), s_rep=list(Sf.s_rep), s_ses=list(Sf.s_ses), centre=list(Sf.c), delta={k: list(v) for k, v in Sf.delta.items()}), open(OD + 'bayes_params.json', 'w'), indent=1)
    # physics-rule consistency of single-fault symptoms (POOL): share of repetitions moving in the rule direction by more than one scale b
    cons = []
    for (k, d), cs in RULES.items():
        j = SYM.index(k)
        for l in CONDS:
            if l.rsplit('_', 1)[0] not in cs: continue
            z = (Sf.V[Sf.lab == l, j] - Sf.c[j]) / Sf.b[j]; ok = (z > 1) if d == 'up' else (z < -1)
            cons.append(dict(symptom=k, direction=d, fault=l, share=float(ok.mean()), median_z=float(np.median(z))))
    C = pd.DataFrame(cons); C.to_csv(OD + 'rule_consistency.csv', index=False)
    # bearing shift of pneumatic symptoms vs healthy-session difference
    bs = {l: list(np.round(np.abs(Sf.delta[l][:5]) / (2 * Sf.s_ses[:5] + 1e-9), 2)) for l in CONDS if l.startswith('B1')}
    json.dump(dict(bearing_shift_over_session=bs, symptoms=SYM[:5]), open(OD + 'session_test.json', 'w'), indent=1)
    pd.set_option('display.width', 220)
    print(D.groupby(['family', 'model', 'lam'])[['label', 'loc']].mean().round(3).unstack('lam').to_string())
    print(json.dumps({m: sel[m]['tuned'] for m in sel})); print(hs.to_string()); print(C.groupby(['symptom', 'direction']).share.mean().round(2).to_string()); print(bs)
    print('b', np.round(Sf.b, 3), 's_rep', np.round(Sf.s_rep, 3), 's_ses', np.round(Sf.s_ses, 3))
