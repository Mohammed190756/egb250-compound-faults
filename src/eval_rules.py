"""Evaluation of the combining-rule comparison (M-b results taken from the main runs)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, pickle, os, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from data6 import *
from eval6 import decode, metrics, TFAM, TY, TCOND, MODELS
R0 = (RES) + '/'; R1 = R0 + 'rules/'
def load(m, r):
    fn = R0 + f'final_{m}.pkl' if r == 'b' else R1 + f'final_{m}_{r}.pkl'
    return pickle.load(open(fn, 'rb')) if os.path.exists(fn) else None
items = pair_items((13, 14, 15)); vr = np.flatnonzero(VAL); VL = [name_of(Y[i]) for i in vr] + [it[2] for it in items]
VY = np.array([label_vec(l) for l in VL]); VG = np.array([0 if l == 'healthy' else 1 for l in VL[:len(vr)]] + [2] * len(items))
rows, vrows, pc, preds = [], [], [], {}
for r in 'abcd':
    for m in MODELS:
        R = load(m, r)
        if R is None or len(R) < 3: continue
        for s in sorted(R):
            P = LABEL_Y[decode(R[s]['test'])]; preds[(r, m, s)] = P
            for f in ('valve-bearing', 'valve-valve'):
                k = TFAM == f; d = metrics(P[k], TY[k]); d.update(rule=r, model=m, seed=s, family=f); rows.append(d)
            if 'val' in R[s]:
                Pv = LABEL_Y[decode(R[s]['val'])]
                for g, gn in ((1, 'real'), (2, 'combined')):
                    k = (VG < 2) if g == 1 else (VG == 2)
                    vrows.append(dict(rule=r, model=m, seed=s, group=gn, label=float((Pv[k] == VY[k]).all(1).mean()), extra=float(((Pv[k] > 0).sum(1) > (VY[k] > 0).sum(1)).mean())))
        for c in np.unique(TCOND):
            k = TCOND == c; pc.append(dict(rule=r, model=m, cond=c, family=TFAM[k][0], label=name_of(TY[k][0]),
                exact=float(np.mean([(preds[(r, m, s)][k] == TY[k]).all(1).mean() for s in range(3)])), loc=float(np.mean([((preds[(r, m, s)][k] > 0) == (TY[k] > 0)).all(1).mean() for s in range(3)]))))
D = pd.DataFrame(rows); V = pd.DataFrame(vrows); PC = pd.DataFrame(pc)
D.to_csv(R1 + 'test_eval_rules.csv', index=False); V.to_csv(R1 + 'val_eval_rules.csv', index=False); PC.to_csv(R1 + 'conditions_rules.csv', index=False)
rng = np.random.default_rng(0); ci = []
for (r, m) in sorted({(a, b) for a, b, _ in preds}):
    for f in ('valve-bearing', 'valve-valve'):
        k = np.flatnonzero(TFAM == f); conds = np.unique(TCOND[k]); idx = {c: k[TCOND[k] == c] for c in conds}
        for metric in ('label', 'loc'):
            ok = np.mean([((preds[(r, m, s)] == TY).all(1) if metric == 'label' else ((preds[(r, m, s)] > 0) == (TY > 0)).all(1)) for s in range(3)], 0)
            bs = [ok[np.concatenate([idx[c] for c in rng.choice(conds, len(conds))])].mean() for _ in range(5000)]
            ci.append(dict(rule=r, model=m, family=f, metric=metric, mean=float(ok[k].mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
CI = pd.DataFrame(ci); CI.to_csv(R1 + 'ci_rules.csv', index=False)
pd.set_option('display.width', 250)
g = D.groupby(['family', 'model', 'rule'])[['label', 'loc', 'F1', 'rec_2S_DV', 'mode_2S_DV']]
print(g.mean().round(2).unstack('rule').to_string())
print(D.groupby(['family', 'rule'])[['label', 'loc']].mean().round(3).unstack('rule'))
if len(V): print(V.groupby(['group', 'model', 'rule'])[['label', 'extra']].mean().round(3).unstack('rule').to_string())
