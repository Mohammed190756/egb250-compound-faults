"""Evaluation: exact label, location, per component, per condition, bootstrap intervals, baselines, validation behaviour."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from data6 import *
OUT = (RES) + '/'
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']
def decode(prs): return (sum(np.log(np.maximum(prs[c][:, LABEL_Y[:, c]], 1e-9)) for c in range(5))).argmax(1)
TIDX = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); TFAM = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120)
TY = Y[TIDX]; TCOND = (meta.campaign + '_P' + meta.P.astype(str)).values[TIDX]; TLAB = np.array([name_of(y) for y in TY])
def metrics(P, T):
    d = dict(label=float((P == T).all(1).mean()), loc=float(((P > 0) == (T > 0)).all(1).mean()))
    tp = ((P > 0) & (T > 0)).sum(); fp = ((P > 0) & (T == 0)).sum(); fn = ((P == 0) & (T > 0)).sum(); d['F1'] = float(2 * tp / max(2 * tp + fp + fn, 1))
    for j, c in enumerate(COMPS):
        t = T[:, j] > 0; p = P[:, j] > 0; d[f'rec_{c}'] = float((t & p).sum() / t.sum()) if t.sum() else np.nan; d[f'fa_{c}'] = float((p & ~t).sum() / max((~t).sum(), 1))
        d[f'mode_{c}'] = float((P[t, j] == T[t, j]).mean()) if t.sum() else np.nan
    return d
rows, preds = [], {}
for m in MODELS:
    for tag, lab in (('', 'with combined rows'), ('_single', 'single faults only')):
        try: R = pickle.load(open(OUT + f'final_{m}{tag}.pkl', 'rb'))
        except FileNotFoundError: continue
        for s in sorted(R):
            P = LABEL_Y[decode(R[s]['test'])]; preds[(m, tag, s)] = P
            for f in ('valve-bearing', 'valve-valve'):
                k = TFAM == f; r = metrics(P[k], TY[k]); r.update(model=m, training=lab, seed=s, family=f); rows.append(r)
D = pd.DataFrame(rows); D.to_csv(OUT + 'test_eval.csv', index=False); pickle.dump(preds, open(OUT + 'test_preds.pkl', 'wb'))
# cluster bootstrap of the mean over seeds, per model and family
rng = np.random.default_rng(0); ci = []
for (m, tag, s0) in sorted({(a, b, 0) for a, b, _ in preds}):
    seeds = [s for (a, b, s) in preds if a == m and b == tag]
    for f in ('valve-bearing', 'valve-valve'):
        k = np.flatnonzero(TFAM == f); conds = np.unique(TCOND[k]); idx = {c: k[TCOND[k] == c] for c in conds}
        for metric in ('label', 'loc'):
            ok = np.mean([((preds[(m, tag, s)] == TY).all(1) if metric == 'label' else ((preds[(m, tag, s)] > 0) == (TY > 0)).all(1)) for s in seeds], 0)
            bs = [ok[np.concatenate([idx[c] for c in rng.choice(conds, len(conds))])].mean() for _ in range(5000)]
            ci.append(dict(model=m, training='with combined rows' if tag == '' else 'single faults only', family=f, metric=metric, mean=float(ok[k].mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
CI = pd.DataFrame(ci); CI.to_csv(OUT + 'test_ci.csv', index=False)
# per condition and per-seed-pooled confusion of names
pc = []
for m in MODELS:
    seeds = [s for (a, b, s) in preds if a == m and b == '']
    for c in np.unique(TCOND):
        k = TCOND == c; ex = np.mean([(preds[(m, '', s)][k] == TY[k]).all(1).mean() for s in seeds]); lo = np.mean([((preds[(m, '', s)][k] > 0) == (TY[k] > 0)).all(1).mean() for s in seeds])
        det = {f'det_{COMPS[j]}': float(np.mean([(preds[(m, '', s)][k][:, j] > 0).mean() for s in seeds])) for j in range(5)}
        pc.append(dict(model=m, cond=c, label=TLAB[k][0], family=TFAM[k][0], exact=ex, loc=lo, **det))
pd.DataFrame(pc).to_csv(OUT + 'test_conditions.csv', index=False)
# constant baseline (uses knowledge of the test sets; reported only to show what the location metric rewards)
const = []
for f, labs in (('valve-bearing', None), ('valve-valve', None)):
    k = TFAM == f; T = TY[k]; best = 0; bestloc = 0
    for lab in sorted(set(TLAB[k])):
        P = np.tile(label_vec(lab), (k.sum(), 1)); best = max(best, (P == T).all(1).mean()); bestloc = max(bestloc, ((P > 0) == (T > 0)).all(1).mean())
    const.append(dict(family=f, loc=float(bestloc), label_best_constant=float(best)))
json.dump(const, open(OUT + 'const_baseline.json', 'w'), indent=1)
# validation behaviour of tuning-phase models: real rows (false alarms) and combined rows
vrows = []
V_lab = None
for m in MODELS:
    try: R = pickle.load(open(OUT + f'final_{m}.pkl', 'rb'))
    except FileNotFoundError: continue
    for s in sorted(R):
        if 'val' not in R[s]: continue
        if V_lab is None:
            items = pair_items((13, 14, 15)); vr = np.flatnonzero(VAL); V_lab = np.array([name_of(Y[i]) for i in vr] + [it[2] for it in items]); VY = np.array([label_vec(l) for l in V_lab]); VG = np.array([0 if l == 'healthy' else 1 for l in V_lab[:len(vr)]] + [2] * len(items))
        P = LABEL_Y[decode(R[s]['val'])]
        for g, gn in ((0, 'healthy'), (1, 'single'), (2, 'combined pair')):
            k = VG == g; vrows.append(dict(model=m, seed=s, group=gn, n=int(k.sum()), label=float((P[k] == VY[k]).all(1).mean()), loc=float(((P[k] > 0) == (VY[k] > 0)).all(1).mean()),
                                           extra_fault=float(((P[k] > 0).sum(1) > (VY[k] > 0).sum(1)).mean())))
pd.DataFrame(vrows).to_csv(OUT + 'val_eval.csv', index=False)
pd.set_option('display.width', 250)
print(D.groupby(['family', 'training', 'model'])[['label', 'loc', 'F1', 'rec_2S_DV', 'rec_B1', 'mode_2S_DV', 'mode_B1']].mean().round(3).to_string())
print(CI.round(3).to_string()); print(const)
if vrows: print(pd.DataFrame(vrows).groupby(['model', 'group'])[['label', 'loc', 'extra_fault']].mean().round(3).to_string())
