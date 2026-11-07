"""Collects every configuration of the study, computes all metrics, intervals and paired tests -> res/final/."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, pickle, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest
sys.path.insert(0, (SRC))
from data6 import *
from reason6 import loglik
R = (RES) + '/'; O = R + 'final/'; os.makedirs(O, exist_ok=True)
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']
TIDX = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); TY = Y[TIDX]
TF = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120); TC = (meta.campaign + '_P' + meta.P.astype(str)).values[TIDX]
CODE = lambda c: c.replace('Multi_P', 'MF-P').replace('Valve_P', 'V-P')
def dec(prs): return LABEL_Y[loglik(prs).argmax(1)]
def load(fn):
    if not os.path.exists(fn): return {}
    Rr = pickle.load(open(fn, 'rb')); return {s: dec(v['test']) for s, v in Rr.items() if isinstance(s, (int, np.integer))}
CONF = {'C2 M-b (reference)': lambda m: load(R + f'final_{m}.pkl'), 'single faults only': lambda m: load(R + f'final_{m}_single.pkl'),
        'C1 random augmentation': lambda m: load(R + f'var/final_{m}_aug.pkl'),
        'C2 M-a': lambda m: load(R + f'rules/final_{m}_a.pkl'), 'C2 M-c': lambda m: load(R + f'rules/final_{m}_c.pkl'),
        'C2 M-d': lambda m: load(R + f'rules/final_{m}_d.pkl'), 'C2 M-e': lambda m: load(R + f'var/final_{m}_coh.pkl'),
        'C2 M-b random pairing': lambda m: load(R + f'var/final_{m}_rand.pkl'), 'C2 M-b without lower bound': lambda m: load(R + f'var/final_{m}_nofloor.pkl'),
        'C2 M-b, test pair structures removed': lambda m: load(R + f'var/final_{m}_unseen.pkl'), 'C2 M-b without A2': lambda m: load(R + f'var/final_{m}_noA2.pkl'),
        'reference: Valve P1 (B)': lambda m: load(R + f'abl/final_{m}_B.pkl'), 'reference: both, pooled (C1)': lambda m: load(R + f'abl/final_{m}_C1.pkl'),
        'reference: both, own campaign (C2)': lambda m: load(R + f'abl/final_{m}_C2.pkl')}
def wa(m): return {s: pickle.load(open(f, 'rb')) for s, f in ((s, R + f'reason/pred_{m}_{s}_RR.pkl') for s in range(3)) if os.path.exists(f)}
def by(m): return {s: pickle.load(open(f, 'rb')) for s, f in ((s, R + f'bayes/pred_{m}_{s}.pkl') for s in range(3)) if os.path.exists(f)}
CONF['C2 M-b + weighted abduction'] = wa; CONF['C2 M-b + Bayesian layer'] = by
def ten(m):
    a = load(R + f'final_{m}.pkl'); a.update(load(R + f'var/final_{m}_seeds.pkl')); return a
CONF['C2 M-b, ten seeds'] = ten
for K in range(1, 6):
    CONF[f'M5 random mask {K}'] = (lambda K: (lambda m: load(R + f'var/final_{m}_mask{K}.pkl') if m == 'GNN' else {}))(K)
def metrics(P, T):
    ex = (P == T).all(1); lo = ((P > 0) == (T > 0)).all(1)
    d = dict(label=float(ex.mean()), loc=float(lo.mean()), MA=float(ex.sum() / max(lo.sum(), 1)))
    tp = ((P > 0) & (T > 0)).sum(); fp = ((P > 0) & (T == 0)).sum(); fn = ((P == 0) & (T > 0)).sum(); d['F1'] = float(2 * tp / max(2 * tp + fp + fn, 1))
    for j, c in enumerate(COMPS):
        t = T[:, j] > 0; p = P[:, j] > 0
        d[f'rec_{c}'] = float((t & p).sum() / t.sum()) if t.sum() else np.nan; d[f'fa_{c}'] = float((p & ~t).sum() / max((~t).sum(), 1)) if (~t).sum() else np.nan
        d[f'mode_{c}'] = float((P[t, j] == T[t, j]).mean()) if t.sum() else np.nan
    return d
rows, preds = [], {}
for cname, fn in CONF.items():
    for m in MODELS:
        if cname == 'C2 M-b, ten seeds' and m == 'RF': continue
        if cname.startswith('C2 M-b +') and False: pass
        Pd = fn(m)
        if not Pd: continue
        preds[(cname, m)] = Pd
        for s, P in Pd.items():
            for f in ('valve-bearing', 'valve-valve'):
                k = TF == f; d = metrics(P[k], TY[k]); d.update(config=cname, model=m, seed=int(s), family=f, n_seeds=len(Pd))
                cond = [(P[(TC == c)] == TY[(TC == c)]).all(1).mean() for c in np.unique(TC[k])]; d['label_cond'] = float(np.mean(cond)); rows.append(d)
M = pd.DataFrame(rows); M.to_csv(O + 'metrics_all.csv', index=False)
# per-condition (mean over seeds) and cluster-bootstrap intervals of the mean over seeds
rng = np.random.default_rng(0); ci, pc = [], []
for (cname, m), Pd in preds.items():
    ex = np.mean([(P == TY).all(1) for P in Pd.values()], 0); lo = np.mean([((P > 0) == (TY > 0)).all(1) for P in Pd.values()], 0)
    for c in np.unique(TC): k = TC == c; pc.append(dict(config=cname, model=m, cond=CODE(c), label_name=name_of(TY[k][0]), family=TF[k][0], exact=float(ex[k].mean()), loc=float(lo[k].mean())))
    for f in ('valve-bearing', 'valve-valve'):
        k = np.flatnonzero(TF == f); conds = np.unique(TC[k]); idx = {c: k[TC[k] == c] for c in conds}
        for met, v in (('label', ex), ('loc', lo)):
            bs = [v[np.concatenate([idx[c] for c in rng.choice(conds, len(conds))])].mean() for _ in range(2000)]
            ci.append(dict(config=cname, model=m, family=f, metric=met, mean=float(v[k].mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
CI = pd.DataFrame(ci); CI.to_csv(O + 'ci_all.csv', index=False); PC = pd.DataFrame(pc); PC.to_csv(O + 'conditions_all.csv', index=False)
# paired tests over the 20 conditions (exact label, mean over seeds) and McNemar over recordings (seed 0)
pairs = [('C2 M-b (reference)', 'single faults only'), ('C1 random augmentation', 'C2 M-b (reference)'), ('C2 M-d', 'C2 M-b (reference)'),
         ('C2 M-b + Bayesian layer', 'C2 M-b (reference)'), ('C2 M-b + weighted abduction', 'C2 M-b (reference)'), ('C2 M-e', 'C2 M-b (reference)'),
         ('C2 M-a', 'C2 M-b (reference)'), ('C2 M-c', 'C2 M-b (reference)')]
tests = []
for a, b in pairs:
    for m in MODELS:
        if (a, m) not in preds or (b, m) not in preds: continue
        for met in ('exact', 'loc'):
            xa = PC[(PC.config == a) & (PC.model == m)].set_index('cond')[met]; xb = PC[(PC.config == b) & (PC.model == m)].set_index('cond')[met]; dlt = (xa - xb.loc[xa.index]).values
            p = wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
            Pa, Pb = preds[(a, m)][min(preds[(a, m)])], preds[(b, m)][min(preds[(b, m)])]
            fa = ((Pa == TY).all(1) if met == 'exact' else ((Pa > 0) == (TY > 0)).all(1)); fb = ((Pb == TY).all(1) if met == 'exact' else ((Pb > 0) == (TY > 0)).all(1))
            n01, n10 = int((~fb & fa).sum()), int((fb & ~fa).sum()); pm = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            tests.append(dict(a=a, b=b, model=m, metric=met, mean_diff=float(dlt.mean()), wilcoxon_p=float(p), a_only=n01, b_only=n10, mcnemar_p=float(pm)))
pd.DataFrame(tests).to_csv(O + 'paired_tests.csv', index=False)
# confusion of the 2S_DV and B1 modes on V+B (reference, seeds pooled)
conf = {}
for m in MODELS:
    if ('C2 M-b (reference)', m) not in preds: continue
    k = TF == 'valve-bearing'; C = np.zeros((5, 5), int); Cb = np.zeros((4, 4), int)
    for P in preds[('C2 M-b (reference)', m)].values():
        for t, p in zip(TY[k][:, 3], P[k][:, 3]): C[t, p] += 1
        for t, p in zip(TY[k][:, 4], P[k][:, 4]): Cb[t, p] += 1
    conf[m] = dict(DV=C.tolist(), B1=Cb.tolist())
json.dump(conf, open(O + 'confusion.json', 'w'))
pd.set_option('display.width', 250)
S = M.groupby(['config', 'family', 'model'])[['label', 'loc']].mean().unstack('family').round(2)
print(S.to_string())

# real-combination cross-validation (upper reference), seed 0 of each fold
cv = []
for m in MODELS:
    P = np.zeros_like(TY); got = np.zeros(len(TY), bool)
    for k in range(1, 6):
        fn = R + f'cv/final_{m}_fold{k}.pkl'
        if not os.path.exists(fn): continue
        Rr = pickle.load(open(fn, 'rb'))
        if 0 not in Rr: continue
        idx = Rr[0]['idx']; pos = np.searchsorted(TIDX, idx) if np.all(np.diff(TIDX) > 0) else np.array([np.flatnonzero(TIDX == i)[0] for i in idx])
        P[pos] = dec(Rr[0]['test']); got[pos] = True
    if not got.all(): continue
    for f in ('valve-bearing', 'valve-valve'):
        k = TF == f; d = metrics(P[k], TY[k]); d.update(model=m, family=f); cv.append(d)
pd.DataFrame(cv).to_csv(O + 'cv.csv', index=False); print(pd.DataFrame(cv)[['model', 'family', 'label', 'loc', 'mode_2S_DV']].round(2).to_string() if cv else 'no cv yet')
