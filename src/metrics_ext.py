"""Extended test metrics for the reference configuration (C2, M-b): label-level, detection, mode and probabilistic
metrics per model (mean over three seeds), the constant rule, and per-component detection (both test families pooled)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from data6 import *
from reason6 import loglik
from sklearn.metrics import roc_auc_score, average_precision_score
R = (RES) + '/'; O = R + 'final/'
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']
TIDX = np.concatenate([FAM['valve-bearing'], FAM['valve-valve']]); TY = Y[TIDX]
TF = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120)
NC = len(COMPS)

def label_metrics(P, T, L=None):
    fp_, ft = P > 0, T > 0
    d = {}
    d['exact'] = (P == T).all(1).mean()
    d['loc'] = (fp_ == ft).all(1).mean()
    d['hamming'] = (fp_ != ft).mean()                                  # share of wrong component fault/no-fault decisions
    inter = (fp_ & ft).sum(1); union = (fp_ | ft).sum(1)
    d['jaccard'] = np.where(union > 0, inter / np.maximum(union, 1), 1.0).mean()
    if L is not None:                                                   # top-3 label accuracy from ranked label scores
        top = np.argsort(-L, 1)[:, :3]
        hit = [(LABEL_Y[top[i]] == T[i]).all(1).any() for i in range(len(T))]
        d['top3'] = float(np.mean(hit))
    tp = (fp_ & ft).sum(); fp = (fp_ & ~ft).sum(); fn = (~fp_ & ft).sum(); tn = (~fp_ & ~ft).sum()
    d['prec'] = tp / max(tp + fp, 1); d['rec'] = tp / max(tp + fn, 1)
    d['f1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    d['spec'] = tn / (tn + fp) if tn + fp else np.nan
    d['bacc'] = (d['rec'] + d['spec']) / 2 if not np.isnan(d['spec']) else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    d['mcc'] = (tp * tn - fp * fn) / den if den > 0 else np.nan
    f1c = []
    for j in range(NC):
        a, b = fp_[:, j], ft[:, j]; t_ = (a & b).sum(); f_ = (a & ~b).sum(); n_ = (~a & b).sum()
        if t_ + f_ + n_ > 0: f1c.append(2 * t_ / (2 * t_ + f_ + n_))
    d['f1_macro'] = np.mean(f1c)
    d['MA'] = (P == T).all(1).sum() / max((fp_ == ft).all(1).sum(), 1)
    # Cohen's kappa over all component mode decisions (healthy = 0)
    a = P.ravel(); b = T.ravel(); po = (a == b).mean()
    cats = np.union1d(a, b); pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    d['kappa'] = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return d

def prob_metrics(prs, T):
    d = {}; auc, ap, ll, br = [], [], [], []
    conf, corr = [], []
    for j in range(NC):
        p = prs[j]; t = T[:, j]; pf = 1 - p[:, 0]; y = (t > 0).astype(int)
        if 0 < y.sum() < len(y): auc.append(roc_auc_score(y, pf)); ap.append(average_precision_score(y, pf))
        ll.append(-np.log(np.clip(p[np.arange(len(t)), t], 1e-12, 1)))
        oh = np.zeros_like(p); oh[np.arange(len(t)), t] = 1; br.append(((p - oh) ** 2).sum(1))
        conf.append(p.max(1)); corr.append(p.argmax(1) == t)
    d['auc'] = np.mean(auc) if auc else np.nan; d['prauc'] = np.mean(ap) if ap else np.nan
    d['n_auc'] = len(auc)
    d['logloss'] = np.concatenate(ll).mean(); d['brier'] = np.concatenate(br).mean()
    c = np.concatenate(conf); k = np.concatenate(corr); bins = np.linspace(0, 1, 11); e = 0
    for lo_, hi_ in zip(bins[:-1], bins[1:]):
        m = (c > lo_) & (c <= hi_)
        if m.any(): e += m.mean() * abs(k[m].mean() - c[m].mean())
    d['ece'] = e
    return d

rows, comp = [], []
for m in MODELS:
    Rr = pickle.load(open(R + f'final_{m}.pkl', 'rb'))
    for s in sorted(k for k in Rr if isinstance(k, (int, np.integer))):
        prs = Rr[s]['test']; L = loglik(prs); P = LABEL_Y[L.argmax(1)]
        for f in ('valve-bearing', 'valve-valve', 'pooled'):
            k = np.ones(len(TF), bool) if f == 'pooled' else TF == f
            d = label_metrics(P[k], TY[k], L[k]); d.update(prob_metrics([p[k] for p in prs], TY[k]))
            d.update(model=m, seed=int(s), family=f); rows.append(d)
        for j, c in enumerate(COMPS):                                   # per-component, both families pooled
            a = P[:, j] > 0; b = TY[:, j] > 0; tp = (a & b).sum(); fp = (a & ~b).sum(); fn = (~a & b).sum()
            comp.append(dict(model=m, seed=int(s), comp=c, prec=tp / (tp + fp) if tp + fp else np.nan, rec=tp / (tp + fn) if tp + fn else np.nan,
                             f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan, fa=fp / (~b).sum() if (~b).sum() else np.nan,
                             mode=(P[b, j] == TY[b, j]).mean() if b.sum() else np.nan, npos=int(b.sum()), nneg=int((~b).sum())))
# constant rule: most frequent true label per family
for f in ('valve-bearing', 'valve-valve'):
    k = TF == f; T = TY[k]; u, cnt = np.unique(T, axis=0, return_counts=True); P = np.repeat(u[cnt.argmax()][None], len(T), 0)
    d = label_metrics(P, T); d.update(model='constant', seed=0, family=f); rows.append(d)
D = pd.DataFrame(rows); C = pd.DataFrame(comp)
D.to_csv(O + 'metrics_ext.csv', index=False); C.to_csv(O + 'metrics_comp.csv', index=False)
pd.set_option('display.width', 250)
cols = ['exact', 'loc', 'hamming', 'jaccard', 'top3', 'prec', 'rec', 'f1', 'f1_macro', 'spec', 'bacc', 'mcc', 'MA', 'kappa', 'auc', 'prauc', 'n_auc', 'logloss', 'brier', 'ece']
print(D.groupby(['family', 'model'])[cols].mean().round(3).to_string())
print(C.groupby(['comp', 'model'])[['prec', 'rec', 'f1', 'fa', 'mode', 'npos', 'nneg']].mean().round(3).to_string())
