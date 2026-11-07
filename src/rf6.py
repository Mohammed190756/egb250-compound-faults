import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, time, itertools, pickle, numpy as np
sys.path.insert(0, (SRC))
from train6 import data, group_weights, decode, OUT, LABEL_Y, NMODE
from sklearn.ensemble import RandomForestClassifier
SPACE = dict(n_estimators=(100, 300), max_features=('sqrt', 0.3), max_depth=(None, 20), min_samples_leaf=(1, 3, 5))
def settings(n=8, seed=0):
    allc = list(itertools.product(*SPACE.values())); rng = np.random.default_rng(seed); d = (300, 'sqrt', None, 1); pick = [d]
    for i in rng.permutation(len(allc)):
        if allc[i] != d and len(pick) < n: pick.append(allc[i])
    return [dict(zip(SPACE, c)) for c in pick]
def fit(D, hp, seed, keep=None):
    keep = np.ones(len(D['y']), bool) if keep is None else keep
    Fw = D['F'][keep].reshape(-1, D['F'].shape[-1]); Yw = np.repeat(D['y'][keep], 9, 0); w = group_weights(np.repeat(D['group'][keep], 9))
    return [RandomForestClassifier(**hp, random_state=seed, n_jobs=1).fit(Fw, Yw[:, c], sample_weight=w) for c in range(5)]
def probs(ms, D, win=False):
    Fa = D['F'].reshape(-1, D['F'].shape[-1]); out = []
    for c, m in enumerate(ms):
        p = np.zeros((len(Fa), NMODE[c])); p[:, m.classes_] = m.predict_proba(Fa); out.append(p if win else p.reshape(len(D['y']), 9, -1).mean(1))
    return out
def val_eval(ms, V):
    pw = probs(ms, V, win=True); yw = np.repeat(V['y'], 9, 0); gw = np.repeat(V['group'], 9)
    ce = sum(-np.log(np.clip(pw[c][np.arange(len(yw)), yw[:, c]], 1e-7, 1)) for c in range(5))
    vl = float(np.mean([ce[gw == g].mean() for g in np.unique(gw)])); pr = [p.reshape(len(V['y']), 9, -1).mean(1) for p in pw]; P = LABEL_Y[decode(pr)]; T = V['y']
    return dict(val_loss=vl, val_exact=float((P == T).all(1).mean()), val_exact_pair=float((P == T).all(1)[V['group'] == 2].mean()), val_exact_real=float((P == T).all(1)[V['group'] < 2].mean())), pr
if __name__ == '__main__':
    D, V = data('tune'); S = settings(); fn = OUT + 'tune_RF.json'; res = json.load(open(fn)) if os.path.exists(fn) else {}
    def go(k, s):
        if f'{k}:{s}' in res: return
        t = time.time(); r, _ = val_eval(fit(D, S[k], s), V); r.update(hp=S[k], sec=time.time() - t); res[f'{k}:{s}'] = r; json.dump(res, open(fn, 'w'), default=str)
        print('RF', k, s, {a: round(b, 3) for a, b in r.items() if a.startswith('val')}, round(r['sec']), flush=True)
    for k in range(len(S)): go(k, 0)
    top = sorted(range(len(S)), key=lambda k: res[f'{k}:0']['val_loss'])[:3]
    for k in top:
        for s in (1, 2): go(k, s)
    best = min(top, key=lambda k: np.mean([res[f'{k}:{s}']['val_loss'] for s in range(3)])); res['selected'] = dict(k=best, hp=S[best], top=top); json.dump(res, open(fn, 'w'), default=str)
    print('SELECTED RF', res['selected'], flush=True)
    Df, T = data('final'); fin = {}; base = {}
    for s in range(3):
        t = time.time(); _, vp = val_eval(fit(D, S[best], s), V)
        fin[s] = dict(test=probs(fit(Df, S[best], s), T), val=vp, sec=time.time() - t, hp=S[best])
        t = time.time(); base[s] = dict(test=probs(fit(Df, S[best], s, keep=Df['group'] < 2), T), sec=time.time() - t, hp=S[best]); print('final RF', s, flush=True)
        pickle.dump(fin, open(OUT + 'final_RF.pkl', 'wb')); pickle.dump(base, open(OUT + 'final_RF_single.pkl', 'wb'))
