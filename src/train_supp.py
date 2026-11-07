"""Supplementary analyses with the main protocol, rule M-b, hyperparameters and e* of the main search.
  abl  : healthy-reference setups (set P6SETUP=B, C1 or C2 in the environment); outputs res/abl/final_<model>_<setup>.pkl
  cv k : real-combination training, grouped fold k (1-5): pool + combined rows + real recordings of the 16 other combinations."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, time, copy, pickle, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
import train6 as T6
from data6 import *
from train6 import prep, group_weights, val_eval, record_probs
from models4 import make, fwd
torch.set_num_threads(1)
R = (RES) + '/'
FOLDS = {1: ['Multi_P10', 'Multi_P11', 'Multi_P13', 'Valve_P25'], 2: ['Multi_P2', 'Multi_P3', 'Multi_P4', 'Valve_P24'], 3: ['Multi_P5', 'Multi_P9', 'Valve_P18', 'Valve_P21'],
         4: ['Multi_P6', 'Multi_P8', 'Valve_P20', 'Valve_P23'], 5: ['Multi_P7', 'Multi_P12', 'Valve_P19', 'Valve_P22']}
COND = (meta.campaign + '_P' + meta.P.astype(str)).values
COMBO = (meta.role == 'combo').values
def run(name, seed, hp, D, V=None, epochs=None):
    """As train6.run, but every real recording (single or real combination) is used in every epoch; combined rows are sampled."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    nf = D['F'].shape[-1]; model = make(name, nf, hp['w'], hp['p']); opt = torch.optim.AdamW(model.parameters(), hp['lr'], weight_decay=hp['wd'])
    Xw = D['X'].reshape(-1, 13, 72); Ew = D['E'].reshape(-1, 2, 300); Fw = D['F'].reshape(-1, nf); Yw = np.repeat(D['y'], 9, 0); Gw = np.repeat(D['group'], 9)
    isreal = np.repeat(D['isreal'], 9); real = np.flatnonzero(isreal); pair = np.flatnonzero(~isreal); hist = []; best = (np.inf, 0, None)
    for ep in range(epochs or T6.MAXEP):
        model.train(); sel = rng.permutation(np.concatenate([real, rng.choice(pair, min(len(real), len(pair)), replace=False)])); w_all = group_weights(Gw[sel]); tl = []
        for b in range(0, len(sel), 64):
            i = sel[b:b + 64]; w = torch.from_numpy(w_all[b:b + 64]).float()
            o = fwd(model, name, torch.from_numpy(Xw[i]), torch.from_numpy(Ew[i]), torch.from_numpy(Fw[i]))
            loss = sum((F.cross_entropy(o[c], torch.from_numpy(Yw[i, c]).long(), reduction='none') * w).sum() / w.sum() for c in range(5))
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss))
        rec = dict(epoch=ep + 1, train_loss=float(np.mean(tl)))
        if V is not None:
            rec.update(val_eval(model, name, V))
            if rec['val_loss'] < best[0]: best = (rec['val_loss'], ep + 1, copy.deepcopy(model.state_dict()))
            hist.append(rec)
            if ep + 1 - best[1] >= T6.PAT: break
        else: hist.append(rec)
    if V is not None: model.load_state_dict(best[2])
    return model, hist, best[1]
def mark(D, n_real): D['isreal'] = np.arange(len(D['y'])) < n_real; return D
def add_real(D, B, idx):
    """Append real recordings (real combinations) to a built set; they count as the pair group for the loss weights."""
    X, E = B.real(idx); lab = [name_of(Y[i]) for i in idx]
    out = dict(X=np.concatenate([D['X'], X]), E=np.concatenate([D['E'], E]), label=np.concatenate([D['label'], lab]), y=np.concatenate([D['y'], Y[idx]]),
               group=np.concatenate([D['group'], np.full(len(idx), 2)]), isreal=np.concatenate([D['isreal'], np.ones(len(idx), bool)]))
    return out
def sets(mode, k=None):
    if mode == 'abl':
        Dt = mark(build(TRS, TRS, range(1, 13)), int(TRS.sum())); Vt = build(TRS, VAL, (13, 14, 15)); mu, sd = prep(Dt); prep(Vt, mu, sd)
        Df = mark(build(POOL, POOL, range(1, 16)), int(POOL.sum())); mu, sd = prep(Df); Ts = test_set(POOL); prep(Ts, mu, sd); return Dt, Vt, Df, Ts
    hold = np.isin(COND, FOLDS[k]) & COMBO; tr = COMBO & ~hold
    Df = mark(build(POOL, POOL, range(1, 16)), int(POOL.sum())); B = Df.pop('_B'); Df = add_real(Df, B, np.flatnonzero(tr)); mu, sd = prep(Df)
    hi = np.flatnonzero(hold); X, E = B.real(hi); Ts = dict(X=X, E=E, idx=hi, y=Y[hi], label=np.array([name_of(Y[i]) for i in hi]), group=np.full(len(hi), 2)); prep(Ts, mu, sd)
    return None, None, Df, Ts
def rf_fit(D, hp, s):
    from sklearn.ensemble import RandomForestClassifier
    Fw = D['F'].reshape(-1, D['F'].shape[-1]); Yw = np.repeat(D['y'], 9, 0); w = group_weights(np.repeat(D['group'], 9))
    return [RandomForestClassifier(**hp, random_state=s, n_jobs=1).fit(Fw, Yw[:, c], sample_weight=w) for c in range(5)]
def rf_probs(ms, D):
    Fa = D['F'].reshape(-1, D['F'].shape[-1]); out = []
    for c, m in enumerate(ms):
        p = np.zeros((len(Fa), NMODE[c])); p[:, m.classes_] = m.predict_proba(Fa); out.append(p.reshape(len(D['y']), 9, -1).mean(1))
    return out
def go(mode, names, k=None):
    tag = SETUP if mode == 'abl' else f'fold{k}'; od = R + ('abl/' if mode == 'abl' else 'cv/'); os.makedirs(od, exist_ok=True)
    Dt, Vt, Df, Ts = sets(mode, k)
    for name in names:
        fn = od + f'final_{name}_{tag}.pkl'; res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
        if name == 'RF':
            hp = json.load(open(T6.OUT + 'tune_RF.json'))['selected']['hp']; hp = {a: (None if b in (None, 'None') else b) for a, b in hp.items()}
        else: J = json.load(open(T6.OUT + f'tune_{name}.json'))['selected']; hp = J['hp']
        for s in (json.load(open((SRC + '/seedcfg.json')))[mode] if name != 'RF' else [0]):
            if s in res: continue
            t = time.time()
            if name == 'RF':
                res[s] = dict(test=rf_probs(rf_fit(Df, hp, s), Ts), idx=Ts.get('idx'))
            else:
                m, h, _ = run(name, s, hp, Df, None, epochs=J['epochs']); res[s] = dict(test=record_probs(m, name, Ts), idx=Ts.get('idx'), hist=h)
            res[s]['sec'] = time.time() - t; pickle.dump(res, open(fn, 'wb')); print(mode, tag, name, s, round(time.time() - t), flush=True)
if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'abl': go('abl', sys.argv[2:])
    else:
        for k in [int(x) for x in sys.argv[2].split(',')]: go('cv', sys.argv[3:], k)
