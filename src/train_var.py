"""Robustness variants of the main configuration (rule M-b, hyperparameters and e* of the main search).
usage: python train_var.py VARIANT MODEL [MODEL ...]
  seeds    : seven further seeds (3-9) of the main configuration
  rand     : random pairing of repetitions (instead of matched repetition numbers)
  nofloor  : energy rule without the lower bound (clipped at 1e-6 of the healthy energy)
  coh      : fully coherent (amplitude) combination of RMS profiles and envelope amplitudes
  unseen   : the component-pair structures of the test sets (2S_DV+B1, 2S_IV+2S_DV, 1S_IVb+2S_DV) removed from all combined rows
  noA2     : accelerometer A2 removed from every input
  maskK    : M5 with random sensor mask K (set P6MASK=K)
  aug      : random compositional augmentation: real rows only; every epoch, random pairs of single-fault windows on different
             components are summed in the normalised input domain, label = component-wise maximum (RF: one fixed draw)"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, time, pickle, numpy as np
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
import data6 as D6
from data6 import *
import train6 as T6
from train6 import prep, run, record_probs
OUT = (RES) + '/var/'; os.makedirs(OUT, exist_ok=True)
VAR = sys.argv[1]
TEST_STRUCT = {('2S_DV', 'B1'), ('2S_IV', '2S_DV'), ('1S_IVb', '2S_DV')}
def items_for(reps):
    if VAR == 'rand':
        rng = np.random.default_rng(7); out = []
        for a, b in PAIRS:
            rb = [r for r in reps if (b, r) in REC]
            for r in reps:
                if (a, r) in REC and rb: out.append((REC[(a, r)], REC[(b, rng.choice(rb))], f'{a}+{b}'))
        return out
    it = pair_items(reps)
    if VAR == 'unseen': it = [x for x in it if tuple(COMP_OF[p] for p in x[2].split('+')) not in TEST_STRUCT]
    return it
_merge = Builder.merge_many
def merge_var(s, I, J, rule='b', rng=None):
    if VAR == 'nofloor':
        Pa, Pb, Ea, Eb = s.P[I], s.P[J], s.E[I], s.E[J]; out = np.empty_like(Pa); ip, idn = s.ip, s.idn
        out[:, :, ip] = Pa[:, :, ip] + Pb[:, :, ip] - s.Hp[ip]
        out[:, :, idn] = np.sqrt(np.maximum(Pa[:, :, idn] ** 2 + Pb[:, :, idn] ** 2 - s.H2[idn], 1e-6 * s.H2[idn]))
        return out, np.sqrt(np.maximum(Ea ** 2 + Eb ** 2 - s.HE2, 1e-6 * s.HE2))
    if VAR == 'coh':
        Pa, Pb, Ea, Eb = s.P[I], s.P[J], s.E[I], s.E[J]; out = np.empty_like(Pa); ip, idn = s.ip, s.idn
        HR = np.sqrt(s.H2); HE = np.sqrt(s.HE2)
        out[:, :, ip] = Pa[:, :, ip] + Pb[:, :, ip] - s.Hp[ip]
        out[:, :, idn] = np.maximum(Pa[:, :, idn] + Pb[:, :, idn] - HR[idn], 1e-3 * HR[idn])
        return out, np.maximum(Ea + Eb - HE, 1e-3 * HE)
    return _merge(s, I, J, rule, rng)
Builder.merge_many = merge_var
def build_v(stat_mask, real_mask, reps):
    B = Builder(stat_mask); ri = np.flatnonzero(real_mask); items = items_for(reps)
    Xr, Er = B.real(ri); Xp, Ep = B.pairs(items, 'b')
    lab = [name_of(Y[i]) for i in ri] + [it[2] for it in items]; grp = [0 if l == 'healthy' else 1 for l in lab[:len(ri)]] + [2] * len(items)
    return dict(X=np.concatenate([Xr, Xp]), E=np.concatenate([Er, Ep]), label=np.array(lab), y=np.array([label_vec(l) for l in lab]), group=np.array(grp)), B
def drop_a2(Q):
    if VAR == 'noA2': Q['X'][..., 1, :] = 0.0   # channel order A1, A2, ...: A2 set to its healthy mean (zero after normalisation)
def data():
    D, B = build_v(POOL, POOL, range(1, 16)); T = test_set(POOL); drop_a2(D); drop_a2(T); mu, sd = prep(D); prep(T, mu, sd); return D, T
import torch, copy
import torch.nn.functional as F
from models4 import make, fwd
from train6 import group_weights
def aug_pairs(D, rng, n):
    """n random window pairs of single faults on different components, combined in the normalised domain."""
    yw = np.repeat(D['y'], 9, 0); single = np.flatnonzero((np.repeat(D['group'], 9) == 1)); comp = (yw[single] > 0).argmax(1)
    a = rng.choice(single, n); b = rng.choice(single, n); ok = comp[np.searchsorted(single, a)] != comp[np.searchsorted(single, b)]; a, b = a[ok], b[ok]
    Xw = D['X'].reshape(-1, 13, 72); Ew = D['E'].reshape(-1, 2, 300)
    X = Xw[a] + Xw[b]; E = Ew[a] + Ew[b]; y = np.maximum(yw[a], yw[b]); return X, E, y
def aug_feats(X, E, mu, sd):
    Fe = summary(X[:, None], E[:, None])[:, 0]; return np.nan_to_num((Fe - mu) / sd).clip(-30, 30).astype(np.float32)
def run_aug(name, seed, hp, D, epochs, mu, sd):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); nf = D['F'].shape[-1]
    model = make(name, nf, hp['w'], hp['p']); opt = torch.optim.AdamW(model.parameters(), hp['lr'], weight_decay=hp['wd'])
    Xw = D['X'].reshape(-1, 13, 72); Ew = D['E'].reshape(-1, 2, 300); Fw = D['F'].reshape(-1, nf); Yw = np.repeat(D['y'], 9, 0); Gw = np.repeat(D['group'], 9)
    for ep in range(epochs):
        model.train(); Xa, Ea, ya = aug_pairs(D, rng, len(Xw)); Fa = aug_feats(Xa, Ea, mu, sd)
        X = np.concatenate([Xw, Xa]); E = np.concatenate([Ew, Ea]); Fx = np.concatenate([Fw, Fa]); Yy = np.concatenate([Yw, ya]); G = np.concatenate([Gw, np.full(len(ya), 2)])
        sel = rng.permutation(len(X)); w_all = group_weights(G[sel])
        for b in range(0, len(sel), 64):
            i = sel[b:b + 64]; w = torch.from_numpy(w_all[b:b + 64]).float()
            o = fwd(model, name, torch.from_numpy(X[i].astype(np.float32)), torch.from_numpy(E[i].astype(np.float32)), torch.from_numpy(Fx[i]))
            loss = sum((F.cross_entropy(o[c], torch.from_numpy(Yy[i, c]).long(), reduction='none') * w).sum() / w.sum() for c in range(5))
            opt.zero_grad(); loss.backward(); opt.step()
    return model
if __name__ == '__main__' and VAR == 'aug':
    names = sys.argv[2:]; B = Builder(POOL); ri = np.flatnonzero(POOL); Xr, Er = B.real(ri); lab = [name_of(Y[i]) for i in ri]
    D = dict(X=Xr, E=Er, y=Y[ri], group=np.array([0 if l == 'healthy' else 1 for l in lab])); T = test_set(POOL); mu, sd = prep(D); prep(T, mu, sd)
    for name in names:
        fn = OUT + f'final_{name}_aug.pkl'; res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
        for s in ((range(3)) if name != 'RF' else [0]):
            if s in res: continue
            t = time.time()
            if name == 'RF':
                from rf6 import probs
                from sklearn.ensemble import RandomForestClassifier
                hp = json.load(open(T6.OUT + 'tune_RF.json'))['selected']['hp']; hp = {a: (None if b in (None, 'None') else b) for a, b in hp.items()}
                rng = np.random.default_rng(s); Xa, Ea, ya = aug_pairs(D, rng, 2145 * 9); Fa = aug_feats(Xa, Ea, mu, sd)
                Fw = np.concatenate([D['F'].reshape(-1, D['F'].shape[-1]), Fa]); Yw = np.concatenate([np.repeat(D['y'], 9, 0), ya]); w = group_weights(np.concatenate([np.repeat(D['group'], 9), np.full(len(ya), 2)]))
                ms = [RandomForestClassifier(**hp, random_state=s, n_jobs=1).fit(Fw, Yw[:, c], sample_weight=w) for c in range(5)]; res[s] = dict(test=probs(ms, T))
            else:
                J = json.load(open(T6.OUT + f'tune_{name}.json'))['selected']; m = run_aug(name, s, J['hp'], D, J['epochs'], mu, sd); res[s] = dict(test=record_probs(m, name, T))
            res[s]['sec'] = time.time() - t; pickle.dump(res, open(fn, 'wb')); print('aug', name, s, round(time.time() - t), flush=True)
    sys.exit()
if __name__ == '__main__':
    names = sys.argv[2:]; D, Ts = data(); tag = VAR if not VAR.startswith('mask') else f"mask{os.environ.get('P6MASK')}"
    print(VAR, 'rows', len(D['y']), 'pairs', int((D['group'] == 2).sum()), flush=True)
    seeds = range(3, 10) if VAR == 'seeds' else range(3)
    for name in names:
        fn = OUT + f'final_{name}_{tag}.pkl'; res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
        CFG = json.load(open((SRC + '/seedcfg.json')))['var']; key = 'mask' if VAR.startswith('mask') else VAR
        sd_ = CFG['seeds'].get(name, []) if VAR == 'seeds' else CFG.get(key, CFG['default'])
        for s in (sd_ if name != 'RF' else [0]):   # RF: one seed (its seed-to-seed spread was below 0.01 in the main configuration)
            if s in res: continue
            t = time.time()
            if name == 'RF':
                from rf6 import fit, probs
                hp = json.load(open(T6.OUT + 'tune_RF.json'))['selected']['hp']; hp = {a: (None if b in (None, 'None') else b) for a, b in hp.items()}
                res[s] = dict(test=probs(fit(D, hp, s), Ts))
            else:
                J = json.load(open(T6.OUT + f'tune_{name}.json'))['selected']; m, h, _ = run(name, s, J['hp'], D, None, epochs=J['epochs'])
                res[s] = dict(test=record_probs(m, name, Ts))
            res[s]['sec'] = time.time() - t; pickle.dump(res, open(fn, 'wb')); print(tag, name, s, round(time.time() - t), flush=True)
