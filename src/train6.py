"""Training: real healthy/single rows + combined two-fault rows, group-balanced loss, five component heads
decoded into one of 164 label names; random search + early stopping on validation (real + combined rows)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, time, copy, itertools, pickle, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
from data6 import *
from models4 import make, fwd
torch.set_num_threads(1)
OUT = (RES) + '/'; MAXEP, PAT = 200, 15
def decode(prs):
    S = sum(np.log(np.maximum(prs[c][:, LABEL_Y[:, c]], 1e-9)) for c in range(5)); return S.argmax(1)
def prep(D, fmu=None, fsd=None):
    Fe = summary(D['X'], D['E'])
    if fmu is None: fmu = Fe.reshape(-1, Fe.shape[-1]).mean(0); fsd = Fe.reshape(-1, Fe.shape[-1]).std(0) + 1e-6
    D['F'] = np.nan_to_num((Fe - fmu) / fsd).clip(-30, 30).astype(np.float32); return fmu, fsd
def group_weights(g):
    n = np.bincount(g, minlength=3).astype(float); w = np.where(n > 0, len(g) / (np.count_nonzero(n) * np.maximum(n, 1)), 0); return w[g]
@torch.no_grad()
def predict(model, name, D, bs=512):
    model.eval(); X = D['X'].reshape(-1, 13, 72); E = D['E'].reshape(-1, 2, 300); Fe = D['F'].reshape(-1, D['F'].shape[-1]); outs = [[] for _ in range(5)]
    for b in range(0, len(X), bs):
        o = fwd(model, name, torch.from_numpy(X[b:b + bs]), torch.from_numpy(E[b:b + bs]), torch.from_numpy(Fe[b:b + bs]))
        for c in range(5): outs[c].append(o[c])
    return [torch.cat(o) for o in outs]
def val_eval(model, name, V):
    o = predict(model, name, V); yw = np.repeat(V['y'], 9, 0); gw = np.repeat(V['group'], 9)
    ce = sum(F.cross_entropy(o[c], torch.from_numpy(yw[:, c]).long(), reduction='none') for c in range(5)).numpy()
    vl = float(np.mean([ce[gw == g].mean() for g in np.unique(gw)]))
    prs = [o[c].softmax(-1).reshape(len(V['y']), 9, -1).mean(1).numpy() for c in range(5)]; P = LABEL_Y[decode(prs)]; T = V['y']
    ex = (P == T).all(1); loc = ((P > 0) == (T > 0)).all(1)
    return dict(val_loss=vl, val_exact=float(ex.mean()), val_loc=float(loc.mean()), val_exact_real=float(ex[V['group'] < 2].mean()),
                val_exact_pair=float(ex[V['group'] == 2].mean()) if (V['group'] == 2).any() else float('nan'))
def run(name, seed, hp, D, V=None, epochs=None, use_pairs=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    nf = D['F'].shape[-1]; model = make(name, nf, hp['w'], hp['p']); opt = torch.optim.AdamW(model.parameters(), hp['lr'], weight_decay=hp['wd'])
    Xw = D['X'].reshape(-1, 13, 72); Ew = D['E'].reshape(-1, 2, 300); Fw = D['F'].reshape(-1, nf); Yw = np.repeat(D['y'], 9, 0); Gw = np.repeat(D['group'], 9)
    real = np.flatnonzero(Gw < 2); pair = np.flatnonzero(Gw == 2); hist = []; best = (np.inf, 0, None)
    for ep in range(epochs or MAXEP):
        t0 = time.time(); model.train()
        sel = np.concatenate([real, rng.choice(pair, len(real), replace=False)]) if (use_pairs and len(pair)) else real
        sel = rng.permutation(sel); w_all = group_weights(Gw[sel]); tl = []
        for b in range(0, len(sel), 64):
            i = sel[b:b + 64]; w = torch.from_numpy(w_all[b:b + 64]).float()
            o = fwd(model, name, torch.from_numpy(Xw[i]), torch.from_numpy(Ew[i]), torch.from_numpy(Fw[i]))
            loss = sum((F.cross_entropy(o[c], torch.from_numpy(Yw[i, c]).long(), reduction='none') * w).sum() / w.sum() for c in range(5))
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss))
        rec = dict(epoch=ep + 1, train_loss=float(np.mean(tl)), sec=time.time() - t0)
        if V is not None:
            rec.update(val_eval(model, name, V))
            if rec['val_loss'] < best[0]: best = (rec['val_loss'], ep + 1, copy.deepcopy(model.state_dict()))
            hist.append(rec)
            if ep + 1 - best[1] >= PAT: break
        else: hist.append(rec)
    if V is not None: model.load_state_dict(best[2])
    return model, hist, best[1]
SPACE = dict(lr=(1e-3, 2e-3, 5e-3), wd=(1e-4, 1e-3, 1e-2), w=(0.5, 1, 2), p=(0.0, 0.1, 0.3))
def settings(n=8, seed=0):
    allc = list(itertools.product(*SPACE.values())); rng = np.random.default_rng(seed); d = (2e-3, 1e-3, 1, 0.0); pick = [d]
    for i in rng.permutation(len(allc)):
        if allc[i] != d and len(pick) < n: pick.append(allc[i])
    return [dict(zip(SPACE, c)) for c in pick]
_C = {}
def data(phase):
    if phase not in _C:
        if phase == 'tune':
            D = build(TRS, TRS, range(1, 13)); V = build(TRS, VAL, (13, 14, 15)); mu, sd = prep(D); prep(V, mu, sd); _C[phase] = (D, V)
        else:
            D = build(POOL, POOL, range(1, 16)); mu, sd = prep(D); T = test_set(POOL); prep(T, mu, sd); _C[phase] = (D, T)
    return _C[phase]
def tune(name):
    D, V = data('tune'); S = settings(); fn = OUT + f'tune_{name}.json'; res = json.load(open(fn)) if os.path.exists(fn) else {}
    def go(k, seed):
        key = f'{k}:{seed}'
        if key in res: return
        t = time.time(); m, h, be = run(name, seed, S[k], D, V)
        res[key] = dict(hp=S[k], hist=h, best_epoch=be, best_val_loss=min(r['val_loss'] for r in h), sec=time.time() - t)
        print(name, k, seed, be, round(res[key]['best_val_loss'], 4), round(h[be - 1]['val_exact_pair'], 3), round(time.time() - t), flush=True); json.dump(res, open(fn, 'w'))
    for k in range(len(S)): go(k, 0)
    top = sorted(range(len(S)), key=lambda k: res[f'{k}:0']['best_val_loss'])[:3]
    for k in top:
        for s in (1, 2): go(k, s)
    best = min(top, key=lambda k: np.mean([res[f'{k}:{s}']['best_val_loss'] for s in range(3)]))
    res['selected'] = dict(k=best, hp=S[best], epochs=int(round(np.mean([res[f'{best}:{s}']['best_epoch'] for s in range(3)]))), top=top)
    json.dump(res, open(fn, 'w')); print('SELECTED', name, res['selected'], flush=True)
@torch.no_grad()
def record_probs(model, name, D):
    o = predict(model, name, D); return [o[c].softmax(-1).reshape(len(D['y']), 9, -1).mean(1).numpy() for c in range(5)]
def sub(Q, k): return {kk: (v[k] if isinstance(v, np.ndarray) and len(v) == len(k) else v) for kk, v in Q.items()}
def final(name, use_pairs=True, tag=''):
    J = json.load(open(OUT + f'tune_{name}.json'))['selected']; D, T = data('final'); fn = OUT + f'final_{name}{tag}.pkl'
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    for seed in range(3):
        if seed in res: continue
        t = time.time(); vrec = None
        if tag:
            Dt, Vt = data('tune'); _, h0, ep = run(name, seed, J['hp'], sub(Dt, Dt['group'] < 2), sub(Vt, Vt['group'] < 2), use_pairs=False)
        else: ep = J['epochs']
        m, h, _ = run(name, seed, J['hp'], D, None, epochs=ep, use_pairs=use_pairs)
        res[seed] = dict(test=record_probs(m, name, T), sec=time.time() - t, epochs=ep, nparam=sum(p.numel() for p in m.parameters()), hp=J['hp'])
        if not tag and seed == 0:   # validation behaviour of a tuning-phase model with the same seed (for false alarms on real validation rows)
            Dt, Vt = data('tune'); mv, hv, bev = run(name, seed, J['hp'], Dt, Vt); res[seed]['val'] = record_probs(mv, name, Vt); res[seed]['val_hist'] = hv
        torch.save(m.state_dict(), f'{(WORK + "/models")}/{name}{tag}_{seed}.pt'); pickle.dump(res, open(fn, 'wb')); print('final', name, tag, seed, ep, round(time.time() - t), flush=True)
if __name__ == '__main__':
    mode = sys.argv[1]
    for name in sys.argv[2:]:
        if mode == 'tune': tune(name)
        elif mode == 'final': final(name)
        elif mode == 'base': final(name, use_pairs=False, tag='_single')
