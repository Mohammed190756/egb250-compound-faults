"""Comparison of combining rules (M-a deviation sum, M-c dominant fault, M-d varied severity) against M-b.
Same data, labels, balance, models and test sets as the main protocol; hyperparameters and training length reused from the M-b search.
M-d: training rows get fresh severities every epoch; validation rows use one fixed draw."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, time, copy, pickle, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
import train6 as T6
from data6 import *
from train6 import prep, group_weights, val_eval, record_probs
from models4 import make, fwd
torch.set_num_threads(1)
OUT = (RES) + '/rules/'; os.makedirs(OUT, exist_ok=True); os.makedirs((WORK + '/models') + '/rules', exist_ok=True)
_C = {}
def data(phase, rule):
    key = (phase, rule)
    if key not in _C:
        if phase == 'tune':
            D = build(TRS, TRS, range(1, 13), rule, 0); V = build(TRS, VAL, (13, 14, 15), rule, 123); mu, sd = prep(D); prep(V, mu, sd); D['_fn'] = (mu, sd); _C[key] = (D, V)
        else:
            D = build(POOL, POOL, range(1, 16), rule, 0); mu, sd = prep(D); D['_fn'] = (mu, sd); Ts = test_set(POOL); prep(Ts, mu, sd); _C[key] = (D, Ts)
    return _C[key]
def regen(D, rule, rng):
    """Fresh M-d draw for every combined row, written in place (window views stay valid)."""
    if rule != 'd': return
    n0 = D['_n0']; X, E = D['_B'].pairs(D['_items'], 'd', rng); mu, sd = D['_fn']
    D['X'][n0:] = X; D['E'][n0:] = E; D['F'][n0:] = np.nan_to_num((summary(X, E) - mu) / sd).clip(-30, 30).astype(np.float32)
def run(name, seed, hp, D, rule, V=None, epochs=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); grng = np.random.default_rng(1000 + seed)
    nf = D['F'].shape[-1]; model = make(name, nf, hp['w'], hp['p']); opt = torch.optim.AdamW(model.parameters(), hp['lr'], weight_decay=hp['wd'])
    Xw = D['X'].reshape(-1, 13, 72); Ew = D['E'].reshape(-1, 2, 300); Fw = D['F'].reshape(-1, nf); Yw = np.repeat(D['y'], 9, 0); Gw = np.repeat(D['group'], 9)
    real = np.flatnonzero(Gw < 2); pair = np.flatnonzero(Gw == 2); hist = []; best = (np.inf, 0, None)
    for ep in range(epochs or T6.MAXEP):
        t0 = time.time(); regen(D, rule, grng); model.train()
        sel = rng.permutation(np.concatenate([real, rng.choice(pair, len(real), replace=False)])); w_all = group_weights(Gw[sel]); tl = []
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
            if ep + 1 - best[1] >= T6.PAT: break
        else: hist.append(rec)
    if V is not None: model.load_state_dict(best[2])
    return model, hist, best[1]
def final(name, rule):
    J = json.load(open(T6.OUT + f'tune_{name}.json'))['selected']; fn = OUT + f'final_{name}_{rule}.pkl'
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    for seed in range(3):
        if seed in res: continue
        t = time.time(); D, Ts = data('final', rule); m, h, _ = run(name, seed, J['hp'], D, rule, None, epochs=J['epochs'])
        res[seed] = dict(test=record_probs(m, name, Ts), sec=time.time() - t, epochs=J['epochs'], hp=J['hp'], hist=h)
        if seed == 0:
            Dt, Vt = data('tune', rule); mv, hv, bev = run(name, seed, J['hp'], Dt, rule, Vt); res[seed]['val'] = record_probs(mv, name, Vt); res[seed]['val_hist'] = hv; res[seed]['val_best_epoch'] = bev
        torch.save(m.state_dict(), f'{(WORK + "/models")}/rules/{name}_{rule}_{seed}.pt'); pickle.dump(res, open(fn, 'wb'))
        print('final', name, rule, seed, J['epochs'], round(time.time() - t), flush=True)
def final_rf(rule):
    from rf6 import fit, probs, val_eval as rf_val
    hp = json.load(open(T6.OUT + 'tune_RF.json'))['selected']['hp']; hp = {k: (None if v in (None, 'None') else v) for k, v in hp.items()}
    fn = OUT + f'final_RF_{rule}.pkl'; res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    Dt, Vt = data('tune', rule); Df, Ts = data('final', rule)
    for s in range(3):
        if s in res: continue
        t = time.time(); r, vp = rf_val(fit(Dt, hp, s), Vt); res[s] = dict(test=probs(fit(Df, hp, s), Ts), val=vp, valm=r, sec=time.time() - t, hp=hp)
        pickle.dump(res, open(fn, 'wb')); print('final RF', rule, s, round(time.time() - t), flush=True)
if __name__ == '__main__':
    rules = sys.argv[1]; names = sys.argv[2:]
    for rule in rules:
        _C.clear()
        for name in names:
            final_rf(rule) if name == 'RF' else final(name, rule)
