"""Sensor-loss analysis of the final models (no retraining): the channels of a sensor group are replaced by their healthy mean
(zero after normalisation, Eq. of the healthy standardisation); summary features are recomputed from the modified inputs.
The Bayesian reasoning layer drops the symptoms whose sensors are missing. Also: attention weights of M4 and M5 on the test sets."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, os, json, pickle, numpy as np, pandas as pd, torch
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
from data6 import *
import train6 as T6
from train6 import prep, record_probs, data
from models4 import make
from reason6 import sym_sets, SYM
from bayes6 import Sig, model_ll
OUT = (RES) + '/sensorloss/'; os.makedirs(OUT, exist_ok=True)
CH = ['A1', 'A2', 'A3', 'A4', 'Mic1', 'Mic2', 'I1', 'I2', 'I3', 'Pr1', 'Pr2', 'Pr3', 'Pr4']
GROUPS = {'none': ([], []), 'pressures': (['Pr1', 'Pr2', 'Pr3', 'Pr4'], []), 'stage-1 valve sensors (A1, Mic1)': (['A1', 'Mic1'], []),
          'stage-2 valve sensors (A2, Mic2)': (['A2', 'Mic2'], []), 'bearing accelerometers (A3, A4)': (['A3', 'A4'], [0, 1]),
          'microphones': (['Mic1', 'Mic2'], []), 'currents': (['I1', 'I2', 'I3'], [])}
SYM_NEEDS = {'dP2': ['Pr2', 'Pr4'], 'dP3': ['Pr3', 'Pr4'], 'puls1': ['Pr1'], 'puls2': ['Pr2'], 'puls3': ['Pr3'], 'env_BPFO': ['A3'], 'env_BPFI': ['A3'], 'env_FTF': ['A3']}
D, T = data('final'); mu, sd = prep(D)   # same normalisation of the summary features as the final models
TY = T['y']; TF = np.array(['valve-bearing'] * 180 + ['valve-valve'] * 120)
tidx = T['idx']; Bf, st, _ = sym_sets(POOL, tidx, []); Sf = Sig(Bf, POOL)
def masked(gch, gen):
    Q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in T.items()}
    for c in gch: Q['X'][..., CH.index(c), :] = 0.0
    for e in gen: Q['E'][..., e, :] = 0.0
    prep(Q, mu, sd); return Q
def ll_sym(gch):
    keep = [k for k, s in enumerate(SYM) if not set(SYM_NEEDS[s]) & set(gch)]
    v = st[:, keep]; MU = Sf.MU[:, keep]; b = Sf.b[keep]
    return -(np.abs(v[:, None, :] - MU[None]) / b).sum(-1)
rows = []
from rf6 import fit, probs
hpRF = json.load(open(T6.OUT + 'tune_RF.json'))['selected']['hp']; hpRF = {a: (None if b in (None, 'None') else b) for a, b in hpRF.items()}
rf = fit(D, hpRF, 0)
nets = {}
for name in ['MLP', 'CNN', 'Dense', 'GNN']:
    J = json.load(open(T6.OUT + f'tune_{name}.json'))['selected']
    for s in range(3):
        m = make(name, D['F'].shape[-1], J['hp']['w'], J['hp']['p']); m.load_state_dict(torch.load(f'{(WORK + "/models")}/{name}_{s}.pt')); m.eval(); nets[(name, s)] = m
for g, (gch, gen) in GROUPS.items():
    Q = masked(gch, gen); Ls = ll_sym(gch)
    preds = {('RF', 0): probs(rf, Q)}
    for (name, s), m in nets.items(): preds[(name, s)] = record_probs(m, name, Q)
    for (name, s), pr in preds.items():
        L0 = model_ll(pr)
        for sysn, Ltot in (('model', L0), ('+ Bayesian layer', L0 + Ls)):
            P = LABEL_Y[Ltot.argmax(1)]
            for f in ('valve-bearing', 'valve-valve'):
                k = TF == f; rows.append(dict(group=g, model=name, seed=s, system=sysn, family=f, label=float((P[k] == TY[k]).all(1).mean()),
                                              loc=float(((P[k] > 0) == (TY[k] > 0)).all(1).mean()), rec_B1=float((P[k][:, 4] > 0)[TY[k][:, 4] > 0].mean()) if (TY[k][:, 4] > 0).any() else np.nan))
    print(g, flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT + 'sensorloss.csv', index=False)
pd.set_option('display.width', 220); print(R.groupby(['family', 'group', 'system'])[['label', 'loc']].mean().round(3).unstack('system').to_string())
# attention of M4 and M5 (seed 0), mean over windows and recordings, per family
att = {}
for name in ['Dense', 'GNN']:
    m = nets[(name, 0)]; X = torch.from_numpy(T['X'].reshape(-1, 13, 72)); E = torch.from_numpy(T['E'].reshape(-1, 2, 300))
    with torch.no_grad(): _, a = m(X, E, return_att=True)
    a = a.numpy().reshape(len(TY), 9, 5, 15).mean(1)
    for f in ('valve-bearing', 'valve-valve'): att[f'{name}|{f}'] = a[TF == f].mean(0).tolist()
json.dump(att, open(OUT + 'attention.json', 'w'))
