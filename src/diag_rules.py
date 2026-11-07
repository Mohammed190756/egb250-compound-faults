"""Diagnostic only: similarity of combined rows (each rule) to the real combination recordings (same measure as Fig. 11)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, numpy as np
sys.path.insert(0, (SRC))
from data6 import *
B = Builder(POOL); out = []
v = lambda X, E: np.concatenate([X.mean((0, 1)).ravel(), E.mean((0, 1)).ravel()])
for fam, idx in FAM.items():
    for lab in sorted({name_of(Y[i]) for i in idx}):
        ri = np.array([i for i in idx if name_of(Y[i]) == lab]); a, b = lab.split('+')
        items = [(REC[(a, r)], REC[(b, r)], lab) for r in range(1, 16) if (a, r) in REC and (b, r) in REC]
        R = v(*B.real(ri)); d = dict(family=fam, label=lab)
        for rule in 'abcd': d[rule] = float(np.corrcoef(R, v(*B.pairs(items, rule, np.random.default_rng(0))))[0, 1])
        out.append(d)
json.dump(out, open((RES) + '/rules/similarity_rules.json', 'w'), indent=1)
import pandas as pd; D = pd.DataFrame(out); print(D.round(2).to_string()); print(D.groupby('family')[list('abcd')].mean().round(3))
print('best rule count', D[list('abcd')].idxmax(1).value_counts().to_dict())
