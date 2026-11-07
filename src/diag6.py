"""Diagnostic only (no choice depends on it): similarity of combined rows to the real combination recordings."""
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
        Xr, Er = B.real(ri); Xp, Ep = B.pairs(items)
        Xa, Ea = B.real(np.array([REC[(a, r)] for r in range(1, 16) if (a, r) in REC])); Xb, Eb = B.real(np.array([REC[(b, r)] for r in range(1, 16) if (b, r) in REC]))
        R = v(Xr, Er); out.append(dict(family=fam, label=lab, combined=float(np.corrcoef(R, v(Xp, Ep))[0, 1]), singleA=float(np.corrcoef(R, v(Xa, Ea))[0, 1]), singleB=float(np.corrcoef(R, v(Xb, Eb))[0, 1])))
json.dump(out, open((RES) + '/superposition6.json', 'w'), indent=1)
for o in out: print(o['label'], round(o['combined'], 2), round(o['singleA'], 2), round(o['singleB'], 2))
