import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, time, numpy as np
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
from train6 import data
import rf6
R = (RES) + '/'
Df, T = data('final'); n = len(T['y'])
J = json.load(open(R + 'tune_RF.json'))['selected']; hp = J['hp']
t = time.perf_counter(); ms = rf6.fit(Df, hp, 0); tr = time.perf_counter() - t
rf6.probs(ms, T); ts = []
for _ in range(3):
    t = time.perf_counter(); rf6.probs(ms, T); ts.append(time.perf_counter() - t)
out = dict(train_s=tr, infer_ms_per_rec=1000 * float(np.median(ts)) / n, nodes=int(sum(e.tree_.node_count for mm in ms for e in mm.estimators_)), hp=hp)
json.dump(out, open(R + 'final/timing_rf.json', 'w'), indent=1); print(out, flush=True)
