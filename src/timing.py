"""Computational cost of the reference configuration: trainable parameters, training time of the final models
(stored, mean of three seeds; RF refitted once with timing) and inference time per test recording (9 windows),
model forward pass only, one CPU thread."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, time, pickle, platform, numpy as np, torch
sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
torch.set_num_threads(1)
from train6 import data, predict, make
import rf6
R = (RES) + '/'; out = {}
Df, T = data('final'); n = len(T['y'])
for m in ['MLP', 'CNN', 'Dense', 'GNN']:
    Rr = pickle.load(open(R + f'final_{m}.pkl', 'rb')); hp = Rr[0]['hp']
    model = make(m, Df['F'].shape[-1], hp['w'], hp['p']); model.load_state_dict(torch.load(f'{(WORK + "/models")}/{m}_0.pt')); model.eval()
    predict(model, m, T); ts = []
    for _ in range(5):
        t = time.perf_counter(); predict(model, m, T); ts.append(time.perf_counter() - t)
    out[m] = dict(nparam=int(sum(p.numel() for p in model.parameters())), train_s=float(np.mean([Rr[s]['sec'] for s in range(3)])),
                  infer_ms_per_rec=1000 * float(np.median(ts)) / n, epochs=int(Rr[0]['epochs']))
    print(m, out[m], flush=True)
J = json.load(open(R + 'tune_RF.json'))['selected']; hp = J['hp']
t = time.perf_counter(); ms = rf6.fit(Df, hp, 0); tr = time.perf_counter() - t
rf6.probs(ms, T); ts = []
for _ in range(5):
    t = time.perf_counter(); rf6.probs(ms, T); ts.append(time.perf_counter() - t)
out['RF'] = dict(nparam=None, nodes=int(sum(e.tree_.node_count for mm in ms for e in mm.estimators_)), train_s=tr, infer_ms_per_rec=1000 * float(np.median(ts)) / n)
out['cpu'] = platform.processor() or platform.machine(); out['threads'] = 1
try:
    out['cpu_model'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')][0]
except Exception: pass
json.dump(out, open(R + 'final/timing.json', 'w'), indent=1); print(out)
