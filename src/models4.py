import numpy as np
"""Model zoo with tunable width (w) and dropout (p)."""
import torch, torch.nn as nn, torch.nn.functional as F
from data4 import COMPS, NMODE, SENS, LINKS, COUPLING
def R(x, w): return max(4, int(round(x * w)))
class Heads(nn.Module):
    def __init__(s, d, w=1, p=0.0):
        super().__init__(); s.h = nn.ModuleList([nn.Sequential(nn.Dropout(p), nn.Linear(d, R(32, w)), nn.GELU(), nn.Linear(R(32, w), n)) for n in NMODE])
    def forward(s, z): return [h(z) for h in s.h]
class MLP(nn.Module):
    def __init__(s, nf, w=1, p=0.0):
        super().__init__(); a, b = R(128, w), R(64, w)
        s.body = nn.Sequential(nn.Linear(nf, a), nn.GELU(), nn.Dropout(p), nn.Linear(a, b), nn.GELU()); s.heads = Heads(b, w, p)
    def forward(s, f): return s.heads(s.body(f))
class CNN(nn.Module):
    def __init__(s, w=1, p=0.0):
        super().__init__(); c1, c2, e1, e2, d = R(32, w), R(64, w), R(16, w), R(32, w), R(64, w)
        s.prof = nn.Sequential(nn.Conv1d(13, c1, 5, padding=2), nn.GELU(), nn.MaxPool1d(2), nn.Conv1d(c1, c2, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        s.env = nn.Sequential(nn.Conv1d(2, e1, 9, padding=4), nn.GELU(), nn.MaxPool1d(4), nn.Conv1d(e1, e2, 9, padding=4), nn.GELU(), nn.AdaptiveMaxPool1d(8), nn.Flatten(), nn.Linear(8 * e2, d))
        s.heads = Heads(c2 + d, w, p)
    def forward(s, x, e): return s.heads(torch.cat([s.prof(x), s.env(e)], 1))
class SensorEnc(nn.Module):
    def __init__(s, d=32, nch=13):
        super().__init__(); s.emb = nn.Embedding(nch, 8); s.gru = nn.GRU(1 + 8, d, batch_first=True); s.out = nn.Linear(d, d)
    def forward(s, x):
        x = F.avg_pool1d(x, 2); B, C, L = x.shape; e = s.emb.weight[None, :, None, :].expand(B, C, L, 8)
        h, _ = s.gru(torch.cat([x[..., None], e], -1).reshape(B * C, L, 9)); return s.out(h.mean(1)).reshape(B, C, -1)
class EnvEnc(nn.Module):
    def __init__(s, d=32, w=1):
        super().__init__(); e1, e2 = R(16, w), R(32, w)
        s.net = nn.Sequential(nn.Conv1d(1, e1, 9, padding=4), nn.GELU(), nn.MaxPool1d(4), nn.Conv1d(e1, e2, 9, padding=4), nn.GELU(), nn.AdaptiveMaxPool1d(8), nn.Flatten(), nn.Linear(8 * e2, d))
    def forward(s, e): B = e.shape[0]; return s.net(e.reshape(B * 2, 1, -1)).reshape(B, 2, -1)
class Diagnoser(nn.Module):
    def __init__(s, mode='gnn', w=1, p=0.0):
        super().__init__(); d = R(32, w); s.mode = mode; s.se = SensorEnc(d); s.ee = EnvEnc(d, w)
        s.q = nn.Parameter(torch.randn(len(COMPS), d) * 0.1); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d)
        mask = torch.zeros(len(COMPS), len(SENS), dtype=torch.bool)
        import os; rs = os.environ.get('P6MASK')   # random mask with the same number of sensors per component (control for M5)
        g = np.random.default_rng(int(rs)) if rs else None
        for i, c in enumerate(COMPS):
            links = LINKS[c] if mode == 'gnn' else SENS
            if g is not None and mode == 'gnn': links = list(g.choice(SENS, len(LINKS[c]), replace=False))
            for sn in links: mask[i, SENS.index(sn)] = True
        s.register_buffer('mask', mask)
        A = torch.eye(len(COMPS))
        if mode == 'gnn':
            cp = COUPLING if g is None else [tuple(COMPS[j] for j in g.choice(5, 2, replace=False)) for _ in range(len(COUPLING))]
            for a, b in cp: A[COMPS.index(a), COMPS.index(b)] = A[COMPS.index(b), COMPS.index(a)] = 1
        else: A = torch.ones(len(COMPS), len(COMPS))
        s.register_buffer('A', A / A.sum(1, keepdim=True)); s.mp = nn.Linear(2 * d, d); s.drop = nn.Dropout(p)
        s.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n)) for n in NMODE])
    def forward(s, x, e, return_att=False):
        h = torch.cat([s.se(x), s.ee(e)], 1)
        att = (s.q[None] @ s.k(h).transpose(1, 2)) / h.shape[-1] ** 0.5
        att = att.masked_fill(~s.mask[None], -1e9).softmax(-1)
        c = att @ s.v(h); c = F.gelu(s.mp(torch.cat([c, s.A[None] @ c], -1))); c = s.drop(c)
        out = [hd(c[:, i]) for i, hd in enumerate(s.heads)]
        return (out, att) if return_att else out
def make(name, nf=225, w=1, p=0.0):
    return {'MLP': lambda: MLP(nf, w, p), 'CNN': lambda: CNN(w, p), 'Dense': lambda: Diagnoser('dense', w, p), 'GNN': lambda: Diagnoser('gnn', w, p)}[name]()
def fwd(model, name, x, e, f): return model(f) if name == 'MLP' else model(x, e)
