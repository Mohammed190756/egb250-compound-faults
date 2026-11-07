"""Split: healthy reference = Bearing P1 only; Valve P1 excluded. Pool = Bearing P1 healthy + bearing singles + valve singles (299).
Validation = R13-R15. Test = Multi P2-P13 (valve+bearing) and Valve P18-P25 (valve+valve)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from nsdata import COMPS, NMODE, SENS, LINKS, COUPLING
P2 = (WORK); VAL_REPS = (13, 14, 15)
meta = pd.read_csv(f'{P2}/meta.csv')
import os
SETUP = os.environ.get('P6SETUP', 'A')   # A: Bearing P1 only (main); B: Valve P1 only; C1/C2: both healthy sessions
VALVE_H = ((meta.campaign == 'Valve') & (meta.role == 'healthy')).values
BEAR_H = ((meta.campaign == 'Bearing') & (meta.role == 'healthy')).values
EXCL = {'A': VALVE_H, 'B': BEAR_H, 'C1': np.zeros(len(meta), bool), 'C2': np.zeros(len(meta), bool)}[SETUP]
POOL = ((meta.role != 'combo').values) & ~EXCL
VAL = POOL & meta.R.isin(VAL_REPS).values
TRS = POOL & ~VAL
HEALTHY = POOL & (meta.role == 'healthy').values
FAM = {'valve-bearing': np.flatnonzero((meta.role == 'combo').values & (meta.campaign == 'Multi').values),
       'valve-valve': np.flatnonzero((meta.role == 'combo').values & (meta.campaign == 'Valve').values)}
Y = meta[[f'y_{c}' for c in COMPS]].values
def align(ref_mask):
    from dataset import align_windows
    from common import D
    A = np.load(f'{D}/angle_profiles.npz', allow_pickle=True); Pr = A['prof']; rec = list(A['records']); ch = list(A['channels'])
    order = [rec.index(r) for r in meta.record]; Pr = Pr[order]; nw = Pr.shape[1]
    ref = np.concatenate([np.arange(i * nw, (i + 1) * nw) for i in np.flatnonzero(ref_mask)])
    Pa, shift, ok = align_windows(Pr, ch, ref)
    return Pa.astype(np.float32), shift, ch
