import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
D=(FEAT)
CH=['A1','A2','A3','A4','Mic1','Mic2','I1','I2','I3','Pr1','Pr2','Pr3','Pr4']
COMPS=['1S_IVb','1S_DV','2S_IV','2S_DV','B1']
def load():
    m=pd.read_csv(f'{D}/recordings.csv'); f=pd.read_parquet(f'{D}/window_features.parquet')
    m['baseline']=np.where(m.campaign=='Valve','Valve','BearingMulti')
    f=f.merge(m[['record','campaign','P','R','role','baseline','timestamp']+[f'y_{c}' for c in COMPS]],on='record')
    feats=[c for c in f.columns if c.split('_')[0] in CH]
    f[feats]=f[feats].replace([np.inf,-np.inf],np.nan)
    return m,f,feats
def zscore_vs_baseline(f,feats):
    """robust z vs the healthy windows of the record's own baseline campaign (label-free at deployment: healthy reference after (re)assembly)."""
    Z=f.copy()
    for b,g in f.groupby('baseline'):
        h=g[g.role=='healthy'][feats]; med=h.median(); mad=(h-med).abs().median()*1.4826
        sd=mad.where(mad>1e-9, h.std()+1e-9)
        Z.loc[g.index,feats]=((g[feats]-med)/sd).values
    return Z
