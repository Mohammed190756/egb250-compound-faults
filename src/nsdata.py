import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import numpy as np, pandas as pd
COMPS=['1S_IVb','1S_DV','2S_IV','2S_DV','B1']; NMODE=[5,5,5,5,4]
SENS=['A1','A2','A3','A4','Mic1','Mic2','I1','I2','I3','Pr1','Pr2','Pr3','Pr4','A3env','A4env']
# ---------------- ontology: physical evidence links (sensor -> component) ----------------
LINKS={'1S_IVb':['Pr1','Pr2','A1','Mic1'],
       '1S_DV': ['Pr2','A1','Mic1'],
       '2S_IV': ['Pr2','A2','Mic2'],            # no pressure transducer at 2S_IV (indirect via Pr2)
       '2S_DV': ['Pr3','Pr2','A2','Mic2'],
       'B1':    ['A3','A3env','A4','A4env']}
COUPLING=[('1S_IVb','1S_DV'),('2S_IV','2S_DV'),('1S_DV','2S_IV')]   # same stage / interstage volume
def load(path=(WORK)):
    Z=np.load(f'{path}/dataset.npz',allow_pickle=True); meta=pd.read_csv(f'{path}/meta.csv')
    P=Z['prof']; env=Z['env']; ch=list(Z['channels']); nrec,nw=P.shape[:2]
    rms=[c for c in ch if not c.startswith('Pr')]
    X=P.copy()
    for c in rms: X[:,:,ch.index(c)]=np.log(np.maximum(X[:,:,ch.index(c)],1e-6))
    E=np.log(np.maximum(env[:,:,2:4,0,:].astype(np.float32),1e-6))   # A3,A4 low band (n,w,2,300)
    # baseline normalisation (healthy windows of the record's own baseline campaign = deployment reference)
    for b in meta.baseline.unique():
        r=np.flatnonzero(meta.baseline==b); h=np.flatnonzero((meta.baseline==b)&(meta.role=='healthy'))
        mu=np.nanmean(X[h],axis=(0,1)); X[r]-=mu
        mue=np.nanmean(E[h],axis=(0,1)); E[r]-=mue
    hsd=np.nanstd(X[meta.role=='healthy'],axis=(0,1,3),keepdims=True)[0]   # per-channel healthy sd
    X/= (hsd+1e-6)
    esd=np.nanstd(E[meta.role=='healthy'],axis=(0,1,3),keepdims=True)[0]; E/=(esd+1e-6)
    X=np.nan_to_num(X).clip(-30,30).astype(np.float32); E=np.nan_to_num(E).clip(-30,30).astype(np.float32)
    Y=meta[[f'y_{c}' for c in COMPS]].values
    return X,E,Y,meta,ch
