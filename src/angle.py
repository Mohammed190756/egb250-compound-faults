import sys; sys.path.insert(0,'.')
from common import *
def nz(x): x=x-x.mean(-1,keepdims=True); return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)
def load_profiles(m):
    A=np.load(f'{D}/angle_profiles.npz',allow_pickle=True); P=A['prof'].copy(); rec=list(A['records']); ch=list(A['channels'])
    mm=m.set_index('record').loc[rec]
    iP=[ch.index(c) for c in ('Pr1','Pr2','Pr3','Pr4')]
    tmpl=nz(np.nanmean(P[np.flatnonzero((mm.campaign=='Valve')&(mm.P==1))][:,:,iP],axis=(0,1)))
    shift=np.zeros(len(rec),int)
    for i in range(len(rec)):   # encoder index angle changes with every reassembly -> align ALL records (label-free)
        x=nz(np.nanmean(P[i][:,iP],0)); sc=[(np.roll(x,s,axis=-1)*tmpl).sum() for s in range(72)]
        shift[i]=int(np.argmax(sc)); P[i]=np.roll(P[i],shift[i],axis=-1)
    return P,rec,ch,shift
