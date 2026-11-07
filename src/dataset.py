import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys; sys.path.insert(0,'.')
from common import *
from angle import nz
PCH=('Pr1','Pr2','Pr3','Pr4')
def align_windows(P, ch, ref_idx, iters=2):
    """Per-window circular alignment of crank-angle profiles to a healthy pressure template (label-free:
    only the healthy reference windows define the template)."""
    iP=[ch.index(c) for c in PCH]
    N=P.reshape(-1,P.shape[2],P.shape[3])            # (rec*win, ch, 72)
    ok=np.isfinite(N).all((1,2))
    tmpl=nz(np.nanmean(N[ref_idx][:,iP],0))
    shift=np.zeros(len(N),int)
    for _ in range(iters):
        X=nz(np.nan_to_num(N[:,iP]))                 # (n,4,72)
        F=np.fft.rfft(X,axis=-1); T=np.fft.rfft(tmpl,axis=-1)
        cc=np.fft.irfft((np.conj(F)*T[None]).sum(1),n=72,axis=-1)   # circular xcorr summed over pressure channels
        shift=np.argmax(cc,axis=-1)
        A=np.stack([np.roll(N[i],shift[i],axis=-1) for i in range(len(N))])
        tmpl=nz(np.nanmean(A[ref_idx][:,iP],0))
    return A.reshape(P.shape), shift.reshape(P.shape[:2]), ok.reshape(P.shape[:2])
def build():
    m,f,feats=load()
    A=np.load(f'{D}/angle_profiles.npz',allow_pickle=True); P=A['prof']; rec=list(A['records']); ch=list(A['channels'])
    mm=m.set_index('record').loc[rec].reset_index()
    nw=P.shape[1]
    hv=np.flatnonzero((mm.campaign=='Valve')&(mm.P==1))
    ref=np.concatenate([np.arange(i*nw,(i+1)*nw) for i in hv])
    Pa,shift,ok=align_windows(P,ch,ref)
    E=np.load(f'{D}/env_orders.npz',allow_pickle=True); env=E['env'].astype(np.float32); O=E['orders']
    assert list(E['records'])==rec
    # window table aligned to (rec, win)
    W=pd.DataFrame([(r,w) for r in rec for w in range(nw)],columns=['record','win']).merge(
        f[['record','win']+feats],on=['record','win'],how='left')
    meta=mm[['record','campaign','P','R','role']+[f'y_{c}' for c in COMPS]].copy()
    meta['baseline']=np.where(meta.campaign=='Valve','Valve','BearingMulti')
    np.savez_compressed((WORK) + '/dataset.npz',prof=Pa.astype(np.float32),shift=shift,ok=ok,env=env,orders=O,channels=np.array(ch))
    meta.to_csv((WORK) + '/meta.csv',index=False); W.to_parquet((WORK) + '/winfeat.parquet')
    return Pa,shift,meta
if __name__=='__main__':
    Pa,shift,meta=build()
    meta['shift_sd']=[np.std(np.unwrap(s*2*np.pi/72))*72/(2*np.pi) for s in shift]
    meta['cond']=meta.campaign.str[0]+meta.P.astype(str).str.zfill(2)
    print(meta.groupby('cond').shift_sd.median().round(2).to_dict())
