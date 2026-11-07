import sys; sys.path.insert(0,'.')
from common import *
def symptoms(f):
    """Physics-defined symptoms per window, expressed relative to the record's baseline-campaign healthy windows.
    Units: robust effect size (difference / healthy std pooled over both baselines) or log-ratio."""
    S=pd.DataFrame(index=f.index)
    base=f.copy()
    base['r2']=f.Pr2_mean/f.Pr4_mean          # interstage pressure normalised by receiver pressure
    base['r3']=f.Pr3_mean/f.Pr4_mean
    defs={'dP2':('r2','diff'),'dP3':('r3','diff'),'puls2':('Pr2_std','log'),'puls3':('Pr3_std','log'),
          'puls1':('Pr1_std','log'),'vib1':('A1_std','log'),'vib2':('A2_std','log'),'vib4':('A4_std','log'),
          'mic1':('Mic1_std','log'),'mic2':('Mic2_std','log'),'cur':('I1_std','log')}
    H=base[f.role=='healthy']
    for k,(c,kind) in defs.items():
        sd=(np.log(H[c]) if kind=='log' else H[c]).groupby(H.baseline).std().mean()
        for b,g in base.groupby('baseline'):
            h=H[H.baseline==b][c]
            v=(np.log(g[c])-np.log(h.median())) if kind=='log' else (g[c]-h.median())
            S.loc[g.index,k]=(v/sd).values
    return S
def env_symptoms(f):
    e=np.load(f'{D}/env_orders.npz',allow_pickle=True); E=e['env'].astype(np.float32); O=e['orders']; rec=list(e['records'])
    ri=np.array([rec.index(r) for r in f.record]); wi=f.win.values
    out={}
    for nm,o in (('BPFO',7.15),('BPFI',9.85),('BPFI_sb',8.85),('FTF',0.42)):
        k=(O>o-0.1)&(O<o+0.1); out[nm]=E[ri,wi,2,0][:,k].max(1)
    P=pd.DataFrame(out,index=f.index)
    S=pd.DataFrame(index=f.index)
    for b,g in f.groupby('baseline'):
        hm=P.loc[g.index[g.role=='healthy']]
        for c in P: S.loc[g.index,'env_'+c]=(np.log(P.loc[g.index,c])-np.log(hm[c]).median())/np.log(hm[c]).std()
    return S
