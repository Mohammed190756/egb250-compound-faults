"""Additivity of real compound faults (diagnostic; no model or setting depends on it).
Deviation quantities in each channel's combining domain (record means over the nine windows):
  pressure Pr1-Pr4:          q = P - mean healthy P
  A1-A4, Mic1-2, I1-I3:      q = R^2 - mean healthy R^2
  envelope spectra A3, A4:   q = E^2 - mean healthy E^2
For every real combination AB with single faults A and B, and repetition r:
  repeat ratio rho = median_r ||q_AB,r - (q_A,r + q_B,r)|| / median_{r != r'} ||q_AB,r - q_AB,r'||
  control/affected error: mean squared error in cells where neither single fault deviates (session offset) vs cells where one does
  bounds band (energy channels): q_A + q_B - 2 sqrt(q_A q_B) <= q_AB <= q_A + q_B + 2 sqrt(q_A q_B) for q_A, q_B > 0 (|C_AB| <= D_A D_B)
  pressure fit: Delta_AB = alpha Delta_A + beta Delta_B (least squares over Pr1-Pr4 and bins), residual share."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, (SRC))
from data6 import *
B = Builder(POOL); ch = B.ch; ip, idn = B.ip, B.idn
CHN = ch + ['A3 env', 'A4 env']
def q_of(i):
    """(9, 15, L) deviation quantities of recording i; L = 72 for profiles, 300 for spectra (kept separately)."""
    P = B.P[i]; E = B.E[i]; qp = np.empty_like(P); qp[:, ip] = P[:, ip] - B.Hp[ip]; qp[:, idn] = P[:, idn] ** 2 - B.H2[idn]
    return np.nanmean(qp, 0), np.nanmean(E ** 2 - B.HE2, 0)   # (13, 72), (2, 300)
def scale(i_list):
    """Healthy repeat scale per channel, to express all channels in comparable units."""
    Q = [q_of(i) for i in i_list]; sp = np.std([q[0] for q in Q], 0).mean(-1); se = np.std([q[1] for q in Q], 0).mean(-1); return np.concatenate([sp, se]) + 1e-12
H = np.flatnonzero(HEALTHY); SC = scale(H)
COND = (meta.campaign + '_P' + meta.P.astype(str)).values
rows, cells = [], []
for fam, idx in FAM.items():
    for c in sorted(set(COND[idx]), key=lambda s: int(s.split('P')[-1])):
        k = idx[COND[idx] == c]; lab = name_of(Y[k[0]]); a, b = lab.split('+')
        reps = [int(meta.R[i]) for i in k if (a, int(meta.R[i])) in REC and (b, int(meta.R[i])) in REC]
        AB = {int(meta.R[i]): q_of(i) for i in k}; QA = {r: q_of(REC[(a, r)]) for r in reps}; QB = {r: q_of(REC[(b, r)]) for r in reps}
        sa = np.std([QA[r][0] for r in reps], 0); sb = np.std([QB[r][0] for r in reps], 0)
        mA = np.mean([QA[r][0] for r in reps], 0); mB = np.mean([QB[r][0] for r in reps], 0)
        eA = np.mean([QA[r][1] for r in reps], 0); eB = np.mean([QB[r][1] for r in reps], 0)
        sea = np.std([QA[r][1] for r in reps], 0); seb = np.std([QB[r][1] for r in reps], 0)
        d = dict(family=fam, cond=c.replace('Multi_P', 'MF-P').replace('Valve_P', 'V-P'), label=lab)
        for j in range(15):
            prof = j < 13; get = (lambda q: q[0][j]) if prof else (lambda q: q[1][j - 13])
            err = [np.linalg.norm(get(AB[r]) - get(QA[r]) - get(QB[r])) for r in reps]
            rr = [np.linalg.norm(get(AB[r1]) - get(AB[r2])) for r1 in AB for r2 in AB if r1 < r2]
            d[f'rho_{CHN[j]}'] = float(np.median(err) / (np.median(rr) + 1e-12))
            # control vs affected cells (mean over repetitions)
            mAj, mBj = (mA[j], mB[j]) if prof else (eA[j - 13], eB[j - 13]); sAj, sBj = (sa[j], sb[j]) if prof else (sea[j - 13], seb[j - 13])
            hs = SC[j]; ctrl = (np.abs(mAj) < 2 * hs) & (np.abs(mBj) < 2 * hs)
            mAB = np.mean([get(AB[r]) for r in AB], 0); e2 = (mAB - mAj - mBj) ** 2
            d[f'ctrl_{CHN[j]}'] = float(e2[ctrl].mean()) if ctrl.any() else np.nan; d[f'aff_{CHN[j]}'] = float(e2[~ctrl].mean()) if (~ctrl).any() else np.nan
            if j in idn or not prof:   # bounds band for energy quantities
                pos = (mAj > 2 * hs) & (mBj > 2 * hs)
                if pos.any():
                    lo = mAj + mBj - 2 * np.sqrt(mAj * mBj); hi = mAj + mBj + 2 * np.sqrt(mAj * mBj)
                    d[f'band_{CHN[j]}'] = float(((mAB >= lo) & (mAB <= hi))[pos].mean()); d[f'nband_{CHN[j]}'] = int(pos.sum())
                    d[f'above_{CHN[j]}'] = float((mAB > hi)[pos].mean()); d[f'below_{CHN[j]}'] = float((mAB < lo)[pos].mean())
        # pressure fit
        DA = np.concatenate([mA[j] for j in ip]); DB = np.concatenate([mB[j] for j in ip]); DAB = np.concatenate([np.mean([AB[r][0][j] for r in AB], 0) for j in ip])
        X = np.stack([DA, DB], 1); coef, *_ = np.linalg.lstsq(X, DAB, rcond=None); res = DAB - X @ coef
        d.update(alpha=float(coef[0]), beta=float(coef[1]), p_resid_share=float((res ** 2).sum() / ((DAB - DAB.mean()) ** 2).sum()),
                 p_add_share=float(((DAB - DA - DB) ** 2).sum() / ((DAB - DAB.mean()) ** 2).sum()))
        rows.append(d)
D = pd.DataFrame(rows); D.to_csv((RES) + '/additivity.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
rho = D[[c for c in D if c.startswith('rho_')]]; print(D[['cond', 'label']].join(rho.round(1)).to_string())
print(D[['cond', 'alpha', 'beta', 'p_resid_share', 'p_add_share']].round(2).to_string())
bc = [c for c in D if c.startswith('band_')]; print(D[['cond'] + bc].round(2).to_string())
ab = [c for c in D if c.startswith('above_')]; bl = [c for c in D if c.startswith('below_')]
print('above', D.groupby('family')[ab].mean().round(2).to_string()); print('below', D.groupby('family')[bl].mean().round(2).to_string())
cr = np.nanmean(D[[c for c in D if c.startswith('ctrl_')]].values / D[[c for c in D if c.startswith('aff_')]].values, 1); D['ctrl_aff'] = cr; print(D[['cond', 'ctrl_aff']].round(2).to_string())
D.to_csv((RES) + '/additivity.csv', index=False)
