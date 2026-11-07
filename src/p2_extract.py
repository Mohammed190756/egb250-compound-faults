"""
EGB-250 compound-fault study: feature extraction from the raw recordings.

Run on the PC that holds the raw data (Anaconda Python 3.9+):
    python p2_extract.py  <raw data folder>  <feature output folder>  [n_processes]

Conventions of the EGB-250 recordings:
  * Mic, current and accelerometer channels: 51,200 S/s (500,000 samples = 9.7656 s)
  * Pressure: 25,000 S/s (10 s); crankshaft reference E1: 500,000 S/s, one rising edge per rev
  * Currents stored in V -> x10 A/V. Valve P1-P10: I1 inverted; valve P11-P17: I2 inverted.
  * Valve P19-P25: E1-derived speed biased low (encoder slip) -> shaft speed from pressure.
  * Valve R8P3 duplicates R7P3 -> skipped. The multi-fault P1 healthy baseline is the same
    acquisition as bearing P1; only bearing P1 is used (a multi-fault P1 folder, if present,
    is skipped).
  * A1/A2 samples at the +-502.4 m/s^2 acquisition limit are counted and flagged.

Outputs (in OUT):
  recordings.csv          one row per record: labels, timestamps, speed estimates, QC flags
  window_features.parquet one row per 1-s window (9 per record): time/frequency descriptors
  env_orders.npz          envelope order spectra of A1-A4 (2 bands) per window, 0-15 orders
  angle_profiles.npz      crank-angle profiles (72 x 5 deg) per window for all 13 channels
  fingerprints.csv        content hash per record (duplicate check)
  extract_log.txt         run log
"""
import os, sys, re, glob, json, hashlib, datetime, time, traceback
import numpy as np, pandas as pd, scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import skew, kurtosis
from multiprocessing import Pool

FS50, FS25, FS500 = 51200.0, 25000.0, 500000.0
WIN_S, NWIN = 1.0, 9
NBIN = 72                                   # crank-angle bins (5 deg)
ORD = np.round(np.arange(0.05, 15.0001, 0.05), 4)   # envelope order grid
ENV_BANDS = {'lo': (2000.0, 8000.0), 'hi': (8000.0, 14000.0)}
SAT = 502.0                                 # m/s^2, acquisition limit is +-502.4

CH50 = [('A1', 'TDMSAnalog50kAcelerometer0'), ('A2', 'TDMSAnalog50kAcelerometer1'),
        ('A3', 'TDMSAnalog50kAcelerometer2'), ('A4', 'TDMSAnalog50kAcelerometer3'),
        ('Mic1', 'TDMSAnalog50kMic0'), ('Mic2', 'TDMSAnalog50kMic1'),
        ('I1', 'TDMSAnalog50kCurrentClamp0'), ('I2', 'TDMSAnalog50kCurrentClamp1'),
        ('I3', 'TDMSAnalog50kCurrentClamp2')]
CH25 = [('Pr1', 'TDMSAnalog25kPressure0'), ('Pr2', 'TDMSAnalog25kPressure1'),
        ('Pr3', 'TDMSAnalog25kPressure2'), ('Pr4', 'TDMSAnalog25kPressure3')]
CHS = [c for c, _ in CH50] + [c for c, _ in CH25]
ACC = ['A1', 'A2', 'A3', 'A4']

# ---------------------------------------------------------------- labels
VALVES = ['1S_IVb', '1S_DV', '2S_IV', '2S_DV']
def labels(campaign, p):
    lab = {v: 0 for v in VALVES}; lab['B1'] = 0
    if campaign == 'Valve':
        if 2 <= p <= 17:
            lab[['2S_DV', '2S_IV', '1S_DV', '1S_IVb'][(p - 2) // 4]] = (p - 2) % 4 + 1
        elif 18 <= p <= 21:
            lab['2S_IV'] = lab['2S_DV'] = p - 17
        elif 22 <= p <= 25:
            lab['1S_IVb'] = lab['2S_DV'] = p - 21
    elif campaign == 'Bearing':
        if p >= 2: lab['B1'] = p - 1
    elif campaign == 'Multi':
        lab['2S_DV'] = (p - 2) % 4 + 1; lab['B1'] = (p - 2) // 4 + 1
    return lab

def split_role(campaign, p):
    """Compositional protocol: healthy + single faults = 'single'; combinations = 'combo'."""
    if campaign == 'Multi' or (campaign == 'Valve' and p >= 18):
        return 'combo'
    return 'healthy' if p == 1 else 'single'

def inverted_clamp(campaign, p):
    if campaign == 'Valve' and 1 <= p <= 10: return 'I1'
    if campaign == 'Valve' and 11 <= p <= 17: return 'I2'
    return ''

# ---------------------------------------------------------------- helpers
def peak_freq(x, fs, lo, hi, pad=8):
    x = x - x.mean(); n = len(x); N = 1 << int(np.ceil(np.log2(n)) + pad // 4 + 1)
    X = np.abs(np.fft.rfft(x * np.hanning(n), N)); f = np.fft.rfftfreq(N, 1 / fs)
    m = np.flatnonzero((f > lo) & (f < hi)); i = m[np.argmax(X[m])]
    if 0 < i < len(X) - 1:                                  # parabolic refinement
        a, b, c = np.log(X[i - 1] + 1e-30), np.log(X[i] + 1e-30), np.log(X[i + 1] + 1e-30)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        return f[i] + d * (f[1] - f[0])
    return f[i]

def shaft_from_pressure(pr, fs=FS25):
    """Shaft speed from harmonics 1-5 of the stage pressures (encoder-independent)."""
    est = []
    for x in pr:
        f1 = peak_freq(x, fs, 11.0, 15.0)
        for k in range(1, 6):
            fk = peak_freq(x, fs, k * f1 - 0.5, k * f1 + 0.5)
            est.append(fk / k)
    return float(np.median(est))

def descriptors(x, fs, f_c, pressure=False):
    out = {}
    mu = x.mean(); xc = x - mu
    sd = xc.std(ddof=1); rms = np.sqrt(np.mean(xc ** 2)); ab = np.abs(xc); mab = ab.mean() + 1e-30
    pk = ab.max()
    out.update(std=sd, skew=skew(xc), kurt=kurtosis(xc, fisher=False), p2p=x.max() - x.min(),
               crest=pk / (rms + 1e-30), shape=rms / mab, impulse=pk / mab)
    X = np.abs(np.fft.rfft(xc * np.hanning(len(xc)))); f = np.fft.rfftfreq(len(xc), 1 / fs)
    P = X ** 2; band = f <= min(20000.0, fs / 2)
    p = P[band] / (P[band].sum() + 1e-30)
    out['centroid'] = float((f[band] * p).sum())
    out['spec_kurt'] = float(kurtosis(P[band], fisher=False))
    for k in range(1, 9):                                   # shaft-order amplitudes
        m = (f > (k - 0.3) * f_c) & (f < (k + 0.3) * f_c)
        out[f'o{k}'] = float(X[m].max() * 2 / len(xc)) if m.any() else np.nan
    if pressure:
        out['mean'] = mu
    return out

def angle_of(t, edges_t, f_true, use_encoder):
    """Shaft revolutions (float) at times t."""
    if use_encoder:
        rev = np.interp(t, edges_t, np.arange(len(edges_t)), left=np.nan, right=np.nan)
    else:   # encoder slipped: linear phase at the pressure-derived speed, anchored at 1st edge
        rev = (t - edges_t[0]) * f_true
        rev[t < edges_t[0]] = np.nan
    return rev

def profiles(x, t, rev, w, use_rms):
    """Mean (or RMS) of x in 72 angle bins over complete revolutions inside window w."""
    m = (t >= w * WIN_S) & (t < (w + 1) * WIN_S) & np.isfinite(rev)
    if m.sum() < 10: return np.full(NBIN, np.nan, np.float32), 0
    r = rev[m]; r0, r1 = np.ceil(r.min()), np.floor(r.max())
    k = (r >= r0) & (r < r1)
    if r1 - r0 < 1: return np.full(NBIN, np.nan, np.float32), 0
    b = ((r[k] % 1.0) * NBIN).astype(int).clip(0, NBIN - 1)
    v = x[m][k]
    if use_rms:
        s = np.bincount(b, v ** 2, NBIN); n = np.bincount(b, None, NBIN)
        return np.sqrt(s / np.maximum(n, 1)).astype(np.float32), int(r1 - r0)
    s = np.bincount(b, v, NBIN); n = np.bincount(b, None, NBIN)
    return (s / np.maximum(n, 1)).astype(np.float32), int(r1 - r0)

def env_order_spectrum(xf, fs, f_true, w):
    n = int(fs * WIN_S); seg = xf[w * n:(w + 1) * n]
    e = np.abs(hilbert(seg)); e = e - e.mean()
    N = 4 * n
    E = np.abs(np.fft.rfft(e * np.hanning(n), N)) * 2 / n
    f = np.fft.rfftfreq(N, 1 / fs)
    return np.interp(ORD * f_true, f, E).astype(np.float16)

# ---------------------------------------------------------------- one record
def campaign_of(path):
    p = path.replace('\\', '/')
    for key, name in (('Valve_Faults', 'Valve'), ('Bearing_Faults', 'Bearing'), ('Multi_Faults', 'Multi')):
        if key in p: return name
    return None

def process(path):
    camp = campaign_of(path)
    fn = os.path.basename(path)
    mt = re.match(r'R(\d+)P(\d+)\.mat$', fn)
    r_i, p_i = int(mt.group(1)), int(mt.group(2))
    M = sio.loadmat(path)
    hdr = M['__header__'].decode(errors='ignore')
    try:
        ts = datetime.datetime.strptime(hdr.split('Created on: ')[1].strip(), '%a %b %d %H:%M:%S %Y')
    except Exception:
        ts = pd.NaT
    d = M['data'][0, 0]
    g50, g25 = d['Analog50k'][0, 0], d['Analog25k'][0, 0]
    sig = {c: g50[k].ravel().astype(np.float64) for c, k in CH50}
    sig.update({c: g25[k].ravel().astype(np.float64) for c, k in CH25})
    for c in ('I1', 'I2', 'I3'):
        sig[c] = sig[c] * 10.0                                   # V -> A
    inv = inverted_clamp(camp, p_i)
    if inv:
        sig[inv] = -sig[inv]                                     # restore 3-phase relationship
    enc = d['Analog500k'][0, 0]['TDMSAnalog500kEncoder0'].ravel()
    edges = np.flatnonzero(np.diff((enc > 0).astype(np.int8)) == 1) + 1
    edges_t = edges / FS500

    f_enc = float(FS500 / np.median(np.diff(edges))) if len(edges) > 2 else np.nan
    f_pr = shaft_from_pressure([sig['Pr2'], sig['Pr3']])
    enc_bias = camp == 'Valve' and 19 <= p_i <= 25
    use_enc = (not enc_bias) and np.isfinite(f_enc) and abs(f_enc / f_pr - 1) < 0.005
    f_true = f_pr

    h = hashlib.sha256()
    for c in ('A1', 'Pr3'):
        h.update(np.ascontiguousarray(sig[c]).tobytes())

    lab = labels(camp, p_i)
    meta = dict(record=f'{camp}_P{p_i}_R{r_i}', campaign=camp, P=p_i, R=r_i, file=path,
                timestamp=ts, role=split_role(camp, p_i), **{f'y_{k}': v for k, v in lab.items()},
                n_faulty=sum(v > 0 for v in lab.values()),
                n50=len(sig['A1']), n25=len(sig['Pr1']), n500=len(enc), n_edges=len(edges),
                f_shaft_encoder=f_enc, f_shaft_pressure=f_pr, f_shaft_used=f_true,
                angle_ref='encoder' if use_enc else 'pressure_speed',
                flag_encoder_bias=int(enc_bias), inverted_clamp=inv,
                f_supply=peak_freq(sig['I3'], FS50, 55, 65),
                i_sum_ratio=float(np.std(sig['I1'] + sig['I2'] + sig['I3']) /
                                  np.mean([np.std(sig[c]) for c in ('I1', 'I2', 'I3')])),
                **{f'sat_{a}': int((np.abs(sig[a]) >= SAT).sum()) for a in ACC},
                pr4_mean=float(sig['Pr4'].mean()), sha256=h.hexdigest())

    t50 = np.arange(len(sig['A1'])) / FS50
    t25 = np.arange(len(sig['Pr1'])) / FS25
    rev50 = angle_of(t50, edges_t, f_true, use_enc)
    rev25 = angle_of(t25, edges_t, f_true, use_enc)

    filt = {}
    for bname, (lo, hi) in ENV_BANDS.items():
        sos = butter(6, [lo, hi], btype='band', fs=FS50, output='sos')
        for a in ACC:
            filt[(a, bname)] = sosfiltfilt(sos, sig[a] - sig[a].mean())

    rows, envs, profs, ncyc = [], [], [], []
    for w in range(NWIN):
        r = dict(record=meta['record'], win=w)
        for c in CHS:
            fs = FS50 if c in dict(CH50) else FS25
            n = int(fs * WIN_S); x = sig[c][w * n:(w + 1) * n]
            for k, v in descriptors(x, fs, f_true, pressure=c.startswith('Pr')).items():
                r[f'{c}_{k}'] = v
        # bearing physics descriptors (envelope order amplitudes, A3 = B1, A4 = B2)
        e_w = np.stack([np.stack([env_order_spectrum(filt[(a, b)], FS50, f_true, w) for b in ENV_BANDS])
                        for a in ACC])                              # (4 acc, 2 bands, n_ord)
        envs.append(e_w)
        pw, nc = [], []
        for c in CHS:
            if c in dict(CH50):
                pr, k = profiles(sig[c], t50, rev50, w, use_rms=True)
            else:
                pr, k = profiles(sig[c], t25, rev25, w, use_rms=False)
            pw.append(pr); nc.append(k)
        profs.append(np.stack(pw)); ncyc.append(min(nc))
        r['n_cycles'] = min(nc)
        rows.append(r)
    return meta, rows, np.stack(envs), np.stack(profs)

def safe(path):
    try:
        return process(path)
    except Exception:
        return ('ERROR', path, traceback.format_exc())

# ---------------------------------------------------------------- main
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('EGB_RAW', 'data/RawData')
    out = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('EGB_FEAT', 'data/features')
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, min(4, (os.cpu_count() or 2) - 1))
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, 'extract_log.txt'), 'w', encoding='utf-8')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

    files = sorted(glob.glob(os.path.join(root, '**', '*.mat'), recursive=True))
    keep = []
    for f in files:
        camp = campaign_of(f); mt = re.match(r'R(\d+)P(\d+)\.mat$', os.path.basename(f))
        if camp is None or mt is None:
            say('skip (unrecognised):', f); continue
        r_i, p_i = int(mt.group(1)), int(mt.group(2))
        if camp == 'Valve' and p_i == 3 and r_i == 8:
            say('skip duplicate (= Valve R7P3):', f); continue
        if camp == 'Multi' and p_i == 1:
            say('skip duplicate baseline (= Bearing P1):', f); continue
        keep.append(f)
    say(f'{len(files)} files found, {len(keep)} to process, {nproc} processes')
    t0 = time.time()

    metas, rows, envs, profs, errs = [], [], [], [], []
    with Pool(nproc) as pool:
        for i, res in enumerate(pool.imap(safe, keep, chunksize=1)):
            if isinstance(res, tuple) and res[0] == 'ERROR':
                errs.append(res[1]); say('ERROR', res[1], res[2]); continue
            m, rw, e, p = res
            metas.append(m); rows.extend(rw); envs.append(e); profs.append(p)
            if (i + 1) % 20 == 0 or i + 1 == len(keep):
                el = time.time() - t0
                say(f'  {i + 1}/{len(keep)} done, {el / 60:.1f} min elapsed, '
                    f'~{el / (i + 1) * (len(keep) - i - 1) / 60:.1f} min left')

    meta = pd.DataFrame(metas)
    meta.to_csv(os.path.join(out, 'recordings.csv'), index=False)
    feats = pd.DataFrame(rows)
    try:
        feats.to_parquet(os.path.join(out, 'window_features.parquet'), index=False)
    except Exception as ex:
        say('parquet unavailable (', ex, ') -> writing csv.gz')
        feats.to_csv(os.path.join(out, 'window_features.csv.gz'), index=False)
    np.savez_compressed(os.path.join(out, 'env_orders.npz'), env=np.stack(envs), orders=ORD,
                        records=meta.record.values, acc=np.array(ACC),
                        bands=np.array([f'{k}:{v[0]:.0f}-{v[1]:.0f}Hz' for k, v in ENV_BANDS.items()]))
    np.savez_compressed(os.path.join(out, 'angle_profiles.npz'), prof=np.stack(profs).astype(np.float32),
                        records=meta.record.values, channels=np.array(CHS),
                        angle_ref=meta.angle_ref.values)
    fp = meta[['record', 'sha256']]
    fp.to_csv(os.path.join(out, 'fingerprints.csv'), index=False)
    dup = fp[fp.sha256.duplicated(keep=False)]

    say('\n==== SUMMARY ====')
    say('records:', len(meta), ' windows:', len(feats), ' errors:', len(errs))
    say(meta.groupby(['campaign', 'role']).size().to_string())
    say('shaft speed (pressure) Hz: mean %.4f sd %.4f' % (meta.f_shaft_pressure.mean(), meta.f_shaft_pressure.std()))
    say('encoder/pressure speed ratio by flag:\n' +
        (meta.f_shaft_encoder / meta.f_shaft_pressure).groupby(meta.flag_encoder_bias).describe().to_string())
    say('supply Hz: %.3f +- %.3f' % (meta.f_supply.mean(), meta.f_supply.std()))
    say('3-phase sum ratio (after clamp fix): median %.3f max %.3f' % (meta.i_sum_ratio.median(), meta.i_sum_ratio.max()))
    say('records with saturation: ' + json.dumps({a: int((meta[f'sat_{a}'] > 0).sum()) for a in ACC}))
    say('angle reference: ' + json.dumps(meta.angle_ref.value_counts().to_dict()))
    say('remaining duplicate fingerprints: %d' % len(dup))
    if len(dup): say(dup.to_string())
    say('done in %.1f min' % ((time.time() - t0) / 60))
    log.close()

if __name__ == '__main__':
    main()
