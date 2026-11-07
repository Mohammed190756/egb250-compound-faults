import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys; sys.path.insert(0, (SRC)); sys.path.insert(0, (SRC))
from col import *
import numpy as np, pandas as pd, scipy.io as sio
from data6 import *
which = sys.argv[1:] or ['split', 'overview', 'signal', 'merge', 'm1m2']
if 'split' in which:
    C = meta.groupby(['campaign', 'P']).first().reset_index(); C['lab'] = C.apply(lambda r: name_of([r['y_' + c] for c in COMPS]), axis=1)
    fig = plt.figure(figsize=(7.0, 4.2)); ax = fig.add_axes([0, 0, 1, 1]); canvas(ax, 100, 72)
    TR, VA, TE = W, '#bdbdbd', '#636363'   # white / light grey / dark grey (IJPHM: no coloured backgrounds)
    def cell(x, y, w, h, r, test, excl=False):
        if excl: ax.add_patch(Rectangle((x, y), w, h, fc=W, ec=GR, lw=0.6, ls='--')); ax.text(x + w / 2, y + h / 2, 'healthy\n(not used)', ha='center', va='center', fontsize=4.9, color=GR); return
        if test: ax.add_patch(Rectangle((x, y), w, h, fc=TE, ec=K, lw=0.5)); ax.text(x + w / 2, y + h / 2, r['lab'].replace('+', '\n+'), ha='center', va='center', fontsize=4.9, color='white', fontweight='bold')
        else:
            ax.add_patch(Rectangle((x, y), w * 0.8, h, fc=TR, ec=K, lw=0.5)); ax.add_patch(Rectangle((x + w * 0.8, y), w * 0.2, h, fc=VA, ec=K, lw=0.5)); ax.text(x + w * 0.4, y + h / 2, r['lab'], ha='center', va='center', fontsize=4.9)
    rows = [('Valve', list(range(1, 18)), 'Valve campaign: 16 single faults (its healthy session is not used)'), ('Bearing', [1, 2, 3, 4], 'Bearing campaign: healthy and 3 single faults'),
            ('Valve', list(range(18, 26)), 'Test 2: 8 real valve + valve combinations'), ('Multi', list(range(2, 14)), 'Test 1: 12 real valve + bearing combinations')]
    y = 64
    for camp, ps, title in rows:
        ax.text(1, y + 2.5, title, fontsize=6.8, fontweight='bold', va='bottom'); per = 9 if len(ps) > 9 else len(ps); w = 10.6; h = 5.0
        for k, p in enumerate(ps):
            r = C[(C.campaign == camp) & (C.P == p)].iloc[0]; xx = 1 + (k % per) * 11; yy = y - (k // per) * 5.8 - h; cell(xx, yy, w, h, r, r['role'] == 'combo', excl=(camp == 'Valve' and p == 1))
        y = yy - 6.3
        if camp == 'Bearing': box(ax, 47, yy - 0.5, 52, 6.5, '', ['144 combined pairs = 48 V+B (16 × 3) + 96 V+V (6 × 4 × 4),', 'built from the training cells and labelled like the tests'], f='aug', ls=6.0)
    ly = yy - 5; ax.set_ylim(ly - 6, 72)
    for x, fc, h_, t in ((1, TR, '', 'training repetitions R1–R12'), (31, VA, '', 'validation repetitions R13–R15'), (63, TE, '', 'real test combinations, never used')):
        ax.add_patch(Rectangle((x, ly), 5, 3, fc=fc, ec=K, lw=0.5, hatch=h_)); ax.text(x + 6, ly + 1.5, t, va='center', fontsize=6.6)
    ax.text(1, ly - 3.5, 'Training: 239 real + 1,713 combined rows; validation: 60 real + 432 combined rows; final training: 299 + 2,145; test: 180 + 120 real recordings.', fontsize=6.5, va='center')
    print('split', save(fig, 'Fig_split'))
if 'overview' in which:
    fig = plt.figure(figsize=(7.2, 4.4)); ax = fig.add_axes([0, 0, 1, 1]); canvas(ax, 100, 62)
    for x, t in ((9, 'Data'), (29, 'Signal processing'), (52, 'Combined training rows'), (84, 'Learning and decoding')): ax.text(x, 60.5, t, ha='center', fontsize=8, fontweight='bold')
    box(ax, 0.5, 36, 17, 20, 'Real recordings', ['healthy (Bearing P1)', '19 single faults', '20 real combinations', '(test only)'], f='data')
    box(ax, 0.5, 10, 17, 20, 'Split', ['train R1–R12', 'validation R13–R15', 'test: real', 'combinations'], f='data')
    box(ax, 21, 38, 16, 18, 'Crank-angle', ['72 bins of 5°', 'aligned to healthy'], f='w'); box(ax, 21, 12, 16, 18, 'Envelope', ['A3, A4: 2–8 kHz', '300 orders'], f='w')
    box(ax, 41, 30, 22, 26, 'Physical superposition', ['pressure: A + B − H', 'energy: A² + B² − H²', '144 pairs × repetitions'], f='aug')
    box(ax, 41, 4, 22, 20, 'Label column', ['164 names: healthy,', '19 single, 144 pairs', 'e.g. 2S_DV_3+B1_1'], f='aug')
    box(ax, 67, 30, 32, 26, 'Models M1–M5', ['M1 random forest · M2 MLP', 'M3 1-D CNN · M4 GRU + attention', 'M5 GRU + physics-masked attention', 'balanced loss, tuned on validation'], f='learn')
    box(ax, 67, 4, 32, 20, 'Decoding', ['five component posteriors →', 'most probable of 164 names', 'test: exact label, location'], f='learn')
    arr(ax, 17.5, 46, 21, 47); arr(ax, 17.5, 42, 21, 21); arr(ax, 37, 47, 41, 45); arr(ax, 37, 21, 41, 40); arr(ax, 52, 30, 52, 24); arr(ax, 63, 43, 67, 43); arr(ax, 63, 14, 67, 14); arr(ax, 83, 30, 83, 24)
    print('overview', save(fig, 'Fig_overview'))
if 'signal' in which or 'merge' in which: B = Builder(POOL)
if 'signal' in which:
    d = sio.loadmat((RAW) + '/Raw_Data_Valve_Faults_DB_011V0/P5/R1P5.mat')['data'][0, 0]
    g25 = d['Analog25k'][0, 0]; pr = {f'Pr{i + 1}': g25[f'TDMSAnalog25kPressure{i}'].ravel() for i in range(4)}
    enc = d['Analog500k'][0, 0]['TDMSAnalog500kEncoder0'].ravel(); ed = np.flatnonzero(np.diff((enc > 0).astype(np.int8)) == 1) / 500000.0
    fig = plt.figure(figsize=(7.0, 6.2)); t = np.arange(len(pr['Pr1'])) / 25000.0; PC = {'Pr1': '#E69F00', 'Pr2': '#0072B2', 'Pr3': '#D55E00', 'Pr4': '#555555'}
    ax = fig.add_axes([0.08, 0.63, 0.38, 0.29]); m = (t >= 1.0) & (t < 1.25)
    for k in PC: ax.plot(t[m] - 1, pr[k][m], color=PC[k], lw=0.9, label=k)
    for e in ed[(ed >= 1) & (ed < 1.25)]: ax.axvline(e - 1, color=GR, lw=0.5, ls='--')
    ax.set_xlabel('Time from 1.00 s of the recording (s)'); ax.set_ylabel('Pressure signal (V)'); ax.legend(ncol=4, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=6.6); panel(ax, '(a) Raw pressures, V-P5 R1 (2S_DV spring breakage)', y=1.14)
    ax = fig.add_axes([0.58, 0.63, 0.38, 0.29]); m = (t >= 1) & (t < 2); r = np.interp(t[m], ed, np.arange(len(ed))); ph = (r % 1) * 360
    for k in ('Pr2', 'Pr3'):
        ax.scatter(ph, pr[k][m], s=0.04, color=tint(PC[k], 0.35), rasterized=True); b = (ph // 5).astype(int); mu = np.bincount(b, pr[k][m], 72) / np.maximum(np.bincount(b, None, 72), 1); ax.plot(np.arange(72) * 5 + 2.5, mu, color=PC[k], lw=1.3, label=f'{k}: bin mean')
    ax.set_xlim(0, 360); ax.set_xlabel('Crank angle (°)'); ax.set_ylabel('Pressure signal (V)'); ax.legend(frameon=False, fontsize=6.6, loc='upper right'); panel(ax, '(b) One second folded onto crank angle', y=1.03)
    i = int(np.flatnonzero(meta.record == 'Valve_P5_R1')[0]); X, E = B.real(np.array([i]))
    ax = fig.add_axes([0.08, 0.07, 0.36, 0.36]); im = ax.imshow(X[0].mean(0), aspect='auto', cmap=DIV, vmin=-6, vmax=6, extent=[0, 360, 12.5, -0.5])
    ax.set_yticks(range(13)); ax.set_yticklabels(B.ch, fontsize=6.4); ax.set_xlabel('Crank angle (°)'); cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label('Deviation from healthy (SD)', fontsize=6.6); cb.ax.tick_params(labelsize=6.2)
    panel(ax, '(c) Model input, 13 × 72', y=1.03)
    ax = fig.add_axes([0.60, 0.07, 0.37, 0.33]); k = ORD <= 12
    for lab, c in (('healthy', GR), ('B1_1', '#0072B2'), ('B1_3', ACC)):
        idx = [j for j in np.flatnonzero(POOL) if name_of(Y[j]) == lab]; ax.plot(ORD[k], B.E[idx][:, :, 0].mean((0, 1))[k] * 1e3, color=c, lw=0.9, label={'healthy': 'healthy', 'B1_1': 'inner-race crack', 'B1_3': 'outer-race crack'}[lab])
    for o, l in ((7.15, 'BPFO'), (9.85, 'BPFI')): ax.axvline(o, color=K, lw=0.5, ls=':'); ax.text(o + 0.12, 55, l, fontsize=6.2)
    ax.set_yscale('log'); ax.set_xlim(0, 12); ax.set_xlabel('Shaft order'); ax.set_ylabel('A3 envelope (mm/s²)'); ax.legend(frameon=False, fontsize=6.4, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3); panel(ax, '(d) Mean envelope spectra', y=1.14)
    print('signal', save(fig, 'Fig_signal'))
if 'merge' in which:
    a = REC[('2S_DV_3', 5)]; b = REC[('B1_3', 5)]; P, Ec = B.merge(a, b)
    fig = plt.figure(figsize=(7.0, 5.0)); ang = np.arange(72) * 5 + 2.5
    L = [('single 2S_DV_3', '#0072B2'), ('single B1_3', '#E69F00'), ('combined row', ACC), ('healthy mean', GR)]
    for k_, (chn, t_) in enumerate((('Pr3', '(a) Pr3 pressure'), ('Pr2', '(b) Pr2 pressure'), ('A3', '(c) A3 RMS profile'))):
        ax = fig.add_axes([0.08 + (k_ % 2) * 0.49, 0.58 - (k_ // 2) * 0.48, 0.40, 0.30]); j = B.ch.index(chn)
        for v, (n, c) in zip((B.P[a][:, j].mean(0), B.P[b][:, j].mean(0), P[:, j].mean(0), B.Hp[j] if chn.startswith('Pr') else np.sqrt(B.H2[j])), L):
            ax.plot(ang, v, color=c, lw=1.4 if n.startswith('comb') else 0.9, ls='--' if n.startswith('healthy') else '-', label=n)
        ax.set_xlim(0, 360); ax.set_xlabel('Crank angle (°)'); ax.set_ylabel('V' if chn.startswith('Pr') else 'm/s²'); panel(ax, t_, y=1.03)
        if k_ == 0: fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    ax = fig.add_axes([0.57, 0.10, 0.41, 0.30]); k = ORD <= 12
    for v, (n, c) in zip((B.E[a][:, 0].mean(0), B.E[b][:, 0].mean(0), Ec[:, 0].mean(0), np.sqrt(B.HE2[0])), L): ax.plot(ORD[k], v[k] * 1e3, color=c, lw=1.3 if n.startswith('comb') else 0.8, ls='--' if n.startswith('healthy') else '-')
    ax.set_yscale('log'); ax.set_xlim(0, 12); ax.set_xlabel('Shaft order'); ax.set_ylabel('mm/s²'); panel(ax, '(d) A3 envelope spectrum', y=1.03)
    fig.text(0.5, -0.01, 'Repetition R5 of each single fault, mean of nine windows. H: healthy mean of Bearing P1.', ha='center', fontsize=6.6)
    print('merge', save(fig, 'Fig_merge'))
if 'm1m2' in which:
    fig = plt.figure(figsize=(7.0, 4.6))
    ax = fig.add_axes([0, 0.52, 1, 0.46]); canvas(ax, 100, 50); panel(ax, '(a) M1: random forest, one forest per component', y=0.98)
    box(ax, 1, 14, 15, 24, 'Input', ['162 summary', 'features per', '1-s window'], f='data')
    box(ax, 21, 30, 15, 10, '', ['bootstrap sample 1']); box(ax, 21, 17, 15, 10, '', ['bootstrap sample 2']); box(ax, 21, 2, 15, 10, '', ['bootstrap sample T'])
    ax.text(28.5, 14.2, '·\n·\n·', ha='center', va='center', fontsize=8, linespacing=0.6)
    def tree(x0, y0, s=1.0):
        pts = [(x0, y0), (x0 - 4 * s, y0 - 4), (x0 + 4 * s, y0 - 4), (x0 - 6 * s, y0 - 8), (x0 - 2 * s, y0 - 8), (x0 + 2 * s, y0 - 8), (x0 + 6 * s, y0 - 8)]
        for a_, b_ in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)): ax.plot([pts[a_][0], pts[b_][0]], [pts[a_][1], pts[b_][1]], color=K, lw=0.6)
        for i, p in enumerate(pts): ax.add_patch(Circle(p, 0.9, fc='#bdbdbd' if i > 2 else W, ec=K, lw=0.6))
    for y_ in (41, 28, 13): tree(48, y_ - 3.5)
    ax.text(48, 12.5, '·\n·\n·', ha='center', va='center', fontsize=8, linespacing=0.6)
    for y_ in (35, 22, 7): arr(ax, 36, y_, 41, y_); arr(ax, 55, y_, 58, 26)
    box(ax, 58, 14, 17, 24, 'Vote', ['average the class', 'probabilities of', 'T trees', '(weighted samples)'], f=tint(MC['RF'], 0.25))
    box(ax, 80, 14, 19, 24, 'Output', ['p_c(m) for each', 'of 5 components', 'window → record', 'mean of 9'], f=tint(MC['RF'], 0.5)); arr(ax, 75, 26, 80, 26)
    ax.text(48, 44.8, 'each split: best of a random feature subset (Gini)', ha='center', fontsize=6.4, style='italic')
    ax = fig.add_axes([0, 0.0, 1, 0.48]); canvas(ax, 100, 50); panel(ax, '(b) M2: multilayer perceptron with one head per component', y=0.98)
    def layer(x, n, label, sub, h=36, y0=6):
        box(ax, x, y0, 9, h, '', [], f=tint(MC['MLP'], 0.2))
        for k in range(min(n, 7)): ax.add_patch(Circle((x + 4.5, y0 + h - 4 - k * (h - 8) / 6), 1.1, fc=W, ec=K, lw=0.6))
        ax.text(x + 4.5, y0 - 1.5, label, ha='center', va='top', fontsize=6.8, fontweight='bold'); ax.text(x + 4.5, y0 - 4.8, sub, ha='center', va='top', fontsize=6.3)
    layer(2, 7, 'input', '162'); layer(22, 7, 'dense + GELU', '128ζ, dropout'); layer(42, 7, 'dense + GELU', '64ζ')
    for x in (11, 31): arr(ax, x, 24, x + 11, 24)
    for k, c in enumerate(COMPS):
        y_ = 40 - k * 8; box(ax, 62, y_, 16, 6, '', [f'head {c}'], ls=6.3); arr(ax, 51, 24, 62, y_ + 3, lw=0.6)
        box(ax, 84, y_, 15, 6, '', [f'softmax, {5 if c != "B1" else 4} modes'], f=tint(MC['MLP'], 0.5), ls=6.0); arr(ax, 78, y_ + 3, 84, y_ + 3, lw=0.6)
    ax.text(80, -1.5, 'head = dropout + dense(64ζ→32ζ) + GELU\n+ dense(32ζ→M_c)', ha='center', va='top', fontsize=6.4, style='italic'); ax.set_ylim(-8, 50)
    print('m1m2', save(fig, 'Fig_M1M2'))
