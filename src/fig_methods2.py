import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys; sys.path.insert(0, (SRC))
from col import *
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
which = sys.argv[1:] or ['overview', 'combine', 'm3', 'm45']
if 'overview' in which:
    fig, ax = plt.subplots(figsize=(7.0, 4.9)); canvas(ax, 100, 70)
    for x, t in ((10, 'Data'), (31.5, 'Inputs'), (55, 'Combination'), (85, 'Diagnosis')): ax.text(x, 68, t, ha='center', fontsize=8.0, fontweight='bold')
    box(ax, 0.5, 40, 19, 24, 'Real data', ['healthy session', '19 single faults', '15 repetitions'], f='data', ts=7.0, ls=6.5)
    box(ax, 0.5, 4, 19, 28, 'Compound', ['20 combinations', '300 recordings', 'test only'], f='test', ts=7.0, ls=6.5)
    box(ax, 23, 47, 17, 17, 'Profiles', ['13 × 72 bins'], ts=7.0, ls=6.5)
    box(ax, 23, 26, 17, 17, 'Spectra', ['A3, A4 envelope'], ts=7.0, ls=6.5)
    box(ax, 23, 4, 17, 18, 'Symptoms', ['8 quantities'], ts=7.0, ls=6.5)
    box(ax, 43, 39, 25, 25, 'C2: systematic', ['144 labelled pairs', 'rules M-a to M-e', 'matched or random'], f='aug', ts=7.0, ls=6.5)
    box(ax, 43, 19, 25, 15, 'C1: random', ['pairs every epoch'], f='aug', ts=7.0, ls=6.5)
    box(ax, 43, 4, 25, 11, '', ['signatures add'], f='aug', ts=7.0, ls=6.5)
    box(ax, 71, 43, 28, 21, 'Models M1–M5', ['forest, perceptron,', 'CNN, GRU-attention'], f='learn', ts=7.0, ls=6.5)
    box(ax, 71, 25, 28, 13, 'Decoding', ['1 of 164 names'], f='learn', ts=7.0, ls=6.5)
    box(ax, 71, 4, 28, 16, 'Reasoning', ['abductive or', 'Bayesian'], f='phys', ts=7.0, ls=6.5)
    for p in (((19.5, 54), (23, 55)), ((19.5, 50), (23, 35)), ((19.5, 46), (23, 13)), ((40, 55), (43, 54)), ((40, 34), (43, 48)), ((40, 34), (43, 27)),
              ((40, 12), (43, 9)), ((68, 52), (71, 53)), ((68, 27), (71, 49)), ((85, 43), (85, 38)), ((85, 25), (85, 20)), ((68, 9), (71, 11))): arr(ax, *p[0], *p[1])
    print('overview', save(fig, 'Fig_overview2'))
if 'combine' in which:
    sys.path.insert(0, (SRC)); from data6 import CONDS, COMP_OF, PAIRS
    fig = plt.figure(figsize=(7.0, 4.4))
    ax = fig.add_axes([0.0, 0.05, 0.42, 0.88]); canvas(ax, 50, 60)
    box(ax, 1, 44, 22, 12, 'Single-fault windows', ['all training repetitions'], f='data', ts=7.2, ls=6.6)
    box(ax, 27, 44, 22, 12, 'Random pair a, b', ['different components'], ts=7.2, ls=6.6)
    box(ax, 1, 22, 48, 14, 'Normalised-domain sum', ['x = x_a + x_b,  e = e_a + e_b', 'mode per component = max(y_a, y_b)'], f='aug', ts=7.2, ls=6.6)
    box(ax, 1, 3, 48, 12, 'New pairs drawn every epoch', ['as many pairs as real windows'], ts=7.2, ls=6.6)
    arr(ax, 23, 50, 27, 50); arr(ax, 38, 44, 25, 36); arr(ax, 25, 22, 25, 15)
    panel(ax, '(a) C1: random compositional augmentation', y=0.98, x=0.02)
    ax = fig.add_axes([0.50, 0.10, 0.48, 0.80]); n = len(CONDS); M = np.full((n, n), np.nan)
    TEST = {('2S_DV', 'B1'), ('2S_IV', '2S_DV'), ('1S_IVb', '2S_DV')}
    for i, a in enumerate(CONDS):
        for j, b in enumerate(CONDS):
            ca, cb = COMP_OF[a], COMP_OF[b]
            if ca == cb: M[i, j] = 0
            else: M[i, j] = 2 if 'B1' in (ca, cb) else 1
    from matplotlib.colors import ListedColormap
    ax.imshow(M, cmap=ListedColormap(['#f7f7f7', '#bdbdbd', '#737373']), vmin=0, vmax=2)
    def is_test(a, b):
        ca, cb = COMP_OF[a], COMP_OF[b]; ma, mb = a.rsplit('_', 1)[1], b.rsplit('_', 1)[1]
        if {ca, cb} == {'2S_DV', 'B1'}: return True
        return {ca, cb} in ({'2S_IV', '2S_DV'}, {'1S_IVb', '2S_DV'}) and ma == mb
    for i, a in enumerate(CONDS):
        for j, b in enumerate(CONDS):
            if i != j and COMP_OF[a] != COMP_OF[b] and is_test(a, b): ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=K, lw=1.4))
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(CONDS, rotation=90, fontsize=5.6); ax.set_yticklabels(CONDS, fontsize=5.6)
    ax.tick_params(length=0)
    hs = [Patch(fc='#bdbdbd', ec=K, lw=0.4, label='valve + valve (96)'), Patch(fc='#737373', ec=K, lw=0.4, label='valve + bearing (48)'), Patch(fc='#f7f7f7', ec=K, lw=0.4, label='same component'), Patch(fc='white', ec=K, lw=1.4, label='structure of a real test condition')]
    ax.legend(handles=hs, frameon=False, fontsize=6.4, loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=2)
    panel(ax, '(b) C2: the 144 admissible pairs', y=1.02)
    print('combine', save(fig, 'Fig_combine'))
if 'm3' in which:
    fig, ax = plt.subplots(figsize=(7.0, 2.9)); canvas(ax, 100, 40)
    top = [('Profiles', '13 × 72', 'data'), ('Conv 5', '64 × 72', 'aug'), ('Pool 2', '64 × 36', 'w'), ('Conv 5', '128 × 36', 'aug'), ('Avg. pool', '128', 'w')]
    bot = [('Spectra', '2 × 300', 'data'), ('Conv 9', '32 × 300', 'aug'), ('Pool 4', '32 × 75', 'w'), ('Conv 9', '64 × 75', 'aug'), ('Max pool', '64 × 8', 'w'), ('Linear', '512 → 128', 'aug')]
    for row, y in ((top, 24), (bot, 3)):
        for i, (t, s_, f) in enumerate(row):
            box(ax, 1 + i * 13.2, y, 11.6, 12, t, [s_], f=f, ts=6.8, ls=6.2)
            if i: arr(ax, 1 + i * 13.2 - 1.6, y + 6, 1 + i * 13.2, y + 6)
    box(ax, 82, 12, 17, 16, 'Concat.', ['256, dropout,', 'five heads'], f=tint('#009E73', 0.45), ts=6.8, ls=6.2)
    arr(ax, 1 + 4 * 13.2 + 11.6, 30, 82, 24); arr(ax, 1 + 5 * 13.2 + 11.6, 9, 82, 16)
    ax.text(50, 38.5, 'Conv k: kernel width k with GELU. Sizes for the selected width ζ = 2 (213,880 parameters).', ha='center', fontsize=6.4, style='italic')
    print('m3', save(fig, 'Fig_M3b'))
if 'm45' in which:
    sys.path.insert(0, (SRC)); from data4 import SENS, LINKS, COMPS
    fig = plt.figure(figsize=(7.0, 5.0)); ax = fig.add_axes([0.0, 0.55, 1.0, 0.43]); canvas(ax, 100, 40)
    box(ax, 1, 22, 15, 15, 'Profile j', ['36 steps of 10°'], f='data', ts=6.8, ls=6.2); box(ax, 21, 22, 20, 15, 'GRU', ['hidden size d = 32ζ', 'shared by channels'], f='learn', ts=6.8, ls=6.2)
    box(ax, 46, 22, 14, 15, 'Mean pool', ['+ linear'], f='learn', ts=6.8, ls=6.2)
    box(ax, 1, 2, 15, 15, 'Spectrum', ['A3 or A4, 300 orders'], f='data', ts=6.8, ls=6.2); box(ax, 21, 2, 20, 15, '1-D CNN', ['two conv. layers', '+ linear → d'], f='learn', ts=6.8, ls=6.2)
    box(ax, 66, 2, 15, 35, 'Sensors', ['15 vectors h_s', '13 profiles', '2 spectra'], f=tint('#56B4E9', 0.6), ts=6.8, ls=6.2)
    box(ax, 85, 2, 14, 35, 'Parts', ['attention', '5 queries q_c', 'messages', '5 heads'], f=tint('#0072B2', 0.35), ts=6.8, ls=6.2)
    for a_, b_ in (((16, 29.5), (21, 29.5)), ((41, 29.5), (46, 29.5)), ((60, 29.5), (66, 25)), ((16, 9.5), (21, 9.5)), ((41, 9.5), (66, 12)), ((81, 19.5), (85, 19.5))): arr(ax, *a_, *b_)
    panel(ax, '(a) Encoders and component stage shared by M4 and M5', y=1.0, x=0.0)
    for k, (t, full) in enumerate((('(b) M4: dense attention', True), ('(c) M5: physics mask', False))):
        ax = fig.add_axes([0.08 + k * 0.47, 0.08, 0.40, 0.36]); Mk = np.ones((5, 15)) if full else np.array([[1 if s in LINKS[c] else 0 for s in SENS] for c in COMPS])
        ax.imshow(Mk, cmap=ListedColormap(['#f7f7f7', '#525252']) if True else None, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(15)); ax.set_xticklabels([s.replace('env', ' env') for s in SENS], rotation=90, fontsize=6.0); ax.set_yticks(range(5)); ax.set_yticklabels(COMPS, fontsize=6.4)
        for x_ in np.arange(-0.5, 15, 1): ax.axvline(x_, color='white', lw=0.6)
        for y_ in np.arange(-0.5, 5, 1): ax.axhline(y_, color='white', lw=0.6)
        ax.tick_params(length=0); ax.set_title(t, loc='left', fontsize=8, fontweight='bold')
    print('m45', save(fig, 'Fig_M45b'))
