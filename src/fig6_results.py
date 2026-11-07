import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json; sys.path.insert(0, (SRC))
from col import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
R = (RES) + '/'; which = sys.argv[1:] or ['train', 'main', 'cond', 'sim']
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']; SH = {'RF': 'M1', 'MLP': 'M2', 'CNN': 'M3', 'Dense': 'M4', 'GNN': 'M5'}
FAMS = (('valve-bearing', 'valve + bearing', 180), ('valve-valve', 'valve + valve', 120))
if 'train' in which:
    TU = {m: json.load(open(R + f'tune_{m}.json')) for m in MODELS}
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_axes([0.08, 0.60, 0.38, 0.32])
    for x, m in enumerate(MODELS):
        J = TU[m]; sel = J['selected']['k']; ks = sorted({int(k.split(':')[0]) for k in J if k != 'selected'})
        for k in ks:
            v = np.mean([J[f'{k}:{s}'].get('best_val_loss', J[f'{k}:{s}'].get('val_loss')) for s in range(3) if f'{k}:{s}' in J])
            ax.scatter(x + np.random.default_rng(k).uniform(-0.22, 0.22), v, s=40 if k == sel else 12, marker='*' if k == sel else 'o', color=MC[m] if k == sel else tint(MC[m], 0.55), edgecolor=K if k == sel else 'none', lw=0.6, zorder=3)
    ax.set_xticks(range(5)); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_yscale('log'); ax.set_ylabel('Validation loss'); panel(ax, '(a) Random search; star: selected', y=1.03)
    keys = [('val_loss', '(b) Validation loss', True), ('val_exact_pair', '(c) Validation: combined rows, exact label', False), ('val_exact_real', '(d) Validation: real rows, exact label', False)]
    pos = [[0.58, 0.60, 0.38, 0.32], [0.08, 0.10, 0.38, 0.32], [0.58, 0.10, 0.38, 0.32]]
    for (k, t, logy), p in zip(keys, pos):
        ax = fig.add_axes(p)
        for m in MODELS[1:]:
            J = TU[m]; sel = J['selected']['k']; H = [J[f'{sel}:{s}']['hist'] for s in range(3)]; L = max(len(h) for h in H)
            mu = np.array([np.mean([h[e][k] for h in H if len(h) > e]) for e in range(L)]); ax.plot(np.arange(1, L + 1), mu, color=MC[m], lw=1.1, ls={'MLP': '-', 'CNN': '--', 'Dense': '-.', 'GNN': ':'}[m], label=LBL[m])
            if k == 'val_loss': e = J['selected']['epochs']; ax.plot(e, mu[e - 1], marker='*', ms=8, color=MC[m], mec=K, mew=0.5)
        if logy: ax.set_yscale('log')
        else: ax.set_ylim(0, 1.02)
        ax.set_xlabel('Epoch'); panel(ax, t, y=1.03)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=4, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.01, 'Curves: selected setting, mean over the seeds still training (early stopping). Stars in (b): training length e* of the final models.', ha='center', fontsize=6.6)
    print('train', save(fig, 'Fig_training'))
if 'main' in which:
    C = pd.read_csv(R + 'test_ci.csv'); const = json.load(open(R + 'const_baseline.json'))
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), gridspec_kw=dict(hspace=0.5, wspace=0.15))
    for c_, (f, ft, n) in enumerate(FAMS):
        for r_, (met, ml) in enumerate((('label', 'Exact-label accuracy'), ('loc', 'Location accuracy'))):
            ax = axs[r_, c_]; x = np.arange(5); w = 0.38
            for j, (tr, al, h) in enumerate((('single faults only', 0.35, '////'), ('with combined rows', 1.0, ''))):
                for i, m in enumerate(MODELS):
                    g = C[(C.model == m) & (C.training == tr) & (C.family == f) & (C.metric == met)]
                    if len(g) == 0: continue
                    g = g.iloc[0]; ax.bar(x[i] + (j - 0.5) * w, g['mean'], w, yerr=[[g['mean'] - g.lo], [g.hi - g['mean']]], color=tint(MC[m], al), ec=K, lw=0.5, hatch=h, error_kw=dict(lw=0.6, capsize=1.2, ecolor=K))
            if met == 'loc':
                cb = [q for q in const if q['family'] == f][0]['loc']; ax.axhline(cb, color=GR, ls='--', lw=0.8); ax.text(4.6, cb + 0.02, 'constant rule', fontsize=6.0, color=GR, ha='right')
            ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_ylim(0, 1.05); ax.set_ylabel(ml if c_ == 0 else '')
            panel(ax, f'({"abcd"[r_ * 2 + c_]}) {ft}: {ml.split()[0].lower()} {ml.split()[1] if ml.startswith("Exact") else ""}'.strip(), y=1.03, fs=8)
    hs = [Patch(fc=tint('#555555', 0.35), ec=K, lw=0.5, hatch='////', label='trained on single faults only (baseline)'), Patch(fc='#555555', ec=K, lw=0.5, label='trained with combined two-fault rows')]
    fig.legend(handles=hs, frameon=False, ncol=2, fontsize=6.8, loc='upper center', bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, 0.0, 'Mean of three seeds; whiskers: 95% cluster-bootstrap interval over conditions. Dashed: constant rule that knows the test set.', ha='center', fontsize=6.6)
    print('main', save(fig, 'Fig_main'))
if 'cond' in which:
    T = pd.read_csv(R + 'test_conditions.csv'); T['num'] = T.cond.str.extract(r'P(\d+)').astype(int)[0]; T['camp'] = T.cond.str[0]
    order = T[T.model == 'RF'].sort_values(['camp', 'num']).cond.tolist(); labs = T[T.model == 'RF'].set_index('cond').loc[order].label.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 5.4), gridspec_kw=dict(wspace=0.06))
    for ax, (met, t) in zip(axs, (('exact', '(a) Exact label'), ('loc', '(b) Location'))):
        M = np.array([[T[(T.model == m) & (T.cond == c)][met].iloc[0] for m in MODELS] for c in order]); im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(5)); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{c.replace('Multi_P', 'MF-P').replace('Valve_P', 'V-P')}  {l}" for c, l in zip(order, labs)] if ax is axs[0] else [], fontsize=6.3)
        for i in range(M.shape[0]):
            for j in range(5): ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=5.8, color='white' if M[i, j] > 0.55 else K)
        ax.axhline(11.5, color=ACC, lw=1.2); ax.set_title(t, fontsize=8, loc='left', fontweight='bold')
    cb = fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02); cb.set_label('Share of recordings (three seeds pooled)', fontsize=7); cb.ax.tick_params(labelsize=6.4)
    print('cond', save(fig, 'Fig_conditions'))
if 'sim' in which:
    S = pd.DataFrame(json.load(open(R + 'superposition6.json'))); T = pd.read_csv(R + 'test_conditions.csv'); acc = T.groupby('label')[['exact', 'loc']].mean()
    S = S.join(acc, on='label'); code = T.drop_duplicates('label').set_index('label').cond.str.replace('Multi_P', 'MF-P').str.replace('Valve_P', 'V-P'); S['code'] = S.label.map(code)
    S['num'] = S.code.str.extract(r'P(\d+)').astype(int)[0]; S = S.sort_values(['family', 'num']).reset_index(drop=True); fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(wspace=0.28))
    ax = axs[0]; x = np.arange(len(S)); w = 0.27
    ax.bar(x - w, S.singleA, w, color='#0072B2', label='valve fault alone'); ax.bar(x, S.singleB, w, color='#E69F00', label='second fault alone'); ax.bar(x + w, S.combined, w, color=ACC, label='combined row')
    ax.set_xticks(x); ax.set_xticklabels(S.code, fontsize=5.6, rotation=90); ax.set_ylabel('Correlation with real input'); ax.set_ylim(-0.1, 1.05); ax.legend(frameon=False, fontsize=6.2, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.07))
    panel(ax, '(a) Similarity to the real combination', y=1.16)
    ax = axs[1]
    for f, mk in (('valve-bearing', 'o'), ('valve-valve', 's')):
        g = S[S.family == f]; ax.scatter(g.combined, g.exact, marker=mk, s=28, color=ACC if f == 'valve-bearing' else '#0072B2', edgecolor=K, lw=0.4, label=f.replace('-', ' + '))
    rr = np.corrcoef(S.combined, S.exact)[0, 1]; ax.text(0.03, 0.95, f'r = {rr:.2f}', transform=ax.transAxes, fontsize=7, va='top')
    ax.set_xlabel('Similarity of combined row to real input'); ax.set_ylabel('Exact-label accuracy (mean of models)'); ax.set_xlim(0.4, 1); ax.set_ylim(-0.03, 1.03); ax.legend(frameon=False, fontsize=6.4, loc='upper left', bbox_to_anchor=(0.0, 0.92))
    panel(ax, '(b) Similarity and accuracy per combination', y=1.16)
    fig.text(0.5, -0.12, 'Diagnostic only: computed after training, no choice was based on it.', ha='center', fontsize=6.6)
    print('sim', save(fig, 'Fig_similarity'))
