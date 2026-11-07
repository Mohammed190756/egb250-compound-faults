import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json; sys.path.insert(0, (SRC))
from col import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, ListedColormap
import numpy as np, pandas as pd
from scipy.stats import spearmanr
R = (RES) + '/'; F = R + 'final/'
MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']; SH = {'RF': 'M1', 'MLP': 'M2', 'CNN': 'M3', 'Dense': 'M4', 'GNN': 'M5'}
which = sys.argv[1:] or ['add', 'cmp', 'conf', 'reason', 'robust', 'attsl']
def order_conds(df, col='cond'):
    n = df[col].str.extract(r'P(\d+)')[0].astype(int); fam = df[col].str.startswith('V'); return df.assign(_f=fam, _n=n).sort_values(['_f', '_n']).drop(columns=['_f', '_n'])
if 'add' in which:
    A = order_conds(pd.read_csv(R + 'additivity.csv')); chans = ['A1', 'A2', 'A3', 'A4', 'Mic1', 'Mic2', 'I1', 'I2', 'I3', 'Pr1', 'Pr2', 'Pr3', 'Pr4', 'A3 env', 'A4 env']
    M = A[[f'rho_{c}' for c in chans]].values
    fig = plt.figure(figsize=(7.0, 7.4)); ax = fig.add_axes([0.13, 0.47, 0.74, 0.48])
    im = ax.imshow(M, cmap='YlOrRd', norm=LogNorm(0.7, 15), aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=5.6, color='white' if M[i, j] > 5 else K)
    ax.set_xticks(range(len(chans))); ax.set_xticklabels(chans, rotation=90, fontsize=6.4); ax.set_yticks(range(len(A))); ax.set_yticklabels([f'{c}  {l}' for c, l in zip(A.cond, A.label)], fontsize=5.6)
    ax.axhline(11.5, color=K, lw=1.0); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label('Repeat ratio ρ', fontsize=7); cb.ax.tick_params(labelsize=6.2)
    panel(ax, '(a) Combination error relative to repetition variation, per channel', y=1.01, x=-0.17)
    ax = fig.add_axes([0.09, 0.06, 0.38, 0.30]); x = np.arange(len(A))
    ax.axhline(1, color=GR, lw=0.7, ls='--'); ax.axhline(0, color=GR, lw=0.5)
    ax.plot(x, A.alpha, 'o', ms=4, color='#0072B2', label='α (valve fault)'); ax.plot(x, A.beta, 's', ms=3.6, color='#E69F00', label='β (second fault)')
    ax.set_xticks(x); ax.set_xticklabels(A.cond, rotation=90, fontsize=5.6); ax.set_ylim(-0.7, 1.6); ax.axvline(11.5, color=K, lw=0.8)
    ax.text(18.5, -0.62, 'V-P22: α = 4.8', fontsize=5.6, ha='center')
    ax.legend(frameon=False, fontsize=6.2, loc='upper left', ncol=2, bbox_to_anchor=(0, 1.02)); ax.set_ylabel('Fitted weight'); panel(ax, '(b) Fitted pressure weights', y=1.04)
    T = pd.read_csv(F + 'conditions_all.csv'); T = T[T.config == 'C2 M-b (reference)'].groupby('cond')['loc'].mean()
    A['locacc'] = A.cond.map(T); ax = fig.add_axes([0.60, 0.06, 0.37, 0.30])
    for fam, mk, c in (('valve-bearing', 'o', ACC), ('valve-valve', 's', '#0072B2')):
        g = A[A.family == fam]; ax.scatter(g.p_add_share, g.locacc, marker=mk, s=22, color=c, edgecolor=K, lw=0.4, label=fam.replace('-', ' + '))
    rs = spearmanr(A.p_add_share, A.locacc); ax.set_xscale('log'); ax.set_xlabel('Pressure combination error (share)'); ax.set_ylabel('Location accuracy (mean of M1–M5)')
    ax.text(0.97, 0.95, f'Spearman $r_s$ = {rs.statistic:.2f}\np = {rs.pvalue:.4f}', transform=ax.transAxes, ha='right', va='top', fontsize=6.4)
    ax.legend(frameon=False, fontsize=6.2, loc='lower left'); ax.set_ylim(-0.03, 1.03); panel(ax, '(c) Additivity and diagnosis', y=1.04)
    print('add', save(fig, 'Fig_additivity'))
if 'cmp' in which:
    C = pd.read_csv(F + 'ci_all.csv'); const = json.load(open(R + 'const_baseline.json'))
    CF = [('single faults only', 'single faults only', 0.15, '////'), ('C1 random augmentation', 'C1 random', 0.4, '....'), ('C2 M-a', 'C2 M-a', 0.55, '\\\\\\\\'),
          ('C2 M-b (reference)', 'C2 M-b', 1.0, ''), ('C2 M-c', 'C2 M-c', 0.65, 'xxxx'), ('C2 M-d', 'C2 M-d', 0.8, '----'), ('C2 M-e', 'C2 M-e', 0.3, '||||')]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6), gridspec_kw=dict(hspace=0.45, wspace=0.12)); w = 0.8 / len(CF)
    for c_, (f, ft) in enumerate((('valve-bearing', 'Valve + bearing'), ('valve-valve', 'Valve + valve'))):
        for r_, (met, ml) in enumerate((('label', 'Exact-label accuracy'), ('loc', 'Location accuracy'))):
            ax = axs[r_, c_]
            for j, (cn, lab, al, h) in enumerate(CF):
                for i, m in enumerate(MODELS):
                    g = C[(C.config == cn) & (C.model == m) & (C.family == f) & (C.metric == met)]
                    if len(g): g = g.iloc[0]; ax.bar(i + (j - (len(CF) - 1) / 2) * w, g['mean'], w, color=CFGC[cn], ec=K, lw=0.35)
            cb = [q for q in const if q['family'] == f][0]; yv = cb['loc'] if met == 'loc' else cb['label_best_constant']
            ax.axhline(yv, color=GR, ls='--', lw=0.8); ax.text(4.55, yv + 0.02, 'constant rule', fontsize=5.8, color=GR, ha='right')
            ax.set_xticks(range(5)); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_ylim(0, 0.7 if met == 'label' else 1.05); ax.set_ylabel(ml if c_ == 0 else '')
            panel(ax, f'({"abcd"[r_ * 2 + c_]}) {ft}', y=1.02)
    hs = [Patch(fc=CFGC[cn], ec=K, lw=0.4, label=lab) for cn, lab, al, h in CF]
    fig.legend(handles=hs, frameon=False, ncol=4, fontsize=6.4, loc='upper center', bbox_to_anchor=(0.5, 1.03))
    print('cmp', save(fig, 'Fig_compare'))
if 'conf' in which:
    J = json.load(open(F + 'confusion.json')); fig, axs = plt.subplots(2, 5, figsize=(7.0, 3.3), gridspec_kw=dict(wspace=0.35, hspace=0.55, height_ratios=[5, 4]))
    for i, m in enumerate(MODELS):
        for r_, (k, lab) in enumerate((('DV', ['N', '1', '2', '3', '4']), ('B1', ['N', '1', '2', '3']))):
            Cm = np.array(J[m][k], float); rows_ = Cm[1:] if k == 'DV' else Cm[1:]; nrm = rows_ / np.maximum(rows_.sum(1, keepdims=True), 1)
            ax = axs[r_, i]; ax.imshow(nrm, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
            for a in range(nrm.shape[0]):
                for b in range(nrm.shape[1]):
                    if nrm[a, b] >= 0.05: ax.text(b, a, f'{nrm[a, b]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5.0, color='white' if nrm[a, b] > 0.6 else K)
            ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, fontsize=6); ax.set_yticks(range(len(lab) - 1)); ax.set_yticklabels(lab[1:], fontsize=6); ax.tick_params(length=0)
            if r_ == 0: ax.set_title(SH[m], fontsize=7.5, fontweight='bold')
            if i == 0: ax.set_ylabel('true ' + ('2S_DV' if k == 'DV' else 'B1') + ' mode', fontsize=6.6)
            if r_ == 1: ax.set_xlabel('predicted', fontsize=6.4)
    print('conf', save(fig, 'Fig_confusion'))
if 'reason' in which:
    S = json.load(open(R + 'bayes/bayes_params.json')); sig = json.load(open(R + 'reason/signatures.json')); syms = ['dP2', 'dP3', 'puls1', 'puls2', 'puls3', 'BPFO', 'BPFI', 'FTF']
    labs = [l for l in sig if l != 'healthy']; Z = np.array([np.array(S['delta'][l]) / np.array(S['b']) for l in labs])
    fig = plt.figure(figsize=(7.0, 5.6)); ax = fig.add_axes([0.10, 0.48, 0.34, 0.47])
    im = ax.imshow(np.clip(Z, -6, 6), cmap=DIV, vmin=-6, vmax=6, aspect='auto')
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if abs(Z[i, j]) >= 2: ax.text(j, i, f'{Z[i, j]:.0f}', ha='center', va='center', fontsize=4.8, color='white' if abs(Z[i, j]) > 4 else K)
    ax.set_xticks(range(8)); ax.set_xticklabels(syms, rotation=90, fontsize=6.2); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.6); ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label('signature δ / ψ', fontsize=6.6); cb.ax.tick_params(labelsize=6)
    panel(ax, '(a) Single-fault signatures', y=1.01, x=-0.28)
    H = pd.read_csv(R + 'bayes/help_harm.csv').groupby(['family', 'metric', 'model'])[['helped', 'harmed']].sum().reset_index()
    ax = fig.add_axes([0.60, 0.58, 0.38, 0.36]); x = np.arange(5); fams = (('valve-bearing', -0.2), ('valve-valve', 0.2))
    for f, off in fams:
        g = H[(H.family == f) & (H.metric == 'label')].set_index('model').loc[MODELS]
        ax.bar(x + off, g.helped, 0.38, color='#1b7837' if f == 'valve-bearing' else '#a6dba0', ec=K, lw=0.4)
        ax.bar(x + off, -g.harmed, 0.38, color='#762a83' if f == 'valve-bearing' else '#d9c4e3', ec=K, lw=0.4)
    ax.axhline(0, color=K, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_ylabel('Recordings (3 seeds)')
    ax.set_ylim(-40, 115)
    ax.legend(handles=[Patch(fc='#1b7837', ec=K, lw=0.4, label='corrected, V+B'), Patch(fc='#a6dba0', ec=K, lw=0.4, label='corrected, V+V'), Patch(fc='#762a83', ec=K, lw=0.4, label='broken, V+B'), Patch(fc='#d9c4e3', ec=K, lw=0.4, label='broken, V+V')], frameon=False, fontsize=5.6, ncol=2, loc='upper right')
    panel(ax, '(b) Exact label: corrected and broken', y=1.03)
    B = pd.read_csv(R + 'bayes/bayes_eval.csv'); ax = fig.add_axes([0.60, 0.08, 0.38, 0.33]); lams = sorted(B.lam.unique()); xl = [l if l > 0 else 0.01 for l in lams]
    for m in MODELS:
        for f, ls in (('valve-bearing', '-'), ('valve-valve', '--')):
            g = B[(B.model == m) & (B.family == f)].groupby('lam')['loc'].mean().loc[lams]; ax.plot(xl, g.values, ls, color=MC[m], lw=1.0, marker=MK[m], ms=2.8)
    ax.set_xscale('log'); ax.set_xticks(xl); ax.set_xticklabels(['0'] + [f'{l:g}' for l in lams[1:]], fontsize=6.4); ax.axvline(1, color=GR, lw=0.6, ls=':')
    ax.set_xlabel('Weight λ of the symptom likelihood'); ax.set_ylabel('Location accuracy'); ax.set_ylim(0, 0.65)
    ax.legend(handles=[Line2D([], [], color=K, ls='-', label='valve + bearing'), Line2D([], [], color=K, ls='--', label='valve + valve')] + [Line2D([], [], color=MC[m], marker=MK[m], ls='none', ms=3, label=SH[m]) for m in MODELS], frameon=False, fontsize=5.6, loc='lower right', ncol=2)
    panel(ax, '(d) Sensitivity to λ (λ = 1 fixed)', y=1.03)
    W = pd.read_csv(F + 'ci_all.csv'); ax = fig.add_axes([0.10, 0.07, 0.40, 0.22]); cfg = [('C2 M-b (reference)', 'model alone', SYSC['model']), ('C2 M-b + weighted abduction', '+ weighted abduction', SYSC['abduction']), ('C2 M-b + Bayesian layer', '+ Bayesian layer', SYSC['bayes'])]
    for j, (cn, lab, colr) in enumerate(cfg):
        for i, m in enumerate(MODELS):
            g = W[(W.config == cn) & (W.model == m) & (W.family == 'valve-bearing') & (W.metric == 'label')]
            if len(g): ax.bar(i + (j - 1) * 0.27, g['mean'].iloc[0], 0.27, color=colr, ec=K, lw=0.35)
    ax.set_xticks(range(5)); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_ylabel('Exact label, V+B'); ax.set_ylim(0, 0.6)
    ax.legend(handles=[Patch(fc=colr, ec=K, lw=0.4, label=lab) for _, lab, colr in cfg], frameon=False, fontsize=6, ncol=3, loc='upper left')
    panel(ax, '(c) Effect of the reasoning layers', y=1.03)
    print('reason', save(fig, 'Fig_reasoning'))
if 'attsl' in which:
    sys.path.insert(0, (SRC)); from data4 import SENS, COMPS
    A = json.load(open(R + 'sensorloss/attention.json')); S = pd.read_csv(R + 'sensorloss/sensorloss.csv')
    fig = plt.figure(figsize=(7.0, 5.4))
    for k, (key, t) in enumerate((('Dense|valve-bearing', '(a) M4 attention, valve + bearing'), ('GNN|valve-bearing', '(b) M5 attention, valve + bearing'))):
        ax = fig.add_axes([0.09 + k * 0.47, 0.60, 0.40, 0.32]); M = np.array(A[key])
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=0.6, aspect='auto')
        for i in range(5):
            for j in range(15):
                if M[i, j] >= 0.05: ax.text(j, i, f'{M[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=4.6, color='white' if M[i, j] > 0.35 else K)
        ax.set_xticks(range(15)); ax.set_xticklabels([s.replace('env', ' env') for s in SENS], rotation=90, fontsize=6); ax.set_yticks(range(5)); ax.set_yticklabels(COMPS, fontsize=6.4); ax.tick_params(length=0)
        ax.set_title(t, loc='left', fontsize=8, fontweight='bold')
    G = ['none', 'pressures', 'stage-1 valve sensors (A1, Mic1)', 'stage-2 valve sensors (A2, Mic2)', 'bearing accelerometers (A3, A4)', 'microphones', 'currents']
    GL = ['none', 'Pr1–Pr4', 'A1, Mic1', 'A2, Mic2', 'A3, A4', 'Mic1, Mic2', 'I1–I3']
    for k, (f, t) in enumerate((('valve-bearing', '(c) Sensors removed, valve + bearing'), ('valve-valve', '(d) Sensors removed, valve + valve'))):
        ax = fig.add_axes([0.09 + k * 0.47, 0.10, 0.40, 0.30])
        for j, (sy, al, h) in enumerate((('model', 1.0, ''), ('+ Bayesian layer', 0.4, '....'))):
            for i, m in enumerate(MODELS):
                v = [S[(S.group == g) & (S.model == m) & (S.system == sy) & (S.family == f)]['loc'].mean() for g in G]
                ax.plot(np.arange(len(G)) + (j - 0.5) * 0.18, v, MK[m], ms=3.2, color=MC[m] if j == 0 else 'white', mec=MC[m] if j == 0 else K, mew=0.5, ls='none')
        ax.set_xticks(range(len(G))); ax.set_xticklabels(GL, rotation=35, ha='right', fontsize=6.4); ax.set_ylim(0, 0.7); ax.set_ylabel('Location accuracy' if k == 0 else '')
        ax.set_title(t, loc='left', fontsize=8, fontweight='bold')
    hs = [Line2D([], [], marker=MK[m], ls='none', color=MC[m], label=SH[m]) for m in MODELS] + [Line2D([], [], marker='o', ls='none', color=K, label='filled: model'), Line2D([], [], marker='o', ls='none', color='white', mec=K, label='open: + Bayesian layer')]
    fig.legend(handles=hs, frameon=False, ncol=7, fontsize=6.2, loc='lower center', bbox_to_anchor=(0.5, -0.03))
    print('attsl', save(fig, 'Fig_attsl'))
if 'robust' in which:
    Mt = pd.read_csv(F + 'metrics_all.csv'); REF = 'C2 M-b (reference)'
    fig = plt.figure(figsize=(7.0, 6.0))
    V = [('C2 M-b random pairing', 'random\npairing'), ('C2 M-b without lower bound', 'no lower\nbound'), ('C2 M-b, test pair structures removed', 'test pairs\nremoved'), ('C2 M-b without A2', 'A2\nremoved')]
    for k, (fam, t) in enumerate((('valve-bearing', '(a) Change in location accuracy, valve + bearing'), ('valve-valve', '(b) Change in location accuracy, valve + valve'))):
        ax = fig.add_axes([0.08 + k * 0.49, 0.60, 0.42, 0.32]); w = 0.16
        for j, (cn, lab) in enumerate(V):
            for i, m in enumerate(MODELS):
                g = Mt[(Mt.config == cn) & (Mt.model == m) & (Mt.family == fam)]
                if not len(g): continue
                r = Mt[(Mt.config == REF) & (Mt.model == m) & (Mt.family == fam) & (Mt.seed.isin(g.seed))]
                ax.bar(j + (i - 2) * w, g['loc'].mean() - r['loc'].mean(), w, color=MC[m], ec=K, lw=0.3)
        ax.axhline(0, color=K, lw=0.6); ax.set_xticks(range(len(V))); ax.set_xticklabels([l for _, l in V], fontsize=6.4); ax.set_ylim(-0.55, 0.3)
        if k == 0: ax.legend(handles=[Patch(fc=MC[m], ec=K, lw=0.3, label=SH[m]) for m in MODELS], frameon=False, fontsize=5.8, ncol=5, loc='lower left')
        ax.set_title(t, loc='left', fontsize=8, fontweight='bold')
    ax = fig.add_axes([0.08, 0.10, 0.42, 0.33]); S = [(REF, 'A'), ('reference: Valve P1 (B)', 'B'), ('reference: both, pooled (C1)', 'C1'), ('reference: both, own campaign (C2)', 'C2')]
    for i, m in enumerate(MODELS):
        for fam, mk, off in (('valve-bearing', 'o', -0.12), ('valve-valve', 's', 0.12)):
            v = []
            for cn, _ in S:
                g = Mt[(Mt.config == cn) & (Mt.model == m) & (Mt.family == fam) & (Mt.seed == 0)]; v.append(g['loc'].mean() if len(g) else np.nan)
            ax.plot(np.arange(4) + off + (i - 2) * 0.03, v, MK[m] + ('-' if fam == 'valve-bearing' else '--'), color=MC[m], ms=3, lw=0.6, mec=K, mew=0.3, mfc=MC[m] if fam == 'valve-bearing' else 'white')
    ax.set_xticks(range(4)); ax.set_xticklabels(['A: Bearing P1', 'B: Valve P1', 'C1: pooled', 'C2: own'], fontsize=6.4); ax.set_ylabel('Location accuracy (seed 0)'); ax.set_ylim(0, 0.75)
    ax.set_title('(c) Healthy reference', loc='left', fontsize=8, fontweight='bold')
    ax.legend(handles=[Line2D([], [], marker=MK[m], ls='none', color=MC[m], mec=K, mew=0.3, label=SH[m]) for m in MODELS], frameon=False, fontsize=5.6, ncol=5, loc='upper center')
    ax = fig.add_axes([0.58, 0.10, 0.40, 0.33]); xs = []
    for i, (m, lab) in enumerate((('Dense', 'M4\n7 seeds'), ('GNN', 'M5\n7 seeds'))):
        g = Mt[(Mt.config == 'C2 M-b, ten seeds') & (Mt.model == m) & (Mt.family == 'valve-bearing')]
        ax.scatter(np.full(len(g), i) + np.linspace(-0.12, 0.12, len(g)), g['loc'], s=14, marker=MK[m], color=MC[m], edgecolor=K, lw=0.3)
    for K_ in range(1, 6):
        g = Mt[(Mt.config == f'M5 random mask {K_}') & (Mt.family == 'valve-bearing')]
        if len(g): ax.scatter([2 + (K_ - 3) * 0.06], [g['loc'].mean()], s=22, marker='X', color=tint(MC['GNN'], 0.4), edgecolor=K, lw=0.3)
    g = Mt[(Mt.config == REF) & (Mt.model == 'GNN') & (Mt.family == 'valve-bearing') & (Mt.seed == 0)]; ax.scatter([2.35], [g['loc'].mean()], s=40, marker='*', color=MC['GNN'], edgecolor=K, lw=0.4)
    ax.set_xticks([0, 1, 2.1]); ax.set_xticklabels(['M4\nseeds', 'M5\nseeds', 'M5: 5 random masks\n(crosses); star: physics'], fontsize=6.4); ax.set_ylim(0, 0.8); ax.set_ylabel('Location accuracy, V+B')
    ax.set_title('(d) Seeds and attention masks', loc='left', fontsize=8, fontweight='bold')
    print('robust', save(fig, 'Fig_robust'))
