import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys, json; sys.path.insert(0, (SRC))
from col import *
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
R = (RES) + '/rules/'; MODELS = ['RF', 'MLP', 'CNN', 'Dense', 'GNN']; SH = {'RF': 'M1', 'MLP': 'M2', 'CNN': 'M3', 'Dense': 'M4', 'GNN': 'M5'}
RS = {'a': ('M-a deviation sum', 0.30, '....'), 'b': ('M-b superposition (main)', 1.0, ''), 'c': ('M-c dominant fault', 0.55, '\\\\\\\\'), 'd': ('M-d varied severity', 0.80, 'xxxx')}
C = pd.read_csv(R + 'ci_rules.csv'); rules = [r for r in 'abcd' if r in set(C.rule)]; const = json.load(open((RES) + '/const_baseline.json'))
fig = plt.figure(figsize=(7.0, 7.6)); FAMS = (('valve-bearing', 'Valve + bearing'), ('valve-valve', 'Valve + valve'))
pos = {(0, 0): [0.07, 0.70, 0.42, 0.22], (0, 1): [0.57, 0.70, 0.42, 0.22], (1, 0): [0.07, 0.39, 0.42, 0.22], (1, 1): [0.57, 0.39, 0.42, 0.22]}
w = 0.8 / len(rules)
for c_, (f, ft) in enumerate(FAMS):
    for r_, (met, ml) in enumerate((('label', 'Exact-label accuracy'), ('loc', 'Location accuracy'))):
        ax = fig.add_axes(pos[(r_, c_)]); x = np.arange(5)
        for j, r in enumerate(rules):
            for i, m in enumerate(MODELS):
                g = C[(C.rule == r) & (C.model == m) & (C.family == f) & (C.metric == met)]
                if len(g) == 0: continue
                g = g.iloc[0]; ax.bar(x[i] + (j - (len(rules) - 1) / 2) * w, g['mean'], w, yerr=[[g['mean'] - g.lo], [g.hi - g['mean']]], color=tint(MC[m], RS[r][1]), ec=K, lw=0.45, hatch=RS[r][2], error_kw=dict(lw=0.5, capsize=0.8, ecolor=K))
        if met == 'loc':
            cb = [q for q in const if q['family'] == f][0]['loc']; ax.axhline(cb, color=GR, ls='--', lw=0.8); ax.text(4.55, cb + 0.02, 'constant rule', fontsize=6.0, color=GR, ha='right')
        ax.set_xticks(x); ax.set_xticklabels([SH[m] for m in MODELS]); ax.set_ylim(0, 1.05); ax.set_ylabel(ml if c_ == 0 else '')
        panel(ax, f'({"abcd"[r_ * 2 + c_]}) {ft}: {ml[0].lower() + ml[1:].replace(" accuracy", "")}', y=1.03, fs=8)
hs = [Patch(fc=tint('#555555', RS[r][1]), ec=K, lw=0.5, hatch=RS[r][2], label=RS[r][0]) for r in rules]
fig.legend(handles=hs, frameon=False, ncol=4, fontsize=6.6, loc='upper center', bbox_to_anchor=(0.5, 0.985))
# (e) similarity of combined rows per rule
S = pd.DataFrame(json.load(open(R + 'similarity_rules.json'))); T = pd.read_csv((RES) + '/test_conditions.csv')
code = T.drop_duplicates('label').set_index('label').cond.str.replace('Multi_P', 'MF-P').str.replace('Valve_P', 'V-P'); S['code'] = S.label.map(code)
S['num'] = S.code.str.extract(r'P(\d+)').astype(int)[0]; S = S.sort_values(['family', 'num']).reset_index(drop=True)
ax = fig.add_axes([0.07, 0.07, 0.92, 0.20]); xx = np.arange(len(S))
MK = {'a': ('o', 'white', 5), 'b': ('D', ACC, 4.5), 'c': ('v', '#8c8c8c', 5), 'd': ('s', '#222222', 4)}
for r in 'abcd':
    mk, fc, ms = MK[r]; ax.plot(xx, S[r], ls='none', marker=mk, mfc=fc, mec=K, mew=0.6, ms=ms, label=RS[r][0].replace(' (main)', ''))
ax.axvline(11.5, color=ACC, lw=1.0); ax.set_xticks(xx); ax.set_xticklabels(S.code, rotation=90, fontsize=6.2); ax.set_xlim(-0.6, len(S) - 0.4)
ax.set_ylabel('Correlation with\nreal input'); ax.set_ylim(0.2, 1.0); ax.legend(frameon=False, ncol=4, fontsize=6.4, loc='lower center', bbox_to_anchor=(0.5, 1.0))
panel(ax, '(e) Similarity of combined rows to the real combination, per rule (diagnostic)', y=1.13, fs=8)
fig.text(0.5, -0.04, 'Bars: mean of three seeds; whiskers: 95% cluster-bootstrap interval over conditions. All rules use the M-b hyperparameters.', ha='center', fontsize=6.6)
print(save(fig, 'Fig_rules'))
