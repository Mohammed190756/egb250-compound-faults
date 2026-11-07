import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import to_rgb
import matplotlib.text as _mt
FS = 1.38   # all font sizes scaled so that text prints at 9-10 pt when the 7-in figure is placed at 16 cm
_orig = _mt.Text.set_fontsize
def _set_fs(self, fontsize):
    try: fontsize = float(fontsize) * FS
    except (TypeError, ValueError): pass
    return _orig(self, fontsize)
_mt.Text.set_fontsize = _set_fs; _mt.Text.set_size = _set_fs
plt.rcParams.update({'font.family': 'Liberation Serif', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': 0.7, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2, 'legend.fontsize': 7.2, 'axes.spines.top': False, 'axes.spines.right': False,
    'hatch.linewidth': 0.6, 'lines.linewidth': 1.2, 'axes.edgecolor': '#222222', 'text.color': '#111111', 'axes.labelcolor': '#111111',
    'xtick.color': '#222222', 'ytick.color': '#222222', 'axes.grid': False})
K = '#111111'; W = '#ffffff'; GR = '#8c8c8c'; LG = '#e6e6e6'
MC = {'RF': '#E69F00', 'MLP': '#CC79A7', 'CNN': '#009E73', 'Dense': '#56B4E9', 'GNN': '#0072B2'}
ACC = '#D55E00'
LBL = {'RF': 'M1 RF', 'MLP': 'M2 MLP', 'CNN': 'M3 CNN', 'Dense': 'M4 dense att.', 'GNN': 'M5 masked att.'}
SEQ = 'Blues'; DIV = 'RdBu_r'
FILL = {'w': W, 'l': W, 'learn': W, 'phys': W, 'data': W, 'test': W, 'aug': W}   # diagrams: white boxes, black outlines (IJPHM: no coloured backgrounds)
import matplotlib as _mpl
_vir = _mpl.colormaps['viridis']
# one colour per configuration (plain bars); lightness differs so bars stay distinct in greyscale
CFGC = {'single faults only': '#d9d9d9', 'C1 random augmentation': _vir(0.92), 'C2 M-a': _vir(0.72), 'C2 M-b (reference)': '#08306b',
        'C2 M-c': _vir(0.52), 'C2 M-d': _vir(0.33), 'C2 M-e': _vir(0.15)}
SYSC = {'model': '#08306b', 'abduction': '#4292c6', 'bayes': '#c6dbef'}   # model alone, + weighted abduction, + Bayesian layer
MK = {'RF': 'o', 'MLP': 's', 'CNN': '^', 'Dense': 'D', 'GNN': 'v'}       # marker per model (greyscale-safe)
def tint(c, a=0.4):
    r, g, b = to_rgb(c); return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)
OUT = (WORK + '/figs') + '/'
def save(fig, name):
    fig.savefig(OUT + name + '.png', dpi=1000, bbox_inches='tight', facecolor='white'); fig.savefig(OUT + name + '_preview.png', dpi=150, bbox_inches='tight', facecolor='white'); plt.close(fig)
    from PIL import Image; Image.MAX_IMAGE_PIXELS = None; im = Image.open(OUT + name + '.png').convert('RGB')
    if im.size[0] * im.size[1] > 39.5e6:
        s = (39.5e6 / (im.size[0] * im.size[1])) ** 0.5; im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    im.save(OUT + name + '.png', dpi=(1000, 1000)); return im.size
def canvas(ax, xl=100, yl=100): ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis('off')
def box(ax, x, y, w, h, title='', lines=(), f='w', ts=7.6, ls=6.8, lw=0.9, ls_style='-', ec=K, tc=K):
    if f == 'test': lw = max(lw, 1.6)   # test boxes: thick outline instead of a dark fill
    tc = K; ec = K
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.8', fc=W, ec=ec, lw=lw, ls=ls_style))
    if title and lines:
        ax.text(x + w / 2, y + h - 1.0, title, ha='center', va='top', fontsize=ts, fontweight='bold', color=tc)
        ax.text(x + w / 2, y + h - 1.0 - ts * 0.62, '\n'.join(lines), ha='center', va='top', fontsize=ls, linespacing=1.3, color=tc)
    elif title: ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=ts, fontweight='bold', color=tc)
    else: ax.text(x + w / 2, y + h / 2, '\n'.join(lines), ha='center', va='center', fontsize=ls, linespacing=1.3, color=tc)
def arr(ax, x1, y1, x2, y2, label='', ls='-', lx=0, ly=0, fs=6.6, rad=0.0, lw=0.8, c=K):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=lw, color=c, ls=ls, connectionstyle=f'arc3,rad={rad}'))
    if label: ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=fs, ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.3))
def panel(ax, t, x=0.0, y=1.02, fs=8.5): ax.text(x, y, t, transform=ax.transAxes, fontsize=fs, fontweight='bold', va='bottom')
