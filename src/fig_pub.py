"""Figures of the test rig adapted from published open-access (CC BY 4.0) studies of the same rig:
Cerrada et al. (2020), Medina et al. (2022), Medina et al. (2024) and D. Cabrera et al. (2024)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import SRC, RAW, FEAT, WORK, RES, IMG
import sys; sys.path.insert(0, (SRC))
from col import *
from PIL import Image; Image.MAX_IMAGE_PIXELS = None
import numpy as np
I = (IMG) + '/'
WHITE = {'c20-004': [(260, 596, 296, 626), (880, 598, 916, 631)], 'c20-002': [(668, 314, 698, 339), (668, 674, 698, 702)]}   # remove the source panel letters
def load(k, up=3000):
    im = Image.open(I + k + '.png').convert('RGB')
    if k in WHITE:
        from PIL import ImageDraw
        d = ImageDraw.Draw(im)
        for b in WHITE[k]: d.rectangle(b, fill='white')
    if max(im.size) < up: s = up / max(im.size); im = im.resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
    return np.asarray(im)
def img(ax, k, t):
    ax.imshow(load(k), interpolation='lanczos'); ax.axis('off'); ax.set_title(t, loc='left', fontsize=8.5, fontweight='bold', pad=3)
# Figure: compressor and valves
fig = plt.figure(figsize=(7.0, 6.4))
img(fig.add_axes([0.0, 0.53, 1.0, 0.44]), 'c20-004', '(a) Compressor: external parts and internal view')
img(fig.add_axes([0.0, 0.0, 0.47, 0.47]), 'm24f4-000', '(b) Positions of the valves')
img(fig.add_axes([0.50, 0.0, 0.50, 0.47]), 'c20-002', '(c) Plate valve: parts and assembly')
print('rig', save(fig, 'Fig_rig'))
# Figure: sensors
fig = plt.figure(figsize=(7.0, 2.75))
img(fig.add_axes([0.0, 0.0, 0.62, 0.90]), 'm22-000', '(a) Sensor layout')
img(fig.add_axes([0.64, 0.0, 0.36, 0.90]), 'd24-000', '(b) Installed sensors')
print('sensors', save(fig, 'Fig_sensors'))
# Figure: seeded faults
rows = [('m24a-019', 'B1_1: inner-race crack, EDM slot 2.0 × 1.0 mm'), ('m24a-013', 'B1_2: roller crack, EDM slot 1.0 × 0.5 mm'),
        ('m24a-017', 'B1_3: outer-race crack, EDM slot 2.0 × 1.0 mm'), ('m24a-007', 'Mode 1: valve seat wear, seat depth reduced from 5.00 to 3.56 mm'),
        ('m24a-005', 'Mode 2: plate corrosion, through hole of 2.5 mm'), ('m24a-011', 'Mode 3: plate fracture, cut through the plate'), ('m24a-009', 'Mode 4: spring breakage, cut spring')]
fig = plt.figure(figsize=(7.0, 5.6))
for i, (k, t) in enumerate(rows[:3]):
    ax = fig.add_axes([0.0, 1 - (i + 1) / 3 + 0.05, 0.50, 1 / 3 - 0.11]); ax.imshow(load(k), interpolation='lanczos'); ax.axis('off'); ax.set_anchor('W')
    ax.set_title(f'({"abc"[i]}) ' + t.split(',')[0], loc='left', fontsize=8.5, fontweight='bold', pad=2)
    ax.text(0, -0.06, t.split(', ')[1], transform=ax.transAxes, va='top', fontsize=7.6)
for i, (k, t) in enumerate(rows[3:]):
    ax = fig.add_axes([0.56, 1 - (i + 1) / 4 + 0.04, 0.42, 1 / 4 - 0.085]); ax.imshow(load(k), interpolation='lanczos'); ax.axis('off'); ax.set_anchor('W')
    ax.set_title(f'({"defg"[i]}) ' + t.split(',')[0], loc='left', fontsize=8.5, fontweight='bold', pad=2)
    ax.text(0, -0.05, t.split(', ')[1], transform=ax.transAxes, va='top', fontsize=7.6)
print('faults', save(fig, 'Fig_faults'))
