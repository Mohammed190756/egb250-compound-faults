"""Folder layout. Set the environment variables, or keep the defaults (relative to the repository)."""
import os
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
RAW = os.environ.get('EGB_RAW', os.path.join(ROOT, 'data', 'RawData'))            # raw .mat recordings (not distributed)
FEAT = os.environ.get('EGB_FEAT', os.path.join(ROOT, 'data', 'features'))        # output of p2_extract.py
WORK = os.environ.get('EGB_WORK', os.path.join(ROOT, 'work'))                    # dataset.npz, models, figures
RES = os.environ.get('EGB_RES', os.path.join(ROOT, 'results'))                   # result files
IMG = os.environ.get('EGB_IMG', os.path.join(ROOT, 'third_party_images'))        # images extracted from the cited CC BY papers
for _d in (WORK, os.path.join(WORK, 'models'), os.path.join(WORK, 'models', 'rules'), os.path.join(WORK, 'figs'), RES):
    os.makedirs(_d, exist_ok=True)
