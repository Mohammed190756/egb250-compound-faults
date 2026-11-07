# egb250-compound-faults

Code and result files for the study *Learning unseen two-fault combinations of a reciprocating compressor from single-fault recordings: combination methods, physical additivity and Bayesian symptom reasoning*.

The models are trained on real healthy and single-fault recordings of the EGB-250 two-stage reciprocating compressor and on two-fault examples combined from them. They are evaluated on 300 real compound-fault recordings that never enter training or tuning:

- 180 valve-plus-bearing recordings;
- 120 valve-plus-valve recordings.

## Data

The EGB-250 reciprocating-compressor dataset was provided by GIDTEC, Universidad Politécnica Salesiana (Cuenca, Ecuador), under a data-sharing agreement, and is used with GIDTEC's permission. The dataset is the property of GIDTEC and is **not** included in this repository. Researchers who wish to use it should contact the authors of:

> Medina, R., Cerrada, M., Yang, S., Cabrera, D., Estupiñan, E., & Sánchez, R.-V. (2022). Fault classification in a reciprocating compressor and a centrifugal pump using non-linear entropy features. *Mathematics*, 10(17), 3033. https://doi.org/10.3390/math10173033

## Layout

| Folder | Content |
|---|---|
| `src/` | All code (one flat folder, so scripts import each other directly) |
| `results/` | Result files used for every number, table and figure of the paper |
| `run_all.sh` | Order in which the scripts were run |

Folders are set by environment variables (see `src/_paths.py`):

| Variable | Folder |
|---|---|
| `EGB_RAW` | Raw `.mat` recordings |
| `EGB_FEAT` | Extracted features |
| `EGB_WORK` | Intermediate dataset, model weights and figures |
| `EGB_RES` | Result files |

## Pipeline

| Step | Scripts | Paper |
|---|---|---|
| Feature extraction from the raw recordings: sampling rates, current scaling, shaft speed, crank angle, nine 1-s windows per recording | `p2_extract.py` | Sec. 4 |
| Crank-angle profiles, envelope order spectra, normalisation | `dataset.py`, `common.py`, `angle.py`, `nsdata.py`, `data4.py` | Sec. 5 |
| Combination of single faults: signal model, rules M-a to M-e, pairing schemes, symptoms | `data6.py`, `symptoms.py` | Sec. 6 |
| Models M1 to M5 and label decoding | `models4.py`, `rf6.py`, `train6.py` | Sec. 7 |
| Weighted abduction and Bayesian symptom likelihood | `reason6.py`, `bayes6.py` | Sec. 8 |
| Protocol, metrics, intervals and paired tests | `eval6.py`, `eval_rules.py`, `eval_final.py`, `metrics_ext.py` | Secs. 9–10 |
| Additivity of real compound faults | `additivity.py`, `diag6.py`, `diag_rules.py` | Secs. 9.5, 10.2 |
| Robustness, real-combination reference, sensor loss, cost | `train_var.py`, `train_supp.py`, `train_rules.py`, `sensorloss.py`, `timing.py`, `rf_timing.py` | Secs. 10.6–10.8, App. C |
| Figures | `col.py`, `fig6_methods.py`, `fig_methods2.py`, `fig6_results.py`, `fig_results2.py`, `fig6_rules.py`, `fig_pub.py` | Figs. 1, 5–18 |

The model names in the code map to the paper as follows:

| Code | Paper | Model |
|---|---|---|
| `RF` | M1 | Random forest |
| `MLP` | M2 | Multilayer perceptron |
| `CNN` | M3 | Convolutional network |
| `Dense` | M4 | GRU with dense attention |
| `GNN` | M5 | GRU with physics-masked attention |

**Settings.** The random seeds and configuration values are included:

- `seedcfg.json`;
- the hyperparameter searches in `results/tune_*.json`.

**Figures 2–4 of the paper** reproduce figures of published open-access (CC BY 4.0) studies of the same test rig. `fig_pub.py` assembles them from images that the user extracts from those papers into `EGB_IMG`. The images themselves are not distributed here.

## Requirements

- Python 3.10 or later.
- The packages in `requirements.txt`.

The extraction step reads the raw data and takes about 30 minutes with four processes. All later steps run on a CPU.

## Licence

MIT (see `LICENSE`). Please cite the archived release (see `CITATION.cff`) and the paper.
