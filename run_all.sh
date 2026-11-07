#!/usr/bin/env bash
# Order used to produce the results. Set EGB_RAW, EGB_FEAT, EGB_WORK and EGB_RES first (see src/_paths.py).
set -e
cd "$(dirname "$0")/src"
python p2_extract.py "$EGB_RAW" "$EGB_FEAT" 4          # 1. features from the raw recordings
python dataset.py                                     # 2. crank-angle profiles, envelope spectra, window table
M="MLP CNN Dense GNN"                                 # M2, M3, M4, M5 (M1 = RF)
python train6.py tune $M                              # 3. hyperparameter search on the validation repetitions
python train6.py final $M; python train6.py base $M   # 4. final models (reference rule M-b) and single-fault baseline
python rf6.py                                         #    M1: search, final model and single-fault baseline
python train_rules.py acd $M RF                       # 5. other combining rules
for V in unseen coh rand nofloor noA2 aug; do python train_var.py $V $M RF; done   # 6. robustness variants
python train_var.py seeds $M
for K in 1 2 3 4 5; do P6MASK=$K python train_var.py mask GNN; done
for S in B C1 C2; do P6SETUP=$S python train_supp.py abl $M RF; done
python train_supp.py cv 1,2,3,4,5,6,7,8 $M RF         # 7. training with real combinations (upper reference)
python reason6.py; python bayes6.py                   # 8. weighted abduction and Bayesian symptom likelihood
python additivity.py; python diag6.py; python diag_rules.py   # 9. additivity of real compound faults
python eval6.py; python eval_rules.py; python eval_final.py; python metrics_ext.py   # 10. metrics, intervals, tests
python sensorloss.py; python timing.py; python rf_timing.py   # 11. sensor loss and computational cost
python fig6_methods.py; python fig_methods2.py; python fig6_results.py; python fig_results2.py; python fig6_rules.py   # 12. figures
