# Model comparison (temporal test set, decision @ 0.5)

| model | roc_auc | pr_auc | brier | precision_post | recall_post | f1_post | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.500 | 0.165 | 0.250 | 0.000 | 0.000 | 0.000 | 0.835 |
| creator_historical | 0.445 | 0.166 | 0.250 | 0.032 | 0.029 | 0.031 | 0.693 |
| logreg3 | 0.479 | 0.158 | 0.238 | 0.148 | 0.294 | 0.197 | 0.603 |
| logreg_full | 0.522 | 0.167 | 0.198 | 0.179 | 0.176 | 0.178 | 0.730 |
| hgb | 0.476 | 0.157 | 0.285 | 0.159 | 0.706 | 0.260 | 0.336 |

- **Deployed: `logreg_full`** (lowest OOF Brier). OOF Brier: {'logreg_full': 0.2368, 'hgb': 0.2815}
- Operating point: t_low=0.444, t_high=0.563 (precision target reached: True); precision_post=0.067, recall_post=0.029, coverage Post/Unsure/No=0.07/0.23/0.69
- ROC-AUC 95% CI: (0.4499337808404061, 0.5874537227644977); PR-AUC 95% CI: (0.13145289570852753, 0.22430018535664858)
- Random 5-fold CV AUC (signal, no drift): **0.6470633746772163**
- Leave-one-creator-out AUC: 0.5701690829825972
- Fame demo — creator-only AUC under label A: 0.6301573330399384, under label C: 0.4868051783451023
