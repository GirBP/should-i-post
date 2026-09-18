# Methods from other domains — results & honest verdicts

Pre-publication Model-A baseline for reference: **best-A LOCO ROC-AUC ≈ 0.588 (breakout) / 0.61 (ER)**; Model B (day-1) ≈ 0.95. Each method below is judged vs that.

## Trajectory / point-process — when is the day-14 outcome determined? (forecasting + cascade)
**Verdict:** the outcome is **front-loaded** — day-1 alone already gives ~91% of the day-14 separability (AUC ~0.71), day-7 ~99%. Log-logistic growth-curve extrapolation ≈ the raw early value (no gain at daily granularity). Explains why pre-pub (day-0) is hard and Model B (day-1) is easy; matches SEISMIC (~15% err after 1h) and SMTPD (day-1 → SRC 0.95) from the literature.
| section       |   day |   auc_vs_breakout14 |   frac_of_day14_auc |   auc_extrap_to_14 |   auc_raw_day_d |    n |
|:--------------|------:|--------------------:|--------------------:|-------------------:|----------------:|-----:|
| early_window  |     0 |              0.5956 |               0.417 |           nan      |        nan      |  nan |
| early_window  |     1 |              0.7097 |               0.914 |           nan      |        nan      |  nan |
| early_window  |     2 |              0.7185 |               0.953 |           nan      |        nan      |  nan |
| early_window  |     3 |              0.722  |               0.968 |           nan      |        nan      |  nan |
| early_window  |     5 |              0.725  |               0.981 |           nan      |        nan      |  nan |
| early_window  |     7 |              0.7267 |               0.989 |           nan      |        nan      |  nan |
| early_window  |    14 |              0.7293 |               1     |           nan      |        nan      |  nan |
| growth_extrap |     1 |            nan      |             nan     |             0.7188 |          0.7188 | 8000 |
| growth_extrap |     3 |            nan      |             nan     |             0.7308 |          0.7314 | 8000 |
| growth_extrap |     7 |            nan      |             nan     |             0.7358 |          0.7355 | 8000 |

## Learning-to-rank vs calibrated classifier (information retrieval)
**Verdict:** the LambdaMART ranker does **not** beat the calibrated classifier's score for within-creator ranking (Spearman ~0.16 vs ~0.18); the classifier also gives calibration for free. Per-author Spearman ~0.18 = modest pre-publication ordering ability, consistent with the content-only ceiling.
| scorer                    |   mean_author_spearman |   mean_ndcg@3 |   n_authors |
|:--------------------------|-----------------------:|--------------:|------------:|
| LambdaMART(rank:pairwise) |                 0.1597 |        0.4451 |        1495 |
| classifier(prob)          |                 0.1799 |        0.4492 |        1495 |

## Multi-task (shared trunk) vs single-task (deep learning)
**Verdict:** multi-task **hurts** here (breakout AUC drops vs single-task) — the ER/follower heads pull the shared representation off the breakout target. Honest negative; single-task is better.
| model                  |   breakout_temporal_auc |    std |
|:-----------------------|------------------------:|-------:|
| single-task (breakout) |                  0.5584 | 0.0064 |
| multi-task (3 heads)   |                  0.5302 | 0.0094 |

## Deep / foundation tabular vs GBT (tabular-DL)
**Verdict:** see table — tests whether FT-Transformer / TabPFN-v2 beat HGB/XGB on our features (expectation from §5: they do not, the ceiling is information not model).
| model          |     auc | note                                 |
|:---------------|--------:|:-------------------------------------|
| hgb            |   0.551 | PCA-64; (xgb~0.555 see model_family) |
| FT-Transformer |   0.545 | custom, PCA-64                       |
| TabPFN-v2      | nan     | unavailable: TabPFNLicenseError      |

## Causal debiasing — IPS by author size (causal ML)
**Verdict:** IPS reweighting changes **nothing** (AUC and prediction-vs-author-size correlation unchanged) — the within-creator label has already removed the fame/popularity bias, so reweighting is redundant. Confirms the label, not IPS, does the debiasing.
| target        | weighting        |   temporal_auc |   pred_vs_authorsize_corr |
|:--------------|:-----------------|---------------:|--------------------------:|
| y_breakout_wc | unweighted       |         0.5763 |                    0.1639 |
| y_breakout_wc | IPS(author-size) |         0.576  |                    0.1678 |

## Summary
- **Confirmed the ceiling is information-bound:** stronger/other-domain models do not beat the simple calibrated GBT on pre-publication content (LTR, multi-task, tabular-DL all ≤ baseline).
- **Trajectory analysis pinpoints the missing signal:** day-1 determines ~91% of the outcome → the gap is early-watch dynamics, which is exactly the SnapUGC watch%/retention transfer target (P3.2).
- **IPS redundant** under the within-creator label (honest negative, as predicted).