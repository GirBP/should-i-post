# ShouldIPost? — auto-generated results

_Generated from `reports/*` by `experiments/make_report.py`. AUC = ROC-AUC; LOCO = leave-one-creator-out; temporal = train<valid<test by date. CIs are 95% bootstrap._


## 1. Baselines (temporal test)

| setting                     |   roc_auc |   pr_auc |   brier |
|:----------------------------|----------:|---------:|--------:|
| y_breakout_wc/majority      |    0.5    |   0.4991 |  0.25   |
| y_breakout_wc/creator_prior |    0.3381 |   0.398  |  0.2831 |
| y_breakout_wc/caption_logit |    0.5133 |   0.5082 |  0.2499 |
| y_er_wc/majority            |    0.5    |   0.4684 |  0.2503 |
| y_er_wc/creator_prior       |    0.3587 |   0.3805 |  0.2743 |
| y_er_wc/caption_logit       |    0.5236 |   0.4837 |  0.2499 |


## 2. Staged ablation — Model A (marginal lift of each block)

| target        | stage        | model   |   loco_auc |   loco_ci_lo |   loco_ci_hi |   temporal_auc |   marginal |
|:--------------|:-------------|:--------|-----------:|-------------:|-------------:|---------------:|-----------:|
| y_breakout_wc | TAB          | logreg  |     0.5278 |       0.5255 |       0.5307 |         0.5248 |   nan      |
| y_breakout_wc | +priors      | logreg  |     0.5281 |       0.5255 |       0.5305 |         0.518  |     0.0003 |
| y_breakout_wc | +text        | logreg  |     0.5282 |       0.5257 |       0.5308 |         0.52   |     0.0001 |
| y_breakout_wc | +creator_fit | logreg  |     0.5289 |       0.5264 |       0.5314 |         0.5219 |     0.0007 |
| y_breakout_wc | +trend       | logreg  |     0.529  |       0.5265 |       0.5315 |         0.5221 |     0.0001 |
| y_breakout_wc | +retrieval   | logreg  |     0.5289 |       0.5264 |       0.5314 |         0.5221 |    -0.0001 |
| y_er_wc       | TAB          | logreg  |     0.5412 |       0.5387 |       0.5439 |         0.5416 |   nan      |
| y_er_wc       | +priors      | logreg  |     0.5478 |       0.5453 |       0.5503 |         0.5599 |     0.0066 |
| y_er_wc       | +text        | logreg  |     0.5456 |       0.5432 |       0.5479 |         0.5558 |    -0.0022 |
| y_er_wc       | +creator_fit | logreg  |     0.546  |       0.5437 |       0.5484 |         0.5569 |     0.0004 |
| y_er_wc       | +trend       | logreg  |     0.5461 |       0.5437 |       0.5484 |         0.5568 |     0.0001 |
| y_er_wc       | +retrieval   | logreg  |     0.5461 |       0.5437 |       0.5485 |         0.557  |     0      |


## 2b. Best Model-A config (TAB + text + creator-fit + trend, no priors; full data)

| target        | model   |   loco_auc | loco_ci          |   temporal_auc |   brier |
|:--------------|:--------|-----------:|:-----------------|---------------:|--------:|
| y_breakout_wc | logreg  |     0.5692 | [0.5666, 0.5719] |         0.5736 |  0.2506 |
| y_breakout_wc | hgb     |     0.5805 | [0.5778, 0.583]  |         0.5713 |  0.2461 |
| y_er_wc       | logreg  |     0.6    | [0.5979, 0.6024] |         0.614  |  0.2391 |
| y_er_wc       | hgb     |     0.6062 | [0.6038, 0.6088] |         0.6231 |  0.2379 |


## 3. Text encoders — marginal lift over TAB

| target        | encoder   |   loco_auc |   temporal_auc |   lift_over_TAB |
|:--------------|:----------|-----------:|---------------:|----------------:|
| y_breakout_wc | none      |     0.5278 |         0.5248 |          0      |
| y_breakout_wc | minilm    |     0.5642 |         0.5559 |          0.0364 |
| y_breakout_wc | bge       |     0.5674 |         0.5705 |          0.0396 |
| y_breakout_wc | e5        |     0.5722 |         0.5674 |          0.0444 |
| y_breakout_wc | svd       |     0.5611 |         0.5555 |          0.0333 |
| y_er_wc       | none      |     0.5412 |         0.5416 |          0      |
| y_er_wc       | minilm    |     0.5917 |         0.6058 |          0.0505 |
| y_er_wc       | bge       |     0.5981 |         0.6102 |          0.0569 |
| y_er_wc       | e5        |     0.6001 |         0.6153 |          0.0589 |
| y_er_wc       | svd       |     0.5943 |         0.6008 |          0.0531 |


## 4. Model family (fixed features)

| target        | model    |   loco_auc |   temporal_auc |   brier |
|:--------------|:---------|-----------:|---------------:|--------:|
| y_breakout_wc | logreg   |     0.5544 |         0.552  |  0.2555 |
| y_breakout_wc | hgb      |     0.5574 |         0.5579 |  0.2471 |
| y_breakout_wc | xgb      |     0.5584 |         0.5542 |  0.2481 |
| y_breakout_wc | catboost |     0.5618 |         0.5666 |  0.2464 |
| y_er_wc       | logreg   |     0.5877 |         0.5986 |  0.2467 |
| y_er_wc       | hgb      |     0.5851 |         0.6064 |  0.2408 |
| y_er_wc       | xgb      |     0.5879 |         0.6057 |  0.2403 |
| y_er_wc       | catboost |     0.5878 |         0.6084 |  0.2397 |


## 5. Optuna-tuned XGBoost vs default

- best LOCO AUC **0.5598** vs default 0.5518 (n=40001, 30 trials)
- best params: `{"n_estimators": 402, "max_depth": 6, "learning_rate": 0.01219502741029983, "subsample": 0.7126171624067774, "colsample_bytree": 0.83955878476835, "reg_lambda": 6.620243604606745, "min_child_weight": 12}`


## 6. LLM-as-judge — marginal lift

| target        | config         |   loco_auc |   temporal_auc |   lift_vs_TAB |
|:--------------|:---------------|-----------:|---------------:|--------------:|
| y_breakout_wc | TAB            |     0.512  |         0.5017 |        0      |
| y_breakout_wc | judge_only     |     0.4884 |         0.4999 |       -0.0236 |
| y_breakout_wc | TAB+judge      |     0.4967 |         0.5104 |       -0.0153 |
| y_breakout_wc | TAB+text       |     0.5212 |         0.5638 |        0.0092 |
| y_breakout_wc | TAB+text+judge |     0.5132 |         0.5486 |        0.0012 |
| y_breakout_wc | TAB+hook       |     0.5141 |         0.5027 |        0.0021 |
| y_breakout_wc | TAB+clarity    |     0.5199 |         0.5128 |        0.0079 |
| y_breakout_wc | TAB+trend_fit  |     0.4996 |         0.5012 |       -0.0124 |
| y_breakout_wc | TAB+arousal    |     0.5193 |         0.5137 |        0.0073 |
| y_breakout_wc | TAB+saturation |     0.5112 |         0.5013 |       -0.0008 |
| y_breakout_wc | TAB+cta        |     0.5065 |         0.5085 |       -0.0055 |
| y_er_wc       | TAB            |     0.5091 |         0.5141 |        0      |
| y_er_wc       | judge_only     |     0.526  |         0.5173 |        0.0169 |
| y_er_wc       | TAB+judge      |     0.5432 |         0.5176 |        0.0341 |
| y_er_wc       | TAB+text       |     0.5533 |         0.5471 |        0.0442 |
| y_er_wc       | TAB+text+judge |     0.5658 |         0.5438 |        0.0567 |
| y_er_wc       | TAB+hook       |     0.5164 |         0.5095 |        0.0073 |
| y_er_wc       | TAB+clarity    |     0.5161 |         0.5382 |        0.007  |
| y_er_wc       | TAB+trend_fit  |     0.5092 |         0.5186 |        0.0001 |
| y_er_wc       | TAB+arousal    |     0.5315 |         0.5139 |        0.0224 |
| y_er_wc       | TAB+saturation |     0.514  |         0.509  |        0.0049 |
| y_er_wc       | TAB+cta        |     0.509  |         0.5185 |       -0.0001 |


## 7. Multimodal ablation + fusion

_n=262 videos, 228 authors, base rate 0.672_

| target        | config                   | model   |   loco_auc |   temporal_auc |   marginal |   n_features |
|:--------------|:-------------------------|:--------|-----------:|---------------:|-----------:|-------------:|
| y_breakout_wc | TAB                      | hgb     |     0.5301 |         0.5576 |   nan      |           37 |
| y_breakout_wc | +TEXT                    | hgb     |     0.5551 |         0.5718 |     0.025  |          805 |
| y_breakout_wc | +AUDIO                   | hgb     |     0.5708 |         0.5812 |     0.0157 |         1413 |
| y_breakout_wc | +VIDEO_HOOK              | hgb     |     0.5721 |         0.6    |     0.0013 |         2447 |
| y_breakout_wc | +SigLIP                  | hgb     |     0.6237 |         0.5129 |     0.0516 |         3215 |
| y_breakout_wc | +creator_fit             | hgb     |     0.6237 |         0.5129 |     0      |         3217 |
| y_breakout_wc | +trend                   | hgb     |     0.6237 |         0.4753 |     0      |         3219 |
| y_breakout_wc | video=CLIP               | hgb     |     0.5328 |         0.3271 |   nan      |          549 |
| y_breakout_wc | video=SigLIP             | hgb     |     0.6403 |         0.5435 |   nan      |          805 |
| y_breakout_wc | video=hookCLIP           | hgb     |     0.583  |         0.6047 |   nan      |          549 |
| y_breakout_wc | audio=CLAP               | hgb     |     0.592  |         0.5882 |   nan      |          549 |
| y_breakout_wc | audio=lowlevel           | hgb     |     0.5599 |         0.4518 |   nan      |          133 |
| y_breakout_wc | visual_lowlevel          | hgb     |     0.5629 |         0.5294 |   nan      |           47 |
| y_breakout_wc | fusion=late(mean-logreg) | late    |     0.5738 |       nan      |   nan      |          nan |
| y_breakout_wc | fusion=early(HGB)        | hgb     |     0.6237 |         0.5129 |   nan      |         3215 |
| y_breakout_wc | fusion=neural(MLP)       | mlp     |     0.5282 |         0.5271 |   nan      |         3215 |
| y_breakout_wc | B_full(+day1)            | hgb     |     0.9152 |         0.9176 |   nan      |         3219 |
| y_er_wc       | TAB                      | hgb     |     0.4728 |         0.4097 |   nan      |           37 |
| y_er_wc       | +TEXT                    | hgb     |     0.569  |         0.3449 |     0.0962 |          805 |
| y_er_wc       | +AUDIO                   | hgb     |     0.5253 |         0.4514 |    -0.0437 |         1413 |
| y_er_wc       | +VIDEO_HOOK              | hgb     |     0.5321 |         0.625  |     0.0068 |         2447 |
| y_er_wc       | +SigLIP                  | hgb     |     0.5614 |         0.6435 |     0.0293 |         3215 |
| y_er_wc       | +creator_fit             | hgb     |     0.5614 |         0.6435 |     0      |         3217 |
| y_er_wc       | +trend                   | hgb     |     0.5614 |         0.6435 |     0      |         3219 |
| y_er_wc       | video=CLIP               | hgb     |     0.5453 |         0.5162 |   nan      |          549 |
| y_er_wc       | video=SigLIP             | hgb     |     0.5378 |         0.6065 |   nan      |          805 |
| y_er_wc       | video=hookCLIP           | hgb     |     0.5066 |         0.6134 |   nan      |          549 |
| y_er_wc       | audio=CLAP               | hgb     |     0.4519 |         0.4606 |   nan      |          549 |
| y_er_wc       | audio=lowlevel           | hgb     |     0.4851 |         0.4745 |   nan      |          133 |
| y_er_wc       | visual_lowlevel          | hgb     |     0.5731 |         0.5255 |   nan      |           47 |
| y_er_wc       | fusion=late(mean-logreg) | late    |     0.4909 |       nan      |   nan      |          nan |
| y_er_wc       | fusion=early(HGB)        | hgb     |     0.5614 |         0.6435 |   nan      |         3215 |
| y_er_wc       | fusion=neural(MLP)       | mlp     |     0.5229 |         0.5532 |   nan      |         3215 |
| y_er_wc       | B_full(+day1)            | hgb     |     0.7161 |         0.6644 |   nan      |         3219 |


## 8. Deployable A->B system (calibrated; test set)

**Model A** — AUC 0.5683 [0.5623, 0.5746], PR-AUC 0.5526, Brier 0.246, precision@Post 0.5662
  bands {'t_low': 0.4159, 't_high': 0.5473}  →  {'Post': {'coverage': 0.294, 'n': 9176, 'base_rate': 0.566}, 'Unsure': {'coverage': 0.658, 'n': 20511, 'base_rate': 0.482}, 'Do not post': {'coverage': 0.048, 'n': 1497, 'base_rate': 0.323}, 'abstain_coverage': 0.658}

**Model B** — AUC 0.7502 [0.745, 0.7558], PR-AUC 0.7381, Brier 0.2011, precision@Post 0.668
  bands {'t_low': 0.5109, 't_high': 0.5119}  →  {'Post': {'coverage': 0.51, 'n': 15916, 'base_rate': 0.668}, 'Unsure': {'coverage': 0.072, 'n': 2259, 'base_rate': 0.504}, 'Do not post': {'coverage': 0.417, 'n': 13009, 'base_rate': 0.292}, 'abstain_coverage': 0.072}


## 9. Label robustness — horizon + label definition + author-history (P0.2/P0.3)

| section     | setting               |   loco_auc |   base_rate |   agree_vs_H14 |   fame_leak_auc |    bestA |   bestA+authhist |   authhist_only |   lift |
|:------------|:----------------------|-----------:|------------:|---------------:|----------------:|---------:|-----------------:|----------------:|-------:|
| horizon     | H=7                   |     0.5708 |       0.498 |          0.964 |        nan      | nan      |         nan      |        nan      |    nan |
| horizon     | H=10                  |     0.5706 |       0.498 |          0.978 |        nan      | nan      |         nan      |        nan      |    nan |
| horizon     | H=14                  |     0.5704 |       0.498 |          0.995 |        nan      | nan      |         nan      |        nan      |    nan |
| horizon     | H=21                  |     0.5696 |       0.498 |          0.977 |        nan      | nan      |         nan      |        nan      |    nan |
| label_def   | within_creator_median |     0.5678 |       0.503 |        nan     |          0.5044 | nan      |         nan      |        nan      |    nan |
| label_def   | global_top30_breakout |     0.5945 |       0.3   |        nan     |          0.664  | nan      |         nan      |        nan      |    nan |
| label_def   | abs_logviews_top30    |     0.7153 |       0.3   |        nan     |          0.8565 | nan      |         nan      |        nan      |    nan |
| author_hist | y_breakout_wc         |   nan      |     nan     |        nan     |        nan      |   0.5678 |           0.5678 |          0.4973 |      0 |
| author_hist | y_er_wc               |   nan      |     nan     |        nan     |        nan      |   0.5982 |           0.5982 |          0.4991 |     -0 |


## 10. Other-domain methods (P1)

**Learning-to-rank vs classifier (per-author Spearman / NDCG@3):**
| scorer                    |   mean_author_spearman |   mean_ndcg@3 |   n_authors |
|:--------------------------|-----------------------:|--------------:|------------:|
| LambdaMART(rank:pairwise) |                 0.1597 |        0.4451 |        1495 |
| classifier(prob)          |                 0.1799 |        0.4492 |        1495 |

**Trajectory / point-process — when is day-14 breakout determined:**
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

**Multi-task (shared trunk) vs single-task:**
| model                  |   breakout_temporal_auc |    std |
|:-----------------------|------------------------:|-------:|
| single-task (breakout) |                  0.5584 | 0.0064 |
| multi-task (3 heads)   |                  0.5302 | 0.0094 |

**Deep / foundation tabular vs GBT:**
| model          |     auc | note                                 |
|:---------------|--------:|:-------------------------------------|
| hgb            |   0.551 | PCA-64; (xgb~0.555 see model_family) |
| FT-Transformer |   0.545 | custom, PCA-64                       |
| TabPFN-v2      | nan     | unavailable: TabPFNLicenseError      |

**Causal debiasing (IPS by author size):**
| target        | weighting        |   temporal_auc |   pred_vs_authorsize_corr |
|:--------------|:-----------------|---------------:|--------------------------:|
| y_breakout_wc | unweighted       |         0.5763 |                    0.1639 |
| y_breakout_wc | IPS(author-size) |         0.576  |                    0.1678 |
