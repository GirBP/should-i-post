# Extra methods sweep (temporal test, within-creator breakout)

_sample n=60001, best-A features, encoder=bge. Ceiling is information-bound: same-information methods cluster near baseline; only new-information beats it._

| method                          | family             |      auc |   secs | note                                                                     |
|:--------------------------------|:-------------------|---------:|-------:|:-------------------------------------------------------------------------|
| best-A HGB (reference)          | baseline           |   0.5666 |  nan   | nan                                                                      |
| gradient_boosting               | cpu-classifier     |   0.563  |  799.6 | nan                                                                      |
| logreg_balanced                 | cpu-classifier     |   0.5623 |   42.9 | nan                                                                      |
| lda                             | cpu-classifier     |   0.5576 |    7   | nan                                                                      |
| deep_ensemble(5xMLP)            | torch              |   0.5502 |  nan   | nan                                                                      |
| gaussian_nb                     | cpu-classifier     |   0.5481 |    0.6 | nan                                                                      |
| random_forest                   | cpu-classifier     |   0.5468 |  375.2 | nan                                                                      |
| focal_loss_mlp                  | torch              |   0.5425 |  nan   | nan                                                                      |
| bagging                         | cpu-classifier     |   0.5419 | 1335.2 | nan                                                                      |
| extra_trees                     | cpu-classifier     |   0.5401 |  178.7 | nan                                                                      |
| best-A + graph_hashtag_neighbor | graph              |   0.534  |  nan   | nan                                                                      |
| mlp_sklearn                     | cpu-classifier     |   0.53   |   52.1 | nan                                                                      |
| adaboost                        | cpu-classifier     |   0.5186 |  226.5 | nan                                                                      |
| sgd_log                         | cpu-classifier     |   0.517  |   11.2 | nan                                                                      |
| svc_rbf                         | cpu-classifier     |   0.5131 |  256   | nan                                                                      |
| knn                             | cpu-classifier     |   0.5031 |    8.1 | nan                                                                      |
| RULE-BASED (expert system)      | rules              |   0.4932 |  nan   | nan                                                                      |
| qda                             | cpu-classifier     | nan      |  nan   | LinAlgError: The covariance matrix of class 0 is not full rank. Increase |
| TabPFN-v2                       | tabular-foundation | nan      |  nan   | skip: TabPFNLicenseError                                                 |
| TabNet                          | tabular-dl         | nan      |  nan   | skip: ModuleNotFoundError                                                |