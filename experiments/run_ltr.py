#!/usr/bin/env python3
"""P1.3 — Learning-to-rank (information-retrieval domain).

The popularity-prediction field measures RANK quality (Spearman/NDCG), not just
classification. We reframe pre-publication scoring as within-creator ranking:
given an author's candidates, order them by expected breakout. Compares an
XGBoost LambdaMART ranker (rank:pairwise) to our calibrated classifier's score,
on the temporal test split, by mean per-author Spearman + NDCG@3.

Output: reports/ltr.csv (+ stdout).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, config as C
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"]
    return TAB + ["semantic", "creator_fit", "trend_fit"]


def per_author_rank_metrics(df_idx, score, target_val, authors, min_n=3):
    sp, nd = [], []
    s = pd.DataFrame({"a": authors, "score": score, "y": target_val})
    for a, g in s.groupby("a"):
        if len(g) < min_n or g["y"].nunique() < 2:
            continue
        rho = spearmanr(g["score"], g["y"]).correlation
        if rho == rho:
            sp.append(rho)
        gain = g["y"].values - g["y"].values.min()        # NDCG needs non-negative relevance
        nd.append(ndcg_score([gain], [g["score"].values], k=3))
    return float(np.mean(sp)) if sp else float("nan"), float(np.mean(nd)) if nd else float("nan"), len(sp)


def main():
    from xgboost import XGBRanker, XGBClassifier
    df = D.load(eligible_only=True); F.add_derived(df)
    blocks = best_blocks()
    tr, va, te = S.temporal_masks(df)
    tri, tei = np.where(tr)[0], np.where(te)[0]
    yreg = df["breakout_wc"].astype(float).values            # continuous relevance
    ybin = df["y_breakout_wc"].astype(int).values
    auth = df["author_id"].values
    X, _ = F.build_blocks(df, blocks, tri, ybin)
    X = np.nan_to_num(X)

    # --- LambdaMART ranker: sort train by author to form contiguous groups ---
    order = np.argsort(auth[tri], kind="stable")
    tri_s = tri[order]
    grp = pd.Series(auth[tri_s]).groupby(pd.Series(auth[tri_s]), sort=False).size().values
    # relevance must be non-negative ints for ranker -> rank within author to 0..k
    rel = pd.Series(yreg[tri_s]).groupby(pd.Series(auth[tri_s]), sort=False).rank(method="dense").astype(int).values - 1
    rk = XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1)
    rk.fit(X[tri_s], rel, group=grp)
    rk_score = rk.predict(X[tei])

    # --- classifier score for comparison ---
    clf = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=-1,
                        eval_metric="logloss")
    clf.fit(X[tri], ybin[tri])
    clf_score = clf.predict_proba(X[te])[:, 1]

    rows = []
    for name, sc in [("LambdaMART(rank:pairwise)", rk_score), ("classifier(prob)", clf_score)]:
        rho, nd, k = per_author_rank_metrics(tei, sc, yreg[te], auth[te])
        rows.append({"scorer": name, "mean_author_spearman": round(rho, 4),
                     "mean_ndcg@3": round(nd, 4), "n_authors": k})
        print(f"  {name}: Spearman={rho:.4f}  NDCG@3={nd:.4f}  (authors={k})", flush=True)
    pd.DataFrame(rows).to_csv(C.REPORTS / "ltr.csv", index=False)
    print("saved -> reports/ltr.csv")


if __name__ == "__main__":
    main()
