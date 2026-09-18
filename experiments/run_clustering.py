"""Assess clustering as an approach for the pre-publication task, empirically.

Three clustering uses are tested (temporal split, fold-safe = fit on train only):
1. CONTENT-CLUSTER prior: KMeans on text embeddings -> cluster's train success-rate as a
   feature (a learned, soft version of the topic/hashtag target-encoding we already found
   HURTS cross-creator transfer). Marginal over best-A.
2. COLD-AUTHOR backstop: does the cluster prior help specifically for authors with little
   own history (where the within-creator signal is thin)? — the one place clustering should help.
3. UNSUPERVISED-ONLY: cluster membership alone as the predictor (sanity: clustering is not a
   supervised predictor; should trail the supervised model).

Output: reports/clustering.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, config as C
from sip.modeling import make_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"], n
    return TAB + ["semantic", "creator_fit", "trend_fit"], "svd"


def auc_on(Xtr, ytr, Xte, yte):
    m = make_model("logreg").fit(np.nan_to_num(Xtr), ytr)
    return round(float(roc_auc_score(yte, m.predict_proba(np.nan_to_num(Xte))[:, 1])), 4)


def main():
    df = D.load(eligible_only=True); F.add_derived(df)
    blocks, tn = best_blocks()
    y = df["y_breakout_wc"].astype(int).values
    tr, va, te = S.temporal_masks(df)
    tri = np.where(tr)[0]
    Xbest, _ = F.build_blocks(df, blocks, tri, y)
    Xemb, _ = F.build_blocks(df, [f"text_emb:{tn}"] if tn != "svd" else ["semantic"], tri, y)

    # fold-safe KMeans on train embeddings
    pca = PCA(n_components=min(64, Xemb.shape[1]), random_state=0).fit(np.nan_to_num(Xemb[tri]))
    Z = pca.transform(np.nan_to_num(Xemb))
    K = 50
    km = KMeans(n_clusters=K, random_state=0, n_init=4).fit(Z[tri])
    cl = km.predict(Z)
    rate = pd.Series(y[tri]).groupby(cl[tri]).mean()
    gm = float(y[tri].mean())
    cl_rate = pd.Series(cl).map(rate).fillna(gm).values.reshape(-1, 1)
    size = pd.Series(cl).map(pd.Series(cl[tri]).value_counts()).fillna(0).values.reshape(-1, 1)
    cl_feat = np.c_[cl_rate, np.log1p(size)]            # cluster success-prior + saturation

    rows = []
    a_best = auc_on(Xbest[tri], y[tri], Xbest[te], y[te])
    a_cl = auc_on(np.c_[Xbest, cl_feat][tri], y[tri], np.c_[Xbest, cl_feat][te], y[te])
    a_only = auc_on(cl_feat[tri], y[tri], cl_feat[te], y[te])
    rows.append({"setting": "best-A", "auc": a_best})
    rows.append({"setting": "best-A + content-cluster prior", "auc": a_cl, "lift": round(a_cl - a_best, 4)})
    rows.append({"setting": "cluster prior ONLY (unsupervised-derived)", "auc": a_only})

    # cold-author backstop: authors with few prior videos
    nprior = F.build_blocks(df, ["author_history"], tri, y)[0][:, 2]   # author_nprior_log
    cold = (nprior <= np.log1p(2))                                      # <=2 prior videos
    cold_te = te & cold
    if cold_te.sum() > 200:
        cti = np.where(cold_te)[0]
        b = round(float(roc_auc_score(y[cti], make_model("logreg").fit(np.nan_to_num(Xbest[tri]), y[tri])
                                       .predict_proba(np.nan_to_num(Xbest[cti]))[:, 1])), 4)
        cfeat = np.c_[Xbest, cl_feat]
        bc = round(float(roc_auc_score(y[cti], make_model("logreg").fit(np.nan_to_num(cfeat[tri]), y[tri])
                                        .predict_proba(np.nan_to_num(cfeat[cti]))[:, 1])), 4)
        rows.append({"setting": f"COLD authors (<=2 prior, n={int(cold_te.sum())}): best-A", "auc": b})
        rows.append({"setting": "COLD authors: best-A + cluster prior", "auc": bc, "lift": round(bc - b, 4)})

    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "clustering.csv", index=False)
    print(out.to_string(index=False))
    print("saved -> reports/clustering.csv")


if __name__ == "__main__":
    main()
