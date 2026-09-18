"""P1.6 — Causal debiasing (lite): inverse-propensity reweighting by author size.

Popularity-bias = the model leaning on 'big author' rather than 'good creative'.
We reweight training by inverse propensity of the author-size bucket and check
whether (a) breakout AUC and (b) the prediction's correlation with author size
change. Expected/honest result: under the WITHIN-CREATOR label the fame signal is
already removed, so IPS adds little — quantifies that the label, not reweighting,
is doing the debiasing. Output: reports/ips.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, config as C
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"]
    return TAB + ["semantic", "creator_fit", "trend_fit"]


def main():
    from sip.modeling import make_model
    df = D.load(eligible_only=True); F.add_derived(df)
    tr, va, te = S.temporal_masks(df)
    tri, tei = np.where(tr)[0], np.where(te)[0]
    foll = pd.to_numeric(df["followers_at_post"], errors="coerce").fillna(1).clip(lower=1).values
    size = np.log1p(foll)
    X, _ = F.build_blocks(df, best_blocks(), tri, df["y_breakout_wc"].astype(int).values)
    X = np.nan_to_num(X)
    rows = []
    for target in ("y_breakout_wc",):
        y = df[target].astype(int).values
        # propensity = P(large-author bucket); IPS weight = 1/propensity, normalized
        buckets = pd.qcut(size[tri], 5, labels=False, duplicates="drop")
        freq = pd.Series(buckets).value_counts(normalize=True)
        w = (1.0 / freq.reindex(buckets).values); w = w / w.mean()
        for name, weights in [("unweighted", None), ("IPS(author-size)", w)]:
            m = make_model("hgb")
            m.fit(X[tri], y[tri], sample_weight=weights)
            p = m.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y[te], p)
            bias = spearmanr(p, size[te]).correlation   # corr of prediction with author size
            rows.append({"target": target, "weighting": name, "temporal_auc": round(auc, 4),
                         "pred_vs_authorsize_corr": round(float(bias), 4)})
            print(f"  {target} {name}: AUC={auc:.4f}  pred~size corr={bias:+.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.REPORTS / "ips.csv", index=False)
    print("saved -> reports/ips.csv")


if __name__ == "__main__":
    main()
