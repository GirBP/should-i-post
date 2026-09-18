"""Assess the "model the author's history (+ other authors) to predict beating the
author's median" idea, empirically.

The proposed TARGET ('beats the author's own median views') == our primary
within-creator breakout label. The proposed FEATURES == author history. We already
have author_history (lifetime aggregate) + creator_fit (similarity to own past
winners) + a global cross-author model. The genuinely NEW piece is a SEQUENTIAL /
RECENCY view of the author's recent form. This quantifies its marginal value, and
contrasts the within-creator target vs an absolute-views target to expose the fame trap.

Output: reports/author_history_model.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, eval as E, config as C
from sip.modeling import make_model

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"], n
    return TAB + ["semantic", "creator_fit", "trend_fit"], "svd"


def author_recency(df):
    """Closed-window sequential author features (uses sip.features.block_author_recency, which
    only aggregates prior videos whose H-day outcome was observable by post time — leak-safe)."""
    return F.block_author_recency(df, np.arange(len(df)), df["y_breakout_wc"].astype(int).values)


def loco_auc(df, X, y, model="logreg"):
    oof = np.full(len(df), np.nan)
    for tri, tei in S.loco_folds(df):
        if len(np.unique(y[tri])) < 2:
            continue
        m = make_model(model).fit(np.nan_to_num(X[tri]), y[tri])
        oof[tei] = m.predict_proba(np.nan_to_num(X[tei]))[:, 1]
    ok = ~np.isnan(oof)
    return round(E._safe_auc(y[ok], oof[ok]), 4)


def main():
    df = D.load(eligible_only=True); F.add_derived(df)
    blocks, tn = best_blocks()
    y = df["y_breakout_wc"].astype(int).values
    Xbest, _ = F.build_blocks(df, blocks, np.arange(len(df)), y)
    Xhist, _ = F.build_blocks(df, ["author_history"], np.arange(len(df)), y)
    Xrec, _ = author_recency(df)

    # absolute-views target to expose the fame trap
    v = df["views_at_H"].values.astype(float)
    y_abs = (v >= np.quantile(v, 0.70)).astype(int)

    configs = {
        "best-A (content+creator-fit)": Xbest,
        "+author_history(lifetime)": np.c_[Xbest, Xhist],
        "+author_history+recency(sequence)": np.c_[Xbest, Xhist, Xrec],
        "author-history ONLY": Xhist,
        "author-history+recency ONLY": np.c_[Xhist, Xrec],
    }
    rows = []
    for name, X in configs.items():
        a_rel = loco_auc(df, X, y)            # within-creator (beats own median) -- the right target
        a_abs = loco_auc(df, X, y_abs)        # absolute top-30% views -- the fame-leaky target
        rows.append({"config": name, "auc_beats_own_median": a_rel, "auc_absolute_views": a_abs})
        print(f"  {name:38} relative={a_rel}  absolute={a_abs}", flush=True)
    pd.DataFrame(rows).to_csv(C.REPORTS / "author_history_model.csv", index=False)
    print("saved -> reports/author_history_model.csv")


if __name__ == "__main__":
    main()
