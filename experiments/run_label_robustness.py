#!/usr/bin/env python3
"""P0.2/P0.3 — label robustness + author-history ablation.

Closes A2 (label choice), A4 (horizon sensitivity), A5 (author-history feature).

1. HORIZON sensitivity: rebuild the within-creator breakout label at H in {7,10,14,21},
   run the best Model-A config, report LOCO AUC + label agreement vs H=14. (Also notes
   that a FIXED observation horizon already controls the video-age confounder, so the
   Wu-2016 age-normalization is largely moot for this dataset — we verify stability instead.)
2. LABEL DEFINITION: within-creator-median vs global top-30% raw breakout vs abs-log-views,
   each scored by (a) content AUC (best-A features) and (b) FAME-LEAK AUC (author-size only).
   Confirms the within-creator label is the fame-neutral one.
3. AUTHOR-HISTORY: best-A vs best-A + author_history (both targets), and author_history-alone
   vs each label — shows it carries popularity-bias (predicts abs-views) but adds little to the
   within-creator target (the parry works).

Output: reports/label_robustness.csv (+ experiments_log rows).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, experiment as X, eval as E, splits as S, config as C

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_text():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return f"text_emb:{n}", n
    return "semantic", "svd"


def views_at_H(df, H):
    e = pd.read_parquet(C.RAW / "engagement_daily.parquet",
                        columns=["video_id", "days_since_post", "play_count"])
    e["video_id"] = e["video_id"].astype(str)
    cum = (e[e.days_since_post <= H].sort_values(["video_id", "days_since_post"])
           .groupby("video_id").tail(1).set_index("video_id")["play_count"])
    return cum.reindex(df["video_id"].astype(str).values).values


def loco_auc(df, X_or_blocks, y, model="logreg"):
    """Quick LOCO OOF AUC for a fixed numeric matrix or block list."""
    oof = np.full(len(df), np.nan)
    for tri, tei in S.loco_folds(df):
        if len(np.unique(y[tri])) < 2:
            continue
        if isinstance(X_or_blocks, np.ndarray):
            Xall = X_or_blocks
        else:
            Xall, _ = F.build_blocks(df, X_or_blocks, tri, y)
        from sip.modeling import make_model
        m = make_model(model).fit(np.nan_to_num(Xall[tri]), y[tri])
        oof[tei] = m.predict_proba(np.nan_to_num(Xall[tei]))[:, 1]
    ok = ~np.isnan(oof)
    return E._safe_auc(y[ok], oof[ok])


def main():
    df = D.load(eligible_only=True); F.add_derived(df)
    text, tn = best_text()
    cfit = f"creator_fit:{tn}" if tn != "svd" else "creator_fit"
    bestA = TAB + [text, cfit, "trend_fit"]
    foll = pd.to_numeric(df["followers_at_post"], errors="coerce").fillna(0).values
    rows = []

    # ---- 1. horizon sensitivity ----
    base = df["y_breakout_wc"].astype(int).values  # H=14 reference
    for H in C.H_GRID:
        v = views_at_H(df, H)
        bo = pd.Series(v / np.clip(foll, 1, None), index=df.index)
        _, yH = D._within_creator_binary(bo, df["author_id"])
        yH = yH.astype(int).values
        auc = loco_auc(df, bestA, yH)
        agree = float((yH == base).mean())
        rows.append({"section": "horizon", "setting": f"H={H}", "loco_auc": round(auc, 4),
                     "base_rate": round(float(yH.mean()), 3), "agree_vs_H14": round(agree, 3)})
        print(f"  H={H}: AUC={auc:.4f} base={yH.mean():.3f} agree={agree:.3f}", flush=True)

    # ---- 2. label-definition comparison (H=14) ----
    v14 = df["views_at_H"].values.astype(float)
    raw_bo = v14 / np.clip(foll, 1, None)
    labels = {
        "within_creator_median": base,
        "global_top30_breakout": (raw_bo >= np.quantile(raw_bo, 0.70)).astype(int),
        "abs_logviews_top30": (v14 >= np.quantile(v14, 0.70)).astype(int),
    }
    fame_X = np.c_[np.log1p(foll), F.build_blocks(df, ["author_history"], np.arange(len(df)), base)[0]]
    for name, yl in labels.items():
        content = loco_auc(df, bestA, yl)
        fame = loco_auc(df, fame_X, yl)
        rows.append({"section": "label_def", "setting": name, "loco_auc": round(content, 4),
                     "fame_leak_auc": round(fame, 4), "base_rate": round(float(yl.mean()), 3)})
        print(f"  {name}: content={content:.4f} fame_leak={fame:.4f} base={yl.mean():.3f}", flush=True)

    # ---- 3. author-history ablation ----
    for target in ("y_breakout_wc", "y_er_wc"):
        yt = df[target].astype(int).values
        a = loco_auc(df, bestA, yt)
        ah = loco_auc(df, bestA + ["author_history"], yt)
        only = loco_auc(df, ["author_history"], yt)
        rows.append({"section": "author_hist", "setting": f"{target}", "bestA": round(a, 4),
                     "bestA+authhist": round(ah, 4), "authhist_only": round(only, 4),
                     "lift": round(ah - a, 4)})
        print(f"  {target}: bestA={a:.4f} +authhist={ah:.4f} (lift {ah-a:+.4f}) only={only:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(C.REPORTS / "label_robustness.csv", index=False)
    print("saved -> reports/label_robustness.csv")


if __name__ == "__main__":
    main()
