"""P2.1/P2.2 — segment error-analysis, calibration-by-segment, reliability plots,
conformal coverage check (assignment §8 requirement).

Uses the deployable A→B artifact. On the temporal test split:
  * per-segment AUC + Brier + ECE + base rate, by topic / creator-size bucket / duration bucket
  * reliability-curve PNGs for Model A and Model B (calibrated)
  * split-conformal coverage: does the 1-alpha prediction set actually cover on test?

Outputs: reports/error_analysis.md, reports/error_analysis.csv, reports/plots/reliability_{A,B}.png
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
from sip import data as D, features as F, splits as S, eval as E, config as C
from sklearn.metrics import roc_auc_score, brier_score_loss


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def seg_metrics(y, p, mask):
    y, p = y[mask], p[mask]
    if len(y) < 50 or len(np.unique(y)) < 2:
        return None
    return {"n": int(len(y)), "base_rate": round(float(y.mean()), 3),
            "auc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(brier_score_loss(y, p)), 4),
            "ece": round(ece(y, p), 4)}


def main():
    art = joblib.load(C.MODELS / "deployable.joblib")
    df = D.load(eligible_only=True); F.add_derived(df)
    tr, va, te = S.temporal_masks(df)
    tri = np.where(tr)[0]
    y = df["y_breakout_wc"].astype(int).values
    rows = []
    preds = {}
    for kind in ("A", "B"):
        m = art[kind]
        X, _ = F.build_blocks(df, m["blocks"], tri, y)
        p = m["cal"].predict_proba_pos(X[te])
        preds[kind] = p
        yt = y[te]
        # overall
        rows.append({"model": kind, "segment": "ALL", **seg_metrics(yt, p, np.ones(len(yt), bool))})
        # by topic (top categories)
        topic = df["topic"].astype(str).values[te]
        for t in pd.Series(topic).value_counts().head(8).index:
            s = seg_metrics(yt, p, topic == t)
            if s: rows.append({"model": kind, "segment": f"topic={t[:18]}", **s})
        # by creator-size bucket
        foll = pd.to_numeric(df["followers_at_post"], errors="coerce").fillna(1).values[te]
        cb = np.asarray(pd.qcut(np.log1p(foll), 4, labels=["small", "mid", "large", "xlarge"],
                                duplicates="drop"))
        for b in ["small", "mid", "large", "xlarge"]:
            s = seg_metrics(yt, p, cb == b)
            if s: rows.append({"model": kind, "segment": f"creator={b}", **s})
        # by duration bucket
        dur = pd.to_numeric(df["duration_s"], errors="coerce").fillna(0).values[te]
        for nm, lo, hi in [("<7s", 0, 7), ("7-15s", 7, 15), ("15-30s", 15, 30), ("30-60s", 30, 60), (">60s", 60, 1e9)]:
            s = seg_metrics(yt, p, (dur >= lo) & (dur < hi))
            if s: rows.append({"model": kind, "segment": f"dur={nm}", **s})

    pd.DataFrame(rows).to_csv(C.REPORTS / "error_analysis.csv", index=False)

    # reliability plots
    (C.REPORTS / "plots").mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for kind in ("A", "B"):
        cen, obs, w = E.reliability_curve(y[te], preds[kind], bins=10)
        plt.figure(figsize=(4, 4))
        plt.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        plt.plot(cen, obs, "o-", color="#1d9e75")
        plt.xlabel("predicted P"); plt.ylabel("observed frequency")
        plt.title(f"Model {kind} reliability (test)"); plt.tight_layout()
        plt.savefig(C.REPORTS / "plots" / f"reliability_{kind}.png", dpi=110); plt.close()

    # conformal coverage on test
    cov = {}
    for kind in ("A", "B"):
        conf = art[kind]["conformal"]
        s = conf.predict_set(preds[kind])      # 1={0},2={1},3={0,1},0={}
        yt = y[te]
        in_set = ((s == 1) & (yt == 0)) | ((s == 2) & (yt == 1)) | (s == 3)
        cov[kind] = {"target_coverage": round(1 - conf.alpha, 3),
                     "empirical_coverage": round(float(in_set.mean()), 3),
                     "abstain_rate(setsize!=1)": round(float(((s == 3) | (s == 0)).mean()), 3)}

    md = ["# Error analysis & calibration by segment (temporal test)\n",
          "## Conformal coverage check", "```", json.dumps(cov, indent=2), "```\n",
          "## Per-segment metrics", pd.DataFrame(rows).to_markdown(index=False),
          "\nReliability plots: `reports/plots/reliability_A.png`, `reliability_B.png`."]
    (C.REPORTS / "error_analysis.md").write_text("\n".join(md))
    print("conformal coverage:", cov)
    print("saved -> reports/error_analysis.{md,csv} + plots/")


if __name__ == "__main__":
    main()
