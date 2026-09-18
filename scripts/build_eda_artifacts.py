#!/usr/bin/env python3
"""Compute the dataset-journey artifacts consumed by notebooks/RESULTS.ipynb.

Produces, from the real data on disk (no model needed):
  reports/dataset_comparison.csv   the brief's dataset vs. the dataset of record, side by side
  reports/lingbow_eda.json         scale / creators / temporal / content-coverage / fame-skew of lingbow
  reports/plots/eda_*.png          timeline, engagement trajectory, views distribution, creator long-tail

Old set = data/raw/train.csv (datahiveai/Tiktok-Videos, shipped with the brief).
New set = data/raw/lingbow/* + data/processed/canonical.parquet (dataset of record).
Numbers about the old set's deployed AUC (0.52) are quoted from the prior run recorded in the brief.
Regenerate:  python scripts/build_eda_artifacts.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _chartstyle as CS
CS.apply()
FIG_W, H_S = CS.FIG_W, CS.H_S

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
PLOTS = REP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
GREEN, BLUE, RED = CS.GREEN, CS.BLUE, CS.RED


def eta_squared(values, groups):
    """Share of variance in `values` explained by `groups` (fame proxy strength)."""
    s = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    grand = s["v"].mean()
    ss_tot = ((s["v"] - grand) ** 2).sum()
    ss_bet = s.groupby("g")["v"].apply(lambda x: len(x) * (x.mean() - grand) ** 2).sum()
    return float(ss_bet / ss_tot) if ss_tot > 0 else float("nan")


# ----------------------------------------------------------------- OLD (brief's set)
old = {}
old_path = RAW / "train.csv"
if old_path.exists():
    o = pd.read_csv(old_path)
    vc = o["author_unique_id"].value_counts()
    plays = o["play_count"]
    yr = pd.to_datetime(o["create_time"], unit="s", errors="coerce").dt.year
    old = {
        "name": "datahiveai/Tiktok-Videos",
        "rows": int(len(o)),
        "creators": int(o["author_unique_id"].nunique()),
        "videos_per_creator_median": float(vc.median()),
        "top_creator_share": float(vc.iloc[0] / len(o)),
        "top5_play_share": float(plays.sort_values(ascending=False).head(5).sum() / plays.sum()),
        "repost_all_zero": bool((o["repost_count"].fillna(0) == 0).all()),
        "create_year_min": int(yr.min()), "create_year_max": int(yr.max()),
        "garbage_epoch_rows": int((yr < 2011).sum()),
        "has_daily_engagement": False,          # single cumulative snapshot only
        "fame_eta2_logplays": eta_squared(np.log1p(plays), o["author_unique_id"]),
        "deployed_loco_auc": 0.52,              # prior run quoted in the brief (CI [0.45, 0.59])
    }

# ----------------------------------------------------------------- NEW (dataset of record)
can = pd.read_parquet(PROC / "canonical.parquet")
eng = pd.read_parquet(RAW / "lingbow" / "engagement_daily.parquet")
cre = pd.read_parquet(RAW / "lingbow" / "creator_daily.parquet")

vpc = can.groupby("author_id").size()
dt = pd.to_datetime(can["create_dt"])
content_cols = {"transcript": "transcript", "gpt_summary": "gpt_summary", "topic": "topic"}
coverage = {k: float(can[c].astype(str).str.strip().replace("nan", "").ne("").mean())
            for k, c in content_cols.items() if c in can.columns}
coverage["hashtags"] = float(can["hashtag_count"].gt(0).mean()) if "hashtag_count" in can.columns else float("nan")

# engagement trajectory by days_since_post (0..30). The MEAN is inflated by a few late-accumulating
# viral outliers, so we also keep the MEDIAN (the typical video) — it is the honest front-loading curve.
sub = eng[eng["days_since_post"].between(0, 30)].groupby("days_since_post")["play_count"]
traj, traj_med = sub.mean(), sub.median()
day1_frac = float(traj.get(1, np.nan) / traj.get(14, np.nan)) if 14 in traj.index else float("nan")
day1_frac_med = float(traj_med.get(1, np.nan) / traj_med.get(14, np.nan)) if 14 in traj_med.index else float("nan")

new = {
    "name": "lingbow/tiktok-video-engagement-200k",
    "rows": int(len(can)),
    "creators": int(can["author_id"].nunique()),
    "videos_per_creator_median": float(vpc.median()),
    "videos_per_creator_p90": float(vpc.quantile(0.90)),
    "date_min": str(dt.min().date()), "date_max": str(dt.max().date()),
    "has_daily_engagement": True, "engagement_days": int(eng["days_since_post"].max()),
    "creator_daily_rows": int(len(cre)),
    "content_coverage": coverage,
    "fame_eta2_logviews": eta_squared(np.log1p(can["views_at_H"]), can["author_id"]),
    "followers_median": float(can["followers_at_post"].median()),
    "followers_p99": float(can["followers_at_post"].quantile(0.99)),
    "day1_over_day14_views_mean": day1_frac,
    "day1_over_day14_views_median": day1_frac_med,
    "eligible": int(can["elig"].sum()) if "elig" in can.columns else int(len(can)),
}
json.dump({"old": old, "new": new}, open(REP / "lingbow_eda.json", "w"), indent=2)

# ----------------------------------------------------------------- comparison table
def fmt(x):
    # space (not comma) thousands separator so the value never collides with CSV delimiters
    return f"{x:,}".replace(",", " ") if isinstance(x, int) else (f"{x:.2f}" if isinstance(x, float) else str(x))

rows = [
    ("Videos", fmt(old.get("rows", "—")), fmt(new["rows"])),
    ("Creators", fmt(old.get("creators", "—")), fmt(new["creators"])),
    ("Median videos / creator", fmt(round(old.get("videos_per_creator_median", float("nan")), 1)),
     fmt(round(new["videos_per_creator_median"], 1))),
    ("Top-creator share of rows", f"{old.get('top_creator_share', float('nan')):.0%}", "<1%"),
    ("Day-by-day engagement (0–30)", "no (one snapshot)", f"yes ({new['engagement_days']} days)"),
    ("Creator follower state over time", "no", f"yes ({new['creator_daily_rows']:,} rows)"),
    ("Pre-publication content fields", "caption / duration", "caption / transcript / gpt-summary / topic / emotions / music / hashtags"),
    ("Horizon-H within-creator label feasible", "no", "yes (H∈{7,10,14,21})"),
    ("Day-1 Model B feasible", "no", "yes"),
    ("LOCO cross-creator eval feasible", "no (4 groups)", f"yes ({new['creators']:,} groups)"),
    ("Creator-identity share of log-views (eta²)", f"{old.get('fame_eta2_logplays', float('nan')):.2f}",
     f"{new['fame_eta2_logviews']:.2f}"),
    ("Deployed pre-pub ROC-AUC (prior run)", f"{old.get('deployed_loco_auc', float('nan')):.2f} (≈random)", "see §5 (≈0.57)"),
]
pd.DataFrame(rows, columns=["dimension", "datahiveai/Tiktok-Videos (brief)",
                            "lingbow/…-200k (chosen)"]).to_csv(REP / "dataset_comparison.csv", index=False)

# ----------------------------------------------------------------- plots
# 1. videos over time (monthly)
fig, ax = plt.subplots(figsize=(FIG_W, H_S))
dt.dt.to_period("M").value_counts().sort_index().rename(lambda p: str(p)).plot.bar(ax=ax, color=BLUE)
ax.set_title("lingbow: videos posted per month"); ax.set_xlabel(""); ax.set_ylabel("videos")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(PLOTS / "eda_timeline.png"); plt.close()

# 2. engagement trajectory (front-loading -> why a day-1 model is so much stronger than pre-pub)
fig, ax = plt.subplots(figsize=(FIG_W, H_S))
(traj_med / traj_med.max()).plot(ax=ax, color=GREEN, marker="o", ms=3, label="median video")
(traj / traj.max()).plot(ax=ax, color=BLUE, marker="o", ms=3, ls="--", label="mean (outlier-inflated)")
ax.axvline(1, ls="--", c="gray", lw=1); ax.text(1.4, 0.15, "day 1", color="gray")
ax.set_title("Cumulative views by age (normalised) — reach is front-loaded")
ax.set_xlabel("days since post"); ax.set_ylabel("share of day-30 views"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PLOTS / "eda_engagement_traj.png"); plt.close()

# 3. views distribution (log) — the fame skew that forces a within-creator label
fig, ax = plt.subplots(figsize=(FIG_W, H_S))
np.log10(can["views_at_H"].clip(lower=1)).plot.hist(bins=60, ax=ax, color=BLUE, alpha=0.85)
ax.set_title("Views @ H=14 (log10) — heavy-tailed, dominated by creator fame")
ax.set_xlabel("log10 views"); ax.set_ylabel("videos"); plt.tight_layout()
plt.savefig(PLOTS / "eda_views_dist.png"); plt.close()

# 4. creator long-tail (videos per creator) vs the old set's 4 creators
fig, ax = plt.subplots(figsize=(FIG_W, H_S))
vpc.clip(upper=vpc.quantile(0.99)).plot.hist(bins=50, ax=ax, color=GREEN, alpha=0.85)
ax.set_title(f"lingbow: videos per creator across {new['creators']:,} creators (old set had 4)")
ax.set_xlabel("videos per creator (capped at p99)"); ax.set_ylabel("creators"); plt.tight_layout()
plt.savefig(PLOTS / "eda_creator_videos.png"); plt.close()

print("wrote reports/dataset_comparison.csv, reports/lingbow_eda.json, reports/plots/eda_*.png")
print(json.dumps({"old": old, "new": new}, indent=2, default=str))
