#!/usr/bin/env python3
"""Analytical artifacts consumed by notebooks/RESULTS.ipynb (computed from the real data + model):

  reports/feature_correlation.csv / plots/feature_correlation.png
      Spearman correlation among the interpretable pre-publication features (redundancy structure).
  reports/feature_influence.csv / plots/feature_influence.png
      Univariate signed effect of each pre-publication feature on within-creator breakout
      (single-feature ROC-AUC minus 0.5, with the sign of the Spearman correlation).
  plots/decision_bands.png
      Why the Post / Unsure / Do-not-post thresholds sit where they do: precision and coverage as a
      function of the threshold on the held-out probabilities, plus the cost-ratio -> threshold curve.

Regenerate:  PYTHONPATH=src python scripts/build_analysis_artifacts.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for _chartstyle
import _chartstyle as CS
CS.apply()
FIG_W, H_S, H_M, H_SQ = CS.FIG_W, CS.H_S, CS.H_M, CS.H_SQ

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
PLOTS = REP / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
GREEN, BLUE, RED, GREY = CS.GREEN, CS.BLUE, CS.RED, CS.GREY

# ----------------------------------------------------------------- interpretable feature frame
can = pd.read_parquet(PROC / "canonical.parquet")
if "elig" in can.columns:
    can = can[can["elig"] == 1]
can = can[can["y_breakout_wc"].notna()].copy()
y = can["y_breakout_wc"].astype(int).values

dt = pd.to_datetime(can["create_dt"])
music_freq = can["music_id"].map(can["music_id"].value_counts())
F = pd.DataFrame({
    "duration_s": pd.to_numeric(can["duration"], errors="coerce"),
    "caption_len": can["desc"].fillna("").astype(str).str.len(),
    "word_count": pd.to_numeric(can.get("word_count"), errors="coerce"),
    "hashtag_count": pd.to_numeric(can.get("hashtag_count"), errors="coerce"),
    "emoji_count": pd.to_numeric(can.get("emoji_count"), errors="coerce"),
    "question_count": pd.to_numeric(can.get("question_count"), errors="coerce"),
    "speaking_rate": pd.to_numeric(can.get("speaking_rate"), errors="coerce"),
    "joy": pd.to_numeric(can.get("joy"), errors="coerce"),
    "anger": pd.to_numeric(can.get("anger"), errors="coerce"),
    "surprise": pd.to_numeric(can.get("surprise"), errors="coerce"),
    "sadness": pd.to_numeric(can.get("sadness"), errors="coerce"),
    "fear": pd.to_numeric(can.get("fear"), errors="coerce"),
    "arousal": pd.to_numeric(can.get("anger"), errors="coerce").fillna(0)
               + pd.to_numeric(can.get("surprise"), errors="coerce").fillna(0)
               + pd.to_numeric(can.get("fear"), errors="coerce").fillna(0),
    "is_weekend": (dt.dt.dayofweek >= 5).astype(float),
    "is_english": can.get("is_english", pd.Series(index=can.index)).astype("float"),
    "created_by_ai": can.get("created_by_ai", pd.Series(index=can.index)).astype("float"),
    "is_ads": can.get("is_ads", pd.Series(index=can.index)).astype("float"),
    "music_popularity_log": np.log1p(music_freq.astype(float)),
    "creator_followers_log": np.log1p(pd.to_numeric(can["followers_at_post"], errors="coerce")),
}).apply(lambda c: c.fillna(c.median()))

# ----------------------------------------------------------------- 1. correlation structure
corr = F.corr(method="spearman")
corr.round(3).to_csv(REP / "feature_correlation.csv")
fig, ax = plt.subplots(figsize=(FIG_W, H_SQ))
ax.grid(False)
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
ax.set_title("Spearman correlation among pre-publication features")
fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout()
plt.savefig(PLOTS / "feature_correlation.png"); plt.close()

# ----------------------------------------------------------------- 2. univariate influence on success
rows = []
for col in F.columns:
    x = F[col].values.astype(float)
    if np.nanstd(x) == 0:
        continue
    auc = roc_auc_score(y, x)                       # single-feature ranking AUC
    sp = pd.Series(x).corr(pd.Series(y), method="spearman")
    rows.append({"feature": col, "univariate_auc": round(float(auc), 4),
                 "signed_lift": round(float(auc - 0.5), 4),
                 "spearman_vs_success": round(float(sp), 4),
                 "direction": "↑ helps" if sp > 0 else "↓ hurts"})
infl = pd.DataFrame(rows).sort_values("signed_lift", key=lambda s: s.abs(), ascending=False)
infl.to_csv(REP / "feature_influence.csv", index=False)

fig, ax = plt.subplots(figsize=(FIG_W, H_M))
d = infl.sort_values("signed_lift")
colors = [GREEN if v > 0 else RED for v in d["signed_lift"]]
ax.barh(d["feature"], d["signed_lift"], color=colors)
ax.axvline(0, c="gray", lw=1)
ax.set_xlabel("single-feature ROC-AUC − 0.5  (green = raises success, red = lowers)")
ax.set_title("Univariate influence of each pre-publication feature on within-creator breakout")
plt.tight_layout(); plt.savefig(PLOTS / "feature_influence.png"); plt.close()

# ----------------------------------------------------------------- 3. decision-band rationale
m = __import__("joblib").load(ROOT / "models" / "deployable.joblib")
fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 2 * H_S))           # stacked -> uniform width
for ax, k, color in [(axes[0], "A", BLUE), (axes[1], "B", GREEN)]:
    p = np.asarray(m[k]["val_probs"], float); yv = np.asarray(m[k]["val_y"], int)
    bands = m[k]["bands"]
    ts = np.linspace(p.min() + 1e-3, p.max() - 1e-3, 60)
    prec = [yv[p >= t].mean() if (p >= t).sum() > 30 else np.nan for t in ts]   # precision@Post
    cov = [(p >= t).mean() for t in ts]                                          # fraction posted
    ax.plot(ts, prec, color=color, label="precision@Post = P(breakout | p≥t)")
    ax.plot(ts, cov, color=GREY, ls="--", label="coverage = share posted")
    ax.axhline(0.65, c=RED, ls=":", lw=1, label="precision target 0.65")
    ax.axvline(bands["t_high"], c=color, lw=1)
    ax.text(bands["t_high"], 0.05, f" t_high={bands['t_high']:.3f}", color=color)
    ax.axvline(bands["t_low"], c=GREY, lw=1)
    ax.set_title(f"Model {k}: precision and coverage vs threshold"); ax.set_xlabel("threshold t")
    ax.set_ylim(0, 1); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(PLOTS / "decision_bands.png"); plt.close()

# cost-ratio -> Elkan threshold curve
fig, ax = plt.subplots(figsize=(FIG_W, H_S))
r = np.linspace(0.2, 5, 100); ax.plot(r, r / (1 + r), color=BLUE)
ax.axvline(1, c=GREY, ls="--"); ax.axhline(0.5, c=GREY, ls="--")
ax.scatter([1], [0.5], color=RED, zorder=5); ax.text(1.05, 0.42, "symmetric cost → 0.50")
ax.set_xlabel("cost ratio r = cost(false Post) / cost(missed hit)")
ax.set_ylabel("Elkan-optimal Post threshold  r/(1+r)")
ax.set_title("Cost-sensitive threshold: how selectivity maps to the Post cutoff")
plt.tight_layout(); plt.savefig(PLOTS / "decision_threshold_cost.png"); plt.close()

# ----------------------------------------------------------------- 4. reliability curves (unified style)
def reliability(p, yv, k, color):
    edges = np.quantile(p, np.linspace(0, 1, 11))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        msk = idx == b
        if msk.sum() > 0:
            xs.append(p[msk].mean()); ys.append(yv[msk].mean())
    fig, ax = plt.subplots(figsize=(FIG_W, H_S))
    ax.plot([0, 1], [0, 1], ls="--", c=GREY, label="perfect calibration")
    ax.plot(xs, ys, marker="o", color=color, label=f"Model {k}")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed breakout rate")
    ax.set_title(f"Model {k}: reliability (calibration) on the test split"); ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(PLOTS / f"reliability_{k}.png"); plt.close()

for k, color in [("A", BLUE), ("B", GREEN)]:
    reliability(np.asarray(m[k]["val_probs"], float), np.asarray(m[k]["val_y"], int), k, color)

# ----------------------------------------------------------------- 5. SHAP (unified width), best-effort
# Human-readable labels: raw names like "bge522" (an embedding dimension) or "dow_sin" are correct but
# opaque on a plot, so relabel them for the reader.
PRETTY = {"char_len": "caption length", "word_len": "word count", "n_hashtags": "hashtag count",
          "n_mentions": "mention count", "has_question": "question hook", "has_exclam": "exclamation",
          "n_emoji": "emoji count", "has_url": "off-platform URL", "has_cta": "call-to-action",
          "digit_ratio": "digit ratio", "caption_is_empty": "empty caption", "allcaps_ratio": "ALL-CAPS ratio",
          "duration_s": "duration (s)", "dur_vshort": "very short", "dur_short": "short",
          "dur_mid": "mid-length", "dur_long": "long", "dur_vlong": "very long",
          "hour_sin": "posting hour", "hour_cos": "posting hour", "dow_sin": "day of week",
          "dow_cos": "day of week", "is_weekend": "weekend", "month_sin": "month", "month_cos": "month",
          "is_english_i": "English caption", "created_by_ai_i": "AI-generated", "is_ads_i": "marked as ad",
          "aspect": "aspect ratio", "log_play_d1": "day-1 plays", "log_like_d1": "day-1 likes",
          "er_d1": "day-1 engagement rate"}
def _lab(n):
    return ("text-emb " + n[3:]) if n.startswith("bge") else PRETTY.get(n, n)

try:
    import shap
    bg = np.asarray(m["A"]["shap_bg"], float)
    sv = shap.TreeExplainer(m["A"]["model"])(bg)
    sv.feature_names = [_lab(n) for n in m["A"]["names"]]
    plt.figure(figsize=(FIG_W, H_M))
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("Model A: SHAP feature impact (background sample)")
    plt.tight_layout(); plt.savefig(PLOTS / "shap_beeswarm.png"); plt.close()
    plt.figure(figsize=(FIG_W, H_M))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title("Model A: SHAP explanation for one example")
    plt.tight_layout(); plt.savefig(PLOTS / "shap_waterfall.png"); plt.close()
    # SHAP controls its own canvas; normalise the two PNGs to the shared pixel width
    from PIL import Image
    target_w = int(FIG_W * CS.DPI)
    for nm in ("shap_beeswarm.png", "shap_waterfall.png"):
        im = Image.open(PLOTS / nm)
        h = round(im.height * target_w / im.width)
        im.resize((target_w, h), Image.LANCZOS).save(PLOTS / nm, dpi=(CS.DPI, CS.DPI))
    print("regenerated SHAP plots in unified style")
except Exception as e:
    print(f"SHAP regeneration skipped ({type(e).__name__}: {e}); keeping existing shap_*.png")

# ----------------------------------------------------------------- 6. importance by GROUP, per model
CAP = {"char_len", "word_len", "n_hashtags", "n_mentions", "has_question", "has_exclam", "n_emoji",
       "has_url", "has_cta", "digit_ratio", "caption_is_empty", "allcaps_ratio"}
TIM = {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month_sin", "month_cos"}
DUR = {"duration_s", "dur_vshort", "dur_short", "dur_mid", "dur_long", "dur_vlong"}
MET = {"is_english_i", "created_by_ai_i", "is_ads_i", "aspect"}
DAY1 = {"log_play_d1", "log_like_d1", "er_d1", "log_play_d1_vs_author"}
GORDER = ["text embedding (BGE)", "day-1 engagement", "caption text stats", "duration",
          "posting time", "meta flags"]


def _grp(n):
    if n.startswith("bge"):
        return "text embedding (BGE)"
    if n in DAY1:
        return "day-1 engagement"
    if n in CAP:
        return "caption text stats"
    if n in TIM:
        return "posting time"
    if n in DUR:
        return "duration"
    if n in MET:
        return "meta flags"
    return "other"


inventory = {}
try:
    import shap
    for k, color in [("A", BLUE), ("B", GREEN)]:
        names = list(m[k]["names"]); bg = np.asarray(m[k]["shap_bg"], float)
        sv = np.abs(np.asarray(shap.TreeExplainer(m[k]["model"])(bg).values)).mean(0)  # mean |SHAP| / feature
        g = {}
        for nm, v in zip(names, sv):
            g[_grp(nm)] = g.get(_grp(nm), 0.0) + float(v)
        order = [x for x in GORDER if x in g]
        fig, ax = plt.subplots(figsize=(FIG_W, H_S))
        ax.barh(order[::-1], [g[x] for x in order[::-1]], color=color)
        ax.set_xlabel("summed mean |SHAP| — contribution to the score")
        ax.set_title(f"Model {k}: feature-group importance")
        plt.tight_layout(); plt.savefig(PLOTS / f"shap_group_{k}.png"); plt.close()
        cnt = {}
        for nm in names:
            cnt[_grp(nm)] = cnt.get(_grp(nm), 0) + 1
        inventory[k] = {"total": len(names), "by_group": cnt}
    print("wrote grouped importance shap_group_A/B.png")
except Exception as e:
    print(f"grouped importance skipped ({type(e).__name__}: {e})")
inventory["tabular_names"] = {"caption": sorted(CAP), "posting time": sorted(TIM),
                             "duration": sorted(DUR), "meta": sorted(MET),
                             "day-1 (Model B only)": ["log_play_d1", "log_like_d1", "er_d1"]}
json.dump(inventory, open(REP / "feature_inventory.json", "w"), indent=1)

# ----------------------------------------------------------------- summary json
summary = {
    "n_eligible": int(len(can)),
    "n_features_examined": int(F.shape[1]),
    "top_positive": infl[infl.signed_lift > 0].head(5)[["feature", "signed_lift"]].values.tolist(),
    "top_negative": infl[infl.signed_lift < 0].head(5)[["feature", "signed_lift"]].values.tolist(),
    "max_abs_offdiag_corr": round(float(corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max()), 3),
}
json.dump(summary, open(REP / "feature_analysis_summary.json", "w"), indent=2)
print("wrote feature_correlation.csv, feature_influence.csv, feature_analysis_summary.json, plots/*")
print(json.dumps(summary, indent=2))
