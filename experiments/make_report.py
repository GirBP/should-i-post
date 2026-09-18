#!/usr/bin/env python3
"""Consolidate all experiment outputs into reports/RESULTS_AUTO.md.

Reads whatever exists in reports/ and renders the headline tables (baselines,
staged ablation, text encoders, model family, Optuna, LLM-judge, multimodal,
deployable A->B + decision bands) so the report always reflects real runs.
"""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C

R = C.REPORTS
out = ["# ShouldIPost? — auto-generated results\n",
       "_Generated from `reports/*` by `experiments/make_report.py`. "
       "AUC = ROC-AUC; LOCO = leave-one-creator-out; temporal = train<valid<test by date. "
       "CIs are 95% bootstrap._\n"]


def section(title):
    out.append(f"\n## {title}\n")


def df_md(path, cols=None):
    if not Path(path).exists():
        out.append(f"_(missing {Path(path).name})_\n"); return None
    df = pd.read_csv(path)
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    out.append(df.to_markdown(index=False)); out.append("")
    return df


def jload(path):
    return json.load(open(path)) if Path(path).exists() else None


section("1. Baselines (temporal test)")
b = jload(R / "baselines.json")
if b:
    rows = []
    for k, v in b.items():
        m = v if "roc_auc" in v else (v.get("temporal") or {})
        rows.append({"setting": k, "roc_auc": m.get("roc_auc"),
                     "pr_auc": m.get("pr_auc"), "brier": m.get("brier")})
    out.append(pd.DataFrame(rows).to_markdown(index=False)); out.append("")

section("2. Staged ablation — Model A (marginal lift of each block)")
df_md(R / "ablation_A.csv", ["target", "stage", "model", "loco_auc",
                             "loco_ci_lo", "loco_ci_hi", "temporal_auc", "marginal"])

section("2b. Best Model-A config (TAB + text + creator-fit + trend, no priors; full data)")
df_md(R / "best_A.csv", ["target", "model", "loco_auc", "loco_ci", "temporal_auc", "brier"])

section("3. Text encoders — marginal lift over TAB")
df_md(R / "text_encoders.csv", ["target", "encoder", "loco_auc", "temporal_auc", "lift_over_TAB"])

section("4. Model family (fixed features)")
df_md(R / "model_family.csv", ["target", "model", "loco_auc", "temporal_auc", "brier"])

section("5. Optuna-tuned XGBoost vs default")
o = jload(R / "optuna_best.json")
if o:
    out.append(f"- best LOCO AUC **{o['best_loco_auc']}** vs default {o['default_loco_auc']} "
               f"(n={o['n']}, {o['n_trials']} trials)")
    out.append(f"- best params: `{json.dumps(o['best_params'])}`\n")

section("6. LLM-as-judge — marginal lift")
df_md(R / "judge_lift.csv", ["target", "config", "loco_auc", "temporal_auc", "lift_vs_TAB"])

section("7. Multimodal ablation + fusion")
mm = jload(R / "multimodal_metrics.json")
if mm:
    out.append(f"_n={mm['n']} videos, {mm['authors']} authors, base rate {mm['base_rate']:.3f}_\n")
df_md(R / "multimodal_ablation.csv", ["target", "config", "model", "loco_auc",
                                      "temporal_auc", "marginal", "n_features"])

section("8. Deployable A->B system (calibrated; test set)")
d = jload(R / "deployable_metrics.json")
if d:
    for k in ("A", "B"):
        m = d[k]["metrics"]
        out.append(f"**Model {k}** — AUC {m['roc_auc']} {m['roc_auc_ci']}, PR-AUC {m['pr_auc']}, "
                   f"Brier {m['brier']}, precision@Post {m['precision_at_post']}")
        out.append(f"  bands {d[k]['bands']}  →  {d[k]['band_report']}\n")

section("9. Label robustness — horizon + label definition + author-history (P0.2/P0.3)")
df_md(R / "label_robustness.csv")

section("10. Other-domain methods (P1)")
out.append("**Learning-to-rank vs classifier (per-author Spearman / NDCG@3):**")
df_md(R / "ltr.csv")
out.append("**Trajectory / point-process — when is day-14 breakout determined:**")
df_md(R / "trajectory_pp.csv")
out.append("**Multi-task (shared trunk) vs single-task:**")
df_md(R / "multitask.csv")
out.append("**Deep / foundation tabular vs GBT:**")
df_md(R / "tabular_dl.csv")
out.append("**Causal debiasing (IPS by author size):**")
df_md(R / "ips.csv")

(R / "RESULTS_AUTO.md").write_text("\n".join(out))
print("wrote", R / "RESULTS_AUTO.md")
