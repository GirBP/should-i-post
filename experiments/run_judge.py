#!/usr/bin/env python3
"""Assemble LLM-as-judge scores and measure their marginal lift.

1. Collect data/processed/judge_out/*.json -> data/processed/judge_scores.parquet
2. On the judged subset, compare (LOCO + temporal, with CIs):
     TAB  vs  TAB+judge(6)  vs  judge-only  vs  TAB+text(bge)+judge
   and a per-dimension add-one ablation (which judged dimension helps most).
Outputs reports/judge_lift.csv and appends to experiments_log.md.
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, experiment as X, features as F, config as C

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def assemble():
    rows = []
    for f in glob.glob(str(C.PROC / "judge_out" / "*.json")):
        try:
            rows += json.load(open(f))
        except Exception as e:
            print("skip", f, e)
    df = pd.DataFrame(rows).drop_duplicates("video_id")
    keep = ["video_id"] + F.JUDGE_DIMS
    df = df[[c for c in keep if c in df.columns]].copy()
    df.columns = ["video_id"] + [f"judge_{c}" for c in df.columns if c != "video_id"]
    for c in df.columns:
        if c != "video_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["video_id"] = df["video_id"].astype(str)
    df.to_parquet(C.PROC / "judge_scores.parquet", index=False)
    print(f"assembled {len(df)} judged videos -> judge_scores.parquet")
    return df


def main():
    j = assemble()
    F._judge_cache.cache_clear()
    df = D.load(eligible_only=True)
    sub = df[df["video_id"].astype(str).isin(set(j["video_id"]))].reset_index(drop=True)
    F.add_derived(sub)
    print(f"judged subset matched to canonical: {len(sub)} videos, {sub.author_id.nunique()} authors")
    has_bge = (C.PROC / "emb_bge.parquet").exists()
    text = ["text_emb:bge"] if has_bge else ["semantic"]
    rows = []
    for target in ("y_breakout_wc", "y_er_wc"):
        configs = [
            ("TAB", TAB),
            ("judge_only", ["judge"]),
            ("TAB+judge", TAB + ["judge"]),
            ("TAB+text", TAB + text),
            ("TAB+text+judge", TAB + text + ["judge"]),
        ]
        base = None
        for name, blocks in configs:
            r = X.run(sub, blocks, "hgb", target=target, model_kind="A", n_boot=500,
                      label=f"judge:{name}[{target}]", log=True, verbose=True)
            auc = r["loco"]["roc_auc"]
            if name == "TAB":
                base = auc
            rows.append({"target": target, "config": name, "loco_auc": auc,
                         "loco_ci": r["loco"]["roc_auc_ci"], "temporal_auc": r["temporal"]["roc_auc"],
                         "lift_vs_TAB": None if base is None else round(auc - base, 4)})
        # per-dimension add-one
        for d in F.JUDGE_DIMS:
            r = X.run(sub, TAB + [f"judge:{d}"], "hgb", target=target, model_kind="A",
                      n_boot=400, label=f"judge1:{d}[{target}]", log=True, verbose=False)
            rows.append({"target": target, "config": f"TAB+{d}", "loco_auc": r["loco"]["roc_auc"],
                         "loco_ci": r["loco"]["roc_auc_ci"], "temporal_auc": r["temporal"]["roc_auc"],
                         "lift_vs_TAB": round(r["loco"]["roc_auc"] - base, 4)})
    pd.DataFrame(rows).to_csv(C.REPORTS / "judge_lift.csv", index=False)
    print("judge_lift ->", C.REPORTS / "judge_lift.csv")


if __name__ == "__main__":
    main()
