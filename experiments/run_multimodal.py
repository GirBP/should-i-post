#!/usr/bin/env python3
"""Multimodal ablation + fusion (Model A and B) on the extracted video subset.

Requires data/multimodal/features.parquet (multimodal/extract.py) and
data/processed/emb_bge.parquet (scripts/encode_text.py).

Produces reports/multimodal_ablation.csv + reports/multimodal_metrics.json:
  * staged ablation  TAB -> +TEXT -> +AUDIO -> +VIDEO/HOOK -> +creator_fit -> +trend
  * encoder comparisons  CLIP vs SigLIP ; hook-only vs full ; CLAP vs low-level audio
  * fusion  early (concat+HGB) vs late (mean of per-modality logreg) vs neural (MLP)
  * Model B (+day-1)
All on temporal + LOCO with bootstrap CIs, vs §1 prior numbers.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, experiment as X, features as F, eval as E, splits as S, config as C

TAB = ["caption", "emotion", "duration", "timing", "meta"]
TEXT = ["text_emb:bge"]
AUDIO = ["mm:clap", "mm:aud"]
VIDEO = ["mm:vclip", "mm:hclip", "mm:vll"]      # CLIP frames + hook + low-level visual
SIG = ["mm:vsig"]


def subset():
    f = pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])
    have = set(f["video_id"].astype(str))
    df = D.load(eligible_only=True)
    sub = df[df["video_id"].astype(str).isin(have)].reset_index(drop=True)
    F.add_derived(sub)
    return sub


def run_cfg(sub, blocks, name, target, model="hgb", kind="A", nb=500):
    r = X.run(sub, blocks, model, target=target, model_kind=kind, n_boot=nb,
              label=f"mm:{name}[{target}]", log=True, verbose=True)
    return {"target": target, "config": name, "model": model,
            "loco_auc": r["loco"]["roc_auc"], "loco_ci": r["loco"]["roc_auc_ci"],
            "temporal_auc": r["temporal"]["roc_auc"], "brier": r["temporal"]["brier"],
            "n_features": r["n_features"]}, r


def late_fusion(sub, modality_blocklists, target):
    """Mean of per-modality logreg OOF probabilities."""
    y = sub[target].astype(int).values
    oofs = []
    for blocks in modality_blocklists:
        r = X.run(sub, blocks, "logreg", target=target, model_kind="A",
                  do_temporal=False, n_boot=1, log=False, verbose=False)
        oofs.append(r["_oof"])
    P = np.nanmean(np.vstack(oofs), axis=0)
    ok = ~np.isnan(P)
    return E.metrics(y[ok], P[ok], n_boot=500)


def main():
    if not (C.MM / "features.parquet").exists():
        print("no multimodal features yet — run multimodal/extract.py"); return
    sub = subset()
    print(f"multimodal subset: {len(sub)} videos, {sub.author_id.nunique()} authors")
    print("label balance:", sub["y_breakout_wc"].mean().round(3),
          "| split:", sub["split_temporal"].value_counts().to_dict())
    if len(sub) < 300:
        print("WARN: subset very small; CIs will be wide.")
    cfit = "creator_fit:bge" if (C.PROC / "emb_bge.parquet").exists() else "creator_fit"
    rows = []; detail = {}

    for target in ("y_breakout_wc", "y_er_wc"):
        # staged ablation
        stages = [
            ("TAB", TAB), ("+TEXT", TAB + TEXT), ("+AUDIO", TAB + TEXT + AUDIO),
            ("+VIDEO_HOOK", TAB + TEXT + AUDIO + VIDEO),
            ("+SigLIP", TAB + TEXT + AUDIO + VIDEO + SIG),
            ("+creator_fit", TAB + TEXT + AUDIO + VIDEO + SIG + [cfit]),
            ("+trend", TAB + TEXT + AUDIO + VIDEO + SIG + [cfit, "trend_fit"]),
        ]
        prev = None
        for name, blocks in stages:
            row, r = run_cfg(sub, blocks, name, target)
            row["marginal"] = None if prev is None else round(row["loco_auc"] - prev, 4)
            prev = row["loco_auc"]; rows.append(row)
        # encoder comparisons (single-modality, over TAB)
        for name, blocks in [("video=CLIP", TAB + ["mm:vclip"]), ("video=SigLIP", TAB + ["mm:vsig"]),
                             ("video=hookCLIP", TAB + ["mm:hclip"]),
                             ("audio=CLAP", TAB + ["mm:clap"]), ("audio=lowlevel", TAB + ["mm:aud"]),
                             ("visual_lowlevel", TAB + ["mm:vll"])]:
            row, _ = run_cfg(sub, blocks, name, target); rows.append(row)
        # fusion
        mods = [TAB, TEXT, AUDIO, VIDEO]
        lf = late_fusion(sub, mods, target)
        rows.append({"target": target, "config": "fusion=late(mean-logreg)", "model": "late",
                     "loco_auc": lf["roc_auc"], "loco_ci": lf["roc_auc_ci"],
                     "temporal_auc": None, "brier": None, "n_features": None})
        row, _ = run_cfg(sub, TAB + TEXT + AUDIO + VIDEO + SIG, "fusion=early(HGB)", target, model="hgb")
        rows.append(row)
        row, _ = run_cfg(sub, TAB + TEXT + AUDIO + VIDEO + SIG, "fusion=neural(MLP)", target, model="mlp")
        rows.append(row)
        # Model B
        row, _ = run_cfg(sub, TAB + TEXT + AUDIO + VIDEO + SIG + ["day1"], "B_full(+day1)",
                         target, model="hgb", kind="B")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(C.REPORTS / "multimodal_ablation.csv", index=False)
    json.dump({"n": int(len(sub)), "authors": int(sub.author_id.nunique()),
               "base_rate": float(sub["y_breakout_wc"].mean()),
               "rows": rows}, open(C.REPORTS / "multimodal_metrics.json", "w"), indent=2, default=str)
    print("multimodal_ablation ->", C.REPORTS / "multimodal_ablation.csv")


if __name__ == "__main__":
    main()
