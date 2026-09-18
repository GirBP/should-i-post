"""Fast, class-balanced end-to-end classification pipeline on REAL TikTok videos.

Honed on 100 videos; scales to N by `--n` (same code path for 100 or 10,000).
Full chain, nothing faked:
  real video analysis (SigLIP frames + CLAP audio) + text embedding + tabular
    -> class-balanced sample (50/50)
    -> StratifiedGroupKFold by author (out-of-fold: every video scored by a model
       that never saw it, and never saw that creator — the honest 'unseen video' setup)
    -> calibrated probability + cost-based Post / Do-not-post / Unsure verdict
    -> metrics (ROC-AUC + CI, precision@Post, confusion) + per-video assessment JSON.

Run:
    PYTHONPATH=src python experiments/fast_pipeline.py            # 100 videos
    PYTHONPATH=src python experiments/fast_pipeline.py --n 200    # up to the balanced max
    PYTHONPATH=src python experiments/fast_pipeline.py --n 10000  # scales unchanged (needs that many with features)
"""
import os, sys, json, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("SIP_DEVICE", "cpu")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, config as C, eval as E
from sip.modeling import make_model

TARGET = "y_breakout_wc"
BLOCKS = ["caption", "duration", "timing", "meta", "missing", "text_emb:minilm", "mm:vsig", "mm:clap"]
SEED = 0


def balanced_sample(df, n):
    """Take n/2 positive + n/2 negative (seeded, deterministic)."""
    rng = np.random.RandomState(SEED)
    per = n // 2
    out = []
    for cls in (1, 0):
        pool = df[df[TARGET] == cls]
        take = min(per, len(pool))
        out.append(pool.sample(take, random_state=rng))
    sub = pd.concat(out).sample(frac=1.0, random_state=rng).reset_index(drop=True)  # shuffle
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    # 1. real TikTok videos that have BOTH a downloaded video (SigLIP/CLAP features) and a label
    df = D.load(eligible_only=True); df["video_id"] = df["video_id"].astype(str)
    mm_ids = set(pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])["video_id"].astype(str))
    have = df[df["video_id"].isin(mm_ids)].copy()
    pos, neg = int((have[TARGET] == 1).sum()), int((have[TARGET] == 0).sum())
    max_bal = 2 * min(pos, neg)
    n = min(args.n, max_bal)
    if n < args.n:
        print(f"[note] only {max_bal} class-balanced videos available (pos={pos}, neg={neg}); using {n}.")
    sub = balanced_sample(have, n)
    F.add_derived(sub)
    y = sub[TARGET].astype(int).values
    groups = sub["author_id"].values
    print(f"videos: {len(sub)}  | balance: {dict(pd.Series(y).value_counts())}  | "
          f"creators: {pd.Series(groups).nunique()}  | features blocks: {BLOCKS}", flush=True)

    # 2. out-of-fold predictions — no author in both train and test (honest 'unseen creator')
    n_groups = pd.Series(groups).nunique()
    if n_groups >= args.folds:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        split = splitter.split(sub, y, groups)
        proto = "StratifiedGroupKFold by author (leakage-safe)"
    else:
        split = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED).split(sub, y)
        proto = "StratifiedKFold (too few creators to group)"
    oof = np.full(len(sub), np.nan)
    for tri, tei in split:
        Xall, names = F.build_blocks(sub, BLOCKS, tri, y, target_col=TARGET)  # encoders fit on train fold only
        m = make_model(args.model); m.fit(Xall[tri], y[tri])
        oof[tei] = m.predict_proba(Xall[tei])[:, 1]
    print(f"protocol: {proto}  | {len(names)} features", flush=True)

    # 3. metrics + cost-based decision bands (Post precision target 0.60)
    met = E.metrics(y, oof, n_boot=1000, post_threshold=0.5)
    bands = E.find_decision_bands(y, oof, precision_target=0.60, min_coverage=0.05)
    rep = E.band_report(y, oof, bands)
    verdicts = E.apply_bands(oof, bands)  # array of 'Post' / 'Do not post' / 'Unsure'

    # 4. per-video assessment (real video -> real verdict vs real outcome)
    assess = []
    for i in range(len(sub)):
        assess.append({
            "video_id": sub["video_id"].iloc[i],
            "author_id": str(sub["author_id"].iloc[i]),
            "caption": (str(sub.get("description", pd.Series([""] * len(sub))).iloc[i]) or "")[:120],
            "duration_s": float(sub["duration"].iloc[i]) if "duration" in sub else None,
            "prob_success": round(float(oof[i]), 4),
            "verdict": verdicts[i],
            "true_breakout": int(y[i]),
            "correct": bool((oof[i] >= bands["t_high"]) == bool(y[i])) if verdicts[i] != "Unsure" else None,
        })
    out = {
        "n": len(sub), "balance": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
        "protocol": proto, "n_features": len(names), "model": args.model,
        "bands": {k: round(float(v), 4) for k, v in bands.items() if isinstance(v, (int, float))},
        "metrics": {"roc_auc": met["roc_auc"], "roc_auc_ci": met["roc_auc_ci"],
                    "pr_auc": met.get("pr_auc"), "brier": met.get("brier")},
        "band_report": rep, "assessments": assess,
    }
    path = C.REPORTS / f"fast_pipeline_{len(sub)}.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=2)

    # 5. console summary
    print(f"\n=== RESULT (n={len(sub)}, real TikTok, {proto.split(' ')[0]} OOF) ===")
    print(f"  ROC-AUC        : {met['roc_auc']}  CI {met['roc_auc_ci']}")
    print(f"  PR-AUC / Brier : {met.get('pr_auc')} / {met.get('brier')}")
    for band in ("Post", "Unsure", "Do not post"):
        r = rep.get(band, {})
        print(f"  {band:12}: {int(r.get('n',0)):3d} videos ({r.get('coverage',0)*100:.0f}%)  "
              f"actual-breakout-rate {r.get('base_rate','-')}")
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
