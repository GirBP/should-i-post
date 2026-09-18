#!/usr/bin/env python3
"""Train the ONE unified deployable classifier (replaces deployable.joblib A/B + the 262-video
deployable_mm.joblib pair shown as two competing scores).

One classifier, ALL features, trained on the ~12.5k real TikTok that have extracted video features:
  hand      caption stats, duration bins, timing, meta, missing-modality flags
  text      BGE embedding of caption+transcript (Whisper supplies the transcript live)
  video     SigLIP frame embedding (768-D)
  audio     CLAP embedding (512-D)

Two heads of the SAME feature set (day-1 is strictly optional at inference):
  U    unified, no day-1            -> used when the user did NOT opt into day-1
  Ud1  unified + day1_basic counts  -> used ONLY when the user opted in (manual or auto-from-link)

Pipeline mirrors build_deployable.py: temporal TRAIN -> isotonic calibration on VALID ->
conformal + cost bands on VALID -> honest metrics on TEST.
Artifact -> models/unified.joblib ; metrics -> reports/unified_metrics.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, eval as E, config as C
from sip.modeling import make_model, Calibrated, ConformalAbstention

ENC_NAME, ENC_MODEL = "bge", "BAAI/bge-base-en-v1.5"
BLOCKS_U = ["caption", "duration", "timing", "meta", "missing",
            f"text_emb:{ENC_NAME}", "mm:vsig", "mm:clap"]
BLOCKS_UD1 = BLOCKS_U + ["day1_basic"]
TARGET = "y_breakout_wc"


def _creator_masks(df, seed=C.SEED):
    """70/15/15 split by AUTHOR (no creator in two parts). The mm subset is the NEWEST download
    window, so a temporal split is degenerate here (train and test collapse into days of each
    other); the honest protocol for this set is leave-creators-out, as in the LOCO experiments."""
    rng = np.random.default_rng(seed)
    authors = df["author_id"].unique()
    rng.shuffle(authors)
    n = len(authors)
    a_tr = set(authors[: int(0.70 * n)])
    a_va = set(authors[int(0.70 * n): int(0.85 * n)])
    grp = df["author_id"].values
    tr = np.isin(grp, list(a_tr))
    va = np.isin(grp, list(a_va))
    te = ~(tr | va)
    return tr, va, te


def _train_one(df, blocks, kind):
    tr, va, te = _creator_masks(df)
    tri = np.where(tr)[0]
    y = df[TARGET].astype(int).values
    X, names = F.build_blocks(df, blocks, tri, y, target_col=TARGET)
    base = make_model("hgb").fit(X[tri], y[tri])
    cal = Calibrated(base, method="isotonic").fit_calibration(X[va], y[va])
    p_va = cal.predict_proba_pos(X[va])
    conf = ConformalAbstention(alpha=C.CONFORMAL_ALPHA).fit(p_va, y[va])
    bands = E.find_decision_bands(y[va], p_va, precision_target=C.PRECISION_AT_POST_TARGET)
    p_te = cal.predict_proba_pos(X[te])
    metrics = E.metrics(y[te], p_te, n_boot=1000, post_threshold=bands["t_high"])
    rep = E.band_report(y[te], p_te, bands)
    rng = np.random.default_rng(C.SEED)
    bg = X[tri][rng.choice(len(tri), min(200, len(tri)), replace=False)]
    print(f"  [{kind}] n_feat={len(names)}  test AUC={metrics['roc_auc']}  "
          f"Brier={metrics['brier']}  bands={bands}", flush=True)
    return {"blocks": blocks, "names": names, "model": base, "cal": cal, "conformal": conf,
            "bands": bands, "target": TARGET, "kind": kind, "metrics": metrics, "band_report": rep,
            "shap_bg": bg.astype(np.float32),
            "val_probs": p_va.astype(np.float32), "val_y": y[np.where(va)[0]].astype(np.int8)}


def main():
    df = D.load(eligible_only=True)
    df["video_id"] = df["video_id"].astype(str)
    mm_ids = set(pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])["video_id"].astype(str))
    df = df[df["video_id"].isin(mm_ids)].reset_index(drop=True)
    F.add_derived(df)
    y = df[TARGET].astype(int)
    print(f"unified training set: {len(df):,} TikTok with video features "
          f"(pos={int(y.sum())}, neg={int((1 - y).sum())}, creators={df['author_id'].nunique()})")

    U = _train_one(df, BLOCKS_U, "U")
    Ud1 = _train_one(df, BLOCKS_UD1, "Ud1")

    # factor directions for rationales (train-only correlations)
    tr, _, _ = _creator_masks(df)
    ytr = y.values[tr]
    factor_signs = {}
    for f in (F.CAPTION + F.DURATION + ["is_english_i", "hour_sin", "is_weekend"]):
        v = df[f].astype(float).values[tr]
        if np.std(v) > 0:
            factor_signs[f] = round(float(np.corrcoef(v, ytr)[0, 1]), 4)

    # example bank (similar past videos by BGE) from TRAIN rows only
    e = pd.read_parquet(C.PROC / f"emb_{ENC_NAME}.parquet")
    e["video_id"] = e["video_id"].astype(str)
    bank_df = df.iloc[np.where(tr)[0]][["video_id", "desc", TARGET]].sample(
        min(3000, int(tr.sum())), random_state=C.SEED)
    bcols = [c for c in e.columns if c != "video_id"]
    bank_emb = (e.set_index("video_id")[bcols]
                .reindex(bank_df["video_id"].values).fillna(0.0).values.astype(np.float32))
    example_bank = {"emb": bank_emb, "y": bank_df[TARGET].astype(int).values,
                    "caption": [str(s)[:80] for s in bank_df["desc"].fillna("").values]}

    artifact = {
        "version": "3.0.0-unified",
        "label_definition": ("breakout within-creator at H=14: views@14 / followers_at_post, "
                             "centred by the creator's own median, binarised at the global median; "
                             "fame-neutral reach proxy."),
        "bge_model": ENC_MODEL, "encoder_name": ENC_NAME,
        "U": {k: U[k] for k in ("blocks", "names", "model", "cal", "conformal", "bands",
                                "target", "metrics", "band_report", "shap_bg", "val_probs", "val_y")},
        "Ud1": {k: Ud1[k] for k in ("blocks", "names", "model", "cal", "conformal", "bands",
                                    "target", "metrics", "band_report", "shap_bg", "val_probs", "val_y")},
        "factor_signs": factor_signs,
        "example_bank": example_bank,
        "trained_on": {"n": int(len(df)), "creators": int(df["author_id"].nunique()),
                       "source": "real TikTok with extracted SigLIP/CLAP features"},
    }
    joblib.dump(artifact, C.MODELS / "unified.joblib")
    out = {"U": {"metrics": U["metrics"], "bands": U["bands"], "band_report": U["band_report"]},
           "Ud1": {"metrics": Ud1["metrics"], "bands": Ud1["bands"], "band_report": Ud1["band_report"]},
           "n": int(len(df))}
    json.dump(out, open(C.REPORTS / "unified_metrics.json", "w"), indent=2)
    print("saved -> models/unified.joblib ; reports/unified_metrics.json")


if __name__ == "__main__":
    main()
