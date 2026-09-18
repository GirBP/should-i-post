#!/usr/bin/env python3
"""Train and persist the deployable A -> B decision system.

Design choice for deployability: the deployable feature set uses ONLY blocks that
are reproducible at inference without dataset-fitted state —
  caption (deterministic from text), timing (post time), duration, meta,
  text_emb:bge (a public pretrained encoder).
Dataset-prior encoders (topic/music/hashtag target-encoding, creator_fit, trend)
are excluded from the *deployable* model because a brand-new candidate cannot
reproduce them cold; their research value is measured separately in the ablation.
HistGradientBoosting is used so any genuinely-missing feature is NaN-tolerant.

Pipeline: fit on temporal TRAIN -> isotonic calibration on VALID -> split-conformal
abstention on VALID -> cost-based decision bands on VALID -> evaluate on TEST.
Artifact -> models/deployable.joblib ; metrics -> reports/deployable_metrics.json
"""
import sys, json
from pathlib import Path
import numpy as np
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, eval as E, config as C
from sip.modeling import make_model, Calibrated, ConformalAbstention

_ENCODER_IDS = {"bge": "BAAI/bge-base-en-v1.5", "e5": "intfloat/e5-base-v2",
                "minilm": "sentence-transformers/all-MiniLM-L6-v2"}


def _pick_encoder():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return n, _ENCODER_IDS[n]
    return None, None


ENC_NAME, ENC_MODEL = _pick_encoder()
BLOCKS_A = ["caption", "timing", "duration", "meta", f"text_emb:{ENC_NAME}"]
# Model B uses only day-1 signals the user can actually supply at inference (no author-relative
# term, which needs the creator's historical median and is unavailable cold) — honest deployed AUC.
BLOCKS_B = BLOCKS_A + ["day1_basic"]


def _train_one(df, blocks, target, kind):
    tr, va, te = S.temporal_masks(df)
    tri = np.where(tr)[0]
    y = df[target].astype(int).values
    X, names = F.build_blocks(df, blocks, tri, y)
    base = make_model("hgb").fit(X[tri], y[tri])
    cal = Calibrated(base, method="isotonic").fit_calibration(X[va], y[va])
    p_va = cal.predict_proba_pos(X[va])
    conf = ConformalAbstention(alpha=C.CONFORMAL_ALPHA).fit(p_va, y[va])
    bands = E.find_decision_bands(y[va], p_va, precision_target=C.PRECISION_AT_POST_TARGET)
    p_te = cal.predict_proba_pos(X[te])
    metrics = E.metrics(y[te], p_te, n_boot=1000, post_threshold=bands["t_high"])
    rep = E.band_report(y[te], p_te, bands)
    centers, frac, w = E.reliability_curve(y[te], p_te)
    rng = np.random.default_rng(C.SEED)
    bg = X[tri][rng.choice(len(tri), min(200, len(tri)), replace=False)]   # SHAP background
    return {
        "blocks": blocks, "names": names, "model": base, "cal": cal,
        "conformal": conf, "bands": bands, "target": target, "kind": kind,
        "metrics": metrics, "band_report": rep,
        "reliability": {"pred": centers.tolist(), "obs": frac.tolist(), "n": w.tolist()},
        "shap_bg": bg.astype(np.float32),
        "val_probs": p_va.astype(np.float32), "val_y": y[va].astype(np.int8),
    }


def main():
    df = D.load(eligible_only=True)
    F.add_derived(df)
    if ENC_NAME is None:
        print("WARN: no text embedding cache — run scripts/encode_text.py first"); return
    print(f"training deployable on {len(df):,} eligible videos using encoder={ENC_NAME}")

    # interpretable factor directions (train correlation with success) for rationales
    tr, _, _ = S.temporal_masks(df)
    y_tr = df["y_breakout_wc"].astype(int).values[tr]
    factor_signs = {}
    for f in (F.CAPTION + F.DURATION + ["is_english_i", "hour_sin", "is_weekend"]):
        v = df[f].astype(float).values[tr]
        if np.std(v) > 0:
            factor_signs[f] = round(float(np.corrcoef(v, y_tr)[0, 1]), 4)

    # example bank for example-based evidence (similar past winners/losers)
    import pandas as pd
    emb_path = C.PROC / f"emb_{ENC_NAME}.parquet"
    e = pd.read_parquet(emb_path); e["video_id"] = e["video_id"].astype(str)
    bank_df = df.iloc[tr][["video_id", "desc", "y_breakout_wc"]].copy()
    bank_df = bank_df.sample(min(3000, len(bank_df)), random_state=C.SEED)
    bcols = [c for c in e.columns if c != "video_id"]
    emb_map = e.set_index("video_id")[bcols]
    bank_emb = emb_map.reindex(bank_df["video_id"].astype(str).values).fillna(0.0).values.astype(np.float32)
    example_bank = {
        "emb": bank_emb,
        "y": bank_df["y_breakout_wc"].astype(int).values,
        "caption": [str(s)[:80] for s in bank_df["desc"].fillna("").values],
    }

    A = _train_one(df, BLOCKS_A, "y_breakout_wc", "A")
    B = _train_one(df, BLOCKS_B, "y_breakout_wc", "B")
    print(f"  Model A test AUC={A['metrics']['roc_auc']}  Brier={A['metrics']['brier']}")
    print(f"  Model B test AUC={B['metrics']['roc_auc']}  Brier={B['metrics']['brier']}")
    print(f"  A bands={A['bands']}  band_report={A['band_report']}")

    artifact = {
        "version": "2.0.0-lingbow",
        "label_definition": ("breakout within-creator at H=14: views@14 / followers_at_post, "
                             "centred by the creator's own median, binarised at the global median; "
                             "fame-neutral reach proxy."),
        "bge_model": ENC_MODEL, "encoder_name": ENC_NAME,
        "A": {k: A[k] for k in ("blocks", "names", "model", "cal", "conformal", "bands",
                                "target", "metrics", "band_report", "shap_bg", "val_probs", "val_y")},
        "B": {k: B[k] for k in ("blocks", "names", "model", "cal", "conformal", "bands",
                                "target", "metrics", "band_report", "shap_bg", "val_probs", "val_y")},
        "caption_feature_names": F.CAPTION,
        "factor_signs": factor_signs,
        "example_bank": example_bank,
    }
    joblib.dump(artifact, C.MODELS / "deployable.joblib")
    out = {"A": {"metrics": A["metrics"], "bands": A["bands"], "band_report": A["band_report"],
                 "reliability": A["reliability"]},
           "B": {"metrics": B["metrics"], "bands": B["bands"], "band_report": B["band_report"],
                 "reliability": B["reliability"]},
           "label_definition": artifact["label_definition"]}
    json.dump(out, open(C.REPORTS / "deployable_metrics.json", "w"), indent=2)
    print("saved -> models/deployable.joblib ; reports/deployable_metrics.json")


if __name__ == "__main__":
    main()
