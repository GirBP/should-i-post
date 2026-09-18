#!/usr/bin/env python3
"""Build the deployable MULTIMODAL Model A (text + SigLIP frames + CLAP/low-level audio + low-level
visual) on the 262 real videos that have extracted features.

Honesty by design (N=262 -> wide CI):
  - the headline number is the LOCO (leave-one-creator-out) OOF ROC-AUC with bootstrap CI;
  - isotonic calibration + decision bands are fitted on the LOCO OOF probabilities (unbiased),
    then the base model is refit on ALL rows for deployment;
  - the artifact carries an explicit small-sample caveat and is used in the product only as the
    file-mode "video boost" head next to the population text model (deployable.joblib).

Output: models/deployable_mm.joblib
Run:    PYTHONPATH=src python experiments/build_deployable_mm.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C, data as D, features as F, eval as E, experiment as X  # noqa: E402
from sip.modeling import make_model, Calibrated, ConformalAbstention  # noqa: E402

TAB = ["caption", "emotion", "duration", "timing", "meta"]
BLOCKS_MM = TAB + ["text_emb:bge", "mm:vsig", "mm:clap", "mm:aud", "mm:vll"]
TARGET = "y_breakout_wc"


def subset() -> pd.DataFrame:
    have = set(pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])["video_id"].astype(str))
    df = D.load(eligible_only=True)
    sub = df[df["video_id"].astype(str).isin(have)].reset_index(drop=True)
    F.add_derived(sub)
    return sub


def main():
    sub = subset()
    y = sub[TARGET].astype(int).values
    print(f"multimodal deployable: n={len(sub)} videos, hits={int(y.sum())}, "
          f"authors={sub['author_id'].nunique()}")

    # 1) honest LOCO evaluation + OOF probabilities (same runner as the ablation study)
    r = X.run(sub, BLOCKS_MM, "hgb", target=TARGET, model_kind="A", n_boot=1000,
              label=f"mm-deploy:{'+'.join(BLOCKS_MM)}", log=True, verbose=True)
    oof = np.asarray(r["_oof"], dtype=float)
    ok = ~np.isnan(oof)
    loco_auc, loco_ci = r["loco"]["roc_auc"], r["loco"]["roc_auc_ci"]
    print(f"LOCO OOF AUC {loco_auc} CI {loco_ci}  (oof coverage {ok.mean():.2f})")

    # 2) calibration + bands + conformal on the UNBIASED OOF probabilities
    cal_iso = Calibrated(None, method="isotonic")          # container; we fit the mapper directly
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[ok], y[ok])
    bands = E.find_decision_bands(y[ok], iso.predict(oof[ok]),
                                  precision_target=C.PRECISION_AT_POST_TARGET)
    conf = ConformalAbstention(alpha=C.CONFORMAL_ALPHA).fit(iso.predict(oof[ok]), y[ok])
    p_cal = iso.predict(oof[ok])
    metrics = E.metrics(y[ok], p_cal, n_boot=1000, post_threshold=bands["t_high"])
    print(f"OOF-calibrated: AUC {metrics['roc_auc']} CI {metrics['roc_auc_ci']} "
          f"brier {metrics['brier']} bands {bands}")

    # 3) deployment fit: base model on ALL rows (max data), calibration = the OOF isotonic above
    tri = np.arange(len(sub))
    Xall, names = F.build_blocks(sub, BLOCKS_MM, tri, y)
    base = make_model("hgb").fit(Xall, y)
    cal_iso.base = base
    cal_iso._cal = iso                                     # isotonic mapper fitted on OOF

    rng = np.random.default_rng(C.SEED)
    bg = Xall[rng.choice(len(sub), min(200, len(sub)), replace=False)]
    artifact = {
        "version": "mm-1.0",
        "caveat": ("Trained on N=262 real videos with extracted SigLIP/CLAP features; "
                   "LOCO CI is wide — use as the file-mode video boost next to the "
                   "population text model, not as its replacement."),
        "label_definition": ("breakout within-creator at H=14: views@14 / followers_at_post, "
                             "centred by the creator's own median, binarised at the global median"),
        "bge_model": "BAAI/bge-base-en-v1.5", "encoder_name": "bge",
        "blocks": BLOCKS_MM, "names": names, "model": base, "cal": cal_iso,
        "conformal": conf, "bands": bands, "target": TARGET,
        "metrics": {"loco_auc": loco_auc, "loco_ci": loco_ci,
                    "oof_calibrated": {k: metrics[k] for k in
                                       ("roc_auc", "roc_auc_ci", "pr_auc", "brier")},
                    "n": int(len(sub))},
        "shap_bg": bg.astype(np.float32),
        "val_probs": p_cal.astype(np.float32), "val_y": y[ok].astype(np.int8),
    }
    out = C.MODELS / "deployable_mm.joblib"
    joblib.dump(artifact, out)
    print(f"saved {out}  ({out.stat().st_size/1e6:.1f} MB, {len(names)} features)")


if __name__ == "__main__":
    main()
