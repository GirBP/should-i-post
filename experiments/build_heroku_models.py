"""Fit + save the deployable calculator models for the Heroku site.

These use ONLY manually-enterable signals (caption text, duration, posting time, +day-1 counts) —
NOT SigLIP/CLAP, which need the raw video. So they are small, torch-light, and fit Heroku:
  Model A (pre-publication): caption + duration + timing + meta + missing + MiniLM(text)
  Model B (day-1 keep/delete): A's blocks + leak-free day-1 counts

Saves models/model_a_calc.joblib and models/model_b_calc.joblib (estimator + feature spec +
decision bands), plus reports/heroku_models.json. Fits on ALL of the balanced 10k (deployment),
with a quick StratifiedGroupKFold AUC for the honest headline number.
"""
import os, sys, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("SIP_DEVICE", "cpu")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, config as C, eval as E
from sip.modeling import make_model

TARGET, SEED, N = "y_breakout_wc", 0, 10000
A_BLOCKS = ["caption", "duration", "timing", "meta", "missing"]   # hand features only: torch-free, live-servable, no train/serve skew
B_BLOCKS = A_BLOCKS + ["day1_basic"]


def balanced(df):
    rng = np.random.RandomState(SEED); per = N // 2
    return pd.concat([df[df[TARGET] == c].sample(min(per, (df[TARGET] == c).sum()), random_state=rng)
                      for c in (1, 0)]).sample(frac=1.0, random_state=rng).reset_index(drop=True)


def cv_auc(sub, blocks, y, groups):
    p = np.full(len(sub), np.nan)
    for tri, tei in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(sub, y, groups):
        X, _ = F.build_blocks(sub, blocks, tri, y, target_col=TARGET)
        p[tei] = make_model("hgb").fit(X[tri], y[tri]).predict_proba(X[tei])[:, 1]
    return p


def main():
    df = D.load(eligible_only=True); df["video_id"] = df["video_id"].astype(str)
    df = df[df["video_id"].isin(set(pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])["video_id"].astype(str)))]
    sub = balanced(df); F.add_derived(sub)
    y = sub[TARGET].astype(int).values; groups = sub["author_id"].values
    out = {"n": len(sub)}
    C.MODELS = getattr(C, "MODELS", C.ROOT / "models"); C.MODELS.mkdir(exist_ok=True)

    for tag, blocks, kind in [("a_calc", A_BLOCKS, "A"), ("b_calc", B_BLOCKS, "B")]:
        p = cv_auc(sub, blocks, y, groups)                         # honest OOF metric
        bands = E.find_decision_bands(y, p, precision_target=0.60, min_coverage=0.05)
        Xall, names = F.build_blocks(sub, blocks, np.arange(len(sub)), y, target_col=TARGET)
        model = make_model("hgb").fit(Xall, y)                     # final fit on ALL data for deployment
        art = {"model": model, "blocks": blocks, "feature_names": names, "kind": kind,
               "bands": {k: float(v) for k, v in bands.items() if isinstance(v, (int, float))},
               "target": TARGET, "n_train": len(sub)}
        path = C.MODELS / f"model_{tag}.joblib"; joblib.dump(art, path)
        auc = roc_auc_score(y, p)
        out[tag] = {"auc": round(auc, 4), "ci": list(E.bootstrap_ci(y, p, roc_auc_score, n_boot=1000)),
                    "n_features": len(names), "bands": art["bands"], "path": str(path.name)}
        print(f"  {tag:8} AUC {auc:.3f}  {len(names)}f  -> {path.name}", flush=True)

    json.dump(out, open(C.REPORTS / "heroku_models.json", "w"), indent=2)
    print("saved -> reports/heroku_models.json ; models/model_{a,b}_calc.joblib")


if __name__ == "__main__":
    main()
