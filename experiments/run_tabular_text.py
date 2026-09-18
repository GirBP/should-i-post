#!/usr/bin/env python3
"""Tabular + text experiment suite (Model A, pre-publication).

Produces, with bootstrap CIs on BOTH protocols (LOCO + temporal):
  reports/baselines.json         majority / creator-prior / caption-logistic
  reports/model_family.csv       logreg vs HGB vs XGB vs CatBoost (fixed features)
  reports/text_encoders.csv      3 sentence encoders + SVD, marginal lift over TAB
  reports/ablation_A.csv         staged TAB -> +priors -> +text -> +creator_fit -> +trend -> +retrieval
  reports/optuna_best.json       tuned XGBoost vs default
Every run is also appended to reports/experiments_log.md.

Usage:  python experiments/run_tabular_text.py [--sections all|ablation|family|encoders|baselines|optuna]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, experiment as X, eval as E, features as F, config as C, splits as S
from sip.modeling import make_model

TAB = ["caption", "emotion", "duration", "timing", "meta"]
PRIORS = ["topic", "music", "hashtags"]


def stratified_sample(df, n, seed=0):
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    return (df.groupby(["split_temporal", "y_breakout_wc"], group_keys=False)
              .sample(frac=frac, random_state=seed).reset_index(drop=True))


def have_emb(name):
    return (C.PROC / f"emb_{name}.parquet").exists()


def best_text():
    for n in ("bge", "e5", "minilm"):
        if have_emb(n):
            return f"text_emb:{n}", n
    return "semantic", "svd"


# ---------------------------------------------------------------- baselines
def baselines(df):
    out = {}
    tr, va, te = S.temporal_masks(df)
    for target in ("y_breakout_wc", "y_er_wc"):
        y = df[target].astype(int).values
        # majority / base-rate predictor
        p_major = np.full(te.sum(), y[tr].mean())
        out[f"{target}/majority"] = E.metrics(y[te], p_major, n_boot=300)
        # creator prior: author's train success rate (target-encode author_id)
        g = pd.Series(y[tr]).groupby(df["author_id"].values[tr]).mean()
        gm = y[tr].mean()
        p_prior = df["author_id"].map(g).fillna(gm).values[te]
        out[f"{target}/creator_prior"] = E.metrics(y[te], p_prior, n_boot=300)
        # caption-only logistic
        r = X.run(df, ["caption"], "logreg", target=target, model_kind="A",
                  n_boot=300, label=f"caption-logit[{target}]", log=True, verbose=False)
        out[f"{target}/caption_logit"] = {"loco": r.get("loco"), "temporal": r.get("temporal")}
    json.dump(out, open(C.REPORTS / "baselines.json", "w"), indent=2)
    print("baselines ->", C.REPORTS / "baselines.json")
    return out


# ---------------------------------------------------------------- model family
def model_family(df, sample=60000):
    sub = stratified_sample(df, sample)
    text, _tn = best_text()
    feats = TAB + [text]
    rows = []
    for target in ("y_breakout_wc", "y_er_wc"):
        for mdl in ("logreg", "hgb", "xgb", "catboost"):
            try:
                r = X.run(sub, feats, mdl, target=target, model_kind="A", n_boot=300,
                          label=f"family:{mdl}[{target}]", log=True, verbose=True)
                rows.append({"target": target, "model": mdl,
                             "loco_auc": r["loco"]["roc_auc"], "loco_ci": r["loco"]["roc_auc_ci"],
                             "temporal_auc": r["temporal"]["roc_auc"], "brier": r["temporal"]["brier"]})
            except Exception as e:
                print("  family fail", mdl, target, type(e).__name__, str(e)[:80])
    pd.DataFrame(rows).to_csv(C.REPORTS / "model_family.csv", index=False)
    print("model_family ->", C.REPORTS / "model_family.csv")
    return rows


# ---------------------------------------------------------------- text encoders
def text_encoders(df, sample=None):
    sub = stratified_sample(df, sample)
    rows = []
    encoders = [("none", None)] + [(n, f"text_emb:{n}") for n in ("minilm", "bge", "e5") if have_emb(n)]
    encoders += [("svd", "semantic")]
    for target in ("y_breakout_wc", "y_er_wc"):
        base = None
        for name, blk in encoders:
            feats = TAB + ([blk] if blk else [])
            r = X.run(sub, feats, "logreg", target=target, model_kind="A", n_boot=400,
                      label=f"enc:{name}[{target}]", log=True, verbose=True)
            auc = r["loco"]["roc_auc"]
            if name == "none":
                base = auc
            rows.append({"target": target, "encoder": name, "loco_auc": auc,
                         "loco_ci": r["loco"]["roc_auc_ci"], "temporal_auc": r["temporal"]["roc_auc"],
                         "lift_over_TAB": None if base is None else round(auc - base, 4)})
    pd.DataFrame(rows).to_csv(C.REPORTS / "text_encoders.csv", index=False)
    print("text_encoders ->", C.REPORTS / "text_encoders.csv")
    return rows


# ---------------------------------------------------------------- staged ablation
def ablation(df, sample=None, model="logreg"):
    sub = stratified_sample(df, sample)
    text, _tn = best_text()
    cfit = f"creator_fit:{_tn}" if _tn != "svd" else "creator_fit"
    retr = f"retrieval:{_tn}" if _tn != "svd" else "retrieval"
    stages = [
        ("TAB", TAB),
        ("+priors", TAB + PRIORS),
        ("+text", TAB + PRIORS + [text]),
        ("+creator_fit", TAB + PRIORS + [text, cfit]),
        ("+trend", TAB + PRIORS + [text, cfit, "trend_fit"]),
        ("+retrieval", TAB + PRIORS + [text, cfit, "trend_fit", retr]),
    ]
    rows = []
    for target in ("y_breakout_wc", "y_er_wc"):
        prev = None
        for name, feats in stages:
            r = X.run(sub, feats, model, target=target, model_kind="A", n_boot=500,
                      label=f"abl:{name}[{target}]", log=True, verbose=True)
            auc = r["loco"]["roc_auc"]
            rows.append({"target": target, "stage": name, "model": model,
                         "loco_auc": auc, "loco_ci_lo": r["loco"]["roc_auc_ci"][0],
                         "loco_ci_hi": r["loco"]["roc_auc_ci"][1],
                         "temporal_auc": r["temporal"]["roc_auc"], "brier": r["temporal"]["brier"],
                         "marginal": None if prev is None else round(auc - prev, 4)})
            prev = auc
    pd.DataFrame(rows).to_csv(C.REPORTS / "ablation_A.csv", index=False)
    print("ablation_A ->", C.REPORTS / "ablation_A.csv")
    return rows


# ---------------------------------------------------------------- optuna
def optuna_tune(df, sample=40000, n_trials=30):
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score
    sub = stratified_sample(df, sample)
    text, _tn = best_text()
    y = sub["y_breakout_wc"].astype(int).values
    folds = S.loco_folds(sub)
    # precompute per-fold matrices once (features fold-safe)
    fold_data = []
    for tri, tei in folds:
        Xall, _ = F.build_blocks(sub, TAB + [text], tri, y)
        fold_data.append((Xall, tri, tei))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            tree_method="hist", n_jobs=-1, eval_metric="logloss", random_state=C.SEED)
        aucs = []
        for Xall, tri, tei in fold_data:
            m = XGBClassifier(**params); m.fit(Xall[tri], y[tri])
            aucs.append(roc_auc_score(y[tei], m.predict_proba(Xall[tei])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # default for comparison
    default_auc = np.mean([roc_auc_score(y[tei], make_model("xgb").fit(Xall[tri], y[tri])
                          .predict_proba(Xall[tei])[:, 1]) for Xall, tri, tei in fold_data])
    out = {"n": len(sub), "n_trials": n_trials, "best_loco_auc": round(study.best_value, 4),
           "default_loco_auc": round(float(default_auc), 4), "best_params": study.best_params}
    json.dump(out, open(C.REPORTS / "optuna_best.json", "w"), indent=2)
    print("optuna ->", out["best_loco_auc"], "vs default", out["default_loco_auc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="all")
    ap.add_argument("--ablation-sample", type=int, default=0, help="0 = full eligible data")
    args = ap.parse_args()
    df = D.load(eligible_only=True)
    F.add_derived(df)
    print(f"loaded {len(df):,} eligible videos; emb: "
          f"{[n for n in ('minilm','bge','e5') if have_emb(n)]}")
    secs = args.sections.split(",") if args.sections != "all" else \
        ["baselines", "encoders", "ablation", "family", "optuna"]
    abl_n = args.ablation_sample or None
    if "baselines" in secs: baselines(df)
    if "encoders" in secs: text_encoders(df, sample=abl_n)
    if "ablation" in secs: ablation(df, sample=abl_n, model="logreg")
    if "family" in secs: model_family(df)
    if "optuna" in secs: optuna_tune(df)
    print("DONE tabular+text suite")


if __name__ == "__main__":
    main()
