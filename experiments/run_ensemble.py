#!/usr/bin/env python3
"""Final ensemble (matrix item 12): OOF stacking of the best model families.

Stacks HGB + XGBoost + CatBoost via out-of-fold probabilities with a logistic
meta-learner, on the best inference-tractable feature set, for both targets and
both splits (LOCO + temporal). Compares to the best single family.
Outputs reports/ensemble.csv and appends to experiments_log.md.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, eval as E, config as C
from sip.modeling import make_model
from sklearn.linear_model import LogisticRegression

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_text():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return f"text_emb:{n}"
    return "semantic"


def oof_stack(df, blocks, target, families=("hgb", "xgb", "catboost"), sample=60000):
    if sample and len(df) > sample:
        df = df.groupby(["split_temporal", target], group_keys=False).sample(
            frac=sample / len(df), random_state=0).reset_index(drop=True)
    y = df[target].astype(int).values
    folds = S.loco_folds(df)
    base_oof = {f: np.full(len(df), np.nan) for f in families}
    for tri, tei in folds:
        X, _ = F.build_blocks(df, blocks, tri, y)
        for f in families:
            m = make_model(f).fit(X[tri], y[tri])
            base_oof[f][tei] = m.predict_proba(X[tei])[:, 1]
    M = np.column_stack([base_oof[f] for f in families])
    ok = ~np.isnan(M).any(1)
    # meta-learner via nested OOF on the stacked features
    meta_oof = np.full(len(df), np.nan)
    for tri, tei in S.loco_folds(df):
        tr = tri[ok[tri]]; te = tei[ok[tei]]
        if len(np.unique(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=500).fit(M[tr], y[tr])
        meta_oof[te] = lr.predict_proba(M[te])[:, 1]
    res = {"target": target}
    for f in families:
        res[f"{f}_auc"] = round(E._safe_auc(y[ok], base_oof[f][ok]), 4)
    mk = ~np.isnan(meta_oof)
    res["stack_auc"] = round(E._safe_auc(y[mk], meta_oof[mk]), 4)
    res["stack_ci"] = E.bootstrap_ci(y[mk], meta_oof[mk], E._safe_auc, 400)
    res["n"] = int(len(df))
    return res


def main():
    df = D.load(eligible_only=True)
    F.add_derived(df)
    blocks = TAB + [best_text()]
    rows = [oof_stack(df, blocks, t) for t in ("y_breakout_wc", "y_er_wc")]
    pd.DataFrame(rows).to_csv(C.REPORTS / "ensemble.csv", index=False)
    for r in rows:
        print(r)
    print("ensemble ->", C.REPORTS / "ensemble.csv")


if __name__ == "__main__":
    main()
