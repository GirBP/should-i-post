"""Best pre-publication Model-A config (TAB + text + creator-fit + trend, no priors),
recomputed with the leak-safe closed-window creator-fit. Writes reports/best_A.csv.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, experiment as X, config as C

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def main():
    df = D.load(eligible_only=True); F.add_derived(df)
    tn = next((n for n in ("bge", "minilm", "e5") if (C.PROC / f"emb_{n}.parquet").exists()), "svd")
    text = f"text_emb:{tn}" if tn != "svd" else "semantic"
    cfit = f"creator_fit:{tn}" if tn != "svd" else "creator_fit"
    blocks = TAB + [text, cfit, "trend_fit"]
    rows = []
    for target in ("y_breakout_wc", "y_er_wc"):
        for mdl in ("logreg", "hgb"):
            r = X.run(df, blocks, mdl, target=target, model_kind="A", n_boot=500,
                      label=f"bestA-noPriors-closedwin:{mdl}[{target}]", log=True, verbose=True)
            rows.append({"target": target, "model": mdl, "loco_auc": r["loco"]["roc_auc"],
                         "loco_ci": r["loco"]["roc_auc_ci"], "temporal_auc": r["temporal"]["roc_auc"],
                         "brier": r["temporal"]["brier"]})
    pd.DataFrame(rows).to_csv(C.REPORTS / "best_A.csv", index=False)
    print("saved -> reports/best_A.csv")


if __name__ == "__main__":
    main()
