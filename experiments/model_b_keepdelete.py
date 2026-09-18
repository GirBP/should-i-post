"""Model B — day-1 keep/delete decision (organic, no paid boosting).

24h after posting, given LEAK-FREE day-1 signal (raw counts + engagement rate; NOT the
author-relative feature the audit flagged), predict whether the video will break out and decide:
  KEEP    — likely to succeed, or uncertain (default; never delete on doubt)
  DELETE  — day-1 signal strongly says a dud unlikely to recover

Cost asymmetry: a false DELETE (killing a would-be hit) is catastrophic and irreversible;
a false KEEP (a dud sits on the profile) is cheap. So we DELETE only when precision-of-dud is
high — and report exactly how many true breakouts the policy would wrongly delete (must be ~0).

    PYTHONPATH=src python experiments/model_b_keepdelete.py --n 10000
"""
import os, sys, argparse, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("SIP_DEVICE", "cpu")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, config as C, leakage as L
from sip.modeling import make_model

TARGET, SEED = "y_breakout_wc", 0
A_BLOCKS = ["caption", "duration", "timing", "meta", "missing", "text_emb:minilm", "mm:vsig", "mm:clap"]


def oof(sub, blocks, y, groups, folds):
    p = np.full(len(sub), np.nan)
    for tri, tei in folds:
        X, _ = F.build_blocks(sub, blocks, tri, y, target_col=TARGET)
        m = make_model("hgb").fit(X[tri], y[tri]); p[tei] = m.predict_proba(X[tei])[:, 1]
    return p


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); a = [roc_auc_score(y[i], p[i]) for i in
        (rng.randint(0, len(y), len(y)) for _ in range(n)) if len(np.unique(y[i])) == 2]
    return round(np.percentile(a, 2.5), 3), round(np.percentile(a, 97.5), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10000); args = ap.parse_args()
    df = D.load(eligible_only=True); df["video_id"] = df["video_id"].astype(str)
    mm = set(pd.read_parquet(C.MM / "features.parquet", columns=["video_id"])["video_id"].astype(str))
    have = df[df["video_id"].isin(mm)].copy()
    rng = np.random.RandomState(SEED); per = args.n // 2
    sub = pd.concat([have[have[TARGET] == c].sample(min(per, (have[TARGET] == c).sum()), random_state=rng)
                     for c in (1, 0)]).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    F.add_derived(sub); y = sub[TARGET].astype(int).values; groups = sub["author_id"].values
    L.assert_model_b(F.DAY1_BASIC)  # confirm leak-free day-1 set
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(sub, y, groups))
    print(f"n={len(sub)} | day-1 feats (leak-free)={F.DAY1_BASIC}", flush=True)

    pA = oof(sub, A_BLOCKS, y, groups, folds)
    pB = oof(sub, A_BLOCKS + ["day1_basic"], y, groups, folds)
    for nm, p in [("Model A (pre-pub)", pA), ("Model B (+day-1)", pB)]:
        lo, hi = boot_ci(y, p); print(f"  {nm:20} ROC-AUC {roc_auc_score(y, p):.3f}  CI [{lo}, {hi}]", flush=True)

    # ---- KEEP / DELETE policy on Model B: delete the bottom-k while the DELETE set stays ≥X% duds ----
    order = np.argsort(pB)                        # ascending prob -> most likely duds first
    hits_cum = np.cumsum(y[order] == 1)           # true hits among the bottom-k
    ks = np.arange(1, len(order) + 1)
    dud_prec = 1 - hits_cum / ks                  # set-level dud-precision of "delete bottom k"
    n_break = int(y.sum())
    print("\n=== KEEP / DELETE policy (Model B) — safety-first ===")
    print("  policy: DELETE the bottom-k by day-1 prob; pick the LARGEST k whose delete-set is ≥X% true duds")
    for prec_target in (0.99, 0.97, 0.95, 0.90):
        ok = np.where(dud_prec >= prec_target)[0]
        k = int(ok.max() + 1) if len(ok) else 0
        lost = int(hits_cum[k - 1]) if k else 0
        print(f"  delete-set ≥{prec_target} duds: DELETE {k/len(sub)*100:5.1f}% of videos "
              f"| true hits wrongly deleted: {lost} ({lost/max(n_break,1)*100:.2f}% of all hits)")
    print("\n  → the rest are KEPT (default; never delete on doubt). Higher bar = safer but fewer deletes.")


if __name__ == "__main__":
    main()
