"""Per-class breakdown: do the video/audio embeddings actually help the BREAKOUT class?

A single ROC-AUC hides class-specific behaviour. This script fixes the sample and the
leave-one-creator-out folds, then trains the SAME HistGradientBoosting on different feature
SETS and reports, for each, the full per-class picture:

  * ROC-AUC (ranking) and PR-AUC / average-precision for the POSITIVE class (breakout)
  * precision / recall / F1 / support for class 0 AND class 1 (at t=0.5 and at F1-optimal t)
  * macro-F1, confusion matrix
  * bootstrap 95% CI on the DELTA vs the hand-feature baseline for
    {positive-class F1, positive-class recall, PR-AUC, ROC-AUC} — is any gain real or noise?

Feature sets (same rows, same folds, same model):
  hand              caption+duration+timing+meta+missing   (deployed Model A, torch-free)
  hand+video        + SigLIP video embedding (768-D)
  hand+audio        + CLAP audio embedding (512-D)
  hand+video+audio  + both
  video_only        SigLIP only
  audio_only        CLAP only
  video+audio_only  both, no hand features

    PYTHONPATH=src python experiments/perclass_eval.py --n 10000 \
        --mm-parquet data/multimodal/features_tiktok.parquet
"""
import os, sys, json, argparse, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("SIP_DEVICE", "cpu")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, confusion_matrix)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, config as C
from sip.modeling import make_model

TARGET, SEED = "y_breakout_wc", 0
HAND = ["caption", "duration", "timing", "meta", "missing"]


def load(n, mm_parquet):
    df = D.load(eligible_only=True); df["video_id"] = df["video_id"].astype(str)
    mm = pd.read_parquet(mm_parquet); mm["video_id"] = mm["video_id"].astype(str)
    vsig = [c for c in mm.columns if c.startswith("vsig")]
    clap = [c for c in mm.columns if c.startswith("clap")]
    have = df[df["video_id"].isin(set(mm["video_id"]))].copy()
    rng = np.random.RandomState(SEED); per = n // 2
    sub = pd.concat([have[have[TARGET] == c].sample(min(per, int((have[TARGET] == c).sum())), random_state=rng)
                     for c in (1, 0)]).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    F.add_derived(sub)
    y = sub[TARGET].astype(int).values; groups = sub["author_id"].values
    mmi = mm.set_index("video_id")
    Xv = mmi[vsig].reindex(sub["video_id"]).fillna(0).values.astype(np.float32)
    Xa = mmi[clap].reindex(sub["video_id"]).fillna(0).values.astype(np.float32)
    return sub, y, groups, Xv, Xa


def oof(sub, y, folds, blocks, extra):
    """Out-of-fold P(breakout). `blocks` = hand blocks (fold-safe); `extra` = dense mm cols or None."""
    p = np.full(len(y), np.nan)
    for tri, tei in folds:
        mats = []
        if blocks:
            Xh, _ = F.build_blocks(sub, blocks, tri, y, target_col=TARGET); mats.append(Xh)
        if extra is not None:
            mats.append(extra)
        X = np.concatenate(mats, axis=1).astype(np.float32)
        m = make_model("hgb").fit(X[tri], y[tri])
        p[tei] = m.predict_proba(X[tei])[:, 1]
    return p


def f1_opt_threshold(y, p):
    ts = np.linspace(0.05, 0.95, 181)
    best_t, best_f = 0.5, -1
    for t in ts:
        pr, rc, f, _ = precision_recall_fscore_support(y, (p >= t).astype(int), labels=[0, 1], zero_division=0)
        if f[1] > best_f:
            best_f, best_t = f[1], t
    return float(best_t)


def per_class(y, p, t):
    pred = (p >= t).astype(int)
    pr, rc, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "t": round(t, 3),
        "neg": {"precision": round(pr[0], 3), "recall": round(rc[0], 3), "f1": round(f[0], 3), "support": int(s[0])},
        "pos": {"precision": round(pr[1], 3), "recall": round(rc[1], 3), "f1": round(f[1], 3), "support": int(s[1])},
        "macro_f1": round(float(f.mean()), 3),
        "confusion": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }


def summary(y, p, t):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),   # positive-class average precision
        "at_0.5": per_class(y, p, 0.5),
        "at_f1opt": per_class(y, p, t),
    }


def boot_delta(y, base_p, cfg_p, t=0.5, n=1000, seed=0):
    """95% CI on config-minus-hand deltas over paired bootstrap resamples.
    Class-threshold metrics (pos-F1, pos-recall) are read at the SHARED operating point t
    (default 0.5); PR-AUC and ROC-AUC are threshold-free."""
    rng = np.random.RandomState(seed)
    d_f1, d_rec, d_pr, d_auc = [], [], [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        yi = y[i]
        if len(np.unique(yi)) < 2:
            continue
        _, rb, fb, _ = precision_recall_fscore_support(yi, (base_p[i] >= t).astype(int), labels=[0, 1], zero_division=0)
        _, rc, fc, _ = precision_recall_fscore_support(yi, (cfg_p[i] >= t).astype(int), labels=[0, 1], zero_division=0)
        d_f1.append(fc[1] - fb[1]); d_rec.append(rc[1] - rb[1])
        d_pr.append(average_precision_score(yi, cfg_p[i]) - average_precision_score(yi, base_p[i]))
        d_auc.append(roc_auc_score(yi, cfg_p[i]) - roc_auc_score(yi, base_p[i]))
    def ci(a):
        a = np.array(a); return [round(float(np.percentile(a, 2.5)), 4),
                                 round(float(np.median(a)), 4), round(float(np.percentile(a, 97.5)), 4)]
    return {"pos_f1_delta": ci(d_f1), "pos_recall_delta": ci(d_rec),
            "pr_auc_delta": ci(d_pr), "roc_auc_delta": ci(d_auc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--mm-parquet", default=str(C.MM / "features_tiktok.parquet"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    sub, y, groups, Xv, Xa = load(args.n, args.mm_parquet)
    Xva = np.concatenate([Xv, Xa], axis=1)
    folds = list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED).split(sub, y, groups))
    print(f"n={len(y)} | balance={dict(pd.Series(y).value_counts())} | creators={pd.Series(groups).nunique()} | "
          f"video={Xv.shape[1]}D audio={Xa.shape[1]}D | mm={args.mm_parquet}", flush=True)

    configs = [
        ("hand",             HAND, None),
        ("hand+video",       HAND, Xv),
        ("hand+audio",       HAND, Xa),
        ("hand+video+audio", HAND, Xva),
        ("video_only",       [],   Xv),
        ("audio_only",       [],   Xa),
        ("video+audio_only", [],   Xva),
    ]

    results, probs, thr = {}, {}, {}
    for name, blocks, extra in configs:
        p = oof(sub, y, folds, blocks, extra)
        t = f1_opt_threshold(y, p)
        probs[name], thr[name] = p, t
        results[name] = summary(y, p, t)
        s = results[name]
        print(f"\n=== {name} ===")
        print(f"  ROC-AUC {s['roc_auc']:.3f} | PR-AUC(pos) {s['pr_auc']:.3f}")
        for tag in ("at_0.5", "at_f1opt"):
            b = s[tag]
            print(f"  @t={b['t']:<4}  POS  P {b['pos']['precision']:.2f} R {b['pos']['recall']:.2f} "
                  f"F1 {b['pos']['f1']:.2f}   |  NEG  P {b['neg']['precision']:.2f} R {b['neg']['recall']:.2f} "
                  f"F1 {b['neg']['f1']:.2f}   |  macro-F1 {b['macro_f1']:.2f}")

    print("\n=== DELTA vs hand (bootstrap 95% CI: [lo, median, hi]; class metrics @ t=0.5) ===")
    deltas = {}
    base = probs["hand"]
    for name in ["hand+video", "hand+audio", "hand+video+audio"]:
        d = boot_delta(y, base, probs[name], t=0.5)
        deltas[name] = d
        real = lambda ci: "REAL (CI excludes 0)" if (ci[0] > 0 or ci[2] < 0) else "noise (CI spans 0)"
        print(f"\n  {name} - hand:")
        print(f"    pos-F1     Δ {d['pos_f1_delta']}   {real(d['pos_f1_delta'])}")
        print(f"    pos-recall Δ {d['pos_recall_delta']}   {real(d['pos_recall_delta'])}")
        print(f"    PR-AUC     Δ {d['pr_auc_delta']}   {real(d['pr_auc_delta'])}")
        print(f"    ROC-AUC    Δ {d['roc_auc_delta']}   {real(d['roc_auc_delta'])}")

    out = {"n": len(y), "balance": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()},
           "mm_parquet": args.mm_parquet, "folds": args.folds,
           "configs": results, "deltas_vs_hand": deltas}
    path = C.REPORTS / "perclass_eval.json"
    json.dump(out, open(path, "w"), indent=2)
    # flat CSV for quick scanning
    rows = []
    for name, s in results.items():
        for tag in ("at_0.5", "at_f1opt"):
            b = s[tag]
            rows.append({"config": name, "roc_auc": s["roc_auc"], "pr_auc": s["pr_auc"], "threshold": b["t"],
                         "pos_precision": b["pos"]["precision"], "pos_recall": b["pos"]["recall"], "pos_f1": b["pos"]["f1"],
                         "neg_precision": b["neg"]["precision"], "neg_recall": b["neg"]["recall"], "neg_f1": b["neg"]["f1"],
                         "macro_f1": b["macro_f1"]})
    pd.DataFrame(rows).to_csv(C.REPORTS / "perclass_eval.csv", index=False)
    print(f"\nsaved -> {path}\nsaved -> {C.REPORTS / 'perclass_eval.csv'}")


if __name__ == "__main__":
    main()
