"""Evaluation: metrics, bootstrap CIs, reliability, and decision bands.

Metrics match the business decision: ranking quality (ROC/PR-AUC), the precision
of a "Post" call (precision@Post), winner recall, and honest probabilities
(Brier + reliability). 95% CIs are bootstrap (stratified resample of test rows).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_score, recall_score)


def _safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0):
    """95% percentile CI for metric `fn(y,p)` via row resampling."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def precision_recall_at(y, p, threshold):
    yhat = (np.asarray(p) >= threshold).astype(int)
    if yhat.sum() == 0:
        return float("nan"), 0.0, 0
    prec = precision_score(y, yhat, zero_division=0)
    rec = recall_score(y, yhat, zero_division=0)
    return float(prec), float(rec), int(yhat.sum())


def reliability_curve(y, p, bins=10):
    """Return (bin_centers, frac_pos, weights) for a calibration plot."""
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    centers, frac, w = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append(p[m].mean()); frac.append(y[m].mean()); w.append(int(m.sum()))
    return np.array(centers), np.array(frac), np.array(w)


def metrics(y, p, n_boot=1000, post_threshold=0.5, seed=0):
    """Full metric bundle with bootstrap CIs. Returns a JSON-friendly dict."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    auc = _safe_auc(y, p)
    pr = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else float("nan")
    prec, rec, npost = precision_recall_at(y, p, post_threshold)
    auc_lo, auc_hi = bootstrap_ci(y, p, _safe_auc, n_boot, seed)
    # per-class precision/recall/F1 + macro-F1 + accuracy at the operating threshold
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    yhat = (p >= post_threshold).astype(int)
    pr_c, rc_c, f1_c, _ = precision_recall_fscore_support(y, yhat, labels=[0, 1], zero_division=0)
    return {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "roc_auc": round(auc, 4), "roc_auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
        "pr_auc": round(pr, 4), "brier": round(brier, 4),
        "precision_at_post": round(prec, 4) if prec == prec else None,
        "recall_at_post": round(rec, 4), "n_post": npost,
        "accuracy": round(float(accuracy_score(y, yhat)), 4),
        "f1_pos": round(float(f1_c[1]), 4), "f1_neg": round(float(f1_c[0]), 4),
        "macro_f1": round(float(f1_c.mean()), 4),
        "precision_neg": round(float(pr_c[0]), 4), "recall_neg": round(float(rc_c[0]), 4),
        "post_threshold": post_threshold,
    }


def find_decision_bands(y_val, p_val, precision_target=0.65, min_coverage=0.05):
    """Pick (t_low, t_high) on validation.

    t_high = lowest threshold whose precision@Post >= target (with enough volume);
    t_low  = symmetric "do not post" cutoff where precision of the *negative* call
             is comparably high.  Between them -> "Unsure".
    Falls back to quantiles if the target is unreachable.
    """
    y = np.asarray(y_val).astype(int); p = np.asarray(p_val, dtype=float)
    n = len(y)
    order = np.unique(np.round(p, 4))
    # t_high: smallest t with precision(p>=t) >= target and coverage >= min_coverage
    t_high = None
    for t in order:
        sel = p >= t
        if sel.sum() >= max(min_coverage * n, 20):
            prec = y[sel].mean()
            if prec >= precision_target:
                t_high = float(t); break
    if t_high is None:
        t_high = float(np.quantile(p, 0.80))
    # t_low: largest t with precision of "do not post" (y==0 | p<t) >= target
    t_low = None
    for t in order[::-1]:
        sel = p < t
        if sel.sum() >= max(min_coverage * n, 20):
            neg_prec = 1.0 - y[sel].mean()
            if neg_prec >= precision_target:
                t_low = float(t); break
    if t_low is None:
        t_low = float(np.quantile(p, 0.20))
    if t_low > t_high:
        t_low, t_high = float(np.quantile(p, 0.20)), float(np.quantile(p, 0.80))
    return {"t_low": round(t_low, 4), "t_high": round(t_high, 4)}


def apply_bands(p, bands):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), "Unsure", dtype=object)
    out[p >= bands["t_high"]] = "Post"
    out[p < bands["t_low"]] = "Do not post"
    return out


def band_report(y, p, bands):
    """Coverage + precision of each band on a held-out set."""
    dec = apply_bands(p, bands)
    y = np.asarray(y).astype(int)
    rep = {}
    n = len(y)
    for label in ("Post", "Unsure", "Do not post"):
        m = dec == label
        cov = float(m.mean())
        if m.sum() == 0:
            rep[label] = {"coverage": 0.0, "n": 0, "base_rate": None}
            continue
        rep[label] = {"coverage": round(cov, 3), "n": int(m.sum()),
                      "base_rate": round(float(y[m].mean()), 3)}
    rep["abstain_coverage"] = rep["Unsure"]["coverage"]
    return rep
