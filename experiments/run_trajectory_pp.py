"""P1.1/P1.2 — trajectory & point-process view of popularity (forecasting + cascade domains).

Daily cumulative views (day 0..30) let us ask the key A3 question: WHEN is the day-14
breakout outcome determined? Two analyses:

1. EARLY-WINDOW determinacy: AUC of predicting the day-14 within-creator breakout label
   using only early breakout (views@d / followers) for d in {0,1,2,3,5,7}. The curve shows
   how much each early day adds — i.e. how front-loaded the signal is (and why the pre-pub
   model is hard while Model B is easy).
2. GROWTH-CURVE / point-process extrapolation: fit a log-logistic growth curve to days 0..d,
   extrapolate to day-14, threshold at the creator median; compare AUC to (a) the raw day-d
   value and (b) the literature's self-exciting intuition (early increments predict the tail).
   Run on a sample for the per-video curve fit.

Output: reports/trajectory_pp.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, config as C
from sklearn.metrics import roc_auc_score as _ra


def roc_auc_score(y, s):
    return _ra(y, np.nan_to_num(np.asarray(s, float), nan=0.0, posinf=1e9, neginf=-1e9))


def cum_views_by_day(vids, days):
    e = pd.read_parquet(C.RAW / "engagement_daily.parquet",
                        columns=["video_id", "days_since_post", "play_count"])
    e["video_id"] = e["video_id"].astype(str)
    out = {}
    for d in days:
        cum = (e[e.days_since_post <= d].sort_values(["video_id", "days_since_post"])
               .groupby("video_id").tail(1).set_index("video_id")["play_count"])
        out[d] = cum.reindex(vids).values
    return out


def log_logistic_extrap(cum_series_days, cum_values, target_day):
    """Fit V(t)=A / (1+(t0/t)^b) to (days, cumulative) and predict at target_day."""
    from scipy.optimize import curve_fit
    t = np.asarray(cum_series_days, float) + 1.0
    v = np.asarray(cum_values, float)
    if np.all(v <= 0) or len(t) < 3:
        return v[-1] if len(v) else 0.0
    def f(t, A, t0, b):
        return A / (1.0 + (t0 / t) ** b)
    try:
        A0 = max(v[-1] * 1.5, 1.0)
        p, _ = curve_fit(f, t, v, p0=[A0, 2.0, 1.5], maxfev=2000,
                         bounds=([v[-1], 0.1, 0.2], [v[-1] * 50 + 10, 60, 6]))
        return float(f(target_day + 1.0, *p))
    except Exception:
        return float(v[-1])


def main():
    df = D.load(eligible_only=True)
    vids = df["video_id"].astype(str).values
    foll = pd.to_numeric(df["followers_at_post"], errors="coerce").fillna(1).clip(lower=1).values
    y = df["y_breakout_wc"].astype(int).values
    days = [0, 1, 2, 3, 5, 7]
    cv = cum_views_by_day(vids, days + [14])
    rows = []

    # --- 1. early-window determinacy (full data, single-feature AUC) ---
    auc14 = roc_auc_score(y, np.nan_to_num(cv[14]) / foll)  # ceiling: day-14 itself
    for d in days:
        early_bo = np.log1p(np.nan_to_num(cv[d]) / foll)
        auc = roc_auc_score(y, early_bo)
        rows.append({"section": "early_window", "day": d, "auc_vs_breakout14": round(auc, 4),
                     "frac_of_day14_auc": round((auc - 0.5) / (auc14 - 0.5), 3)})
        print(f"  day {d}: AUC={auc:.4f} ({(auc-0.5)/(auc14-0.5):.0%} of day-14 separability)", flush=True)
    rows.append({"section": "early_window", "day": 14, "auc_vs_breakout14": round(auc14, 4), "frac_of_day14_auc": 1.0})

    # --- 2. growth-curve extrapolation (sample) ---
    rng = np.random.default_rng(0)
    samp = rng.choice(len(df), min(8000, len(df)), replace=False)
    for d in [1, 3, 7]:
        fit_days = [x for x in days if x <= d]
        pred = np.array([log_logistic_extrap(fit_days, [cv[x][i] for x in fit_days], 14) for i in samp])
        raw = np.nan_to_num(cv[d][samp])
        ys = y[samp]
        auc_fit = roc_auc_score(ys, np.log1p(pred / foll[samp]))
        auc_raw = roc_auc_score(ys, np.log1p(raw / foll[samp]))
        rows.append({"section": "growth_extrap", "day": d, "auc_extrap_to_14": round(auc_fit, 4),
                     "auc_raw_day_d": round(auc_raw, 4), "n": len(samp)})
        print(f"  extrap day0..{d}->14: AUC={auc_fit:.4f} vs raw day-{d} {auc_raw:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(C.REPORTS / "trajectory_pp.csv", index=False)
    print("saved -> reports/trajectory_pp.csv")


if __name__ == "__main__":
    main()
