#!/usr/bin/env python3
"""Internal, temporally-safe trend-fit features (same-platform momentum).

External trend datasets (e.g. ronantakizawa/tiktok-trending-hashtags) are only
yearly-rank granularity — useless for videos all posted within 2024-06..11. So we
derive momentum FROM lingbow itself: for each video's music_id and hashtags, how
much more were they used in the days just BEFORE this post than in an earlier
baseline window. Only earlier posts are used (strictly t' < t) -> no leakage.

  trend_music   = log1p(recent_uses) - log1p(baseline_rate*recent_window)  for the sound
  trend_hashtag = same, averaged over the video's hashtags

Output: data/processed/trend_features.parquet (video_id, trend_music, trend_hashtag)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C, data as D

W_RECENT = 7        # days just before the post
W_BASE = 28         # baseline window length before the recent window


def momentum(times_sorted, t, w_recent=W_RECENT, w_base=W_BASE):
    """Momentum of a token at time t given sorted earlier-use times (days)."""
    lo_r = t - w_recent
    lo_b = t - w_recent - w_base
    i_t = np.searchsorted(times_sorted, t, "left")
    i_r = np.searchsorted(times_sorted, lo_r, "left")
    i_b = np.searchsorted(times_sorted, lo_b, "left")
    recent = i_t - i_r                       # uses in [t-7, t)
    base = i_r - i_b                          # uses in [t-35, t-7)
    base_rate = base / max(w_base, 1)         # per-day baseline
    return float(np.log1p(recent) - np.log1p(base_rate * w_recent))


def main():
    df = D.build()                            # all videos (use full history for context)
    df = df[df["create_dt"].notna()].copy()
    days = (df["create_dt"].astype("datetime64[ns]").astype("int64") / 86400e9).values   # days since epoch (float)
    df["_day"] = days
    vid = df["video_id"].astype(str).values

    # --- music momentum ---
    tm = np.zeros(len(df), np.float32)
    music = df["music_id"].astype(str).values
    order = np.argsort(music, kind="stable")
    music_s = music[order]; days_s = days[order]
    # boundaries of equal music groups
    bounds = np.r_[0, np.where(music_s[1:] != music_s[:-1])[0] + 1, len(music_s)]
    for gi in range(len(bounds) - 1):
        sl = slice(bounds[gi], bounds[gi + 1])
        idx = order[sl]
        gtimes = np.sort(days[idx])
        for k in idx:
            tm[k] = momentum(gtimes, days[k]) if len(gtimes) > 1 else 0.0

    # --- hashtag momentum ---
    def hnames(x):
        try:
            return [h["hashtag_name"] for h in x] if x is not None and len(x) > 0 else []
        except Exception:
            return []
    htags = df["hashtags"].map(hnames).values
    # build per-hashtag sorted time arrays
    tag_times = {}
    for tags, dday in zip(htags, days):
        for h in tags:
            tag_times.setdefault(h, []).append(dday)
    tag_times = {h: np.sort(np.array(v)) for h, v in tag_times.items() if len(v) > 1}
    th = np.zeros(len(df), np.float32)
    for i, tags in enumerate(htags):
        ms = [momentum(tag_times[h], days[i]) for h in tags if h in tag_times]
        th[i] = float(np.mean(ms)) if ms else 0.0

    out = pd.DataFrame({"video_id": vid, "trend_music": tm, "trend_hashtag": th})
    out.to_parquet(C.PROC / "trend_features.parquet", index=False)
    print(f"trend features {out.shape} -> {C.PROC/'trend_features.parquet'}")
    print(out[["trend_music", "trend_hashtag"]].describe().round(3))


if __name__ == "__main__":
    main()
