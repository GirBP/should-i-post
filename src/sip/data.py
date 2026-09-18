"""Raw lingbow (3 tables) -> one canonical analysis frame.

Produces, per video:
  * labels      y_breakout_wc (primary), y_er_wc (resonance), y_follgrow (aux)
                + their continuous values for diagnostics
  * eligibility elig (views@H >= MIN_PLAY and follower-at-post known)
  * day-1       log_play_d1, log_like_d1, er_d1, log_play_d1_vs_author  (Model B only)
  * splits      split_temporal in {train,valid,test}; author_id for LOCO
  * content     all pre-publication raw fields (text kept raw for fold-safe encoders)

Leakage discipline: engagement counters never enter as content; the only
post-publication signals are the explicit day<=1 features (clearly named *_d1),
and the label uses day-H counters strictly after all features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C


def _follower_asof(videos: pd.DataFrame, creator: pd.DataFrame, when: pd.Series) -> pd.Series:
    """follower_count for each video's author as of `when` (backward merge_asof)."""
    L = pd.DataFrame({
        "video_id": videos["video_id"].values,
        "author_id": videos["author_id"].values,
        "_w": pd.to_datetime(when, errors="coerce").values,
    }).dropna(subset=["_w"]).sort_values("_w")
    R = creator.assign(date=pd.to_datetime(creator["date"], errors="coerce"))
    R = R[["author_id", "date", "follower_count"]].dropna().sort_values("date")
    m = pd.merge_asof(L, R, left_on="_w", right_on="date", by="author_id", direction="backward")
    return pd.Series(m["follower_count"].values, index=m["video_id"]).reindex(videos["video_id"].values)


def _within_creator_binary(value: pd.Series, author: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Centre a per-video value by the creator's own median, binarize at global median.

    Returns (centered_value, binary_label). The per-creator median defines the
    creator's personal baseline (a label normalisation, not a feature).
    """
    centered = value - value.groupby(author.values).transform("median")
    label = (centered > centered.median()).astype("int8")
    return centered, label


def build(force: bool = False) -> pd.DataFrame:
    """Build (and cache) the canonical frame. Idempotent unless force=True."""
    if C.CANONICAL.exists() and not force:
        return pd.read_parquet(C.CANONICAL)

    v = pd.read_parquet(C.RAW / "videos.parquet", columns=C.VIDEO_KEEP)
    v["video_id"] = v["video_id"].astype(str)
    v["author_id"] = v["author_id"].astype(str)
    v["create_dt"] = pd.to_datetime(v["create_time"], errors="coerce")

    eng_cols = ["video_id", "days_since_post", "play_count", "like_count",
                "comment_count", "share_count", "collect_count"]
    e = pd.read_parquet(C.RAW / "engagement_daily.parquet", columns=eng_cols)
    e["video_id"] = e["video_id"].astype(str)
    e = e.sort_values(["video_id", "days_since_post"])

    c = pd.read_parquet(C.RAW / "creator_daily.parquet",
                        columns=["author_id", "date", "follower_count"])
    c["author_id"] = c["author_id"].astype(str)

    ENG = ["like_count", "comment_count", "share_count", "collect_count"]

    # --- cumulative counters at horizon H (last row with days_since_post <= H) ---
    cumH = e[e.days_since_post <= C.H].groupby("video_id").tail(1).set_index("video_id")
    playH = cumH["play_count"].reindex(v["video_id"].values)
    engH = cumH[ENG].sum(axis=1).reindex(v["video_id"].values)

    # --- follower as of post, and at post+H (for follower growth) ---
    foll0 = _follower_asof(v, c, v["create_dt"])
    follH = _follower_asof(v, c, v["create_dt"] + pd.to_timedelta(C.H, unit="D"))
    foll0 = pd.Series(foll0.values, index=v.index)
    follH = pd.Series(follH.values, index=v.index)
    playH = pd.Series(playH.values, index=v.index)
    engH = pd.Series(engH.values, index=v.index)

    out = v.copy()

    # ---- primary label: breakout within-creator (views@H / followers@post) ----
    breakout = playH / foll0.clip(lower=1)
    out["breakout_wc"], out["y_breakout_wc"] = _within_creator_binary(breakout, out["author_id"])

    # ---- secondary: within-creator engagement-rate (resonance) ----
    er = engH / playH.clip(lower=1)
    out["er_wc"], out["y_er_wc"] = _within_creator_binary(er, out["author_id"])

    # ---- auxiliary: follower growth over H ----
    follgrow = (follH - foll0) / foll0.clip(lower=1)
    out["follgrow"], out["y_follgrow"] = _within_creator_binary(follgrow, out["author_id"])

    # ---- eligibility (floor on reach + known follower count) ----
    out["elig"] = ((playH >= C.MIN_PLAY) & foll0.notna()).astype("int8")
    out["followers_at_post"] = foll0.values
    out["views_at_H"] = playH.values

    # ---- day-1 features (Model B ONLY; never used by Model A) ----
    d1 = (e[e.days_since_post <= 1].sort_values(["video_id", "days_since_post"])
          .groupby("video_id").tail(1).set_index("video_id"))
    play_d1 = d1["play_count"].reindex(out["video_id"].values).values
    like_d1 = d1["like_count"].reindex(out["video_id"].values).values
    eng_d1 = d1[ENG].sum(axis=1).reindex(out["video_id"].values).values
    out["log_play_d1"] = np.log1p(np.nan_to_num(play_d1))
    out["log_like_d1"] = np.log1p(np.nan_to_num(like_d1))
    out["er_d1"] = np.divide(eng_d1, np.clip(play_d1, 1, None),
                             out=np.zeros_like(eng_d1, float), where=~np.isnan(play_d1))
    # day-1 reach relative to the author's own typical day-1 reach (fame-neutral)
    lp = pd.Series(out["log_play_d1"].values, index=out.index)
    out["log_play_d1_vs_author"] = (lp - lp.groupby(out["author_id"].values).transform("median")).values

    # ---- temporal split by create_date quantiles ----
    q1, q2 = out["create_dt"].quantile(C.TEMPORAL_Q[0]), out["create_dt"].quantile(C.TEMPORAL_Q[1])
    split = np.where(out["create_dt"] <= q1, "train",
            np.where(out["create_dt"] <= q2, "valid", "test"))
    out["split_temporal"] = split

    out.to_parquet(C.CANONICAL, index=False)
    return out


def load(eligible_only: bool = True) -> pd.DataFrame:
    df = build()
    if eligible_only:
        df = df[df["elig"] == 1].reset_index(drop=True)
    return df


def text_field(df: pd.DataFrame) -> pd.Series:
    """Concatenated free text for semantic encoders (caption + summary + transcript)."""
    parts = [df.get(c, "").fillna("").astype(str) for c in ("desc", "gpt_summary", "transcript")]
    return (parts[0] + " " + parts[1] + " " + parts[2]).str.strip()


if __name__ == "__main__":
    import time
    t = time.time()
    df = build(force=True)
    print(f"built canonical {df.shape} in {time.time()-t:.1f}s -> {C.CANONICAL}")
    elig = df[df.elig == 1]
    print(f"eligible: {len(elig):,} ({len(elig)/len(df):.1%})  authors={elig.author_id.nunique():,}")
    for t_ in ("y_breakout_wc", "y_er_wc", "y_follgrow"):
        print(f"  base rate {t_}: {elig[t_].mean():.3f}")
    print("temporal split:", df.split_temporal.value_counts().to_dict())
