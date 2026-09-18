"""Self-learning flywheel: track posted videos by URL, poll their public stats, self-label at
label maturity, and periodically retrain a challenger model — all local, no human annotation.

Design (validated against external sources — see reports/flywheel_design.md):
  - track(url): extract content features once (caption/transcript/summary), snapshot the live
    counters, and estimate the creator's baseline = median views of their ~15 recent videos.
  - poll(): refresh counters for tracked videos (2x/day cadence is the caller's job; we enforce a
    minimum gap). At age >= 1 day the day-1 counters are frozen (Model B training signal). At age >=
    MATURITY_DAYS the within-creator label is computed: views_now > creator_median -> 1 else 0.
    Delayed-feedback rule (Ktena et al. 2019): rows never train before their label matures.
  - retrain_challenger(): with >= MIN_NEW matured rows, refit the text Model A on base + new rows and
    report holdout metrics next to the champion. NO auto-promotion (champion/challenger discipline):
    a human promotes by replacing models/deployable.joblib after reading the report.
  - selection-bias guard (feedback-loop literature): if the tracked pool is > BIAS_MAX_POS positive
    or negative, status() raises a warning — track ordinary videos, not only trending ones.

Storage: SQLite at data/flywheel.db (tables: videos, snapshots, baselines).
Nothing here raises on network/extractor failures — every step degrades and records status.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

from . import config as C

DB = C.ROOT / "data" / "flywheel.db"
MATURITY_DAYS = int(os.environ.get("SIP_FLYWHEEL_MATURITY", C.H))       # label horizon (14)
MIN_POLL_GAP_H = 6                       # do not hammer: min hours between snapshots per video
BASELINE_N = 15                          # creator recent videos for the median baseline
MIN_NEW = int(os.environ.get("SIP_FLYWHEEL_MIN_NEW", 25))               # rows required to retrain
BIAS_MAX_SHARE = 0.8                     # warn if labeled pool is >80% one class


# ---------------------------------------------------------------- store
def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(DB)
    cx.execute("""CREATE TABLE IF NOT EXISTS videos(
        video_id TEXT PRIMARY KEY, url TEXT, creator TEXT, caption TEXT, extra_text TEXT,
        duration_s REAL, aspect REAL, upload_date TEXT, added_at TEXT,
        status TEXT DEFAULT 'tracking',            -- tracking | labeled | failed
        day1_views REAL, day1_likes REAL, label INTEGER, labeled_at TEXT, note TEXT)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT, ts TEXT,
        views REAL, likes REAL, comments REAL)""")
    cx.execute("""CREATE TABLE IF NOT EXISTS baselines(
        creator TEXT PRIMARY KEY, median_views REAL, n INTEGER, updated_at TEXT)""")
    return cx


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _meta(url: str) -> dict | None:
    """Metadata-only fetch: live counters + uploader + upload_date. None on failure."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                               "socket_timeout": 20}) as ydl:
            m = ydl.extract_info(url, download=False)
        return {"id": str(m.get("id")), "views": m.get("view_count"), "likes": m.get("like_count"),
                "comments": m.get("comment_count"), "creator": m.get("uploader") or m.get("channel"),
                "upload_date": m.get("upload_date"), "caption": m.get("description") or m.get("title") or "",
                "duration": m.get("duration"), "width": m.get("width"), "height": m.get("height")}
    except Exception:
        return None


def _creator_baseline(creator: str) -> tuple[float, int] | None:
    """Median views of the creator's recent videos (the within-creator bar). None on failure."""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                               "extract_flat": True, "playlistend": BASELINE_N,
                               "socket_timeout": 25}) as ydl:
            m = ydl.extract_info(f"https://www.tiktok.com/@{creator}", download=False)
        views = [e.get("view_count") for e in (m.get("entries") or []) if e.get("view_count")]
        if len(views) < 5:
            return None
        return float(median(views)), len(views)
    except Exception:
        return None


# ---------------------------------------------------------------- public API
def track(url: str, extract_content: bool = True, language: str | None = None) -> dict:
    """Start tracking a posted video: content features once + snapshot0 + creator baseline."""
    meta = _meta(url)
    if not meta or not meta.get("id"):
        return {"ok": False, "error": "could not fetch video metadata (rate-limit or bad URL)"}
    cx = _conn()
    vid = meta["id"]
    if cx.execute("SELECT 1 FROM videos WHERE video_id=?", (vid,)).fetchone():
        cx.close()
        return {"ok": True, "video_id": vid, "note": "already tracked"}

    extra_text = ""
    note = ""
    if extract_content:
        try:
            from . import extract as EX
            inp = EX.extract(url, language=language)          # downloads + whisper + llm (best-effort)
            extra_text = inp.get("extra_text") or ""
        except Exception as e:
            note = f"content extraction skipped: {type(e).__name__}"

    creator = meta.get("creator") or ""
    base = _creator_baseline(creator) if creator else None
    if base:
        cx.execute("INSERT OR REPLACE INTO baselines VALUES(?,?,?,?)",
                   (creator, base[0], base[1], _now()))
    aspect = (meta["width"] / meta["height"]) if (meta.get("width") and meta.get("height")) else None
    cx.execute("""INSERT INTO videos(video_id,url,creator,caption,extra_text,duration_s,aspect,
                  upload_date,added_at,note) VALUES(?,?,?,?,?,?,?,?,?,?)""",
               (vid, url, creator, meta["caption"], extra_text, meta.get("duration"), aspect,
                meta.get("upload_date"), _now(), note))
    cx.execute("INSERT INTO snapshots(video_id,ts,views,likes,comments) VALUES(?,?,?,?,?)",
               (vid, _now(), meta.get("views"), meta.get("likes"), meta.get("comments")))
    cx.commit(); cx.close()
    return {"ok": True, "video_id": vid, "creator": creator,
            "views_now": meta.get("views"),
            "creator_median_views": base[0] if base else None,
            "baseline_n": base[1] if base else 0,
            "transcript_extracted": bool(extra_text), "note": note or None}


def _age_days(row) -> float:
    """Video age in days: from upload_date when known, else from added_at."""
    up, added = row
    try:
        if up:
            d0 = datetime.strptime(up, "%Y%m%d").replace(tzinfo=timezone.utc)
        else:
            d0 = datetime.fromisoformat(added)
    except Exception:
        d0 = datetime.fromisoformat(added)
    return (datetime.now(timezone.utc) - d0).total_seconds() / 86400.0


def poll(max_videos: int = 20, sleep_s: float = 2.0) -> dict:
    """Refresh counters for tracked videos (respecting MIN_POLL_GAP_H), freeze day-1, label matured."""
    cx = _conn()
    rows = cx.execute("""SELECT video_id,url,creator,upload_date,added_at,day1_views
                         FROM videos WHERE status='tracking' LIMIT ?""", (max_videos,)).fetchall()
    out = {"polled": 0, "skipped": 0, "labeled": 0, "failed": 0}
    for vid, url, creator, up, added, day1 in rows:
        last = cx.execute("SELECT MAX(ts) FROM snapshots WHERE video_id=?", (vid,)).fetchone()[0]
        if last and (datetime.now(timezone.utc) - datetime.fromisoformat(last)
                     ).total_seconds() < MIN_POLL_GAP_H * 3600:
            out["skipped"] += 1
            continue
        meta = _meta(url)
        if not meta:
            out["failed"] += 1
            continue
        cx.execute("INSERT INTO snapshots(video_id,ts,views,likes,comments) VALUES(?,?,?,?,?)",
                   (vid, _now(), meta.get("views"), meta.get("likes"), meta.get("comments")))
        age = _age_days((up, added))
        if day1 is None and age >= 1.0:                       # freeze the day-1 signal (Model B row)
            cx.execute("UPDATE videos SET day1_views=?, day1_likes=? WHERE video_id=?",
                       (meta.get("views"), meta.get("likes"), vid))
        if age >= MATURITY_DAYS:                              # label maturity (delayed feedback rule)
            b = cx.execute("SELECT median_views FROM baselines WHERE creator=?",
                           (creator,)).fetchone()
            if b and meta.get("views") is not None:
                label = int(float(meta["views"]) > float(b[0]))
                cx.execute("""UPDATE videos SET status='labeled', label=?, labeled_at=?
                              WHERE video_id=?""", (label, _now(), vid))
                out["labeled"] += 1
            else:
                cx.execute("UPDATE videos SET note='matured but no baseline' WHERE video_id=?", (vid,))
        out["polled"] += 1
        time.sleep(sleep_s)                                   # polite pacing
    cx.commit(); cx.close()
    return out


def status() -> dict:
    """Pool counts + bias guard + drift hint. Cheap, no network."""
    cx = _conn()
    n = dict(cx.execute("SELECT status, COUNT(*) FROM videos GROUP BY status").fetchall())
    lab = cx.execute("SELECT label, COUNT(*) FROM videos WHERE label IS NOT NULL GROUP BY label").fetchall()
    labels = {int(k): v for k, v in lab}
    pos, neg = labels.get(1, 0), labels.get(0, 0)
    warn = None
    tot = pos + neg
    if tot >= 10 and max(pos, neg) / tot > BIAS_MAX_SHARE:
        warn = (f"selection bias: {max(pos,neg)}/{tot} labels are one class — track ordinary "
                f"videos too, not only trending ones (feedback-loop risk)")
    vids = cx.execute("""SELECT video_id, creator, status, label, day1_views, added_at
                         FROM videos ORDER BY added_at DESC LIMIT 25""").fetchall()
    cx.close()
    return {"counts": n, "labels": {"hit": pos, "flop": neg},
            "ready_to_retrain": tot >= MIN_NEW, "min_new": MIN_NEW,
            "bias_warning": warn,
            "recent": [{"video_id": v, "creator": c, "status": s, "label": l,
                        "day1_views": d1, "added_at": a} for v, c, s, l, d1, a in vids]}


def tracked_day1(url: str) -> dict | None:
    """The GENUINE day-1 counters captured by polling (frozen at age>=1d). None if the video is not
    tracked or has not reached the 24h freeze yet. This is the only accurate day-1 source for an
    already-posted video — the live URL only exposes current cumulative counts."""
    cx = _conn()
    row = cx.execute("SELECT day1_views, day1_likes FROM videos WHERE url=? AND day1_views IS NOT NULL",
                     (url,)).fetchone()
    cx.close()
    if not row:
        return None
    return {"play_count": float(row[0]), "like_count": float(row[1] or 0)}


def _new_rows_frame():
    """Matured rows -> a DataFrame matching the text Model A training schema."""
    import pandas as pd
    cx = _conn()
    df = pd.read_sql_query("""SELECT video_id, creator AS author_id, caption AS desc,
                              extra_text, duration_s AS duration, aspect AS ratio,
                              upload_date, label AS y_breakout_wc
                              FROM videos WHERE status='labeled'""", cx)
    cx.close()
    if df.empty:
        return df
    df["create_dt"] = pd.to_datetime(df["upload_date"], format="%Y%m%d", errors="coerce",
                                     utc=True).fillna(pd.Timestamp.now(tz="UTC"))
    df["is_english"] = True                                    # unknown -> neutral defaults
    df["created_by_ai"] = False
    df["is_ads"] = False
    return df


def retrain_challenger() -> dict:
    """Refit the text Model A on base data + matured flywheel rows; write a challenger report.
    NEVER swaps the champion — promotion is a human decision after reading the report."""
    import joblib
    import pandas as pd
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from . import data as D, features as F, splits as S
    from .modeling import make_model, Calibrated

    new = _new_rows_frame()
    if len(new) < MIN_NEW:
        return {"ok": False, "reason": f"insufficient matured rows: {len(new)} < {MIN_NEW}"}

    base_df = D.load(eligible_only=True)
    F.add_derived(base_df)
    tr, va, te = S.temporal_masks(base_df)
    y_base = base_df["y_breakout_wc"].astype(int).values

    blocks = ["caption", "timing", "duration", "meta", "text_emb:bge"]
    champ = joblib.load(C.MODELS / "deployable.joblib")
    Xbase, names = F.build_blocks(base_df, blocks, np.where(tr)[0], y_base)
    assert list(names) == list(champ["A"]["names"]), "feature order drifted vs champion"

    # champion metrics on the fixed temporal test — the comparison anchor
    y_te = y_base[te]
    p_champ = champ["A"]["cal"].predict_proba_pos(Xbase[te])
    champ_auc = roc_auc_score(y_te, p_champ)

    # new rows go through the LIVE inference path (same caption stats + same BGE on caption+extra_text),
    # which guarantees identical feature semantics for rows that are not in the precomputed parquet.
    from . import inference as INF
    X_new = np.vstack([
        INF._feature_vector({"caption": r["desc"], "duration_s": r["duration"],
                             "aspect": r["ratio"], "post_time": str(r["create_dt"]),
                             "extra_text": r["extra_text"]})[0]
        for _, r in new.iterrows()])
    y_new = new["y_breakout_wc"].astype(int).values

    # challenger: train on base-train + ALL matured new rows, calibrate on base-valid
    Xc = np.vstack([Xbase[tr], X_new])
    y_comb = np.concatenate([y_base[tr], y_new])
    model = make_model("hgb").fit(Xc, y_comb)
    cal = Calibrated(model, method="isotonic").fit_calibration(Xbase[va], y_base[va])
    p_chal = cal.predict_proba_pos(Xbase[te])
    chal_auc = roc_auc_score(y_te, p_chal)

    report = {
        "ts": _now(), "n_new_rows": int(len(new)),
        "champion": {"auc_temporal_test": round(float(champ_auc), 4)},
        "challenger": {"auc_temporal_test": round(float(chal_auc), 4),
                       "brier": round(float(brier_score_loss(y_te, p_chal)), 4)},
        "verdict": ("challenger >= champion — consider promotion (manual)"
                    if chal_auc >= champ_auc else "challenger worse — keep champion"),
        "promotion": "manual: replace models/deployable.joblib only after review",
    }
    joblib.dump({"model": model, "cal": cal, "names": names, "blocks": blocks,
                 "trained_at": _now(), "n_new": int(len(new))},
                C.MODELS / "challenger.joblib")
    (C.REPORTS / "flywheel_challenger.json").write_text(json.dumps(report, indent=1))
    return {"ok": True, **report}
