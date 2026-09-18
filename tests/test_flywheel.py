"""Flywheel logic tests — no network: fake metadata via monkeypatch, temp DB per test.

Run:  PYTHONPATH=src pytest tests/test_flywheel.py -q
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import flywheel as FW  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(FW, "DB", tmp_path / "fw.db")
    return tmp_path / "fw.db"


def _fake_meta(views=1000, likes=50, creator="alice", days_old=0):
    up = (datetime.now(timezone.utc) - timedelta(days=days_old)).strftime("%Y%m%d")
    return {"id": "v1", "views": views, "likes": likes, "comments": 3, "creator": creator,
            "upload_date": up, "caption": "hi #fyp", "duration": 15, "width": 9, "height": 16}


def test_track_stores_video_baseline_and_snapshot(tmp_db, monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: _fake_meta())
    monkeypatch.setattr(FW, "_creator_baseline", lambda c: (500.0, 15))
    r = FW.track("https://x/video/1", extract_content=False)
    assert r["ok"] and r["video_id"] == "v1" and r["creator_median_views"] == 500.0
    st = FW.status()
    assert st["counts"].get("tracking") == 1
    # duplicate tracking is idempotent
    assert FW.track("https://x/video/1", extract_content=False)["note"] == "already tracked"


def test_poll_freezes_day1_and_labels_at_maturity(tmp_db, monkeypatch):
    # video already MATURITY_DAYS old -> one poll should label it against the baseline
    monkeypatch.setattr(FW, "_meta", lambda url: _fake_meta(views=900, days_old=FW.MATURITY_DAYS + 1))
    monkeypatch.setattr(FW, "_creator_baseline", lambda c: (500.0, 15))
    FW.track("https://x/video/1", extract_content=False)
    monkeypatch.setattr(FW, "MIN_POLL_GAP_H", 0)              # allow immediate re-poll in test
    out = FW.poll(sleep_s=0)
    assert out["labeled"] == 1
    st = FW.status()
    assert st["labels"]["hit"] == 1                            # 900 > 500 -> beats own median
    row = st["recent"][0]
    assert row["status"] == "labeled" and row["day1_views"] == 900


def test_flop_label_when_below_creator_median(tmp_db, monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: _fake_meta(views=100, days_old=FW.MATURITY_DAYS + 1))
    monkeypatch.setattr(FW, "_creator_baseline", lambda c: (500.0, 15))
    FW.track("https://x/video/1", extract_content=False)
    monkeypatch.setattr(FW, "MIN_POLL_GAP_H", 0)
    FW.poll(sleep_s=0)
    assert FW.status()["labels"]["flop"] == 1


def test_poll_respects_min_gap(tmp_db, monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: _fake_meta())
    monkeypatch.setattr(FW, "_creator_baseline", lambda c: (500.0, 15))
    FW.track("https://x/video/1", extract_content=False)
    out = FW.poll(sleep_s=0)                                   # snapshot0 is fresh -> must skip
    assert out["skipped"] == 1 and out["polled"] == 0


def test_retrain_refuses_without_enough_rows(tmp_db):
    r = FW.retrain_challenger()
    assert r["ok"] is False and "insufficient" in r["reason"]


def test_bias_warning_on_one_sided_pool(tmp_db, monkeypatch):
    import sqlite3
    cx = FW._conn()
    for i in range(12):                                        # 12 hits, 0 flops
        cx.execute("INSERT INTO videos(video_id,creator,status,label,added_at) VALUES(?,?,?,?,?)",
                   (f"v{i}", "a", "labeled", 1, FW._now()))
    cx.commit(); cx.close()
    assert FW.status()["bias_warning"] is not None
