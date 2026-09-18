"""Day-1 handling in URL mode — server helpers, no network (FW._meta monkeypatched).

Contract under test (webapp/server.py, unified classifier era):
  - day-1 is STRICTLY opt-in: no auto-activation exists any more (_auto_day1 removed);
    the UI checkbox probes /api/probe and sends the counts explicitly.
  - /api/probe returns age + counters and flags the ~1-day window where current
    counts are a valid day-1 proxy (auto_ok).
  - manual day-1 on an old URL -> age guard warns (counts are cumulative, not 24h).

Run:  PYTHONPATH=src pytest tests/test_auto_day1.py -q
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from webapp import server as S  # noqa: E402
from sip import flywheel as FW  # noqa: E402


def _ymd(days_ago: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d")


def test_age_days_parses():
    assert 9.0 <= S._age_days(_ymd(10)) <= 11.0   # tolerance: age carries a day-fraction + UTC/local offset
    assert S._age_days(None) is None
    assert S._age_days("garbage") is None


def test_no_auto_activation_helper_exists():
    """Day-1 must be opt-in only: the auto-activation helper is gone by design."""
    assert not hasattr(S, "_auto_day1")


def test_probe_flags_one_day_window(monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: {"views": 5000, "likes": 200, "upload_date": _ymd(1)})
    r = S.probe({"url": "https://x/v/2"})
    assert r["ok"] and r["over_one_day"] and r["auto_ok"]
    assert r["views"] == 5000 and r["likes"] == 200


def test_probe_old_video_not_auto_ok(monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: {"views": 999999, "likes": 5000, "upload_date": _ymd(619)})
    r = S.probe({"url": "https://x/v/3"})
    assert r["over_one_day"] and not r["auto_ok"]   # cumulative counts != day-1


def test_probe_fresh_video_not_auto_ok(monkeypatch):
    monkeypatch.setattr(FW, "_meta", lambda url: {"views": 100, "likes": 5, "upload_date": _ymd(0)})
    r = S.probe({"url": "https://x/v/4"})
    assert not r["over_one_day"] and not r["auto_ok"]  # day-1 not matured yet


def test_day_one_mapping():
    assert S._day_one("8000", "600") == {"play_count": 8000.0, "like_count": 600.0}
    assert S._day_one(None, "600") is None            # no plays -> no day-1 at all
    assert S._day_one("0", "600") is None


def test_manual_day1_on_old_url_warns():
    w = S._day1_age_guard({"play_count": 900000, "like_count": 50000}, _ymd(619))
    assert w and "24h" in w


def test_no_guard_on_fresh_or_missing_date():
    assert S._day1_age_guard({"play_count": 1}, _ymd(1)) is None
    assert S._day1_age_guard(None, _ymd(619)) is None
