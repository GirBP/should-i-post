"""Regression test: the deployed A->B model must beat the trivial baseline and stay
calibrated. Reads reports/deployable_metrics.json (fast; skipped if not built)."""
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C  # noqa: E402

_P = C.REPORTS / "deployable_metrics.json"
pytestmark = pytest.mark.skipif(not _P.exists(), reason="deployable_metrics.json not built")


def _m():
    return json.load(open(_P))


def test_model_a_beats_baseline():
    a = _m()["A"]["metrics"]
    assert a["roc_auc"] > 0.53            # beats majority/caption baseline (~0.50-0.51)
    assert a["brier"] < 0.25              # calibrated, better than the 0.25 constant


def test_model_b_beats_model_a():
    # honest day-1 model (day1_basic features only, leak-free): better than pre-publication A,
    # but not the inflated ~0.95 that relied on a leaky author-relative feature (see AUDIT.md)
    m = _m()
    assert m["B"]["metrics"]["roc_auc"] > 0.70
    assert m["B"]["metrics"]["roc_auc"] > m["A"]["metrics"]["roc_auc"]
    assert m["B"]["metrics"]["brier"] < 0.23


def test_bands_valid():
    for k in ("A", "B"):
        bands = _m()[k]["bands"]
        assert 0.0 <= bands["t_low"] <= bands["t_high"] <= 1.0
