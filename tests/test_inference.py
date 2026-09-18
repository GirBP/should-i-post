"""Inference robustness: the stateless predictor must degrade gracefully on
partial / invalid input and never crash. Skipped if the artifact is not built.

Run: SIP_DEVICE=cpu PYTHONPATH=src pytest tests/test_inference.py -q
"""
import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("SIP_DEVICE", "cpu")  # avoid MPS contention in CI
from sip import config as C  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (C.MODELS / "deployable.joblib").exists(),
    reason="deployable.joblib not built (run experiments/build_deployable.py)")

VALID = {"Post", "Do not post", "Unsure"}


@pytest.fixture(scope="module")
def inf():
    from sip import inference
    return inference


def _check(res):
    assert res["recommendation"] in VALID
    assert 0.0 <= res["probability"] <= 1.0
    assert isinstance(res["rationale"], str) and res["rationale"]
    assert "what_model_cannot_know" in res
    for f in res["factors"]:
        assert "note" in f


def test_full_caption(inf):
    _check(inf.predict({"caption": "wait for it 😱 #fyp", "duration_s": 15}))


def test_empty_caption_warns(inf):
    r = inf.predict({"caption": "", "duration_s": None})
    _check(r)
    assert any("caption" in w.lower() for w in r["warnings"])  # graceful warning


def test_missing_duration(inf):
    _check(inf.predict({"caption": "a normal caption about cooking"}))


def test_weird_input_no_crash(inf):
    _check(inf.predict({"caption": "🔥🔥🔥", "duration_s": 2, "post_time": "2024-08-01T20:00:00"}))
    _check(inf.predict({"caption": "x" * 5000, "duration_s": 999}))


def test_day_one_switches_to_model_b(inf):
    r = inf.predict({"caption": "dance #fyp", "duration_s": 12},
                    day_one={"play_count": 40000, "like_count": 5000, "comment_count": 200,
                             "share_count": 600, "collect_count": 300})
    _check(r)
    assert r["model"].startswith("B")


def test_examples_and_factors(inf):
    r = inf.predict({"caption": "wait for it 😱 storytime #fyp", "duration_s": 18})
    assert isinstance(r["examples"], list)            # example-based evidence present
    assert all("note" in f for f in r["factors"])     # per-prediction factors (SHAP or fallback)


def test_cost_ratio_shifts_threshold(inf):
    inp = {"caption": "a cooking video", "duration_s": 20}
    lenient = inf.predict(inp, cost_ratio=0.3)["bands"]["t_high"]
    strict = inf.predict(inp, cost_ratio=4.0)["bands"]["t_high"]
    assert strict > lenient                            # higher cost(false Post) -> more selective


def test_malformed_inputs_never_crash(inf):
    """Regression for the type-mismatch crashers found in app testing — must degrade, not raise."""
    bad_inps = [
        {"caption": 12345}, {"caption": ["a", "b"]},
        {"duration_s": "abc"}, {"aspect": "9:16"},
        {"post_time": "not-a-date"}, {"post_time": "1191191"}, {"post_time": 123456789},
        {"extra_text": 99}, {"caption": "ok", "duration_s": float("nan")},
    ]
    for inp in bad_inps:
        _check(inf.predict(inp))                        # contract holds, no exception
    # malformed day_one and cost_ratio must also be safe
    _check(inf.predict({"caption": "x"}, day_one={"play_count": "abc", "like_count": None}))
    for cr in ("abc", 0, -1, float("nan")):
        b = inf.predict({"caption": "x"}, cost_ratio=cr)["bands"]
        assert 0.0 <= b["t_low"] <= b["t_high"] <= 1.0  # bands stay valid
