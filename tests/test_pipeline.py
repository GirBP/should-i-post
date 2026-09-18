"""Core pipeline tests: label freeze, splits, feature determinism, fold-safety.

Run:  PYTHONPATH=src pytest tests/ -q
These use a small synthetic frame so they run without the dataset.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import features as F, splits as S, eval as E, data as D  # noqa: E402


def synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    authors = [f"a{i}" for i in range(40)]
    base = pd.to_datetime("2024-06-01")
    df = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n)],
        "author_id": rng.choice(authors, n),
        "create_dt": base + pd.to_timedelta(rng.integers(0, 150, n), unit="D"),
        "desc": rng.choice(["hello #fun follow me!", "what is this?", "", "buy now www.x.com 😀😀"], n),
        "duration": rng.integers(5, 90, n).astype(float),
        "ratio": rng.choice([0.56, 1.0], n),
        "is_english": rng.choice([True, False], n),
        "created_by_ai": False, "is_ads": False,
        "topic": rng.choice(["A", "B", "C"], n),
        "music_id": rng.choice(["m1", "m2", "m3"], n),
        "hashtags": [[{"hashtag_name": h} for h in rng.choice(["x", "y", "z"], 2)] for _ in range(n)],
        "transcript": "blah blah", "gpt_summary": "a video about things", "speaking_rate": rng.random(n),
        "anger": rng.random(n), "joy": rng.random(n), "surprise": rng.random(n),
        "sadness": rng.random(n), "disgust": rng.random(n), "fear": rng.random(n),
        "y_breakout_wc": rng.integers(0, 2, n),
        "log_play_d1": rng.random(n), "log_like_d1": rng.random(n),
        "er_d1": rng.random(n), "log_play_d1_vs_author": rng.random(n),
    })
    return df


# ---------------------------------------------------------------- leakage
def test_leakage_guard_blocks_day1_in_model_a():
    from sip.leakage import assert_model_a, LeakageError
    with pytest.raises(LeakageError):
        assert_model_a(["char_len", "log_play_d1"])
    with pytest.raises(LeakageError):
        assert_model_a(["er_d1"])
    assert assert_model_a(["char_len", "duration_s", "bge0"]) is True


def test_leakage_guard_blocks_horizon_in_model_b():
    from sip.leakage import assert_model_b, LeakageError
    assert assert_model_b(["log_play_d1", "er_d1", "char_len"]) is True  # day-1 ok
    with pytest.raises(LeakageError):
        assert_model_b(["play_count"])                                   # horizon counter
    with pytest.raises(LeakageError):
        assert_model_b(["views_at_H"])                                   # label component


def test_loco_authors_disjoint():
    df = synth()
    for tri, tei in S.loco_folds(df):
        assert set(df["author_id"].values[tri]).isdisjoint(df["author_id"].values[tei])


# ---------------------------------------------------------------- determinism
def test_feature_build_deterministic():
    df = synth()
    y = df["y_breakout_wc"].values
    tri = np.arange(300)
    X1, n1 = F.build_blocks(df.copy(), ["caption", "emotion", "timing"], tri, y)
    X2, n2 = F.build_blocks(df.copy(), ["caption", "emotion", "timing"], tri, y)
    assert n1 == n2
    np.testing.assert_allclose(X1, X2)


def test_target_encoding_is_fold_safe():
    """topic target-encoding for a test row must equal the TRAIN-only group mean,
    i.e. test labels must not influence the encoding."""
    df = synth()
    y = df["y_breakout_wc"].values
    tri = np.arange(300)
    X, names = F.build_blocks(df, ["topic"], tri, y)
    te_idx = names.index("topic_te")
    # recompute expected encoding from train only
    g = pd.Series(y[tri]).groupby(df["topic"].values[tri]).mean()
    gm = y[tri].mean()
    expected = df["topic"].map(g).fillna(gm).values
    np.testing.assert_allclose(X[:, te_idx], expected, rtol=1e-5)


def test_creator_fit_closed_window_no_lookahead():
    """A prior winner must be INVISIBLE to creator_fit until its H-day label is observable
    (prior.create_dt + H <= now). This guards the lookahead leak fixed after the audit:
    v1 posted < H days after a winning v0 must see 0 prior wins; v2 posted > H days later sees it."""
    from sip import config as C
    base = pd.to_datetime("2024-06-01")
    df = pd.DataFrame({
        "video_id": ["v0", "v1", "v2"], "author_id": ["a", "a", "a"],
        "create_dt": [base, base + pd.Timedelta(days=5), base + pd.Timedelta(days=C.H + 10)],
        "desc": ["dance tutorial routine", "dance tutorial routine", "cooking pasta recipe"],
        "gpt_summary": ["", "", ""], "transcript": ["", "", ""],
        "y_breakout_wc": [1, 0, 0],     # v0 is a winner
    })
    y = df["y_breakout_wc"].values
    X, names = F.block_creator_fit(df, np.arange(len(df)), y)   # call block directly (skip add_derived)
    fit = X[:, names.index("creator_fit")]; nprior = X[:, names.index("n_prior_wins_log")]
    assert fit[0] == 0 and nprior[0] == 0                 # first video: nothing prior
    assert nprior[1] == 0, "v1 (<H days after winner) must NOT see v0's label yet (no lookahead)"
    assert nprior[2] > 0, "v2 (>H days after winner) should see the now-observable prior winner"


# ---------------------------------------------------------------- metrics
def test_metrics_and_bands():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 500), 0, 1)
    m = E.metrics(y, p, n_boot=100)
    assert 0.5 <= m["roc_auc"] <= 1.0
    assert m["roc_auc_ci"][0] <= m["roc_auc"] <= m["roc_auc_ci"][1]
    bands = E.find_decision_bands(y, p, precision_target=0.6)
    assert bands["t_low"] <= bands["t_high"]
    rep = E.band_report(y, p, bands)
    assert 0 <= rep["abstain_coverage"] <= 1


def test_rule_based_scorer():
    """Rule-based expert system returns valid probabilities and ranks a clearly-good
    candidate above a clearly-bad one."""
    from sip.rules import RuleBasedScorer
    rs = RuleBasedScorer()
    good = pd.DataFrame([{"has_question": 1, "has_cta": 1, "n_hashtags": 4, "duration_s": 15,
                          "char_len": 60, "caption_is_empty": 0, "n_emoji": 2, "allcaps_ratio": 0.0,
                          "has_url": 0, "is_weekend": 1}])
    bad = pd.DataFrame([{"has_question": 0, "has_cta": 0, "n_hashtags": 0, "duration_s": 200,
                         "char_len": 0, "caption_is_empty": 1, "n_emoji": 0, "allcaps_ratio": 0.9,
                         "has_url": 1, "is_weekend": 0}])
    pg = rs.predict_proba(good); pb = rs.predict_proba(bad)
    assert pg.shape == (1, 2) and 0 <= pg[0, 1] <= 1
    np.testing.assert_allclose(pg.sum(1), 1.0, atol=1e-6)
    assert pg[0, 1] > pb[0, 1]                          # good ranked above bad


def test_label_definition_within_creator():
    """Within-creator binary: centring by creator median then global-median split."""
    val = pd.Series([1.0, 2, 3, 10, 20, 30])
    auth = pd.Series(["a", "a", "a", "b", "b", "b"])
    centered, label = D._within_creator_binary(val, auth)
    # creator medians removed -> a:{-1,0,1}, b:{-10,0,10}; global median of centered ~0
    assert label.tolist() == [0, 0, 1, 0, 0, 1]
