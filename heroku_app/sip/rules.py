"""Transparent rule-based ("expert system") scorer — a heuristic baseline that needs
no training, encoding the domain knowledge from the literature review (hook, CTA,
3-5 niche hashtags not #fyp spam, watchable duration, non-empty readable caption,
no off-platform URL). Mirrors commercial "viral score 0-100" tools, but is here a
HONEST baseline the learned model must beat.

Operates on the derived columns produced by sip.features.add_derived(). Exposes a
sklearn-like predict_proba so it plugs into sip.eval / the inference contract.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class RuleBasedScorer:
    """Weighted, interpretable rules -> a 0..1 'post-worthiness' score. Direction of each
    rule is domain-justified (and matches the train correlations we measured)."""

    def __init__(self):
        self.rules = [
            ("question hook", lambda d: d["has_question"], +0.30),
            ("call-to-action", lambda d: d["has_cta"], +0.25),
            ("3-5 hashtags (niche)", lambda d: ((d["n_hashtags"] >= 3) & (d["n_hashtags"] <= 6)).astype(float), +0.20),
            ("no hashtags", lambda d: (d["n_hashtags"] == 0).astype(float), -0.20),
            ("hashtag spam (>10)", lambda d: (d["n_hashtags"] > 10).astype(float), -0.25),
            ("watchable duration 7-30s", lambda d: ((d["duration_s"] >= 7) & (d["duration_s"] <= 30)).astype(float), +0.25),
            ("too long (>90s)", lambda d: (d["duration_s"] > 90).astype(float), -0.25),
            ("caption length 20-150", lambda d: ((d["char_len"] >= 20) & (d["char_len"] <= 150)).astype(float), +0.15),
            ("empty caption", lambda d: d["caption_is_empty"], -0.35),
            ("moderate emoji 1-5", lambda d: ((d["n_emoji"] >= 1) & (d["n_emoji"] <= 5)).astype(float), +0.10),
            ("ALL-CAPS spam", lambda d: (d["allcaps_ratio"] > 0.5).astype(float), -0.20),
            ("off-platform URL", lambda d: d["has_url"], -0.20),
            ("evening/weekend post", lambda d: d["is_weekend"], +0.05),
        ]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        s = np.zeros(len(df), dtype=float)
        for _, fn, w in self.rules:
            s = s + w * np.asarray(fn(df), dtype=float)
        return _sig(s)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.score(df)
        return np.c_[1 - p, p]

    def explain(self, row: pd.Series):
        """Per-candidate fired rules (for UI/debug)."""
        out = []
        for name, fn, w in self.rules:
            v = float(np.asarray(fn(pd.DataFrame([row])))[0])
            if v:
                out.append({"rule": name, "weight": w, "fired": True})
        return sorted(out, key=lambda r: -abs(r["weight"]))
