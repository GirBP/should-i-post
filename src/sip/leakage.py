"""Automated leakage guards. A single leak invalidates a result, so these are
asserted in code (and exercised by the test suite).

Rules enforced:
  * Model A feature matrices may contain ONLY pre-publication content + creator
    state as of create_date.  No engagement counters, no day-1 signals.
  * Model B may add day<=1 signals (clearly named *_d1) but never day>1 / horizon
    counters that overlap the label window.
  * Fold-safe fitting: encoders / target-encoders / scalers fit on train folds
    only (enforced by construction in sip.features + sip.experiment; this module
    provides a runtime check helper for tests).
"""
from __future__ import annotations
from . import config as C

# Tokens that mark a post-publication / horizon signal forbidden in Model A.
_FORBIDDEN_A = tuple(t for t in C.ENGAGEMENT_TOKENS)
# Day-1 signals Model B is allowed to use.
_ALLOWED_B_DAY1 = ("log_play_d1", "log_like_d1", "er_d1", "log_play_d1_vs_author")
# Horizon/label columns forbidden everywhere as features.
_FORBIDDEN_LABEL = ("breakout_wc", "er_wc", "follgrow", "views_at_H",
                    "y_breakout_wc", "y_er_wc", "y_follgrow")


class LeakageError(AssertionError):
    pass


def assert_model_a(feature_names):
    """Raise if a Model-A feature set contains any post-publication signal."""
    bad = []
    for f in feature_names:
        lf = str(f).lower()
        if any(tok in lf for tok in _FORBIDDEN_A):
            bad.append(f)
        if f in _FORBIDDEN_LABEL:
            bad.append(f)
    if bad:
        raise LeakageError(f"Model A leak: forbidden post-publication features {sorted(set(bad))}")
    return True


def assert_model_b(feature_names):
    """Raise if a Model-B feature set uses a horizon counter (day>1) or a label."""
    bad = []
    for f in feature_names:
        if f in _FORBIDDEN_LABEL:
            bad.append(f); continue
        lf = str(f).lower()
        # allow the whitelisted day-1 signals; flag any other engagement counter
        if f in _ALLOWED_B_DAY1:
            continue
        if any(tok in lf for tok in ("play_count", "like_count", "comment_count",
                                     "share_count", "collect_count", "download_count",
                                     "whatsapp_share_count")):
            bad.append(f)
    if bad:
        raise LeakageError(f"Model B leak: horizon/label features {sorted(set(bad))}")
    return True


def assert_disjoint_authors(train_authors, test_authors):
    """LOCO sanity: no author appears in both train and test."""
    overlap = set(train_authors) & set(test_authors)
    if overlap:
        raise LeakageError(f"LOCO leak: {len(overlap)} authors in both train and test")
    return True
