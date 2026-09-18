"""Single source of truth for paths, label definition, splits and seeds.

Every constant the pipeline depends on lives here so that data prep, training,
evaluation and inference cannot silently disagree.
"""
from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "lingbow"
PROC = ROOT / "data" / "processed"
MM = ROOT / "data" / "multimodal"               # manifest + extracted features
VIDEOS = ROOT / "data" / "videos"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for _p in (PROC, REPORTS, MODELS, MM):
    _p.mkdir(parents=True, exist_ok=True)

CANONICAL = PROC / "canonical.parquet"          # built by sip.data.build()
EXPLOG = REPORTS / "experiments_log.md"

# ---------------------------------------------------------------- label
# Primary target: breakout within-creator at horizon H.
#   breakout = views@H / followers_at_post          (reach beyond own audience)
#   within-creator centering: subtract the creator's own median breakout, then
#   binarize at the global median of the centered value (~0).  This makes the
#   label fame-neutral (the creator's personal baseline is removed) and is the
#   exact definition behind the established §1 numbers, so results stay comparable.
# Note: the per-creator median is a *label normalisation* (defining what counts as
# success for that creator), not a predictive feature, so it is not a feature leak.
H = 14                                           # horizon in days (validate {7,10,14,21})
H_GRID = (7, 10, 14, 21)
MIN_PLAY = 50                                    # eligibility floor on views@H
MIN_VIDEOS = 50                                  # creators with >= own threshold else global

# ---------------------------------------------------------------- splits
# Temporal split by create_date quantiles (train<valid<test) ~ 70/15/15.
TEMPORAL_Q = (0.70, 0.85)
LOCO_FOLDS = 4                                   # GroupKFold by author for leave-one-creator-out
SEED = 0

# ---------------------------------------------------------------- decision policy
# Cost-based bands tuned on validation to hit a precision@Post target.
PRECISION_AT_POST_TARGET = 0.65
CONFORMAL_ALPHA = 0.10                           # target abstention error rate

# ---------------------------------------------------------------- columns
# Raw lingbow columns retained in the canonical frame (pre-publication content
# + ids + raw text for fold-safe encoders). Engagement counters are excluded;
# they enter only as explicit day-1 (Model B) features built in sip.data.
VIDEO_KEEP = [
    "video_id", "author_id", "create_time", "create_date", "duration", "ratio",
    "desc_language", "is_english", "desc", "sticker_text", "hashtags",
    "created_by_ai", "is_ads", "music_id", "music_title", "music_author",
    "transcript", "word_count", "emoji_count", "question_count", "hashtag_count",
    "speaking_rate", "gpt_summary", "topic",
    "anger", "joy", "surprise", "sadness", "disgust", "fear",
]

# Feature-name prefixes/sets used by leakage guards: anything matching DAY1_* or
# these engagement tokens must never appear in a Model-A feature matrix.
ENGAGEMENT_TOKENS = (
    "play_count", "like_count", "comment_count", "share_count", "collect_count",
    "download_count", "whatsapp_share_count", "_d1", "day1", "engagement",
)
