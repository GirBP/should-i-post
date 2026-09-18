# Model card — ShouldIPost? (pre-publication TikTok success evaluator)

## Overview
A two-model triage system that scores a TikTok **before it is posted** and outputs
**Post / Do not post / Unsure** with a *calibrated* probability, the main factors, and an
explicit note on what the model can and cannot know at prediction time.

- **Model A (cold-start, pre-publication):** content + creator-state-as-of-post-time only.
- **Model B (early-signal, +day-1):** Model A's inputs plus day-≤1 engagement; for the
  amplify / keep / cut decision after posting.
- **Decision policy:** isotonic calibration → split-conformal abstention → cost-based bands
  (Post / Do not post / Unsure) tuned for precision@Post on validation. Framed as
  explore–exploit: post the uncertain ones organically, let Model B resolve them on day 1,
  amplify the winners.
- **Explainability & controls:** each prediction returns **per-prediction SHAP** factors
  (embedding dims aggregated into one "semantic text" bucket for readability), **example-based
  evidence** (nearest past videos with their realised hit/flop outcome), a data-grounded
  rationale, and the can/can't-know note. The decision threshold is a **cost-ratio knob**
  (cost of a false "Post" ÷ cost of a missed hit) the team can turn at inference without
  retraining — the model is fixed, only the Post/Unsure/Don't bands move (Elkan-optimal
  threshold on the calibrated validation set).

## Intended use & users
Content/marketing teams triaging a backlog of candidates and deciding what to amplify.
**Not** an oracle of virality — a triage + abstain tool. Prototype only (data is CC BY-NC).

## Data
- **`lingbow/tiktok-video-engagement-200k`** — 209,543 videos, 1,872 creators, posted
  2024-06…11; three linked tables (videos, engagement_daily 0–30, creator_daily).
- The earlier `datahiveai/Tiktok-Videos` (4 creators, 2,060 rows) is a dead end (deployed
  model ROC-AUC ≈ 0.52, CI [0.45, 0.59] ≈ random) and is **not** used. The leftover
  `reports/metrics.json` and `models/metadata.json` describe that superseded 4-creator prototype
  (kept only for history); the deployed system is `models/deployable.joblib` with
  `reports/deployable_metrics.json`.
- Raw video+audio for a balanced, shuffled, leakage-aware subset are downloaded by `video_id`
  (yt-dlp) for the multimodal track; coverage is partial (TikTok rate-limits / deletions) and
  reported honestly.

## Target (label)
**Primary:** *breakout within-creator* at horizon H=14 — `views@14 / followers_at_post`,
centred by the creator's own median and binarised at the global median. This is **fame-neutral**
(the creator's personal baseline is removed; fame-leak AUC ≈ 0.51 vs 0.72–0.86 for naive labels —
0.72 raw breakout, 0.79 absolute views, 0.86 absolute log-views), business-aligned (reach beyond the
creator's own audience), and the
day-1 model predicts it strongly. The per-creator median is a label *normalisation*, not a
feature, so it is not a feature leak.
**Secondary / auxiliary:** within-creator engagement-rate (resonance) and follower growth.

## Features (modular, each ablated)
Tabular (caption, emotion, duration, timing, meta), topic/music/hashtag priors (fold-safe
target-encoding), text embeddings (MiniLM / BGE / e5), **creator-fit** (temporally-safe
similarity to the creator's own past winners), **trend-fit** (internal trailing-window
momentum of the sound/hashtags — same-platform, leakage-safe), retrieval (kNN memory bank),
**LLM-as-judge** (hook/clarity/trend-fit/arousal/saturation/CTA + rationale), and the
**multimodal** track (CLIP & SigLIP frames + a dedicated hook window, CLAP audio embeddings,
plus low-level visual: scene cuts, motion, faces, brightness; and low-level audio/prosody:
mel stats, RMS, ZCR, spectral, tempo, pitch, speech/music ratio).

The **deployable** model deliberately uses only inference-reproducible blocks
(deterministic caption/timing/duration/meta + a public sentence encoder) so a brand-new
candidate can be scored cold; HistGradientBoosting makes any genuinely-missing feature
NaN-tolerant. Dataset-prior encoders (target-encoding, creator-fit, trend) are measured in
the ablation but excluded from the cold-start deployable model because a new candidate
cannot reproduce them.

## Evaluation
Two protocols on every experiment: **temporal** (train < valid < test by date) and
**leave-one-creator-out (LOCO)**. Metrics: ROC-AUC, PR-AUC, precision@Post, winner recall,
Brier + reliability, abstention coverage, all with 95% bootstrap CIs. Automated leakage guards
(`sip.leakage`) assert no engagement/day-1 signal enters Model A and no horizon counter enters
Model B; fold-safe encoders are fit on training folds only.

Full headline numbers, ablation tables, and decision bands: see `reports/RESULTS.md`,
`reports/RESULTS_AUTO.md`, `reports/experiments_log.md`, and `reports/*.csv`.

## Key honest findings
- **Pre-publication reach (breakout) is information-bound, not model-bound.** Tabular/semantic
  Model A sits at ≈ 0.53–0.57 LOCO; tuned XGBoost ≈ HGB ≈ MLP — a stronger model does not beat
  a simple one. An apparent creator-fit/recency lever was found (post-audit) to be a temporal lookahead leak; leak-free it adds ≈0 (see reports/RESULTS.md §13).
- **Resonance (within-creator ER) is more learnable** than reach (≈ 0.63 with rich content):
  the **LLM-as-judge helps ER but not breakout** — judged content quality predicts whether a
  video resonates with the existing audience, not whether the algorithm pushes it to new ones.
- **Model B (day-1), leak-free ≈ 0.75** (the pre-audit ≈0.95 used a leaky author-relative feature): the day-1
  amplify/cut decision and the pre-publication abstain, not a pre-publication oracle.
- The missing pre-publication drivers (first-seconds watch-time/completion, algorithmic
  seeding, external trend timing) are simply not observable before posting.

## Limitations & ethics
- "Success" here is a **reach proxy** — no clicks, conversions, or revenue in the data.
  Validated against reach and follower growth; commercial value would need other data.
- Data is **CC BY-NC 4.0** — prototype/research only, not commercial deployment.
- Coverage of raw video is partial (TikTok deletions + rate-limits since 2024).
- Predictions are weakest with no caption/transcript; the system warns and abstains rather
  than guessing.

## AI tools / hosted models / paid compute
- Local, free: HistGradientBoosting/XGBoost/CatBoost, sentence-transformers (MiniLM/BGE/e5),
  open_clip (CLIP/SigLIP), CLAP, librosa — all on Apple-Silicon MPS.
- **LLM-as-judge** scores were produced by Claude (Sonnet) agents inside the Claude Code
  workflow runner (no external API key; counted as session tokens, not a separate paid API).
- **Paid compute: $0** (no cloud GPU used; everything ran locally within the $100 budget).
