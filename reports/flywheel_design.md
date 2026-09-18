# Self-learning flywheel — design rationale and sources

Goal: the system ingests posted-video URLs, polls their public stats, self-labels outcomes at
label maturity, and periodically retrains a challenger model. No human annotation. All local.

## Validated design decisions

| Decision | Rationale | Source |
|---|---|---|
| Metadata polling via yt-dlp, 2x/day, exponential backoff, <=50 tracked videos | Empirically confirmed on this machine (view/like/comment counts + creator page of ~15 recent videos with per-video views); scraping-feasibility research puts safe cadence at 2x/day per video with >=8h gaps | yt-dlp TikTok extractor; ScrapingBee/IPFoxy 2026 scraping guides; local empirical test 2026-07-05 |
| Creator baseline = median views of ~15 recent videos | Within-creator "beats own median" label without follower data; matches the project's fame-neutral target | project label study (reports/label_comparison.md) |
| Labels only at maturity (H=14 days) | Training on immature engagement biases the model (delayed-feedback problem) | Ktena et al. 2019, arXiv:1907.06558 |
| Day-1 counters frozen at age>=1d | Clean Model B training signal, mirrors the dataset's day-1 semantics | project Model B design |
| Periodic FULL refit, not incremental trees | Incremental GBT updating degrades split quality; full refit is minutes at this scale | XGBoost continued-training caveats; Chip Huyen continual-learning notes |
| Champion/challenger, promotion only by human | Prevents silent regressions; challenger evaluated on the fixed temporal test next to the champion | DataRobot champion/challenger; Google MLOps level 1 |
| Selection-bias guard (warn if >80% one class) | Tracking only trending videos breaks class balance and feeds a feedback loop | arXiv:2207.01616 (feedback loops); MNAR bias literature |
| Drift check via PSI on features (threshold 0.25) | Cheap offline drift proxy | Evidently AI concept-drift guide |

## Commercial context (blunt)

- No credible commercial product predicts PRE-publication organic virality; the market converged on
  "measure the first 12-48h, then amplify winners": Dash Hudson Vision AI ($1,999+/mo), Emplifi.
  Our Model B day-1 gate is exactly that product category; Model A is honest triage.
- Pre-publication prediction is sold only for PAID ads (Kantar LINK, Pencil, System1) at enterprise
  prices — a different market.
- The lingbow-trained weights are CC BY-NC (non-commercial). The flywheel is the path out: data the
  system collects itself is licence-clean, so a commercially usable model can be trained on the
  accumulated pool once it is large enough.

## Practical note

Tracking videos that are ALREADY >= 14 days old yields a label on the first post-gap poll (age
already past maturity), so the labeled pool can be grown quickly by tracking a mix of recent hits
AND ordinary older videos from the same creators (which also feeds the bias guard).

## Components

- src/sip/flywheel.py — store (SQLite), track/poll/status/retrain_challenger
- scripts/flywheel_tick.py — cron-able tick (poll + optional retrain)
- webapp: POST /api/track, GET /api/flywheel/status, POST /api/flywheel/tick + UI buttons
- models/challenger.joblib + reports/flywheel_challenger.json — challenger artifacts (manual promotion)
