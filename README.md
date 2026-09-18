# ShouldIPost?

A **pre-publication** success evaluator for TikTok. Given a *not-yet-posted* candidate
(caption, planned duration/time, optionally a video file or URL) it returns
**Post / Do not post / Unsure**, a **calibrated** probability, the main factors, an LLM-style
rationale, and an honest note on what the model can and cannot know before posting.

It is engineered to be **honest about a hard problem**: pre-publication virality is
information-bound (the real drivers — first-seconds watch-time, algorithmic seeding, trend
timing — are not observable yet), so the system *abstains* when the signal is weak, never uses
post-publication metrics as Model-A inputs, is *calibrated* rather than confident, and locates
the reliable value in the **day-1 amplify/cut decision (Model B)**.

> **Leakage discipline:** likes/views are the *label*, never a Model-A input. Pasting a URL
> reads only pre-publication metadata (caption, duration). Automated guards enforce this.

---

## Quickstart — the product (one command)

Everything local, offline-capable, free. Apple-Silicon (MPS) by default. Python 3.10–3.14.

```bash
cd shouldipost
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg                      # audio/video extraction
# optional, fully offline LLM fields (summary/topic/emotions):
#   brew install ollama && ollama pull llama3.2:3b

./start.sh                               # ollama + backend + model warmup + opens the browser
./start.sh stop                          # shut down
./start.sh status                        # health + warmup state
```

`start.sh` launches the FastAPI app (`webapp/server.py`) on <http://127.0.0.1:8770>, pre-warms
every heavy model at startup (BGE, Whisper, SigLIP, CLAP) so the live session has **zero cold
starts**, and prefers a local Ollama LLM (no API key). The header shows a green **● ready** dot
when warm.

**Two scoring modes** (segmented toggle on the page):
- **Video file — not posted yet:** upload the mp4 + the caption you would post; the backend
  transcribes it (Whisper), derives LLM fields (Ollama), extracts SigLIP frames + CLAP audio,
  and scores with the unified classifier (hand features + text + video + audio).
- **Link — already posted:** the backend pulls everything from the URL itself (metadata +
  transcript + fields) and scores the same way; optional day-1 plays/likes switch the classifier
  to its day-1 head (`/api/probe` tells the UI when current counts are a valid day-1 proxy).

**Self-learning flywheel** (URL mode): *Track for self-learning* stores the video, polls its
public stats, freezes the day-1 signal, self-labels it at maturity (H=14, beats the creator's
own median), and accumulated labels retrain a *challenger* — promotion stays a human decision.
See `reports/flywheel_design.md`; run a tick with `PYTHONPATH=src python scripts/flywheel_tick.py`.

### Research pipeline (reproduce the models)

```bash
./run_all.sh                             # data + tabular/text experiments + deployable model + tests
./run_all.sh --with-multimodal           # + download videos, extract video/audio, multimodal ablation
```

Outputs land in `reports/` (tables, CIs), `models/deployable.joblib` (the A→B system), and
`reports/experiments_log.md` (one row per run). The extractor is validated on 262 real videos with
known outcomes: `scripts/validate_extractor.py` → `reports/extractor_validation.json`; RESULTS §13.

The webapp itself scores through a separate, newer artifact, `models/unified.joblib` (hand
features + BGE text + SigLIP + CLAP, one classifier with an optional day-1 head): build it with
`python experiments/build_unified.py` after `./run_all.sh --with-multimodal` has produced
`data/multimodal/features.parquet` and the text embeddings; metrics land in
`reports/unified_metrics.json`.

## Directory layout

```
src/sip/          importable library — the pipeline (data, features, models, inference)
scripts/          one-off/reproducible steps: download data, encode text, build reports/notebooks
experiments/      training + evaluation runs, one script per experiment, logged to reports/
multimodal/       batch video/audio pipeline: download by video_id -> extract -> features.parquet
notebooks/        research narrative and the module-notebook pipeline (see notebooks/README.md)
tests/            pytest: leakage guards, label freeze, feature determinism, fold-safety, inference
webapp/           FastAPI backend + static frontend — the demo UI (./start.sh)
heroku_app/       slim, torch-free deployment (Models A/B calculator only, no video)
deploy/           DigitalOcean Droplet stack (Caddy + docker compose) for the full product
reports/          generated tables, metrics, plots, and the results notebook's source artifacts
docs/             GitHub Pages: the public results page
data/             not committed; see data/README.md for source, license, and how to rebuild it
models/           not committed; trained artifacts, rebuilt by experiments/*.py
requirements/     base/dev/multimodal dependency sets (requirements.txt points at base)
```

### Library layout (`src/sip/`)

```
config.py     constants + paths (single source of truth: H=14, splits, seeds, bands)
data.py       raw lingbow (3 tables) -> canonical frame (labels, day-1, temporal split)
splits.py     temporal (date) + leave-one-creator-out (LOCO) protocols
features.py   modular feature blocks (the unit of ablation); fold-safe encoders
modeling.py   model factory + isotonic/Platt calibration + split-conformal abstention
eval.py       metrics + bootstrap CIs + reliability + decision bands
leakage.py    automated leakage guards (asserted in code and tests)
experiment.py one-call runner: blocks + model + target + split -> metrics (logs each run)
inference.py  STATELESS scoring of a URL / video file / caption (graceful degradation)
extract.py    video -> features: ffprobe/yt-dlp + ffmpeg + faster-whisper + pluggable LLM
flywheel.py   self-learning tracker: poll -> freeze day-1 -> self-label at maturity -> retrain
rules.py      rule-based baseline scorer
```

Data prep (`sip.data`/`scripts`), training/eval (`experiments`), and inference
(`sip.inference`/`webapp`) are physically separate.

## Deployment

Three ways to run this, for different needs:

- **Root `Dockerfile`** — the full product (manual entry + live video file/URL, SigLIP/CLAP/Whisper).
  No request-time limits, so it needs a host without them: see `deploy/DEPLOY_DIGITALOCEAN.md` for
  a DigitalOcean Droplet behind Caddy (`deploy/docker-compose.yml`). Use this when you want the
  complete demo, including video upload/URL scoring, reachable at a public URL.
- **`heroku_app/`** — a slim, torch-free image with only the hand-feature Models A/B calculator
  (no video, <256 MB RAM). Fits Heroku's slug size and 30 s request timeout, and is cheap to run
  24/7. Use this for an always-on public URL when live video isn't needed.
- **`deploy/`** — DigitalOcean Droplet scripts and compose files for both of the above:
  `docker-compose.yml` runs the full root `Dockerfile` behind Caddy (auto-HTTPS);
  `slim.docker-compose.yml` runs `heroku_app/` instead, on the cheapest droplet size. See
  `deploy/MINIMAL_COST.md` for splitting the two by cost (cheap calculator always on, full video
  app on demand only).

## How "success" is defined

**Primary label — breakout within-creator (H=14):** `views@14 / followers_at_post`, centred by
the creator's own median and binarised at the global median. It is **fame-neutral** (removing
the creator's personal baseline; a creator-only predictor scores ≈0.34–0.50, i.e. fame is *not*
the signal), business-aligned (reach beyond the creator's own audience), and the day-1 model
predicts it strongly. Secondary targets: within-creator engagement-rate (**resonance**) and
follower growth. See `reports/model_card.md`.

## Evaluation protocol (applied to every experiment)

- **Two splits:** temporal (train<valid<test by date) **and** LOCO (GroupKFold by author).
- **Metrics with 95% bootstrap CIs:** ROC-AUC, PR-AUC, precision@Post, winner recall,
  Brier + reliability, abstention coverage.
- **Leakage guards** (`sip.leakage`, exercised by `tests/`): Model A gets only pre-publication
  content + creator state as-of-post; Model B may add day-≤1 signals; encoders fit on train folds.

## Results (headline)

Full, auto-generated tables: **`reports/RESULTS.md`** + `reports/RESULTS_AUTO.md` +
`reports/*.csv`. Honest summary:

- **Pre-publication reach (breakout) is information-bound.** Model A ≈ 0.53–0.57 LOCO
  (temporal-test 0.568); tuned XGBoost ≈ HGB ≈ MLP (a stronger model does not beat a simple one).
  The signal is mostly **TAB + text**; creator-fit adds only ≈+0.01 once leak-safe (audit §13).
- **Resonance (ER) is more learnable** (≈ 0.60–0.63); the **LLM-as-judge helps ER, not breakout**.
- **Model B (+day-1) is strong** (temporal-test **≈ 0.75, leak-free**) — the reliable product value.
  (An earlier ≈0.94 relied on a leaky author-relative day-1 feature, since removed — RESULTS §13.)
- Multimodal video+audio lift over text+tabular is measured in `reports/multimodal_ablation.csv`.

## Decision policy

Isotonic calibration → split-conformal abstention → cost-based bands tuned for precision@Post
on validation → **Post / Do not post / Unsure**. Product framing is explore–exploit: post the
uncertain ones organically, let Model B resolve them on day 1, amplify the winners.

## Dataset & license

[`lingbow/tiktok-video-engagement-200k`](https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k)
— 209,543 videos, 1,872 creators, 2024-06…11. **License: CC BY-NC 4.0 — research/non-commercial
only.** Raw data and downloaded videos are not committed (`.gitignore`).

## AI tools / hosted models / paid compute

Built with **Claude Code** (Opus) as engineer; every metric is from running the local pipeline.
**Runtime is fully local & free:** local **Ollama** (default `llama3.2:3b`, no API key) for the
optional caption fields, local **BGE** text embeddings, local Whisper / SigLIP / CLAP, and
scikit-learn / XGBoost / CatBoost — all on Apple-Silicon MPS. The LLM backend is pluggable
(`SIP_LLM=ollama` default; `gemini|deepseek|haiku` optional with a key; `none` = transcript-only).
The **LLM-as-judge** research features were generated once by Claude (Sonnet) agents in the Claude
Code workflow runner — no external API key, session tokens only. **Paid APIs: none. Paid compute:
$0** (no cloud GPU; the $100 budget was unused).
