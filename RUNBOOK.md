# RUNBOOK — reproduce everything

## One command
```bash
cd shouldipost
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg                       # only needed for --with-multimodal

./run_all.sh                              # data + tabular/text experiments + deployable model + tests
./run_all.sh --with-multimodal           # + download videos, extract video/audio, multimodal ablation
./run_all.sh --with-judge                # + LLM-as-judge marginal-lift (needs judge scores on disk)
```
All results land in `reports/` (CSV/JSON + `RESULTS.md`, `RESULTS_AUTO.md`, `experiments_log.md`),
the deployable model in `models/deployable.joblib`. Demo: `./start.sh` (FastAPI webapp).

## Stages (what run_all.sh does)
1. **Data** — `scripts/download_raw_lingbow.py` (needs Hugging Face) → `data/raw/lingbow/*`;
   `python -m sip.data` → `data/processed/canonical.parquet` (labels, day-1, splits).
2. **Embeddings + trend** — `scripts/encode_text.py` (MiniLM/BGE/e5 → `emb_*.parquet`);
   `scripts/build_trend_features.py` (internal momentum).
3. **Tests** — `pytest tests/` (leakage guards, label freeze, feature determinism, inference).
4. **Experiments** — `experiments/run_tabular_text.py` (baselines, encoders, ablation, family,
   Optuna); `experiments/run_ensemble.py`; `experiments/run_judge.py` (after the judge workflow);
   `experiments/build_deployable.py` (calibrated A→B + conformal + bands);
   `experiments/make_report.py` (consolidates `reports/RESULTS_AUTO.md`).
5. **Multimodal** (`--with-multimodal`) — `multimodal/extract.py` (disk-bounded
   download→extract→delete: CLIP/SigLIP/CLAP + low-level); `experiments/run_multimodal.py`.

## The webapp's model

`./start.sh` scores through `models/unified.joblib`, not `models/deployable.joblib`. Build it
after a `--with-multimodal` run (it needs `data/multimodal/features.parquet` and the text
embeddings from stage 2):
```bash
PYTHONPATH=src python experiments/build_unified.py   # -> models/unified.joblib, reports/unified_metrics.json
```

## LLM-as-judge
Scores were produced by Claude agents in the Claude Code workflow runner (no API key). To
reproduce without that runner, drop one JSON array per batch into `data/processed/judge_out/`
(schema: `video_id, hook, clarity, trend_fit, arousal, saturation, cta, rationale`) and run
`experiments/run_judge.py`.

## Notes / known limits
- TikTok rate-limits sustained downloads (IP block) → multimodal coverage is partial; the
  extractor is polite + resumable (`multimodal/extract.py --polite`, `multimodal/polite_download.py`).
- Heavy artifacts (`data/`, `models/*`, `emb_*.parquet`, downloaded videos) are git-ignored;
  regenerate via the steps above.
- Apple-Silicon MPS by default; set `SIP_DEVICE=cpu` to force CPU (e.g. for tests).
