#!/usr/bin/env bash
# ShouldIPost? — single reproducible entrypoint.
#   ./run_all.sh                  data + tabular/text experiments + deployable model + tests
#   ./run_all.sh --with-multimodal   also download videos + extract video/audio + run multimodal ablation
#   ./run_all.sh --with-judge        also (re)run the LLM-as-judge marginal-lift experiment
# Heavy steps are resumable; re-running skips finished artifacts.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src
WITH_MM=0; WITH_JUDGE=0
for a in "$@"; do
  [ "$a" = "--with-multimodal" ] && WITH_MM=1
  [ "$a" = "--with-judge" ] && WITH_JUDGE=1
done

echo "== 0. venv + deps =="
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

echo "== 1. data: raw lingbow -> canonical frame (labels, day-1, splits) =="
[ -f data/raw/lingbow/videos.parquet ] || python scripts/download_raw_lingbow.py
python -m sip.data                                  # builds data/processed/canonical.parquet

echo "== 2. text embeddings (3 encoders) + internal trend momentum =="
python scripts/encode_text.py                       # emb_{minilm,bge,e5}.parquet (resumable)
python scripts/build_trend_features.py              # trend_features.parquet

echo "== 3. tests (leakage guards, label freeze, feature determinism) =="
python -m pytest tests/ -q

echo "== 4. tabular + text experiment suite (-> reports/) =="
python experiments/run_tabular_text.py              # baselines, encoders, ablation_A, family, optuna
python experiments/run_ensemble.py                  # stacking ensemble (matrix item 12)
# legacy §1 experiments (label choice, cross-creator transfer, focused creator-fit) — cited in RESULTS
python experiments/label_comparison.py || true
python experiments/cross_creator_theories.py || true   # also builds data/processed/_feat_cache.parquet
python experiments/creator_fit.py || true
python experiments/model_power.py || true

echo "== 5. deployable A->B model (calibration + conformal + decision bands) =="
python experiments/build_deployable.py              # models/deployable.joblib
python experiments/make_report.py                   # consolidate -> reports/RESULTS_AUTO.md
echo "== 5b. consolidated research notebook + static site report =="
python scripts/build_results_notebook.py || true
pip install -q nbconvert ipykernel 2>/dev/null || true
jupyter nbconvert --to notebook --execute --inplace notebooks/RESULTS.ipynb --ExecutePreprocessor.timeout=300 || true
jupyter nbconvert --to html notebooks/RESULTS.ipynb --output-dir docs --output results.html || true

if [ "$WITH_JUDGE" = "1" ]; then
  echo "== 6. LLM-as-judge marginal lift (needs data/processed/judge_out/*.json) =="
  python experiments/run_judge.py || echo "  (judge_out missing — run the judge workflow first)"
fi

if [ "$WITH_MM" = "1" ]; then
  echo "== 7. multimodal: download + extract (disk-bounded) + ablation =="
  command -v ffmpeg >/dev/null || { echo "need ffmpeg: brew install ffmpeg"; exit 1; }
  pip install -q open_clip_torch timm opencv-python-headless librosa soundfile yt-dlp
  python multimodal/extract.py --polite --chunk 200 --models clip,sig,clap
  python experiments/run_multimodal.py
else
  echo "== 7. multimodal skipped (add --with-multimodal) =="
fi

echo "DONE. Results in reports/ ; deployable model in models/ ; demo: ./start.sh"
