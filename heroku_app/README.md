# ShouldIPost? — Heroku app (minimal, torch-free)

Self-contained deployable: two hand-feature models (no video, no embeddings → ~1.6 MB slug,
no 30 s-timeout risk), an Apple-minimal calculator, model downloads, and a model card.

- **Model A** (pre-publication): caption + duration + timing → Post / Do-not-post / Unsure.
- **Model B** (+ day-1 counts): Keep / Delete (organic account hygiene; deletes only high-confidence duds).
- Feature building reuses the bundled `sip/` so train and serve match exactly.

## Run locally
```bash
cd heroku_app
pip install -r requirements.txt
uvicorn app:app --port 8000     # -> http://127.0.0.1:8000
```

## Deploy to Heroku (this folder is the deploy root)
```bash
cd heroku_app
git init && git add -A && git commit -m "ShouldIPost deploy"
heroku login
heroku create <your-app-name>          # or: heroku git:remote -a <existing-app>
git push heroku main                    # (use `master` if that is your branch)
heroku open
```
The Procfile (`web: uvicorn app:app --host 0.0.0.0 --port $PORT`), `requirements.txt` (slim) and
`runtime.txt` (python-3.12) are already here. Files served: `/` (UI), `/api/score`,
`/download/a`, `/download/b`, `/api/health`.

## One caveat — scikit-learn version
The `.joblib` models were pickled with scikit-learn 1.9.x; Heroku installs the latest that matches
`scikit-learn>=1.4`. HistGradientBoosting unpickles across minor versions (with an
`InconsistentVersionWarning`), so it normally just works. If loading ever errors on a version
mismatch, re-save the two models under the deployed scikit-learn version
(`experiments/build_heroku_models.py` in the main repo) and copy them into `models/`.

## What is NOT here (by design)
Live video-file / URL analysis (SigLIP/CLAP/Whisper) is intentionally excluded — it needs a
multi-GB stack and 60–180 s per video, which exceeds Heroku's slug limit and 30 s request timeout.
Run that path locally via the main repo's `./start.sh`, or on a separate GPU service.
