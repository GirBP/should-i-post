# Deploy ShouldIPost? on DigitalOcean

Heroku could host only the slim manual-entry calculator: its **30 s request timeout**, **500 MB
slug limit**, and **no GPU** rule out live video (SigLIP + CLAP + Whisper need a multi-GB stack and
30–120 s per clip). A DigitalOcean **Droplet** has none of those limits — so it runs the *full*
product: manual entry **plus** live video-file / URL analysis with the multimodal video-boost model.

> The GitHub Student Developer Pack (the same pack behind the Heroku offer) includes a
> **$200 DigitalOcean credit for 12 months** — enough to run this for the better part of a year.

## Why a Droplet, not App Platform

| | Droplet (recommended) | App Platform (PaaS) |
|---|---|---|
| Long requests (60–120 s video) | ✅ no proxy timeout (we set 300 s) | ⚠️ platform request timeout — a sync video call is risky |
| Image / disk size | ✅ 25–80 GB disk, image size irrelevant | ✅ Docker image (large ok) |
| RAM for torch warmup | ✅ pick 4–8 GB | ⚠️ cheap tiers 512 MB–1 GB are too small |
| GPU | optional GPU Droplet | ❌ none |
| Ops | you run Docker + Caddy (scripted below) | fully managed |

Droplet wins for the video path; App Platform is fine only if you drop live video or move it to a
background worker. A plain **CPU** Droplet is enough — no GPU required (inference is 30–120 s/clip,
and with no timeout that is acceptable). A GPU Droplet only makes it snappier.

## Sizing

- **Minimum:** `s-2vcpu-4gb` (~$24/mo) + the 4 GB swap `setup_droplet.sh` adds. Warmup loads
  SigLIP + CLAP + Whisper + BGE together (~3–4 GB peak).
- **Comfortable:** `s-4vcpu-8gb` (~$48/mo) — faster extraction, no swap pressure.
- $200 credit ⇒ ~8 months at 4 GB, ~4 months at 8 GB.

## Steps

1. **Create the Droplet** — Ubuntu 24.04, 4 GB+, in the control panel (or `doctl compute droplet create`).

2. **Provision Docker** (one-time):
   ```bash
   ssh root@<droplet-ip> 'bash -s' < deploy/setup_droplet.sh
   ```

3. **Ship the repo** (needs `models/unified.joblib` — the one classifier — present locally;
   `.dockerignore` drops `.venv`, `data/`, `demo_videos/` and the superseded models, so the upload is small):
   ```bash
   rsync -az --filter=':- .dockerignore' ./ root@<droplet-ip>:/opt/shouldipost/
   ```

4. **Launch** (from the droplet):
   ```bash
   ssh root@<droplet-ip>
   cd /opt/shouldipost/deploy
   # HTTP on the IP:
   docker compose up -d --build
   # …or with a domain for automatic HTTPS (point an A record at the droplet first):
   SITE_ADDRESS=shouldipost.example.com docker compose up -d --build
   ```
   First build ~10–15 min (torch + transformers + open_clip). First request triggers a one-time
   model download (~2–4 GB) into the persisted `hf_cache` volume; watch it warm:
   ```bash
   curl -s http://<droplet-ip>/api/health      # {"warmup":"warming"} -> "warm"
   ```

5. **Open** `http://<droplet-ip>/` (or `https://your.domain`). Manual entry, URL scoring, and file
   upload (≤200 MB) all work.

## Environment

- `SIP_LLM=none` (default) → transcript-only extraction; **never fails**, no key needed.
- For LLM-derived fields (summary/topic/emotions) set an API backend in `deploy/.env`:
  ```
  SIP_LLM=deepseek
  DEEPSEEK_API_KEY=sk-...
  ```
  (or `SIP_LLM=haiku` + `ANTHROPIC_API_KEY`). Or run Ollama in a sidecar container for fully-local LLM.

## Operate

```bash
docker compose logs -f app            # tail
docker compose restart app            # restart (hf_cache persists → no re-download)
docker compose down                   # stop
docker compose up -d --build          # redeploy after `rsync` of new code/models
```

## Notes

- **Slimmer image:** `catboost`, `xgboost`, `optuna`, `shap`, `datasets` in `requirements.txt` are
  training-only; a serve-only requirements file would cut build time and image size. Left in for now
  so train and serve share one dependency set.
- **The slim Heroku app still exists** under `heroku_app/` if you also want a tiny always-on
  calculator; it and this full Droplet can coexist.
