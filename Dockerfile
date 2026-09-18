# ShouldIPost? — full product image (FastAPI webapp: manual entry + live video file/URL).
# Runs the real pipeline: metadata + Whisper transcript + SigLIP/CLAP video-boost + models A/B.
# No Heroku slug/30s limits here — built for a DigitalOcean Droplet (or any Docker host).
#
#   Build:  docker build -t shouldipost .
#   Run:    docker run --rm -p 8080:8080 shouldipost      # -> http://localhost:8080
# Needs models/deployable.joblib (+ deployable_mm.joblib) in the build context (./run_all.sh builds them).
FROM python:3.12-slim

# ffmpeg: audio extraction for Whisper + frame decode.  curl: container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY requirements/ requirements/
# Core deps + the multimodal extras (SigLIP via open_clip, CLAP audio, frame/audio IO) that
# requirements.txt lists as a separate install for the video path.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir open_clip_torch timm opencv-python-headless librosa soundfile audioread

COPY . .
# SIP_LLM=none -> transcript-only extraction, never fails (no Ollama/API key on the box).
# HF_HOME on a mounted volume so SigLIP/CLAP/Whisper/BGE download once and persist across restarts.
ENV PYTHONPATH=/app/src SIP_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE \
    SIP_LLM=none SIP_WARMUP=1 HF_HOME=/app/.hf_cache PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:8080/api/health || exit 1
CMD ["sh", "-c", "uvicorn webapp.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
