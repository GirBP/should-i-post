# Slim, torch-free image (Models A/B calculator only) — runs in <256 MB RAM.
# Meant for the smallest, always-on droplet; the heavy video app is the root Dockerfile.
#   Build:  docker build -t shouldipost-slim heroku_app/
#   Run:    docker run --rm -p 80:8080 shouldipost-slim
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
