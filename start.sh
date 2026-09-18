#!/bin/bash
# ShouldIPost — one-command product launch (local, offline-capable).
#   ./start.sh          start everything (ollama + backend + warmup) and open the browser
#   ./start.sh stop     stop everything
#   ./start.sh status   health + warmup state
set -e
cd "$(dirname "$0")"
PORT="${SIP_PORT:-8770}"
URL="http://127.0.0.1:$PORT"

stop_all() {
  pkill -f "uvicorn webapp.server" 2>/dev/null && echo "backend stopped" || echo "backend not running"
}

case "${1:-start}" in
  stop)   stop_all; exit 0 ;;
  status) curl -s "$URL/api/health" | python3 -m json.tool 2>/dev/null || echo "backend not running"; exit 0 ;;
esac

# 1) local LLM (optional but recommended: summary/topic/emotions fully offline)
if command -v ollama >/dev/null 2>&1; then
  curl -s -o /dev/null http://localhost:11434/api/tags 2>/dev/null || { nohup ollama serve >/tmp/ollama.log 2>&1 & sleep 2; }
  export SIP_LLM="${SIP_LLM:-ollama}"
  echo "ollama: up (backend $SIP_LLM, model ${SIP_LLM_MODEL:-llama3.2:3b})"
else
  echo "ollama not installed -> transcript-only mode (still works)"
fi

# 2) backend (with startup warmup of BGE/Whisper/SigLIP/CLAP)
source .venv/bin/activate
export PYTHONPATH=src SIP_DEVICE="${SIP_DEVICE:-cpu}" KMP_DUPLICATE_LIB_OK=TRUE
stop_all >/dev/null 2>&1 || true
nohup uvicorn webapp.server:app --host 127.0.0.1 --port "$PORT" >/tmp/shouldipost.log 2>&1 &
printf "starting backend"
for i in $(seq 1 20); do
  sleep 1; printf "."
  curl -s -o /dev/null "$URL/api/health" 2>/dev/null && break
done
echo " up: $URL"

# 3) wait for warmup so the presentation has zero cold starts
printf "warming models (BGE, Whisper, SigLIP, CLAP)"
for i in $(seq 1 60); do
  STATE=$(curl -s "$URL/api/health" | python3 -c "import sys,json;print(json.load(sys.stdin).get('warmup',''))" 2>/dev/null)
  [ "$STATE" = "warm" ] && break
  sleep 3; printf "."
done
echo " $STATE"
curl -s "$URL/api/health" | python3 -m json.tool

# 4) open the product
open "$URL" 2>/dev/null || true
echo "ready: $URL   (./start.sh stop — вимкнути)"
