# ShouldIPost web app (local, backend + frontend)

Full local application: a FastAPI backend that extracts features from a video and runs the deployed
Model A / Model B, plus a static frontend with a two-mode toggle.

- **Not posted yet (file):** upload the video + the caption you would post (which is not in the file);
  the backend extracts transcript (Whisper), summary/topic/emotions (optional LLM), duration, aspect.
- **Already posted (URL):** paste the link; the backend fetches metadata, downloads audio and extracts
  everything itself.
- Optional day-1 plays/likes switch scoring to the stronger Model B.

## Run
```bash
cd shouldipost && source .venv/bin/activate
export PYTHONPATH=src SIP_DEVICE=cpu KMP_DUPLICATE_LIB_OK=TRUE
export SIP_LLM=deepseek DEEPSEEK_API_KEY=...   # optional: adds summary/topic/emotions. Omit -> transcript-only.
uvicorn webapp.server:app --port 8000
# open http://127.0.0.1:8000
```
Needs `ffmpeg` (brew install ffmpeg). First video loads the Whisper model (~10-30 s).

## Layout
- `server.py` — FastAPI: routes + thin service layer (calls src/sip/extract + src/sip/inference).
- `static/index.html` `static/style.css` `static/app.js` — frontend.

All ML logic stays in `src/sip`; the web layer only validates input, calls the extractor + model, and
shapes one JSON response.
