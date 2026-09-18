"""ShouldIPost web app — FastAPI backend + static frontend.

Two scoring modes, both running the real deployed model:
  - published  (URL)  -> the backend extracts every feature itself (metadata + Whisper transcript
                         + LLM fields) and returns them for display, then scores.
  - unpublished (file) -> the user uploads the video file and supplies the fields that cannot be read
                         from it (the caption they would post); the backend extracts the rest.

Thin layer: routes validate input, call sip.extract (video -> features) and sip.inference (features ->
recommendation), and shape one JSON response. All ML logic lives in src/sip; nothing is duplicated here.

Run:  PYTHONPATH=src uvicorn webapp.server:app --port 8000   (or:  python -m webapp.server)
LLM fields need SIP_LLM + a key (SIP_LLM=deepseek DEEPSEEK_API_KEY=...); without a key it is
transcript-only and never fails.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import extract as EX, inference as INF  # noqa: E402

app = FastAPI(title="ShouldIPost", version="1.0")
STATIC = Path(__file__).resolve().parent / "static"
MODELS = Path(__file__).resolve().parents[1] / "models"

MAX_MB = 200
_WARM = {"state": "cold", "steps": []}


def _warmup():
    """Pre-load every heavy component once at startup so the live presentation has no cold starts:
    champion + BGE, the mm artifact, the Whisper model, and the SigLIP/CLAP encoders. Each step is
    independent — a failure records the step as skipped but never blocks the app."""
    _WARM["state"] = "warming"
    sample = next(iter((STATIC / "samples").glob("*.mp4")), None)
    steps = [
        ("unified+bge", lambda: (INF._load_unified(), INF._bge("warmup"))),
        ("whisper", lambda: sample and EX._transcribe(str(sample))),
        ("siglip+clap", lambda: sample and EX.mm_features(str(sample))),
    ]
    for name, fn in steps:
        try:
            fn()
            _WARM["steps"].append(name)
        except Exception as e:
            _WARM["steps"].append(f"{name}: skipped ({type(e).__name__})")
    _WARM["state"] = "warm"


@app.on_event("startup")
def _on_start():
    if os.environ.get("SIP_WARMUP", "1") != "0":
        import threading
        threading.Thread(target=_warmup, daemon=True).start()


def _day_one(plays, likes) -> dict | None:
    """Map the two UI fields onto the day-1 dict Model B expects (missing counters default to 0)."""
    try:
        p = float(plays)
    except (TypeError, ValueError):
        return None
    if p <= 0:
        return None
    like = 0.0
    try:
        like = max(0.0, float(likes))
    except (TypeError, ValueError):
        pass
    return {"play_count": p, "like_count": like}


def _age_days(upload_date) -> float | None:
    """Video age in days from a YYYYMMDD upload_date. None if unparseable/missing."""
    if not upload_date:
        return None
    try:
        from datetime import datetime, timezone
        d0 = datetime.strptime(str(upload_date), "%Y%m%d").replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - d0).total_seconds() / 86400.0
    except Exception:
        return None


def _day1_age_guard(day_one: dict | None, upload_date) -> str | None:
    """MANUAL day-1 on an old URL: the numbers a user reads from the page are CURRENT cumulative
    counts, not the 24h snapshot Model B was trained on -> out-of-distribution. Warn when >2 days."""
    if not day_one:
        return None
    age = _age_days(upload_date)
    if age is not None and age > 2:
        return (f"This video was posted ~{int(age)} days ago. Model B needs the counts AT THE 24h "
                f"MARK, not current totals — the URL exposes only cumulative counts, so day-1 numbers "
                f"typed here are likely out-of-distribution. Get true day-1 via 'Track for "
                f"self-learning' (snapshots at 24h). Treat this Model B score as unreliable.")
    return None


def _score(inp: dict, cost_ratio: float | None, day_one: dict | None = None,
           notes: list | None = None, mm_feats: dict | None = None) -> dict:
    """Run THE unified classifier (hand + BGE + SigLIP + CLAP; + day-1 head only when the user
    opted in) and merge the extracted fields for display. Day-1 is never auto-activated."""
    extracted = inp.pop("extracted", {})
    inp.pop("upload_date", None)
    inp.pop("video_path", None)
    res = INF.predict_unified(inp, mm_feats=mm_feats, day_one=day_one, cost_ratio=cost_ratio)
    res["inputs"] = {"caption": inp.get("caption"), "duration_s": inp.get("duration_s"),
                     "aspect": inp.get("aspect")}
    res["extracted"] = extracted
    for n in (notes or []):
        if n:
            res.setdefault("warnings", []).append(n)
    return res


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "llm_backend": EX._backend(),
            "warmup": _WARM["state"], "warm_steps": _WARM["steps"]}


@app.post("/api/score/url")
def score_url(payload: dict) -> dict:
    """Published video: extract everything from the URL (metadata + Whisper transcript + SigLIP/CLAP
    on the downloaded file), then score with the unified classifier. Day-1 counts are used ONLY when
    the caller sends them (checkbox/manual) — never auto-activated."""
    url = (payload or {}).get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid http(s) video URL is required.")
    cost = payload.get("cost_ratio")
    d1 = payload.get("day1") or {}
    try:
        inp = EX.extract(url, language=payload.get("language"))
    except Exception as e:                                   # yt-dlp / network / rate-limit
        raise HTTPException(status_code=502, detail=f"Could not fetch or process the URL: {e}")
    upload_date = inp.get("upload_date")
    vpath = inp.get("video_path")                            # temp download reused for video features
    mm = None
    if vpath and os.path.exists(vpath):
        try:
            mm = EX.mm_features(vpath)
        finally:
            try:
                os.remove(vpath)
            except OSError:
                pass
    day_one = _day_one(d1.get("plays"), d1.get("likes"))     # None unless the user opted in
    notes = [_day1_age_guard(day_one, upload_date)] if day_one else []
    return _score(inp, cost, day_one, notes, mm_feats=mm)


@app.post("/api/score/file")
async def score_file(file: UploadFile = File(...), caption: str = Form(""),
                     post_time: str = Form(""), cost_ratio: float | None = Form(None),
                     plays: str = Form(""), likes: str = Form(""), language: str = Form("")) -> dict:
    """Unpublished video: user uploads the file + the caption they would post; extract the rest."""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    data = await file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_MB} MB).")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        inp = EX.extract(path, caption=caption or "", language=language or None)
        if post_time.strip():
            inp["post_time"] = post_time.strip()
        mm = EX.mm_features(path)                    # SigLIP + CLAP for the unified classifier
        return _score(inp, cost_ratio, _day_one(plays, likes), mm_feats=mm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------- self-learning flywheel
@app.post("/api/track")
def track_video(payload: dict) -> dict:
    """Start tracking a POSTED video for the self-learning loop (poll -> self-label -> retrain)."""
    url = (payload or {}).get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid http(s) video URL is required.")
    from sip import flywheel as FW
    res = FW.track(url, extract_content=bool(payload.get("extract_content", True)),
                   language=payload.get("language"))
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=res.get("error", "tracking failed"))
    return res


@app.get("/api/flywheel/status")
def flywheel_status() -> dict:
    from sip import flywheel as FW
    return FW.status()


@app.post("/api/flywheel/tick")
def flywheel_tick() -> dict:
    """Poll + label now (the button behind the demo); retraining stays a manual/CLI decision."""
    from sip import flywheel as FW
    out = FW.poll()
    return {"poll": out, "status": FW.status()}


@app.post("/api/probe")
def probe(payload: dict):
    """Auto-check a video URL BEFORE scoring: how old is it (older than a day?) and what are its
    current view/like counts — so the UI can decide whether the day-1 signal is usable and offer to
    auto-fill it (checkbox) instead of manual entry. `auto_ok` marks the ~1-day window where the
    current cumulative counts are a valid day-1 proxy (matches _auto_day1)."""
    url = (payload or {}).get("url", "").strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="A valid http(s) video URL is required.")
    from sip import flywheel as FW
    m = FW._meta(url)
    if not m:
        raise HTTPException(status_code=502, detail="Could not read the URL (rate-limited or unavailable).")
    age = _age_days(m.get("upload_date"))
    return {"ok": True, "upload_date": m.get("upload_date"),
            "age_days": round(age, 2) if age is not None else None,
            "over_one_day": bool(age is not None and age >= 1.0),
            "auto_ok": bool(age is not None and 1.0 <= age <= 2.0),
            "views": m.get("views"), "likes": m.get("likes")}


@app.get("/download/{which}")
def download(which: str):
    """Download the one deployed model — the unified classifier."""
    if which.lower() != "u" or not (MODELS / "unified.joblib").exists():
        return JSONResponse({"error": "use /download/u"}, status_code=404)
    return FileResponse(MODELS / "unified.joblib", filename="unified.joblib",
                        media_type="application/octet-stream")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webapp.server:app", host="127.0.0.1", port=8000, reload=False)
