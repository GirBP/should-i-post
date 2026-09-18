"""ShouldIPost? — minimal Heroku backend (torch-free, hand-feature Models A & B).

Two tiny models (no video, no embeddings → slim slug, no 30s-timeout risk):
  A (pre-publication): caption/duration/timing/meta → Post / Do-not-post / Unsure
  B (+day-1):          A's features + leak-free day-1 counts → Keep / Delete

Feature building reuses the bundled `sip` package so train and serve match exactly.
"""
import os, sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # bundled sip/
os.environ.setdefault("SIP_DEVICE", "cpu")
from sip import features as F                     # noqa: E402

A = joblib.load(HERE / "models" / "model_a_calc.joblib")
B = joblib.load(HERE / "models" / "model_b_calc.joblib")

app = FastAPI(title="ShouldIPost?", version="2.0")


def _row(caption, duration_s, post_iso, day1=None):
    """One-row lingbow-shaped frame from user input; add_derived + day-1 fill the rest."""
    try:
        dt = datetime.fromisoformat(post_iso) if post_iso else datetime.now(timezone.utc)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    df = pd.DataFrame([{
        "video_id": "live", "desc": caption or "", "transcript": "",
        "duration": float(duration_s or 0),
        "create_time": int(dt.timestamp()), "create_dt": pd.Timestamp(dt).tz_convert(None),
        "speaking_rate": 0.0,
        "anger": 0.0, "joy": 0.0, "surprise": 0.0, "sadness": 0.0, "disgust": 0.0, "fear": 0.0,
        "is_english": True, "created_by_ai": False, "is_ads": False, "ratio": 9 / 16,
    }])
    F.add_derived(df)
    if day1:                                        # match data.load: log1p + er = eng/plays
        plays = float(day1.get("plays") or 0); likes = float(day1.get("likes") or 0)
        eng = likes + float(day1.get("comments") or 0) + float(day1.get("shares") or 0)
        df["log_play_d1"] = np.log1p(plays)
        df["log_like_d1"] = np.log1p(likes)
        df["er_d1"] = eng / max(plays, 1.0)
    return df


def _verdict(art, df, kind):
    X, _ = F.build_blocks(df, art["blocks"], np.array([0]), np.array([0]), target_col=art["target"])
    p = float(art["model"].predict_proba(X)[:, 1][0])
    tl, th = art["bands"]["t_low"], art["bands"]["t_high"]
    if kind == "A":
        label = "Post" if p >= th else ("Do not post" if p <= tl else "Unsure")
    else:                                            # keep/delete: default KEEP, delete only clear duds
        label = "Delete" if p <= tl else "Keep"
    return {"probability": round(p, 4), "verdict": label, "bands": {"t_low": round(tl, 3), "t_high": round(th, 3)}}


@app.get("/api/health")
def health():
    return {"ok": True, "models": ["A", "B"], "features_A": len(A["feature_names"]), "features_B": len(B["feature_names"])}


@app.post("/api/score")
def score(payload: dict):
    cap = (payload or {}).get("caption", ""); dur = payload.get("duration"); t = payload.get("post_time", "")
    day1 = payload.get("day1") or None
    out = {"model_A": _verdict(A, _row(cap, dur, t), "A")}
    if day1 and (day1.get("plays") or day1.get("likes")):
        out["model_B"] = _verdict(B, _row(cap, dur, t, day1), "B")
    out["note"] = ("Pre-publication reach is information-bound (~0.58 AUC): the model nudges + abstains, "
                   "it is not a virality oracle. Day-1 signal (Model B, ~0.75) is the reliable decision.")
    return JSONResponse(out)


@app.get("/download/{which}")
def download(which: str):
    f = {"a": "model_a_calc.joblib", "b": "model_b_calc.joblib"}.get(which.lower())
    if not f:
        return JSONResponse({"error": "use /download/a or /download/b"}, status_code=404)
    return FileResponse(HERE / "models" / f, filename=f, media_type="application/octet-stream")


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
