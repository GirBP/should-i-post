"""Stateless inference: score a candidate TikTok BEFORE it is posted.

Accepts a TikTok URL, a local video file, or caption+metadata, validates and
degrades gracefully on partial input, and returns a calibrated recommendation
(Post / Do not post / Unsure) with the main factors and an explicit note on what
the model can and cannot know at prediction time.

Separate from training: loads models/deployable.joblib and recomputes only the
inference-reproducible features (deterministic caption/timing/duration/meta +
a public BGE text embedding).
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib
from . import config as C, features as F

_ARTIFACT = None
_BGE = None

_CANNOT_KNOW = (
    "What the model CANNOT see before posting: the first-seconds watch-time and "
    "completion rate, how the algorithm seeds the video, external trend timing, and "
    "the creator's live audience mood. Pre-publication virality is information-bound: "
    "the reliable signal arrives on day 1 (Model B). Treat this as triage, not an oracle."
)


def _load():
    global _ARTIFACT
    if _ARTIFACT is None:
        path = C.MODELS / "deployable.joblib"
        if not path.exists():
            raise FileNotFoundError("models/deployable.joblib missing — run experiments/build_deployable.py")
        _ARTIFACT = joblib.load(path)
    return _ARTIFACT


def _bge(text: str) -> np.ndarray:
    global _BGE
    art = _load()
    if _BGE is None:
        from sentence_transformers import SentenceTransformer
        import os, torch
        forced = os.environ.get("SIP_DEVICE")
        dev = forced or ("mps" if torch.backends.mps.is_available() else "cpu")
        _BGE = SentenceTransformer(art["bge_model"], device=dev)
    v = _BGE.encode([text or ""], normalize_embeddings=True, show_progress_bar=False)[0]
    return v.astype(np.float32)


# ---------------------------------------------------------------- input adapters
def _safe_float(x, default=np.nan):
    """Coerce to float; return default (NaN) for None/''/non-numeric — never raises."""
    if x is None or x == "":
        return default
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _detect_lang_en(text: str) -> float:
    if not text:
        return np.nan
    # cheap heuristic: ascii-letter ratio (avoids a heavy langdetect dep)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return np.nan
    return float(sum(c.isascii() for c in letters) / len(letters) > 0.8)


def from_url(url: str) -> dict:
    """Pull caption + duration from a TikTok URL via yt-dlp metadata (no full download)."""
    import yt_dlp
    info = {}
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        meta = ydl.extract_info(url, download=False)
    info["caption"] = meta.get("description") or meta.get("title") or ""
    info["duration_s"] = meta.get("duration")
    w, h = meta.get("width"), meta.get("height")
    info["aspect"] = (w / h) if (w and h) else np.nan
    info["post_time"] = None
    info["upload_date"] = meta.get("upload_date")     # YYYYMMDD; used only to warn about day-1 misuse
    return info


def from_file(path: str, caption: str = "") -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    cap.release()
    dur = (n / fps) if fps else np.nan
    return {"caption": caption, "duration_s": float(dur) if dur else np.nan,
            "aspect": (w / h) if h else np.nan, "post_time": None}


# ---------------------------------------------------------------- feature vector
def _feature_vector(inp: dict) -> tuple[np.ndarray, dict]:
    """Build the Model-A feature vector in the artifact's name order. Returns
    (vector, computed_caption_features) for explanation."""
    art = _load()
    caption = inp.get("caption")
    caption = "" if caption is None else (caption if isinstance(caption, str) else str(caption))
    pt = inp.get("post_time")
    if isinstance(pt, datetime):
        dt = pt
    elif isinstance(pt, str) and pt.strip():
        try:
            dt = datetime.fromisoformat(pt)
        except Exception:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    EM = F._EMOJI
    cap = {
        "char_len": len(caption), "word_len": len(caption.split()),
        "n_hashtags": len(re.findall(r"#\w+", caption)),
        "n_mentions": len(re.findall(r"@[\w.]+", caption)),
        "has_question": int("?" in caption), "has_exclam": int("!" in caption),
        "n_emoji": len(EM.findall(caption)), "has_url": int(bool(re.search(r"https?://|www\.", caption))),
        "has_cta": int(bool(F._CTA.search(caption))),
        "digit_ratio": sum(c.isdigit() for c in caption) / max(len(caption), 1),
        "caption_is_empty": int(len(caption) == 0),
        "allcaps_ratio": (sum(w.isupper() and len(w) > 1 for w in caption.split())
                          / max(len(caption.split()), 1)),
    }
    dur = _safe_float(inp.get("duration_s"))
    duration = {"duration_s": dur}
    for nm, lo, hi in [("dur_vshort", 0, 7), ("dur_short", 7, 15), ("dur_mid", 15, 30),
                       ("dur_long", 30, 60), ("dur_vlong", 60, 1e9)]:
        duration[nm] = float(lo <= dur < hi) if dur == dur else np.nan
    timing = {"hour_sin": np.sin(2 * np.pi * dt.hour / 24), "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
              "dow_sin": np.sin(2 * np.pi * dt.weekday() / 7), "dow_cos": np.cos(2 * np.pi * dt.weekday() / 7),
              "is_weekend": int(dt.weekday() >= 5),
              "month_sin": np.sin(2 * np.pi * dt.month / 12), "month_cos": np.cos(2 * np.pi * dt.month / 12)}
    asp = _safe_float(inp.get("aspect"))
    meta = {"is_english_i": _detect_lang_en(caption),
            "created_by_ai_i": 0.0, "is_ads_i": 0.0, "aspect": asp}
    blockvals = {**cap, **timing, **duration, **meta}
    # text embedding (live)
    extra = inp.get("extra_text")
    extra = "" if extra is None else (extra if isinstance(extra, str) else str(extra))
    text_for_emb = (caption + " " + extra).strip()
    bge = _bge(text_for_emb)
    vec = []
    for name in art["A"]["names"]:
        if name in blockvals:
            vec.append(blockvals[name])
        elif name.startswith(art.get("encoder_name", "bge")):
            vec.append(bge[int("".join(ch for ch in name if ch.isdigit()))])
        else:
            vec.append(np.nan)
    return np.array(vec, dtype=np.float32).reshape(1, -1), cap, bge


# ---------------------------------------------------------------- explanation
def _factors(cap: dict, inp: dict) -> list[dict]:
    art = _load(); signs = art.get("factor_signs", {})
    present = {
        "has_question": cap["has_question"], "has_exclam": cap["has_exclam"],
        "has_cta": cap["has_cta"], "n_emoji": min(cap["n_emoji"], 1),
        "caption_is_empty": cap["caption_is_empty"], "is_english_i": 1,
    }
    dur = inp.get("duration_s")
    out = []
    for f, on in present.items():
        if f in signs and on:
            s = signs[f]
            out.append({"factor": f, "direction": "↑" if s > 0 else "↓",
                        "corr": s, "note": _human(f, s)})
    if dur and dur == dur:
        band = next((n for n, lo, hi in [("very short (<7s)", 0, 7), ("short (7–15s)", 7, 15),
                     ("mid (15–30s)", 15, 30), ("long (30–60s)", 30, 60), ("very long (>60s)", 60, 1e9)]
                    if lo <= dur < hi), "?")
        out.append({"factor": "duration", "direction": "·", "corr": None,
                    "note": f"Duration {dur:.0f}s ({band})."})
    out.sort(key=lambda d: -abs(d["corr"]) if d["corr"] else 0)
    return out[:6]


def _human(f, s):
    pos = s > 0
    table = {
        "has_question": ("A question hook tends to help.", "A question hook tends not to help here."),
        "has_exclam": ("Excitement (!) is mildly positive.", "Excitement (!) is mildly negative here."),
        "has_cta": ("A call-to-action helps.", "A heavy call-to-action slightly hurts reach."),
        "n_emoji": ("Emoji presence is mildly positive.", "Emoji presence is mildly negative."),
        "caption_is_empty": ("Empty caption hurts.", "Empty caption is fine here."),
        "is_english_i": ("English caption.", "English caption."),
    }
    return table.get(f, ("", ""))[0 if pos else 1]


# ---------------------------------------------------------------- main API
def predict(inp: dict, day_one: dict | None = None, cost_ratio: float | None = None) -> dict:
    """Score one candidate. `inp` keys: caption, duration_s, aspect, post_time, extra_text.
    Optional `day_one` keys -> switches to the stronger Model B.
    `cost_ratio` = cost(false Post)/cost(missed hit); >1 makes "Post" more selective.
    Returns recommendation, calibrated probability, SHAP factors, example evidence, rationale."""
    art = _load()
    x, cap, bge = _feature_vector(inp)
    use = "B" if day_one else "A"
    m = art[use]
    if use == "B":
        x = _augment_dayone(x, m, day_one)
    p = float(m["cal"].predict_proba_pos(x)[0])
    bands = _bands_for_cost(m, cost_ratio)
    dec = _decide(p, bands, m["conformal"])
    shap_factors = _shap_factors(x, m)
    factors = shap_factors or _factors(cap, inp)        # SHAP if available, else correlation fallback
    return {
        "recommendation": dec, "probability": round(p, 4),
        "model": ("B (day-1 signal)" if use == "B" else "A (pre-publication)"),
        "bands": bands, "cost_ratio": cost_ratio, "factors": factors,
        "examples": _examples(bge),
        "what_model_cannot_know": _CANNOT_KNOW,
        "label_meaning": art["label_definition"],
        "calibrated": True, "warnings": _validate(inp),
        "rationale": _rationale(dec, p, factors, use),
    }


# ---------------------------------------------------------------- multimodal (video boost)
_ARTIFACT_MM = None


def _load_mm():
    global _ARTIFACT_MM
    if _ARTIFACT_MM is None:
        p = C.MODELS / "deployable_mm.joblib"
        _ARTIFACT_MM = joblib.load(p) if p.exists() else False
    return _ARTIFACT_MM or None


def predict_mm(inp: dict, mm_feats: dict, cost_ratio: float | None = None) -> dict | None:
    """Score with the multimodal Model A (text + SigLIP + CLAP), trained on 262 real videos.
    `mm_feats` comes from sip.extract.mm_features(file). Returns None when the artifact or the
    features are unavailable — callers treat the video boost as optional."""
    art = _load_mm()
    if art is None or not mm_feats:
        return None
    x_txt, cap, bge = _feature_vector(inp)            # shared tabular + bge values (by name below)
    base_names = _load()["A"]["names"]
    txtvals = dict(zip(base_names, x_txt.ravel()))
    emo = (inp.get("extracted") or {}).get("emotions") or {}
    arous = (emo.get("anger", np.nan) + emo.get("surprise", np.nan) + emo.get("fear", np.nan)
             if emo else np.nan)
    vec = []
    for nm in art["names"]:
        if nm in mm_feats:
            vec.append(_safe_float(mm_feats[nm]))
        elif nm in txtvals:
            vec.append(txtvals[nm])
        elif nm in ("anger", "joy", "surprise", "sadness", "disgust", "fear"):
            vec.append(_safe_float(emo.get(nm)))
        elif nm == "arousal":
            vec.append(_safe_float(arous))
        else:                                          # speaking_rate etc. -> unknown at inference
            vec.append(np.nan)
    X = np.array(vec, dtype=np.float32).reshape(1, -1)
    p = round(float(art["cal"].predict_proba_pos(X)[0]), 4)   # decide on the displayed precision
    bands = _bands_for_cost(art, cost_ratio) if "val_probs" in art else art["bands"]
    dec = _decide(p, bands, art.get("conformal"))
    m = art.get("metrics", {})
    return {
        "recommendation": dec, "probability": round(p, 4),
        "model": "A-mm (video boost: text + SigLIP frames + CLAP audio)",
        "bands": bands,
        "auc_note": (f"LOCO AUC {m.get('loco_auc')} CI {m.get('loco_ci')} on n={m.get('n')} real "
                     f"videos — directional boost, wide CI"),
        "caveat": art.get("caveat"),
    }


# ---------------------------------------------------------------- SHAP per-prediction
_EXPL = {}
def _shap_factors(x, m, topn=6):
    """Top SHAP contributors for this prediction; embedding dims aggregated into one
    'semantic text' bucket so the explanation stays human-readable. [] if shap fails."""
    try:
        import shap
        key = id(m["model"])
        if key not in _EXPL:
            try:
                _EXPL[key] = ("tree", shap.TreeExplainer(m["model"]))
            except Exception:
                bg = np.nan_to_num(np.asarray(m.get("shap_bg")))
                f = lambda d: m["model"].predict_proba(d)[:, 1]   # noqa: E731
                _EXPL[key] = ("kernel", shap.Explainer(f, bg))
        kind, expl = _EXPL[key]
        xi = np.nan_to_num(x)
        if kind == "tree":
            sv = expl.shap_values(xi)
            sv = sv[1] if isinstance(sv, list) else sv
        else:
            sv = expl(xi, max_evals=2 * x.shape[1] + 1).values
        sv = np.asarray(sv).reshape(-1)
        names = m["names"]; enc = _load().get("encoder_name", "bge")
        agg = {}
        for nm, val in zip(names, sv):
            key2 = "semantic text" if nm.startswith(enc) else nm
            agg[key2] = agg.get(key2, 0.0) + float(val)
        items = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:topn]
        return [{"factor": k, "direction": "↑" if v > 0 else "↓",
                 "shap": round(v, 4), "corr": round(v, 4),
                 "note": f"{_pretty(k)} {'raises' if v > 0 else 'lowers'} the score"} for k, v in items]
    except Exception:
        return []


def _pretty(n):
    return {"semantic text": "Caption/transcript content", "char_len": "Caption length",
            "n_hashtags": "Hashtag count", "has_question": "Question hook", "has_cta": "Call-to-action",
            "duration_s": "Duration", "is_weekend": "Weekend posting"}.get(n, n)


# ---------------------------------------------------------------- example-based evidence
def _examples(bge, k=3):
    """Nearest past videos (by content embedding) with their realised outcome."""
    art = _load(); bank = art.get("example_bank")
    if bank is None or bge is None:
        return []
    E = np.asarray(bank["emb"], dtype=np.float32)
    q = np.asarray(bge, dtype=np.float32)
    if E.shape[1] != q.shape[0]:
        return []
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    qn = q / (np.linalg.norm(q) + 1e-8)
    sims = En @ qn
    top = np.argsort(-sims)[:k]
    return [{"outcome": "hit" if bank["y"][i] == 1 else "flop",
             "similarity": round(float(sims[i]), 3),
             "caption": bank["caption"][i]} for i in top]


# ---------------------------------------------------------------- cost-ratio knob
def _bands_for_cost(m, cost_ratio):
    """Recompute Post/Don't bands from a cost ratio via the Bayes-optimal threshold on the
    calibrated validation set; default (None) returns the stored precision-tuned bands."""
    r = _safe_float(cost_ratio, default=None) if cost_ratio is not None else None
    if r is None or r <= 0 or "val_probs" not in m:     # invalid/absent -> stored precision-tuned bands
        return m["bands"]
    t_high = r / (1.0 + r)                               # Elkan optimal Post threshold
    t_low = min(m["bands"]["t_low"], t_high - 0.05)
    t_low = max(min(t_low, t_high), 0.01)               # keep 0 < t_low <= t_high
    return {"t_low": round(float(t_low), 4), "t_high": round(float(t_high), 4)}


def _augment_dayone(x, m, day_one):
    # append day-1 block columns in artifact order (inputs coerced safely)
    g = lambda k: max(_safe_float(day_one.get(k, 0), 0.0), 0.0)  # noqa: E731
    play = g("play_count"); pc = max(play, 1)
    eng = sum(g(k) for k in ("like_count", "comment_count", "share_count", "collect_count"))
    amed = _safe_float(day_one.get("author_median_log_play"), np.log1p(play))
    vals = {"log_play_d1": np.log1p(play), "log_like_d1": np.log1p(g("like_count")),
            "er_d1": eng / pc, "log_play_d1_vs_author": np.log1p(play) - amed}
    extra = [vals.get(n, np.nan) for n in m["names"] if n in vals]
    base = list(x.ravel())[: len(m["names"]) - len(extra)]
    return np.array(base + extra, dtype=np.float32).reshape(1, -1)


def _decide(p, bands, conformal):
    if p >= bands["t_high"]:
        return "Post"
    if p < bands["t_low"]:
        return "Do not post"
    return "Unsure"


def _validate(inp):
    w = []
    cap = inp.get("caption")
    if not (str(cap) if cap is not None else "").strip():
        w.append("No caption provided — relying on weak priors only; result is low-confidence.")
    if _safe_float(inp.get("duration_s")) != _safe_float(inp.get("duration_s")):  # NaN -> unknown
        w.append("Duration unknown — imputed.")
    if not str(inp.get("extra_text") or "").strip():
        w.append("No transcript/on-screen text — text embedding uses the caption only.")
    return w


def _rationale(dec, p, factors, use):
    lead = {"Post": "Worth posting", "Do not post": "Likely below this creator's bar",
            "Unsure": "Too close to call pre-publication"}[dec]
    pos = [f["note"] for f in factors if f["corr"] and f["corr"] > 0][:2]
    neg = [f["note"] for f in factors if f["corr"] and f["corr"] < 0][:2]
    bits = []
    if pos:
        bits.append("In favour: " + " ".join(pos))
    if neg:
        bits.append("Against: " + " ".join(neg))
    tail = ("Strong day-1 signal drives this." if use == "B"
            else "Pre-publication signal is inherently weak; consider posting organically and "
                 "letting the day-1 model decide whether to amplify.")
    return f"{lead} (p={p:.2f}). " + " ".join(bits) + " " + tail
