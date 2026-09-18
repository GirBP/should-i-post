"""Best-effort video -> pre-publication features, feeding sip.inference.predict().

Turns a real, unposted video into the dict Model A expects, so the user can run the
end-to-end flow (upload / URL -> backend extracts features -> recommendation) with no
manual field entry.

Pipeline:
  metadata (yt-dlp / cv2)                       -> caption, duration, aspect, post_time
  audio (ffmpeg 16 kHz) -> faster-whisper       -> transcript
  LLM (transcript + caption)                    -> summary, topic, emotions, hook, cta

Design note (why this is enough): the deployed Model A's only content lever is the BGE
embedding of `caption + extra_text` (see sip.inference._feature_vector). So the extractor's
primary job is producing clean text -> extra_text = transcript + " " + summary. topic and
emotions are surfaced for interpretability; they are NOT inputs to the deployed Model A.

LLM backend is pluggable via env SIP_LLM:
  haiku     -> Anthropic Haiku          (dev / prompt tuning)
  deepseek  -> DeepSeek (OpenAI-compat)  (final product; cheap)
  none/auto -> skip LLM if no key        (transcript-only; never fails)
Keys via env: ANTHROPIC_API_KEY / DEEPSEEK_API_KEY. Nothing here raises on a missing
dependency or a bad API response — every step degrades to a safe default.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from . import inference as INF

# The 9 lingbow content categories (values as stored in canonical.parquet["topic"]).
TOPICS = ["Beauty_Fashion", "Cooking_Food", "Dance_Music", "Life_hacks_Personal_growth",
          "Lifestyle", "Movies_TV_Books", "Others", "Shopping_Products", "Sports_Fitness"]
EMOTIONS = ["anger", "joy", "surprise", "sadness", "disgust", "fear"]

_WHISPER = None


# --------------------------------------------------------------- audio -> transcript
def _transcribe(video_path: str, language: str | None = None, task: str = "transcribe",
                model_size: str | None = None) -> str:
    """ffmpeg (16 kHz mono) -> faster-whisper. Returns '' on any failure (no raise).
    `language` (e.g. 'uk') forces the spoken language so Whisper does not mis-detect it (a common
    Ukrainian->Russian error); None means auto-detect. task='translate' outputs English."""
    global _WHISPER
    wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav = tmp.name
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000",
                        "-t", "180", wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if _WHISPER is None:
            from faster_whisper import WhisperModel
            size = model_size or os.environ.get("SIP_WHISPER", "base")
            _WHISPER = WhisperModel(size, device="cpu", compute_type="int8")
        segments, _ = _WHISPER.transcribe(wav, beam_size=1, language=language, task=task)
        return " ".join(s.text.strip() for s in segments).strip()
    except Exception:
        return ""
    finally:
        if wav and os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass


# --------------------------------------------------------------- LLM backend (pluggable)
_SYS = (
    "You extract structured metadata from a short-form video's transcript for a virality model. "
    "Reply with ONLY one minified JSON object, no prose, no code fences. Schema: "
    '{"summary": string, "topic": one of ' + json.dumps(TOPICS) + ", "
    '"emotions": {"anger": 0..1, "joy": 0..1, "surprise": 0..1, "sadness": 0..1, '
    '"disgust": 0..1, "fear": 0..1}, "hook": 0..1, "cta": 0..1}. '
    'summary = <=200 chars describing what the video is about. hook = does the opening grab '
    'attention. cta = is there a call to action. If unsure use "Others" and 0.0.'
)
_FEWSHOT_U = ('Caption: morning routine ✨ #fyp\n'
              'Transcript: hey guys so today I want to show you my five minute morning routine to feel productive')
_FEWSHOT_A = ('{"summary":"A quick five-minute morning productivity routine.","topic":"Lifestyle",'
              '"emotions":{"anger":0,"joy":0.6,"surprise":0.1,"sadness":0,"disgust":0,"fear":0},'
              '"hook":0.4,"cta":0.2}')


def _ollama_up() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        return False


def _backend() -> str:
    b = (os.environ.get("SIP_LLM") or "auto").lower()
    if b in ("ollama", "gemini", "haiku", "deepseek", "none"):
        return b
    if _ollama_up():                              # auto: prefer the local model, then keys
        return "ollama"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "haiku"
    return "none"


def _call_llm(user: str) -> str:
    """Return the raw model text, or '' if the backend is unavailable/fails."""
    b = _backend()
    try:
        if b in ("deepseek", "gemini", "ollama"):
            from openai import OpenAI
            if b == "ollama":                         # local model, OpenAI-compatible endpoint
                client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
                model = os.environ.get("SIP_LLM_MODEL", "llama3.2:3b")   # fast local default; phi4 = quality
            elif b == "gemini":                       # Google AI Studio, OpenAI-compatible endpoint
                client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                                api_key=os.environ["GEMINI_API_KEY"])
                model = os.environ.get("SIP_LLM_MODEL", "gemini-2.0-flash")
            else:
                client = OpenAI(base_url="https://api.deepseek.com",
                                api_key=os.environ["DEEPSEEK_API_KEY"])
                model = os.environ.get("SIP_LLM_MODEL", "deepseek-chat")
            r = client.chat.completions.create(
                model=model,
                temperature=0, max_tokens=400,
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": _FEWSHOT_U},
                          {"role": "assistant", "content": _FEWSHOT_A},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content or ""
        if b == "haiku":
            import anthropic
            client = anthropic.Anthropic()
            r = client.messages.create(
                model=os.environ.get("SIP_LLM_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=400, temperature=0, system=_SYS,
                messages=[{"role": "user", "content": _FEWSHOT_U},
                          {"role": "assistant", "content": _FEWSHOT_A},
                          {"role": "user", "content": user}])
            return "".join(getattr(bl, "text", "") for bl in r.content)
    except Exception:
        return ""
    return ""


def _clip01(x, default=0.0):
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


def _parse(raw: str, transcript: str) -> dict:
    """Parse the LLM JSON; fall back to safe defaults so the pipeline never breaks."""
    default = {"summary": transcript[:160], "topic": "Others",
               "emotions": {e: 0.0 for e in EMOTIONS}, "hook": 0.0, "cta": 0.0,
               "llm_used": False}
    if not raw:
        return default
    try:
        m = re.search(r"\{.*\}", raw, re.S)          # tolerate stray text around the JSON
        d = json.loads(m.group(0) if m else raw)
        topic = d.get("topic") if d.get("topic") in TOPICS else "Others"
        em = d.get("emotions") or {}
        return {
            "summary": str(d.get("summary") or transcript[:160])[:300],
            "topic": topic,
            "emotions": {e: _clip01(em.get(e, 0.0)) for e in EMOTIONS},
            "hook": _clip01(d.get("hook")), "cta": _clip01(d.get("cta")),
            "llm_used": True,
        }
    except Exception:
        return default


def llm_extract(transcript: str, caption: str = "") -> dict:
    user = f"Caption: {caption or '(none)'}\nTranscript: {(transcript or '(none)')[:2000]}\nReturn JSON."
    return _parse(_call_llm(user), transcript or "")


# --------------------------------------------------------------- metadata (no cv2)
def _probe(path: str) -> dict:
    """duration + aspect via ffprobe — avoids importing cv2 alongside faster-whisper's `av`
    (they bundle conflicting libavdevice builds). Safe defaults on any failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out)
        s = (d.get("streams") or [{}])[0]
        w, h = s.get("width"), s.get("height")
        dur = (d.get("format") or {}).get("duration")
        return {"duration_s": float(dur) if dur else None,
                "aspect": (w / h) if (w and h) else None}
    except Exception:
        return {"duration_s": None, "aspect": None}


# --------------------------------------------------------------- source -> video file
def _fetch_to_file(url: str) -> str | None:
    """Best-effort download of a video to a temp mp4 via yt-dlp (TikTok may rate-limit)."""
    try:
        import yt_dlp
        out = str(Path(tempfile.gettempdir()) / "sip_dl_%(id)s.%(ext)s")
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "outtmpl": out,
                               "format": "mp4/best", "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception:
        return None


# --------------------------------------------------------------- multimodal features (video boost)
_MM = None


def _mm_module():
    """Lazy-load multimodal/extract.py (repo root) as a module; None if unavailable."""
    global _MM
    if _MM is None:
        try:
            import importlib.util
            p = Path(__file__).resolve().parents[2] / "multimodal" / "extract.py"
            spec = importlib.util.spec_from_file_location("mm_extract", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _MM = mod
        except Exception:
            _MM = False
    return _MM or None


_ENC = None


def mm_features(video_path: str) -> dict | None:
    """SigLIP frames + CLAP/low-level audio + low-level visual features for one local video file.
    Column names match data/multimodal/features.parquet (vsig*, clap*, aud_*, vll_*), so the vector
    feeds models/deployable_mm.joblib directly. Returns None if encoders/deps are unavailable —
    the caller then simply skips the video-boost score (graceful)."""
    global _ENC
    MM = _mm_module()
    if MM is None:
        return None
    try:
        if _ENC is None:
            dev = "cpu"
            try:
                import torch
                if not os.environ.get("SIP_DEVICE") == "cpu" and torch.backends.mps.is_available():
                    dev = "mps"
            except Exception:
                pass
            _ENC = MM.Encoders(dev, which=("sig", "clap"))
        uni, hook, seq, meta = MM.read_frames(video_path)
        if not uni:
            return None
        row = {}
        for j, x in enumerate(_ENC.sig_emb(uni)):
            row[f"vsig{j}"] = float(x)
        row.update(MM.visual_lowlevel(seq, meta, None))
        y48, y16, wav = MM.extract_wav(video_path)
        try:
            for j, x in enumerate(_ENC.clap_emb(y48, 48000)):
                row[f"clap{j}"] = float(x)
            row.update(MM.audio_lowlevel(y16, 16000))
        finally:
            if wav and os.path.exists(wav):
                try:
                    os.remove(wav)
                except OSError:
                    pass
        return row
    except Exception:
        return None


# --------------------------------------------------------------- main entry
def extract(source: str, caption: str = "", transcribe: bool = True, language: str | None = None) -> dict:
    """Video (local path or URL) -> features. Returns a dict with:
      - the predict()-ready keys: caption, duration_s, aspect, post_time, extra_text
      - `extracted`: {transcript, summary, topic, emotions, hook, cta, llm_used, language} for display
    `language`: None/'auto' auto-detect; a code like 'uk' forces the spoken language (avoids Whisper
    mis-detecting it); 'translate' transcribes to English. Never raises: every stage degrades safely.
    """
    lang, task = None, "transcribe"
    if language and language not in ("auto", ""):
        if language == "translate":
            task = "translate"
        else:
            lang = language

    is_url = str(source).startswith(("http://", "https://"))
    if is_url:
        meta = INF.from_url(source)                  # caption + duration + aspect from yt-dlp metadata
        vpath = _fetch_to_file(source) if transcribe else None
    else:
        meta = {"caption": caption, "post_time": None, **_probe(source)}   # ffprobe, no cv2
        vpath = source if transcribe else None
    cap = meta.get("caption") or caption or ""

    transcript = _transcribe(vpath, lang, task) if vpath else ""
    ex = llm_extract(transcript, cap)

    # extra_text feeds the BGE embedding — transcript + summary is the real content signal.
    extra_text = (transcript + " " + ex["summary"]).strip()

    inp = {"caption": cap, "duration_s": meta.get("duration_s"),
           "aspect": meta.get("aspect"), "post_time": meta.get("post_time"),
           "extra_text": extra_text or None}
    inp["upload_date"] = meta.get("upload_date")     # None for local files; drives the day-1 guard
    inp["video_path"] = vpath if is_url else None    # URL: temp download reused for SigLIP/CLAP; caller removes
    inp["extracted"] = {"transcript": transcript, **{k: ex[k] for k in
                        ("summary", "topic", "emotions", "hook", "cta", "llm_used")},
                        "backend": _backend(), "language": (language or "auto")}
    return inp
