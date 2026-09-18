#!/usr/bin/env python3
"""Comprehensive, disk-bounded multimodal feature extractor.

For each video in the manifest, in chunks:
  1. ensure the mp4 (download with yt-dlp if missing),
  2. extract VIDEO + AUDIO features (one rich pass),
  3. checkpoint to data/multimodal/features.parquet,
  4. DELETE the mp4 (unless --keep) so disk stays bounded.

Resumable: video_ids already in features.parquet are skipped.

Feature blocks (column prefixes -> sip.features "mm:<prefix>"):
  vclip*  CLIP ViT-B/32 image embedding, mean-pooled over uniform frames (512)
  hclip*  CLIP over the HOOK window (first ~3 s)                          (512)
  vsig*   SigLIP ViT-B/16 image embedding over uniform frames            (768)
  vll_*   low-level visual: scene cuts, motion, brightness, colour, faces
  clap*   CLAP audio embedding                                           (512)
  aud_*   low-level audio: mel stats, RMS, ZCR, spectral, tempo, prosody, speech/music

Usage:
  python multimodal/extract.py --limit 40 --keep        # timed probe (keeps files)
  python multimodal/extract.py --chunk 200              # full run, disk-bounded
  python multimodal/extract.py --models clip,clap       # subset of encoders
"""
import argparse, os, subprocess, tempfile, time, warnings, gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
# prefer the stratified/shuffled manifest (balanced across split x label) if present
_MM = ROOT / "data" / "multimodal" / "manifest_mm.csv"
MAN = _MM if _MM.exists() else ROOT / "data" / "multimodal" / "manifest.csv"
VID = ROOT / "data" / "videos"
OUTF = ROOT / "data" / "multimodal" / "features.parquet"
STATUS = ROOT / "data" / "multimodal" / "extract_status.csv"
N_UNIFORM, N_HOOK, HOOK_S = 8, 4, 3.0


def pick_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------- download
def ensure_video(vid, url):
    import yt_dlp
    dst = VID / f"{vid}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return vid, "cached"
    opts = {"outtmpl": str(VID / f"{vid}.%(ext)s"), "format": "mp4/best[ext=mp4]/best",
            "quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True,
            "retries": 2, "socket_timeout": 20, "merge_output_format": "mp4"}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return vid, ("ok" if dst.exists() else "missing_after_dl")
    except Exception as e:
        m = str(e).lower()
        if "ip address is blocked" in m or "rate" in m:
            return vid, "blocked"
        return vid, ("unavailable" if any(k in m for k in ("not available", "deleted", "private")) else "dlerror")


# --------------------------------------------------------------- frames
def read_frames(path):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total <= 0:
        cap.release(); return [], [], [], {"fps": fps, "w": w, "h": h, "nframes": 0}
    uni_idx = np.linspace(0, total - 1, N_UNIFORM).astype(int)
    hook_last = int(min(total - 1, HOOK_S * fps))
    hook_idx = np.linspace(0, max(hook_last, 1), N_HOOK).astype(int)
    want = sorted(set(uni_idx.tolist()) | set(hook_idx.tolist()))
    raw = {}
    for idx in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            raw[idx] = fr  # BGR ndarray
    cap.release()
    uni = [Image.fromarray(cv2.cvtColor(raw[i], cv2.COLOR_BGR2RGB)) for i in uni_idx if i in raw]
    hook = [Image.fromarray(cv2.cvtColor(raw[i], cv2.COLOR_BGR2RGB)) for i in hook_idx if i in raw]
    seq = [raw[i] for i in sorted(raw)]  # BGR for low-level
    return uni, hook, seq, {"fps": fps, "w": w, "h": h, "nframes": total}


_FACE = None
def visual_lowlevel(seq, meta, duration):
    import cv2
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if not seq:
        return {f"vll_{k}": 0.0 for k in ["scene_cuts", "motion", "bright_mean", "bright_std",
                "colorful", "sat_mean", "face_frac", "face_max_area", "aspect", "nframes_log"]}
    grays = [cv2.cvtColor(cv2.resize(f, (160, 284)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in seq]
    diffs = [np.mean(np.abs(grays[i] - grays[i - 1])) / 255.0 for i in range(1, len(grays))]
    motion = float(np.mean(diffs)) if diffs else 0.0
    cuts = float(np.sum(np.array(diffs) > 0.20)) if diffs else 0.0
    bright = [g.mean() / 255.0 for g in grays]
    sats, colorful = [], []
    faces = 0; max_area = 0.0
    for f in seq:
        hsv = cv2.cvtColor(cv2.resize(f, (160, 284)), cv2.COLOR_BGR2HSV)
        sats.append(hsv[..., 1].mean() / 255.0)
        b, g, r = cv2.split(cv2.resize(f, (160, 284)).astype(np.float32))
        rg = np.abs(r - g); yb = np.abs(0.5 * (r + g) - b)
        colorful.append(float(np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)) / 255.0)
        det = _FACE.detectMultiScale(cv2.cvtColor(cv2.resize(f, (160, 284)), cv2.COLOR_BGR2GRAY), 1.2, 4)
        if len(det):
            faces += 1
            a = max((wd * ht) for (_, _, wd, ht) in det) / (160 * 284)
            max_area = max(max_area, float(a))
    dur = max(float(duration) if duration and duration == duration else 1.0, 1.0)
    return {
        "vll_scene_cuts": cuts / dur, "vll_motion": motion,
        "vll_bright_mean": float(np.mean(bright)), "vll_bright_std": float(np.std(bright)),
        "vll_colorful": float(np.mean(colorful)), "vll_sat_mean": float(np.mean(sats)),
        "vll_face_frac": faces / len(seq), "vll_face_max_area": max_area,
        "vll_aspect": (meta["w"] / meta["h"]) if meta["h"] else 0.0,
        "vll_nframes_log": float(np.log1p(meta["nframes"])),
    }


# --------------------------------------------------------------- audio
def extract_wav(path):
    """Return (y48 @48kHz for CLAP, y16 @16kHz for low-level, wavpath)."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "48000", "-t", "60", wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y48, _ = librosa.load(wav, sr=48000)
        y16 = librosa.resample(y48, orig_sr=48000, target_sr=16000) if y48.size else y48
        return y48, y16, wav
    except Exception:
        return None, None, wav


def audio_lowlevel(y, sr, n_mels=40):
    import librosa
    keys = (["aud_mel%d_m" % i for i in range(n_mels)] + ["aud_mel%d_s" % i for i in range(n_mels)] +
            ["aud_rms_m", "aud_rms_s", "aud_zcr_m", "aud_zcr_s", "aud_cent_m", "aud_cent_s",
             "aud_bw_m", "aud_roll_m", "aud_flat_m", "aud_tempo", "aud_hpr",
             "aud_f0_m", "aud_f0_s", "aud_voiced_frac", "aud_loud_range", "aud_silence"])
    if y is None or y.size < sr // 2:
        return {k: 0.0 for k in keys}
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels))
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    try:
        tempo = float(librosa.beat.beat_track(y=y, sr=sr)[0])
    except Exception:
        tempo = 0.0
    try:
        h, p = librosa.effects.hpss(y)
        hpr = float(np.sum(h**2) / (np.sum(p**2) + 1e-8))  # harmonic/percussive (speech vs music proxy)
    except Exception:
        hpr = 0.0
    try:
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
        f0_m, f0_s = (float(np.mean(f0)), float(np.std(f0))) if f0.size else (0.0, 0.0)
        voiced = float(np.mean((f0 > 70) & (f0 < 350))) if f0.size else 0.0
    except Exception:
        f0_m = f0_s = voiced = 0.0
    out = {}
    out.update({f"aud_mel{i}_m": float(mel[i].mean()) for i in range(n_mels)})
    out.update({f"aud_mel{i}_s": float(mel[i].std()) for i in range(n_mels)})
    out.update({"aud_rms_m": float(rms.mean()), "aud_rms_s": float(rms.std()),
                "aud_zcr_m": float(zcr.mean()), "aud_zcr_s": float(zcr.std()),
                "aud_cent_m": float(cent.mean()), "aud_cent_s": float(cent.std()),
                "aud_bw_m": float(bw.mean()), "aud_roll_m": float(roll.mean()),
                "aud_flat_m": float(flat.mean()), "aud_tempo": tempo, "aud_hpr": np.log1p(hpr),
                "aud_f0_m": f0_m, "aud_f0_s": f0_s, "aud_voiced_frac": voiced,
                "aud_loud_range": float(np.percentile(rms, 95) - np.percentile(rms, 5)),
                "aud_silence": float(np.mean(rms < 0.01))})
    return out


# --------------------------------------------------------------- model wrappers
class Encoders:
    def __init__(self, dev, which):
        self.dev = dev; self.which = which
        import torch; self.torch = torch
        if "clip" in which:
            import open_clip
            self.clip, _, self.clip_prep = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self.clip = self.clip.to(dev).eval()
        if "sig" in which:
            import open_clip
            self.sig, _, self.sig_prep = open_clip.create_model_and_transforms(
                "ViT-B-16-SigLIP", pretrained="webli")
            self.sig = self.sig.to(dev).eval()
        if "clap" in which:
            from transformers import ClapModel, ClapProcessor
            self.clap = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
            self.clap_proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def _img(self, model, prep, frames, dim):
        if not frames:
            return np.zeros(dim, np.float32)
        x = self.torch.stack([prep(f) for f in frames]).to(self.dev)
        with self.torch.no_grad():
            e = model.encode_image(x).float().mean(0)
            e = e / (e.norm() + 1e-8)
        return e.cpu().numpy().astype(np.float32)

    def clip_emb(self, frames):
        return self._img(self.clip, self.clip_prep, frames, 512)

    def sig_emb(self, frames):
        return self._img(self.sig, self.sig_prep, frames, 768)

    def clap_emb(self, y, sr):
        if y is None or y.size < sr // 2:
            return np.zeros(512, np.float32)
        with self.torch.no_grad():
            inp = self.clap_proc(audio=y, sampling_rate=sr, return_tensors="pt")
            inp = {k: v.to(self.dev) for k, v in inp.items()}
            out = self.clap.get_audio_features(**inp)
            e = out if isinstance(out, self.torch.Tensor) else out.pooler_output
            e = e.float()[0]
            e = e / (e.norm() + 1e-8)
        return e.cpu().numpy().astype(np.float32)


def extract_one(vid, enc, which, durations):
    path = VID / f"{vid}.mp4"
    if not (path.exists() and path.stat().st_size > 0):
        return None
    def _safe(fn, default):
        try:
            return fn()
        except Exception as e:
            print(f"   [{vid}] {fn.__name__ if hasattr(fn,'__name__') else 'step'} fail: "
                  f"{type(e).__name__} {str(e)[:50]}")
            return default
    try:
        uni, hook, seq, meta = read_frames(path)
        if not uni:
            return None
        row = {"video_id": vid}
        if "clip" in which:
            for j, x in enumerate(_safe(lambda: enc.clip_emb(uni), np.zeros(512, np.float32))): row[f"vclip{j}"] = x
            for j, x in enumerate(_safe(lambda: enc.clip_emb(hook), np.zeros(512, np.float32))): row[f"hclip{j}"] = x
        if "sig" in which:
            for j, x in enumerate(_safe(lambda: enc.sig_emb(uni), np.zeros(768, np.float32))): row[f"vsig{j}"] = x
        row.update(_safe(lambda: visual_lowlevel(seq, meta, durations.get(vid)), {}))
        y48, y16, wav = extract_wav(path)
        if "clap" in which:
            for j, x in enumerate(_safe(lambda: enc.clap_emb(y48, 48000), np.zeros(512, np.float32))): row[f"clap{j}"] = x
        row.update(_safe(lambda: audio_lowlevel(y16, 16000), {}))
        if os.path.exists(wav): os.remove(wav)
        return row
    except Exception as e:
        print(f"   extract fail {vid}: {type(e).__name__} {str(e)[:60]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default="clip,sig,clap")
    ap.add_argument("--keep", action="store_true", help="do not delete mp4 after extraction")
    ap.add_argument("--manifest", default=str(MAN))
    ap.add_argument("--no-download", action="store_true", help="extract only already-present mp4s")
    ap.add_argument("--polite", action="store_true",
                    help="download one-at-a-time with sleep + exponential backoff on IP block")
    ap.add_argument("--sleep", type=float, default=1.5, help="polite: base sleep between downloads (s)")
    args = ap.parse_args()
    which = set(args.models.split(","))
    VID.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest, dtype={"video_id": str})
    if args.limit:
        man = man.head(args.limit)
    durations = {}
    try:
        vdur = pd.read_parquet(ROOT / "data/raw/lingbow/videos.parquet", columns=["video_id", "duration"])
        durations = dict(zip(vdur["video_id"].astype(str), pd.to_numeric(vdur["duration"], errors="coerce")))
    except Exception:
        pass

    done = set()
    if OUTF.exists():
        done = set(pd.read_parquet(OUTF, columns=["video_id"])["video_id"].astype(str))
    todo = [r for _, r in man.iterrows() if str(r["video_id"]) not in done]
    print(f"to extract: {len(todo)} / {len(man)} (done {len(done)})  models={sorted(which)}")

    dev = pick_device(); print("device", dev)
    enc = Encoders(dev, which)
    t0 = time.time(); n_ok = 0; backoff = 60.0

    def flush(rows):
        if not rows:
            return
        df = pd.DataFrame(rows)
        if OUTF.exists():
            df = pd.concat([pd.read_parquet(OUTF), df], ignore_index=True).drop_duplicates("video_id")
        df.to_parquet(OUTF, index=False)

    for ci in range(0, len(todo), args.chunk):
        chunk = todo[ci:ci + args.chunk]
        # 1. acquire mp4s
        if not args.no_download:
            miss = [(str(r["video_id"]), r["url"]) for r in chunk
                    if not (VID / f"{r['video_id']}.mp4").exists()]
            if args.polite:
                for vid, url in miss:
                    st = ensure_video(vid, url)[1]
                    if st == "blocked":
                        print(f"  IP blocked -> backoff {backoff:.0f}s", flush=True)
                        time.sleep(backoff); backoff = min(backoff * 2, 1200)
                    else:
                        backoff = max(60.0, backoff * 0.8)
                        time.sleep(args.sleep)
            elif miss:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    list(ex.map(lambda a: ensure_video(*a), miss))
        # 2. extract present (serial on device); delete each after
        rows = []
        for r in chunk:
            vid = str(r["video_id"])
            row = extract_one(vid, enc, which, durations)
            if row is not None:
                rows.append(row); n_ok += 1
            if not args.keep:
                p = VID / f"{vid}.mp4"
                if p.exists():
                    p.unlink()
        flush(rows)
        gc.collect()
        rate = n_ok / max(time.time() - t0, 1e-6)
        print(f"  chunk {ci//args.chunk+1}/{(len(todo)+args.chunk-1)//args.chunk}: "
              f"total_ok={n_ok}  {rate:.2f} vid/s  elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"DONE. extracted {n_ok}. features -> {OUTF}")


if __name__ == "__main__":
    main()
