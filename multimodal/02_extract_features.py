#!/usr/bin/env python3
"""
Крок 2 (локально, Mac MPS/CPU): витягнути мультимодальні ознаки із завантажених відео.

Модальності:
  • ВІДЕО  — рівномірні кадри + окремо «гачок» (перші 3 c) → CLIP ViT-B/32 → ембедінги (pool).
  • АУДІО  — лог-мел статистики (mean/std по смугах) через ffmpeg+librosa (легко, без важких моделей).
  • ТЕКСТ  — desc+gpt_summary+transcript із lingbow → MiniLM (sentence-transformers).

Вихід: data/multimodal/features.parquet  (одна строка на video_id; колонки vemb*, hemb*, aud*, temb*).
Резюмабельно: вже оброблені video_id пропускаються.

Запуск:
    python multimodal/02_extract_features.py --limit 300      # проба
    python multimodal/02_extract_features.py                  # усі завантажені
"""
import argparse, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "data" / "videos"
RAW = ROOT / "data" / "raw" / "lingbow"
OUTF = ROOT / "data" / "multimodal" / "features.parquet"
N_FRAMES = 8
HOOK_SECONDS = 3.0


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def read_frames(path, n=N_FRAMES, hook_s=HOOK_SECONDS):
    """Повертає (uniform_frames[list PIL], hook_frames[list PIL])."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release(); return [], []
    uni_idx = np.linspace(0, total - 1, n).astype(int)
    hook_last = int(min(total - 1, hook_s * fps))
    hook_idx = np.linspace(0, max(hook_last, 1), max(n // 2, 2)).astype(int)
    want = sorted(set(uni_idx.tolist()) | set(hook_idx.tolist()))
    frames = {}
    for idx in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frames[idx] = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    uni = [frames[i] for i in uni_idx if i in frames]
    hook = [frames[i] for i in hook_idx if i in frames]
    return uni, hook


def audio_feats(path, n_mels=40):
    """Лог-мел статистики (mean+std) → 2*n_mels вектор. Тиша/без аудіо → нулі."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ac", "1", "-ar", "16000", wav],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, sr = librosa.load(wav, sr=16000)
        if y.size < sr // 2:
            return np.zeros(2 * n_mels, np.float32)
        m = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels))
        return np.concatenate([m.mean(1), m.std(1)]).astype(np.float32)
    except Exception:
        return np.zeros(2 * n_mels, np.float32)
    finally:
        if os.path.exists(wav):
            os.remove(wav)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    import torch, open_clip
    from sentence_transformers import SentenceTransformer

    dev = pick_device(); print("device:", dev)
    clip, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip = clip.to(dev).eval()
    txt_model = SentenceTransformer("all-MiniLM-L6-v2", device=dev)

    # текст із lingbow
    vtext = pd.read_parquet(RAW / "videos.parquet", columns=["video_id", "desc", "gpt_summary", "transcript"]).set_index("video_id")

    have = {p.stem for p in VID.glob("*.mp4")}
    done = set()
    if OUTF.exists():
        done = set(pd.read_parquet(OUTF, columns=["video_id"])["video_id"].astype(str))
    todo = sorted(have - done)
    if args.limit:
        todo = todo[: args.limit]
    print(f"відео на обробку: {len(todo)} (вже зроблено: {len(done)})")

    def embed_images(frames):
        if not frames:
            return np.zeros(512, np.float32)
        x = torch.stack([prep(f) for f in frames]).to(dev)
        with torch.no_grad():
            e = clip.encode_image(x).float().mean(0)
            e = e / (e.norm() + 1e-8)
        return e.cpu().numpy().astype(np.float32)

    rows, buf = [], []
    for i, vid in enumerate(todo, 1):
        path = VID / f"{vid}.mp4"
        uni, hook = read_frames(path)
        vemb = embed_images(uni); hemb = embed_images(hook)
        aud = audio_feats(path)
        t = vtext.loc[vid] if vid in vtext.index else None
        text = " ".join(str(t[c]) for c in ["desc", "gpt_summary", "transcript"] if t is not None and pd.notna(t[c])) if t is not None else ""
        buf.append((vid, vemb, hemb, aud, text))
        if len(buf) >= args.batch or i == len(todo):
            temb = txt_model.encode([b[4] for b in buf], batch_size=args.batch, show_progress_bar=False, normalize_embeddings=True)
            for (vid_, vemb_, hemb_, aud_, _), te in zip(buf, temb):
                r = {"video_id": vid_}
                r.update({f"vemb{j}": vemb_[j] for j in range(len(vemb_))})
                r.update({f"hemb{j}": hemb_[j] for j in range(len(hemb_))})
                r.update({f"aud{j}": aud_[j] for j in range(len(aud_))})
                r.update({f"temb{j}": float(te[j]) for j in range(len(te))})
                rows.append(r)
            buf = []
            # інкрементальний чекпойнт
            df = pd.DataFrame(rows)
            if OUTF.exists():
                df = pd.concat([pd.read_parquet(OUTF), df], ignore_index=True).drop_duplicates("video_id")
            df.to_parquet(OUTF, index=False); rows = []
            print(f"  {i}/{len(todo)} збережено (усього у файлі: {len(df)})")
    print("Готово:", OUTF)


if __name__ == "__main__":
    main()
