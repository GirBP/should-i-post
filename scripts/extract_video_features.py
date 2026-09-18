"""Parallel SigLIP/CLAP/CLIP feature extraction for a set of local videos.

Generic version of the SnapUGC extractor: point it at a video dir + an id list.
Decode (cv2/ffmpeg) runs in a thread pool; the MPS encoders run on the main thread.

    PYTHONPATH=src python scripts/extract_video_features.py \
        --videos data/videos --ids data/multimodal/manifest_newest.csv \
        --out data/multimodal/features_tiktok.parquet [--workers 8]

Resumable (skips ids already in --out). Columns: video_id + vclip0..511 + vsig0..767 + clap0..511.
"""
import os, sys, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "multimodal"))
import extract as EX  # noqa: E402

COLS = ([f"vclip{i}" for i in range(512)] + [f"vsig{i}" for i in range(768)] + [f"clap{i}" for i in range(512)])
CHUNK = 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "data" / "videos"))
    ap.add_argument("--ids", default=str(ROOT / "data" / "multimodal" / "manifest_newest.csv"))
    ap.add_argument("--out", default=str(ROOT / "data" / "multimodal" / "features_tiktok.parquet"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hook", action="store_true", help="use ONLY the first ~3s (hook) frames + audio")
    args = ap.parse_args()
    VID, OUT = Path(args.videos), Path(args.out)

    ids = pd.read_csv(args.ids, dtype={"video_id": str})["video_id"].astype(str).tolist()
    ids = [v for v in ids if (VID / f"{v}.mp4").exists() and (VID / f"{v}.mp4").stat().st_size > 10000]
    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT, columns=["video_id"])["video_id"].astype(str))
    todo = [v for v in ids if v not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"videos on disk: {len(ids)} | already extracted: {len(done)} | to do: {len(todo)}", flush=True)

    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = EX.Encoders(dev, {"clip", "sig", "clap"})

    def decode(vid):
        p = VID / f"{vid}.mp4"
        try:
            uni, hook, seq, meta = EX.read_frames(p)
            frames = hook if args.hook else uni
            if not frames:
                return None
            y48, y16, wav = EX.extract_wav(p)
            if wav and os.path.exists(wav):
                os.remove(wav)
            if args.hook and y48 is not None and getattr(y48, "size", 0):
                y48 = y48[: 3 * 48000]                      # first 3 s of audio for the hook
            return (vid, frames, y48)
        except Exception:
            return None

    buf, ok, fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(todo), CHUNK):
            batch = todo[start:start + CHUNK]
            for fut in as_completed([pool.submit(decode, v) for v in batch]):
                res = fut.result()
                if res is None:
                    fail += 1; continue
                vid, uni, y48 = res
                try:
                    vclip, vsig = enc.clip_emb(uni), enc.sig_emb(uni)
                    clap = (enc.clap_emb(y48, 48000) if (y48 is not None and getattr(y48, "size", 0))
                            else np.zeros(512, np.float32))
                    buf.append([vid] + np.concatenate([vclip, vsig, clap]).astype(np.float32).tolist())
                    ok += 1
                except Exception:
                    fail += 1
            if buf:
                df = pd.DataFrame(buf, columns=["video_id"] + COLS)
                if OUT.exists():
                    df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
                df.to_parquet(OUT, index=False)
                buf = []
            print(f"  {ok} done, {fail} skipped -> {OUT.name}", flush=True)
    print(f"DONE: {ok} embedded, {fail} skipped. total in {OUT.name}: "
          f"{len(pd.read_parquet(OUT)) if OUT.exists() else 0}", flush=True)


if __name__ == "__main__":
    main()
