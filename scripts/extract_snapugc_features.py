"""Parallel feature extraction for the SnapUGC subset (the heavy 'prepare training' step).

Design (honest about where threads help):
  * DECODE (cv2 frame sampling + ffmpeg->wav) is CPU/subprocess-bound and releases the GIL,
    so it runs in a THREAD POOL — real parallelism.
  * ENCODE (CLIP/SigLIP/CLAP on MPS) is a single-GPU op, so it runs on the MAIN thread only
    (MPS is not thread-safe). Decode threads run ahead and overlap the GPU work.

Output: data/processed/snapugc_emb.parquet  (video_id + vclip0..511 + vsig0..767 + clap0..511)
Resumable: already-extracted video_ids are skipped. Run again to top up.

Usage:
    PYTHONPATH=src python scripts/extract_snapugc_features.py [WORKERS=6] [LIMIT=0]
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "multimodal"))
from sip import config as C  # noqa: E402
import extract as EX          # noqa: E402

SNAP = C.ROOT / "data" / "snapugc"
VID = SNAP / "videos"
OUT = C.PROC / "snapugc_emb.parquet"
COLS = ([f"vclip{i}" for i in range(512)] + [f"vsig{i}" for i in range(768)] + [f"clap{i}" for i in range(512)])
CHUNK = 400   # flush cadence / memory bound on in-flight decoded frames


def decode(vid):
    """CPU/subprocess work only — safe in a worker thread. Returns (vid, uni, y48) or None."""
    p = VID / f"{vid}.mp4"
    if not p.exists():
        return None
    try:
        uni, hook, seq, meta = EX.read_frames(p)
        if not uni:
            return None
        y48, y16, wav = EX.extract_wav(p)
        if wav and os.path.exists(wav):
            os.remove(wav)
        return (vid, uni, y48)
    except Exception:
        return None


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = EX.Encoders(dev, {"clip", "sig", "clap"})

    labels = pd.read_csv(SNAP / "labels.csv", dtype={"video_id": str})
    ids = labels["video_id"].astype(str).tolist()
    done = set()
    if OUT.exists():
        done = set(pd.read_parquet(OUT, columns=["video_id"])["video_id"].astype(str))
    todo = [v for v in ids if v not in done]
    if limit:
        todo = todo[:limit]
    print(f"to extract: {len(todo)} (already done {len(done)}), workers={workers}, dev={dev}", flush=True)

    buf, n_done, n_fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # bounded batches so in-flight decoded frames don't blow memory
        for start in range(0, len(todo), CHUNK):
            batch = todo[start:start + CHUNK]
            futures = [pool.submit(decode, v) for v in batch]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    n_fail += 1
                    continue
                vid, uni, y48 = res
                try:
                    vclip = enc.clip_emb(uni)                      # 512  (MPS, main thread)
                    vsig = enc.sig_emb(uni)                        # 768
                    clap = (enc.clap_emb(y48, 48000) if (y48 is not None and getattr(y48, "size", 0))
                            else np.zeros(512, np.float32))        # 512 (silent/no-audio -> zeros)
                    buf.append([vid] + np.concatenate([vclip, vsig, clap]).astype(np.float32).tolist())
                    n_done += 1
                except Exception:
                    n_fail += 1
            # flush chunk -> append to parquet
            if buf:
                df = pd.DataFrame(buf, columns=["video_id"] + COLS)
                if OUT.exists():
                    df = pd.concat([pd.read_parquet(OUT), df], ignore_index=True)
                df.to_parquet(OUT, index=False)
                buf = []
            print(f"  progress: {n_done} done, {n_fail} skipped -> {OUT.name}", flush=True)

    print(f"DONE: {n_done} embedded, {n_fail} skipped. total in parquet: "
          f"{len(pd.read_parquet(OUT)) if OUT.exists() else 0}", flush=True)


if __name__ == "__main__":
    main()
