#!/usr/bin/env python3
"""Patient, gentle, download-ONLY pre-fetcher for the multimodal subset.

TikTok rate-limits sustained bursts (IP block). This downloads manifest_mm videos
one at a time with a polite delay and long exponential backoff when blocked, so it
accumulates whatever it can over time without hammering. No GPU — pairs with
`multimodal/extract.py --no-download` which consumes + deletes the files.

Usage:  python multimodal/polite_download.py --cap 3000 --sleep 2.5
"""
import argparse, os, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "multimodal" / "manifest_mm.csv"
VID = ROOT / "data" / "videos"


def dl(vid, url):
    import yt_dlp
    dst = VID / f"{vid}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return "cached"
    opts = {"outtmpl": str(VID / f"{vid}.%(ext)s"), "format": "mp4/best[ext=mp4]/best",
            "quiet": True, "no_warnings": True, "noplaylist": True, "noprogress": True,
            "retries": 1, "socket_timeout": 20, "merge_output_format": "mp4"}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return "ok" if dst.exists() else "missing"
    except Exception as e:
        m = str(e).lower()
        if "blocked" in m or "rate" in m:
            return "blocked"
        return "unavailable" if any(k in m for k in ("not available", "deleted", "private")) else "err"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3000, help="stop after this many on disk")
    ap.add_argument("--sleep", type=float, default=2.5)
    args = ap.parse_args()
    VID.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(MAN, dtype={"video_id": str}).to_dict("records")
    backoff = 300.0
    got = sum(1 for p in VID.glob("*.mp4"))
    print(f"start: {got} on disk; target cap {args.cap}", flush=True)
    for r in rows:
        if sum(1 for _ in VID.glob("*.mp4")) >= args.cap:
            print("cap reached", flush=True); break
        vid = str(r["video_id"])
        if (VID / f"{vid}.mp4").exists():
            continue
        st = dl(vid, r["url"])
        if st == "blocked":
            print(f"blocked -> sleep {backoff:.0f}s", flush=True)
            time.sleep(backoff); backoff = min(backoff * 1.7, 1800)
        elif st == "ok":
            got += 1; backoff = max(300.0, backoff * 0.7)
            if got % 50 == 0:
                print(f"  got {got}", flush=True)
            time.sleep(args.sleep)
        else:
            time.sleep(args.sleep)
    print(f"done: {sum(1 for _ in VID.glob('*.mp4'))} on disk", flush=True)


if __name__ == "__main__":
    main()
