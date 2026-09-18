"""Live terminal dashboard for the TikTok download — run and watch.

    PYTHONPATH=src python scripts/watch_download.py

Shows, refreshing every ~3s: downloaded vs target, class balance, current rate
(videos/s + MB/s over the last interval), disk free, and an ETA. Ctrl-C to quit.
Read-only — it just watches data/videos/ and the manifest, never touches the download.
"""
import os, time, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "data" / "videos"
MAN = ROOT / "data" / "multimodal" / "manifest_newest.csv"
TARGET = 10000
BAR = 42


def bar(frac, width=BAR, ch="█", empty="░"):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return ch * n + empty * (width - n)


def main():
    man = pd.read_csv(MAN, dtype={"video_id": str})
    ids = man["video_id"].tolist()
    ylab = dict(zip(man["video_id"], man["y"]))
    def scan():
        p = [v for v in ids if (VID / f"{v}.mp4").exists() and (VID / f"{v}.mp4").stat().st_size > 10000]
        return p, sum((VID / f"{v}.mp4").stat().st_size for v in p)
    _p0, _b0 = scan()
    prev_n, prev_bytes, prev_t = len(_p0), _b0, time.time()   # seed so the first frame's rate is 0, not bogus
    try:
        while True:
            present = [v for v in ids if (VID / f"{v}.mp4").exists()
                       and (VID / f"{v}.mp4").stat().st_size > 10000]
            n = len(present)
            pos = sum(1 for v in present if ylab.get(v) == 1)
            neg = n - pos
            total_bytes = sum((VID / f"{v}.mp4").stat().st_size for v in present)
            now = time.time(); dt = max(now - prev_t, 1e-6)
            vps = (n - prev_n) / dt
            mbps = (total_bytes - prev_bytes) / dt / 1e6
            prev_n, prev_bytes, prev_t = n, total_bytes, now
            free_gb = shutil.disk_usage(ROOT).free / 1e9
            remain = max(TARGET - n, 0)
            eta = remain / vps / 60 if vps > 0.05 else float("inf")

            os.system("clear" if os.name != "nt" else "cls")
            print("  ┌─────────────────────────────────────────────────────────────┐")
            print("  │        ShouldIPost — завантаження TikTok (live)             │")
            print("  └─────────────────────────────────────────────────────────────┘\n")
            print(f"  Найновіші балансовані:  {n:,} / {len(ids):,}  (ціль {TARGET:,})")
            print(f"  [{bar(n/len(ids))}]  {n/len(ids)*100:4.1f}%\n")
            bmax = max(pos, neg, 1)
            print(f"  ✅ зайшло  (pos)  {pos:>5,}  [{bar(pos/bmax, 30)}]")
            print(f"  ❌ провал  (neg)  {neg:>5,}  [{bar(neg/bmax, 30)}]")
            skew = "збалансовано" if min(pos, neg) / max(pos, neg, 1) > 0.85 else "⚠ перекіс (догойдовує клас)"
            print(f"     баланс: {skew}\n")
            print(f"  Темп:      {vps:5.1f} відео/с   ·   {mbps:6.1f} MB/с")
            print(f"  Диск:      {free_gb:5.1f} GB вільно")
            eta_s = f"{eta:4.0f} хв" if eta != float('inf') else "—"
            print(f"  ETA до {TARGET:,}: {eta_s}")
            print("\n  (Ctrl-C щоб вийти · це лише монітор, завантаження не чіпає)")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nвихід.")


if __name__ == "__main__":
    main()
