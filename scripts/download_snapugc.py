"""Download a bounded SnapUGC subset for the watch-retention transfer (P3.2).

SnapUGC (ECCV 2024, https://github.com/dasongli1/SnapUGC_Engagement) ships its data as a
CSV of Snapchat-CDN links (not a monolithic archive). The train split exposes an **ECR**
(engagement-continuation-rate) label per video. Circa-2024 CDN links have partial liveness
(~half still resolve), so we pull a bounded, best-effort subset rather than all 106k.

Usage:
    PYTHONPATH=src python scripts/download_snapugc.py [N_TARGET=150] [MAX_TRY=450]

Produces:
    data/snapugc/videos/<id>.mp4
    data/snapugc/labels.csv   (video_id, nawp, ecr)  -- nawp mirrors ecr (only ECR is published)

Then run the transfer/ablation:
    PYTHONPATH=src python experiments/run_transfer_snapugc.py
"""
import csv, sys, urllib.request, concurrent.futures as cf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "snapugc"
VID = OUT / "videos"
TRAIN_GDRIVE_ID = "1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD"   # SnapUGC train CSV (Id,Title,Description,Download_link,ECR)


def ensure_train_csv() -> Path:
    csv_path = OUT / "snap_train.csv"
    if csv_path.exists() and csv_path.stat().st_size > 10000:
        return csv_path
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        sys.exit("pip install gdown, or manually place the SnapUGC train CSV at " + str(csv_path))
    gdown.download(id=TRAIN_GDRIVE_ID, output=str(csv_path), quiet=False)
    return csv_path


def fetch(row):
    vid, link = row["Id"].strip(), row["Download_link"].strip()
    try:
        ecr = float(row["ECR"])
    except (ValueError, KeyError):
        return None
    dst = VID / f"{vid}.mp4"
    if dst.exists() and dst.stat().st_size > 10000:
        return (vid, ecr)
    try:
        req = urllib.request.Request(link, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=20).read()
        if len(data) < 10000:
            return None
        dst.write_bytes(data)
        return (vid, ecr)
    except Exception:
        return None


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    max_try = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    VID.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ensure_train_csv())))
    stride = max(1, len(rows) // max_try)               # deterministic spread across the file
    cand = rows[::stride][:max_try]
    existing = {p.stem for p in VID.glob("*.mp4") if p.stat().st_size > 10000}
    cand = [r for r in cand if r["Id"].strip() not in existing]   # only NEW rows -> target = new downloads

    got = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(fetch, cand):
            if res:
                got.append(res)
            if len(got) >= target:
                break

    # Rebuild labels.csv from ALL videos on disk so it accumulates across runs/strides.
    ecr_by_id = {r["Id"].strip(): r["ECR"] for r in rows if r.get("ECR")}
    on_disk = [p.stem for p in VID.glob("*.mp4") if p.stat().st_size > 10000]
    labeled = [(vid, ecr_by_id[vid]) for vid in on_disk if vid in ecr_by_id]
    with open(OUT / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "nawp", "ecr"])
        for vid, ecr in labeled:
            w.writerow([vid, ecr, ecr])                 # nawp mirrors ecr (only ECR published)
    print(f"this run fetched {len(got)} new live videos")
    print(f"labels.csv now lists {len(labeled)} on-disk videos -> {OUT / 'labels.csv'}")


if __name__ == "__main__":
    main()
