#!/usr/bin/env python3
"""
Крок 1 (локально): завантажити реальні TikTok-відео за `video_id` з маніфесту.

Навіщо: окремого набору TikTok із самими відеофайлами не існує (ToS публікує лише ID),
тож стандарт для досліджень — узяти ID і докачати mp4 локально (yt-dlp).

Вхід:  data/multimodal/manifest.csv  (згенерований; колонки video_id,url,split,y,...)
Вихід: data/videos/<video_id>.mp4   +   data/multimodal/download_status.csv

Запуск (з кореня проєкту, у venv):
    python multimodal/01_download_videos.py --limit 500     # спершу проба
    python multimodal/01_download_videos.py                 # повний маніфест

Зауваження:
- Дані lingbow за 2024-06..11; станом на 2026 частина відео видалена/приватна →
  очікуй НЕПОВНЕ покриття. Це нормально; працюємо з тим, що завантажилось.
- Поважай ToS і приватність. Використовуй лише для дослідження/прототипу.
"""
import argparse, csv, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "multimodal" / "manifest.csv"
OUT = ROOT / "data" / "videos"
STATUS = ROOT / "data" / "multimodal" / "download_status.csv"


def download_one(row):
    import yt_dlp
    vid = str(row["video_id"]); url = row["url"]
    dst = OUT / f"{vid}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return vid, "cached"
    opts = {
        "outtmpl": str(OUT / f"{vid}.%(ext)s"),
        "format": "mp4/best[ext=mp4]/best",
        "quiet": True, "no_warnings": True, "noplaylist": True,
        "retries": 2, "socket_timeout": 20,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return vid, "ok" if dst.exists() else "missing_after_dl"
    except Exception as e:
        msg = str(e).lower()
        if "not available" in msg or "deleted" in msg or "private" in msg:
            return vid, "unavailable"
        return vid, f"error:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="скільки рядків (0 = всі)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--manifest", default=str(MAN))
    ap.add_argument("--status", default=str(STATUS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man_path, status_path = Path(args.manifest), Path(args.status)

    with open(man_path) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    done = {}
    if status_path.exists():
        with open(status_path) as f:
            done = {r["video_id"]: r["status"] for r in csv.DictReader(f)}
    todo = [r for r in rows if done.get(str(r["video_id"])) not in ("ok", "cached")]
    print(f"до завантаження: {len(todo)} / {len(rows)} (вже ок: {len(rows)-len(todo)})")

    results = dict(done)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            vid, st = fut.result(); results[vid] = st
            if i % 50 == 0:
                ok = sum(1 for v in results.values() if v in ("ok", "cached"))
                print(f"  {i}/{len(todo)} | усього ок: {ok}")
    with open(status_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id", "status"])
        for k, v in results.items():
            w.writerow([k, v])

    ok = sum(1 for v in results.values() if v in ("ok", "cached"))
    print(f"\nГотово. Покриття: {ok}/{len(rows)} = {ok/max(len(rows),1):.1%}")
    from collections import Counter
    print("статуси:", dict(Counter(results.values())))


if __name__ == "__main__":
    main()
