#!/usr/bin/env python3
"""
Докачати СИРИЙ набір lingbow/tiktok-video-engagement-200k у data/raw/lingbow/.

Навіщо: у пісочниці асистента Hugging Face заблоковано (proxy 403), тому повний
EDA «з нуля» і математичне порівняння міток (breakout = перегляди/підписники,
within-creator ER, приріст підписників) потребують, щоб 3 сирі таблиці лежали
локально. Запусти цей скрипт на СВОЇЙ машині (вона дістає HF).

Запуск (з кореня проєкту shouldipost/):
    .venv/bin/python scripts/download_raw_lingbow.py
або:
    python scripts/download_raw_lingbow.py

Розмір: ~кілька сотень МБ (engagement_daily — найбільша, ~6 млн рядків).
"""
from pathlib import Path
import shutil

REPO = "lingbow/tiktok-video-engagement-200k"
FILES = ["videos.parquet", "engagement_daily.parquet", "creator_daily.parquet"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "lingbow"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("Встанови: pip install huggingface_hub")

    OUT.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        print(f"↓ {f} ...", flush=True)
        cached = hf_hub_download(REPO, f, repo_type="dataset")
        dst = OUT / f
        if dst.resolve() != Path(cached).resolve():
            shutil.copy(cached, dst)
        print(f"  → {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
    print("Готово. Тепер сирі таблиці у", OUT)


if __name__ == "__main__":
    main()
