# data/

This directory is not committed — it is rebuilt locally from a licensed dataset. Only this
file and `.gitkeep` markers live in git; everything else is produced by the scripts below.

## Source and license

[`lingbow/tiktok-video-engagement-200k`](https://huggingface.co/datasets/lingbow/tiktok-video-engagement-200k)
on Hugging Face — 209,543 videos, 1,872 creators, posted 2024-06 to 2024-11. Three tables:
`videos.parquet`, `engagement_daily.parquet`, `creator_daily.parquet`.

**License: CC BY-NC 4.0 — research / non-commercial use only.** Do not redistribute the raw
tables or downloaded video files, and do not use them or models trained on them commercially.

An optional transfer-learning subset (SnapUGC) has its own license; see
`scripts/download_snapugc.py` for terms and the download step.

## Reproduce locally

```bash
pip install -r requirements.txt          # needs huggingface_hub
python scripts/download_raw_lingbow.py   # -> data/raw/lingbow/*.parquet (a few hundred MB)
```

`sip.data` reads `data/raw/lingbow/` and builds the canonical frame (labels, day-1 fields,
temporal split) used by every experiment and by `./run_all.sh`.

For the multimodal track (real video/audio features), see `multimodal/README.md` — it
downloads video files by `video_id` into `data/videos/` and extracts SigLIP/CLAP/text features
into `data/multimodal/features.parquet`. The `data/multimodal/manifest*.csv` index files
(video_id/author_id/url + split, derived from the raw lingbow tables) are project-built
selections for that pipeline; they are not tracked in git and are not redistributed, in
keeping with the CC BY-NC terms.

## Layout (created by the scripts above, not by hand)

```
data/
  raw/lingbow/            videos.parquet, engagement_daily.parquet, creator_daily.parquet
  processed/               canonical frame built by sip.data
  videos/                  downloaded mp4s, by video_id (multimodal track)
  multimodal/
    manifest*.csv          video_id/author_id/url index for the multimodal pipeline
    features.parquet       extracted SigLIP/CLAP/text features
  snapugc/                 optional transfer-learning subset (own license)
  flywheel.db              runtime self-learning store, auto-created on first use
```

## Demo videos in the web app

The demo clips used during development are real videos from the lingbow dataset above, so they
are not redistributed in this repository. `webapp/static/samples/README.md` explains how to add
your own clip for the demo button and the startup warmup.
