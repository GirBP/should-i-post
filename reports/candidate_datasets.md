# Candidate datasets to improve the pre-publication predictor

Multi-agent search (8 discovery agents + 22 adversarial verifiers, 30 agents total): **63 candidates
discovered → 22 unique → 11 verified real & usable**. Mapped to our four missing-signal gaps:
**G1** raw pixels+audio / hook, **G2** watch-time / completion / retention, **G3** trend-sound-hashtag
momentum, **G4** cross-platform popularity for transfer.

> **Key framing (honest):** every watch-time dataset below is a **post-exposure interaction log** — it
> cannot supply *pre-publication* features directly. Its value is **transfer/pretraining**: train a
> content→retention/popularity head on its data, then apply that head to a new clip's pixels+audio.
> So the datasets that actually move *our* pre-publication model are the ones that ship **raw media**
> (SnapUGC, MicroLens) — those let "given an unposted video's content, predict engagement". The
> watch-time logs (KuaiRand/KuaiRec/VK-LSVD) are pretraining/validation sources, not feature sources.

## Tier 1 — highest leverage (raw media + engagement → directly improve Model A)

### 1. SnapUGC — best single fit (G1 + G2)
- Snapchat Spotlight short videos, 5–60 s, **raw .mp4 with audio**; ~90k (ECCV 2024) → 120,651 (VQualA 2025: 106k/6k/8.5k).
- Labels: **NAWP** (normalized average watch percentage) + **ECR** (engagement continuation ratio) — genuine watch-time/retention outcomes.
- Why it matters: one of the *very few* sets giving **both** raw pixels+audio (incl. first-seconds hook) **and** real retention labels — exactly our two top gaps.
- Use: pretrain a hook/retention encoder (raw frames+audio → NAWP/ECR), transfer/fine-tune to breakout. Single-platform; no G3/G4. License: no explicit file (Apache-2.0 badge on code) — confirm before redistribution.
- `https://github.com/dasongli1/SnapUGC_Engagement`

### 2. MicroLens-100K / -1M (G1; named in the task brief)
- Content micro-video set: **raw MP4 + extracted audio + cover image + title + comments + pre-extracted multimodal embeddings**; like/view popularity (static snapshot).
- 100K: 19,738 videos / 100k users / 719k interactions; full 1M: ~1M videos.
- Use: pretrain the content encoder on content→popularity, then transfer; or as an auxiliary multimodal head. **No watch-time (G2), no trend (G3).** Domain shift: unnamed platform, videos ~100–400 s (much longer than TikTok hooks).
- License: **academic/research-only (NOT commercial)** — fine for our prototype.
- `https://github.com/westlake-repl/MicroLens`

## Tier 2 — watch-time / completion pretraining (G2; post-exposure → transfer only)

### 3. KuaiRand + KuaiRec — gold standard for completion (G2, partial G3)
- Kuaishou interaction logs. **KuaiRand**: `play_time_ms` + `duration_ms` (→ completion ratio), `long_view`/`valid_play`/`complete_play`, 12 feedback signals, **+1.19M random-exposure (unbiased) impressions**. **KuaiRec**: `watch_ratio` core label + `item_daily_features` (per-video daily counts ≈ coarse momentum, weak G3).
- Use: pretrain a watch-time/completion head; confirm completion is the dominant missing signal. **No raw media (G1).** Chinese-platform OOD; CC-BY vs CC-BY-SA discrepancy to resolve.
- `https://zenodo.org/records/10439422` (KuaiRand) · `https://kuairec.com` (KuaiRec)

### 4. VK-LSVD (deepvk/VK-LSVD) (G2, scale)
- VK (Russian) short video: ~**40.8B** interactions, 10M users, 19.6M videos, 6 months, weekly parquet. `timespent` + `duration` → retention; explicit like/share/bookmark; **64-d content embeddings only (no raw media)**. Apache-2.0.
- Use: large-scale watch-time pretraining / sequence modeling. Single-platform, OOD.
- `https://huggingface.co/datasets/deepvk/VK-LSVD`

### 5. Mr. HiSum — "Most Replayed" (G2, intra-video / hook-shape)
- 31,892 YouTube videos with **frame-level "most replayed" retention curves** (`gtscore`, aggregated over 50k+ viewers) + YouTube-8M frame features (no raw media/audio). NeurIPS 2023, CC BY 4.0.
- Use: model *which seconds* get replayed → a **hook/retention-shape** signal; pretrain a "is the opening replay-worthy" head. Narrow but unique for the hook gap.
- `https://github.com/MRHiSum/MR.HiSum`

### 6. VGGSound (G1-audio, optional)
- ~200k 10 s YouTube clips with audio + 309 sound classes. CC BY 4.0. Use: audio-encoder pretraining — but we already use CLAP, so low marginal value.
- `https://www.robots.ox.ac.uk/~vgg/data/vggsound/`

## Checked & rejected (real, but wrong signals — honest negatives)
- **Trend (G3):** `ronantakizawa/tiktok-trending-hashtags` (annual granularity only), SNAP Twitter/Memetracker hourly (generic, off-platform), Spotify Charts (song streams, not TikTok sounds, 2017–21) → **none give post-time sound/hashtag momentum**; our internal lingbow trailing-window momentum remains the best G3 option.
- **Popularity/recsys benchmarks:** SMPD (image), SMTPD (temporal popularity, no pre-pub content signal), Tenrec (CTR/recsys, no media) — real but not usable for our gaps.
- **Generic pretraining corpora:** Panda-70M, HowTo100M, InternVid, HD-VILA-100M, AudioSet — real and large but **no engagement labels**; generic encoder pretraining offers little over the frozen CLIP/SigLIP/CLAP we already use, and 12 TB+ scale exceeds the $100/local budget.

## Recommendation (what to actually do next)
1. **SnapUGC → biggest expected lift.** Pretrain a raw-pixels+audio **hook/retention** encoder on NAWP/ECR, then fine-tune to breakout on lingbow. Directly attacks G1+G2 with real media + real retention — the combination we lack.
2. **MicroLens** as a second content→popularity pretraining source (G1), with the longer-video domain-shift caveat.
3. **KuaiRand/KuaiRec** to pretrain a completion head and to *empirically confirm* completion is the ceiling driver (G2) — transfer only, OOD.
4. **G3 stays internal** (lingbow momentum); no adequate external trend set exists at the needed granularity.
5. **G4** = treat SnapUGC + MicroLens + KuaiRand + VK-LSVD as a **multi-source transfer mix**, all out-of-distribution vs TikTok — measure transfer, don't assume it.
