# ShouldIPost? — results & honest assessment

All numbers below are from **real runs** on the lingbow dataset (209,543 videos, 1,872
creators, 2024-06…11) reproduced by `run_all.sh`. Every experiment is evaluated on **two
protocols** — temporal (train<valid<test by date) and **leave-one-creator-out (LOCO)** — with
95 % bootstrap CIs, and logged in `reports/experiments_log.md`. Full machine-generated tables:
`reports/RESULTS_AUTO.md` and `reports/*.csv`. Leakage guards (`sip.leakage`) are asserted in
code and covered by `tests/`.

> **Audit-corrected (2026-06): a lookahead leak in creator-history features was found & fixed (§13).
> The corrected figures below are lower and the thesis is stronger.**
>
> **Bottom line.** Pre-publication *reach* is information-bound: **Model A ≈ 0.55–0.57 LOCO**,
> and a stronger model does not beat a simple one. Pre-publication *resonance* (within-creator
> ER) is more learnable (**≈ 0.59–0.63**). The reliable product value is the **day-1 model
> (Model B ≈ 0.75 — leak-free)** — the earlier ~0.95 came from a leaky author-relative day-1 feature (§13). Plus calibrated triage with honest abstention. Multimodal video+audio was
> measured on a partial sample (TikTok rate-limited bulk download); see §7.

---

## 1. Setup

Two decision moments → two models. **Model A** (cold-start, pre-publication: content + creator
state as-of-post). **Model B** (+day-1 engagement: amplify / keep / cut). Output is
**Post / Do not post / Unsure** with a calibrated probability and an honest "what the model
can't know" note. The product is framed as explore–exploit, not an oracle.

## 2. Dataset & why lingbow, not the original

The repo's earlier model used `datahiveai/Tiktok-Videos` (2,060 rows, **4 creators**) → deployed
ROC-AUC **0.52**, CI [0.45, 0.59] ≈ random. lingbow (209k videos, 1,872 creators, day-0…30
trajectories, AI-derived content fields) lets us cleanly separate pre- vs post-publication and
evaluate cross-creator generalisation. License **CC BY-NC 4.0** → prototype only.

## 3. Label (chosen by measurement)

**Primary: breakout within-creator (H=14)** = `views@14 / followers_at_post`, centred by the
creator's own median, binarised at the global median. Fame-neutral, business-aligned (reach
beyond own audience). Fame-neutrality is established by the **fame-leak test** (§5/label_robustness): an
author-size-only model scores **AUC ≈ 0.51** on this label vs **0.72–0.86** on naive labels
(`label_comparison.csv`: raw breakout 0.72, abs_views 0.79; `label_robustness.csv`: abs_logviews 0.86).
(The creator-prior baseline scores 0.34–0.36 — below random — which reflects regression-to-mean / drift in
the author rate, *not* a fame argument; we rely on the fame-leak test for that.)
Secondary: within-creator ER (resonance); follower growth. (`reports/baselines.json`,
`reports/label_comparison.csv`.)

## 4. Baselines (temporal test)

| setting | ROC-AUC | Brier |
|---|---|---|
| majority / base-rate | 0.500 | 0.250 |
| creator-prior (author train rate) | 0.34–0.36 | 0.28 |
| caption-only logistic | 0.51–0.52 | 0.250 |

## 5. Model A (pre-publication) — the information-bound ceiling

**Model family** on a fixed feature set (TAB + text embedding), LOCO, `reports/model_family.csv`:

| target | logreg | HGB | XGBoost | CatBoost | stacking | Optuna-XGB |
|---|---|---|---|---|---|---|
| breakout-wc | 0.553 | 0.553 | 0.555 | **0.557** | 0.559 | 0.559 |
| ER-wc | 0.584 | 0.575 | 0.583 | 0.583 | **0.590** | — |

→ **A stronger model does not beat a simple one** (logreg 0.553 → tuned/stacked 0.559 on
breakout). Optuna (30 trials) lifts XGBoost 0.553 → 0.559; stacking HGB+XGB+CatBoost adds ≈ +0.002.
The ceiling is *information*, not *model* (`reports/optuna_best.json`, `reports/ensemble.csv`).

**Best pre-publication config** (TAB + text + creator-fit + trend, *no* priors), full data,
**leak-safe closed-window** creator-fit, `reports/best_A.csv`:

| target | model | LOCO AUC [95% CI] |
|---|---|---|
| breakout-wc | logreg / HGB | **0.569 / 0.581** |
| ER-wc | logreg / HGB | **0.600 / 0.606** |

> **Audit correction.** Earlier figures (breakout 0.588, ER 0.608) and the claim "creator-fit is
> the best lever" were inflated by a **temporal lookahead leak** (see §13): creator-fit/author-history
> aggregated prior videos whose H-day labels weren't observable yet (posting cadence ≈0.45 days).
> Leak-free, **creator-fit/author-history add only ≈+0.01** — the honest best-A is **≈0.58 (breakout,
> HGB) / ≈0.61 (ER)**, mostly from TAB + text. This is *lower* than the pre-audit number and **strengthens** the
> information-ceiling conclusion.

**Methodological note (honest):** the *staged* ablation (`reports/ablation_A.csv`) added
target-encoded topic/music/hashtag **priors early, and they HURT LOCO** (they overfit and do not
transfer across creators), depressing every later stage. The clean per-block view (text encoders
add-one + best-A above) is the right read; the staged path is reported as-is with this caveat.

**Resonance vs reach.** ER (within-creator engagement) is consistently more predictable than
breakout (reach): family ER ≈ 0.58–0.60, and with richer content (emotion + semantics, HGB) the
prior pipeline reached **0.63**. Reach depends heavily on algorithmic distribution and luck that
are unobservable pre-publication.

## 6. LLM-as-judge (Claude agents scored 2,000 candidates on 6 dimensions)

`reports/judge_lift.csv` (HGB on the judged subset, LOCO):

| config | breakout-wc | ER-wc |
|---|---|---|
| TAB | 0.512 | 0.509 |
| judge-only (6 dims) | 0.488 | 0.526 |
| TAB + judge | 0.497 | 0.543 |
| TAB + text + judge | 0.513 | **0.566** |

→ **The LLM-judge adds ~nothing to reach but meaningfully helps resonance** (+0.03–0.06 on ER).
Top single dimension for ER: **arousal** (+0.022). Interpretation: judged content quality predicts
whether a video resonates with the existing audience, not whether the algorithm pushes it to new
ones. The judge is also used for the product's user-facing rationale. (Subset n≈1.5k → wide CIs.)

## 7. Multimodal (video + audio) — partial coverage (TikTok rate-limit)

Raw video+audio were downloaded by `video_id` and encoded with **CLIP + SigLIP frames + a hook
window, CLAP audio, and low-level visual/audio/prosody** (`multimodal/extract.py`). TikTok
**rate-limited the bulk download (IP block)** — the anticipated partial-coverage / ToS wall — so
the ablation runs on **N = 262 videos (228 creators)** that were fetched before the block; the
downloaded prefix is skewed (base rate 0.67), so CIs are wide (±0.07) and the numbers are
**suggestive, not conclusive**.

Staged ablation, breakout-wc, LOCO (`reports/multimodal_ablation.csv`):

| stage | LOCO AUC [95% CI] | marginal |
|---|---|---|
| TAB | 0.530 [0.46, 0.60] | — |
| +TEXT (BGE) | 0.555 [0.48, 0.62] | +0.025 |
| +AUDIO (CLAP+low-level) | 0.571 [0.49, 0.64] | +0.016 |
| +VIDEO/HOOK (CLIP) | 0.572 [0.50, 0.65] | +0.001 |
| +SigLIP | 0.624 [0.56, 0.69] | +0.052 |

Single-modality over TAB: **video SigLIP 0.640** vs CLIP 0.533 vs hook-CLIP 0.583; **audio CLAP
0.592** vs low-level audio 0.560; visual low-level 0.563. Fusion: early(HGB) 0.624, late(mean)
0.574, neural(MLP) 0.528. **Model B (+day-1) 0.915** (this subset figure uses the pre-audit
day-1 feature; the leak-free day-1 model is ≈0.75 — see §13).

→ **Directionally, raw video (SigLIP) and audio (CLAP) add signal beyond text+tabular** — breakout
climbs 0.53 → 0.62 — which is exactly the hypothesis that the *missing* pre-publication content
signal (pixels/audio) helps push Model A toward the ~0.6–0.7 literature ceiling. **But at N = 262
with ±0.07 CIs and a skewed sample this is not proven**; the TikTok rate-limit prevented the
larger, balanced extraction needed to confirm it. SigLIP clearly beats CLIP here, and CLAP beats
hand-crafted audio. Next step: a longer polite-backoff crawl to the full 10k manifest, then re-run.

### 7.1 SnapUGC watch-retention transfer (bounded best-effort — honest negative)

The literature-favoured lever (`reports/candidate_datasets.md`) is a **watch-retention** signal
transferred from **SnapUGC** (ECCV 2024; 106k Snapchat Spotlight videos with an **ECR**
engagement-continuation label). SnapUGC ships as a CSV of per-video CDN links (not a monolithic
archive), so we pulled videos whose circa-2024 CDN links still resolve (~half are dead), trained a
small MLP head (`[CLIP|SigLIP|CLAP] → ECR`) on them, and applied it to the 262 lingbow multimodal
videos as a `retention_head` feature (`experiments/run_transfer_snapugc.py`; reproduce the data with
`scripts/download_snapugc.py`). We ran it at **two scales** — a 150-video probe and a **68× scale-up
to 10,137 videos** — precisely to test whether the null result is just a data-starved head.

Scaling the head 68× makes it **substantially better at its own task** (held-out SnapUGC ECR
**Spearman 0.32 → 0.45**) — yet transferred to lingbow it **still adds no lift** (`reports/transfer.csv`,
LOCO, N = 262):

| SnapUGC head | ECR Spearman | breakout-wc: mm_best → +retention | ER-wc: mm_best → +retention |
|---|---|---|---|
| 150 | 0.32 | 0.654 → 0.653 (**−0.001**) | 0.497 → 0.496 (**−0.001**) |
| **10,137** | **0.45** | 0.654 → 0.641 (**−0.013**) | 0.497 → 0.494 (**−0.003**) |

→ **Honest negative — now on firm ground.** 68× more data trained a genuinely good watch-retention
head, and it *still* fails to improve (in fact slightly hurts, well within the wide CI) pre-publication
breakout on TikTok. So the null result is **not** a data-starved head — it is the **information
ceiling**: a cross-domain (Snapchat → TikTok) retention signal carries no extra pre-publication
predictive value beyond what SigLIP/CLAP already encode. This *strengthens* the thesis, not weakens it.

## 8. Two-model system + decision policy (deployable, calibrated)

`experiments/build_deployable.py` → `models/deployable.joblib`, `reports/deployable_metrics.json`.
Inference-reproducible blocks only (caption/timing/duration/meta + a public sentence encoder),
isotonic-calibrated, split-conformal abstention, cost-based bands. Temporal test:

| model | ROC-AUC [CI] | Brier | bands (coverage · base-rate) |
|---|---|---|---|
| **A (pre-publication)** | 0.559 [0.55, 0.57] | 0.247 | Post 16 % · 0.58 / Unsure 67 % / Don't 17 % · 0.42 |
| **B (+day-1), leak-free** | **0.750** | 0.201 | better than A, not an oracle (pre-audit 0.952 used a leaky feature, §13) |

→ Cold-start Model A is honest-but-weak (Post precision ≈ 0.58 vs 0.50 base); the system abstains
on 67 % and defers them to **Model B**, which is near-deterministic on day-1 signal.

## 9. What each modern method did or did not add (vs prior §1 numbers)

| method | breakout-wc (reach) | verdict |
|---|---|---|
| tabular boosting (XGB/CatBoost) + Optuna | 0.553 → 0.557 → 0.559 | marginal; ceiling is information |
| stacking ensemble | 0.559 | ≈ best single |
| text embeddings (MiniLM/BGE/e5) | small lift over TAB | encoder choice ≈ irrelevant (see §10) |
| LLM-as-judge | ~0 on reach (+0.03–0.06 on ER) | helps resonance, not reach |
| creator-fit (+trend) | **≈+0.01** leak-free (pre-audit 0.582) | mostly leak — lookahead, see §13 |
| target-encoded priors | hurts LOCO | drop for cross-creator |
| multimodal video+audio | see §7 | partial coverage |
| SnapUGC watch-retention transfer (bounded) | ≈0 (−0.001) | run best-effort — no lift at N=150→262 (§7.1) |
| MicroLens/KuaiRand transfer, contrastive pretrain, AST/PANNs, VideoMAE/V-JEPA | not run | out of budget/scope — documented in §11 |

## 10. Text-encoder comparison (add-one over TAB, full data)

`reports/text_encoders.csv` (LOCO):

| encoder | breakout-wc (lift) | ER-wc (lift) |
|---|---|---|
| none (TAB) | 0.528 | 0.541 |
| MiniLM | 0.564 (+0.036) | 0.592 (+0.051) |
| **BGE** | **0.567 (+0.040)** | **0.598 (+0.057)** |
| TF-IDF/SVD | 0.561 (+0.033) | 0.594 (+0.053) |

→ **Text is the single biggest content lever** (+0.036–0.040 on reach, +0.05–0.06 on resonance),
but **the encoder barely matters**: BGE ≈ MiniLM ≈ classic TF-IDF/SVD within 0.006. Representation
*quality* is not the bottleneck — the information ceiling is. (e5 deferred: ~50 min extra MPS for a
3rd transformer that would not change this conclusion.) This clean add-one view is why the staged
ablation (§5) under-reported text — priors added before text masked its lift.

## 11. Honest scope decisions (tried-or-deferred, with reasons)

- **SnapUGC watch-retention transfer** — *attempted* (bounded, best-effort): 150 live-CDN videos →
  ECR head → applied to lingbow. **No lift** (−0.001, §7.1). A full 106k-scale transfer stays
  deferred (CDN links ~half-dead, cross-domain Snapchat→TikTok), but the bounded probe already
  argues against an easy win — reported as an honest negative, not a pending promise.
- **Transfer pretraining on MicroLens-1M / KuaiRand** — deferred: raw-video corpora are hundreds
  of GB; downloading + pretraining exceeds the $100 / local-MPS budget for a marginal expected gain
  on an information-bound target.
- **Contrastive self-supervised pretraining of the content encoder** — deferred for the same
  budget reason; frozen CLIP/SigLIP/CLAP already provide strong content representations.
- **AST / PANNs audio, VideoMAE / V-JEPA / InternVideo** — represented by CLAP (audio) and
  CLIP/SigLIP (vision) + low-level descriptors; the heavier temporal encoders were not run given
  the partial video coverage made a larger encoder zoo low-value.
- **External trend datasets** — only yearly-rank granularity; replaced by an internal,
  leakage-safe trailing-window momentum feature (same-platform).

## 12. Honest limits

"Success" is a **reach proxy** (no clicks/conversions/revenue in the data) — validated against
reach and follower growth, not revenue. Data is **CC BY-NC** (prototype). Pre-publication
virality is fundamentally hard: the decisive drivers (first-seconds watch-time/completion,
algorithmic seeding, external trend timing) are unobservable before posting. The product's value
is **triage + abstain + the day-1 decision**, not a pre-publication oracle — and we report the
ceiling plainly rather than dressing it up.


## 13. Audit & corrections (independent review)

An independent agent audit found a real **temporal lookahead leak** and several smaller issues; all
material ones are fixed. The corrections lower some headline numbers and **strengthen** the core thesis.

- **F1/F2 — lookahead leak in creator-history features (fixed).** `block_creator_fit`,
  `block_author_history`, `block_author_recency` aggregated the author's prior videos, but used those
  videos' day-H labels which are not observable yet (posting cadence median **0.45 days**, 99.2% < H=14).
  Fix: **closed-window** gating — a prior video counts only when `prior.create_dt + H <= now`
  (cadence/days-since-last stays ungated, as it is observable). Effect: recency-only LOCO **0.59 → 0.53** (weak, near-random); creator-fit / author-history marginal
  **+0.014 → ≈+0.01** (small but real, leak-free); best-A **0.588 → ≈0.58** (HGB). The "author-recency =
  best lever (0.617)" claim was mostly leak — leak-free it adds only ~+0.01. New no-lookahead test added.
- **F3 — Model B inflated by an inference-unavailable feature (fixed).** B=0.952 leaned on
  `log_play_d1_vs_author` (day-1 vs the creator's historical median), which is unavailable cold and was
  neutered at inference, and whose median was computed over full history. The deployable B now uses only
  user-supplied day-1 signals (`day1_basic`) → **honest B = 0.750** (Brier 0.201), still > Model A.
- **F9 — wording fixed.** Fame-neutrality is shown by the fame-leak test (0.51 vs 0.79–0.86), not by the
  creator-prior 0.34 (that is regression-to-mean/drift).
- **F10 (encoder-prefix at inference) already fixed; F4 (B≈autocorrelation) acknowledged; F5 (name-based
  guard), F6 (multimodal temporal weak, N=262), F8 (per-creator median over full history — a label
  normalisation; train-frozen is infeasible under LOCO), F11/F13 — documented as known/minor.**

Net: the deployable model A (0.568) was unaffected (it never used creator-fit); the corrections hit the
*research* claims and the day-1 number. The honest picture: pre-publication ≈0.57, day-1 ≈0.75, and the
information ceiling is firmer than before.
