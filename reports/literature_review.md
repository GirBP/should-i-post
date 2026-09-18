# Literature review — pre-publication virality prediction (grounding our decisions)

Adversarial multi-agent review (8 question-agents → 56 findings → 16 load-bearing claims
verified against **primary sources**, 15/16 confirmed). Every number below was checked by a
verifier agent that fetched the paper. Purpose: replace the inherited "0.6–0.7 ceiling"
assumption with grounded evidence and map it to our results.

## A1 — Is the pre-publication ceiling real? **Yes, and our numbers match the literature.**
- **Pure content-only cold-start is weak.** Khosla et al., *What Makes an Image Popular?* (WWW 2014):
  content-only rank correlation **Spearman 0.31–0.40**; only with **social/author cues** does it reach
  up to **0.81**. (verified vs PDF)
- *On the Feasibility of Predicting News Popularity at Cold Start* (Arapakis et al., SocInfo 2014):
  best content-only classifier **acc 0.797 vs 0.703 majority** → ~9% lift, authors call it
  "essentially infeasible". (verified)
- **SMP Challenge** (Wu et al., ACM MM 2019/overview): winners **SRC 0.59 → 0.77** across years, but
  using **author history + content + multimodal** ensembles (CatBoost), not content-only. (verified)
- **Mapping to us:** our content+creator-fit Model A is **ROC-AUC 0.588 (breakout) / 0.61 (ER)** — squarely
  in the *content-only* regime of the literature. The higher 0.7–0.8 numbers come from adding author/history
  and multimodal — i.e. exactly the levers we measured (author-history +0.014; multimodal SigLIP+CLAP
  directional 0.53→0.62). **The "0.6–0.7 ceiling" is real but is an author+multimodal number, not a
  content-only one.** Our honest framing was correct and is now grounded.

## A3 — Is watch-time/completion the dominant driver? **Yes — strongest single signal, and it's pre-pub-predictable.**
- *Beyond Views* (Wu, Rizoiu, Xie): channel-reputation (author past) **R²=0.42**, all pre-upload features
  **R²=0.449**, but **watch-percentage R²=0.77** — watch-time dwarfs everything. (verified)
- *Delving Deep into Engagement Prediction of Short Videos* (SnapUGC, 90k Snapchat): watch% (NAWP)
  **predictable from pre-post content at SRCC 0.696** (ECR 0.675). (verified) → **this is exactly the P3.2
  transfer opportunity**: a content→watch% head is learnable and would inject the dominant missing signal.
- *Slapping Cats…* (TikTok virality study): creator-profile RF **AUC 0.86** alone vs content 0.81, combined
  0.93 — but creator-profile = audience size, i.e. **the fame component** (consistent with our fame-leak finding).
- **Early-window determinacy:** SEISMIC (Hawkes) predicts final cascade size at **~15% error after 1 hour**;
  HIP forecasts day 91–120 from days 1–90; SMTPD: adding **day-1** early popularity lifts day-30 **SRC 0.85→0.959**.
  → mirrors our Model A 0.59 → **Model B (day-1) 0.95**: the outcome is front-loaded, the reliable signal is early.

## A2 — Label definition. **Our within-creator label is the fame-neutral, leakage-safe choice.**
- Wu et al. 2016 (*Unfolding Temporal Dynamics*) canonical **age-normalized** popularity `s = log2(r/d)+1`
  (r=views, d=days since post); restated by Khosla 2014, DTCN 2017, and a 2024 contrastive-SPP paper. (verified)
- We achieve the **same age control by construction** (fixed horizon H: views@H), and our `run_label_robustness`
  shows the label is **horizon-stable** (AUC ~0.580 at H=7/10/14/21; 96–99% agreement) — so Wu normalization is
  moot here, as predicted.
- Our fame-leak experiment confirms the danger the literature implies: naive labels score high only via fame —
  within-creator **content 0.578 / fame-leak 0.514**; global-top30 **0.79 / 0.87**; abs-views **0.90 / 0.93**.

## A4 — Horizon. **Settled:** within-creator breakout is horizon-insensitive (above) → H=14 is a safe default.

## A5 — Author-history feature. **Legitimate, helps a little, fame parried by the label.**
- Cold-start literature (NxtPost, SocRipple, *Beyond Views* channel-reputation R²=0.42) treats the **author's
  own past outcomes** as the strongest legitimate pre-publication signal — not leakage (known before this post).
- Our `block_author_history`: +0.014 AUC on breakout (0.578→0.592), +0.003 on ER; **alone it scores 0.513**
  (≈ fame-leak) → the within-creator label correctly prevents it from becoming a "big-author" shortcut.

## Methodology grounding (confirms our protocol)
- **Leakage:** Ji et al., *A Critical Study on Data Leakage in RecSys Offline Evaluation* — only
  **split-by-timepoint (temporal)** avoids leakage; random / leave-one-out / by-ratio / by-user all leak.
  → validates our temporal + LOCO discipline and the strict "no post-pub features in Model A" rule.
- **Threshold/calibration:** scikit-learn *Post-tuning the decision threshold for cost-sensitive learning*
  (TunedThresholdClassifierCV) — default 0.5 is wrong under imbalance/asymmetric cost; Elkan-optimal threshold
  ≈ cost ratio. → validates our calibrated, cost-band Post/Unsure/Don't decision policy.

## Net effect on the project
Nothing in the literature contradicts our build; it **confirms** the hard ceiling, the watch-time gap, the
label choice, the leakage protocol, and the threshold policy — and it pinpoints the **one high-value next move**:
a **content→watch%/retention head pretrained on SnapUGC** (NAWP/ECR SRCC ~0.70) transferred to lingbow, which is
the literature-backed way to inject the dominant missing signal (P3.2). Sources are listed inline; full verifier
notes in the run transcript.
