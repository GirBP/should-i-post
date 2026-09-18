#!/usr/bin/env python3
"""Assemble notebooks/RESULTS.ipynb — a single, scientific, corporate-style research report.

Editorial rules (enforced by hand in the cell text below):
  - No emoji and no decorative asterisks anywhere in the prose.
  - Markdown cells carry the narrative; code cells are short and read committed reports/* artifacts so
    the notebook re-executes in seconds without the raw dataset or trained model.

Artifacts are produced by:  run_all.sh, scripts/build_eda_artifacts.py, scripts/build_analysis_artifacts.py
Regenerate this notebook:   python scripts/build_results_notebook.py
                            jupyter nbconvert --to notebook --execute --inplace notebooks/RESULTS.ipynb
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ================================================================= title + abstract
md("""
# ShouldIPost: pre-publication prediction of TikTok video success

A calibrated, leakage-controlled study of whether a short video's success can be estimated before it is
published, and what such an estimate is worth in practice.

## Abstract

We study a cold-start decision: given a finished but unpublished TikTok video, recommend Post,
Do not post, or Unsure, with a calibrated probability and an explanation. The work proceeds in the
order a competent analyst would follow. We first define what "success" should mean and show that a
fame-neutral, within-creator target is the only defensible choice. We then examine the two datasets
named in the brief, demonstrate with evidence that both are unfit for the task, and select a
dataset of record (lingbow, 209,543 videos from 1,872 creators with daily engagement). We perform
exploratory analysis, engineer a modular set of pre-publication features under strict temporal and
leakage controls, and evaluate a wide range of models on two protocols (temporal and
leave-one-creator-out) with bootstrap confidence intervals.

The central empirical result is that pre-publication reach is information-bound, not model-bound: a
simple calibrated gradient-boosting model reaches ROC-AUC of approximately 0.57 (leave-one-creator-out),
and sixteen alternative methods spanning deep networks, boosting, other-domain techniques and feature
engineering all converge on the same wall. A model that observes the first day of engagement is
materially stronger (ROC-AUC about 0.75, leak-free). We therefore frame the product as triage with
honest abstention plus a day-1 amplify-or-cut gate, and we quantify which features carry what little
pre-publication signal exists, how the decision thresholds are derived from asymmetric costs, and what
external data would be required to raise the ceiling.

Every figure and number below is read from a committed artifact under reports/.
""")

# post-audit correction callout
md("""
> Post-audit correction. An independent audit identified a temporal lookahead leak: author-history,
> recency and creator-fit features aggregated prior videos whose horizon-day labels were not yet
> observable when the next video was posted (the median posting gap is about 0.45 days). After a
> closed-window fix, these levers add only about +0.01 to ROC-AUC, recency-only drops from 0.59 to
> about 0.53, and the honest day-1 model is 0.75 rather than the previously reported 0.95. All numbers
> below are the corrected, leak-free figures. The correction strengthens, rather than weakens, the
> information-ceiling conclusion.
""")

md("""
## Contents

0. Problem framing and decision
1. Definition of success (the prediction target)
2. Related work: predicting video virality and diffusion
3. The datasets named in the brief, and why both were rejected
4. Selecting the dataset of record
5. Exploratory data analysis
6. Feature engineering
7. Feature correlation and redundancy
8. Methodology: evaluation protocol and leakage control
9. Modelling and results
10. How each feature influences success
11. Calibration and the decision policy
12. The deployable two-model system
13. From research to product: the end-to-end video extractor and its validation
14. Conclusion, limitations and future work
""")

code("""
import json, warnings
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display as disp
warnings.filterwarnings("ignore")
R = Path("..") / "reports"                       # every artifact read below lives here
csv = lambda n: pd.read_csv(R / n)
js  = lambda n: json.load(open(R / n))
def show(*names):
    for n in names:
        p = R / "plots" / f"{n}.png"
        if p.exists(): disp(Image(filename=str(p)))
import sys; sys.path.insert(0, str(Path("..") / "scripts"))   # shared chart style
import _chartstyle as CS; CS.apply()
GREEN, BLUE, PURPLE, AMBER, RED, GREY = CS.GREEN, CS.BLUE, CS.PURPLE, CS.AMBER, CS.RED, CS.GREY
FIG_W, H_S, H_M = CS.FIG_W, CS.H_S, CS.H_M
""")

# ================================================================= 0. framing
md("""
## 0. Problem framing and decision

The user is a creator, or the team managing a creator, holding a finished video and deciding whether to
publish it. The system returns one of three recommendations:

- Post: the video is likely to do well for this creator.
- Do not post: the video is likely to underperform this creator's own track record.
- Unsure: the pre-publication signal is too weak to commit; publish organically and let the first day of
  engagement decide whether to amplify.

The costs are asymmetric. Recommending Post for a video that flops wastes a publishing slot and erodes
audience goodwill; failing to flag a future hit is a smaller, recoverable miss because the day-1 gate can
still catch it. The system is therefore tuned for precision at Post and is permitted to abstain rather
than force a low-confidence call. This is why the deliverable is a triage-and-abstain policy with a
day-1 follow-up, not a single binary oracle.

Two models serve this flow. Model A is strictly pre-publication: it sees only content and the creator's
state as of the posting time. Model B additionally sees day-1 engagement and drives the
amplify-keep-cut decision once the video is live.

What the model can and cannot know at prediction time. Before posting, the model can see the content
(caption, transcript, summary, topic, emotions, music, hashtags, duration), the posting time, and the
creator's audience size and own track record. It cannot see the signals that the literature shows
dominate outcomes: how long viewers actually watch, the strength of the first one to three seconds as
experienced by a viewer, how the recommender chooses to seed the video, and luck in trend timing. This
gap is the reason pre-publication accuracy is bounded (Section 9) and the reason the product abstains and
defers to the day-1 signal rather than pretending to a certainty it cannot have.
""")

# ================================================================= 1. target
md("""
## 1. Definition of success (the prediction target)

Before any modelling we must decide what "success" means, because a careless definition silently
measures fame instead of quality. We define success as breakout within-creator at horizon H = 14 days:

    breakout = views_at_14_days / followers_at_posting_time

centred by the creator's own median breakout and binarised at the global median. Centring by the
creator's median removes that creator's baseline fame from the label. This is a normalisation of the
target, not a feature, so the model is asked to learn "is this good content for this creator", not "is
this a famous creator".

We compared five candidate labels on the real data. The decisive axis is the fame-leak AUC: how well a
model that sees only the creator's follower count can reproduce the label. A value near 0.5 means the
label is about content; a value near 1.0 means the label is a proxy for fame.
""")
code("""
lc = csv("label_comparison.csv").rename(columns={"мітка": "label"})   # ship an English header
print("Candidate labels. fame_leak_AUC near 0.5 is good (content, not fame); "
      "content_AUC is pre-publication learnability; early_AUC adds day-1 signal.")
display(lc)
""")
md("""
Reading the table:

- Absolute views and raw breakout have high fame-leak (about 0.72 to 0.79): they largely encode who is
  big, so they are rejected.
- Breakout within-creator drives fame-leak down to about 0.51 while remaining business-meaningful
  (reach beyond one's own audience), balanced (base rate 0.50), and orthogonal to engagement rate
  (Spearman about -0.07). It is also strongly predictable from day-1 signal (about 0.75, leak-free), which is what
  makes the day-1 gate work.
- Follower-growth is an equally honest alternative if direct audience acquisition matters more than
  reach; we retain it as a secondary diagnostic.

The within-creator breakout is therefore the primary target throughout. A second virtue, shown in
Section 8, is that its difficulty is horizon-stable across H in {7, 10, 14, 21}, so the fixed horizon
already controls the video-age confounder.

### 1.1 Eligibility, normalisation details and auxiliary targets

A video enters the labelled population only if it is eligible: its views at the horizon reach at least a
floor of 50 (config.MIN_PLAY) and the creator's follower count at posting time is known. This yields
207,509 eligible videos. The success value is centred by the creator's own median breakout and binarised
at the global median of those centred values, computed on the eligible frame. The brief specified a
train-frozen per-creator threshold; we instead define the label once on the full eligible population,
because a per-creator train-frozen threshold is undefined for creators with few or no training videos and
the median is a stable statistic on a balanced target (base rate 0.50). The fame-leak test of about 0.51
confirms this normalisation does not encode creator identity, which is the property that matters; the
feature-level leakage that does matter, a feature seeing the future, is controlled separately in
Section 8.

Two auxiliary targets are retained as diagnostics and for the multi-task experiment: within-creator
engagement rate (resonance among existing fans) and within-creator follower growth (direct audience
acquisition). Both are honest, fame-neutral alternatives. Engagement rate is more learnable
pre-publication, but breakout is the business-aligned reach proxy and remains the primary target.
""")

# ================================================================= 2. related work
md("""
## 2. Related work: predicting video virality and diffusion

This section grounds the project's design choices in the literature. A multi-agent review verified 15 of
16 load-bearing claims against primary sources; the full review with citations is in
reports/literature_review.md. We organise the findings by theme.

### 2.1 The task: pre-publication and cold-start popularity

The Social Media Popularity (SMP) Challenge (Wu et al., ACM MM 2019) formalises exactly this
before-uploading task. Winning systems combine author history, content and multimodal signals in
gradient-boosted ensembles and report Spearman rank correlations rising from roughly 0.59 to 0.77 across
years. Importantly, the field treats the author's own past outcomes as the strongest legitimate
pre-publication signal, not as leakage.

### 2.2 Content-only predictability has a low ceiling

Khosla et al. (What Makes an Image Popular?, WWW 2014) find pure content features reach Spearman of only
0.31 to 0.40, rising to as high as 0.81 once social and author cues are added. Arapakis et al. (On the
Feasibility of Predicting News Popularity at Cold Start, SocInfo 2014) report a best content-only
classifier accuracy of 0.797 against a 0.703 majority baseline, and conclude the task is "essentially
infeasible" from content alone. These ceilings match our cross-creator transfer results, where only
semantics rises clearly above chance (LOCO AUC about 0.557) while timing, duration and emotion barely
transfer (about 0.507 to 0.509).

### 2.3 Watch-time is the dominant driver, and it is unobservable before posting

Wu, Rizoiu and Xie (Beyond Views) decompose the variance: channel reputation explains about R squared
0.42, all pre-upload features together about 0.449, while watch percentage alone explains about 0.77.
Watch-time dwarfs every pre-publication signal, yet it can only be measured after viewers see the
content. The SnapUGC study (Delving Deep into Engagement Prediction of Short Videos, about 90k Snapchat
videos) shows watch percentage is predictable from pre-post content at Spearman about 0.70, which is
precisely the transfer opportunity we identify for future work in Section 14.

### 2.4 Diffusion and self-exciting dynamics make the outcome front-loaded

Self-exciting point-process models such as SEISMIC (Hawkes) predict final cascade size within about 15
percent error after one hour, and SMTPD reports that adding day-1 early popularity lifts day-30
prediction from Spearman 0.85 to 0.96. Our own trajectory analysis agrees: a day-1 model already
attains about 91 percent of the day-14 model's separability above chance. This is the quantitative
reason Model B is easy and Model A is hard.

### 2.5 Evaluation pitfalls

Studies of leakage in recommender evaluation show that only a temporal split (train strictly before
test in time) avoids leakage; random, leave-one-out and by-ratio splits all leak. This validates our
temporal plus leave-one-creator-out discipline. On thresholds, the literature is clear that a default
0.5 cutoff is wrong under class imbalance or asymmetric cost, and that the cost-optimal threshold is
close to the cost ratio, which motivates the calibrated decision bands in Section 11.

In short, the literature predicts a low content-only ceiling, a dominant unobservable watch-time signal,
and strong early-signal determinacy. Our results reproduce all three.
""")

# ================================================================= 3. brief datasets rejected
md("""
## 3. The datasets named in the brief, and why both were rejected

The brief supplied two datasets and explicitly noted that the data is imperfect and should not be
expected to be made perfect. We examined both directly and concluded that neither can support a
pre-publication, cross-creator success model. Recognising this and sourcing a suitable replacement is
itself part of the task.

### 3.1 Dataset 1 — datahiveai/Tiktok-Videos

This is a 2,060-row sample (the full version is paid). Its size is not the fatal problem; its
composition is. All videos come from a handful of mega-creators (in the loaded sample: williesalim,
zachking, mrbeast, addisonre), each with tens of millions of followers and millions of views per video.
On so few, uniformly famous creators it is impossible to learn what makes a creative succeed: there is
no diversity of creators, no small or mid-size accounts, no genuine failures, and no follower variation.
A model would simply learn "this is zachking or mrbeast", which says nothing about a new clip from an
unknown creator. The schema is also thin: no follower field, no audio, no video file, hashtags embedded
in the caption text, engagement counters stored as comma-formatted strings, and duration largely zero.
A model deployed on this set scores ROC-AUC 0.52 with a confidence interval of about [0.45, 0.59], that
is, indistinguishable from chance. The figures below are computed directly from the file.
""")
code("""
eda = js("lingbow_eda.json"); old = eda["old"]
print(f"Dataset 1 — {old['name']}")
print(f"  rows: {old['rows']:,}   creators: {old['creators']}   "
      f"largest creator = {old['top_creator_share']:.0%} of all rows")
print(f"  repost_count all zero: {old['repost_all_zero']}   "
      f"create_time spans {old['create_year_min']}-{old['create_year_max']} "
      f"({old['garbage_epoch_rows']} epoch-zero rows)")
print(f"  day-by-day engagement available: {old['has_daily_engagement']}")
print(f"  deployed pre-publication ROC-AUC = {old['deployed_loco_auc']}  (about random)")
""")
md("""
### 3.2 Dataset 2 — TikTok Video Dataset (commonly yakhyojon/tiktok)

The second dataset has roughly 19,000 rows but was built for a different problem: it is a claim-versus-
opinion classification set that includes an author-ban field. Critically, it contains no information
about creators, music or content categories, and therefore offers no usable pre-publication content
features for popularity. It cannot be repurposed for our task without exactly the signals it lacks.

### 3.3 Implication

Both supplied datasets fail for complementary reasons: Dataset 1 has the right shape of fields but no
population to generalise over, and Dataset 2 has volume but the wrong fields entirely. The task requires
a dataset with many creators, follower information over time, day-by-day engagement, and rich
pre-publication content. None of those are present in the brief's data, so we sourced a replacement.
""")

# ================================================================= 4. dataset of record
md("""
## 4. Selecting the dataset of record

We require a dataset that satisfies four criteria, each of which the brief's data lacked:

1. Many creators, so the cross-creator generalisation question is answerable and leave-one-creator-out
   evaluation has enough folds.
2. Day-by-day engagement, so we can build day-1 features and a horizon-H label without leakage.
3. Creator follower state over time, so success can be normalised by the creator's own audience.
4. Rich pre-publication content beyond a caption.

lingbow/tiktok-video-engagement-200k satisfies all four at scale: 209,543 videos from 1,872 creators
posted between 2024-06-24 and 2024-11-09, with three linked tables (video content with AI-derived
fields, daily engagement for days 0 to 30, and daily creator follower counts). The two datasets named
in the brief and the chosen dataset are compared below.
""")
code("""
cmp = csv("dataset_comparison.csv")
display(cmp)
print("The chosen dataset provides about 100 times the videos, 468 times the creators, a real time "
      "axis, and the content fields needed to ask the pre-publication question honestly.")
""")
md("""
Licence and scope. lingbow is released under CC BY-NC 4.0, which permits research and prototype use only
and forbids commercial deployment. The label is a reach proxy: no dataset in this domain contains
clicks, conversions or revenue. Both constraints are carried through to the model card and the product
copy.
""")

# ================================================================= 5. EDA
md("""
## 5. Exploratory data analysis

The dataset is large, content-rich, and dominated by creator fame. The four views below show when videos
were posted, how cumulative reach accumulates with age, the distribution of views, and the spread of
videos across creators.
""")
code("""
new = js("lingbow_eda.json")["new"]
print(f"{new['name']}")
print(f"  {new['rows']:,} videos   {new['creators']:,} creators   "
      f"{new['date_min']} to {new['date_max']}   {new['eligible']:,} eligible for labelling")
print("  content coverage: " + ", ".join(f"{k} {v:.0%}" for k, v in new['content_coverage'].items()))
print(f"  followers per creator: median {new['followers_median']:,.0f}, "
      f"99th percentile {new['followers_p99']:,.0f}")
print(f"  creator identity explains {new['fame_eta2_logviews']:.0%} of the variance in log(views)")
print(f"  reach is front-loaded: by day 1 the typical (median) video already has "
      f"{new['day1_over_day14_views_median']:.0%} of its day-14 views "
      f"(the mean is {new['day1_over_day14_views_mean']:.0%}, inflated by viral outliers)")
show("eda_timeline", "eda_engagement_traj")
show("eda_views_dist", "eda_creator_videos")
""")
md("""
The most consequential statistic is the last one. On lingbow, creator identity alone explains about 74
percent of the variance in log-views. An absolute-view label would therefore grade fame rather than
content, which is the empirical justification for the within-creator target defined in Section 1. For
contrast, the brief's Dataset 1 showed only 11 percent for the same statistic, but that is an artefact
of its four uniformly-famous creators having almost no between-creator spread; the 1,872-creator
population here reveals the true fame dominance. The front-loaded engagement trajectory, where the
typical video has most of its two-week reach within a day, is the reason a day-1 model is so much
stronger than a pre-publication one.
""")

# ================================================================= 6. feature engineering
md("""
## 6. Feature engineering

Features are produced by a modular extractor in which every block can be toggled independently for
ablation. Each block declares which raw fields it consumes and which features it emits, and every block
is classified as pre-publication (admissible in Model A) or day-1 (admissible only in Model B). The
table summarises the registry.

| Block | Modality | Consumes | Emits (examples) | Scope |
|---|---|---|---|---|
| caption | text | desc | char_len, word_count, n_hashtags, has_question, has_cta, n_emoji, allcaps_ratio | A |
| semantic | text | desc | TF-IDF then TruncatedSVD components sv0..sv59 (fit on train only) | A |
| text_emb:bge / minilm | text | precomputed embeddings | sentence-embedding dimensions | A |
| emotion | affect | anger, joy, surprise, sadness, disgust, fear, speaking_rate | the six emotions, arousal, speaking_rate | A |
| duration | video meta | duration | duration_s and length bins | A |
| timing | temporal | create_dt | cyclic hour, day-of-week, month, is_weekend | A |
| meta | flags | is_english, created_by_ai, is_ads, ratio | indicator features, aspect ratio | A |
| topic | category | topic | target-encoded topic_te (fit on train only) | A |
| hashtags | tags | hashtags | per-tag target encoding, n_hashtags | A |
| music | sound | music_id | target-encoded music_te, music popularity | A |
| creator_fit | similarity | author_id, create_dt, embeddings | similarity to the creator's own past winners; closed-window | A |
| author_history / author_recency | history | author_id, create_dt, views_at_H, followers | past breakout, momentum, posting cadence; closed-window | A |
| trend_fit | trend | precomputed | sound and hashtag momentum at posting time | A |
| retrieval | kNN | embeddings | mean success and similarity of k nearest past videos | A |
| judge | LLM | precomputed | hook, clarity, trend-fit, arousal, saturation, call-to-action | A |
| day1 / day1_basic | early signal | day-1 counters | log plays, log likes, engagement rate | B |
| mm: / retention_head | multimodal | precomputed | video, audio, hook embeddings; transferred watch-time head | A or B |

Three design points matter for correctness:

1. Fold-safe fitting. Every learned transform (TF-IDF and SVD, target encodings, kNN and PCA indices) is
   fit on training rows only and then applied to all rows, with a global-mean fallback for unseen
   categories. This prevents the test labels from influencing the encoding.
2. Closed-window temporal gating. The creator_fit, author_history and author_recency blocks aggregate a
   creator's prior videos, but only those whose horizon-day outcome was already observable at the time
   the current video was posted (prior.create_dt + H is at or before now). This is the fix for the
   lookahead leak described in the audit callout; posting cadence itself is left ungated because it is
   observable immediately.
3. Modality separation. Day-1 counters live only in Model B, and the label horizon window is disjoint
   from the day-1 window, so Model B cannot see its own label.

Scope of what is evaluated. The text, affect, timing, meta, topic, hashtags, music, creator-fit,
author-history, trend-fit, retrieval and judge blocks are evaluated on the full population (Sections 9.1
to 9.7). The multimodal mm: blocks are evaluated only on the N = 262 subset that could be downloaded
(Section 9.8). The retention_head block, which would transfer a watch-time signal from an external
dataset, is wired but not yet evaluated; it is documented as future work in Section 14.
""")

# ================================================================= 7. correlation
md("""
## 7. Feature correlation and redundancy

Before attributing predictive value to features it is useful to understand their correlation structure,
because correlated features share, rather than add, information. The heatmap below shows Spearman
correlations among the interpretable pre-publication features.
""")
code("""
fa = js("feature_analysis_summary.json")
print(f"Examined {fa['n_features_examined']} interpretable pre-publication features on "
      f"{fa['n_eligible']:,} eligible videos.")
print(f"Largest absolute off-diagonal correlation: {fa['max_abs_offdiag_corr']}")
show("feature_correlation")
""")
md("""
The structure is benign. The only strong correlations are inside the affect block: arousal is by
construction the sum of anger, surprise and fear, so it correlates with its components (about 0.74 to
0.78), and joy is negatively correlated with fear (about -0.82). The remaining features are close to
independent, so the engineered set is not redundant and the modelling in Section 9 is not dominated by
collinearity. The correlation matrix itself is saved to reports/feature_correlation.csv.
""")

# ================================================================= 8. methodology
md("""
## 8. Methodology: evaluation protocol and leakage control

Every experiment is evaluated under two protocols, and both are always reported:

- Temporal split: train, validation and test are ordered in time, which mirrors deployment, where one
  predicts the future from the past.
- Leave-one-creator-out (LOCO): no creator appears in both training and test, which measures genuine
  generalisation to unseen creators rather than memorisation of known ones.

Metrics are ROC-AUC, PR-AUC, the Brier score with reliability curves, precision at Post, and abstention
coverage, each reported with bootstrap 95 percent confidence intervals. Leakage is controlled by
automated assertions: Model A is rejected if any feature name matches a post-publication engagement
counter or a label component, and Model B is rejected if it sees any signal beyond day 1. The label uses
day-H counters strictly after all features. The horizon is validated for stability across H in
{7, 10, 14, 21}.
""")
code("""
lr = csv("label_robustness.csv")
print("Horizon stability of the within-creator breakout target:")
display(lr[lr.section == "horizon"][["setting", "loco_auc", "agree_vs_H14"]])
print("Label definition: content learnability (LOCO AUC) versus fame-leak AUC:")
display(lr[lr.section == "label_def"][["setting", "loco_auc", "fame_leak_auc"]])
""")

# ================================================================= 9. modelling and results
md("""
## 9. Modelling and results

### 9.1 Simple baselines

Every later number is anchored against three trivial baselines required by the brief: a majority-class
predictor, a creator-prior (predict the creator's own historical base rate), and a caption-only logistic
regression. On the fame-neutral within-creator target these sit at or below chance, which is the floor
the content and day-1 models must clear.
""")
code("""
bl = js("baselines.json")
rows = []
for key, v in bl.items():
    tgt, name = key.split("/")
    rec = v.get("temporal", v) if "temporal" in v else v        # caption_logit is nested by split
    rows.append({"target": tgt, "baseline": name, "roc_auc": rec.get("roc_auc"),
                 "pr_auc": rec.get("pr_auc"), "precision@Post": rec.get("precision_at_post"),
                 "brier": rec.get("brier")})
display(pd.DataFrame(rows))
print("Majority is 0.50 by construction; the creator-prior scores below 0.5 (regression to the mean: "
      "a creator's recent base rate anti-predicts beating their own median); caption-only logistic is "
      "about 0.51 to 0.52. These are the reference points for every model below.")
""")
md("""
### 9.2 The pre-publication ceiling does not move with the model

Across model families the leave-one-creator-out AUC clusters around 0.55 to 0.57. Hyperparameter tuning
with Optuna adds about +0.006 and stacking about +0.002, both inside the bootstrap confidence intervals.
The ceiling is set by the available information, not the model.
""")
code("""
fam = csv("model_family.csv")
display(fam[fam.target == "y_breakout_wc"][["model", "loco_auc", "temporal_auc", "brier"]])
o = js("optuna_best.json"); e = csv("ensemble.csv")
print(f"Optuna-XGB {o['best_loco_auc']} versus default {o['default_loco_auc']}; "
      f"stacking {e[e.target=='y_breakout_wc']['stack_auc'].iloc[0]}")
d = fam[fam.target == "y_breakout_wc"].set_index("model")["loco_auc"].sort_values()
ax = d.plot.barh(color=GREEN, figsize=(FIG_W, H_S))
ax.axvline(0.5, ls="--", c=GREY); ax.set_xlim(0.5, 0.6); ax.set_xlabel("LOCO ROC-AUC")
ax.set_title("Model A across families clusters near 0.55 to 0.57"); plt.tight_layout(); plt.show()
""")
md("""
### 9.3 What content signal helps, and which modalities transfer across creators

Text embeddings add the most among content modalities (about +0.04 on reach and +0.05 on engagement
rate), but the encoder choice barely matters: BGE, MiniLM and a TF-IDF/SVD baseline are within noise of
one another. Encoder quality is not the bottleneck; information is. The second table isolates which kind
of knowledge transfers to unseen creators (leave-one-creator-out): only semantics (the GPT summary and
transcript) rises clearly above chance, while timing, duration and emotion barely transfer. This is the
empirical version of the literature's content-only ceiling.
""")
code("""
display(csv('text_encoders.csv')[['target', 'encoder', 'loco_auc', 'lift_over_TAB']])
cc = csv("cross_creator_theories.csv").sort_values("oof_auc_LOCO", ascending=False)
print("Cross-creator transfer per modality (LOCO out-of-fold AUC of each theory in isolation):")
display(cc)
""")
md("""
### 9.4 An LLM judge helps resonance, not reach

An LLM scoring hook strength, clarity, arousal and call-to-action lifts within-creator engagement rate by
about +0.03 to +0.06 but adds essentially nothing to breakout. Judged content quality predicts whether
existing fans engage, not whether the algorithm pushes the video to new audiences.
""")
code("display(csv('judge_lift.csv')[['target', 'config', 'loco_auc', 'lift_vs_TAB']])")
md("""
### 9.5 Author history as a sequence: a cautionary tale

Recency and momentum over a creator's own past videos appeared to be the strongest pre-publication
lever, reaching 0.617 in the leaky pre-audit run, until the audit
identified the lookahead leak. Under the closed-window fix, author-history and recency add only about
+0.01, and recency alone is about 0.53, that is, near random. The lesson is that on fast-cadence data it
is not enough to use only earlier posts; the earlier posts' labels must also have been observable. The
corrected table shows the lift gone.
""")
code("""
ah = csv("author_history_model.csv")
display(ah)
ax = ah.set_index("config")[["auc_beats_own_median", "auc_absolute_views"]].plot.bar(
    figsize=(FIG_W, H_S), color=[GREEN, RED])
ax.set_ylabel("LOCO ROC-AUC"); ax.legend(["beats own median (fame-neutral)", "absolute views (fame)"])
ax.set_title("Author history: the fame trap"); plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.show()
""")
md("""
### 9.6 The full-data ablation ladder

This is the required per-modality ablation on the full population: starting from tabular features, each
modality is added in turn and its marginal lift is measured on both protocols with confidence intervals.
It shows directly that text is the only substantial lever and that priors, trend, retrieval and
creator-fit each add little. (The multimodal extension of this ladder, limited to N = 262 videos, is in
Section 9.8.)
""")
code("""
ab = csv("ablation_A.csv")
for tgt in ab["target"].unique():
    print(f"Target: {tgt}")
    display(ab[ab.target == tgt][["stage", "loco_auc", "loco_ci_lo", "loco_ci_hi", "temporal_auc", "marginal"]])
""")
md("""
### 9.7 A method sweep across families and domains

Beyond the boosting and deep classifiers above, we evaluated methods imported from adjacent domains:
learning-to-rank (information retrieval), multi-task shared-trunk networks, causal inverse-propensity
reweighting, clustering features, a rule-based expert system, deep ensembles and tabular deep models
(FT-Transformer; TabPFN and TabNet did not install on the environment). None beats the simple baseline.
The only directional exception is the small-sample multimodal probe in Section 9.8. A useful by-product,
consistent with the diffusion literature, is that a day-1 model already recovers about 91 percent of the
day-14 model's AUC above chance.
""")
code("""
em = csv("extra_methods.csv")
scored = em.dropna(subset=["auc"]).sort_values("auc")
n_methods = int((~scored["method"].str.contains("reference", case=False, na=False)).sum())
ax = scored.set_index("method")["auc"].plot.barh(figsize=(FIG_W, H_M), color=PURPLE)
ax.axvline(0.5, ls="--", c=GREY); ax.set_xlim(0.48, 0.58); ax.set_xlabel("ROC-AUC (temporal test)")
ax.set_title(f"{n_methods} methods versus the best-A baseline, all at the same wall")
plt.tight_layout(); plt.show()
skipped = em[em["auc"].isna()]["method"].astype(str).tolist()
if skipped: print("Could not fit (reported as honest negatives):", ", ".join(skipped))
""")
code("""
# Other-domain methods reported on their native metrics
print("Learning-to-rank, within-author ranking quality (Spearman and NDCG@3):")
display(csv("ltr.csv"))
print("Multi-task shared-trunk versus single-task (temporal AUC):")
display(csv("multitask.csv"))
print("Causal IPS reweighting by author size (temporal AUC and correlation of prediction with size):")
display(csv("ips.csv"))
print("Clustering-derived features:")
display(csv("clustering.csv"))
print("Tabular deep models:")
display(csv("tabular_dl.csv"))
tj = csv("trajectory_pp.csv")
print("Predictability is front-loaded (day-1 model AUC as a fraction of the day-14 model's):")
display(tj[tj.section == "early_window"][["day", "auc_vs_breakout14", "frac_of_day14_auc"]])
""")
md("""
### 9.8 Multimodal content

On the videos we could fetch, SigLIP frames plus CLAP audio push breakout from about 0.53 to 0.62, which
is suggestive that the missing raw-content signal helps but is not conclusive. Free downloading was
rate-limited, so this runs on N = 262 videos with wide confidence intervals (about plus or minus 0.07).
At this sample size the temporal-split column is unstable and some values fall below chance; the
leave-one-creator-out column with its wide interval is the figure to read. This is the project's main
open question and the reason the documented budget targets data acquisition rather than compute.
""")
code("""
mm = js("multimodal_metrics.json")
print(f"N = {mm['n']} videos, {mm['authors']} creators, base rate {mm['base_rate']}")
a = csv("multimodal_ablation.csv")
ladder = ["TAB", "+TEXT", "+AUDIO", "+VIDEO_HOOK", "+SigLIP", "+creator_fit", "+trend"]
order = {c: i for i, c in enumerate(ladder)}
lad = a[a.config.isin(ladder)].copy(); lad["_o"] = lad.config.map(order)
print("Modality ablation ladder, both targets, both splits, with LOCO confidence intervals:")
display(lad.sort_values(["target", "_o"])[["target", "config", "loco_auc", "loco_ci", "temporal_auc", "marginal"]])
fus = a[a.config.astype(str).str.startswith("fusion=")]
if len(fus):
    print("Fusion strategies (early, late, neural): none beats the best single modality:")
    display(fus[["target", "config", "loco_auc", "loco_ci", "temporal_auc"]])
""")
md("""
### 9.9 Error analysis and calibration by segment

Aggregate AUC hides where a model works. The table below breaks performance down by topic, creator size
and video duration, reporting discrimination (AUC), the Brier score, and the expected calibration error
(ECE) for both models on the temporal test set. Model A is strongest on very short videos and weakest in
the middle-duration band; calibration is good across segments (ECE mostly below 0.05), with the largest
miscalibration on the smallest segments where estimates are noisiest.
""")
code("""
ea = csv("error_analysis.csv")
for k in ("A", "B"):
    print(f"Model {k}: per-segment AUC, Brier and calibration error (ECE), temporal test:")
    display(ea[ea.model == k][["segment", "n", "base_rate", "auc", "brier", "ece"]])
""")

# ================================================================= 10. feature influence
md("""
## 10. How each feature influences success

Two complementary views answer the question of which features carry signal. The first is a univariate
view: for each interpretable pre-publication feature we measure its single-feature ROC-AUC against the
within-creator breakout label and attach the sign of its Spearman correlation. A value of 0.50 means no
signal; above means the feature raises the odds of success, below means it lowers them.
""")
code("""
infl = csv("feature_influence.csv")
display(infl)
show("feature_influence")
""")
md("""
The picture is consistent with the information-ceiling thesis even at the level of single features.
Caption length and word count carry the most univariate signal, yet the strongest single feature reaches
only about 0.55 AUC. Question marks in the caption, arousal and anger help slightly; longer videos, joy
and larger creator size are slightly negative for within-creator breakout (the last is expected, since
large accounts find it harder to beat their own high median). Most features sit within a hair of chance.

The second view is model-based attribution. SHAP values from the deployed Model A summarise how each
feature shifts the prediction across the population (the beeswarm) and for a single example (the
waterfall). They corroborate the univariate ranking: text-derived features dominate the small amount of
explainable variance, and no feature produces large, consistent shifts.
""")
code('show("shap_beeswarm", "shap_waterfall")')
md("""
### 10.1 Feature-group importance for each model, and the full input inventory

The two deployed models take different inputs, so their importance differs. Aggregating each model's
mean absolute SHAP by feature group makes this legible: for Model A the text embedding of
caption + transcript + summary dominates, with the small tabular blocks (caption stats, duration,
timing, meta) adding little; for Model B the three day-1 engagement features take over as the leading
group, which is why B is so much stronger. The exact input inventory of every model is listed below.
""")
code("""
show("shap_group_A", "shap_group_B")
inv = js("feature_inventory.json")
rows = []
for grp in ["text embedding (BGE)", "caption text stats", "duration", "posting time", "meta flags",
            "day-1 engagement"]:
    rows.append({"feature group": grp,
                 "Model A (797)": inv["A"]["by_group"].get(grp, 0),
                 "Model B (800)": inv["B"]["by_group"].get(grp, 0)})
print("Input feature inventory per model (count of features per group):")
display(pd.DataFrame(rows))
print("The 29 named tabular features (the 768 BGE dims are the text embedding):")
for g, names in inv["tabular_names"].items():
    print(f"  {g}: {', '.join(names)}")
""")
md("""
### 10.2 Which prediction-time features earn a place in the calculator

A deployable calculator can only use inputs a user can supply before posting. We measured, by forward
addition and by drop-column importance on the temporal test set, which of those inputs actually move the
within-creator breakout AUC. The result is decisive and is the honest reason the calculator is
deliberately compact rather than feature-stuffed. Among pre-publication inputs only the topic carries
real signal (about +0.013 AUC added, +0.0045 lost when removed); caption surface statistics add a little
(+0.022 over the 0.5 floor); and the caption text itself (a 4,000-token TF-IDF), the posting time,
duration and flags add essentially nothing, some of them slightly negative. Adding more pre-publication
inputs would be fake precision. The one decisive lever is the first day of engagement: day-1 signals
lift the AUC from about 0.53 to 0.71. This is the information-bound thesis made operational at the level
of the product's inputs, and it is why the calculator leans on Model B for an accurate, business-usable
decision while presenting Model A only as a weak pre-screen.
""")
code("""
imp = csv("web_feature_impact.csv")
print("Forward addition of each pre-publication input group (temporal test AUC):")
display(imp[imp.analysis == "forward_addition"][["group", "test_auc", "marginal_or_drop"]])
print("Drop-column importance (AUC lost when a group is removed from the full pre-publication set):")
display(imp[imp.analysis == "drop_column"][["group", "test_auc", "marginal_or_drop"]])
print("Adding day-1 engagement (Model B):")
display(imp[imp.analysis == "model_B"][["group", "test_auc", "marginal_or_drop"]])
""")

# ================================================================= 11. calibration + decision policy
md("""
## 11. Calibration and the decision policy

A probability is only useful if it is calibrated and if the thresholds that turn it into a decision are
justified. Three mechanisms are combined.

1. Calibration. The base model's scores are mapped to honest probabilities with isotonic regression fit
   on a held-out calibration split. The reliability curves in Section 12 show the result: predicted
   probabilities match observed frequencies.

2. Conformal abstention. A split-conformal procedure with class-conditional (Mondrian) thresholds
   computes, for each class, the (1 - alpha) quantile of the nonconformity score 1 - P(true class) on
   the calibration set. A video is labelled Post if only the success class is in its prediction set, Do
   not post if only the failure class is, and Unsure if both are. This gives a per-class coverage
   guarantee rather than an arbitrary cutoff.

3. Cost-based thresholds. The Post and Do-not-post cutoffs are not fixed at 0.5. The default bands are
   chosen to hit a precision target at Post; alternatively, a deployment can pass a cost ratio r =
   cost(false Post) / cost(missed hit), and the Bayes-optimal threshold follows the Elkan rule
   t_high = r / (1 + r). A symmetric cost (r = 1) recovers 0.5; a higher cost of a wrong Post raises the
   bar. The first figure shows precision and coverage as the threshold moves, with the chosen cutoffs
   marked; the second shows the cost-ratio to threshold mapping.
""")
code('show("decision_bands", "decision_threshold_cost")')
md("""
The asymmetry between the two models is the point. Model A's pre-publication signal is weak, so its bands
are wide and most videos fall into Unsure; this is the system declining to bluff. Model B's day-1 signal
is strong, so its bands are narrow and it commits on most videos. The exact band values and the resulting
coverage are reported next.
""")

# ================================================================= 12. deployable system
md("""
## 12. The deployable two-model system

The deployed artifact combines Model A (pre-publication triage) and Model B (day-1 decision), each
calibrated, with conformal abstention and cost-based bands. The table reports the operating point.
""")
code("""
dm = js("deployable_metrics.json")
for k in ("A", "B"):
    m = dm[k]["metrics"]
    print(f"Model {k}: ROC-AUC {m['roc_auc']} {m.get('roc_auc_ci')}  PR-AUC {m.get('pr_auc')}  "
          f"Brier {m['brier']}  macro-F1 {m.get('macro_f1')}")
    pc = pd.DataFrame({
        "class": ["breakout (positive)", "not-breakout (negative)"],
        "precision": [m.get("precision_at_post"), m.get("precision_neg")],
        "recall": [m.get("recall_at_post"), m.get("recall_neg")],
        "f1": [m.get("f1_pos"), m.get("f1_neg")],
    })
    print(f"  per-class metrics at the operating threshold (macro-F1 = mean of the two F1s):")
    display(pc)
    print(f"  decision coverage per band: {dm[k].get('band_report')}")
show("reliability_A", "reliability_B")
""")
md("""
The operating point makes the asymmetry concrete. Model A sends most videos to Unsure (high abstention
coverage) and only commits a Post when precision is acceptable; this is the system declining to bluff.
Model B, on leak-free day-1 signals only, commits on most videos with higher precision at Post. The
reliability curves confirm both are well calibrated, so the probabilities can be trusted as inputs to
the cost-based policy.

The per-class table reads as expected from the information ceiling. Model A's positive-class (breakout)
F1 is low, about 0.42 — it rarely flags a winner with confidence, and its negative-class F1 (0.62) is
higher only because most videos are correctly left un-posted; this is why the headline metric is ROC-AUC
plus abstention rather than a single F1. Model B is balanced, with both per-class F1 scores around 0.67.
We report F1 per class for completeness, but note that F1 depends on a fixed threshold, whereas the
product is a calibrated three-way decision with abstention, so AUC, PR-AUC, Brier, precision@Post and
band coverage are the metrics that actually match the business decision.

Two reporting caveats. The conformal abstention is calibrated to a target per-class coverage of 1 minus
alpha; the empirical abstention coverage realised on the test set is the per-band coverage printed above.
The operating-point metrics here are on the temporal split; the unseen-creator estimate of the same
models is the leave-one-creator-out figure in Section 9.2 (about 0.57 for Model A), which is the more
conservative number to quote for a brand-new creator.
""")

# ================================================================= 13. video extractor
md("""
## 13. From research to product: the end-to-end video extractor and its validation

The study so far assumed the dataset's curated fields. A real user, however, uploads a video and
expects the system to extract the features itself. We built a best-effort extraction pipeline and
validated it on real videos with known outcomes. Every stage degrades to a safe default, so the
pipeline never fails on a missing dependency, a silent video, or malformed model output.

| Stage | Input | Tool | Output |
|---|---|---|---|
| 1. Metadata | file or URL | ffprobe / yt-dlp | duration, aspect ratio, caption |
| 2. Audio to text | video | ffmpeg 16 kHz + faster-whisper (base, 74M, CPU) | transcript |
| 3. Text to fields | transcript + caption | interchangeable LLM (DeepSeek; or none) | summary, topic (9), emotions (6), hook, CTA |
| 4. Text to embedding | caption + transcript + summary | BGE (109M, 768-dim) | the deployed Model A's only content lever (Section 9) |
| 5. Score | features (+ day-1 if given) | Model A / Model B | Post / Unsure / Do not post |

Validation design. The multimodal track left 262 real videos on disk, each present in the dataset
with its label, curated text and true day-1 counters. This lets us ask three honest questions: does
the deployed system reproduce its population accuracy on raw real videos; how much accuracy is lost
when the extractor's text replaces the curated text; and what would lift the day-1 model further.
Code: scripts/validate_extractor.py; numbers read from reports/extractor_validation.json.
""")
code("""
ev = js("extractor_validation.json"); s = ev["slice"]; f = ev["fidelity"]
c = ev["confound"]; a = ev["author_relative_day1"]; cov = f["coverage"]
print(f"Validation on {ev['n_slice']} real videos ({ev['n_hits']} hits, {ev['n_english']} English).")

acc = pd.DataFrame([
    ["Model A — caption only", s["aucA_caption_only"]],
    ["Model A — + Whisper transcript (extractor floor)", f["aucA_extracted_text"]],
    ["Model A — curated dataset text", s["aucA_dataset_text"]],
    ["Model A — population reference", s["population_reference"]["A"]],
    ["Model B — real day-1 counters", s["aucB_real_day1"]],
    ["Model B — population reference", s["population_reference"]["B"]],
], columns=["configuration", "ROC-AUC on the 262-video slice"])
display(acc)

fid = pd.DataFrame([
    ["both transcripts present", cov["both_present"]],
    ["Whisper recovers a MISSING dataset transcript", cov["whisper_recovers_missing_dataset_transcript"]],
    ["dataset has text but Whisper silent", cov["dataset_only_whisper_silent"]],
    ["both empty", cov["both_empty"]],
    ["token-Jaccard, median (both present)", f["token_jaccard_median_both"]],
    ["BGE cosine, median (both present)", f["bge_cosine_median_both"]],
    ["per-video |delta p| median / p90", f"{f['abs_dp_median']} / {f['abs_dp_p90']}"],
    ["decision agreement at 0.5", f["decision_agreement_at_0p5"]],
], columns=["extractor fidelity (Whisper vs dataset transcript)", "value"])
display(fid)

labels = ["A: caption only", "A: +Whisper (extractor)", "A: curated text", "A: population ref",
          "B: real day-1", "B: population ref"]
vals = [s["aucA_caption_only"], f["aucA_extracted_text"], s["aucA_dataset_text"],
        s["population_reference"]["A"], s["aucB_real_day1"], s["population_reference"]["B"]]
cols = [GREEN, GREEN, BLUE, GREY, GREEN, GREY]
ax = plt.subplots(figsize=(FIG_W, H_S))[1]
ax.barh(labels[::-1], vals[::-1], color=cols[::-1])
ax.axvline(0.5, ls="--", c=GREY); ax.set_xlim(0.5, 0.8); ax.set_xlabel("ROC-AUC on the 262-video slice")
ax.set_title("Real videos: extracted text nearly matches curated; day-1 reproduces the population")
plt.tight_layout(); plt.show()
print(f"Author-relative day-1 (upgrade path): AUC {a['auc_slice']} slice / {a['auc_test']} test — {a['note']}")
print(f"(raw day-1 plays alone {c['auc_day1_alone_slice']}/{c['auc_day1_alone_test']}; "
      f"Spearman with followers {c['spearman_day1_followers_slice']} — absolute reach mostly encodes creator size)")
""")
md("""
Reading the results. First, the deployed system reproduces its population behaviour on raw real
videos: Model B scores essentially its population AUC on the slice, and Model A sits in the same
band as the multimodal probe of Section 9.8, so nothing about real files breaks the models. Second,
the extractor is an adequate substitute for curated fields, and in one respect exceeds them: where
the dataset also has a transcript the Whisper text matches it closely (token overlap 0.86, embedding
cosine 0.92), and for more than half of the videos the dataset transcript is missing entirely while
Whisper recovers real speech, so the extractor's coverage is wider than the source data's. Replacing
curated text with extracted text costs about one point of AUC and leaves roughly three quarters of
decisions unchanged; the residual loss concentrates in the English subset, where the missing LLM
summary matters most, which is the honest floor of a transcript-only extractor (the pluggable LLM
summary is the natural next increment).
Third, the study surfaced a concrete product upgrade: day-1 plays expressed relative to the
creator's own historical median are a far stronger day-1 signal than absolute counts, because
absolute reach mostly encodes creator size. The number required is one the creator knows about
their own account, so a single optional input field would raise the day-1 gate's accuracy
substantially; we document this as the next step rather than shipping it, because it requires a
retrained Model B and a fallback policy for creators without history.
""")

# ================================================================= 14. conclusion
md("""
## 14. Conclusion, limitations and future work

Conclusion. Pre-publication virality is information-bound. Our content and creator-history Model A
reaches about 0.57 leave-one-creator-out, exactly where the literature's content-only regime predicts,
and sixteen alternative methods confirm the ceiling. The higher numbers reported in the field come from
author history, multimodal content and early signals, which are precisely the levers we measured. The
product's defensible value is triage with honest abstention plus the day-1 amplify-or-cut decision at
about 0.75, stated plainly rather than dressed up. A small N = 262 multimodal probe suggests raw content
could lift Model A toward 0.62, but it is not conclusive.

Limitations. Success is a reach proxy; the data contains no clicks, conversions or revenue. The dataset
is CC BY-NC 4.0, so this is a research prototype, not a deployable commercial product. The multimodal
result rests on 262 videos.

Methods deliberately not run. For honesty about coverage of the brief's experiment matrix, the following
were not executed and are deferred: contrastive or self-supervised pretraining of the content encoder;
transfer pretraining on MicroLens or KuaiRand and adaptation back to lingbow; and the heavier video and
audio encoder zoo (VideoMAE, V-JEPA, InternVideo, AST, PANNs). All of these require raw video and audio
at a scale beyond the 262 clips we could download under the rate limit, so running them would not have
been measured on a meaningful sample. They are listed here rather than left silently absent.

Future work. The single way past the ceiling is more information, not more modelling. The most promising
source is SnapUGC, which ships raw video and audio together with real watch-time labels (NAWP and ECR);
pretraining a hook-and-retention head on it and transferring to lingbow directly attacks the two signals
we lack, and would activate the retention_head block left wired but unevaluated in Section 6. MicroLens
and KuaiRand are secondary transfer sources. These are catalogued, with licences and caveats, in
reports/candidate_datasets.md.

Reproducibility. One entrypoint rebuilds everything: run_all.sh, with --with-multimodal for the heavy
path. The data-journey and analysis artifacts come from scripts/build_eda_artifacts.py and
scripts/build_analysis_artifacts.py; this report is regenerated by scripts/build_results_notebook.py.
Every figure above reads a committed artifact under reports/.
""")

nb["cells"] = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}}
(ROOT / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, ROOT / "notebooks" / "RESULTS.ipynb")
print("wrote notebooks/RESULTS.ipynb")
