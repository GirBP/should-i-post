#!/usr/bin/env python3
"""Validation study for the end-to-end video extractor on real on-disk videos.

Uses the videos under data/videos/ (fetched by the multimodal track) for which the dataset
provides ground truth: the within-creator breakout label, curated text fields
(transcript, gpt_summary) and the real day-1 counters. Produces
reports/extractor_validation.json with four measurements:

  1. slice        deployed Model A / Model B AUC on these videos using dataset-curated
                  features, against the population reference (A 0.568 / B 0.750);
  2. confound     absolute day-1 plays vs creator size (why raw day-1 alone is weak);
  3. author_rel   day-1 plays relative to the creator's own TRAIN-median: the documented,
                  leakage-free upgrade path for Model B (the creator knows this number);
  4. fidelity     Whisper transcripts vs the dataset's reference transcripts (token Jaccard,
                  BGE cosine) and the prediction cost of extractor-built text vs curated text
                  (AUC delta, |delta-p|, decision agreement).

ASR cache: data/processed/whisper_262.parquet (video_id, whisper_text, asr_seconds).
If absent, transcribes with the production path (sip.extract._transcribe; slow, ~20 min).

Run:  PYTHONPATH=src python scripts/validate_extractor.py
"""
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sip import inference as INF, features as F  # noqa: E402

ASR_CACHE = ROOT / "data" / "processed" / "whisper_262.parquet"


def load_slice():
    ids = sorted(p.stem for p in (ROOT / "data" / "videos").glob("*.mp4"))
    can = pd.read_parquet(ROOT / "data" / "processed" / "canonical.parquet")
    can["video_id"] = can["video_id"].astype(str)
    S = can[can.video_id.isin(ids)].copy().reset_index(drop=True)
    T = can[(can.split_temporal == "test") & (can.elig == 1)].copy()
    tr_med = can[can.split_temporal == "train"].groupby("author_id")["log_play_d1"].median()
    return S, T, tr_med


def asr_transcripts(S):
    if ASR_CACHE.exists():
        return pd.read_parquet(ASR_CACHE)
    from sip import extract as EX
    rows = []
    for vid in S.video_id:
        rows.append({"video_id": vid,
                     "whisper_text": EX._transcribe(str(ROOT / "data" / "videos" / f"{vid}.mp4"))})
    df = pd.DataFrame(rows)
    df.to_parquet(ASR_CACHE, index=False)
    return df


def tabular_row(r):
    """The non-embedding part of the deployed Model A feature vector (mirrors sip.inference)."""
    caption = r.desc if isinstance(r.desc, str) else ""
    dt = datetime.fromisoformat(str(r.create_dt))
    cap = {"char_len": len(caption), "word_len": len(caption.split()),
           "n_hashtags": len(re.findall(r"#\w+", caption)),
           "n_mentions": len(re.findall(r"@[\w.]+", caption)),
           "has_question": int("?" in caption), "has_exclam": int("!" in caption),
           "n_emoji": len(F._EMOJI.findall(caption)),
           "has_url": int(bool(re.search(r"https?://|www\.", caption))),
           "has_cta": int(bool(F._CTA.search(caption))),
           "digit_ratio": sum(c.isdigit() for c in caption) / max(len(caption), 1),
           "caption_is_empty": int(len(caption) == 0),
           "allcaps_ratio": (sum(w.isupper() and len(w) > 1 for w in caption.split())
                             / max(len(caption.split()), 1))}
    dur = float(r.duration) if r.duration == r.duration else np.nan
    d = {"duration_s": dur}
    for nm, lo, hi in [("dur_vshort", 0, 7), ("dur_short", 7, 15), ("dur_mid", 15, 30),
                       ("dur_long", 30, 60), ("dur_vlong", 60, 1e9)]:
        d[nm] = float(lo <= dur < hi) if dur == dur else np.nan
    t = {"hour_sin": np.sin(2 * np.pi * dt.hour / 24), "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
         "dow_sin": np.sin(2 * np.pi * dt.weekday() / 7), "dow_cos": np.cos(2 * np.pi * dt.weekday() / 7),
         "is_weekend": int(dt.weekday() >= 5),
         "month_sin": np.sin(2 * np.pi * dt.month / 12), "month_cos": np.cos(2 * np.pi * dt.month / 12)}
    try:
        asp = float(r.ratio)                      # lingbow `ratio` may hold strings like "540p"
    except (TypeError, ValueError):
        asp = np.nan
    asp = asp if asp == asp else np.nan
    m = {"is_english_i": INF._detect_lang_en(caption), "created_by_ai_i": 0.0,
         "is_ads_i": 0.0, "aspect": asp}
    return {**cap, **t, **d, **m}


def main():
    S, T, tr_med = load_slice()
    art = INF._load(); _ = INF._bge("warm")
    names = art["A"]["names"]; enc = art.get("encoder_name", "bge")
    y = S.y_breakout_wc.values
    out = {"n_slice": int(len(S)), "n_hits": int(S.y_breakout_wc.sum()),
           "n_english": int(S.is_english.astype(bool).sum())}

    # ---- 2) confound + 3) author-relative
    out["confound"] = {
        "auc_day1_alone_slice": round(float(roc_auc_score(y, S.log_play_d1)), 4),
        "auc_day1_alone_test": round(float(roc_auc_score(T.y_breakout_wc, T.log_play_d1)), 4),
        "spearman_day1_followers_slice": round(float(spearmanr(S.log_play_d1, np.log1p(S.followers_at_post)).statistic), 4),
        "spearman_day1_followers_test": round(float(spearmanr(T.log_play_d1, np.log1p(T.followers_at_post)).statistic), 4)}
    rel_s = S.log_play_d1 - S.author_id.map(tr_med)
    rel_t = T.log_play_d1 - T.author_id.map(tr_med)
    ms, mt = rel_s.notna(), rel_t.notna()
    out["author_relative_day1"] = {
        "auc_slice": round(float(roc_auc_score(y[ms], rel_s[ms])), 4),
        "auc_test": round(float(roc_auc_score(T.y_breakout_wc[mt], rel_t[mt])), 4),
        "note": "day-1 plays minus the creator's own TRAIN-median; needs one number the creator knows"}

    # ---- shared: embeddings + vectors
    W = asr_transcripts(S)
    S2 = S.merge(W, on="video_id", how="inner").reset_index(drop=True)
    y2 = S2.y_breakout_wc.values
    caps = [r.desc if isinstance(r.desc, str) else "" for _, r in S2.iterrows()]
    ds_tr = S2.transcript.astype(str).replace({"nan": "", "None": ""}).fillna("")
    wh_tr = S2.whisper_text.astype(str).fillna("")
    text_x = [(c + " " + w).strip() for c, w in zip(caps, wh_tr)]
    text_d = [(c + " " + f"{g if isinstance(g, str) else ''} {t}".strip()).strip()
              for c, g, t in zip(caps, S2.gpt_summary, ds_tr)]
    text_c = [c.strip() for c in caps]
    B = INF._BGE
    E = {k: B.encode(v, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
         for k, v in {"x": text_x, "d": text_d, "c": text_c,
                      "wt": list(wh_tr), "dt": list(ds_tr)}.items()}
    tabs = [tabular_row(r) for _, r in S2.iterrows()]

    def X(emb):
        return np.array([[tv[n] if n in tv else
                          (emb[i][int("".join(ch for ch in n if ch.isdigit()))] if n.startswith(enc) else np.nan)
                          for n in names] for i, tv in enumerate(tabs)], dtype=np.float32)

    p_x = art["A"]["cal"].predict_proba_pos(X(E["x"]))
    p_d = art["A"]["cal"].predict_proba_pos(X(E["d"]))
    p_c = art["A"]["cal"].predict_proba_pos(X(E["c"]))

    # ---- 1) slice scoring incl. Model B with real day-1
    eng = pd.read_parquet(ROOT / "data" / "raw" / "lingbow" / "engagement_daily.parquet")
    eng["video_id"] = eng["video_id"].astype(str)
    d1 = eng[(eng.days_since_post == 1) & eng.video_id.isin(S2.video_id)].set_index("video_id")
    Xd_full = X(E["d"])
    pB = np.full(len(S2), np.nan)
    for i, r in S2.iterrows():
        if r.video_id in d1.index:
            e = d1.loc[r.video_id]
            xb = INF._augment_dayone(Xd_full[i:i + 1], art["B"],
                                     {k: float(e[k]) for k in ("play_count", "like_count",
                                      "comment_count", "share_count", "collect_count")})
            pB[i] = float(art["B"]["cal"].predict_proba_pos(xb)[0])
    mB = ~np.isnan(pB)
    out["slice"] = {
        "aucA_dataset_text": round(float(roc_auc_score(y2, p_d)), 4),
        "aucA_caption_only": round(float(roc_auc_score(y2, p_c)), 4),
        "aucB_real_day1": round(float(roc_auc_score(y2[mB], pB[mB])), 4),
        "population_reference": {"A": 0.5683, "B": 0.7502}}

    # ---- 4) ASR + prediction fidelity
    toks = lambda s: set(re.findall(r"[a-z0-9']+", str(s).lower()))  # noqa: E731  (punctuation-free)
    wh_has = wh_tr.str.strip().ne("").values
    ds_has = ds_tr.str.strip().ne("").values
    both = wh_has & ds_has
    jac = np.array([len(toks(a) & toks(b)) / max(len(toks(a) | toks(b)), 1)
                    for a, b in zip(wh_tr, ds_tr)])
    cos = (E["wt"] * E["dt"]).sum(1)
    en = S2.is_english.astype(bool).values
    dp = np.abs(p_x - p_d)
    out["fidelity"] = {
        "coverage": {"both_present": int(both.sum()),
                     "whisper_recovers_missing_dataset_transcript": int((wh_has & ~ds_has).sum()),
                     "dataset_only_whisper_silent": int((~wh_has & ds_has).sum()),
                     "both_empty": int((~wh_has & ~ds_has).sum())},
        "token_jaccard_median_both": round(float(np.median(jac[both])), 3),
        "bge_cosine_median_both": round(float(np.median(cos[both])), 3),
        "bge_cosine_median_both_en": round(float(np.median(cos[both & en])), 3),
        "bge_cosine_median_both_nonen": round(float(np.median(cos[both & ~en])), 3),
        "aucA_extracted_text": round(float(roc_auc_score(y2, p_x)), 4),
        "aucA_dataset_text": round(float(roc_auc_score(y2, p_d)), 4),
        "aucA_extracted_text_en": round(float(roc_auc_score(y2[en], p_x[en])), 4),
        "aucA_dataset_text_en": round(float(roc_auc_score(y2[en], p_d[en])), 4),
        "abs_dp_median": round(float(np.median(dp)), 4),
        "abs_dp_p90": round(float(np.percentile(dp, 90)), 4),
        "decision_agreement_at_0p5": round(float(np.mean((p_x >= 0.5) == (p_d >= 0.5))), 4),
        "note": "extracted = caption + Whisper transcript (no LLM summary; the extractor floor)"}

    dest = ROOT / "reports" / "extractor_validation.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
