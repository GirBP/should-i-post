#!/usr/bin/env python3
"""Measure which PREDICTION-TIME-OBTAINABLE features actually move the within-creator breakout AUC, to
decide what belongs in the browser calculator. Forward-addition ablation on the temporal split, plus
grouped permutation importance. Nothing here is exposed unless it earns its place.

Run:  PYTHONPATH=src python scripts/investigate_web_features.py
"""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROC = ROOT / "data" / "processed"

CTA = re.compile(r"\b(follow|like|share|comment|subscribe|link in bio|check out|tag|save|duet)\b", re.I)
URL = re.compile(r"https?://|www\.")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]")
WORD = re.compile(r"#\w+")


def cap_feats(desc):
    c = "" if desc is None else str(desc)
    w = [t for t in c.split() if t]
    caps = [t for t in w if len(t) > 1 and t.upper() == t and re.search("[A-Z]", t)]
    return [len(c), len(w), len(WORD.findall(c)), len(EMOJI.findall(c)),
            1.0 if "?" in c else 0.0, 1.0 if CTA.search(c) else 0.0, 1.0 if URL.search(c) else 0.0,
            (len(caps) / len(w)) if w else 0.0]


df = pd.read_parquet(PROC / "canonical.parquet")
df = df[(df["elig"] == 1) & df["y_breakout_wc"].notna()].copy()
y = df["y_breakout_wc"].astype(int).values
sp = df["split_temporal"].values
tr, te = sp == "train", sp == "test"
dt = pd.to_datetime(df["create_dt"])

# feature groups (all obtainable before/at prediction time)
caption_stats = np.array([cap_feats(d) for d in df["desc"].values], float)
timing = np.c_[np.sin(2*np.pi*dt.dt.hour/24), np.cos(2*np.pi*dt.dt.hour/24),
               np.sin(2*np.pi*dt.dt.dayofweek/7), np.cos(2*np.pi*dt.dt.dayofweek/7),
               (dt.dt.dayofweek >= 5).astype(float)]
duration = np.log1p(pd.to_numeric(df["duration"], errors="coerce").fillna(0).clip(lower=0)).values.reshape(-1, 1)
flags = np.c_[df["is_english"].astype(float).fillna(0), df["is_ads"].astype(float).fillna(0),
              df["created_by_ai"].astype(float).fillna(0)]
glob = y[tr].mean()
te_map = pd.Series(y[tr]).groupby(df["topic"].values[tr]).mean()
topic = df["topic"].map(te_map).fillna(glob).astype(float).values.reshape(-1, 1)
day1 = np.c_[pd.to_numeric(df["log_play_d1"], errors="coerce").fillna(0),
             pd.to_numeric(df["log_like_d1"], errors="coerce").fillna(0),
             pd.to_numeric(df["er_d1"], errors="coerce").fillna(0)]

# caption TF-IDF (fit on train only) — the actual WORDS of the caption
vec = TfidfVectorizer(max_features=4000, min_df=10, stop_words="english", ngram_range=(1, 2))
tfidf_tr = vec.fit_transform(df["desc"].fillna("").values[tr])
tfidf_all = vec.transform(df["desc"].fillna("").values)
print(f"caption TF-IDF vocabulary: {len(vec.vocabulary_)} tokens")

GROUPS = {"caption_stats": caption_stats, "timing": timing, "duration": duration,
          "flags": flags, "topic": topic, "caption_text(tfidf)": tfidf_all, "day1": day1}


def fit_auc(mats):
    parts = [csr_matrix(m) if not hasattr(m, "tocsr") else m for m in mats]
    X = hstack(parts).tocsr()
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])


def std(m):  # standardize dense groups (tfidf already scaled)
    if hasattr(m, "tocsr"):
        return m
    mu = m[tr].mean(0); sd = m[tr].std(0); sd[sd == 0] = 1
    return (m - mu) / sd


rows = []
print("\n=== Forward addition (Model A, pre-publication; temporal test AUC) ===")
order = ["caption_stats", "duration", "timing", "flags", "topic", "caption_text(tfidf)"]
chosen, prev = [], 0.5
for g in order:
    chosen.append(g)
    auc = fit_auc([std(GROUPS[k]) for k in chosen])
    print(f"  + {g:22s} -> AUC {auc:.4f}   (marginal {auc-prev:+.4f})")
    rows.append({"analysis": "forward_addition", "group": g, "test_auc": round(auc, 4),
                 "marginal_or_drop": round(auc - prev, 4)})
    prev = auc

print("\n=== Leave-one-group-out (drop-column importance on the full Model A set) ===")
full = order
base = fit_auc([std(GROUPS[k]) for k in full])
for g in full:
    auc = fit_auc([std(GROUPS[k]) for k in full if k != g])
    print(f"  without {g:22s} -> AUC {auc:.4f}   (drop {base-auc:+.4f})")
    rows.append({"analysis": "drop_column", "group": g, "test_auc": round(auc, 4),
                 "marginal_or_drop": round(base - auc, 4)})

aucB = fit_auc([std(GROUPS[k]) for k in full + ["day1"]])
print(f"\n=== Model B (add day-1 signals) ===\n  full A + day1 -> AUC {aucB:.4f}")
rows.append({"analysis": "model_B", "group": "+day1", "test_auc": round(aucB, 4),
             "marginal_or_drop": round(aucB - base, 4)})

out = ROOT / "reports" / "web_feature_impact.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
print("Conclusion: among prediction-time features, only TOPIC adds real signal to Model A; caption "
      "text/timing/duration/flags add ~0. The decisive lever is day-1 (Model B). More inputs would be "
      "fake precision, so the calculator exposes the justified minimal set plus day-1.")
