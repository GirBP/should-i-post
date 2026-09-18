#!/usr/bin/env python3
"""Train compact, browser-deployable logistic surrogates of Model A and Model B and export them to
docs/web_model.json for the GitHub Pages calculator.

Why a surrogate: the full Model A uses text embeddings (BGE) and video-derived fields (emotion,
transcript) that cannot run in a static web page. This trains an honest logistic model on exactly the
features a user can supply before posting (caption text statistics, duration, posting time, topic, flags)
for Model A, plus day-1 plays and likes for Model B. Feature construction here mirrors the JavaScript in
docs/index.html one-to-one so train-time and browser-time features match. Reported AUC is the surrogate's
own held-out score, not the full model's.

Run:  PYTHONPATH=src python scripts/build_web_model.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sip import eval as E  # find_decision_bands

PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

CTA = re.compile(r"\b(follow|like|share|comment|subscribe|link in bio|check out|tag|save|duet)\b", re.I)
URL = re.compile(r"https?://|www\.")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]")
WORD = re.compile(r"#\w+")


def caption_feats(desc):
    """Pure-text features, computed identically to the browser port."""
    c = "" if desc is None else str(desc)
    words = [w for w in c.split() if w]
    caps = [w for w in words if len(w) > 1 and w.upper() == w and re.search("[A-Z]", w)]
    return {
        "char_len": len(c),
        "word_count": len(words),
        "n_hashtags": len(WORD.findall(c)),
        "n_emoji": len(EMOJI.findall(c)),
        "has_question": 1.0 if "?" in c else 0.0,
        "has_cta": 1.0 if CTA.search(c) else 0.0,
        "has_url": 1.0 if URL.search(c) else 0.0,
        "allcaps_ratio": (len(caps) / len(words)) if words else 0.0,
    }


def time_feats(dt):
    h = dt.hour + dt.minute / 60.0
    dow = dt.dayofweek
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": 1.0 if dow >= 5 else 0.0,
    }


df = pd.read_parquet(PROC / "canonical.parquet")
df = df[(df["elig"] == 1) & df["y_breakout_wc"].notna()].copy()
dt = pd.to_datetime(df["create_dt"])
y = df["y_breakout_wc"].astype(int).values

# --- topic target encoding on TRAIN only ---
tr = df["split_temporal"].values == "train"
glob = float(y[tr].mean())
te = pd.Series(y[tr]).groupby(df["topic"].values[tr]).mean()
topic_te_map = {str(k): float(v) for k, v in te.items()}
topic_te = df["topic"].map(topic_te_map).fillna(glob).astype(float).values

# --- assemble the feature frame (caption + time + flags + topic + day1) ---
cap = pd.DataFrame([caption_feats(d) for d in df["desc"].values], index=df.index)
tim = pd.DataFrame([time_feats(t) for t in dt], index=df.index)
base = pd.concat([cap, tim], axis=1)
base["log_duration"] = np.log1p(pd.to_numeric(df["duration"], errors="coerce").fillna(0).clip(lower=0))
base["is_english"] = df["is_english"].astype(float).fillna(0)
base["is_ads"] = df["is_ads"].astype(float).fillna(0)
base["created_by_ai"] = df["created_by_ai"].astype(float).fillna(0)
base["topic_te"] = topic_te

A_FEATS = list(base.columns)
B_EXTRA = ["log_play_d1", "log_like_d1", "er_d1"]      # user-suppliable day-1 signals
for c in B_EXTRA:
    base[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

split = df["split_temporal"].values
LABELS = {  # human-readable factor names for the UI
    "char_len": "caption length", "word_count": "word count", "n_hashtags": "hashtag count",
    "n_emoji": "emoji count", "has_question": "question hook", "has_cta": "call to action",
    "has_url": "off-platform link", "allcaps_ratio": "ALL-CAPS ratio", "hour_sin": "posting hour",
    "hour_cos": "posting hour", "dow_sin": "day of week", "dow_cos": "day of week",
    "is_weekend": "weekend post", "log_duration": "video length", "is_english": "English caption",
    "is_ads": "marked as ad", "created_by_ai": "AI-generated", "topic_te": "topic prior",
    "log_play_d1": "day-1 plays", "log_like_d1": "day-1 likes", "er_d1": "day-1 engagement rate",
}


def train(feature_names, tag):
    X = base[feature_names].values.astype(float)
    Xtr, Xva, Xte = X[split == "train"], X[split == "valid"], X[split == "test"]
    ytr, yva, yte = y[split == "train"], y[split == "valid"], y[split == "test"]
    mean = Xtr.mean(0); std = Xtr.std(0); std[std == 0] = 1.0
    clf = LogisticRegression(max_iter=2000, C=1.0).fit((Xtr - mean) / std, ytr)
    p_te = clf.predict_proba((Xte - mean) / std)[:, 1]
    p_va = clf.predict_proba((Xva - mean) / std)[:, 1]
    bands = E.find_decision_bands(yva, p_va, precision_target=0.60)
    auc = roc_auc_score(yte, p_te)
    print(f"Model {tag}: test ROC-AUC {auc:.4f}  Brier {brier_score_loss(yte, p_te):.4f}  "
          f"n_test={len(yte)}  bands={bands}")
    return {
        "features": feature_names,
        "labels": [LABELS.get(f, f) for f in feature_names],
        "mean": [round(float(v), 6) for v in mean],
        "std": [round(float(v), 6) for v in std],
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
        "bands": bands,
        "metrics": {"roc_auc": round(float(auc), 4),
                    "brier": round(float(brier_score_loss(yte, p_te)), 4),
                    "n_test": int(len(yte))},
    }


out = {
    "version": "web-1.0",
    "note": "Logistic surrogates trained on browser-computable features only; honest held-out AUC below.",
    "topic_te": {**topic_te_map, "__global__": round(glob, 6)},
    "topics": sorted(topic_te_map.keys()),
    "A": train(A_FEATS, "A"),
    "B": train(A_FEATS + B_EXTRA, "B"),
}
DOCS.mkdir(exist_ok=True)
json.dump(out, open(DOCS / "web_model.json", "w"), indent=1)

# Inline the model into docs/index.html between sentinels so the calculator works offline and on Pages.
idx = DOCS / "index.html"
if idx.exists():
    html = idx.read_text()
    compact = json.dumps(out, separators=(",", ":"))
    new = re.sub(r"/\*MODEL_JSON_START\*/.*?/\*MODEL_JSON_END\*/",
                 "/*MODEL_JSON_START*/" + compact.replace("\\", "\\\\") + "/*MODEL_JSON_END*/",
                 html, count=1, flags=re.S)
    idx.write_text(new)
    print(f"inlined model into {idx}")
print(f"wrote {DOCS/'web_model.json'}  (A {len(out['A']['features'])} feats, B {len(out['B']['features'])} feats)")
