#!/usr/bin/env python3
"""
Крок 3 (локально): злити мультимодальні ознаки з мітками й оцінити Моделі A/B.

Мітка/спліт/автор беруться з маніфесту (breakout within-creator, y вже пораховано).
Ознаки:
  TAB   — табличний контент із lingbow (тривалість, емоції, speaking_rate, лічильники)
  TEXT  — temb* (MiniLM),  VIDEO — vemb* (CLIP),  HOOK — hemb* (перші 3 c),  AUDIO — aud* (мел)
  DAY1  — сигнали першої доби (лише для Моделі B)

Абляція показує, що додає кожна модальність понад табличний baseline (~0.55).

Запуск:
    python multimodal/03_train_eval.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "lingbow"
MAN = ROOT / "data" / "multimodal" / "manifest.csv"
FEAT = ROOT / "data" / "multimodal" / "features.parquet"
OUT = ROOT / "reports" / "multimodal_metrics.json"
H = 14


def block(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

    man = pd.read_csv(MAN, dtype={"video_id": str})
    F = pd.read_parquet(FEAT); F["video_id"] = F["video_id"].astype(str)
    df = man.merge(F, on="video_id", how="inner")
    print(f"злито відео з ознаками: {len(df)} (з {len(man)} у маніфесті) — покриття {len(df)/len(man):.1%}")

    # --- TAB (табличний контент) + DAY1 із lingbow ---
    v = pd.read_parquet(RAW / "videos.parquet",
        columns=["video_id", "duration", "speaking_rate", "word_count", "emoji_count",
                 "question_count", "hashtag_count", "anger", "joy", "surprise", "sadness", "disgust", "fear"])
    v["video_id"] = v["video_id"].astype(str)
    df = df.merge(v, on="video_id", how="left")
    TAB = ["duration", "speaking_rate", "word_count", "emoji_count", "question_count", "hashtag_count",
           "anger", "joy", "surprise", "sadness", "disgust", "fear"]

    e = pd.read_parquet(RAW / "engagement_daily.parquet",
                        columns=["video_id", "days_since_post", "play_count", "like_count",
                                 "comment_count", "share_count", "collect_count"])
    e["video_id"] = e["video_id"].astype(str)
    d1 = e[e.days_since_post <= 1].sort_values(["video_id", "days_since_post"]).groupby("video_id").tail(1).set_index("video_id")
    df["log_play_d1"] = np.log1p(d1["play_count"].reindex(df.video_id).values)
    df["log_like_d1"] = np.log1p(d1["like_count"].reindex(df.video_id).values)
    er = (d1[["like_count", "comment_count", "share_count", "collect_count"]].sum(1) / d1["play_count"].clip(lower=1))
    df["er_d1"] = er.reindex(df.video_id).values
    DAY1 = ["log_play_d1", "log_like_d1", "er_d1"]

    TEXT, VIDEO, HOOK, AUDIO = block(df, "temb"), block(df, "vemb"), block(df, "hemb"), block(df, "aud")
    y = df["y"].astype(int).values
    groups = df["author_id"].values
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values

    def evaluate(cols, name, model="hgb"):
        X = df[cols].astype(float).fillna(df[cols].astype(float).median())
        # LOCO (GroupKFold) AUC
        oof = np.full(len(df), np.nan)
        gkf = GroupKFold(n_splits=4)
        for tri, tei in gkf.split(X, y, groups):
            m = _mk(model)
            m.fit(X.iloc[tri].values, y[tri]); oof[tei] = m.predict_proba(X.iloc[tei].values)[:, 1]
        loco = roc_auc_score(y, oof)
        # темпоральний тест
        m = _mk(model); m.fit(X[tr].values, y[tr]); p = m.predict_proba(X[te].values)[:, 1]
        res = {"name": name, "n_features": len(cols),
               "loco_auc": round(loco, 3),
               "temporal_auc": round(roc_auc_score(y[te], p), 3),
               "temporal_pr_auc": round(average_precision_score(y[te], p), 3),
               "brier": round(brier_score_loss(y[te], p), 3)}
        print(f"  {name:34} LOCO={res['loco_auc']:.3f}  temporal={res['temporal_auc']:.3f}")
        return res

    def _mk(model):
        if model == "logreg":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0)

    print("\n=== Модель A (до публікації): абляція модальностей ===")
    A = {}
    A["tab_baseline"] = evaluate(TAB, "TAB (baseline)")
    A["text"] = evaluate(TAB + TEXT, "TAB+TEXT")
    A["audio"] = evaluate(TAB + TEXT + AUDIO, "TAB+TEXT+AUDIO")
    A["video"] = evaluate(TAB + TEXT + VIDEO + HOOK, "TAB+TEXT+VIDEO+HOOK")
    A["full"] = evaluate(TAB + TEXT + VIDEO + HOOK + AUDIO, "A_FULL (усі модальності)")
    print("\n=== Модель B (+ перша доба) ===")
    B = evaluate(TAB + TEXT + VIDEO + HOOK + AUDIO + DAY1, "B_FULL (A + день1)")

    out = {"n": len(df), "coverage_of_manifest": round(len(df) / len(man), 3),
           "base_rate_test": round(float(y[te].mean()), 3),
           "model_A_ablation": A, "model_B": B,
           "note": "breakout within-creator; LOCO=GroupKFold за авторами; temporal=train->test за датою"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
    print("\nЗбережено:", OUT)


if __name__ == "__main__":
    main()
