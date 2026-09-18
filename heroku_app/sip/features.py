"""Modular, independently-toggleable feature blocks (the unit of ablation).

Each block is a callable ``fn(df, train_idx) -> (X: np.ndarray, names: list[str])``.
Frozen blocks (deterministic transforms, pretrained embeddings) ignore train_idx.
Fold-safe blocks (target encoders, TF-IDF/SVD) fit ONLY on rows in train_idx and
transform all rows — so the experiment runner can call them per fold with that
fold's training indices and never leak.

Blocks are grouped:
  MODEL A (pre-publication):
    caption emotion duration timing meta topic music hashtags semantic
    text_emb:<name> creator_fit trend_fit
  MODEL B (adds post-publication day-1):
    day1
  MULTIMODAL (loaded from data/multimodal/features.parquet):
    mm_text mm_video mm_hook mm_audio mm_clap mm_lowlevel
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from . import config as C, data as D

_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]")
_CTA = re.compile(r"\b(follow|like|share|comment|subscribe|link in bio|check out|tag|save|duet)\b", re.I)


# ------------------------------------------------------------------ derived cols
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic per-video columns (idempotent)."""
    if "_derived" in df.attrs:
        return df
    desc = df["desc"].fillna("").astype(str)
    df["char_len"] = desc.str.len()
    df["word_len"] = desc.str.split().map(len)
    df["n_hashtags"] = desc.str.count(r"#\w+")
    df["n_mentions"] = desc.str.count(r"@[\w.]+")
    df["has_question"] = desc.str.contains(r"\?").astype(int)
    df["has_exclam"] = desc.str.contains("!", regex=False).astype(int)
    df["n_emoji"] = desc.map(lambda s: len(_EMOJI.findall(s)))
    df["has_url"] = desc.str.contains(r"https?://|www\.").astype(int)
    df["has_cta"] = desc.map(lambda s: int(bool(_CTA.search(s))))
    df["digit_ratio"] = desc.map(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    df["caption_is_empty"] = (df["char_len"] == 0).astype(int)
    df["allcaps_ratio"] = desc.map(
        lambda s: (sum(w.isupper() and len(w) > 1 for w in s.split()) / max(len(s.split()), 1)))
    for em in ["anger", "joy", "surprise", "sadness", "disgust", "fear"]:
        df[em] = pd.to_numeric(df[em], errors="coerce")
    df["arousal"] = df[["anger", "surprise", "fear"]].sum(axis=1)
    df["speaking_rate"] = pd.to_numeric(df["speaking_rate"], errors="coerce")
    df["duration_s"] = pd.to_numeric(df["duration"], errors="coerce")
    for name, lo, hi in [("dur_vshort", 0, 7), ("dur_short", 7, 15), ("dur_mid", 15, 30),
                         ("dur_long", 30, 60), ("dur_vlong", 60, 1e9)]:
        df[name] = ((df["duration_s"] >= lo) & (df["duration_s"] < hi)).astype(int)
    ct = df["create_dt"]
    df["hour_sin"] = np.sin(2 * np.pi * ct.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ct.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ct.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ct.dt.dayofweek / 7)
    df["is_weekend"] = (ct.dt.dayofweek >= 5).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * ct.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ct.dt.month / 12)
    df["is_english_i"] = (df["is_english"] == True).astype(int)  # noqa: E712
    df["created_by_ai_i"] = (df["created_by_ai"] == True).astype(int)  # noqa: E712
    df["is_ads_i"] = (df["is_ads"] == True).astype(int)  # noqa: E712
    df["aspect"] = pd.to_numeric(df["ratio"], errors="coerce")
    df.attrs["_derived"] = True
    return df


def _cols(df, names):
    return df[names].astype(float).values, list(names)


# ------------------------------------------------------------- frozen A blocks
CAPTION = ["char_len", "word_len", "n_hashtags", "n_mentions", "has_question",
           "has_exclam", "n_emoji", "has_url", "has_cta", "digit_ratio",
           "caption_is_empty", "allcaps_ratio"]
EMOTION = ["anger", "joy", "surprise", "sadness", "disgust", "fear", "arousal", "speaking_rate"]
DURATION = ["duration_s", "dur_vshort", "dur_short", "dur_mid", "dur_long", "dur_vlong"]
TIMING = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month_sin", "month_cos"]
META = ["is_english_i", "created_by_ai_i", "is_ads_i", "aspect"]


# --------------------------------------------------------- fold-safe A blocks
def _target_encode(df, idcol, train_idx, y):
    g = pd.Series(y).iloc[train_idx].groupby(df[idcol].iloc[train_idx].values).mean()
    gm = float(np.mean(y[train_idx]))
    return df[idcol].map(g).fillna(gm).values.reshape(-1, 1)


def block_topic(df, train_idx, y):
    return _target_encode(df, "topic", train_idx, y), ["topic_te"]


def block_music(df, train_idx, y):
    te = _target_encode(df, "music_id", train_idx, y)
    pop = df["music_id"].map(df["music_id"].iloc[train_idx].value_counts()).fillna(0).values.reshape(-1, 1)
    return np.c_[te, np.log1p(pop)], ["music_te", "music_pop_log"]


def _hashtag_names(x):
    try:
        return [h["hashtag_name"] for h in x] if x is not None and len(x) > 0 else []
    except Exception:
        return []


def block_hashtags(df, train_idx, y):
    if "_htags" not in df.columns:
        df["_htags"] = df["hashtags"].map(_hashtag_names)
    sub_tags = df["_htags"].iloc[train_idx].values
    sub_y = y[train_idx]
    num, den = {}, {}
    for tags, yy in zip(sub_tags, sub_y):
        for h in tags:
            num[h] = num.get(h, 0) + yy; den[h] = den.get(h, 0) + 1
    gm = float(sub_y.mean()); rate = {h: num[h] / den[h] for h in num}

    def val(tags):
        r = [rate[h] for h in tags if h in rate]
        return np.mean(r) if r else gm
    te = df["_htags"].map(val).values.reshape(-1, 1)
    return np.c_[te, df["n_hashtags"].values.reshape(-1, 1)], ["htag_te", "n_hashtags"]


def block_semantic(df, train_idx, y, n_components=60):
    """TF-IDF -> TruncatedSVD fit on TRAIN text only (fold-safe)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    txt = D.text_field(df)
    min_df = 5 if len(train_idx) > 1000 else 1
    tf = TfidfVectorizer(max_features=8000, min_df=min_df, stop_words="english")
    Xtr = tf.fit_transform(txt.iloc[train_idx])
    k = max(1, min(n_components, Xtr.shape[1] - 1, len(train_idx) - 1))
    svd = TruncatedSVD(n_components=k, random_state=C.SEED).fit(Xtr)
    Z = svd.transform(tf.transform(txt))
    return Z, [f"sv{i}" for i in range(k)]


# ----------------------------------------------- pretrained embedding caches
@lru_cache(maxsize=8)
def _emb_cache(name: str):
    """Load a precomputed embedding parquet keyed by video_id. Returns (dict, dim)."""
    path = C.PROC / f"emb_{name}.parquet"
    if not path.exists():
        return None, 0
    e = pd.read_parquet(path); e["video_id"] = e["video_id"].astype(str)
    cols = [c for c in e.columns if c != "video_id"]
    arr = e[cols].values.astype(np.float32)
    return dict(zip(e["video_id"].values, arr)), len(cols)


def block_text_emb(df, train_idx, y, name="bge"):
    cache, dim = _emb_cache(name)
    if cache is None:
        raise FileNotFoundError(f"missing emb_{name}.parquet — run scripts/encode_text.py")
    M = np.zeros((len(df), dim), np.float32)
    vids = df["video_id"].astype(str).values
    for i, v in enumerate(vids):
        if v in cache:
            M[i] = cache[v]
    return M, [f"{name}{i}" for i in range(dim)]


# ----------------------------------------------- creator-fit (closed-window, leak-safe)
def block_creator_fit(df, train_idx, y, space="semantic"):
    """Cosine of candidate to the centroid of the SAME author's past winners (y==1) whose
    H-day outcome was ALREADY OBSERVABLE at this video's post time (prior.create_dt + H <=
    create_dt). The closed window is essential: posting cadence is ~0.45 days, so a naive
    'strictly earlier' rule would use labels that don't exist yet (lookahead leak)."""
    if space == "semantic":
        Z, _ = block_semantic(df, train_idx, y)
    else:
        Z, _ = block_text_emb(df, train_idx, y, name=space)
    E = Z.astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    days = (df["create_dt"].astype("datetime64[ns]").astype("int64") / 86400e9).values
    order = np.argsort(days)
    auth = df["author_id"].values
    fit = np.zeros(len(df), np.float32); nprev = np.zeros(len(df), np.float32)
    from collections import defaultdict, deque
    dim = E.shape[1]
    pend = defaultdict(deque)                                   # winners whose label isn't observable yet
    sums = defaultdict(lambda: np.zeros(dim, np.float32)); cnts = defaultdict(int)
    for i in order:
        a = auth[i]; di = days[i]
        while pend[a] and pend[a][0][0] <= di - C.H:            # release now-observable winners
            sums[a] += pend[a].popleft()[1]; cnts[a] += 1
        if cnts[a] > 0:
            cc = sums[a] / cnts[a]; cc = cc / (np.linalg.norm(cc) + 1e-8)
            fit[i] = float(E[i] @ cc); nprev[i] = cnts[a]
        if y[i] == 1:
            pend[a].append((di, E[i]))
    return np.c_[fit, np.log1p(nprev)], ["creator_fit", "n_prior_wins_log"]


def block_retrieval(df, train_idx, y, name="bge", k=50, pca_dim=64):
    """Retrieval-augmented prediction (multimodal memory bank, à la MMRA/M3TR):
    predict from the mean label of the k nearest TRAIN videos in embedding space.
    Fold-safe: neighbours and PCA are fit on train rows only. Returns the kNN
    success estimate + the mean neighbour similarity (confidence)."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    try:
        Z, _ = block_text_emb(df, train_idx, y, name=name)
    except FileNotFoundError:
        Z, _ = block_semantic(df, train_idx, y)
    Z = Z.astype(np.float32)
    p = PCA(n_components=min(pca_dim, Z.shape[1]), random_state=C.SEED).fit(Z[train_idx])
    Zr = p.transform(Z)
    Zr = Zr / (np.linalg.norm(Zr, axis=1, keepdims=True) + 1e-8)
    # cap the memory bank for tractable brute kNN (a sampled bank is still valid)
    ref = train_idx
    if len(train_idx) > 25000:
        rng = np.random.default_rng(C.SEED)
        ref = rng.choice(train_idx, 25000, replace=False)
    nn = NearestNeighbors(n_neighbors=min(k, len(ref)), metric="cosine").fit(Zr[ref])
    ytr = y[ref]
    dist, idx = nn.kneighbors(Zr)                 # idx into ref space
    knn_y = ytr[idx].mean(axis=1).reshape(-1, 1)
    sim = (1.0 - dist).mean(axis=1).reshape(-1, 1)
    return np.c_[knn_y, sim], ["knn_success", "knn_sim"]


def block_trend_fit(df, train_idx, y):
    """Is the music / hashtags rising at post time? Loaded from a precomputed
    trend table (scripts/build_trend_features.py). Falls back to zeros."""
    path = C.PROC / "trend_features.parquet"
    if not path.exists():
        return np.zeros((len(df), 2), np.float32), ["trend_music", "trend_hashtag"]
    t = pd.read_parquet(path); t["video_id"] = t["video_id"].astype(str)
    t = t.set_index("video_id").reindex(df["video_id"].astype(str).values)
    return t[["trend_music", "trend_hashtag"]].fillna(0).values, ["trend_music", "trend_hashtag"]


def block_author_history(df, train_idx, y):
    """Author's STRICTLY-PAST realised popularity (NxtPost / SocRipple cold-start
    backfill). Legitimate pre-publication signal: it uses outcomes of the author's
    EARLIER videos (known before this post), never this video's. Carries a
    popularity-bias ('big author') component -> parried by the within-creator label
    and reported as such. Returns running-mean log breakout, log views, log #prior."""
    views = pd.to_numeric(df.get("views_at_H"), errors="coerce").fillna(0).values
    foll = pd.to_numeric(df.get("followers_at_post"), errors="coerce").fillna(0).values
    raw_breakout = np.log1p(views / np.clip(foll, 1, None))
    raw_pop = np.log1p(views)
    days = (df["create_dt"].astype("datetime64[ns]").astype("int64") / 86400e9).values
    order = np.argsort(days)
    auth = df["author_id"].values
    from collections import defaultdict, deque
    pend = defaultdict(deque)                                   # prior videos whose label isn't observable yet
    sb = defaultdict(float); sp = defaultdict(float); cn = defaultdict(int)
    hb = np.zeros(len(df), np.float32); hp = np.zeros(len(df), np.float32); nn = np.zeros(len(df), np.float32)
    for i in order:
        a = auth[i]; di = days[i]
        while pend[a] and pend[a][0][0] <= di - C.H:            # only labels observable by now (closed window)
            _, b_, p_ = pend[a].popleft(); sb[a] += b_; sp[a] += p_; cn[a] += 1
        if cn[a] > 0:
            hb[i] = sb[a] / cn[a]; hp[i] = sp[a] / cn[a]; nn[i] = cn[a]
        pend[a].append((di, raw_breakout[i], raw_pop[i]))
    return np.c_[hb, hp, np.log1p(nn)], ["author_hist_breakout", "author_hist_pop", "author_nprior_log"]


def block_author_recency(df, train_idx, y):
    """Strictly-past SEQUENTIAL author features: recent form (last-3 mean), exp-weighted
    recency, momentum (recent - lifetime), posting cadence. Best pre-publication lever found
    (lifts within-creator breakout to ~0.62 vs 0.59 for the lifetime aggregate). Legitimate
    (uses only the author's earlier outcomes) but needs author history -> 0 for cold authors."""
    views = pd.to_numeric(df.get("views_at_H"), errors="coerce").fillna(0).values
    foll = pd.to_numeric(df.get("followers_at_post"), errors="coerce").fillna(1).clip(lower=1).values
    bo = np.log1p(views / foll)
    days = (df["create_dt"].astype("datetime64[ns]").astype("int64") / 86400e9).values
    order = np.argsort(days)
    auth = df["author_id"].values
    from collections import defaultdict, deque
    pend = defaultdict(deque)          # (day, breakout) outcomes not yet observable
    hist = defaultdict(list)           # observable breakout values, in order
    lsum = defaultdict(float); ln = defaultdict(int); last_day = {}
    rec3 = np.zeros(len(df)); ewm = np.zeros(len(df)); mom = np.zeros(len(df)); dsl = np.zeros(len(df))
    for i in order:
        a = auth[i]; di = days[i]
        while pend[a] and pend[a][0][0] <= di - C.H:            # release outcomes whose H-day label exists
            v = pend[a].popleft()[1]; hist[a].append(v); lsum[a] += v; ln[a] += 1
        if hist[a]:
            last3 = hist[a][-3:]; rec3[i] = np.mean(last3)
            w = np.array([0.6 ** k for k in range(len(hist[a]))][::-1]); ewm[i] = np.dot(w, hist[a]) / w.sum()
            mom[i] = rec3[i] - lsum[a] / ln[a]
        if a in last_day:
            dsl[i] = di - last_day[a]                            # cadence is observable immediately (ungated)
        pend[a].append((di, bo[i])); last_day[a] = di
    return np.c_[rec3, ewm, mom, np.log1p(dsl)], \
        ["auth_recent3", "auth_ewm", "auth_momentum", "auth_days_since_last_log"]


# ----------------------------------------------------------- Model B day-1
DAY1 = ["log_play_d1", "log_like_d1", "er_d1", "log_play_d1_vs_author"]
# inference-reproducible subset: all three are computable from the user's day-1 counts alone
# (the author-relative term needs the creator's historical median, which is unavailable cold).
DAY1_BASIC = ["log_play_d1", "log_like_d1", "er_d1"]


# ----------------------------------------------------------- multimodal blocks
@lru_cache(maxsize=4)
def _mm_cache():
    path = C.MM / "features.parquet"
    if not path.exists():
        return None
    f = pd.read_parquet(path); f["video_id"] = f["video_id"].astype(str)
    return f.set_index("video_id")


@lru_cache(maxsize=2)
def _judge_cache():
    path = C.PROC / "judge_scores.parquet"
    if not path.exists():
        return None
    j = pd.read_parquet(path); j["video_id"] = j["video_id"].astype(str)
    return j.set_index("video_id")


JUDGE_DIMS = ["hook", "clarity", "trend_fit", "arousal", "saturation", "cta"]


def block_judge(df, train_idx, y, dims=None):
    """LLM-as-judge scores (precomputed). `dims` selects a subset for per-dimension
    ablation; default = all six."""
    j = _judge_cache()
    if j is None:
        raise FileNotFoundError("missing data/processed/judge_scores.parquet — run experiments/run_judge.py")
    cols = [f"judge_{d}" for d in (dims or JUDGE_DIMS)]
    sub = j[[c for c in cols if c in j.columns]].reindex(df["video_id"].astype(str).values)
    return sub.fillna(sub.median()).values.astype(np.float32), list(sub.columns)


@lru_cache(maxsize=2)
def _retention_cache():
    path = C.PROC / "snapugc_retention.parquet"
    if not path.exists():
        return None
    r = pd.read_parquet(path); r["video_id"] = r["video_id"].astype(str)
    return r.set_index("video_id")


def block_retention_head(df, train_idx, y):
    """Transfer feature: predicted watch%/retention from a head pretrained on SnapUGC
    (raw frames+audio -> NAWP/ECR), applied to this video's content embeddings. Injects
    the literature's dominant-but-missing signal (watch-time). Loaded from a precomputed
    table (experiments/run_transfer_snapugc.py). Falls back to zeros if absent."""
    r = _retention_cache()
    cols = ["pred_nawp", "pred_ecr"]
    if r is None:
        return np.zeros((len(df), 2), np.float32), cols
    sub = r.reindex(df["video_id"].astype(str).values)
    return sub[[c for c in cols if c in r.columns]].fillna(sub.median()).values.astype(np.float32), \
        [c for c in cols if c in r.columns]


def block_mm(df, train_idx, y, prefix):
    mm = _mm_cache()
    if mm is None:
        raise FileNotFoundError("missing data/multimodal/features.parquet — run multimodal extractor")
    cols = [c for c in mm.columns if c.startswith(prefix)]
    if not cols:
        return np.zeros((len(df), 1), np.float32), [f"{prefix}_missing"]
    sub = mm[cols].reindex(df["video_id"].astype(str).values)
    return sub.fillna(0.0).values.astype(np.float32), cols


def block_missing(df, train_idx, y):
    """Missing-modality indicators (Little & Rubin missing-indicator method): one binary flag
    per modality so the model can learn whether an ABSENT modality is itself informative,
    instead of conflating 'no audio' with an embedding that happens to sit near zero. The
    embedding stays zero-imputed (standard for tree models); the flag makes absence explicit."""
    desc = df["desc"].fillna("").astype(str).str.strip()
    tr = (df["transcript"] if "transcript" in df.columns else df["desc"]).fillna("").astype(str).str.strip()
    has_caption = (desc != "").astype(np.float32).values
    has_transcript = (tr != "").astype(np.float32).values
    has_audio = np.ones(len(df), np.float32)
    has_video = np.ones(len(df), np.float32)
    mm = _mm_cache()
    if mm is not None:
        ids = df["video_id"].astype(str).values
        clap = [c for c in mm.columns if c.startswith("clap")]
        vsig = [c for c in mm.columns if c.startswith("vsig")]
        if clap:
            has_audio = (mm[clap].reindex(ids).abs().sum(1).fillna(0.0) > 0).astype(np.float32).values
        if vsig:
            has_video = (mm[vsig].reindex(ids).abs().sum(1).fillna(0.0) > 0).astype(np.float32).values
    X = np.column_stack([has_caption, has_transcript, has_audio, has_video])
    return X.astype(np.float32), ["has_caption", "has_transcript", "has_audio", "has_video"]


# ------------------------------------------------------------ block registry
def build_blocks(df, blocks, train_idx, y, target_col=None):
    """Assemble a feature matrix from a list of block specs. Fold-safe blocks use
    `y` over `train_idx` only. Returns (X, names)."""
    add_derived(df)
    y = np.asarray(y)
    mats, names = [], []
    for b in blocks:
        if b == "caption":
            X, n = _cols(df, CAPTION)
        elif b == "emotion":
            X, n = _cols(df, EMOTION)
        elif b == "duration":
            X, n = _cols(df, DURATION)
        elif b == "timing":
            X, n = _cols(df, TIMING)
        elif b == "meta":
            X, n = _cols(df, META)
        elif b == "missing":
            X, n = block_missing(df, train_idx, y)
        elif b == "topic":
            X, n = block_topic(df, train_idx, y)
        elif b == "music":
            X, n = block_music(df, train_idx, y)
        elif b == "hashtags":
            X, n = block_hashtags(df, train_idx, y)
        elif b == "semantic":
            X, n = block_semantic(df, train_idx, y)
        elif b.startswith("text_emb:"):
            X, n = block_text_emb(df, train_idx, y, name=b.split(":", 1)[1])
        elif b == "creator_fit" or b.startswith("creator_fit:"):
            space = b.split(":", 1)[1] if ":" in b else "semantic"
            X, n = block_creator_fit(df, train_idx, y, space=space)
        elif b.startswith("retrieval:") or b == "retrieval":
            name = b.split(":", 1)[1] if ":" in b else "bge"
            X, n = block_retrieval(df, train_idx, y, name=name)
        elif b == "trend_fit":
            X, n = block_trend_fit(df, train_idx, y)
        elif b == "author_history":
            X, n = block_author_history(df, train_idx, y)
        elif b == "author_recency":
            X, n = block_author_recency(df, train_idx, y)
        elif b == "retention_head":
            X, n = block_retention_head(df, train_idx, y)
        elif b == "day1":
            X, n = _cols(df, DAY1)
        elif b == "day1_basic":
            X, n = _cols(df, DAY1_BASIC)
        elif b == "judge" or b.startswith("judge:"):
            dims = b.split(":", 1)[1].split(",") if ":" in b else None
            X, n = block_judge(df, train_idx, y, dims=dims)
        elif b.startswith("mm:"):
            X, n = block_mm(df, train_idx, y, prefix=b.split(":", 1)[1])
        else:
            raise ValueError(f"unknown block {b}")
        mats.append(np.nan_to_num(X.astype(np.float32))); names.extend(n)
    return (np.concatenate(mats, axis=1) if mats else np.zeros((len(df), 0))), names
