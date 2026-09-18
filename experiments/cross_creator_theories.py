import pandas as pd, numpy as np, warnings, time, re, os
warnings.filterwarnings("ignore"); t0=time.time()
log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}",flush=True)
R="data/raw/lingbow"; H,min_videos,q,min_play=14,50,0.5,50
FEAT="data/processed/_feat_cache.parquet"; RES="reports/cross_creator_theories.csv"

if os.path.exists(FEAT):
    D=pd.read_parquet(FEAT); log(f"features from cache {D.shape}")
else:
    cols=["video_id","author_id","create_time","duration","desc","music_id","topic",
          "anger","joy","surprise","sadness","disgust","fear","speaking_rate","hashtags",
          "transcript","gpt_summary"]
    v=pd.read_parquet(f"{R}/videos.parquet",columns=cols)
    e=pd.read_parquet(f"{R}/engagement_daily.parquet",columns=["video_id","days_since_post","play_count","like_count","comment_count","share_count","collect_count"]).sort_values(["video_id","days_since_post"])
    c=pd.read_parquet(f"{R}/creator_daily.parquet",columns=["author_id","date","follower_count"])
    log("loaded raw")
    ENG=["like_count","comment_count","share_count","collect_count"]
    cumH=e[e.days_since_post<=H].groupby("video_id").tail(1).set_index("video_id")
    d=v.set_index("video_id").copy(); d["create_dt"]=pd.to_datetime(d["create_time"],errors="coerce")
    def foll_asof(when):
        L=pd.DataFrame({"vid":d.index.values,"author_id":d["author_id"].values,"_w":when.values}).dropna(subset=["_w"]).sort_values("_w")
        Rr=c.assign(date=pd.to_datetime(c["date"],errors="coerce"))[["author_id","date","follower_count"]].dropna().sort_values("date")
        m=pd.merge_asof(L,Rr,left_on="_w",right_on="date",by="author_id",direction="backward")
        return pd.Series(m["follower_count"].values,index=m["vid"]).reindex(d.index)
    foll=foll_asof(d["create_dt"]); playH=cumH["play_count"].reindex(d.index)
    bo=playH/foll.clip(lower=1); bo_wc=bo-bo.groupby(d["author_id"].values).transform("median")
    F=pd.DataFrame(index=d.index)
    F["author_id"]=d["author_id"]; F["y"]=(bo_wc>bo_wc.median()).astype(int)
    F["elig"]=((playH>=min_play)&foll.notna()).astype(int)
    # caption
    desc=d["desc"].fillna("").astype(str); EM=re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
    F["char_len"]=desc.str.len(); F["word_len"]=desc.str.split().map(len); F["n_hashtags"]=desc.str.count(r"#\w+")
    F["n_mentions"]=desc.str.count(r"@[\w.]+"); F["has_question"]=desc.str.contains(r"\?").astype(int)
    F["has_exclam"]=desc.str.contains("!",regex=False).astype(int); F["n_emoji"]=desc.map(lambda s:len(EM.findall(s)))
    F["has_url"]=desc.str.contains(r"https?://|www\.").astype(int)
    # emotion (+arousal)
    for em in ["anger","joy","surprise","sadness","disgust","fear"]: F[em]=pd.to_numeric(d[em],errors="coerce")
    F["arousal"]=F[["anger","surprise","fear"]].sum(axis=1)
    F["speaking_rate"]=pd.to_numeric(d["speaking_rate"],errors="coerce")
    # duration + timing
    F["duration_s"]=pd.to_numeric(d["duration"],errors="coerce"); ct=d["create_dt"]
    F["hour_sin"]=np.sin(2*np.pi*ct.dt.hour/24); F["hour_cos"]=np.cos(2*np.pi*ct.dt.hour/24)
    F["dow_sin"]=np.sin(2*np.pi*ct.dt.dayofweek/7); F["dow_cos"]=np.cos(2*np.pi*ct.dt.dayofweek/7); F["is_weekend"]=(ct.dt.dayofweek>=5).astype(int)
    # ids for fold-safe priors
    F["topic"]=d["topic"].astype(str); F["music_id"]=d["music_id"].astype(str)
    def hnames(x):
        try: return [h["hashtag_name"] for h in x] if x is not None and len(x)>0 else []
        except Exception: return []
    F["htags"]=d["hashtags"].map(hnames)
    # semantic SVD (unsupervised, fit on all — прийнятно для розвідки)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    txt=(d["gpt_summary"].fillna("")+" "+d["transcript"].fillna("")).astype(str)
    tf=TfidfVectorizer(max_features=8000,min_df=5,stop_words="english"); X=tf.fit_transform(txt)
    sv=TruncatedSVD(n_components=60,random_state=0).fit_transform(X)
    for i in range(60): F[f"sv{i}"]=sv[:,i]
    log("features built")
    F.to_parquet(FEAT); D=F; log("features cached")

# ---------- evaluation ----------
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

D=D[D["elig"]==1].copy()
y=D["y"].values; groups=D["author_id"].values
gkf=GroupKFold(n_splits=4); folds=list(gkf.split(D,y,groups))

CAP=["char_len","word_len","n_hashtags","n_mentions","has_question","has_exclam","n_emoji","has_url"]
EMO=["anger","joy","surprise","sadness","disgust","fear","arousal","speaking_rate"]
DUR=["duration_s"]; TIM=["hour_sin","hour_cos","dow_sin","dow_cos","is_weekend"]
SEM=[f"sv{i}" for i in range(60)]

def te_col(idcol, tr, all_idx):              # fold-safe target encoding by id
    g=D.iloc[tr].groupby(idcol)["y"].mean(); gm=D.iloc[tr]["y"].mean()
    return D[idcol].map(g).fillna(gm).values

def htag_te(tr):                              # mean train-breakout over video's hashtags
    sub=D.iloc[tr]; m={}; cnt={}
    for tags,yy in zip(sub["htags"].values, sub["y"].values):
        for h in tags: m[h]=m.get(h,0)+yy; cnt[h]=cnt.get(h,0)+1
    gm=sub["y"].mean(); rate={h:m[h]/cnt[h] for h in m}
    def val(tags): 
        r=[rate[h] for h in tags if h in rate]; return np.mean(r) if r else gm
    return D["htags"].map(val).values

def oof_auc(build):                           # build(tr)->X (DataFrame aligned to D), model
    oof=np.full(len(D),np.nan)
    for tr,te in folds:
        X=build(tr); mdl=make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400))
        Xtr=np.nan_to_num(X[tr]); Xte=np.nan_to_num(X[te])
        if len(np.unique(y[tr]))<2: continue
        mdl.fit(Xtr,y[tr]); oof[te]=mdl.predict_proba(Xte)[:,1]
    ok=~np.isnan(oof); return roc_auc_score(y[ok],oof[ok])

def cols_build(cols):
    M=D[cols].astype(float).values
    return lambda tr: M

theories={
 "T1 тема (cross-creator prior)": lambda: oof_auc(lambda tr: np.c_[te_col("topic",tr,None)]),
 "T2 емоції/arousal": lambda: oof_auc(cols_build(EMO)),
 "T3 підпис/гачок": lambda: oof_auc(cols_build(CAP)),
 "T4 тривалість": lambda: oof_auc(cols_build(DUR)),
 "T5 час публікації": lambda: oof_auc(cols_build(TIM)),
 "T6 музика (prior+популярність)": lambda: oof_auc(lambda tr: np.c_[te_col("music_id",tr,None), D["music_id"].map(D.iloc[tr]["music_id"].value_counts()).fillna(0).values]),
 "T7 хештеги (prior)": lambda: oof_auc(lambda tr: np.c_[htag_te(tr), D["n_hashtags"].values]),
 "T8 семантика (SVD60)": lambda: oof_auc(cols_build(SEM)),
}
done=set()
if os.path.exists(RES):
    prev=pd.read_csv(RES); done=set(prev["theory"]); 
else:
    prev=pd.DataFrame(columns=["theory","oof_auc_LOCO"])
log(f"base rate={y.mean():.3f}, eligible={len(D)}, already done={len(done)}")
rows=[]
for name,fn in theories.items():
    if name in done: continue
    a=fn(); rows.append({"theory":name,"oof_auc_LOCO":round(a,3)}); log(f"{name}: AUC={a:.3f}")
    pd.concat([prev,pd.DataFrame(rows)],ignore_index=True).to_csv(RES,index=False)
print("\n=== РЕЗУЛЬТАТИ (LOCO, чужі автори -> наш) ===")
print(pd.read_csv(RES).to_string(index=False))
log("DONE")
