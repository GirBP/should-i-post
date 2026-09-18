import pandas as pd, numpy as np, warnings, time, re
warnings.filterwarnings("ignore"); t0=time.time()
log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}",flush=True)
R="data/raw/lingbow"; H,d1,min_videos,q,min_play=14,1,50,0.5,50
cols=["video_id","author_id","create_time","duration","is_english","desc","music_id",
      "topic","anger","joy","surprise","sadness","disgust","fear",
      "word_count","emoji_count","question_count","hashtag_count","speaking_rate",
      "created_by_ai","is_ads","sticker_text","transcript","gpt_summary"]
v=pd.read_parquet(f"{R}/videos.parquet",columns=cols)
e=pd.read_parquet(f"{R}/engagement_daily.parquet",columns=["video_id","days_since_post","play_count","like_count","comment_count","share_count","collect_count"]).sort_values(["video_id","days_since_post"])
c=pd.read_parquet(f"{R}/creator_daily.parquet",columns=["author_id","date","follower_count"])
log("loaded")
ENG=["like_count","comment_count","share_count","collect_count"]
cumH=e[e.days_since_post<=H].groupby("video_id").tail(1).set_index("video_id")
erH=(cumH[ENG].sum(axis=1)/cumH["play_count"].clip(lower=1))
d=v.set_index("video_id").copy(); d["create_dt"]=pd.to_datetime(d["create_time"],errors="coerce")
q70,q85=d["create_dt"].quantile([0.70,0.85]); d["split"]=np.where(d["create_dt"]>=q85,"test",np.where(d["create_dt"]>=q70,"valid","train"))
trm0=d["split"]=="train"
# labels
erH_d=erH.reindex(d.index)
er_tr=erH_d.reindex(d.index[trm0]).dropna(); a_tr=d.loc[er_tr.index,"author_id"]; cnt=a_tr.value_counts(); good=cnt[cnt>=min_videos].index
thr_a=er_tr.groupby(a_tr.values).quantile(q); thr_a=thr_a[thr_a.index.isin(good)].to_dict(); thr_g=float(er_tr.quantile(q))
y_er=(erH_d>d["author_id"].map(lambda a:thr_a.get(a,thr_g))).astype("float")
def foll_asof(when):
    L=pd.DataFrame({"vid":d.index.values,"author_id":d["author_id"].values,"_w":when.values}).dropna(subset=["_w"]).sort_values("_w")
    Rr=c.assign(date=pd.to_datetime(c["date"],errors="coerce"))[["author_id","date","follower_count"]].dropna().sort_values("date")
    m=pd.merge_asof(L,Rr,left_on="_w",right_on="date",by="author_id",direction="backward")
    return pd.Series(m["follower_count"].values,index=m["vid"]).reindex(d.index)
foll=foll_asof(d["create_dt"]); playH=cumH["play_count"].reindex(d.index)
bo=playH/foll.clip(lower=1); bo_wc=bo-bo.groupby(d["author_id"].values).transform("median")
y_bo=(bo_wc>bo_wc[trm0].median()).astype("float")
log("labels ready")
# ---- feature tiers ----
desc=d["desc"].fillna("").astype(str)
A0=pd.DataFrame(index=d.index)
EMOJI=re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")
A0["char_len"]=desc.str.len(); A0["word_len"]=desc.str.split().map(len)
A0["n_hashtags"]=desc.str.count(r"#\w+"); A0["has_question"]=desc.str.contains(r"\?").astype(int)
A0["has_exclam"]=desc.str.contains("!",regex=False).astype(int); A0["n_emoji"]=desc.map(lambda s:len(EMOJI.findall(s)))
A0["duration_s"]=pd.to_numeric(d["duration"],errors="coerce")
ct=d["create_dt"]; A0["hour_sin"]=np.sin(2*np.pi*ct.dt.hour/24); A0["hour_cos"]=np.cos(2*np.pi*ct.dt.hour/24)
A0["dow"]=ct.dt.dayofweek; A0["is_weekend"]=(ct.dt.dayofweek>=5).astype(int)
A0["is_english"]=pd.to_numeric(d["is_english"],errors="coerce").fillna(0).astype(int)
mfreq=d.loc[trm0,"music_id"].value_counts(); A0["music_freq"]=d["music_id"].map(mfreq).fillna(0.0)
# A1: native GenAI content
A1=A0.copy()
for em in ["anger","joy","surprise","sadness","disgust","fear"]: A1[em]=pd.to_numeric(d[em],errors="coerce")
for nm in ["word_count","emoji_count","question_count","hashtag_count","speaking_rate"]: A1[nm]=pd.to_numeric(d[nm],errors="coerce")
A1["created_by_ai"]=pd.to_numeric(d["created_by_ai"],errors="coerce").fillna(0); A1["is_ads"]=pd.to_numeric(d["is_ads"],errors="coerce").fillna(0)
A1["has_transcript"]=d["transcript"].fillna("").astype(str).str.len().gt(0).astype(int)
A1["sticker_len"]=d["sticker_text"].fillna("").astype(str).str.len()
top=pd.get_dummies(d["topic"].astype(str),prefix="t"); A1=pd.concat([A1,top],axis=1)
log("A0/A1 ready")
# A2: + TF-IDF(gpt_summary+transcript) -> SVD80 (fit on train)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
txt=(d["gpt_summary"].fillna("")+" "+d["transcript"].fillna("")).astype(str)
tf=TfidfVectorizer(max_features=10000,min_df=5,stop_words="english")
Xtr_txt=tf.fit_transform(txt[trm0]); svd=TruncatedSVD(n_components=80,random_state=0); svd.fit(Xtr_txt)
emb=pd.DataFrame(svd.transform(tf.transform(txt)),index=d.index,columns=[f"sv{i}" for i in range(80)])
A2=pd.concat([A1,emb],axis=1)
log(f"A2 ready (text svd evr={svd.explained_variance_ratio_.sum():.2f})")
# ---- eval ----
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
trm=trm0&(playH>=min_play); tem=(d["split"]=="test")&(playH>=min_play)
def auc(X,y):
    Xtr=X.loc[trm].astype(float).fillna(X.loc[trm].astype(float).median()); Xte=X.loc[tem].astype(float).fillna(X.loc[trm].astype(float).median())
    ytr,yte=y[trm],y[tem]
    m=HistGradientBoostingClassifier(max_depth=4,learning_rate=0.06,max_iter=250,l2_regularization=1.0)
    m.fit(Xtr,ytr.astype(int)); return roc_auc_score(yte.astype(int),m.predict_proba(Xte)[:,1])
print("\n=== Модель A (до публікації): чи піднімає багатший контент? ===")
print(f"{'label':28} {'A0 поточна':>11} {'A1 +контент':>12} {'A2 +семантика':>14}")
for lname,y in [("breakout(within-creator)",y_bo),("within-creator ER",y_er)]:
    r=[auc(A0,y),auc(A1,y),auc(A2,y)]; print(f"{lname:28} {r[0]:>11.3f} {r[1]:>12.3f} {r[2]:>14.3f}")
    log(f"done {lname}")
log("ALL DONE")
