import pandas as pd, numpy as np, warnings, json, time
warnings.filterwarnings("ignore")
t0=time.time(); R="data/raw/lingbow"
log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)

H, d1, min_play, min_videos, q = 14, 1, 50, 50, 0.5
v=pd.read_parquet(f"{R}/videos.parquet", columns=["video_id","author_id","create_time","duration","is_english","desc","music_id"])
c=pd.read_parquet(f"{R}/creator_daily.parquet", columns=["author_id","date","follower_count"])
e=pd.read_parquet(f"{R}/engagement_daily.parquet",
   columns=["video_id","days_since_post","play_count","like_count","comment_count","share_count","collect_count"])
log(f"loaded v={v.shape} c={c.shape} e={e.shape}")

e=e.sort_values(["video_id","days_since_post"])
def cum_at(Hh):
    return e[e.days_since_post<=Hh].groupby("video_id").tail(1).set_index("video_id")
cumH=cum_at(H); cum1=cum_at(d1)
ENG=["like_count","comment_count","share_count","collect_count"]
erH=(cumH[ENG].sum(axis=1)/cumH["play_count"].clip(lower=1))
log("cumH/cum1 ready")

d=v.set_index("video_id").copy()
d["create_dt"]=pd.to_datetime(d["create_time"],errors="coerce")
q70,q85=d["create_dt"].quantile([0.70,0.85])
d["split"]=np.where(d["create_dt"]>=q85,"test",np.where(d["create_dt"]>=q70,"valid","train"))

# within-creator ER label (train-frozen thresholds)
erH_d=erH.reindex(d.index)
tr_v=d.index[d["split"]=="train"]
er_tr=erH_d.reindex(tr_v).dropna(); a_tr=d.loc[er_tr.index,"author_id"]
cnt=a_tr.value_counts(); good=cnt[cnt>=min_videos].index
thr_a=er_tr.groupby(a_tr.values).quantile(q); thr_a=thr_a[thr_a.index.isin(good)].to_dict()
thr_g=float(er_tr.quantile(q))
ref=d["author_id"].map(lambda a:thr_a.get(a,thr_g))
y_er=(erH_d>ref).astype("float")
log("ER label ready")

# features A
import re
EMOJI=re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")
desc=d["desc"].fillna("").astype(str)
F=pd.DataFrame(index=d.index)
F["char_len"]=desc.str.len(); F["word_len"]=desc.str.split().map(len)
F["n_hashtags"]=desc.str.count(r"#\w+"); F["n_mentions"]=desc.str.count(r"@[\w.]+")
F["has_question"]=desc.str.contains(r"\?").astype(int); F["has_exclam"]=desc.str.contains("!",regex=False).astype(int)
F["n_emoji"]=desc.map(lambda s:len(EMOJI.findall(s))); F["has_url"]=desc.str.contains(r"https?://|www\.").astype(int)
F["digit_ratio"]=desc.map(lambda s:sum(ch.isdigit() for ch in s)/len(s) if s else 0.0)
F["caption_is_empty"]=(F["char_len"]==0).astype(int)
F["duration_s"]=pd.to_numeric(d["duration"],errors="coerce")
ct=d["create_dt"]
F["hour_sin"]=np.sin(2*np.pi*ct.dt.hour/24); F["hour_cos"]=np.cos(2*np.pi*ct.dt.hour/24)
F["dow_sin"]=np.sin(2*np.pi*ct.dt.dayofweek/7); F["dow_cos"]=np.cos(2*np.pi*ct.dt.dayofweek/7)
F["is_weekend"]=(ct.dt.dayofweek>=5).astype(int)
F["is_english"]=pd.to_numeric(d["is_english"],errors="coerce").fillna(0).astype(int)
mfreq=d.loc[d.split=="train","music_id"].value_counts(); F["music_freq"]=d["music_id"].map(mfreq).fillna(0.0)
A=list(F.columns)
# B extra
F["log_play_d1"]=np.log1p(cum1["play_count"].reindex(d.index))
F["log_like_d1"]=np.log1p(cum1["like_count"].reindex(d.index))
F["er_d1"]=(cum1[ENG].sum(axis=1)/cum1["play_count"].clip(lower=1)).reindex(d.index)
amean=F.loc[d.split=="train"].assign(a=d["author_id"]).groupby("a")["log_play_d1"].mean()
F["log_play_d1_vs_author"]=F["log_play_d1"]-d["author_id"].map(amean).fillna(F.loc[d.split=="train","log_play_d1"].mean())
B=A+["log_play_d1","log_like_d1","er_d1","log_play_d1_vs_author"]
log("features ready")

# followers asof
def foll_asof(when):
    L=pd.DataFrame({"vid":d.index.values,"author_id":d["author_id"].values,"_w":when.values}).dropna(subset=["_w"]).sort_values("_w")
    Rr=c.assign(date=pd.to_datetime(c["date"],errors="coerce"))[["author_id","date","follower_count"]].dropna().sort_values("date")
    m=pd.merge_asof(L,Rr,left_on="_w",right_on="date",by="author_id",direction="backward")
    return pd.Series(m["follower_count"].values,index=m["vid"]).reindex(d.index)
d["foll_post"]=foll_asof(d["create_dt"]); d["foll_H"]=foll_asof(d["create_dt"]+pd.Timedelta(days=H))
log(f"followers ready (coverage {d['foll_post'].notna().mean():.3f})")

playH=cumH["play_count"].reindex(d.index)
score=pd.DataFrame(index=d.index)
score["within-creator ER"]=erH_d-erH_d.groupby(d["author_id"].values).transform("median")
score["breakout"]=playH/d["foll_post"].clip(lower=1)
score["follower_grow"]=(d["foll_H"]-d["foll_post"])/d["foll_post"].clip(lower=1)
score["abs_views"]=playH
score["breakout (within-creator)"]=score["breakout"]-score["breakout"].groupby(d["author_id"].values).transform("median")

elig=(playH>=min_play)&y_er.notna()&d.index.isin(cum1.index)&d["foll_post"].notna()
trm=elig&(d["split"]=="train"); tem=elig&(d["split"]=="test")
log(f"eligible={int(elig.sum())} train={int(trm.sum())} test={int(tem.sum())}")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def binz(s): return (s>s[trm].median()).astype("float")
def fauc(y,cols):
    Xtr=F.loc[trm,cols].fillna(F.loc[trm,cols].median()); Xte=F.loc[tem,cols].fillna(F.loc[trm,cols].median())
    ytr,yte=y[trm],y[tem]
    if ytr.dropna().nunique()<2 or yte.dropna().nunique()<2: return float("nan")
    m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.08,max_iter=150,l2_regularization=1.0)
    m.fit(Xtr,ytr.astype(int)); return roc_auc_score(yte.astype(int),m.predict_proba(Xte)[:,1])
def sauc(y,x):
    yy,xx=y[tem],x[tem]; ok=yy.notna()&xx.notna()
    if yy[ok].nunique()<2: return float("nan")
    a=roc_auc_score(yy[ok].astype(int),xx[ok]); return max(a,1-a)
rows=[]
for name in score.columns:
    y=binz(score[name])
    rows.append({"мітка":name,"base_rate":round(float(y[elig].mean()),3),
      "fame_leak_AUC":round(sauc(y,d["foll_post"]),3),
      "content_AUC_A":round(fauc(y,A),3),"early_AUC_B":round(fauc(y,B),3),
      "spearman_vs_ER":round(spearmanr(score[name],score["within-creator ER"],nan_policy="omit").correlation,3)})
    log(f"done {name}")
out=pd.DataFrame(rows)
print("\n=== РЕЗУЛЬТАТ: порівняння міток (eligible, тест) ===")
print(out.to_string(index=False))
out.to_csv("reports/label_comparison.csv",index=False)
print("\nSAVED reports/label_comparison.csv")
log("ALL DONE")
