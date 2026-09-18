import pandas as pd, numpy as np, warnings, time, os
warnings.filterwarnings("ignore"); t0=time.time()
log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}",flush=True)
RES="reports/model_power.csv"
D=pd.read_parquet("data/processed/_feat_cache.parquet"); D=D[D["elig"]==1].copy()
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
y=D["y"].values; groups=D["author_id"].values
num=["char_len","word_len","n_hashtags","n_mentions","has_question","has_exclam","n_emoji","has_url",
     "anger","joy","surprise","sadness","disgust","fear","arousal","speaking_rate","duration_s",
     "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend"]+[f"sv{i}" for i in range(60)]
Xb=np.nan_to_num(D[num].astype(float).values)
folds=list(GroupKFold(n_splits=3).split(D,y,groups))
def add_priors(tr):
    sub=D.iloc[tr]; tg=sub.groupby("topic")["y"].mean(); mg=sub.groupby("music_id")["y"].mean(); gm=sub["y"].mean()
    return np.c_[Xb, D["topic"].map(tg).fillna(gm).values, D["music_id"].map(mg).fillna(gm).values]
def run(name):
    oof=np.full(len(D),np.nan)
    for tr,te in folds:
        X=add_priors(tr)
        if name=="logreg":
            sc=StandardScaler().fit(X[tr]); m=LogisticRegression(max_iter=500); m.fit(sc.transform(X[tr]),y[tr]); p=m.predict_proba(sc.transform(X[te]))[:,1]
        elif name=="hgb_tuned":
            m=HistGradientBoostingClassifier(max_depth=None,learning_rate=0.03,max_iter=600,l2_regularization=2.0,max_leaf_nodes=63,early_stopping=True,validation_fraction=0.1)
            m.fit(X[tr],y[tr]); p=m.predict_proba(X[te])[:,1]
        elif name=="mlp":
            idx=tr if len(tr)<=60000 else np.random.RandomState(0).choice(tr,60000,replace=False)
            sc=StandardScaler().fit(X[idx]); m=MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=80,early_stopping=True)
            m.fit(sc.transform(X[idx]),y[idx]); p=m.predict_proba(sc.transform(X[te]))[:,1]
        oof[te]=p
        log(f"  {name} fold done")
    np.save(f"data/processed/_oof_{name}.npy",oof)
    ok=~np.isnan(oof); return roc_auc_score(y[ok],oof[ok])
done={}
if os.path.exists(RES):
    for _,r in pd.read_csv(RES).iterrows(): done[r["model"]]=r["LOCO_AUC"]
for name in ["logreg","hgb_tuned","mlp"]:
    if name in done: continue
    a=run(name); done[name]=round(a,3)
    pd.DataFrame([{"model":k,"LOCO_AUC":v} for k,v in done.items()]).to_csv(RES,index=False)
    log(f"{name}: AUC={done[name]}")
# stack if all present
if all(os.path.exists(f"data/processed/_oof_{n}.npy") for n in ["logreg","hgb_tuned","mlp"]) and "stack" not in done:
    O=[np.load(f"data/processed/_oof_{n}.npy") for n in ["logreg","hgb_tuned","mlp"]]
    s=np.nanmean(O,axis=0); ok=~np.isnan(s); done["stack"]=round(roc_auc_score(y[ok],s[ok]),3)
    pd.DataFrame([{"model":k,"LOCO_AUC":v} for k,v in done.items()]).to_csv(RES,index=False)
print("\n=== Модель A (breakout-wc, LOCO): максимальна потужність ===")
print(pd.read_csv(RES).to_string(index=False)); log("DONE")
