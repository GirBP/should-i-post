#!/usr/bin/env python3
"""
Експеримент: creator-fit — чи допомагає схожість кандидата на ВЛАСНІ минулі хіти автора.
Темпорально-безпечно: для кожного відео беремо центроїд лише СТРОГО ранніших відео того ж
автора з y==1 (минулі переможці) у семантичному просторі (SVD), і косинус-схожість до нього.
Це within-creator перенос (не cross-creator) — останній неперевірений важіль до публікації.

Залежить від кешу ознак (data/processed/_feat_cache.parquet), що його будує
experiments/cross_creator_theories.py. Запуск:  python experiments/creator_fit.py
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore"); t0=time.time()
log=lambda m: print(f"[{time.time()-t0:5.1f}s] {m}", flush=True)
ROOT=Path(__file__).resolve().parents[1]

cache=ROOT/"data/processed/_feat_cache.parquet"
if not cache.exists(): cache=Path("/tmp/feat_theory.parquet")
D=pd.read_parquet(cache)
v=pd.read_parquet(ROOT/"data/raw/lingbow/videos.parquet", columns=["video_id","create_date"]).set_index("video_id")
D=D.join(v); D["create_dt"]=pd.to_datetime(D["create_date"],errors="coerce")
D=D[D["elig"]==1].copy()
SV=[f"sv{i}" for i in range(60)]
E=D[SV].values.astype(np.float32); E=E/ (np.linalg.norm(E,axis=1,keepdims=True)+1e-8)   # L2-норма
log(f"завантажено {len(D)} відео")

# creator-fit: косинус до центроїда СТРОГО ранніших переможців того ж автора
D=D.reset_index(drop=False).rename(columns={"index":"video_id"}) if "video_id" not in D.columns else D.reset_index(drop=True)
order=np.argsort(D["create_dt"].values.astype("datetime64[ns]"))
auth=D["author_id"].values; y=D["y"].values
fit=np.zeros(len(D),np.float32); nprev=np.zeros(len(D),np.float32)
from collections import defaultdict
sums=defaultdict(lambda: np.zeros(60,np.float32)); cnts=defaultdict(int)
for i in order:
    a=auth[i]
    if cnts[a]>0:
        c=sums[a]/cnts[a]; c=c/(np.linalg.norm(c)+1e-8)
        fit[i]=float(E[i]@c); nprev[i]=cnts[a]
    if y[i]==1:                     # додаємо ПІСЛЯ оцінки → лише минуле
        sums[a]+=E[i]; cnts[a]+=1
D["creator_fit"]=fit; D["n_prior_wins"]=np.log1p(nprev)
log(f"creator-fit пораховано (частка з історією: {(nprev>0).mean():.2f})")

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
groups=D["author_id"].values
def loco(cols):
    X=D[cols].astype(float).values; oof=np.full(len(D),np.nan)
    for tr,te in GroupKFold(4).split(X,y,groups):
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
        m.fit(np.nan_to_num(X[tr]),y[tr]); oof[te]=m.predict_proba(np.nan_to_num(X[te]))[:,1]
    ok=~np.isnan(oof); return roc_auc_score(y[ok],oof[ok])

res={
 "semantic_only (SVD60)": round(loco(SV),3),
 "creator_fit_only": round(loco(["creator_fit","n_prior_wins"]),3),
 "semantic + creator_fit": round(loco(SV+["creator_fit","n_prior_wins"]),3),
}
print("\n=== creator-fit (LOCO, breakout-wc) ===")
for k,val in res.items(): print(f"  {k:28} AUC={val}")
import json; (ROOT/"reports").mkdir(exist_ok=True)
json.dump(res, open(ROOT/"reports/creator_fit.json","w"), ensure_ascii=False, indent=2)
log("DONE -> reports/creator_fit.json")
