"""Run a broad sweep of so-far-unused methods in PARALLEL, + a rule-based system.

CPU classifiers run in a ProcessPool (true multiprocessing); torch models (MPS) run
sequentially after; methods needing missing data / fragile installs are attempted and
skipped gracefully with a reason. All on the within-creator breakout target, a fixed
best-A feature set, temporal split (single fit -> fast across many methods).

Output: reports/extra_methods.csv + reports/extra_methods.md
Honest framing: the ceiling is information-bound, so same-information algorithms are
expected to land near the ~0.55-0.59 baseline; only new-information methods can exceed it.
"""
import os, sys, time, warnings, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sip import data as D, features as F, splits as S, config as C
from sklearn.metrics import roc_auc_score

TMP = Path("/tmp/sip_extra"); TMP.mkdir(exist_ok=True)
TAB = ["caption", "emotion", "duration", "timing", "meta"]


def _enc():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return n
    return "svd"


# ---- module-level CPU model zoo (each must expose predict_proba) ----
def _model(name):
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,
                                  HistGradientBoostingClassifier)
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    z = {
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=0),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, n_jobs=1, random_state=0),
        "gradient_boosting": GradientBoostingClassifier(random_state=0),
        "adaboost": AdaBoostClassifier(random_state=0),
        "bagging": BaggingClassifier(n_estimators=50, n_jobs=1, random_state=0),
        "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "sgd_log": SGDClassifier(loss="log_loss", max_iter=50, random_state=0),
        "knn": KNeighborsClassifier(n_neighbors=50, n_jobs=1),
        "gaussian_nb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(),
        "mlp_sklearn": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, random_state=0),
        "svc_rbf": SVC(probability=True, random_state=0),
    }
    return z[name]


SLOW = {"svc_rbf", "gaussian_boosting"}  # subsample train for these


def _fit_eval(name):
    t = time.time()
    Xtr = np.load(TMP / "Xtr.npy"); ytr = np.load(TMP / "ytr.npy")
    Xte = np.load(TMP / "Xte.npy"); yte = np.load(TMP / "yte.npy")
    if name in ("svc_rbf",) and len(Xtr) > 8000:        # rbf-SVC O(n^2) -> subsample
        rng = np.random.default_rng(0); idx = rng.choice(len(Xtr), 8000, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    try:
        m = _model(name).fit(Xtr, ytr)
        auc = float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        return {"method": name, "family": "cpu-classifier", "auc": round(auc, 4),
                "secs": round(time.time() - t, 1)}
    except Exception as e:
        return {"method": name, "family": "cpu-classifier", "auc": None, "note": f"{type(e).__name__}: {str(e)[:60]}"}


def main():
    import concurrent.futures as cf
    df = D.load(eligible_only=True); F.add_derived(df)
    # stratified 60k sample for a fast, broad sweep
    n = 60000
    sub = (df.groupby(["split_temporal", "y_breakout_wc"], group_keys=False)
             .sample(frac=min(1.0, n / len(df)), random_state=0).reset_index(drop=True))
    enc = _enc()
    blocks = TAB + ([f"text_emb:{enc}", f"creator_fit:{enc}", "trend_fit"] if enc != "svd"
                    else ["semantic", "creator_fit", "trend_fit"])
    y = sub["y_breakout_wc"].astype(int).values
    tr, va, te = S.temporal_masks(sub); tri = np.where(tr)[0]
    X, names = F.build_blocks(sub, blocks, tri, y)
    X = np.nan_to_num(X).astype(np.float32)
    np.save(TMP / "Xtr.npy", X[tr]); np.save(TMP / "ytr.npy", y[tr])
    np.save(TMP / "Xte.npy", X[te]); np.save(TMP / "yte.npy", y[te])
    yte = y[te]
    print(f"sample={len(sub)} feats={X.shape[1]} enc={enc} | running CPU zoo in parallel", flush=True)

    results = []
    # baseline ref
    from sip.modeling import make_model
    base = make_model("hgb").fit(X[tr], y[tr])
    results.append({"method": "best-A HGB (reference)", "family": "baseline",
                    "auc": round(float(roc_auc_score(yte, base.predict_proba(X[te])[:, 1])), 4)})

    # ---- CPU zoo in parallel ----
    zoo = ["random_forest", "extra_trees", "gradient_boosting", "adaboost", "bagging",
           "logreg_balanced", "sgd_log", "knn", "gaussian_nb", "lda", "qda", "mlp_sklearn", "svc_rbf"]
    with cf.ProcessPoolExecutor(max_workers=min(8, (os.cpu_count() or 4) - 1)) as ex:
        for r in ex.map(_fit_eval, zoo):
            print("  ", r, flush=True); results.append(r)

    # ---- rule-based system (no training) ----
    from sip.rules import RuleBasedScorer
    rs = RuleBasedScorer()
    p_rule = rs.predict_proba(sub.iloc[te])[:, 1]
    results.append({"method": "RULE-BASED (expert system)", "family": "rules",
                    "auc": round(float(roc_auc_score(yte, p_rule)), 4)})

    # ---- graph feature: 1-hop hashtag-neighbour train success (fold-safe) ----
    try:
        results.append(_graph_neighbor(sub, tr, te, y, yte))
    except Exception as e:
        results.append({"method": "graph_hashtag_neighbor", "family": "graph", "auc": None, "note": str(e)[:60]})

    # ---- torch methods (MPS, sequential) ----
    try:
        results += _torch_methods(X, y, tr, te, sub)
    except Exception as e:
        results.append({"method": "torch_methods", "family": "torch", "auc": None, "note": str(e)[:80]})

    # ---- fragile installs: TabPFN / AutoGluon / TabNet (try, skip gracefully) ----
    results += _try_installs(X[tr], y[tr], X[te], yte)

    out = pd.DataFrame(results)
    out.to_csv(C.REPORTS / "extra_methods.csv", index=False)
    md = ["# Extra methods sweep (temporal test, within-creator breakout)\n",
          f"_sample n={len(sub)}, best-A features, encoder={enc}. Ceiling is information-bound: "
          "same-information methods cluster near baseline; only new-information beats it._\n",
          out.sort_values("auc", ascending=False, na_position="last").to_markdown(index=False)]
    (C.REPORTS / "extra_methods.md").write_text("\n".join(md))
    print("saved -> reports/extra_methods.csv / .md", flush=True)


def _graph_neighbor(sub, tr, te, y, yte):
    """Mean TRAIN success of videos sharing >=1 hashtag (1-hop graph neighbourhood)."""
    def hn(x):
        try:
            return [h["hashtag_name"] for h in x] if x is not None and len(x) else []
        except Exception:
            return []
    tags = sub["hashtags"].map(hn)
    tag_rate = {}
    for t_list, yy, is_tr in zip(tags, y, tr):
        if is_tr:
            for h in t_list:
                a = tag_rate.setdefault(h, [0, 0]); a[0] += yy; a[1] += 1
    gm = float(y[tr].mean())
    def val(tl):
        r = [tag_rate[h][0] / tag_rate[h][1] for h in tl if h in tag_rate and tag_rate[h][1] >= 5]
        return np.mean(r) if r else gm
    feat = sub["hashtags"].map(hn).map(val).values.reshape(-1, 1)
    Xtr = np.c_[np.load(TMP / "Xtr.npy"), feat[tr]]; Xte = np.c_[np.load(TMP / "Xte.npy"), feat[te]]
    from sip.modeling import make_model
    m = make_model("hgb").fit(Xtr, y[tr])
    return {"method": "best-A + graph_hashtag_neighbor", "family": "graph",
            "auc": round(float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1])), 4)}


def _torch_methods(X, y, tr, te, sub):
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.tensor(Xn[tr], device=dev); yt = torch.tensor(y[tr].astype(np.float32), device=dev)
    Xe = torch.tensor(Xn[te], device=dev); yte = y[te]
    d = X.shape[1]; out = []

    def mlp():
        return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)).to(dev)

    def train(m, loss_fn, epochs=80):
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); bs = 4096; nN = len(Xt)
        for _ in range(epochs):
            m.train(); perm = torch.randperm(nN, device=dev)
            for i in range(0, nN, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss_fn(m(Xt[idx]).squeeze(1), yt[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            return torch.sigmoid(m(Xe).squeeze(1)).cpu().numpy()

    bce = nn.BCEWithLogitsLoss()
    # deep ensemble (5 MLPs)
    ps = []
    for s in range(5):
        torch.manual_seed(s); ps.append(train(mlp(), bce, 60))
    out.append({"method": "deep_ensemble(5xMLP)", "family": "torch",
                "auc": round(float(roc_auc_score(yte, np.mean(ps, 0))), 4)})
    # focal loss
    def focal(logits, tgt, g=2.0):
        p = torch.sigmoid(logits); ce = bce(logits, tgt)
        pt = p * tgt + (1 - p) * (1 - tgt)
        return (((1 - pt) ** g) * nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none")).mean()
    out.append({"method": "focal_loss_mlp", "family": "torch",
                "auc": round(float(roc_auc_score(yte, train(mlp(), focal, 80))), 4)})
    return out


def _try_installs(Xtr, ytr, Xte, yte):
    res = []
    # TabPFN
    try:
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        from tabpfn import TabPFNClassifier
        rng = np.random.default_rng(0); s = rng.choice(len(Xtr), min(8000, len(Xtr)), replace=False)
        from sklearn.decomposition import PCA
        p = PCA(64, random_state=0).fit(Xtr[s]); a = p.transform(Xtr[s]); b = p.transform(Xte)
        clf = TabPFNClassifier(); clf.fit(a, ytr[s])
        res.append({"method": "TabPFN-v2", "family": "tabular-foundation",
                    "auc": round(float(roc_auc_score(yte, clf.predict_proba(b)[:, 1])), 4)})
    except Exception as e:
        res.append({"method": "TabPFN-v2", "family": "tabular-foundation", "auc": None, "note": f"skip: {type(e).__name__}"})
    # pytorch-tabnet
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier(verbose=0); clf.fit(Xtr, ytr, max_epochs=30, patience=8)
        res.append({"method": "TabNet", "family": "tabular-dl",
                    "auc": round(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])), 4)})
    except Exception as e:
        res.append({"method": "TabNet", "family": "tabular-dl", "auc": None, "note": f"skip: {type(e).__name__}"})
    return res


if __name__ == "__main__":
    main()
