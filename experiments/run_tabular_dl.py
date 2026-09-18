"""P1.4 — Deep / foundation tabular models vs GBT (tabular-DL domain).

Tests whether modern tabular DL beats gradient boosting on our pre-publication
features: TabPFN-v2 (tabular foundation model, in-context) and a compact
FT-Transformer (feature-tokenizer transformer). Both vs HGB/XGB on the same
best-A feature set, temporal split. Graceful skip + honest note if a package
won't install on Python 3.14. Output: reports/tabular_dl.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid torch+xgboost libomp segfault on macOS
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, config as C
from sip.modeling import make_model
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"]
    return TAB + ["semantic", "creator_fit", "trend_fit"]


def ft_transformer_auc(Xtr, ytr, Xte, yte, dev, d_token=32, depth=3, heads=4, epochs=40):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    d = Xtr.shape[1]
    Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
    Xte_t = torch.tensor(Xte, device=dev)

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(d, d_token) * 0.02)
            self.b = nn.Parameter(torch.zeros(d, d_token))
            self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
            enc = nn.TransformerEncoderLayer(d_token, heads, d_token * 2, dropout=0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, depth)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b               # (B, d, d_token)
            z = torch.cat([self.cls.expand(x.size(0), -1, -1), tok], 1)
            z = self.tr(z)
            return self.head(z[:, 0]).squeeze(1)

    m = FT().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    bs, n = 4096, Xtr_t.shape[0]
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(); loss = bce(m(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = np.concatenate([torch.sigmoid(m(Xte_t[i:i + bs])).cpu().numpy()
                            for i in range(0, Xte_t.shape[0], bs)])
    return roc_auc_score(yte, p)


def main():
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    df = D.load(eligible_only=True); F.add_derived(df)
    tr, va, te = S.temporal_masks(df)
    tri, tei = np.where(tr)[0], np.where(te)[0]
    y = df["y_breakout_wc"].astype(int).values
    X, _ = F.build_blocks(df, best_blocks(), tri, y)
    X = np.nan_to_num(X).astype(np.float32)
    # standardize + PCA (keeps DL light; TabPFN feature cap)
    mu, sd = X[tri].mean(0), X[tri].std(0) + 1e-6
    Xn = (X - mu) / sd
    pca = PCA(n_components=min(64, X.shape[1]), random_state=0).fit(Xn[tri])
    Xp = pca.transform(Xn).astype(np.float32)
    rows = []

    # GBT reference on same PCA features (HGB only; XGBoost is run in run_tabular_text/
    # run_ensemble where torch is NOT imported — torch+xgboost double-load libomp and
    # hard-segfault in one process on macOS-ARM even with KMP_DUPLICATE_LIB_OK).
    m = make_model("hgb").fit(Xp[tri], y[tri])
    auc = roc_auc_score(y[te], m.predict_proba(Xp[te])[:, 1])
    rows.append({"model": "hgb", "auc": round(auc, 4), "note": "PCA-64; (xgb~0.555 see model_family)"})
    print(f"  hgb: {auc:.4f}", flush=True)

    # FT-Transformer
    try:
        auc = ft_transformer_auc(Xp[tri], y[tri].astype(np.float32), Xp[te], y[te], dev)
        rows.append({"model": "FT-Transformer", "auc": round(auc, 4), "note": "custom, PCA-64"})
        print(f"  FT-Transformer: {auc:.4f}", flush=True)
    except Exception as e:
        rows.append({"model": "FT-Transformer", "auc": None, "note": f"failed: {type(e).__name__}"})
        print("  FT-Transformer failed:", e)

    # TabPFN-v2 (sample to its limits)
    try:
        from tabpfn import TabPFNClassifier
        rng = np.random.default_rng(0)
        s = rng.choice(tri, min(10000, len(tri)), replace=False)
        clf = TabPFNClassifier(device=dev)
        clf.fit(Xp[s], y[s])
        st = rng.choice(tei, min(5000, len(tei)), replace=False)
        auc = roc_auc_score(y[st], clf.predict_proba(Xp[st])[:, 1])
        rows.append({"model": "TabPFN-v2", "auc": round(auc, 4), "note": f"train {len(s)}, test {len(st)}"})
        print(f"  TabPFN-v2: {auc:.4f}", flush=True)
    except Exception as e:
        rows.append({"model": "TabPFN-v2", "auc": None, "note": f"unavailable: {type(e).__name__}"})
        print("  TabPFN-v2 unavailable:", str(e)[:100])

    pd.DataFrame(rows).to_csv(C.REPORTS / "tabular_dl.csv", index=False)
    print("saved -> reports/tabular_dl.csv")


if __name__ == "__main__":
    main()
