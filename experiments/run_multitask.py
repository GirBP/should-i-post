"""P1.5 — Multi-task learning (shared-trunk) vs single-task (deep-learning domain).

A shared MLP trunk with three heads (breakout, ER, follower-growth) trained jointly,
vs the same trunk trained on breakout only. Tests whether auxiliary engagement targets
regularize the pre-publication representation. Temporal split, MPS. Output: reports/multitask.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, splits as S, config as C
from sklearn.metrics import roc_auc_score

TAB = ["caption", "emotion", "duration", "timing", "meta"]


def best_blocks():
    for n in ("bge", "minilm", "e5"):
        if (C.PROC / f"emb_{n}.parquet").exists():
            return TAB + [f"text_emb:{n}", f"creator_fit:{n}", "trend_fit"]
    return TAB + ["semantic", "creator_fit", "trend_fit"]


def main():
    import torch, torch.nn as nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    df = D.load(eligible_only=True); F.add_derived(df)
    tr, va, te = S.temporal_masks(df)
    tri = np.where(tr)[0]
    targets = ["y_breakout_wc", "y_er_wc", "y_follgrow"]
    Y = np.stack([df[t].astype(float).values for t in targets], 1)
    X, names = F.build_blocks(df, best_blocks(), tri, df["y_breakout_wc"].astype(int).values)
    X = np.nan_to_num(X).astype(np.float32)
    mu, sd = X[tri].mean(0), X[tri].std(0) + 1e-6
    Xn = (X - mu) / sd

    Xtr = torch.tensor(Xn[tri], device=dev); Ytr = torch.tensor(Y[tri], dtype=torch.float32, device=dev)
    Xte = torch.tensor(Xn[te], device=dev); yte = Y[te][:, 0].astype(int)
    d = X.shape[1]

    def make():
        trunk = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(128, 64), nn.ReLU()).to(dev)
        heads = nn.ModuleList([nn.Linear(64, 1).to(dev) for _ in range(3)])
        return trunk, heads

    def train(n_heads):
        trunk, heads = make()
        params = list(trunk.parameters()) + [p for h in heads[:n_heads] for p in h.parameters()]
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()
        for ep in range(60):
            trunk.train(); opt.zero_grad()
            z = trunk(Xtr)
            loss = sum(bce(heads[k](z).squeeze(1), Ytr[:, k]) for k in range(n_heads))
            loss.backward(); opt.step()
        trunk.eval()
        with torch.no_grad():
            p = torch.sigmoid(heads[0](trunk(Xte)).squeeze(1)).cpu().numpy()
        return roc_auc_score(yte, p)

    rows = []
    for name, nh in [("single-task (breakout)", 1), ("multi-task (3 heads)", 3)]:
        aucs = [train(nh) for _ in range(3)]  # 3 seeds-ish (dropout/init variance)
        rows.append({"model": name, "breakout_temporal_auc": round(float(np.mean(aucs)), 4),
                     "std": round(float(np.std(aucs)), 4)})
        print(f"  {name}: AUC={np.mean(aucs):.4f} ±{np.std(aucs):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.REPORTS / "multitask.csv", index=False)
    print("saved -> reports/multitask.csv")


if __name__ == "__main__":
    main()
