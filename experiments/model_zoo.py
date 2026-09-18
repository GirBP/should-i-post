"""Literature-grounded model comparison for multimodal pre-publication classification.

Same balanced sample, same leave-one-creator-out folds, same bootstrap CI — four models:
  HGB     — gradient-boosted trees (strong on tabular, axis-aligned → weak on dense embeddings)
  linear  — L2 logistic regression on standardized features (canonical embedding probe)
  mlp     — 2-hidden MLP, dropout + weight-decay + early stopping
  fusion  — per-modality encoders → Gated Multimodal Unit (Arevalo et al., 2017) → head

Scientific canon: StandardScaler fit on each TRAIN fold only, StratifiedGroupKFold by author
(leave-one-creator-out), out-of-fold predictions, 1000× bootstrap 95% CI, Brier for calibration.
The multimodal (video/audio) embeddings load from --mm-parquet so the SAME harness compares
whole-video vs hook-only feature sets by swapping that file.

    PYTHONPATH=src python experiments/model_zoo.py --n 10000 --mm-parquet data/multimodal/features_tiktok.parquet
"""
import os, sys, argparse, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); os.environ.setdefault("SIP_DEVICE", "cpu")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from sip import data as D, features as F, config as C
from sip.modeling import make_model

TARGET, SEED = "y_breakout_wc", 0
TAB_TEXT = ["caption", "duration", "timing", "meta", "missing", "text_emb:minilm"]  # non-mm blocks


def load_xy(n, mm_parquet):
    df = D.load(eligible_only=True); df["video_id"] = df["video_id"].astype(str)
    mm = pd.read_parquet(mm_parquet); mm["video_id"] = mm["video_id"].astype(str)
    vsig = [c for c in mm.columns if c.startswith("vsig")]; clap = [c for c in mm.columns if c.startswith("clap")]
    have = df[df["video_id"].isin(set(mm["video_id"]))].copy()
    rng = np.random.RandomState(SEED); per = n // 2
    sub = pd.concat([have[have[TARGET] == c].sample(min(per, (have[TARGET] == c).sum()), random_state=rng)
                     for c in (1, 0)]).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    F.add_derived(sub); y = sub[TARGET].astype(int).values; groups = sub["author_id"].values
    Xtt, names = F.build_blocks(sub, TAB_TEXT, np.arange(len(sub)), y, target_col=TARGET)
    mmi = mm.set_index("video_id")
    Xv = mmi[vsig].reindex(sub["video_id"]).fillna(0).values.astype(np.float32)
    Xa = mmi[clap].reindex(sub["video_id"]).fillna(0).values.astype(np.float32)
    # modality spans for the fusion model: [tab_text | video | audio]
    spans = {"tabtext": (0, Xtt.shape[1]), "video": (Xtt.shape[1], Xtt.shape[1] + Xv.shape[1]),
             "audio": (Xtt.shape[1] + Xv.shape[1], Xtt.shape[1] + Xv.shape[1] + Xa.shape[1])}
    X = np.concatenate([Xtt, Xv, Xa], axis=1).astype(np.float32)
    return X, y, groups, spans


def torch_mlp(Xtr, ytr, Xte, spans=None, gated=False, epochs=120, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed); dev = "cpu"
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte)
    d = Xtr.shape[1]
    if gated and spans:
        class Fusion(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.ModuleDict({k: nn.Sequential(nn.Linear(b - a, 64), nn.ReLU()) for k, (a, b) in spans.items()})
                s.gate = nn.Linear(64 * len(spans), len(spans))
                s.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
                s.spans = spans
            def forward(s, x):
                hs = [s.enc[k](x[:, a:b]) for k, (a, b) in s.spans.items()]
                g = torch.softmax(s.gate(torch.cat(hs, 1)), 1)               # gated multimodal unit
                fused = sum(g[:, i:i+1] * hs[i] for i in range(len(hs)))
                return s.head(fused).squeeze(1)
        net = Fusion()
    else:
        net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
        _f = net.forward
        net.forward = lambda x: _f(x).squeeze(1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    idx = np.arange(len(Xtr)); rng = np.random.RandomState(seed); rng.shuffle(idx)
    val = idx[: len(idx) // 6]; tr = idx[len(idx) // 6:]
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xtr_t[tr]), ytr_t[tr]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(Xtr_t[val]), ytr_t[val]).item()
        if vl < best - 1e-4: best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > 15: break
    if best_state: net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xte_t)).numpy()


def boot_ci(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed); aucs = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) == 2: aucs.append(roc_auc_score(y[i], p[i]))
    return round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--mm-parquet", default=str(C.MM / "features_tiktok.parquet"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="whole")
    args = ap.parse_args()
    X, y, groups, spans = load_xy(args.n, args.mm_parquet)
    print(f"n={len(y)} | features={X.shape[1]} | creators={pd.Series(groups).nunique()} | mm={args.mm_parquet}", flush=True)
    folds = list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED).split(X, y, groups))

    def cv(fit_predict, standardize):
        oof = np.full(len(y), np.nan)
        for tri, tei in folds:
            Xtr, Xte = X[tri], X[tei]
            if standardize:
                sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            oof[tei] = fit_predict(Xtr.astype(np.float32), y[tri], Xte.astype(np.float32))
        return oof

    runs = {
        "HGB (trees)":      (lambda a, b, c: make_model("hgb").fit(a, b).predict_proba(c)[:, 1], False),
        "Linear probe":     (lambda a, b, c: LogisticRegression(max_iter=2000, C=1.0).fit(a, b).predict_proba(c)[:, 1], True),
        "MLP (2-hidden)":   (lambda a, b, c: torch_mlp(a, b, c), True),
        "Gated fusion":     (lambda a, b, c: torch_mlp(a, b, c, spans=spans, gated=True), True),
    }
    print(f"\n=== MODEL ZOO — {args.tag} features, LOCO OOF, N={len(y)} ===")
    rows = []
    for name, (fp, std) in runs.items():
        oof = cv(fp, std); auc = roc_auc_score(y, oof); lo, hi = boot_ci(y, oof); br = brier_score_loss(y, oof)
        print(f"  {name:16} ROC-AUC {auc:.3f}  CI [{lo}, {hi}]  Brier {br:.3f}", flush=True)
        rows.append({"tag": args.tag, "model": name, "auc": round(auc, 4), "ci_lo": lo, "ci_hi": hi, "brier": round(br, 4)})
    out = C.REPORTS / f"model_zoo_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False); print(f"saved -> {out}")


if __name__ == "__main__":
    main()
