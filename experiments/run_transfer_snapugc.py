"""P3.2 — SnapUGC watch%/retention transfer (the literature-backed top lever).

Pretrain a small head on SnapUGC (raw frames+audio -> NAWP/ECR watch-retention labels),
then APPLY it to lingbow videos' content embeddings to inject the dominant-but-missing
watch-time signal as a pre-publication feature, and ablate it.

Pipeline (all local):
  1. Requires data/snapugc/labels.csv (video_id,nawp,ecr) + data/snapugc/videos/<id>.mp4.
  2. Extract CLIP+SigLIP+CLAP embeddings for SnapUGC (reuse multimodal.extract.Encoders).
  3. Train an MLP head: [vclip|vsig|clap] -> (nawp, ecr).   (torch/MPS)
  4. Apply the head to lingbow multimodal features.parquet (same embedding columns) ->
     data/processed/snapugc_retention.parquet (video_id, pred_nawp, pred_ecr).
  5. Ablate on the lingbow multimodal subset: best-A(mm) vs +retention_head.

Graceful: if SnapUGC data is absent, prints how to get it and exits 0 (so the batch
never breaks). Output: reports/transfer.csv
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C

SNAP = C.ROOT / "data" / "snapugc"
EMB_COLS = ([f"vclip{i}" for i in range(512)] + [f"vsig{i}" for i in range(768)] + [f"clap{i}" for i in range(512)])


def emb_for_snapugc(labels):
    """List aligned to labels rows: 1792-d embedding (or None). Prefers precomputed
    data/processed/snapugc_emb.parquet (scripts/extract_snapugc_features.py, parallel);
    only falls back to inline extraction for videos not yet in it."""
    ids = labels["video_id"].astype(str).tolist()
    pre = {}
    emb_path = C.PROC / "snapugc_emb.parquet"
    if emb_path.exists():
        d = pd.read_parquet(emb_path)
        M = d[EMB_COLS].values.astype(np.float32)
        pre = {str(v): M[i] for i, v in enumerate(d["video_id"].astype(str))}
        print(f"loaded {len(pre)} precomputed SnapUGC embeddings", flush=True)
    missing = [v for v in ids if v not in pre]
    enc = None
    if missing:                                    # lazy encoder only for not-yet-extracted ids
        sys.path.insert(0, str(C.ROOT / "multimodal"))
        import importlib, torch
        ex = importlib.import_module("extract")
        enc = ex.Encoders("mps" if torch.backends.mps.is_available() else "cpu", {"clip", "sig", "clap"})
    rows = []
    for vid in ids:
        if vid in pre:
            rows.append(pre[vid]); continue
        p = SNAP / "videos" / f"{vid}.mp4"
        if not p.exists():
            rows.append(None); continue
        uni, hook, seq, meta = ex.read_frames(p)
        if not uni:
            rows.append(None); continue
        vclip = enc.clip_emb(uni); vsig = enc.sig_emb(uni)
        y48, y16, wav = ex.extract_wav(p)
        clap = enc.clap_emb(y48, 48000) if (y48 is not None and getattr(y48, "size", 0)) else np.zeros(512, np.float32)
        if wav and os.path.exists(wav):
            os.remove(wav)
        rows.append(np.concatenate([vclip, vsig, clap]).astype(np.float32))
    return rows


def main():
    if not (SNAP / "labels.csv").exists():
        print("SnapUGC not found. Build a bounded subset first:")
        print("  PYTHONPATH=src python scripts/download_snapugc.py   # -> data/snapugc/{videos,labels.csv}")
        print("  (SnapUGC train CSV = per-video Snapchat-CDN links + ECR label; ~half still resolve.)")
        print("Skipping (bounded transfer already measured: no lift — reports/transfer.csv, RESULTS §7.1).")
        return
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    labels = pd.read_csv(SNAP / "labels.csv", dtype={"video_id": str})

    embs = emb_for_snapugc(labels)
    keep = [i for i, e in enumerate(embs) if e is not None]
    Xs = np.stack([embs[i] for i in keep]); ys = labels.iloc[keep][["nawp", "ecr"]].values.astype(np.float32)
    print(f"SnapUGC usable: {len(keep)}/{len(labels)}", flush=True)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    Xsn = (Xs - mu) / sd
    n = len(Xsn); tr = np.arange(n) < int(0.85 * n)
    Xt = torch.tensor(Xsn[tr], device=dev); Yt = torch.tensor(ys[tr], device=dev)

    head = nn.Sequential(nn.Linear(Xs.shape[1], 256), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4); mse = nn.MSELoss()
    for _ in range(200):
        head.train(); opt.zero_grad()
        loss = mse(head(Xt), Yt); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pv = head(torch.tensor(Xsn[~tr], device=dev)).cpu().numpy()
    val_sp = [round(float(spearmanr(pv[:, k], ys[~tr][:, k]).correlation), 3) for k in range(2)]
    print(f"head val Spearman NAWP/ECR: {val_sp}", flush=True)

    # apply to lingbow multimodal features
    mm = pd.read_parquet(C.MM / "features.parquet"); mm["video_id"] = mm["video_id"].astype(str)
    have = [c for c in EMB_COLS if c in mm.columns]
    Xl = ((mm[have].values.astype(np.float32) - mu[[EMB_COLS.index(c) for c in have]])
          / sd[[EMB_COLS.index(c) for c in have]])
    with torch.no_grad():
        pl = head(torch.tensor(Xl, device=dev)).cpu().numpy()
    pd.DataFrame({"video_id": mm["video_id"].values, "pred_nawp": pl[:, 0], "pred_ecr": pl[:, 1]}) \
        .to_parquet(C.PROC / "snapugc_retention.parquet", index=False)

    # ablation on lingbow multimodal subset
    from sip import data as D, features as F, experiment as X
    F._retention_cache.cache_clear()
    df = D.load(eligible_only=True)
    sub = df[df["video_id"].astype(str).isin(set(mm["video_id"]))].reset_index(drop=True); F.add_derived(sub)
    TAB = ["caption", "emotion", "duration", "timing", "meta"]
    rows = []
    for target in ("y_breakout_wc", "y_er_wc"):
        for name, blocks in [("mm_best", TAB + ["text_emb:minilm", "mm:vsig", "mm:clap"]),
                             ("mm_best+retention", TAB + ["text_emb:minilm", "mm:vsig", "mm:clap", "retention_head"])]:
            r = X.run(sub, blocks, "hgb", target=target, model_kind="A", n_boot=400,
                      label=f"transfer:{name}[{target}]", log=True, verbose=True)
            rows.append({"target": target, "config": name, "loco_auc": r["loco"]["roc_auc"],
                         "loco_ci": r["loco"]["roc_auc_ci"], "n": len(sub)})
    pd.DataFrame(rows).to_csv(C.REPORTS / "transfer.csv", index=False)
    print("saved -> reports/transfer.csv ; snapugc_retention.parquet")

    # --- record this run into the scale-up ledger (source of truth for the notebook section) ---
    import json
    n_head = len(keep)
    def g(t, c):
        return round(float(next(r["loco_auc"] for r in rows if r["target"] == t and r["config"] == c)), 4)
    entry = {
        "label": f"scale-up ({n_head} SnapUGC)",
        "n_snapugc_head": int(n_head),
        "head_spearman_ecr": val_sp[1],
        "breakout": {"mm_best": g("y_breakout_wc", "mm_best"),
                     "mm_best_retention": g("y_breakout_wc", "mm_best+retention"),
                     "delta": round(g("y_breakout_wc", "mm_best+retention") - g("y_breakout_wc", "mm_best"), 4)},
        "er": {"mm_best": g("y_er_wc", "mm_best"),
               "mm_best_retention": g("y_er_wc", "mm_best+retention"),
               "delta": round(g("y_er_wc", "mm_best+retention") - g("y_er_wc", "mm_best"), 4)},
    }
    sj = C.REPORTS / "snapugc_scaleup.json"
    data = json.load(open(sj)) if sj.exists() else {"ablation": {"runs": []}}
    data.setdefault("extraction", {})["n_embedded"] = int(n_head)
    runs = data.setdefault("ablation", {}).setdefault("runs", [])
    runs = [r for r in runs if r.get("n_snapugc_head") != int(n_head)] + [entry]
    data["ablation"]["runs"] = runs
    json.dump(data, open(sj, "w"), indent=2)
    print(f"updated reports/snapugc_scaleup.json (n_snapugc_head={n_head})")


if __name__ == "__main__":
    main()
