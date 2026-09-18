#!/usr/bin/env python3
"""Cache sentence embeddings for the canonical text field (caption+summary+transcript)
with three encoders, for the "3 text encoders marginal lift" experiment and for the
main pipeline / creator-fit.  Output: data/processed/emb_{name}.parquet (float16,
keyed by video_id). Resumable: skips an encoder whose file already exists.

  minilm  sentence-transformers/all-MiniLM-L6-v2   (384d, fast baseline)
  bge     BAAI/bge-base-en-v1.5                     (768d, strong)
  e5      intfloat/e5-base-v2                        (768d, 'passage: ' prefix)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import config as C, data as D

ENCODERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-base-en-v1.5", ""),
    "e5": ("intfloat/e5-base-v2", "passage: "),
}


def main(names=None):
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    df = D.load(eligible_only=True)
    vids = df["video_id"].astype(str).values
    txt = D.text_field(df).tolist()
    print(f"encoding {len(txt):,} texts on {dev}")
    for name in (names or ENCODERS):
        out = C.PROC / f"emb_{name}.parquet"
        if out.exists():
            print(f"  {name}: exists, skip"); continue
        model_id, prefix = ENCODERS[name]
        t = time.time()
        m = SentenceTransformer(model_id, device=dev)
        m.max_seq_length = min(getattr(m, "max_seq_length", 256), 256)  # cap MPS memory
        bs = 64 if name != "minilm" else 256
        inp = [prefix + s for s in txt] if prefix else txt
        emb = m.encode(inp, batch_size=bs, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
        cols = {f"{name}{i}": emb[:, i].astype(np.float16) for i in range(emb.shape[1])}
        pd.DataFrame({"video_id": vids, **cols}).to_parquet(out, index=False)
        print(f"  {name}: {emb.shape} -> {out}  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
