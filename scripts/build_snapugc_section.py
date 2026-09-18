"""Generate a standalone notebook section from the SnapUGC scale-up ledger.

Reads reports/snapugc_scaleup.json (source of truth, updated by run_transfer_snapugc.py)
and writes notebooks/snapugc_scaleup.ipynb — a self-contained section ready to review or to
paste into notebooks/RESULTS.ipynb. Numbers are baked into the markdown (readable without
running) AND recomputed by a code cell (reproducible). Re-run after the 10k ablation completes.

    PYTHONPATH=src python scripts/build_snapugc_section.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "snapugc_scaleup.json"
OUT = ROOT / "notebooks" / "snapugc_scaleup.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in src.strip("\n").split("\n")]}


def main():
    d = json.load(open(LEDGER))
    dl, ex, ab = d["download"], d["extraction"], d["ablation"]
    runs = ab["runs"]

    # ---- narrative markdown (numbers baked in) ----
    head = [
        "## Scale-up study — SnapUGC watch-retention transfer at higher data volume",
        "",
        f"**Question.** Does giving the transfer head *more* data change the honest-negative "
        f"result? The first pass used only **{dl['baseline_videos']}** SnapUGC videos. Here we scale "
        f"to **{dl['videos']:,}** (**{dl['scale_x']}×**) and re-measure.",
        "",
        "**Data (legally, no block-evasion).** SnapUGC (ECCV 2024) ships as a CSV of per-video "
        f"Snapchat-CDN links + an **ECR** label; ~{int(dl['cdn_live_frac_approx']*100)}% of the 2024 "
        f"links still resolve. Pulled with a multithreaded downloader "
        f"(~35 videos/s, ~40 MB/s) → **{dl['videos']:,} videos, ~{dl['size_gb']} GB**.",
        "",
        "**Features.** " + ex["method"] + f" → {'+'.join(ex['encoders'])} = **{ex['dim']}-d** per video.",
        "",
        "> *Multithreading note:* download is network-I/O → threads gave ~24×. Feature extraction is "
        "single-GPU (MPS) → threads only parallelise video decode, so ~2–4×, not 24×.",
    ]

    # ---- results table (baked) ----
    tbl = ["| SnapUGC videos (head) | head Spearman (ECR) | breakout Δ (mm_best → +retention) | ER Δ |",
           "|---|---|---|---|"]
    for r in sorted(runs, key=lambda x: x["n_snapugc_head"]):
        b, e = r["breakout"], r["er"]
        tbl.append(f"| {r['n_snapugc_head']:,} | {r['head_spearman_ecr']} | "
                   f"{b['mm_best']} → {b['mm_best_retention']} (**{b['delta']:+}**) | "
                   f"{e['mm_best']} → {e['mm_best_retention']} (**{e['delta']:+}**) |")

    # ---- verdict (data-driven) ----
    scaled = [r for r in runs if r["n_snapugc_head"] >= 1000]
    if scaled:
        s = max(scaled, key=lambda x: x["n_snapugc_head"])
        moved = abs(s["breakout"]["delta"]) >= 0.01 or abs(s["er"]["delta"]) >= 0.01
        verdict = (f"**Result at {s['n_snapugc_head']:,} videos:** breakout Δ={s['breakout']['delta']:+}, "
                   f"ER Δ={s['er']['delta']:+}. " + (
                       "The extra data **moves the needle** — see the table."
                       if moved else
                       "**Still no lift.** 68× more transfer data does not rescue the signal — the "
                       "ceiling is information, not sample size. The honest negative holds, now on far "
                       "firmer ground (a well-trained head on thousands of videos, not 150)."))
    else:
        verdict = ("**10k ablation pending** (feature extraction running). The 150-video baseline shows "
                   "no lift; this section auto-updates once the scaled run writes its numbers to the ledger.")

    cells = [
        md(*head),
        md("### Results", "", *tbl, "", verdict),
        md("### Reproduce (loads the ledger `reports/snapugc_scaleup.json`)"),
        code(
            "import json, pandas as pd, matplotlib.pyplot as plt\n"
            "d = json.load(open('../reports/snapugc_scaleup.json'))\n"
            "runs = sorted(d['ablation']['runs'], key=lambda x: x['n_snapugc_head'])\n"
            "df = pd.DataFrame([{'n_snapugc': r['n_snapugc_head'],\n"
            "                    'head_spearman_ecr': r['head_spearman_ecr'],\n"
            "                    'breakout_delta': r['breakout']['delta'],\n"
            "                    'er_delta': r['er']['delta']} for r in runs])\n"
            "display(df)\n"
            "ax = df.set_index('n_snapugc')[['breakout_delta','er_delta']].plot.bar(rot=0)\n"
            "ax.axhline(0, color='k', lw=0.8); ax.set_ylabel('LOCO AUC lift from retention_head')\n"
            "ax.set_title('SnapUGC retention transfer: lift vs head-training size'); plt.tight_layout(); plt.show()"
        ),
        md("**Takeaway.** " + verdict.replace("**", "")),
    ]

    for i, c in enumerate(cells):
        c["id"] = f"snapugc-scaleup-{i}"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.parent.mkdir(exist_ok=True)
    json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}  ({len(cells)} cells, {len(runs)} run(s) in ledger)")


if __name__ == "__main__":
    main()
