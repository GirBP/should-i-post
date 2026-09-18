"""One-call experiment runner: feature spec + model + target + split -> metrics.

Runs both protocols (LOCO out-of-fold and temporal train->test), enforces the
leakage guard for the model kind, computes metrics with bootstrap CIs, and
appends a row to reports/experiments_log.md.
"""
from __future__ import annotations
import json
import time
import numpy as np
from . import config as C, features as F, splits as S, eval as E, leakage as L
from .modeling import make_model


def _fit_predict(model_name, Xtr, ytr, Xte):
    m = make_model(model_name)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run(df, blocks, model_name="hgb", target="y_breakout_wc", model_kind="A",
        do_loco=True, do_temporal=True, n_boot=1000, label=None, log=True, verbose=True):
    """Return {"label","blocks","model","target","loco":{...},"temporal":{...}}."""
    t0 = time.time()
    y = df[target].astype(int).values
    label = label or "+".join(blocks)

    # leakage guard on the (split-independent) feature names
    _, names = F.build_blocks(df, blocks, np.arange(len(df)), y, target_col=target)
    (L.assert_model_a if model_kind == "A" else L.assert_model_b)(names)

    res = {"label": label, "blocks": blocks, "model": model_name, "target": target,
           "model_kind": model_kind, "n_features": len(names)}

    if do_loco:
        oof = np.full(len(df), np.nan)
        for tri, tei in S.loco_folds(df):
            L.assert_disjoint_authors(df["author_id"].values[tri], df["author_id"].values[tei])
            if len(np.unique(y[tri])) < 2:
                continue
            # fit fold-safe encoders on this fold's train, transform all rows
            Xall, _ = F.build_blocks(df, blocks, tri, y, target_col=target)
            oof[tei] = _fit_predict(model_name, Xall[tri], y[tri], Xall[tei])
        ok = ~np.isnan(oof)
        res["loco"] = E.metrics(y[ok], oof[ok], n_boot=n_boot)
        res["_oof"] = oof

    if do_temporal:
        tr, va, te = S.temporal_masks(df)
        tri = np.where(tr)[0]
        Xall, _ = F.build_blocks(df, blocks, tri, y, target_col=target)
        if len(np.unique(y[tri])) >= 2:
            p = _fit_predict(model_name, Xall[tri], y[tri], Xall[te])
            res["temporal"] = E.metrics(y[te], p, n_boot=n_boot)
            res["_test_p"] = p
            res["_test_mask"] = te

    res["seconds"] = round(time.time() - t0, 1)
    if verbose:
        lc = res.get("loco", {}); tc = res.get("temporal", {})
        print(f"  {label:42s} LOCO={lc.get('roc_auc','-')} "
              f"temporal={tc.get('roc_auc','-')}  ({res['seconds']}s, {len(names)}f)")
    if log:
        append_log(res)
    return res


def append_log(res):
    """Append one row to the markdown experiment log (created with a header)."""
    C.EXPLOG.parent.mkdir(exist_ok=True)
    if not C.EXPLOG.exists():
        C.EXPLOG.write_text(
            "# Experiments log\n\nOne row per run. AUC with 95% bootstrap CI. "
            "LOCO = leave-one-creator-out (GroupKFold by author); temporal = train<valid<test by date.\n\n"
            "| label | model | target | kind | nfeat | LOCO AUC [CI] | temporal AUC [CI] | temporal Brier | s |\n"
            "|---|---|---|---|---|---|---|---|---|\n")

    def fmt(m):
        if not m:
            return "—"
        ci = m.get("roc_auc_ci", [None, None])
        return f"{m.get('roc_auc')} [{ci[0]}, {ci[1]}]"
    lc, tc = res.get("loco"), res.get("temporal")
    row = (f"| {res['label']} | {res['model']} | {res['target']} | {res['model_kind']} | "
           f"{res['n_features']} | {fmt(lc)} | {fmt(tc)} | "
           f"{tc.get('brier','—') if tc else '—'} | {res.get('seconds','')} |\n")
    with open(C.EXPLOG, "a") as f:
        f.write(row)


def save_json(results, path):
    clean = []
    for r in results:
        clean.append({k: v for k, v in r.items() if not k.startswith("_")})
    path = C.REPORTS / path if not str(path).startswith("/") else path
    json.dump(clean, open(path, "w"), ensure_ascii=False, indent=2)
    return path
