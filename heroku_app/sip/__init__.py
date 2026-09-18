"""ShouldIPost? — pre-publication TikTok success evaluator.

A modular, leakage-guarded pipeline on the lingbow/tiktok-video-engagement-200k
dataset. Clear separation of concerns:

    sip.config     constants + paths (single source of truth)
    sip.data       raw lingbow -> canonical analysis frame (labels, day-1, splits)
    sip.splits     temporal and leave-one-creator-out (LOCO) splits
    sip.features   modular, independently-toggleable feature blocks (fold-safe)
    sip.eval       metrics + bootstrap CIs + reliability + decision bands
    sip.modeling   model factory, calibration, split-conformal abstention
    sip.leakage    automated leakage guards (asserted in code + tests)
    sip.experiment one-call experiment runner + experiments_log.md appender

Everything is reproducible from `run_all.sh`. The heavy multimodal path lives in
`multimodal/` and writes a features table that `sip.features` consumes.
"""
__all__ = ["config", "data", "splits", "features", "eval", "modeling", "leakage", "experiment"]
