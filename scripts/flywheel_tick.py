#!/usr/bin/env python3
"""One flywheel tick: poll tracked videos, freeze day-1 signals, label matured ones, and retrain a
challenger when enough labels accumulated. Run manually or on a schedule (e.g. launchd/cron 2x/day):

    PYTHONPATH=src python scripts/flywheel_tick.py [--retrain]

Polling cadence, backoff and label maturity live in src/sip/flywheel.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import flywheel as FW  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="also retrain the challenger if ready")
    args = ap.parse_args()

    print("poll:", json.dumps(FW.poll()))
    st = FW.status()
    print("status:", json.dumps({k: st[k] for k in ("counts", "labels", "ready_to_retrain",
                                                    "bias_warning")}))
    if args.retrain and st["ready_to_retrain"]:
        print("retrain:", json.dumps(FW.retrain_challenger(), indent=1))
    elif args.retrain:
        print(f"retrain skipped: need >= {st['min_new']} matured labels")


if __name__ == "__main__":
    main()
