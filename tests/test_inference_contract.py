#!/usr/bin/env python3
"""
Adversarial test harness for SIP inference API contract.
Tests caption + metadata edge cases, validates return contract.
"""

import sys
import traceback
import math
from datetime import datetime, timedelta

# Setup environment
import os
os.environ["PYTHONPATH"] = "src"
os.environ["SIP_DEVICE"] = "cpu"

from sip import inference as I


def check_contract(result, case_desc):
    """Verify result dict matches expected contract."""
    errors = []

    # Required keys
    required = {"recommendation", "probability", "factors", "examples", "rationale", "warnings"}
    if not isinstance(result, dict):
        return f"Result is not dict: {type(result)}"

    missing = required - set(result.keys())
    if missing:
        return f"Missing keys: {missing}"

    # Check types and ranges
    if result["recommendation"] not in {"Post", "Do not post", "Unsure"}:
        return f"Invalid recommendation: {result['recommendation']}"

    prob = result["probability"]
    if not isinstance(prob, (int, float)):
        return f"probability is {type(prob)}, not number"
    if math.isnan(prob):
        return f"probability is NaN"
    if not (0 <= prob <= 1):
        return f"probability {prob} outside [0,1]"

    if not isinstance(result["factors"], list):
        return f"factors is {type(result['factors'])}, not list"

    if not isinstance(result["examples"], list):
        return f"examples is {type(result['examples'])}, not list"

    if not isinstance(result["rationale"], str):
        return f"rationale is {type(result['rationale'])}, not str"

    if not isinstance(result["warnings"], list):
        return f"warnings is {type(result['warnings'])}, not list"

    return None


def run_test(inp_dict, case_desc):
    """Run single test case, return (status, exception_info_if_any)."""
    try:
        result = I.predict(inp_dict, day_one=None, cost_ratio=None)
        contract_error = check_contract(result, case_desc)
        if contract_error:
            return "CONTRACT_FAIL", contract_error
        return "OK", None
    except Exception as e:
        return "EXCEPTION", (type(e).__name__, str(e), traceback.format_exc())


def main():
    test_cases = []

    # === Caption edge cases ===
    test_cases.append(({}, "empty dict (all keys missing)"))
    test_cases.append(({"caption": ""}, "empty caption"))
    test_cases.append(({"caption": "   \t\n  "}, "whitespace-only caption"))
    test_cases.append(({"caption": None}, "None caption"))
    test_cases.append(({"caption": "x" * 50000}, "50k char caption"))
    test_cases.append(({"caption": "😱🔥💀"}, "emoji-only caption"))
    test_cases.append(({"caption": "#a #b #c #d #e"}, "hashtag-only caption"))
    test_cases.append(({"caption": "https://example.com/very/long/url?param=value&other=stuff"}, "URL-only caption"))
    test_cases.append(({"caption": "Hello\nWorld\n\nMultiline"}, "multiline with newlines"))
    test_cases.append(({"caption": "Cyrillic: Привет мир"}, "Cyrillic caption"))
    test_cases.append(({"caption": "Chinese: 你好世界"}, "Chinese caption"))
    test_cases.append(({"caption": "Arabic: مرحبا بالعالم"}, "Arabic RTL caption"))
    test_cases.append(({"caption": "Control\x00char\x01null"}, "caption with null/control chars"))
    test_cases.append(({"caption": 12345}, "caption as int"))
    test_cases.append(({"caption": ["list", "not", "string"]}, "caption as list"))

    # === Duration edge cases ===
    test_cases.append(({"duration_s": 0}, "duration_s = 0"))
    test_cases.append(({"duration_s": -5}, "duration_s = negative"))
    test_cases.append(({"duration_s": 1e9}, "duration_s = 1e9 (huge)"))
    test_cases.append(({"duration_s": float('nan')}, "duration_s = NaN"))
    test_cases.append(({"duration_s": float('inf')}, "duration_s = infinity"))
    test_cases.append(({"duration_s": None}, "duration_s = None"))
    test_cases.append(({"duration_s": "abc"}, "duration_s = string 'abc'"))

    # === Aspect ratio edge cases ===
    test_cases.append(({"aspect": None}, "aspect = None"))
    test_cases.append(({"aspect": 0}, "aspect = 0"))
    test_cases.append(({"aspect": -1.5}, "aspect = negative"))
    test_cases.append(({"aspect": 1e10}, "aspect = huge"))
    test_cases.append(({"aspect": "9:16"}, "aspect as string"))

    # === post_time edge cases ===
    test_cases.append(({"post_time": "not-a-date"}, "post_time = invalid string"))
    test_cases.append(({"post_time": "2099-12-31T23:59:59Z"}, "post_time = far future"))
    future = (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z"
    test_cases.append(({"post_time": future}, "post_time = 1 year in future"))
    test_cases.append(({"post_time": None}, "post_time = None"))
    test_cases.append(({"post_time": 123456789}, "post_time = unix timestamp (int)"))

    # === extra_text edge cases ===
    test_cases.append(({"extra_text": ""}, "extra_text = empty string"))
    test_cases.append(({"extra_text": "x" * 100000}, "extra_text = 100k chars"))
    test_cases.append(({"extra_text": None}, "extra_text = None"))

    # === Combined adversarial cases ===
    test_cases.append((
        {
            "caption": "Bad\x00caption",
            "duration_s": -999,
            "aspect": 0,
            "post_time": "garbage",
            "extra_text": None
        },
        "combined: all invalid"
    ))
    test_cases.append((
        {
            "caption": "😱" * 1000,
            "duration_s": float('nan'),
            "aspect": None,
            "post_time": None
        },
        "combined: emoji + NaN + None values"
    ))

    # Run all tests
    results = []
    for inp, desc in test_cases:
        status, exc_info = run_test(inp, desc)
        results.append((status, desc, exc_info))
        status_str = "OK  " if status == "OK" else status
        print(f"{status_str} | {desc}")
        if exc_info:
            if isinstance(exc_info, tuple) and len(exc_info) == 3:
                exc_type, exc_msg, _ = exc_info
                print(f"       -> {exc_type}: {exc_msg}")
            else:
                print(f"       -> {exc_info}")

    # === Summary ===
    print("\n" + "="*70)
    total = len(results)
    ok_count = sum(1 for s, _, _ in results if s == "OK")
    exception_count = sum(1 for s, _, _ in results if s == "EXCEPTION")
    contract_fail_count = sum(1 for s, _, _ in results if s == "CONTRACT_FAIL")

    print(f"SUMMARY: {total} cases total")
    print(f"  OK: {ok_count}")
    print(f"  EXCEPTION (uncaught): {exception_count}")
    print(f"  CONTRACT_FAIL (bad return): {contract_fail_count}")
    print()

    # Print failures
    failures = [r for r in results if r[0] != "OK"]
    if failures:
        print("FAILURES:")
        for status, desc, exc_info in failures:
            print(f"\n  {status}: {desc}")
            if exc_info:
                if isinstance(exc_info, tuple) and len(exc_info) == 3:
                    exc_type, exc_msg, tb = exc_info
                    print(f"    Exception: {exc_type}: {exc_msg}")
                    if False:  # Set True to see full traceback
                        print(f"    Traceback:\n{tb}")
                else:
                    print(f"    Error: {exc_info}")

    print("\n" + "="*70)
    verdict = "PASS" if exception_count == 0 and contract_fail_count == 0 else "FAIL"
    print(f"VERDICT: {verdict}")
    if exception_count > 0 or contract_fail_count > 0:
        print(f"  Inference API does NOT degrade gracefully.")
        print(f"  {exception_count} uncaught exceptions, {contract_fail_count} contract violations.")
    else:
        print(f"  Inference API degrades gracefully on all {total} adversarial inputs.")


if __name__ == "__main__":
    main()
