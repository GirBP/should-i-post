"""Extractor robustness: the LLM-output parser must never break the pipeline, whatever a
weak model returns. Pure-logic tests — no ffmpeg / whisper / API calls.

Run:  PYTHONPATH=src pytest tests/test_extract.py -q
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sip import extract as EX  # noqa: E402


def _valid(d):
    return (d["topic"] in EX.TOPICS
            and isinstance(d["summary"], str)
            and all(0.0 <= d["emotions"][e] <= 1.0 for e in EX.EMOTIONS)
            and 0.0 <= d["hook"] <= 1.0 and 0.0 <= d["cta"] <= 1.0)


def test_parser_handles_weak_model_outputs():
    cases = [
        '{"summary":"pasta","topic":"Cooking_Food","emotions":{"joy":0.8},"hook":0.5,"cta":0.1}',
        '```json\n{"topic":"Dance_Music","summary":"a dance"}\n```',              # code fences
        'Sure! {"topic":"Sports_Fitness","summary":"gym"} hope it helps',         # prose around JSON
        '{"summary":"only a summary"}',                                           # missing fields
        '{"topic":"NotARealTopic","emotions":{"joy":"high","anger":2}}',          # bad values
        '',                                                                       # empty
        'complete garbage, not json',                                             # garbage
    ]
    for raw in cases:
        assert _valid(EX._parse(raw, "fallback transcript")), raw


def test_bad_topic_falls_back_to_others():
    assert EX._parse('{"topic":"Politics"}', "t")["topic"] == "Others"


def test_emotions_clamped_to_unit_interval():
    d = EX._parse('{"emotions":{"anger":5,"joy":-1,"fear":"x"}}', "t")
    assert d["emotions"]["anger"] == 1.0 and d["emotions"]["joy"] == 0.0 and d["emotions"]["fear"] == 0.0


def test_empty_llm_output_marks_llm_unused():
    d = EX._parse("", "some transcript text")
    assert d["llm_used"] is False and d["summary"] == "some transcript text"


def test_backend_selection_from_env(monkeypatch):
    monkeypatch.setenv("SIP_LLM", "none")
    assert EX._backend() == "none"
    monkeypatch.setenv("SIP_LLM", "deepseek")
    assert EX._backend() == "deepseek"
    monkeypatch.setenv("SIP_LLM", "ollama")
    assert EX._backend() == "ollama"
    monkeypatch.delenv("SIP_LLM", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(EX, "_ollama_up", lambda: False)   # no local server in this scenario
    assert EX._backend() == "none"                       # auto with nothing available -> none
    monkeypatch.setattr(EX, "_ollama_up", lambda: True)
    assert EX._backend() == "ollama"                     # auto prefers the local model


def test_call_llm_none_backend_returns_empty(monkeypatch):
    monkeypatch.setenv("SIP_LLM", "none")
    assert EX._call_llm("anything") == ""                # no network, no raise
