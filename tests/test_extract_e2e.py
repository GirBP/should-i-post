"""End-to-end smoke of the video -> features -> recommendation pipeline.

Synthesises a tiny speech video (macOS `say` + ffmpeg), transcribes it (Whisper) and scores it
through the deployed Model A — proving F16 works at runtime, not just at the parser level.
Skips cleanly where the heavy tools are absent (CI without ffmpeg/whisper/say), so the suite stays green
everywhere.

Run:  PYTHONPATH=src pytest tests/test_extract_e2e.py -q
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _have(cmd):
    return shutil.which(cmd) is not None


@pytest.mark.skipif(not (_have("ffmpeg") and _have("say")),
                    reason="needs ffmpeg + macOS `say` to synthesise a speech video")
def test_video_to_recommendation_end_to_end(tmp_path):
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        pytest.skip("faster-whisper not installed")

    os.environ.setdefault("SIP_LLM", "none")        # transcript-only: no API key needed for the smoke
    os.environ.setdefault("SIP_DEVICE", "cpu")
    from sip import extract as EX, inference as INF

    aiff, mp4 = tmp_path / "s.aiff", tmp_path / "clip.mp4"
    subprocess.run(["say", "-o", str(aiff),
                    "Hey everyone, a quick five minute morning routine to feel productive. Follow for more."],
                   check=True)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=240x240:d=6",
                    "-i", str(aiff), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(mp4)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    inp = EX.extract(str(mp4))
    ex = inp["extracted"]
    assert len(ex["transcript"]) > 10, "Whisper should return a transcript from the speech"
    assert inp["extra_text"], "transcript must populate extra_text (the Model-A content lever)"
    assert inp["duration_s"] and inp["duration_s"] > 0

    res = INF.predict(inp)
    assert res["recommendation"] in ("Post", "Unsure", "Do not post")
    assert 0.0 <= res["probability"] <= 1.0
    assert "what_model_cannot_know" in res and res["factors"] is not None
