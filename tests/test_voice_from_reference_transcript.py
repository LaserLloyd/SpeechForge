"""POST /v1/voices/from-reference for an engine that needs the clip's transcript (S37)."""

from __future__ import annotations

import numpy as np
import pytest

from speechforge.audio import float_to_pcm16, wav_bytes
from tests.conftest import FakeTts, make_voice
from tests.test_api import Running, build, env_token, ready  # noqa: F401 - fixture


class WordsTts(FakeTts):
    """Clones from words: records what build_voice was given."""

    needs_transcript = True

    def __init__(self):
        super().__init__()
        self.built: list[tuple[int, int, str | None]] = []

    def build_voice(self, reference, rate, transcript=None):
        self.built.append((len(reference), rate, transcript))
        return make_voice().to_npz()


def _clip() -> bytes:
    t = np.arange(24000 * 5) / 24000
    speech = (0.3 * np.sin(2 * np.pi * 200 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)))
    return wav_bytes(float_to_pcm16(speech.astype(np.float32)), 24000)


@pytest.mark.usefixtures("env_token")
async def test_the_rig_transcribes_the_clip_when_the_caller_sends_no_words(config, studioforge):
    config.voices.allow_reference_clip = True
    tts = WordsTts()
    app, _, stt = await build(config, studioforge, tts=tts)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/from-reference", content=_clip())
        assert r.status_code == 201, r.text
        doc = r.json()
        assert doc["transcript_source"] == "stt" and doc["transcript"].startswith("heard ")
        assert doc["transcribe_ms"] is not None and doc["id"].startswith("sha256:")
        n, rate, words = tts.built[-1]
        assert rate == 24000 and words == doc["transcript"]
        # The STT got the clip at ITS rate (16 kHz), not at 24 kHz.
        assert stt.calls == 1 and doc["transcript"] == f"heard {round(n * 16000 / 24000)} samples"


@pytest.mark.usefixtures("env_token")
async def test_the_caller_can_send_the_transcript_in_a_header(config, studioforge):
    config.voices.allow_reference_clip = True
    tts = WordsTts()
    app, _, stt = await build(config, studioforge, tts=tts)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/from-reference", content=_clip(),
                         headers={"X-Reference-Transcript": "  The words I spoke.  "})
        assert r.status_code == 201, r.text
        doc = r.json()
        assert doc["transcript"] == "The words I spoke." and doc["transcript_source"] == "caller"
        assert "transcribe_ms" not in doc and stt.calls == 0
        assert tts.built[-1][2] == "The words I spoke."
        r = await c.post("/v1/voices/from-reference", content=_clip(),
                         headers={"X-Reference-Transcript": "x" * 2001})
        assert r.status_code == 400


@pytest.mark.usefixtures("env_token")
async def test_an_engine_without_the_need_is_untouched(config, studioforge):
    config.voices.allow_reference_clip = True
    tts = FakeTts()
    app, _, stt = await build(config, studioforge, tts=tts)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/from-reference", content=_clip(),
                         headers={"X-Reference-Transcript": "ignored"})
        assert r.status_code == 201
        assert "transcript" not in r.json() and stt.calls == 0
