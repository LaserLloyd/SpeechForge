"""POST /v1/voices/surprise (S34): keep an improvised voice from an engine's take()."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from speechforge.audio import read_wav
from tests.conftest import FakeTts, make_voice
from tests.test_api import Running, build, env_token, ready  # noqa: F401 - fixture


class Improviser(FakeTts):
    name = "drama3-tts"

    def __init__(self):
        super().__init__()
        self.takes: list[tuple[str, int | None]] = []

    def take(self, text, seed=None):
        self.takes.append((text, seed))
        pcm = np.full(2400, 1000, dtype="<i2")
        return make_voice().to_npz(), pcm


@pytest.mark.usefixtures("env_token")
async def test_surprise_stores_the_take_as_a_voice(config, studioforge):
    config.voices.allow_reference_clip = True
    tts = Improviser()
    app, *_ = await build(config, studioforge, tts=tts)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/surprise", json={"text": "  Hello there. ", "seed": 4})
        assert r.status_code == 201, r.text
        doc = r.json()
        assert doc["text"] == "Hello there." and doc["id"].startswith("sha256:")
        assert tts.takes == [("Hello there.", 4)]
        audio, rate = read_wav(base64.b64decode(doc["preview_wav_base64"]))
        assert rate == tts.sample_rate and len(audio) == 2400
        assert doc["audio_ms"] == int(2400 * 1000 / tts.sample_rate)
        vid = doc["id"].split(":", 1)[1]
        assert (await c.head(f"/v1/voices/{vid}")).status_code == 200
        r = await c.post("/v1/voices/surprise", json={"text": ""})
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
        r = await c.post("/v1/voices/surprise", json={"text": "x", "seed": "no"})
        assert r.status_code == 400
        r = await c.post("/v1/voices/surprise", content=b"not json",
                         headers={"content-type": "application/json"})
        assert r.status_code == 400


@pytest.mark.usefixtures("env_token")
async def test_surprise_is_refused_without_the_opt_in_or_the_engine(config, studioforge):
    config.voices.allow_reference_clip = False
    app, *_ = await build(config, studioforge, tts=Improviser())
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/surprise", json={"text": "Hi."})
        assert r.status_code == 403 and r.json()["error"]["code"] == "reference_clip_disabled"
    config.voices.allow_reference_clip = True
    app, *_ = await build(config, studioforge, tts=FakeTts())   # no take()
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/surprise", json={"text": "Hi."})
        assert r.status_code == 501 and r.json()["error"]["code"] == "unsupported"


@pytest.mark.parametrize("bad", [[], "text", 7])
@pytest.mark.usefixtures("env_token")
async def test_surprise_body_must_be_an_object(config, studioforge, bad):
    config.voices.allow_reference_clip = True
    app, *_ = await build(config, studioforge, tts=Improviser())
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/surprise", json=bad)
        assert r.status_code == 400
