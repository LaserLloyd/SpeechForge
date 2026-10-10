"""The HTTP contract (docs/API.md), with fake engines and a fake StudioForge."""

from __future__ import annotations

import asyncio
import hashlib
import threading

import httpx
import numpy as np
import pytest

from speechforge.app import create_app
from speechforge.audio import wav_bytes
from speechforge.lease import LeaseClient
from speechforge.pipeline import Speaker
from tests.conftest import FakeStt, FakeTts

TOKEN = "client-token-for-tests"


@pytest.fixture
def env_token(monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_TOKEN", TOKEN)


async def build(config, studioforge, *, tts=None, stt=None):
    tts = tts or FakeTts()
    stt = stt or FakeStt()
    lc = LeaseClient(
        "http://sf.test",
        holder="speechforge",
        priority=2,
        idle_ttl_s=900,
        vacate_url="http://127.0.0.1:1236/internal/vacate",
        vacate_token="vt",
        transport=httpx.MockTransport(studioforge.handler),
    )
    app = create_app(
        config,
        speaker_factory=lambda d: Speaker(tts, stt),
        lease_client=lc,
        selftest=None,
        residency_kw=FAKE_GPU,
    )
    return app, tts, stt


FAKE_GPU = {
    "free_check": lambda d, gib: (True, "free"),
    "facts": lambda d: {"name": "Fake 3090", "sm": "sm_86"},
    "live": lambda d: {"gpu_util": 3, "vram_free_mib": 18000, "vram_used_mib": 6000},
}


class Running:
    """The app with its lifespan, behind an httpx client (peer = 127.0.0.1)."""

    def __init__(self, app, *, peer="127.0.0.1", token: str | None = TOKEN):
        self.app = app
        self.peer = peer
        self.token = token

    async def __aenter__(self):
        self._ctx = self.app.router.lifespan_context(self.app)
        await self._ctx.__aenter__()
        transport = httpx.ASGITransport(app=self.app, client=(self.peer, 5000))
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.client = httpx.AsyncClient(transport=transport, base_url="http://sf", headers=headers)
        self.client.sf_app = self.app
        return self.client

    async def __aexit__(self, *exc):
        await self.client.aclose()
        await self._ctx.__aexit__(None, None, None)


async def ready(client):
    assert await client.sf_app.state.residency.wait_ready(5), "never ready"


async def test_health_is_open_and_says_nothing_private(config, studioforge):
    app, *_ = await build(config, studioforge)
    async with Running(app, peer="100.1.2.3", token=None) as c:
        r = await c.get("/health")
        assert r.status_code == 200 and r.json() == {
            "ok": True,
            "version": r.json()["version"],
            "api": 1,
        }


async def test_token_required_and_checked(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    async with Running(app, peer="100.1.2.3", token=None) as c:
        assert (await c.get("/ready")).status_code == 401
        r = await c.get("/ready", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_token"
        r = await c.get("/ready", headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 503 and r.json()["ready"] is False  # nothing loaded yet
        assert r.headers["retry-after"]


async def test_no_token_means_loopback_only(config, studioforge, monkeypatch):
    monkeypatch.delenv("SPEECHFORGE_TOKEN", raising=False)
    app, *_ = await build(config, studioforge)
    async with Running(app, peer="127.0.0.1", token=None) as c:
        assert (await c.get("/ready")).status_code == 503
    async with Running(app, peer="192.168.50.9", token=None) as c:
        assert (await c.get("/ready")).status_code == 401


async def test_ready_never_carries_secrets(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        await ready(c)
        r = await c.get("/ready")
        assert r.status_code == 200 and r.json()["ready"] is True
        text = r.text
        assert TOKEN not in text and "vacate_token" not in text and "vacate_url" not in text
        assert r.json()["gpu"]["sm"] == "sm_86"


async def test_metrics_loopback_and_token_only_and_no_content(
    config, studioforge, env_token, voice_bytes
):
    app, *_ = await build(config, studioforge)
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        await c.post("/v1/audio/speech", json={"input": "Secret words here.", "voice": vid})
        r = await c.get("/metrics")
        assert r.status_code == 200 and "speechforge_" in r.text
        assert "Secret" not in r.text and vid not in r.text and TOKEN not in r.text
    async with Running(app, peer="100.1.2.3") as c:
        assert (await c.get("/metrics")).status_code == 403
    async with Running(app, token="wrong") as c:
        assert (await c.get("/metrics")).status_code == 401


async def test_speech_streams_pcm_with_headers(config, studioforge, env_token, voice_bytes):
    app, tts, _ = await build(config, studioforge, tts=FakeTts(chunks=4))
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        assert (await c.put(f"/v1/voices/{vid}", content=voice_bytes)).status_code == 201
        assert (await c.put(f"/v1/voices/{vid}", content=voice_bytes)).status_code == 200
        r = await c.post(
            "/v1/audio/speech", json={"input": "Hello there.", "voice": f"sha256:{vid}"}
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/pcm"
        assert r.headers["x-sample-rate"] == "24000"
        pcm = np.frombuffer(r.content, "<i2")
        assert len(pcm) == 4 * 960 and pcm[0] == 1 and pcm[-1] == 4


async def test_speech_wav_and_render(config, studioforge, env_token, voice_bytes):
    app, *_ = await build(config, studioforge)
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        r = await c.post(
            "/v1/audio/speech", json={"input": "Hi.", "voice": vid, "response_format": "wav"}
        )
        assert r.content[:4] == b"RIFF"
        r = await c.post(
            "/v1/audio/speech", json={"input": "Hi.", "voice": vid, "render": True, "seed": 3}
        )
        assert len(r.content) == 4800


async def test_unknown_voice_is_404_before_anything(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "0" * 64})
        assert r.status_code == 404 and r.json()["error"]["code"] == "voice_not_found"
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": "not-a-hash"})
        assert r.status_code == 400


async def test_not_ready_is_an_immediate_503_with_retry_after(
    config, studioforge, env_token, voice_bytes
):
    app, *_ = await build(config, studioforge)
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": vid})
        assert r.status_code == 503 and r.headers.get("retry-after")
        assert r.json()["error"]["code"] == "not_ready"


async def test_gpu_refused_is_503_with_studioforge_wait(config, studioforge, env_token):
    studioforge.refuse = (409, "lease_conflict", 90)
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        pcm = np.zeros(1600, "<i2").tobytes()
        r = await c.post("/v1/audio/transcriptions/raw", content=pcm)
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_ready"
        await asyncio.sleep(0.05)
        r = await c.post("/v1/audio/transcriptions/raw", content=pcm)
        assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_unavailable"
        assert r.headers["retry-after"] == "90"


async def test_error_before_audio_is_json_not_a_broken_stream(
    config, studioforge, env_token, voice_bytes
):
    app, *_ = await build(config, studioforge, tts=FakeTts(fail_at=0))
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": vid})
        assert r.status_code == 500 and r.json()["error"]["code"] == "synthesis_failed"


async def test_error_mid_stream_breaks_the_stream(config, studioforge, env_token, voice_bytes):
    app, *_ = await build(config, studioforge, tts=FakeTts(chunks=3, fail_at=2))
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        with pytest.raises(Exception):  # noqa: B017 - any transport error: the client must see one
            await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": vid})


async def test_busy_when_all_streams_taken(config, studioforge, env_token, voice_bytes):
    config.tts.max_http_streams = 1
    gate = threading.Event()

    class Gated(FakeTts):
        def stream(self, *a, **kw):
            yield np.full(960, 1, dtype="<i2")
            gate.wait(5)
            yield from super().stream(*a, **kw)

    app, *_ = await build(config, studioforge, tts=Gated(chunks=2))
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        await ready(c)
        await c.put(f"/v1/voices/{vid}", content=voice_bytes)
        first = asyncio.create_task(
            c.post("/v1/audio/speech", json={"input": "Long.", "voice": vid})
        )
        await asyncio.sleep(0.1)
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": vid})
        assert r.status_code == 503 and r.json()["error"]["code"] == "busy"
        assert r.headers["retry-after"] == "1"
        gate.set()
        assert (await first).status_code == 200
        r = await c.post("/v1/audio/speech", json={"input": "Hi.", "voice": vid})
        assert r.status_code == 200  # the slot came back when the stream ended


async def test_raw_transcription(config, studioforge, env_token):
    app, _, stt = await build(config, studioforge)
    pcm = (np.zeros(16000, "<i2")).tobytes()
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/audio/transcriptions/raw", content=pcm)
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "heard 16000 samples" and body["audio_ms"] == 1000
        r = await c.post("/v1/audio/transcriptions/raw?sample_rate=8000", content=pcm)
        assert r.status_code == 400
        r = await c.post("/v1/audio/transcriptions/raw", content=b"\x00")
        assert r.status_code == 400


async def test_raw_transcription_length_cap(config, studioforge, env_token):
    config.stt.max_audio_s = 1.0
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/audio/transcriptions/raw", content=b"\x00\x00" * 40000)
        assert r.status_code == 413


async def test_openai_multipart_transcription(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    wav = wav_bytes(np.zeros(48000, "<i2"), 48000)  # 1 s at 48 kHz -> resampled to 16 kHz
    async with Running(app) as c:
        await ready(c)
        r = await c.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav, "audio/wav")},
            data={"model": "parakeet-tdt-0.6b-v2"},
        )
        assert r.status_code == 200 and r.json() == {"text": "heard 16000 samples"}
        r = await c.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", wav, "audio/wav")},
            data={"response_format": "text"},
        )
        assert r.text == "heard 16000 samples"
        r = await c.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"junk", "x")})
        assert r.status_code == 400


async def test_voice_routes(config, studioforge, env_token, voice_bytes):
    app, *_ = await build(config, studioforge)
    vid = hashlib.sha256(voice_bytes).hexdigest()
    async with Running(app) as c:
        assert (await c.head(f"/v1/voices/{vid}")).status_code == 404
        r = await c.put(f"/v1/voices/{'1' * 64}", content=voice_bytes)
        assert r.status_code == 400 and r.json()["error"]["code"] == "voice_hash_mismatch"
        assert (await c.put(f"/v1/voices/{vid}", content=voice_bytes)).status_code == 201
        assert (await c.head(f"/v1/voices/{vid}")).status_code == 200
        assert (await c.delete(f"/v1/voices/{vid}")).json()["deleted"] is True
        junk = b"PK\x03\x04 not really"
        r = await c.put(f"/v1/voices/{hashlib.sha256(junk).hexdigest()}", content=junk)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_voice"


async def test_voice_from_reference_is_off_by_default(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    clip = wav_bytes(np.zeros(24000 * 4, "<i2"), 24000)
    async with Running(app) as c:
        r = await c.post("/v1/voices/from-reference", content=clip)
        assert r.status_code == 403 and r.json()["error"]["code"] == "reference_clip_disabled"


async def test_voice_from_reference_when_allowed(config, studioforge, env_token):
    config.voices.allow_reference_clip = True
    app, *_ = await build(config, studioforge)
    clip = wav_bytes(np.zeros(24000 * 4, "<i2"), 24000)
    async with Running(app) as c:
        await ready(c)
        r = await c.post("/v1/voices/from-reference", content=clip)
        assert r.status_code == 201
        vid = r.json()["id"]
        assert (await c.head(f"/v1/voices/{vid.split(':')[1]}")).status_code == 200
        assert vid in (await c.get("/v1/voices")).json()["ids"]
        short = wav_bytes(np.zeros(24000, "<i2"), 24000)
        assert (await c.post("/v1/voices/from-reference", content=short)).status_code == 400


async def test_voice_put_size_cap(config, studioforge, env_token):
    config.voices.max_bytes = 70_000
    app, *_ = await build(config, studioforge)
    big = b"x" * 80_000
    async with Running(app) as c:
        r = await c.put(f"/v1/voices/{hashlib.sha256(big).hexdigest()}", content=big)
        assert r.status_code == 413


async def test_vacate_only_from_loopback(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    async with Running(app, peer="100.1.2.3") as c:
        r = await c.post(
            "/internal/vacate", json={"lease_id": "x"}, headers={"X-SF-Vacate-Token": "vt"}
        )
        assert r.status_code == 403


async def test_vacate_route(config, studioforge, env_token):
    app, tts, _ = await build(config, studioforge)
    async with Running(app, token=None) as c:
        c.headers["Authorization"] = f"Bearer {TOKEN}"
        await ready(c)
        lid = next(iter(studioforge.leases))
        r = await c.post(
            "/internal/vacate", json={"lease_id": lid}, headers={"X-SF-Vacate-Token": "wrong"}
        )
        assert r.status_code == 403
        r = await c.post(
            "/internal/vacate",
            json={"lease_id": lid},
            headers={"X-SF-Vacate-Token": "vt", "X-SF-Lease-Id": lid},
        )
        assert r.status_code == 202 and r.json()["held"] is True
        for _ in range(100):
            if not studioforge.leases:
                break
            await asyncio.sleep(0.02)
        assert studioforge.leases == {} and tts.closed
        r = await c.post("/v1/audio/transcriptions/raw", content=b"\x00\x00" * 1600)
        assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_vacated"


async def test_shutdown_releases_the_lease(config, studioforge, env_token):
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        await ready(c)
        assert len(studioforge.leases) == 1
    assert studioforge.leases == {}


async def test_boot_releases_stale_leases(config, studioforge, env_token):
    studioforge.leases = {"old": {"id": "old", "devices": [3], "holder": "speechforge"}}
    app, *_ = await build(config, studioforge)
    async with Running(app) as c:
        await c.get("/health")
        assert "old" not in studioforge.leases
