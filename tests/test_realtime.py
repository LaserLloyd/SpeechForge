"""sf.1 over a real WebSocket (Starlette TestClient), fake engines, fake StudioForge."""

from __future__ import annotations

import base64
import hashlib
import json
import time

import httpx
import numpy as np
import pytest
from starlette.testclient import TestClient, WebSocketDenialResponse

from speechforge.app import create_app
from speechforge.engines.turn import EnergyVad
from speechforge.lease import LeaseClient
from speechforge.pipeline import Speaker
from speechforge.protocol import KIND_MIC, KIND_TTS, pack_audio, unpack_audio
from tests.conftest import FakeEou, FakeStt, FakeTts

TOKEN = "realtime-test-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
FAKE_GPU = {
    "free_check": lambda d, gib: (True, "free"),
    "facts": lambda d: {"name": "Fake 3090", "sm": "sm_86"},
    "live": lambda d: {"gpu_util": 5, "vram_free_mib": 18000, "vram_used_mib": 6000},
}


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_TOKEN", TOKEN)


def build(config, studioforge, *, tts=None, stt=None, eou=None):
    tts = tts or FakeTts(chunks=5)
    stt = stt or FakeStt()
    eou = eou or FakeEou()
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
        lease_client=lc,
        selftest=None,
        residency_kw=FAKE_GPU,
        speaker_factory=lambda d: Speaker(tts, stt, eou=eou, vad=EnergyVad()),
    )
    return app, tts, stt, eou


def connect(tc, headers=AUTH, protocols=("sf.1",)):
    return tc.websocket_connect("/v1/realtime", headers=dict(headers), subprotocols=list(protocols))


def hello(ws, **extra):
    ws.send_text(
        json.dumps(
            {"type": "hello", "proto": "sf.1", "client": {"name": "test", "version": "1"}, **extra}
        )
    )
    while True:
        msg = ws.receive_json()
        if msg["type"] == "ready" or (msg["type"] == "error" and msg.get("fatal")):
            return msg


def until(ws, kind, limit=500):
    """Receive until a control frame of ``kind``; returns (frame, audio frames, others)."""
    audio, others = [], []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio.append(unpack_audio(msg["bytes"]))
            continue
        data = json.loads(msg["text"])
        if data["type"] == kind:
            return data, audio, others
        others.append(data)
    raise AssertionError(f"no {kind}")


def put_voice(ws, data: bytes) -> str:
    vid = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode()
    step = 40_000
    parts = [b64[i : i + step] for i in range(0, len(b64), step)]
    for i, p in enumerate(parts):
        ws.send_text(
            json.dumps(
                {"type": "voice.put", "id": vid, "part": i, "parts": len(parts), "data_b64": p}
            )
        )
    stored, _, _ = until(ws, "voice.stored")
    assert stored["id"] == f"sha256:{vid}"
    return vid


def speech(seconds: float, start_pts: int = 0) -> list[bytes]:
    n = int(seconds * 16000)
    t = np.arange(n) / 16000
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    return [
        pack_audio(KIND_MIC, 0, 0, i // 320, start_pts + i, pcm[i : i + 320])
        for i in range(0, n, 320)
    ]


def silence(seconds: float, start_pts: int) -> list[bytes]:
    n = int(seconds * 16000)
    z = np.zeros(n, "<i2")
    return [
        pack_audio(KIND_MIC, 0, 0, i // 320, start_pts + i, z[i : i + 320])
        for i in range(0, n, 320)
    ]


# --------------------------------------------------------------------- upgrade
@pytest.mark.parametrize(
    ("headers", "protos", "status"),
    [
        ({**AUTH, "Origin": "https://evil.example"}, ("sf.1",), 403),
        ({}, ("sf.1",), 401),
        ({"Authorization": "Bearer wrong"}, ("sf.1",), 401),
        (AUTH, ("sf.2",), 400),
    ],
)
def test_upgrade_refusals(config, studioforge, headers, protos, status):
    app, *_ = build(config, studioforge)
    with (
        TestClient(app) as tc,
        pytest.raises(WebSocketDenialResponse) as exc,
        connect(tc, headers, protos),
    ):
        pass
    assert exc.value.status_code == status


def test_capacity_refusal_has_retry_after(config, studioforge):
    config.realtime.max_sessions = 1
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        assert hello(ws)["type"] == "ready"
        with pytest.raises(WebSocketDenialResponse) as exc, connect(tc):
            pass
        assert exc.value.status_code == 503 and exc.value.headers["retry-after"]


def test_upgrade_rate_limit(config, studioforge):
    config.server.upgrades_per_ip_per_min = 2
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc:
        for _ in range(2):
            with pytest.raises(WebSocketDenialResponse), connect(tc, {}):
                pass
        with pytest.raises(WebSocketDenialResponse) as exc, connect(tc):
            pass
        assert exc.value.status_code == 429


def test_first_frame_must_be_hello(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        ws.send_text(json.dumps({"type": "ping", "t": 1}))
        err = ws.receive_json()
        assert err["type"] == "error" and err["fatal"] and err["code"] == "bad_frame"


def test_ready_frame(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        r = hello(ws)
        assert r["type"] == "ready" and r["proto"] == "sf.1"
        assert r["gpu"] == {"index": 3, "name": "Fake 3090", "sm": "sm_86"}
        assert r["limits"]["max_text"] == 2000 and r["limits"]["sessions"] == 3
        assert r["lease"]["state"] == "held" and r["lease"]["id"]
        assert {"stt", "tts", "eou"} <= set(r["engines"])
        assert r["resume_token"] and r["session"]


def test_refused_lease_is_a_pre_upgrade_503_after_first_try(config, studioforge):
    studioforge.refuse = (409, "lease_conflict", 90)
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc:
        with connect(tc) as ws:
            err = hello(ws)
            assert err["type"] == "error" and err["code"] == "not_ready" and err["fatal"]
            assert err["retry_after_s"] == 90
        with pytest.raises(WebSocketDenialResponse) as exc, connect(tc):
            pass
        assert exc.value.status_code == 503 and exc.value.headers["retry-after"] == "90"


# --------------------------------------------------------------------- STT
def test_partials_eou_final_and_reset(config, studioforge):
    app, _, stt, eou = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        for f in speech(1.0):
            ws.send_bytes(f)
            time.sleep(0.016)  # about real time: the ingest cap is 1.25x
        vad, _, _ = until(ws, "stt.vad")
        assert vad["speech"] is True
        part, _, _ = until(ws, "stt.partial")
        assert part["tail"].startswith("heard") and part["upto"] > 0
        assert part["lag_ms"] >= 0 and "stable_words" in part and "punct_final" in part
        for f in silence(0.4, 16000):
            ws.send_bytes(f)
            time.sleep(0.016)
        e, _, _ = until(ws, "eou")
        assert e["p"] == pytest.approx(0.93) and e["pause_ms"] >= 150 and e["at"] <= 16000 + 512
        ws.send_text(json.dumps({"type": "stt.flush", "id": "f1"}))
        fin, _, _ = until(ws, "stt.final")
        assert fin["id"] == "f1" and fin["text"].startswith("final")
        assert fin["upto"] == 16000 + int(0.4 * 16000)
        assert fin["words"][0][0] == "final" and fin["words"][0][1] >= 0
        assert eou.calls == 2  # one warm-up call at load + one pause
        ws.send_text(json.dumps({"type": "stt.reset"}))
        ws.send_text(json.dumps({"type": "ping", "t": 1.5}))
        pong, _, _ = until(ws, "pong")
        assert pong["t"] == 1.5 and "srv_t" in pong and pong["gpu_util"] == 5
        assert pong["vram_free_mib"] == 18000 and "queue" in pong


def test_ingest_is_rate_limited(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        for f in speech(3.0):  # 3 s sent as fast as possible: > 1.25 x real time
            ws.send_bytes(f)
        ws.send_text(json.dumps({"type": "ping", "t": 1}))
        until(ws, "pong")
        mgr = app.state.sessions
        s = next(iter(mgr.sessions.values()))
        assert s.stt.dropped_samples > 0


def test_bad_binary_frame_is_an_error_not_a_close(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        ws.send_bytes(b"\x00" * 20)
        err, _, _ = until(ws, "error")
        assert err["code"] == "bad_frame" and not err["fatal"]
        ws.send_text(json.dumps({"type": "nonsense"}))
        err, _, _ = until(ws, "error")
        assert err["code"] == "bad_frame"
        ws.send_text(json.dumps({"type": "ping", "t": 2}))
        assert until(ws, "pong")[0]["t"] == 2


def test_error_flood_closes_1008(config, studioforge):
    config.realtime.error_flood = 5
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        for _ in range(5):
            ws.send_text("{}")
        closed = None
        for _ in range(20):
            msg = ws.receive()
            if msg["type"] == "websocket.close":
                closed = msg
                break
        assert closed is not None and closed["code"] == 1008


# --------------------------------------------------------------------- TTS
def open_ctx(ws, vid, ctx=1, gen=1, **kw):
    ws.send_text(
        json.dumps(
            {
                "type": "tts.open",
                "ctx": ctx,
                "gen": gen,
                "voice": f"sha256:{vid}",
                "turn_id": "t1",
                **kw,
            }
        )
    )


def test_tts_stream_frames_words_done(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=5))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid)
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "Hello there, "}))
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "friend.", "final": False}))
        ws.send_text(json.dumps({"type": "tts.close", "ctx": 1}))
        done, audio, others = until(ws, "tts.done")
        kinds = [o["type"] for o in others]
        assert "tts.started" in kinds and "tts.words" in kinds
        started = next(o for o in others if o["type"] == "tts.started")
        assert started["ttfa_ms"] >= 0 and started["turn_id"] == "t1"
        words = next(o for o in others if o["type"] == "tts.words")
        assert [w[0] for w in words["words"]] == ["Hello", "there", "friend"]
        assert words["words"][0][1] == 0 and words["words"][-1][2] <= 5 * 40
        assert tts.texts == ["Hello there, friend."]  # one unit: cut at the sentence end
        assert all(f.kind == KIND_TTS and f.ctx == 1 and f.gen == 1 for f in audio)
        assert [f.seq for f in audio] == list(range(len(audio)))
        assert [f.pts for f in audio] == [i * 960 for i in range(len(audio))]
        assert all(len(f.pcm) == 960 for f in audio)
        assert done["audio_ms"] == 5 * 40


def test_flush_renders_buffer_and_acknowledges(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=2))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid)
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "no punctuation yet"}))
        ws.send_text(json.dumps({"type": "tts.flush", "ctx": 1, "id": 7}))
        fl, audio, _ = until(ws, "tts.flushed")
        assert fl["id"] == 7 and fl["audio_ms"] == 80 and len(audio) == 2
        assert tts.texts == ["no punctuation yet"]


def test_unknown_voice_is_an_error(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        open_ctx(ws, "0" * 64)
        err, _, _ = until(ws, "error")
        assert err["code"] == "unknown_voice" and err["ctx"] == 1 and not err["fatal"]


def test_cancel_stops_quickly_and_is_idempotent(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=10_000, step_s=0.005))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid, gen=4)
        ws.send_text(json.dumps({"type": "credit", "ctx": 1, "ahead_ms": 10_000}))
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "Long.", "final": True}))
        until(ws, "tts.started")
        ws.send_text(json.dumps({"type": "cancel", "gen": 4}))
        c, _, _ = until(ws, "cancelled")
        assert c["gen"] == 4 and c["ctxs"] == [1] and c["stop_ms"] <= 50
        steps = tts.steps
        time.sleep(0.1)
        assert tts.steps - steps <= 1  # nothing ran after the cancel
        ws.send_text(json.dumps({"type": "cancel", "gen": 4}))
        c2, _, _ = until(ws, "cancelled")
        assert c2["ctxs"] == []


def test_credit_bounds_how_far_tts_runs_ahead(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=1000, step_s=0.001))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid)
        ws.send_text(json.dumps({"type": "credit", "ctx": 1, "ahead_ms": 400, "played_ms": 0}))
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "Long.", "final": True}))
        until(ws, "tts.started")
        time.sleep(0.3)
        assert tts.steps <= 400 // 40 + 1  # stalled at the credit, not rendering ahead
        from speechforge.metrics import METRICS

        assert METRICS.counter("tts_credit_stall_total") >= 1
        ws.send_text(json.dumps({"type": "credit", "ctx": 1, "ahead_ms": 400, "played_ms": 400}))
        time.sleep(0.3)
        assert 400 // 40 < tts.steps <= 800 // 40 + 1
        ws.send_text(json.dumps({"type": "cancel", "ctx": 1}))
        until(ws, "cancelled")


def test_pause_freezes_and_resume_continues(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=1000, step_s=0.002))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid)
        ws.send_text(json.dumps({"type": "credit", "ctx": 1, "ahead_ms": 10_000}))
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "Long.", "final": True}))
        until(ws, "tts.started")
        ws.send_text(json.dumps({"type": "tts.pause", "ctx": 1}))
        ws.send_text(json.dumps({"type": "ping", "t": 1}))
        until(ws, "pong")
        time.sleep(0.05)
        frozen = tts.steps
        time.sleep(0.2)
        assert tts.steps - frozen <= 1
        ws.send_text(json.dumps({"type": "tts.resume", "ctx": 1}))
        time.sleep(0.2)
        assert tts.steps > frozen + 5
        ws.send_text(json.dumps({"type": "cancel", "ctx": 1}))
        until(ws, "cancelled")


def test_context_limits_and_text_caps(config, studioforge, voice_bytes):
    config.realtime.max_text_per_ctx = 20
    app, *_ = build(config, studioforge, tts=FakeTts(chunks=1000, step_s=0.005))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid, ctx=1, gen=1)
        open_ctx(ws, vid, ctx=2, gen=2)  # queued
        open_ctx(ws, vid, ctx=3, gen=3)
        err, _, _ = until(ws, "error")
        assert err["code"] == "busy" and err["ctx"] == 3
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "x" * 21}))
        err, _, _ = until(ws, "error")
        assert err["code"] == "too_long"
        ws.send_text(json.dumps({"type": "cancel", "gen": 2}))
        c, _, _ = until(ws, "cancelled")
        assert c["ctxs"] == [1, 2]


def test_queued_context_starts_when_the_active_one_is_done(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=3))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        open_ctx(ws, vid, ctx=1, gen=1)
        open_ctx(ws, vid, ctx=2, gen=1)
        for c in (1, 2):
            ws.send_text(
                json.dumps({"type": "tts.text", "ctx": c, "text": f"Unit {c}.", "final": True})
            )
            ws.send_text(json.dumps({"type": "tts.close", "ctx": c}))
        d1, a1, _ = until(ws, "tts.done")
        d2, a2, _ = until(ws, "tts.done")
        assert (d1["ctx"], d2["ctx"]) == (1, 2)
        assert {f.ctx for f in a1} == {1} and {f.ctx for f in a2} <= {2}


def test_oversize_message_is_too_long(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "x" * 70_000}))
        err, _, _ = until(ws, "error")
        assert err["code"] == "too_long"


def test_voice_has_list_delete(config, studioforge, voice_bytes):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        vid = put_voice(ws, voice_bytes)
        ws.send_text(json.dumps({"type": "voice.has", "id": vid}))
        assert until(ws, "voice.has")[0]["present"] is True
        ws.send_text(json.dumps({"type": "voice.list"}))
        assert f"sha256:{vid}" in until(ws, "voice.list")[0]["ids"]
        ws.send_text(json.dumps({"type": "voice.delete", "id": vid}))
        assert until(ws, "voice.deleted")[0]["deleted"] is True
        bad = b"not the bytes"
        ws.send_text(
            json.dumps(
                {"type": "voice.put", "id": "a" * 64, "data_b64": base64.b64encode(bad).decode()}
            )
        )
        assert until(ws, "error")[0]["code"] == "bad_frame"


# --------------------------------------------------------------------- lifecycle
def test_vacate_announces_drains_and_closes(config, studioforge, voice_bytes):
    app, tts, *_ = build(config, studioforge, tts=FakeTts(chunks=3))
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        lid = next(iter(studioforge.leases))
        r = tc.post("/internal/vacate", json={"lease_id": lid}, headers={"X-SF-Vacate-Token": "vt"})
        # TestClient's peer is "testclient", not loopback: the route refuses it.
        assert r.status_code == 403
        res = app.state.residency
        tc.portal.call(res.vacate, lid, "vt")
        st, _, _ = until(ws, "state")
        assert st["lease"] == "vacating"
        err, _, _ = until(ws, "error")
        assert err["code"] == "vacated" and err["fatal"] and err["retry_after_s"] == 120
    assert studioforge.leases == {} and tts.closed


def test_resume_reattaches_the_same_session(config, studioforge):
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc:
        with connect(tc) as ws:
            r = hello(ws)
        with connect(tc) as ws2:
            r2 = hello(ws2, resume={"session": r["session"], "token": r["resume_token"]})
            assert r2["type"] == "ready" and r2["session"] == r["session"]
            assert r2["resume_token"] != r["resume_token"]
        with connect(tc) as ws3:
            r3 = hello(ws3, resume={"session": r["session"], "token": "wrong"})
            assert r3["type"] == "ready" and r3["session"] != r["session"]
        # past the resume window a detached session is expired by the residency tick
        config.realtime.resume_window_s = 0.0
        tc.portal.call(app.state.residency.tick)
        assert app.state.sessions.live() == []
        assert app.state.residency._sessions == 0


def test_idle_session_is_closed(config, studioforge):
    config.realtime.idle_close_s = 0.3
    app, *_ = build(config, studioforge)
    with TestClient(app) as tc, connect(tc) as ws:
        hello(ws)
        err = ws.receive_json()
        assert err["type"] == "error" and err["fatal"]


def test_two_sessions_speak_at_once(config, studioforge, voice_bytes):
    app, *_ = build(config, studioforge, tts=FakeTts(chunks=20, step_s=0.002))
    with TestClient(app) as tc, connect(tc) as a, connect(tc) as b:
        hello(a)
        hello(b)
        va, vb = put_voice(a, voice_bytes), put_voice(b, voice_bytes)
        for ws, vid in ((a, va), (b, vb)):
            open_ctx(ws, vid)
            ws.send_text(json.dumps({"type": "tts.text", "ctx": 1, "text": "Hi.", "final": True}))
            ws.send_text(json.dumps({"type": "tts.close", "ctx": 1}))
        da, aa, _ = until(a, "tts.done")
        db, ab, _ = until(b, "tts.done")
        assert len(aa) == len(ab) == 20
