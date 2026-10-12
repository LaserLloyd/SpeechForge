"""Small pure units: frame codec, control checks, auth rotation, models.lock,
metrics, log redaction, rolling STT bookkeeping, word estimates, supervisor,
blue/green switch, bench verdicts, self-test."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import struct
import sys
import time

import numpy as np
import pytest

from speechforge import models
from speechforge.auth import OVERLAP_S, TokenStore, bearer, write_token
from speechforge.config import Config
from speechforge.engines.parakeet import words_from_tokens
from speechforge.engines.turn import EnergyVad, whisper_features
from speechforge.logsetup import JsonFormatter, redact
from speechforge.metrics import Registry
from speechforge.protocol import (
    HEADER_SIZE,
    KIND_MIC,
    KIND_TTS,
    FrameError,
    check_control,
    pack_audio,
    unpack_audio,
)
from speechforge.stt_stream import RATE, RollingStt
from speechforge.tts_ctx import estimate_words, unit_ready


# --------------------------------------------------------------------- codec
def test_frame_roundtrip_header_layout():
    pcm = np.arange(-5, 315, dtype="<i2")
    buf = pack_audio(KIND_TTS, 7, 300, 2**32 + 5, 123456, pcm)
    assert len(buf) == HEADER_SIZE + 2 * len(pcm)
    magic, kind, ctx, gen, seq, pts, n = struct.unpack_from("<BBHHIIH", buf)
    assert (magic, kind, ctx, gen, seq, pts, n) == (0x53, 2, 7, 300, 5, 123456, 320)  # seq wraps
    f = unpack_audio(buf)
    assert f.kind == KIND_TTS and np.array_equal(f.pcm, pcm)


@pytest.mark.parametrize(
    "buf",
    [
        b"",
        b"S\x01" + b"\x00" * 10,
        b"X" + pack_audio(KIND_MIC, 0, 0, 0, 0, np.zeros(4, "<i2"))[1:],
        pack_audio(KIND_MIC, 0, 0, 0, 0, np.zeros(4, "<i2"))[:-2],
        b"S\x09" + pack_audio(KIND_MIC, 0, 0, 0, 0, np.zeros(1, "<i2"))[2:],
    ],
)
def test_bad_frames(buf):
    with pytest.raises(FrameError):
        unpack_audio(buf)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"type":"nope"}',
        '{"type":"tts.open","ctx":1,"gen":1}',
        '{"type":"tts.open","ctx":true,"gen":1,"voice":"x"}',
        '{"type":"tts.text","ctx":70000,"text":"x"}',
        '{"type":"cancel"}',
        '{"type":"credit","ctx":1,"ahead_ms":"3"}',
    ],
)
def test_control_rejects(text):
    with pytest.raises(FrameError):
        check_control(text)


def test_control_accepts_additive_fields():
    msg = check_control('{"type":"cancel","gen":3,"future_field":1}')
    assert msg["gen"] == 3


# --------------------------------------------------------------------- auth
def test_token_rotation_overlap(tmp_path, monkeypatch):
    monkeypatch.delenv("SPEECHFORGE_TOKEN", raising=False)
    env = tmp_path / "speechforge.env"
    env.write_text("# keep me\nSPEECHFORGE_SF_KEY=k\n")
    old = write_token(env, rotate=False)
    now = [time.time()]
    store = TokenStore(env, clock=lambda: now[0])
    assert store.check(old) and not store.check("nope") and not store.check("")
    new = write_token(env, rotate=True)
    os.utime(env, ns=(time.time_ns(), time.time_ns() + 10**9))  # make the mtime change visible
    assert store.check(new) and store.check(old)
    now[0] += OVERLAP_S + 1
    assert store.check(new) and not store.check(old)
    text = env.read_text()
    assert "SPEECHFORGE_SF_KEY=k" in text and "# keep me" in text
    if os.name != "nt":
        assert env.stat().st_mode & 0o777 == 0o600


def test_bearer_parsing():
    assert bearer("Bearer abc") == "abc" and bearer("bearer  abc ") == "abc"
    assert bearer("Basic abc") is None and bearer(None) is None and bearer("Bearer ") is None


# --------------------------------------------------------------------- models.lock
def test_lock_is_complete_and_pinned():
    lock = models.load_lock()
    files = lock["files"]
    assert all(len(f["sha256"]) == 64 and f["size"] > 0 for f in files)
    assert all(
        ("/resolve/" in f["url"] or "/raw/" in f["url"]) and "/main/" not in f["url"] for f in files
    )
    cfg = Config(data_dir="x")
    need = {f["dest"] for f in models.files_for(cfg)}
    assert "chatterbox-turbo/onnx/language_model.onnx_data" in need
    assert "parakeet-tdt-0.6b-v2/encoder-model.onnx.data" in need
    assert "smart-turn/smart-turn-v3.2-gpu.onnx" in need
    assert "silero-vad/silero_vad.onnx" in need
    assert not any("int8" in d or "_q4" in d for d in need)


def _fake_lock(monkeypatch, data: bytes):
    entry = {
        "groups": ["vad"],
        "url": "https://example.invalid/x/resolve/abc/f.onnx",
        "dest": "silero-vad/silero_vad.onnx",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    monkeypatch.setattr(models, "load_lock", lambda: {"files": [entry]})
    return entry


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_verifies_before_rename(tmp_path, monkeypatch):
    data = b"model bytes" * 100
    _fake_lock(monkeypatch, data)
    cfg = Config(data_dir=tmp_path)
    got = models.fetch(cfg, groups=["vad"], opener=lambda req, timeout: _Resp(data))
    assert len(got) == 1 and got[0].read_bytes() == data
    assert models.verify(cfg, ["vad"]) == {"silero-vad/silero_vad.onnx": "ok"}
    # tampered bytes never replace a file
    got[0].unlink()
    with pytest.raises(models.ModelError):
        models.fetch(cfg, groups=["vad"], opener=lambda req, timeout: _Resp(b"evil" * 275))
    assert not got[0].exists() and not list((tmp_path / "models").rglob("*.partial"))


def test_verify_detects_changes_and_rehashes_stale_stamps(tmp_path, monkeypatch):
    data = b"abc" * 1000
    _fake_lock(monkeypatch, data)
    cfg = Config(data_dir=tmp_path)
    path = tmp_path / "models" / "silero-vad" / "silero_vad.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    assert models.verify(cfg, ["vad"])["silero-vad/silero_vad.onnx"] == "rehashed"  # no stamp
    assert models.verify(cfg, ["vad"])["silero-vad/silero_vad.onnx"] == "ok"
    # Age the stamp instead of shrinking the window to nanoseconds: on Windows
    # two time.time() calls can be closer than any positive window.
    stamp = tmp_path / "state" / "model-stamps" / "silero-vad__silero_vad.onnx.json"
    aged = json.loads(stamp.read_text())
    aged["at"] -= 10 * 86400
    stamp.write_text(json.dumps(aged))
    assert models.verify(cfg, ["vad"])["silero-vad/silero_vad.onnx"] == "rehashed"
    path.write_bytes(b"abd" * 1000)  # same size, different bytes, new mtime
    # The fast path trusts size + mtime_ns; a rewrite inside one filesystem
    # timestamp tick would look unchanged, so move the mtime on explicitly.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    with pytest.raises(models.ModelError):
        models.verify(cfg, ["vad"])


# --------------------------------------------------------------------- metrics / logs
def test_metrics_render():
    r = Registry()
    r.inc("session_errors_total", code="busy")
    r.set("sessions_open", 2)
    for v in (10, 20, 300):
        r.observe("tts_ttfa_ms", v)
    text = r.render()
    assert 'speechforge_session_errors_total{code="busy"} 1' in text
    assert "speechforge_sessions_open 2" in text
    assert 'speechforge_tts_ttfa_ms_bucket{le="+Inf"} 3' in text
    assert r.quantile("tts_ttfa_ms", 0.5) == 20


def test_log_redaction():
    tok = "Zx9" * 15
    assert tok not in redact(f"Authorization: Bearer {tok}")
    assert tok not in redact(f"vacate_token={tok} x-sf-vacate-token: {tok}")
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "token=%s", (tok,), None)
    rec.turn_id = "t9"
    line = json.loads(JsonFormatter().format(rec))
    assert tok not in line["msg"] and line["turn_id"] == "t9"


def _asyncio_record(msg: str, exc: BaseException | None) -> logging.LogRecord:
    info = (type(exc), exc, None) if exc is not None else None
    return logging.LogRecord("asyncio", logging.ERROR, "f", 1, msg, (), info)


def test_proactor_reset_noise_is_dropped_and_counted():
    from speechforge.logsetup import ProactorResetFilter
    from speechforge.metrics import METRICS

    f = ProactorResetFilter()
    cb = "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)"
    before = METRICS.counter("asyncio_conn_reset_total")
    assert f.filter(_asyncio_record(cb, ConnectionResetError(10054, "reset"))) is False
    assert f.filter(_asyncio_record(cb, ConnectionAbortedError())) is False
    # anything else from asyncio -- or the same callback failing another way -- still logs
    assert f.filter(_asyncio_record(cb, RuntimeError("real"))) is True
    assert f.filter(_asyncio_record("Task exception was never retrieved", OSError())) is True
    assert f.filter(_asyncio_record("Exception in callback x()", ConnectionResetError())) is True
    assert f.filter(_asyncio_record(cb, None)) is True
    other = logging.LogRecord("speechforge", logging.ERROR, "f", 1, cb, (), None)
    assert f.filter(other) is True
    assert METRICS.counter("asyncio_conn_reset_total") == before + 2


# --------------------------------------------------------------------- rolling STT
def _speech(n):
    return (0.3 * np.sin(np.arange(n) * 0.2) * 32767).astype("<i2")


def test_rolling_stt_pause_partial_final_and_carry_over():
    t = [100.0]
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: t[0])
    ev = []
    for i in range(0, RATE, 320):
        ev += s.push(i, _speech(320))
    assert ev[0] == {"type": "stt.vad", "speech": True, "at": 0}
    assert s.partial_due()
    job = s.take_partial()
    assert job.upto == RATE and len(job.tail) == RATE and job.commit is None
    fr = s.apply_partial(job, "hello there friend", 12)
    assert fr["committed"] == "" and fr["tail"] == "hello there friend" and fr["stable_words"] == 0
    assert s.apply_partial(job, "hello there friend", 12) is None  # unchanged: not re-sent
    for i in range(RATE, RATE + 4800, 320):
        ev += s.push(i, np.zeros(320, "<i2"))
    pauses = [e for e in ev if e["type"] == "_pause"]
    assert len(pauses) == 1 and pauses[0]["pause_ms"] >= 150
    eou = s.take_eou(pauses[0]["at"], pauses[0]["pause_ms"])
    assert len(eou.audio) == pauses[0]["at"]
    fin = s.take_final("x")
    # audio after the flush belongs to the next utterance
    s.push(RATE + 4800, _speech(320))
    frame = s.apply_final(fin, "hello there", [("hello", 0.0, 0.4), ("there", 0.4, 0.8)], 30)
    assert frame["upto"] == RATE + 4800 and frame["words"][1] == ["there", 6400, 12800]
    assert s.tail_start == RATE + 4800 and s.end == RATE + 5120


def _paused_stt(text, **kw):
    """A RollingStt one second into speech with ``text`` as its latest partial."""
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, **kw)
    ev = []
    for i in range(0, RATE, 320):
        ev += s.push(i, _speech(320))
    s.apply_partial(s.take_partial(now=1.0), text, 5)
    return s


def _silence_until_pause(s, ms):
    """Feed ``ms`` more silence from where the stream stands; return the ``_pause`` events."""
    ev = []
    for i in range(s.expected, s.expected + ms * RATE // 1000, 320):
        ev += s.push(i, np.zeros(320, "<i2"))
    return [e for e in ev if e["type"] == "_pause"]


def test_pause_gating_punctuated_fires_at_the_short_floor():
    s = _paused_stt("that is the end.", min_silence_ms=300, unpunctuated_silence_ms=800)
    assert _silence_until_pause(s, 250) == []  # under the floor
    ev = _silence_until_pause(s, 100)
    assert len(ev) == 1 and 300 <= ev[0]["pause_ms"] < 400  # once, near the floor


def test_pause_gating_unpunctuated_waits_for_the_long_floor():
    s = _paused_stt("and then he said", min_silence_ms=300, unpunctuated_silence_ms=800)
    assert _silence_until_pause(s, 700) == []  # a mid-sentence breath: not consulted
    ev = _silence_until_pause(s, 200)
    assert len(ev) == 1 and 800 <= ev[0]["pause_ms"] < 900
    # still once per pause
    assert _silence_until_pause(s, 500) == []


def test_pause_gating_resumed_speech_before_the_long_floor_fires_nothing():
    s = _paused_stt("and then he said", min_silence_ms=300, unpunctuated_silence_ms=800)
    assert _silence_until_pause(s, 600) == []
    ev = []
    for i in range(s.expected, s.expected + RATE // 2, 320):
        ev += s.push(i, _speech(320))
    assert not [e for e in ev if e["type"] == "_pause"]
    # the pause clock restarts with the next pause
    assert _silence_until_pause(s, 700) == []


def test_pause_gating_punctuation_arriving_mid_pause_releases_it():
    s = _paused_stt("and then he said", min_silence_ms=300, unpunctuated_silence_ms=800)
    assert _silence_until_pause(s, 400) == []
    s._last_partial_text = "and then he said so."  # the partial catches up during the pause
    ev = _silence_until_pause(s, 100)
    assert len(ev) == 1


def test_pause_gating_client_min_pause_is_clamped_by_the_floor():
    # a client asking for 60 ms still waits for the server's floor
    s = _paused_stt("done.", min_pause_ms=60, min_silence_ms=300, unpunctuated_silence_ms=800)
    assert s.pause_trigger_ms() == 300
    assert _silence_until_pause(s, 250) == []
    # a client asking for more than the floor is honoured
    s = _paused_stt("done.", min_pause_ms=500, min_silence_ms=300, unpunctuated_silence_ms=800)
    assert s.pause_trigger_ms() == 500
    # and a floor-less RollingStt keeps the old behaviour
    s = _paused_stt("and then", min_pause_ms=150)
    assert s.pause_trigger_ms() == 150


def test_rolling_stt_commits_long_tails_at_a_pause():
    s = RollingStt(vad=EnergyVad().stream(), max_tail_s=2.0, clock=lambda: 0.0)
    pts = 0
    for chunk in (_speech(RATE), np.zeros(8000, "<i2"), _speech(RATE + 8000)):
        for i in range(0, len(chunk), 320):
            s.push(pts, chunk[i : i + 320])
            pts += 320
    job = s.take_partial(now=1.0)
    assert job.commit is not None and RATE < job.commit_end < RATE + 8000
    assert job.tail_start == job.commit_end and len(job.tail) <= 2 * RATE


def test_release_partial_puts_a_withdrawn_commit_back():
    s = RollingStt(vad=EnergyVad().stream(), max_tail_s=2.0, clock=lambda: 0.0)
    pts = 0
    for chunk in (_speech(RATE), np.zeros(8000, "<i2"), _speech(RATE + 8000)):
        for i in range(0, len(chunk), 320):
            s.push(pts, chunk[i : i + 320])
            pts += 320
    job = s.take_partial(now=1.0)
    assert job.commit is not None and s.tail_start == job.commit_end
    s.release_partial(job)  # the lane never ran it: the final decodes the whole tail
    assert s.tail_start == 0 and len(s.take_final().tail) == s.end


def test_release_partial_without_a_commit_changes_nothing():
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    job = s.take_partial(now=1.0)
    assert job.commit is None
    s.release_partial(job)
    assert s.tail_start == 0 and len(s.take_final().tail) == RATE


async def test_lane_ticket_withdraws_only_a_job_that_has_not_started():
    import asyncio
    import threading

    from speechforge.lanes import PRIO_EOU, PRIO_PARTIAL, SttLane, Ticket

    lane = SttLane("sf-test-lane")
    try:
        gate = threading.Event()
        ran: list[str] = []
        blocker = lane.submit(PRIO_EOU, lambda: (gate.wait(5), ran.append("blocker"))[0])
        queued_t, started_t = Ticket(), Ticket()
        queued = lane.submit(PRIO_PARTIAL, lambda: ran.append("queued"), queued_t)
        assert queued_t.cancel() is True and queued_t.cancelled
        queued.cancel()
        gate.set()
        await blocker
        # the worker has now passed the withdrawn job without running it
        started = lane.submit(PRIO_PARTIAL, lambda: ran.append("started") or 7, started_t)
        assert await started == 7
        assert started_t.cancel() is False and not started_t.cancelled  # too late: it ran
        await asyncio.sleep(0.05)
        assert ran == ["blocker", "started"]
        # a cancelled ticket whose future is NOT cancelled still never hangs the awaiter
        gate2 = threading.Event()
        blocker2 = lane.submit(PRIO_EOU, lambda: gate2.wait(5))
        t3 = Ticket()
        orphan = lane.submit(PRIO_PARTIAL, lambda: ran.append("orphan"), t3)
        assert t3.cancel()
        gate2.set()
        await blocker2
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(orphan, 2)
        assert "orphan" not in ran
    finally:
        lane.close()


def test_rolling_stt_gaps_and_overlaps():
    s = RollingStt(vad=EnergyVad().stream())
    s.push(0, _speech(320))
    s.push(640, _speech(320))  # 320 missing: zero-filled
    assert s.end == 960
    s.push(640, _speech(320))  # a resend: dropped
    assert s.end == 960


def test_rolling_stt_first_frame_after_reset_adopts_pts_without_zero_fill():
    # A detach resets the stream while the client's mic clock keeps running; the
    # first frame of the resume must become the clock origin, not be preceded by
    # up to MAX_GAP_FILL_S of zero-filled silence.
    s = RollingStt(vad=EnergyVad().stream())
    s.push(0, _speech(320))
    s.reset_all()
    pts = 3 * RATE  # inside the 5 s fill bound: the old code zero-filled 3 s
    s.push(pts, _speech(320))
    assert s.end - s._buf_start == 320 and s._buf_start == pts
    assert s.end == pts + 320 and s.expected == pts + 320
    assert s.utt_start == s.tail_start == pts and s._fed == pts
    assert s._vad_pos >= pts  # the VAD clock follows the mic clock
    assert not s.partial_due(now=10.0) and s.take_partial(now=10.0).tail_start == pts
    s.push(pts + 320, _speech(320))  # contiguous afterwards
    assert s.end - s._buf_start == 640


def test_rolling_stt_small_gap_fill_still_works_after_the_first_frame():
    s = RollingStt(vad=EnergyVad().stream())
    s.reset_all()
    s.push(1000, _speech(320))
    assert s._buf_start == 1000
    s.push(1000 + 320 + 640, _speech(320))  # 640 missing: zero-filled
    assert s.end - s._buf_start == 320 + 640 + 320


def test_stable_words_rule():
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    job = s.take_partial(now=1)
    s.apply_partial(job, "one two three four", 1)
    fr = s.apply_partial(job, "one two three five six", 1)
    assert fr["stable_words"] == 1  # 3 agree, minus 2


def test_words_from_tokens_and_estimates():
    w = words_from_tokens(
        [" He", "y", ",", " we", "lc", "ome", " back"], [0, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9]
    )
    assert [x[0] for x in w] == ["Hey,", "welcome", "back"]
    assert w[0][1:] == (0.0, 0.5) and w[-1][2] == pytest.approx(0.98)
    est = estimate_words("Sure, one moment.", 100, 1000)
    assert [x[0] for x in est] == ["Sure", "one", "moment"]
    assert est[0][1] == 100 and est[-1][2] == 1100


def test_whisper_features_shape():
    f = whisper_features(np.zeros(16000, np.float32) + 0.01)
    assert f.shape == (1, 80, 800) and f.dtype == np.float32


# --------------------------------------------------------------------- supervisor / update
def test_crash_tracker():
    from speechforge.supervise import CrashTracker

    t = CrashTracker(limit=5, window_s=600)
    assert not any(t.crashed(x) for x in (0, 10, 20, 30))
    assert t.crashed(40)
    t2 = CrashTracker(limit=5, window_s=600)
    assert not any(t2.crashed(x * 200) for x in range(10))


def test_child_python_prefers_the_current_release(tmp_path):
    from speechforge.supervise import child_python

    assert child_python(None) == sys.executable
    py = tmp_path / "current" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("")
    assert child_python(tmp_path) == str(py)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink path; junctions on Windows")
def test_switch_and_prune(tmp_path):
    from speechforge import update

    a, b, c = (tmp_path / "releases" / t for t in ("v1", "v2", "v3"))
    for d in (a, b, c):
        d.mkdir(parents=True)
        time.sleep(0.01)
    update.switch(tmp_path, a)
    assert update.current_target(tmp_path) == a.resolve()
    update.switch(tmp_path, c)
    assert update.current_target(tmp_path) == c.resolve()
    update.prune(tmp_path, keep={c.resolve(), a.resolve()})
    assert a.exists() and c.exists() and not b.exists()
    assert update.rollback(tmp_path, tmp_path / "data") == 0
    assert update.current_target(tmp_path) == a.resolve()
    assert (tmp_path / "data" / "state" / "restart.request").exists()


LS_REMOTE = """\
1111111111111111111111111111111111111111\trefs/tags/v0.2.0
2222222222222222222222222222222222222222\trefs/tags/v0.2.0^{}
3333333333333333333333333333333333333333\trefs/tags/v0.10.0
4444444444444444444444444444444444444444\trefs/tags/v0.10.0^{}
5555555555555555555555555555555555555555\trefs/tags/v0.9.1
6666666666666666666666666666666666666666\trefs/tags/v0.10.0-rc1
7777777777777777777777777777777777777777\trefs/tags/nightly
8888888888888888888888888888888888888888\trefs/tags/release-2
"""


class _FakeRun:
    """Stands in for subprocess.run: records argv, answers from a table keyed on
    the subcommand, and can refuse like a failed git call."""

    def __init__(self, answers: dict[str, str | Exception]):
        self.answers = answers
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        import subprocess

        self.calls.append(list(argv))
        key = argv[1] if argv[1] != "-C" else argv[3]
        ans = self.answers.get(key, "")
        if isinstance(ans, Exception):
            raise ans
        return subprocess.CompletedProcess(argv, 0, stdout=ans, stderr="")


def test_release_tags_are_v_only_and_version_sorted():
    from speechforge import update

    tags = update.release_tags(LS_REMOTE)
    assert tags == ["v0.2.0", "v0.9.1", "v0.10.0-rc1", "v0.10.0"]
    assert tags[-1] == "v0.10.0"  # text order would have picked v0.9.1
    assert update.version_key("v1") == update.version_key("v1.0.0")
    assert update.version_key("v1.2.3-rc1") < update.version_key("v1.2.3")
    assert update.version_key("nightly") == ()


def test_source_tags_runs_one_ls_remote(monkeypatch):
    from speechforge import update

    fake = _FakeRun({"ls-remote": LS_REMOTE})
    monkeypatch.setattr(update.subprocess, "run", fake)
    assert update.source_tags("https://example.com/x.git")[-1] == "v0.10.0"
    assert fake.calls == [["git", "ls-remote", "--tags", "https://example.com/x.git"]]


def test_is_url():
    from speechforge.update import is_url

    assert is_url("https://github.com/LaserLloyd/SpeechForge.git")
    assert is_url("ssh://git@example.com/x.git")
    assert is_url("git@example.com:x/y.git")
    assert not is_url(r"C:\releases\speechforge.bundle")
    assert not is_url("/srv/speechforge.bundle")
    assert not is_url("speechforge.bundle")


def test_default_source_prefers_a_live_origin(monkeypatch, tmp_path):
    from speechforge import update

    # A URL origin wins.
    fake = _FakeRun({"remote": "https://example.com/fork.git\n"})
    monkeypatch.setattr(update.subprocess, "run", fake)
    assert update.default_source(tmp_path) == "https://example.com/fork.git"
    assert fake.calls[0][:4] == ["git", "-C", str(tmp_path), "remote"]
    # A bundle origin that is still there wins; one that is gone falls back.
    bundle = tmp_path / "speechforge.bundle"
    bundle.write_bytes(b"")
    monkeypatch.setattr(update.subprocess, "run", _FakeRun({"remote": f"{bundle}\n"}))
    assert update.default_source(tmp_path) == str(bundle)
    gone = tmp_path / "gone.bundle"
    monkeypatch.setattr(update.subprocess, "run", _FakeRun({"remote": f"{gone}\n"}))
    assert update.default_source(tmp_path) == update.DEFAULT_SOURCE
    # No origin at all (a plain checkout without a remote) falls back too.
    import subprocess

    err = subprocess.CalledProcessError(2, ["git"])
    monkeypatch.setattr(update.subprocess, "run", _FakeRun({"remote": err}))
    assert update.default_source(tmp_path) == update.DEFAULT_SOURCE


def test_clone_argv_is_one_shallow_tag_clone(tmp_path):
    from speechforge.update import clone_argv

    argv = clone_argv("https://example.com/x.git", "v0.3.0", tmp_path / "releases" / "v0.3.0")
    assert argv[:2] == ["git", "clone"]
    assert "--branch" in argv and argv[argv.index("--branch") + 1] == "v0.3.0"
    assert argv[argv.index("--depth") + 1] == "1"
    assert argv[-2:] == ["https://example.com/x.git", str(tmp_path / "releases" / "v0.3.0")]
    assert "pull" not in argv


def test_apply_refuses_before_cloning(monkeypatch, tmp_path):
    """An unknown tag, or a source without release tags, stops before anything
    is cloned, synced or started."""
    from speechforge import update

    fake = _FakeRun({"ls-remote": LS_REMOTE})
    monkeypatch.setattr(update.subprocess, "run", fake)
    kw = {"token": "t", "voice": b"", "live_port": 1, "data_dir": tmp_path / "data"}
    assert update.apply(tmp_path, "https://example.com/x.git", "v9.9.9", **kw) == 1
    assert [c[1] for c in fake.calls] == ["ls-remote"]
    fake = _FakeRun({"ls-remote": "abc\trefs/tags/nightly\n"})
    monkeypatch.setattr(update.subprocess, "run", fake)
    assert update.apply(tmp_path, "https://example.com/x.git", None, **kw) == 1
    assert [c[1] for c in fake.calls] == ["ls-remote"]
    assert not (tmp_path / "releases").exists()


def test_update_cli_accepts_source_and_the_old_bundle_name():
    from speechforge.__main__ import build_parser

    ap = build_parser()
    assert ap.parse_args(["update", "--check", "--source", "https://x/y.git"]).source == (
        "https://x/y.git"
    )
    assert ap.parse_args(["update", "--apply", "--bundle", "f.bundle", "--tag", "v1"]).source == (
        "f.bundle"
    )
    assert ap.parse_args(["update", "--rollback"]).source == ""


# --------------------------------------------------------------------- bench / self-test
def test_bench_verdict():
    from speechforge.bench import verdict

    lines, ok = verdict(
        {
            "tts_ttfa_p50_ms": 120,
            "tts_ttfa_p90_ms": 200,
            "cancel_stop_p99_ms": 12,
            "names_correct": 0.95,
        }
    )
    assert ok and any("PASS" in x for x in lines) and any("not measured" in x for x in lines)
    _, ok = verdict({"tts_ttfa_p90_ms": 260})
    assert not ok
    _, ok = verdict({"names_correct": 0.5})
    assert not ok


def test_selftest_with_fakes():
    from speechforge import selftest
    from speechforge.config import SelftestConfig
    from speechforge.pipeline import Speaker
    from tests.conftest import FakeStt, FakeTts

    class GoodStt(FakeStt):
        def transcribe(self, audio):
            return selftest.GOLDEN_TEXT if len(audio) > 100_000 else selftest.TTS_LINE

    res = selftest.run(Speaker(FakeTts(chunks=4), GoodStt()), SelftestConfig(), enforce_timing=True)
    assert res["pass"], res
    assert res["stt_wer"] == 0 and res["roundtrip_wer"] == 0 and res["cancel_ms"] < 50
    bad = selftest.run(
        Speaker(FakeTts(chunks=4), FakeStt()), SelftestConfig(), enforce_timing=False
    )
    assert not bad["pass"] and set(bad["failed"]) == {"stt_wer", "roundtrip_wer"}


def test_wer():
    from speechforge.selftest import wer

    assert wer("the cat sat", "the cat sat") == 0
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert wer("a b", "") == 1.0


# --------------------------------------------------------------- reference clip (S30)


def _sine(freq: float, rate: int, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _bin_amp(x: np.ndarray, rate: int, freq: float) -> float:
    """Amplitude of ``freq`` in ``x`` (Hann-windowed FFT, centre of the clip)."""
    mid = x[len(x) // 4 : -len(x) // 4]
    w = np.hanning(len(mid))
    spec = np.abs(np.fft.rfft(mid * w)) / (w.sum() / 2)
    k = int(round(freq * len(mid) / rate))
    return float(spec[k - 2 : k + 3].max())


@pytest.mark.parametrize("src", [48000, 44100, 22050, 16000])
def test_resample_poly_keeps_frequency_and_amplitude(src):
    from speechforge.audio import resample_poly

    y = resample_poly(_sine(1000.0, src, 1.0), src, 24000)
    assert abs(len(y) - 24000) <= 1
    assert abs(_bin_amp(y, 24000, 1000.0) - 0.5) / 0.5 < 0.01
    # the peak of the spectrum is at 1 kHz, not somewhere else
    mid = y[len(y) // 4 : -len(y) // 4]
    assert (
        abs(np.argmax(np.abs(np.fft.rfft(mid * np.hanning(len(mid))))) * 24000 / len(mid) - 1000)
        < 5
    )


def test_resample_poly_rejects_aliases_that_linear_lets_in():
    from speechforge.audio import resample_linear, resample_poly

    # 13.5 kHz is above the 12 kHz Nyquist of 24 kHz and folds to 10.5 kHz
    x = _sine(13500.0, 48000, 1.0)
    lin = _bin_amp(resample_linear(x, 48000, 24000), 24000, 10500.0)
    poly = _bin_amp(resample_poly(x, 48000, 24000), 24000, 10500.0)
    assert 20 * np.log10(lin / poly) >= 40.0
    assert 20 * np.log10(poly / 0.5) <= -40.0
    # and a 15 kHz tone (folds to 9 kHz), through the non-integer ratio too
    for src in (48000, 44100):
        y = resample_poly(_sine(15000.0, src, 1.0), src, 24000)
        assert 20 * np.log10(max(np.max(np.abs(y[2000:-2000])), 1e-12) / 0.5) <= -40.0


def test_resample_poly_edges_and_noops():
    from speechforge.audio import resample_poly

    x = np.full(4800, 0.25, np.float32)
    y = resample_poly(x, 48000, 24000)
    assert np.allclose(y, 0.25, atol=1e-3)  # a DC level does not ring at the ends
    assert resample_poly(x, 24000, 24000) is x or np.array_equal(resample_poly(x, 24000, 24000), x)
    assert len(resample_poly(np.zeros(0, np.float32), 44100, 24000)) == 0
    assert len(resample_poly(np.ones(5, np.float32), 44100, 24000)) >= 1


def _voiced(seconds: float, amp: float = 0.2, rate: int = 24000) -> np.ndarray:
    return _sine(220.0, rate, seconds, amp) * (0.6 + 0.4 * _sine(3.0, rate, seconds, 1.0))


def _db_of(x: np.ndarray) -> tuple[float, float]:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20 * np.log10(rms), 20 * np.log10(float(np.max(np.abs(x))))


def test_prepare_reference_trims_silence_and_normalises():
    from speechforge.audio import prepare_reference

    rng = np.random.default_rng(1)
    hush = lambda s: (0.0005 * rng.standard_normal(int(24000 * s))).astype(np.float32)  # noqa: E731
    clip = np.concatenate([hush(1.0), _voiced(4.0), hush(0.7)])
    out, st = prepare_reference(clip)
    # 1.0 s lead and 0.7 s tail go, less the 50 ms margin each
    assert 900 <= st["trimmed_lead_ms"] <= 960
    assert 600 <= st["trimmed_tail_ms"] <= 660
    assert abs(len(out) / 24000 - 4.1) < 0.06 and st["duration_s"] == round(len(out) / 24000, 3)
    rms_db, peak_db = _db_of(out)
    assert abs(rms_db - (-20.0)) < 0.2 and peak_db <= -1.0 + 0.05
    assert st["rms_dbfs_in"] < -20.0 and st["gain_db"] > 0
    assert out.dtype == np.float32
    # the signal inside is the input's, scaled: no gate, no reshaping
    k = int(0.05 * 24000)
    a = clip[24000 - k : 24000 - k + len(out)]
    assert np.allclose(
        out / np.max(np.abs(out)), (a - a.mean()) / np.max(np.abs(a - a.mean())), atol=0.02
    )
    assert [w["code"] for w in st["warnings"]] == ["long_lead_silence"]


def test_prepare_reference_gain_is_capped_by_peak():
    from speechforge.audio import prepare_reference

    x = _voiced(4.0, amp=0.02)
    x[24000] = 0.3  # one transient: RMS-matching to -20 would put it far over full scale
    out, st = prepare_reference(x)
    rms_db, peak_db = _db_of(out)
    assert abs(peak_db - (-1.0)) < 0.05 and rms_db < -20.0
    assert abs(st["gain_db"] - (-1.0 - 20 * np.log10(0.3))) < 0.3
    assert np.max(np.abs(out)) < 1.0


def test_prepare_reference_attenuates_hot_clips_and_removes_dc():
    from speechforge.audio import prepare_reference

    x = _voiced(4.0, amp=0.5) + 0.1  # loud and offset
    out, st = prepare_reference(x)
    assert abs(float(out.mean())) < 1e-3 * float(np.max(np.abs(out)))
    assert st["gain_db"] < 0 and abs(_db_of(out)[0] + 20.0) < 0.2
    kept, st2 = prepare_reference(x, dc=False)
    assert abs(float(kept.mean())) > 1e-3
    assert st2["peak_dbfs_in"] == st["peak_dbfs_in"]


def test_prepare_reference_warnings():
    from speechforge.audio import prepare_reference

    codes = lambda x, **kw: [w["code"] for w in prepare_reference(x, **kw)[1]["warnings"]]  # noqa: E731
    assert codes(_voiced(4.0)) == []
    assert "short_clip" in codes(_voiced(2.0))
    assert "quiet_clip" in codes(_voiced(4.0, amp=0.004))
    loud = _voiced(4.0)
    loud[100:104] = 1.0
    assert "clipping" in codes(loud)
    loud[100:102] = 0.2
    assert "clipping" not in codes(loud)  # only two samples left at full scale
    assert codes(np.zeros(24000 * 4, np.float32)) == ["silent_clip"]
    assert codes(np.zeros(0, np.float32)) == ["silent_clip"]
    # a caller that counted clipped samples at the original rate can say so
    assert "clipping" in codes(_voiced(4.0), clipped_samples=5)
    lead = np.concatenate([np.zeros(24000, np.float32), _voiced(4.0)])
    out, st = prepare_reference(lead)
    assert st["trimmed_lead_ms"] > 500 and "long_lead_silence" in [
        w["code"] for w in st["warnings"]
    ]


def test_reference_clip_helper_resamples_then_prepares():
    from speechforge.audio import reference_clip

    x = np.concatenate([np.zeros(48000, np.float32), _voiced(5.0, rate=48000)])
    y, st = reference_clip(x, 48000)
    assert st["prepared"] is True and abs(len(y) / 24000 - 5.1) < 0.1
    raw, st2 = reference_clip(x, 48000, prepare=False)
    assert st2 == {"duration_s": round(len(raw) / 24000, 3), "prepared": False}
    assert abs(len(raw) / 24000 - 6.0) < 0.01
    assert Config(data_dir=".").voices.prepare_reference is True
    with pytest.raises(ValueError):
        Config(data_dir=".", voices={"reference_target_dbfs": -5})


def test_unit_ready_cuts_sentences_not_clauses_or_abbreviations():
    assert unit_ready("Sure, I can help.")
    assert unit_ready("Really?")
    assert unit_ready('He said "wait!" ')
    assert unit_ready("It costs three dollars. ")
    assert unit_ready("It costs 3. ")  # a space after the stop: the sentence ended on a number
    assert not unit_ready("It costs 3.")  # no space yet: probably 3.5
    assert not unit_ready("Ask Dr.")
    assert not unit_ready("See e.g.")
    assert not unit_ready("Meet J.")
    assert not unit_ready("First this; then that")  # clauses stay in one unit
    assert not unit_ready("Here is the plan:")
    assert not unit_ready("no terminator")
