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
from speechforge.tts_ctx import estimate_words


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


def test_rolling_stt_gaps_and_overlaps():
    s = RollingStt(vad=EnergyVad().stream())
    s.push(0, _speech(320))
    s.push(640, _speech(320))  # 320 missing: zero-filled
    assert s.end == 960
    s.push(640, _speech(320))  # a resend: dropped
    assert s.end == 960


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
