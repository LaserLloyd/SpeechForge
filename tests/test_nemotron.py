"""The Nemotron streaming STT engine, on CPU fakes: the ``StreamingSttAdapter``
wiring in ``stt_stream`` (partials from ``feed`` at the cadence, final from
``finalize``, reset per utterance, Parakeet's path untouched), option
validation, the models.lock group, the registry name, the worker protocol
and the subprocess backend (a child interpreter running the worker with a
fake recognizer). No sherpa-onnx, no models, no GPU."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest
from starlette.testclient import TestClient

from speechforge import models
from speechforge.config import Config
from speechforge.engines import base, nemotron, nemotron_worker
from speechforge.engines.base import is_streaming_stt
from speechforge.engines.turn import EnergyVad
from speechforge.stt_stream import RATE, RollingStt
from tests.conftest import FakeStt
from tests.test_realtime import TOKEN, build, connect, hello, silence, speech, until


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_TOKEN", TOKEN)


# ------------------------------------------------------------------ fakes
class FakeHandle:
    """An ``SttStreamHandle``: text = the sample count fed since the last reset."""

    def __init__(self, owner: FakeStreamingStt) -> None:
        self.owner = owner
        self.fed: list[int] = []
        self.resets = 0
        self.finals = 0
        self.closed = False

    @property
    def total(self) -> int:
        return sum(self.fed)

    def feed(self, audio: np.ndarray) -> str:
        assert audio.dtype == np.float32
        self.fed.append(len(audio))
        self.owner.feeds += 1
        return f"heard {self.total} samples"

    def finalize(self):
        self.finals += 1
        n = self.total / RATE
        words = [] if not self.owner.words else [("final", 0.0, n / 2), ("samples", n / 2, n)]
        return f"final {self.total} samples", words

    def reset(self) -> None:
        self.resets += 1
        self.fed = []

    def close(self) -> None:
        self.closed = True


class FakeStreamingStt:
    name = "fake-streaming"
    sample_rate = 16000

    def __init__(self, *, words: bool = True, fallback: bool = False) -> None:
        self.words = words
        self.feeds = 0
        self.transcribes = 0
        self.handles: list[FakeHandle] = []
        self.closed = False
        if fallback:
            self.transcribe_words = self._transcribe_words  # type: ignore[method-assign]

    def create_stream(self) -> FakeHandle:
        h = FakeHandle(self)
        self.handles.append(h)
        return h

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribes += 1
        return f"whole {len(audio)}"

    def _transcribe_words(self, audio: np.ndarray):
        self.transcribes += 1
        n = len(audio) / RATE
        return f"fallback {len(audio)}", [("fallback", 0.0, n)]

    def warm_up(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-streaming", "lock_key": "nemotron-speech-streaming-en-0.6b"}

    def close(self) -> None:
        self.closed = True


def _speech(n):
    return (0.3 * np.sin(np.arange(n) * 0.2) * 32767).astype("<i2")


class _Speaker:
    def __init__(self, stt):
        self.stt = stt

    def transcribe(self, audio):
        return self.stt.transcribe(audio)

    def transcribe_words(self, audio):
        fn = getattr(self.stt, "transcribe_words", None)
        return (self.transcribe(audio), []) if fn is None else fn(audio)


# -------------------------------------------------------------- protocols
def test_streaming_detection_and_protocols():
    assert is_streaming_stt(FakeStreamingStt()) and not is_streaming_stt(FakeStt())
    assert isinstance(FakeStreamingStt(), base.StreamingSttAdapter)
    assert isinstance(FakeStreamingStt(), base.SttAdapter)  # still a plain SttAdapter
    assert isinstance(FakeStreamingStt().create_stream(), base.SttStreamHandle)
    assert isinstance(FakeStt(), base.SttAdapter)
    assert not isinstance(FakeStt(), base.StreamingSttAdapter)


# ------------------------------------------------------------ stt_stream
def test_streaming_partials_feed_only_new_audio_at_the_cadence():
    t = [100.0]
    stt = FakeStreamingStt()
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: t[0], stream=h, partial_ms=160)
    assert h.resets == 1  # reset_all at construction
    sp = _Speaker(stt)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    assert s.partial_due()
    job = s.take_partial()
    assert job.streaming and job.tail_start == 0 and job.upto == RATE and len(job.tail) == RATE
    ctext, cwords, tail = s.decode_partial(job, sp)
    assert (ctext, cwords) == ("", None) and tail == f"heard {RATE} samples"
    assert h.fed == [RATE] and stt.transcribes == 0
    fr = s.apply_partial(job, tail, 3)
    assert fr["committed"] == "" and fr["tail"] == tail and fr["upto"] == RATE
    # more audio before the cadence elapsed: not due; after it: only the new audio is fed
    s.push(RATE, _speech(320))
    assert not s.partial_due()
    t[0] += 0.2
    assert s.partial_due()
    job2 = s.take_partial()
    assert len(job2.tail) == 320 and job2.tail_start == 0 and job2.upto == RATE + 320
    s.decode_partial(job2, sp)
    assert h.fed == [RATE, 320] and h.total == RATE + 320


def test_streaming_withdrawn_partial_leaves_its_audio_for_the_final():
    stt = FakeStreamingStt()
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=h)
    sp = _Speaker(stt)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    s.decode_partial(s.take_partial(now=1.0), sp)  # RATE fed
    for i in range(RATE, 2 * RATE, 320):
        s.push(i, _speech(320))
    withdrawn = s.take_partial(now=2.0)  # would feed the second second ...
    assert len(withdrawn.tail) == RATE
    s.release_partial(withdrawn)  # ... but the lane never ran it
    fin = s.take_final("f1")
    assert len(fin.tail) == RATE and fin.upto == 2 * RATE  # the final feeds it, once
    text, _ = s.decode_final(fin, sp)
    assert h.total == 2 * RATE and text == f"final {2 * RATE} samples"


def test_streaming_final_from_finalize_then_fresh_utterance():
    stt = FakeStreamingStt()
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=h)
    sp = _Speaker(stt)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    s.decode_partial(s.take_partial(now=1.0), sp)  # RATE fed
    for i in range(RATE, RATE + 4800, 320):
        s.push(i, np.zeros(320, "<i2"))
    fin = s.take_final("f1")
    assert fin.streaming and len(fin.tail) == 4800 and fin.tail_start == 0
    assert fin.upto == RATE + 4800
    text, words = s.decode_final(fin, sp)
    assert h.finals == 1 and h.total == RATE + 4800 and stt.transcribes == 0
    assert text == f"final {RATE + 4800} samples" and words[0][0] == "final"
    s.push(RATE + 4800, _speech(320))  # arrives before apply: belongs to the next utterance
    frame = s.apply_final(fin, text, words, 20)
    assert frame["id"] == "f1" and frame["text"] == text and frame["upto"] == RATE + 4800
    # word timings are relative to the utterance start (mic index 0 here)
    assert frame["words"][0][1] == 0 and frame["words"][1][2] == RATE + 4800
    assert h.resets == 2 and h.fed == [] and s.tail_start == RATE + 4800
    # the next partial feeds only what came after upto
    job = s.take_partial(now=2.0)
    assert len(job.tail) == 320 and job.tail_start == RATE + 4800
    s.decode_partial(job, sp)
    assert h.fed == [320]


def test_streaming_word_offsets_follow_the_utterance_start():
    stt = FakeStreamingStt()
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=h)
    sp = _Speaker(stt)
    pts = 0
    for _ in range(0, RATE, 320):
        s.push(pts, _speech(320))
        pts += 320
    frame = s.apply_final(*_final(s, sp), 1)
    assert frame["upto"] == RATE
    for _ in range(0, RATE // 2, 320):
        s.push(pts, _speech(320))
        pts += 320
    fin, text, words = _final(s, sp)
    assert fin.tail_start == RATE
    frame = s.apply_final(fin, text, words, 1)
    assert frame["words"][0][1] == RATE  # second utterance's first word starts at its start


def _final(s, sp, fid=None):
    fin = s.take_final(fid)
    text, words = s.decode_final(fin, sp)
    return fin, text, words


def test_streaming_reset_and_gap_reset_the_handle():
    stt = FakeStreamingStt()
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=h)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    s.decode_partial(s.take_partial(now=1.0), _Speaker(stt))
    s.reset_utterance()
    assert h.resets == 2 and h.fed == [] and s.tail_start == RATE and s._fed == RATE
    s.push(RATE, _speech(320))
    job = s.take_partial(now=2.0)
    assert len(job.tail) == 320
    # a > 5 s gap resets everything, the handle included, and feeding resumes at pts
    s.push(RATE + 320 + 6 * RATE, _speech(320))
    assert h.resets == 3 and s._fed == RATE + 320 + 6 * RATE
    s.close_stream()
    assert h.closed and s.stream is None


def test_streaming_final_without_timings_falls_back_to_transcribe_words():
    stt = FakeStreamingStt(words=False, fallback=True)
    h = stt.create_stream()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=h)
    sp = _Speaker(stt)
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    text, words = s.decode_final(s.take_final(), sp)
    assert h.finals == 1 and stt.transcribes == 1
    assert text == f"fallback {RATE}" and words == [("fallback", 0.0, 1.0)]


def test_streaming_final_without_timings_and_no_fallback_sends_empty_words():
    stt = FakeStreamingStt(words=False)
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0, stream=stt.create_stream())
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    text, words = s.decode_final(s.take_final(), _Speaker(stt))
    assert text.startswith("final") and words == [] and stt.transcribes == 0


def test_streaming_long_utterance_keeps_the_start_and_bounds_the_buffer():
    stt = FakeStreamingStt()
    s = RollingStt(
        vad=EnergyVad().stream(), clock=lambda: 0.0, stream=stt.create_stream(), max_utterance_s=5
    )
    pts = 0
    for _ in range(0, 8 * RATE, 320):
        s.push(pts, _speech(320))
        pts += 320
    assert s.tail_start == 0 and s._buf_start >= 2 * RATE  # start kept, buffer cut
    fin = s.take_final()
    assert fin.tail_start == 0 and len(fin.tail) <= 6 * RATE


def test_parakeet_rolling_path_is_unchanged():
    stt = FakeStt()
    s = RollingStt(vad=EnergyVad().stream(), clock=lambda: 0.0)
    assert not s.streaming
    for i in range(0, RATE, 320):
        s.push(i, _speech(320))
    job = s.take_partial(now=1.0)
    assert not job.streaming and job.commit is None
    assert s.decode_partial(job, _Speaker(stt)) == ("", None, f"heard {RATE} samples")
    fin = s.take_final("x")
    assert not fin.streaming and len(fin.tail) == RATE
    text, words = s.decode_final(fin, _Speaker(stt))
    assert text == f"final {RATE} samples" and words[0][0] == "final"
    assert stt.calls == 2


# ----------------------------------------------------------- end to end
def test_realtime_session_streams_partials_and_finals(config, studioforge):
    stt = FakeStreamingStt()
    app, _, _, _ = build(config, studioforge, stt=stt)
    with TestClient(app) as tc, connect(tc) as ws:
        ready = hello(ws)
        assert "lock_key" not in ready["engines"]["stt"] and "sha8" in ready["engines"]["stt"]
        assert len(stt.handles) == 1  # one stream per session, opened at hello
        for f in speech(1.0):
            ws.send_bytes(f)
            time.sleep(0.016)
        part, _, _ = until(ws, "stt.partial")
        assert part["tail"].startswith("heard") and part["committed"] == ""
        for f in silence(0.4, 16000):
            ws.send_bytes(f)
            time.sleep(0.016)
        ws.send_text(json.dumps({"type": "stt.flush", "id": "f1"}))
        fin, _, _ = until(ws, "stt.final")
        assert fin["id"] == "f1" and fin["text"].startswith("final")
        assert fin["upto"] == 16000 + int(0.4 * 16000) and fin["words"][0][0] == "final"
        h = stt.handles[0]
        assert h.finals == 1 and h.total == 0 and stt.transcribes == 0
        # the whole utterance went through the handle exactly once
        ws.send_text(json.dumps({"type": "bye"}))
    assert h.closed


# ---------------------------------------------------------------- options
def _cfg(**kw):
    return Config(data_dir="x", stt={"engine": "nemotron", "nemotron": kw}).stt.nemotron


def test_options_auto_backend_per_platform():
    o = nemotron.resolve_options(_cfg(), provider="cuda", device_id=3, platform="win32")
    assert o.backend == "subprocess" and o.chunk_ms == 160 and o.device_id == 3
    o = nemotron.resolve_options(_cfg(), provider="cuda", device_id=0, platform="linux")
    assert o.backend == "sherpa-onnx"
    o = nemotron.resolve_options(_cfg(), provider="cpu", device_id=0, platform="win32")
    assert o.backend == "sherpa-onnx" and o.revision == nemotron.NEMOTRON_REVISION
    o = nemotron.resolve_options(_cfg(quantization="int8"), provider="cpu", device_id=0)
    assert o.quantization == "int8" and o.revision == nemotron.NEMOTRON_INT8_REVISION


def test_options_refuse_the_dll_clash_and_stray_keys(tmp_path):
    with pytest.raises(nemotron.NemotronConfigError, match="CUDA 12"):
        nemotron.resolve_options(
            _cfg(backend="sherpa-onnx"), provider="cuda", device_id=3, platform="win32"
        )
    with pytest.raises(nemotron.NemotronConfigError, match="python"):
        nemotron.resolve_options(
            _cfg(backend="sherpa-onnx", python=sys.executable), provider="cpu", device_id=0
        )
    with pytest.raises(nemotron.NemotronConfigError, match="does not exist"):
        nemotron.resolve_options(
            _cfg(backend="subprocess", python=str(tmp_path / "nope")), provider="cpu", device_id=0
        )
    with pytest.raises(Exception, match="bogus"):
        _cfg(bogus=1)
    with pytest.raises(Exception, match="chunk_ms"):
        _cfg(chunk_ms=80)


def test_model_files_and_recognizer_kwargs(tmp_path):
    assert nemotron.model_files(160, None) == [
        "tokens.txt",
        "160ms/encoder.onnx",
        "160ms/encoder.data",
        "160ms/decoder.onnx",
        "160ms/joiner.onnx",
    ]
    assert nemotron.model_files(160, "int8")[1:] == [
        "160ms/encoder.int8.onnx",
        "160ms/decoder.int8.onnx",
        "160ms/joiner.int8.onnx",
    ]
    o = nemotron.resolve_options(_cfg(), provider="cpu", device_id=0)
    kw = nemotron.recognizer_kwargs(tmp_path, o)
    assert kw["encoder"].endswith("encoder.onnx") and kw["joiner"].endswith("joiner.onnx")
    assert kw["tokens"].endswith("tokens.txt") and kw["tail_pad_ms"] == 480
    cfg = Config(data_dir=tmp_path, stt={"engine": "nemotron"})
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        nemotron.load(cfg, 0)


def test_lock_group_and_registry():
    cfg = Config(data_dir="x", stt={"engine": "nemotron"})
    assert models.groups_for(cfg)[1] == "stt-nemotron"
    need = {f["dest"] for f in models.files_for(cfg)}
    assert "nemotron-speech-streaming-en-0.6b/160ms/encoder.data" in need
    assert "nemotron-speech-streaming-en-0.6b/tokens.txt" in need
    assert not any("parakeet" in d or "int8" in d for d in need)
    cfg8 = Config(data_dir="x", stt={"engine": "nemotron", "nemotron": {"quantization": "int8"}})
    assert models.groups_for(cfg8)[1] == "stt-nemotron-int8"
    need8 = {f["dest"] for f in models.files_for(cfg8)}
    assert "nemotron-speech-streaming-en-0.6b/160ms/encoder.int8.onnx" in need8
    assert "nemotron-speech-streaming-en-0.6b/tokens.txt" in need8
    lock = models.load_lock()
    entry = lock["models"]["nemotron-speech-streaming-en-0.6b"]
    assert entry["revision"] == nemotron.NEMOTRON_REVISION
    assert entry["revision_int8"] == nemotron.NEMOTRON_INT8_REVISION
    assert "NVIDIA Open Model License" in entry["licence"]
    for f in lock["files"]:
        if f["dest"].startswith("nemotron-"):
            assert f["url"].split("/resolve/")[1].split("/")[0] in (
                nemotron.NEMOTRON_REVISION,
                nemotron.NEMOTRON_INT8_REVISION,
            )
    assert models.groups_for(Config(data_dir="x"))[1] == "stt-fp32"  # Parakeet default
    assert models.engine_shas(cfg)["nemotron-speech-streaming-en-0.6b"]
    base.builtin()
    assert base.STT_ENGINES["nemotron"] is nemotron.load and "parakeet" in base.STT_ENGINES


def test_words_from_sentencepiece_tokens():
    w = nemotron_worker.words_from_sp_tokens(
        ["▁He", "y", ",", "▁wel", "come", "▁back"], [0.0, 0.16, 0.24, 0.48, 0.56, 0.8]
    )
    assert w == [("Hey,", 0.0, 0.48), ("welcome", 0.48, 0.8), ("back", 0.8, 0.88)]
    assert nemotron_worker.words_from_sp_tokens([], []) == []


# ------------------------------------------------------------ the worker
class FakeRecognizer:
    def __init__(self, **kw):
        self.kw = kw
        self.streams: dict[int, list[int]] = {}
        self.n = 0

    def create(self):
        self.n += 1
        self.streams[self.n] = []
        return self.n

    def feed(self, sid, audio):
        self.streams[sid].append(len(audio))
        return f"fed {sum(self.streams[sid])}"

    def finalize(self, sid):
        total = sum(self.streams[sid])
        return f"final {total}", [("final", 0.0, total / 16000)]

    def reset(self, sid):
        self.streams[sid] = []

    def close_stream(self, sid):
        del self.streams[sid]

    def describe(self):
        return {"fake": True, "provider": self.kw.get("provider")}

    def close(self):
        pass


def _cmd(op, **kw):
    return (json.dumps({"op": op, **kw}) + "\n").encode()


def test_worker_protocol_in_memory():
    audio = np.arange(1600, dtype="<f4")
    src = io.BytesIO(
        _cmd("create")
        + _cmd("load", tokens="t", encoder="e", decoder="d", joiner="j", provider="cuda")
        + _cmd("create")
        + _cmd("feed", sid=1, n=1600)
        + audio.tobytes()
        + _cmd("feed", sid=1, n=0)
        + _cmd("finalize", sid=1)
        + _cmd("reset", sid=1)
        + _cmd("feed", sid=1, n=0)
        + _cmd("feed", sid=9, n=0)
        + _cmd("close_stream", sid=1)
        + _cmd("bogus")
        + _cmd("close")
    )
    dst = io.BytesIO()
    logs: list[str] = []
    rc = nemotron_worker.serve(src, dst, make=lambda **kw: FakeRecognizer(**kw), log=logs.append)
    assert rc == 0
    replies = [json.loads(line) for line in dst.getvalue().decode().splitlines()]
    assert replies[0] == {"error": "not loaded", "type": "RuntimeError"}
    assert replies[1] == {"ok": True, "describe": {"fake": True, "provider": "cuda"}}
    assert replies[2] == {"sid": 1}
    assert replies[3] == {"text": "fed 1600"} and replies[4] == {"text": "fed 1600"}
    assert replies[5] == {"text": "final 1600", "words": [["final", 0.0, 0.1]]}
    assert replies[6] == {"ok": True} and replies[7] == {"text": "fed 0"}
    assert replies[8]["type"] == "KeyError" and logs and "feed failed" in logs[0]
    assert replies[9] == {"ok": True} and replies[10]["type"] == "ProtocolError"
    assert replies[11] == {"ok": True}


def test_worker_protocol_bounds_the_feed_frame():
    """A frame count the child would have to allocate blindly is a protocol
    error (exit 2), and the parent refuses it before writing to the pipe."""
    for n in (nemotron_worker.MAX_FEED_SAMPLES + 1, -1, "lots"):
        src = io.BytesIO(_cmd("load") + _cmd("feed", sid=1, n=n))
        dst = io.BytesIO()
        assert nemotron_worker.serve(src, dst, make=lambda **kw: FakeRecognizer(**kw)) == 2
        last = json.loads(dst.getvalue().decode().splitlines()[-1])
        assert last["type"] == "ProtocolError" and "bounds" in last["error"]
    # a frame within the bound is still a feed
    src = io.BytesIO(
        _cmd("load") + _cmd("create") + _cmd("feed", sid=1, n=4) + bytes(16) + _cmd("close")
    )
    dst = io.BytesIO()
    assert nemotron_worker.serve(src, dst, make=lambda **kw: FakeRecognizer(**kw)) == 0

    class NoPipe:
        def poll(self):
            return None

    be = nemotron.SubprocessBackend.__new__(nemotron.SubprocessBackend)
    be._lock = threading.Lock()
    be._proc = NoPipe()
    with pytest.raises(nemotron.WorkerError, match="exceeds"):
        be.feed(1, np.zeros(nemotron_worker.MAX_FEED_SAMPLES + 1, np.float32))


def test_worker_protocol_short_read_exits_2():
    src = io.BytesIO(_cmd("load") + _cmd("feed", sid=1, n=100) + b"\x00" * 8)
    dst = io.BytesIO()
    assert nemotron_worker.serve(src, dst, make=lambda **kw: FakeRecognizer(**kw)) == 2
    assert json.loads(dst.getvalue().decode().splitlines()[-1])["type"] == "ProtocolError"


STUB = """
import sys, time
from speechforge.engines.nemotron_worker import serve
from tests.test_nemotron import FakeRecognizer

class Slow(FakeRecognizer):
    def feed(self, sid, audio):
        if len(audio) == 7:
            time.sleep(30)
        return super().feed(sid, audio)

sys.exit(serve(sys.stdin.buffer, sys.stdout.buffer, make=lambda **kw: Slow(**kw)))
"""


@pytest.mark.timeout(60)
def test_subprocess_backend_round_trip_and_timeout(tmp_path, monkeypatch):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(STUB, encoding="utf-8")
    monkeypatch.setattr(nemotron.SubprocessBackend, "WORKER", stub)
    # the child needs this checkout importable (-I drops PYTHONPATH; the venv's site stays)
    monkeypatch.setattr(
        nemotron.subprocess,
        "Popen",
        _popen_with_path(nemotron.subprocess.Popen),
    )
    opts = nemotron.resolve_options(
        _cfg(backend="subprocess", worker_timeout_s=5), provider="cpu", device_id=0
    )
    kw = nemotron.recognizer_kwargs(tmp_path, opts)
    be = nemotron.SubprocessBackend(kw, python=sys.executable, timeout_s=5)
    try:
        assert be.describe() == {"fake": True, "provider": "cpu"}
        stt = nemotron.NemotronStt(be, opts)
        d = stt.describe()
        assert d["backend"] == "subprocess" and d["chunk_ms"] == 160 and d["device"] == 0
        assert d["revision"] == nemotron.NEMOTRON_REVISION[:8] and d["streaming"]
        s = stt.create_stream()
        assert s.feed(np.zeros(1600, np.float32)) == "fed 1600"
        assert s.feed(np.zeros(0, np.float32)) == "fed 1600"
        text, words = s.finalize()
        assert text == "final 1600" and words == [("final", 0.0, 0.1)]
        s.reset()
        assert s.feed(np.zeros(16, np.float32)) == "fed 16"
        s.close()
        assert stt.transcribe_words(np.zeros(3200, np.float32)) == (
            "final 3200",
            [("final", 0.0, 0.2)],
        )
        assert stt.transcribe(np.zeros(100, np.float32)) == ""  # under 0.1 s: silence
        with pytest.raises(nemotron.WorkerError, match="within"):
            be.feed(be.create(), np.zeros(7, np.float32))
        assert be._proc.poll() is not None  # killed
        with pytest.raises(nemotron.WorkerError, match="exited"):
            be.create()
    finally:
        be.close()


def _popen_with_path(real):
    import os
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])

    def popen(args, **kw):
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + str(Path(root) / "src")
        # -I ignores PYTHONPATH; drop it for the stub so the checkout is importable
        args = [a for a in args if a != "-I"]
        return real(args, env=env, **kw)

    return popen
