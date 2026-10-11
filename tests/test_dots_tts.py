"""The dots-tts adapter on a fake backend: voice format, the D3 refusal,
streaming / stop / credit gate, incremental text, stats, the pickle scan, the
48 -> 24 kHz decimator. No torch, no model, no network."""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import threading
import time
import zipfile
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from speechforge.config import Config
from speechforge.engines import base, dots_tts
from speechforge.engines.dots_tts import (
    MAX_VOICE_BYTES,
    NATIVE_RATE,
    PATCH_SAMPLES,
    Decimator,
    DotsTts,
    DotsVoice,
    check_latent_stats,
    make_voice,
    parse_voice,
)
from speechforge.errors import BadRequest, ConfigError


def clip(seconds: float = 3.0, rate: int = 24000, dtype: str = "<i2") -> np.ndarray:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return (x * 32767).astype("<i2") if dtype == "<i2" else x


def npz(**arrays: Any) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class FakeSession:
    def __init__(self, owner: FakeBackend) -> None:
        self.owner = owner
        self.pieces: list[str] = []
        self.finished = False
        self.closed = False

    def push(self, text: str) -> Iterator[np.ndarray]:
        self.pieces.append(text)
        yield self.owner.chunk(len(self.pieces))

    def finish(self) -> Iterator[np.ndarray]:
        self.finished = True
        yield self.owner.chunk(99)

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """``chunks`` patches of PATCH_SAMPLES float32 per synth; a 1 kHz tone so the
    decimator's output can be checked. Records calls, steps and closure."""

    native_rate = NATIVE_RATE

    def __init__(self, chunks: int = 5, *, text_stream: bool = False, step_s: float = 0.0) -> None:
        self.chunks = chunks
        self.text_stream = text_stream
        self.step_s = step_s
        self.calls: list[dict[str, Any]] = []
        self.steps = 0
        self.gen_closed = False
        self.closed = False
        self.sessions: list[FakeSession] = []

    def chunk(self, i: int) -> np.ndarray:
        t = (np.arange(PATCH_SAMPLES) + i * PATCH_SAMPLES) / NATIVE_RATE
        return (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    def synth(self, voice, voice_id, text, *, seed):
        self.calls.append({"voice_id": voice_id, "text": text, "seed": seed, "voice": voice})
        try:
            for i in range(self.chunks):
                if self.step_s:
                    time.sleep(self.step_s)
                self.steps += 1
                yield self.chunk(i)
        finally:
            self.gen_closed = True

    def open_session(self, voice, voice_id, *, seed):
        s = FakeSession(self)
        self.sessions.append(s)
        return s

    def describe(self):
        return {"backend": "fake"}

    def close(self):
        self.closed = True


@pytest.fixture
def voice() -> DotsVoice:
    return parse_voice(make_voice(clip(), 24000, "A short reference sentence."))


# ----------------------------------------------------------------------------- voices
def test_voice_roundtrip_int16_and_float32():
    v = parse_voice(make_voice(clip(2.0), 24000, " Hello there. "))
    assert v.rate == 24000 and v.transcript == "Hello there." and abs(v.seconds - 2.0) < 1e-6
    assert v.audio.dtype == np.float32 and abs(float(np.max(v.audio)) - 0.5) < 0.01
    f = parse_voice(make_voice(clip(2.0, rate=48000, dtype="f4"), 48000))
    assert f.rate == 48000 and f.transcript == "" and abs(f.seconds - 2.0) < 1e-6
    # a (1, N) float32 array is accepted as mono too
    v2 = parse_voice(npz(audio=clip(1.0, dtype="f4")[None, :], rate=np.array(24000)))
    assert abs(v2.seconds - 1.0) < 1e-6
    assert len(v.wav()) > 44 and v.to_npz()[:2] == b"PK"


@pytest.mark.parametrize(
    "data, needle",
    [
        (make_voice(clip(21.0), 24000), "0.5-20.0 s"),
        (make_voice(clip(0.2), 24000), "0.5-20.0 s"),
        (make_voice(clip(1.0, rate=4000), 4000), "outside"),
        (npz(rate=np.array(24000)), "lacks"),
        (npz(audio=clip(1.0), rate=np.array([24000, 1])), "rate"),
        (npz(audio=np.zeros((2, 24000), "<i2"), rate=np.array(24000)), "mono"),
        (npz(audio=np.zeros(24000, np.float64), rate=np.array(24000)), "mono"),
        (npz(audio=np.full(24000, 1.5, np.float32), rate=np.array(24000)), "[-1, 1]"),
        (npz(audio=np.full(24000, np.nan, np.float32), rate=np.array(24000)), "non-finite"),
        (make_voice(clip(1.0), 24000, "x" * 1001), "1000"),
        (make_voice(clip(1.0), 24000, "bad\x00text"), "control"),
        (npz(audio=clip(1.0), rate=np.array(24000), format=np.array("chatterbox-npz")), "format"),
        (b"not a zip at all", "not a valid"),
        (b"x" * (MAX_VOICE_BYTES + 1), "larger"),
    ],
    ids=[f"case{i}" for i in range(14)],
)
def test_parse_voice_rejects(data, needle):
    with pytest.raises(BadRequest) as exc:
        parse_voice(data)
    assert exc.value.code == "bad_voice" and needle in str(exc.value)


def test_parse_voice_refuses_pickled_objects():
    buf = io.BytesIO()
    np.savez(buf, audio=np.array([{"a": 1}], dtype=object), rate=np.array(24000))
    with pytest.raises(BadRequest):
        parse_voice(buf.getvalue())


def test_build_voice_from_clip_alone_truncates_to_20s():
    tts = DotsTts(FakeBackend())
    data = tts.build_voice(clip(25.0, dtype="f4"), 24000)
    v = parse_voice(data)
    assert v.transcript == "" and abs(v.seconds - 20.0) < 1e-6


# ----------------------------------------------------------------------------- registry, config
def test_registered_under_its_name():
    base.builtin()
    assert base.TTS_ENGINES["dots-tts"] is dots_tts.load
    assert base.VOICE_PARSERS["dots-tts"] is dots_tts.parse_voice


def test_load_refuses_without_the_d3_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(dots_tts, "_make_backend", lambda config, dev: FakeBackend())
    config = Config(data_dir=tmp_path, tts={"engine": "dots-tts"})
    with pytest.raises(RuntimeError) as exc:
        dots_tts.load(config, 3)
    assert "allow_reference_clip" in str(exc.value) and "D3" in str(exc.value)
    config = Config(
        data_dir=tmp_path, tts={"engine": "dots-tts"}, voices={"allow_reference_clip": True}
    )
    tts = dots_tts.load(config, 3)
    assert isinstance(tts, DotsTts) and tts.sample_rate == 24000 and tts.name == "dots-tts"
    assert tts.describe()["native_sample_rate"] == 48000
    config = Config(
        data_dir=tmp_path,
        tts={"engine": "dots-tts", "dots": {"output_rate": 48000}},
        voices={"allow_reference_clip": True},
    )
    assert dots_tts.load(config, 3).sample_rate == 48000
    config = Config(
        data_dir=tmp_path,
        tts={"engine": "dots-tts", "dots": {"double_streaming": True, "backend": "http"}},
        voices={"allow_reference_clip": True},
    )
    with pytest.raises(RuntimeError, match="native"):
        dots_tts.load(config, 3)


def test_config_knobs(tmp_path):
    with pytest.raises(ValidationError, match="loopback"):
        Config(data_dir=tmp_path, tts={"dots": {"url": "http://100.64.0.9:8000"}})
    with pytest.raises(ValidationError):
        Config(data_dir=tmp_path, tts={"dots": {"nope": 1}})
    with pytest.raises(ValidationError):
        Config(data_dir=tmp_path, tts={"dots": {"output_rate": 16000}})
    c = Config(data_dir=tmp_path, tts={"dots": {"url": "http://localhost:8000/", "language": " "}})
    assert c.tts.dots.url == "http://localhost:8000" and c.tts.dots.language is None
    assert c.tts.dots.num_steps == 4 and c.tts.dots.backend == "native"
    c = Config(data_dir=tmp_path, tts={"dots": {"num_steps": None, "guidance_scale": None}})
    assert c.tts.dots.num_steps is None


def test_model_groups_follow_the_engine(tmp_path):
    from speechforge import models

    assert "tts-fp32" in models.groups_for(Config(data_dir=tmp_path))
    g = models.groups_for(Config(data_dir=tmp_path, tts={"engine": "dots-tts"}))
    assert "tts-dots" in g and "tts-fp32" not in g
    g = models.groups_for(
        Config(data_dir=tmp_path, tts={"engine": "dots-tts", "dots": {"double_streaming": True}})
    )
    assert "tts-dots-stts" in g and "tts-dots" not in g
    g = models.groups_for(
        Config(data_dir=tmp_path, tts={"engine": "dots-tts", "dots": {"backend": "http"}})
    )
    assert not any(x.startswith("tts-") for x in g)
    lock = models.load_lock()
    files = [f for f in lock["files"] if "tts-dots" in f["groups"]]
    assert {f["dest"].split("/")[0] for f in files} == {"dots-tts-mf"}
    assert any(f["dest"].endswith("model.safetensors") for f in files)
    assert all(f["url"].count(dots_tts.HF_REVISION) == 1 and len(f["sha256"]) == 64 for f in files)
    stts = [f for f in lock["files"] if "tts-dots-stts" in f["groups"]]
    assert all(dots_tts.HF_STTS_REVISION in f["url"] for f in stts) and stts
    assert lock["models"]["dots.tts-mf"]["revision"] == dots_tts.HF_REVISION


def test_config_example_lists_every_dots_knob():
    import pathlib

    import yaml

    from speechforge.config import DotsTtsConfig

    root = pathlib.Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config.example.yaml").read_text(encoding="utf-8"))
    assert set(raw["tts"]["dots"]) == set(DotsTtsConfig.model_fields)
    Config(**raw)  # the example parses as a whole


# ----------------------------------------------------------------------------- streaming
def test_stream_decimates_and_reports_stats(voice):
    be = FakeBackend(chunks=5)
    tts = DotsTts(be)
    pieces = list(tts.stream("Hello.", "v1", voice, seed=7))
    assert pieces and all(p.dtype == np.dtype("<i2") for p in pieces)
    total = sum(len(p) for p in pieces)
    assert abs(total - 5 * PATCH_SAMPLES // 2) <= 1  # 48 kHz patches -> 24 kHz, nothing lost
    st = tts.last_stats
    assert st["chunks"] == 5 and st["tokens"] == 5 * 4  # 160 ms patches = 4 credit tokens
    assert abs(st["audio_ms"] - 800) <= 1 and st["ttfa_ms"] is not None and st["rtf"] is not None
    assert not st["stopped"] and be.gen_closed
    assert be.calls[0]["seed"] == 7 and be.calls[0]["text"] == "Hello."
    # the 1 kHz tone survives decimation at full amplitude
    out = np.concatenate(pieces).astype(np.float32) / 32767
    assert 0.45 < float(np.max(np.abs(out[200:-200]))) < 0.55


def test_stream_at_native_rate_passes_through(voice):
    tts = DotsTts(FakeBackend(chunks=2), output_rate=48000)
    out = np.concatenate(list(tts.stream("Hi.", "v", voice)))
    assert len(out) == 2 * PATCH_SAMPLES and tts.sample_rate == 48000


def test_stop_between_steps_closes_the_generator(voice):
    be = FakeBackend(chunks=50, step_s=0.002)
    tts = DotsTts(be)
    stop = threading.Event()
    got = 0
    for _ in tts.stream("Long.", "v", voice, stop=stop):
        got += 1
        if got == 3:
            stop.set()
    assert got == 3 and be.steps <= 4 and be.gen_closed and tts.last_stats["stopped"]


def test_credit_gate_holds_the_backend(voice):
    be = FakeBackend(chunks=4)
    tts = DotsTts(be)
    allowed = threading.Event()
    counts: list[int] = []

    def gate(count: int) -> bool:
        counts.append(count)
        return allowed.is_set() or count == 0

    done: list[int] = []
    out: list[np.ndarray] = []

    def run() -> None:
        out.extend(
            tts.stream("Gated.", "v", voice, options={"gate": gate, "on_lm_done": done.append})
        )

    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.05)
    assert be.steps == 1  # the first patch went; the second waits on credit
    allowed.set()
    th.join(timeout=5)
    assert be.steps == 4 and len(out) >= 4
    assert counts[0] == 0 and 4 in counts  # counted in 40 ms tokens: one patch = 4
    assert done == [16]


def test_stop_while_gated(voice):
    be = FakeBackend(chunks=4)
    tts = DotsTts(be)
    stop = threading.Event()

    def gate(count: int) -> bool:
        if count:
            stop.set()
        return count == 0

    pieces = list(tts.stream("x", "v", voice, stop=stop, options={"gate": gate}))
    assert len(pieces) == 1 and tts.last_stats["stopped"] and be.gen_closed


def test_two_contexts_interleave_per_patch(voice):
    be = FakeBackend(chunks=6, step_s=0.003)
    tts = DotsTts(be)
    order: list[str] = []

    def run(tag: str) -> None:
        for _ in tts.stream(tag, tag, voice):
            order.append(tag)

    a, b = threading.Thread(target=run, args=("a",)), threading.Thread(target=run, args=("b",))
    a.start()
    b.start()
    a.join(5)
    b.join(5)
    assert order.count("a") == order.count("b") >= 6  # 6 patches (+ the decimator tail)
    assert order[:6] != ["a"] * 6 and order[:6] != ["b"] * 6  # neither monopolised the lock


def test_text_feed_pushes_pieces_into_one_session(voice):
    be = FakeBackend(text_stream=True)
    tts = DotsTts(be)
    feed = iter(["world, ", "how are you?"])
    pieces = list(tts.stream("Hello ", "v", voice, options={"text_feed": feed}))
    s = be.sessions[0]
    assert s.pieces == ["Hello ", "world, ", "how are you?"] and s.finished and s.closed
    assert len(pieces) >= 4 and be.calls == []
    assert tts.describe()["text_stream"] is True


def test_text_feed_without_sessions_buffers(voice):
    be = FakeBackend(chunks=1)
    tts = DotsTts(be)
    list(tts.stream("Hello ", "v", voice, options={"text_feed": iter(["there", "."])}))
    assert be.calls[0]["text"] == "Hello there." and be.sessions == []


def test_render_warm_up_describe_close(voice):
    be = FakeBackend(chunks=3)
    tts = DotsTts(be)
    pcm = tts.render("Render me.", voice, seed=1)
    assert pcm.dtype == np.dtype("<i2") and abs(len(pcm) - 3 * PATCH_SAMPLES // 2) <= 1
    tts.warm_up()
    assert be.calls[-1]["voice_id"] == "_warmup" and be.calls[-1]["voice"].transcript == ""
    d = tts.describe()
    assert d["engine"] == "dots-tts" and d["revision"] == dots_tts.HF_REVISION
    assert d["voice_format"] == "dotstts-ref" and d["backend"] == "fake"
    tts.close()
    assert be.closed


def test_speaker_speaks_through_the_adapter(voice):
    """The Speaker wrapper passes gate / on_lm_done / seed the same way as to Chatterbox."""
    from speechforge.pipeline import Speaker

    be = FakeBackend(chunks=2)
    sp = Speaker(DotsTts(be), stt=None)
    done: list[int] = []
    out = list(sp.speak("Hi.", "v", voice, seed=3, gate=lambda n: True, on_lm_done=done.append))
    assert out and done == [8] and be.calls[0]["seed"] == 3
    assert sp.last_tts["chunks"] == 2


# ----------------------------------------------------------------------------- decimator
def test_decimator_piecewise_equals_whole_and_filters():
    rate = 48000
    t = np.arange(rate) / rate
    x = (0.5 * np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 20000 * t)).astype(
        np.float32
    )
    whole = Decimator(48000, 24000)
    ref = np.concatenate([whole.process(x), whole.flush()])
    piece = Decimator(48000, 24000)
    parts = [piece.process(c) for c in np.array_split(x, [7, 500, 501, 9000, 30001])]
    parts.append(piece.flush())
    got = np.concatenate(parts)
    assert len(ref) == len(got) == rate // 2
    assert np.allclose(ref, got, atol=1e-5)
    spec = np.abs(np.fft.rfft(ref[1000:13000] * np.hanning(12000)))
    freqs = np.fft.rfftfreq(12000, 1 / 24000)
    tone = spec[np.argmin(np.abs(freqs - 1000))]
    alias = spec[np.argmin(np.abs(freqs - 4000))]  # 20 kHz would fold to 4 kHz
    assert alias < tone / 100
    with pytest.raises(ValueError):
        Decimator(48000, 16000)
    assert Decimator(24000, 24000).process(np.ones(5, np.float32)).tolist() == [1.0] * 5


# ----------------------------------------------------------------------------- pickle scan
def torch_zip(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/data.pkl", payload)
        zf.writestr("archive/version", "3\n")
    return buf.getvalue()


@pytest.mark.parametrize("protocol", [2, 4])
def test_latent_stats_scan_accepts_arrays(tmp_path, protocol):
    stats = {"mean": np.zeros((1, 128), np.float32), "var": np.ones((1, 128), np.float32)}
    p = tmp_path / "latent_stats.pt"
    p.write_bytes(torch_zip(pickle.dumps(stats, protocol=protocol)))
    check_latent_stats(p)
    p.write_bytes(pickle.dumps(stats, protocol=protocol))  # a bare pickle is scanned too
    check_latent_stats(p)


class Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


@pytest.mark.parametrize("protocol", [2, 4])
def test_latent_stats_scan_refuses_code(tmp_path, protocol):
    p = tmp_path / "latent_stats.pt"
    p.write_bytes(torch_zip(pickle.dumps({"mean": Evil()}, protocol=protocol)))
    with pytest.raises(RuntimeError, match="system"):
        check_latent_stats(p)
    p.write_bytes(torch_zip(b"\x80\x02(cbuiltins\neval\n"))
    with pytest.raises(RuntimeError):
        check_latent_stats(p)
    p.write_bytes(b"\x80\x04" + b"\x00" * (dots_tts.MAX_LATENT_STATS_BYTES + 1))
    with pytest.raises(RuntimeError, match="large"):
        check_latent_stats(p)
    with pytest.raises(RuntimeError, match="missing"):
        check_latent_stats(tmp_path / "nope.pt")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("archive/version", "3\n")
    p.write_bytes(buf.getvalue())
    with pytest.raises(RuntimeError, match="no pickle"):
        check_latent_stats(p)


# ----------------------------------------------------------------------------- shim
def test_text_normalizer_shim(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("tn") is not None:
        pytest.skip("WeTextProcessing is installed here")
    for name in [m for m in sys.modules if m == "tn" or m.startswith("tn.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(RuntimeError, match="allow_missing_text_normalizer"):
        dots_tts._normalizer_shim(False)
    assert dots_tts._normalizer_shim(True) is True
    from tn.english.normalizer import Normalizer  # type: ignore[import-not-found]

    with pytest.raises(RuntimeError, match="unavailable"):
        Normalizer().normalize("one")
    for name in (
        "tn",
        "tn.chinese",
        "tn.chinese.normalizer",
        "tn.english",
        "tn.english.normalizer",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_config_error_type_exists():
    assert issubclass(ConfigError, Exception)


# ----------------------------------------------------------------------------- http backend
def test_http_backend_stays_on_the_box_and_streams_pcm(tmp_path, voice):
    """The request carries the reference clip, so the client must ignore proxy
    environment variables and never follow a redirect; the reply is s16le 48 kHz
    yielded per HTTP chunk, with an odd trailing byte carried to the next one."""
    import base64

    import httpx

    config = Config(
        data_dir=tmp_path,
        tts={"engine": "dots-tts", "dots": {"backend": "http", "url": "http://127.0.0.1:8000"}},
        voices={"allow_reference_clip": True},
    )
    be = dots_tts._HttpBackend(config)
    try:
        assert be.client.trust_env is False
        assert be.client.follow_redirects is False
        seen: list[dict] = []
        pcm = (np.arange(-5, 6, dtype="<i2") * 1000).tobytes() + b"\x07"  # 11 samples + stray

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append({"url": str(request.url), "body": body})
            # 3 chunks, one of them cutting a sample in half
            return httpx.Response(200, stream=httpx.ByteStream(pcm))

        be.client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
        out = np.concatenate(list(be.synth(voice, "v1", "Hello.", seed=3)))
        assert len(out) == 11 and out.dtype == np.float32
        assert seen[0]["url"] == "http://127.0.0.1:8000/v1/audio/speech"
        body = seen[0]["body"]
        assert body["stream"] is True and body["response_format"] == "pcm" and body["seed"] == 3
        ref = body["references"][0]
        assert ref["text"] == voice.transcript
        assert ref["audio_path"].startswith("data:audio/wav;base64,")
        assert base64.b64decode(ref["audio_path"].split(",", 1)[1])[:4] == b"RIFF"

        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="loading")

        be.client = httpx.Client(transport=httpx.MockTransport(refuse), trust_env=False)
        with pytest.raises(RuntimeError, match="503"):
            list(be.synth(voice, "v1", "Hello.", seed=None))
    finally:
        be.close()


def test_native_backend_pins_the_hub_offline(monkeypatch, tmp_path):
    """SPEC 8.5: no outbound connection while serving. The native backend sets the
    offline switches before the runtime imports (an operator's own value wins)."""
    import importlib.machinery
    import types

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    # Stop at the torch import: everything before it is what this test checks.
    monkeypatch.setitem(sys.modules, "torch", None)
    tn = types.ModuleType("tn")
    tn.__spec__ = importlib.machinery.ModuleSpec("tn", None)
    monkeypatch.setitem(sys.modules, "tn", tn)
    config = Config(
        data_dir=tmp_path, tts={"engine": "dots-tts"}, voices={"allow_reference_clip": True}
    )
    with pytest.raises(ImportError):
        dots_tts._NativeBackend(config, 0)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"
