"""The Qwen3-TTS adapter on a CPU with no torch: voice parsing, the D3 refusal,
streaming / stop / credit gate, stats, registration and the lock groups."""

from __future__ import annotations

import io
import threading
import time
from typing import Any

import numpy as np
import pytest

from speechforge import models
from speechforge.config import QWEN3_TTS_REVISIONS, Config
from speechforge.engines import base, qwen3_tts
from speechforge.engines.qwen3_tts import (
    MAX_REF_S,
    UNIT_SAMPLES,
    Qwen3Settings,
    Qwen3Tts,
    RefVoice,
    build_voice_bytes,
    chunk_steps_for,
    parse_voice,
)
from speechforge.errors import BadRequest, ConfigError

RATE = 24000


def clip(seconds: float = 3.0, rate: int = RATE) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def voice_bytes(seconds: float = 3.0, rate: int = RATE, transcript: str = "Hello there.") -> bytes:
    return build_voice_bytes(clip(seconds, rate), rate, transcript)


def npz(**arrays: Any) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class FakeBackend:
    """Yields ``pieces`` chunks of ``chunk_size`` codec steps (83 ms each) as
    float32 audio, with faster-qwen3-tts's timing dict shape."""

    sample_rate = RATE

    def __init__(self, *, pieces: int = 4, step_s: float = 0.0) -> None:
        self.pieces, self.step_s = pieces, step_s
        self.prompt_calls: list[tuple[int, int, str, bool]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.render_calls: list[dict[str, Any]] = []
        self.closed_generators = 0
        self.warmups = 0
        self.closed = False

    def clone_prompt(self, audio, rate, transcript, *, xvec_only):
        self.prompt_calls.append((len(audio), rate, transcript, xvec_only))
        return {"prompt_for": len(audio), "xvec": xvec_only}

    def stream(self, text, *, language, prompt, transcript, chunk_size, seed, sampling):
        self.stream_calls.append(
            {
                "text": text,
                "language": language,
                "prompt": prompt,
                "transcript": transcript,
                "chunk_size": chunk_size,
                "seed": seed,
                "sampling": sampling,
            }
        )
        return self._gen(chunk_size)

    def _gen(self, chunk_size):
        samples = chunk_size * RATE // 12
        try:
            for i in range(self.pieces):
                if self.step_s:
                    time.sleep(self.step_s)
                yield (
                    np.full(samples, 0.1 * (i + 1), np.float32),
                    RATE,
                    {
                        "chunk_index": i,
                        "chunk_steps": chunk_size,
                        "prefill_ms": 12.5 if i == 0 else 0,
                        "total_steps_so_far": chunk_size * (i + 1),
                        "is_final": i == self.pieces - 1,
                    },
                )
        finally:
            self.closed_generators += 1

    def render(self, text, *, language, prompt, transcript, seed, sampling):
        self.render_calls.append({"text": text, "seed": seed, "prompt": prompt})
        return np.full(RATE, 0.25, np.float32)

    def warmup(self):
        self.warmups += 1

    def describe(self):
        return {"library": "fake"}

    def close(self):
        self.closed = True


def adapter(**kw) -> tuple[Qwen3Tts, FakeBackend]:
    be = FakeBackend(**{k: v for k, v in kw.items() if k in ("pieces", "step_s")})
    settings = kw.get("settings") or Qwen3Settings()
    return Qwen3Tts(be, settings=settings, revision="abc"), be


# ------------------------------------------------------------------ voice format
def test_voice_roundtrip_keeps_clip_rate_and_transcript():
    v = parse_voice(voice_bytes(3.0, 16000, "  Hello there. "))
    assert isinstance(v, RefVoice)
    assert v.rate == 16000 and abs(v.seconds - 3.0) < 1e-6
    assert v.transcript == "Hello there." and not v.xvec_only
    assert v.audio.dtype == np.float32 and np.abs(v.audio).max() <= 1.0
    assert len(voice_bytes(20.0)) < 2 * 1024 * 1024  # fits the voices.max_bytes upload cap


def test_float32_audio_and_empty_transcript_are_accepted():
    data = npz(audio=clip(2.0), rate=np.int64(RATE), transcript=np.array(""))
    v = parse_voice(data)
    assert v.xvec_only and v.seconds == 2.0


def _t(**kw):
    arrays = {"audio": clip(2.0), "rate": np.int64(RATE), "transcript": np.array("x")}
    arrays.update(kw)
    return npz(**arrays)


BAD_VOICES = {
    "too long": lambda: voice_bytes(MAX_REF_S + 0.5),
    "too short": lambda: voice_bytes(0.5),
    "float64 audio": lambda: _t(audio=clip(2.0).astype(np.float64)),
    "int32 audio": lambda: _t(audio=(clip(2.0) * 32767).astype(np.int32)),
    "missing transcript": lambda: npz(audio=clip(2.0), rate=np.int64(RATE)),
    "missing rate": lambda: npz(audio=clip(2.0), transcript=np.array("x")),
    "missing audio": lambda: npz(rate=np.int64(RATE), transcript=np.array("x")),
    "stereo": lambda: _t(audio=np.stack([clip(2.0), clip(2.0)])),
    "rate too low": lambda: _t(rate=np.int64(4000)),
    "float rate": lambda: _t(rate=np.float64(RATE)),
    "out of range": lambda: _t(audio=clip(2.0) * 40),
    "nan": lambda: _t(audio=np.full(RATE * 2, np.nan, np.float32)),
    "transcript long": lambda: _t(transcript=np.array("x" * 1001)),
    "control chars": lambda: _t(transcript=np.array("bad\x00text")),
    "two transcripts": lambda: _t(transcript=np.array(["a", "b"])),
    "other format": lambda: _t(format=np.array("other")),
    "junk": lambda: b"not a zip at all",
    "truncated": lambda: voice_bytes()[:200],
}


@pytest.mark.parametrize("why", list(BAD_VOICES))
def test_parse_voice_rejects(why):
    with pytest.raises(BadRequest) as exc:
        parse_voice(BAD_VOICES[why]())
    assert exc.value.code == "bad_voice", why


def test_parse_voice_rejects_oversize_bytes():
    big = npz(audio=np.zeros(RATE * 20, np.float32), rate=np.int64(RATE), transcript=np.array("x"))
    assert len(big) > 1024 * 1024
    with pytest.raises(BadRequest):
        parse_voice(big + b"\0" * (qwen3_tts.MAX_VOICE_BYTES + 1))


def test_parse_voice_refuses_pickles():
    buf = io.BytesIO()
    np.savez(
        buf, audio=np.array([{"a": 1}], dtype=object), rate=np.int64(RATE), transcript=np.array("x")
    )
    with pytest.raises(BadRequest):
        parse_voice(buf.getvalue())


def test_build_voice_cuts_an_untranscribed_clip_and_refuses_a_transcribed_long_one():
    tts, _ = adapter()
    v = parse_voice(tts.build_voice(clip(30.0), RATE))
    assert v.seconds == MAX_REF_S and v.xvec_only
    with pytest.raises(BadRequest):
        tts.build_voice(clip(30.0), RATE, "too long to match")
    v2 = parse_voice(tts.build_voice(clip(4.0), RATE, "Four seconds."))
    assert v2.transcript == "Four seconds."


# ------------------------------------------------------------------ D3 refusal and registry
def test_load_refuses_without_the_reference_clip_decision(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)  # any torch import would raise ImportError
    cfg = Config(data_dir=tmp_path, tts={"engine": "qwen3-tts"})
    assert cfg.voices.allow_reference_clip is False
    with pytest.raises(ConfigError) as exc:
        qwen3_tts.load(cfg, 3)
    assert "allow_reference_clip" in str(exc.value) and "D3" in str(exc.value)


def test_load_past_d3_checks_the_model_files_before_importing_torch(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    cfg = Config(
        data_dir=tmp_path, tts={"engine": "qwen3-tts"}, voices={"allow_reference_clip": True}
    )
    with pytest.raises(FileNotFoundError) as exc:
        qwen3_tts.load(cfg, 3)
    assert "qwen3-tts-1.7b" in str(exc.value)
    d = tmp_path / "models" / "qwen3-tts-0.6b"
    (d / "speech_tokenizer").mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"x")
    (d / "speech_tokenizer" / "model.safetensors").write_bytes(b"x")
    (d / "pytorch_model.bin").write_bytes(b"x")
    cfg = Config(
        data_dir=tmp_path,
        tts={"engine": "qwen3-tts", "qwen3": {"model": "0.6b"}},
        voices={"allow_reference_clip": True},
    )
    with pytest.raises(RuntimeError, match="pickle"):
        qwen3_tts.load(cfg, 3)


def test_builtin_registers_the_engine_and_its_parser():
    base.builtin()
    assert base.TTS_ENGINES["qwen3-tts"] is qwen3_tts.load
    assert base.VOICE_PARSERS["qwen3-tts"] is qwen3_tts.parse_voice
    assert base.VOICE_PARSERS["chatterbox"] is not qwen3_tts.parse_voice


def test_adapter_satisfies_the_protocol():
    tts, _ = adapter()
    assert isinstance(tts, base.TtsAdapter)
    assert (
        tts.name == "qwen3-tts" and tts.sample_rate == 24000 and tts.voice_format == "qwen3tts-ref"
    )


def test_backend_rate_must_be_24k():
    be = FakeBackend()
    be.sample_rate = 16000
    with pytest.raises(RuntimeError):
        Qwen3Tts(be)


# ------------------------------------------------------------------ config and lock
def test_config_knobs_and_unknown_keys():
    cfg = Config(
        data_dir="x", tts={"engine": "qwen3-tts", "qwen3": {"chunk_size": 4, "model": "0.6b"}}
    )
    assert cfg.tts.qwen3.chunk_size == 4 and cfg.tts.qwen3.model == "0.6b"
    with pytest.raises(Exception):  # noqa: B017 - pydantic validation error, any flavour
        Config(data_dir="x", tts={"qwen3": {"chunk": 4}})
    with pytest.raises(Exception):  # noqa: B017
        Config(data_dir="x", tts={"qwen3": {"model": "3b"}})


def test_lock_groups_follow_the_engine_choice():
    cfg = Config(data_dir="x", tts={"engine": "qwen3-tts"})
    assert models.tts_group_for(cfg) == "tts-qwen3"
    need = {f["dest"] for f in models.files_for(cfg)}
    assert "qwen3-tts-1.7b/model.safetensors" in need
    assert "qwen3-tts-1.7b/speech_tokenizer/model.safetensors" in need
    assert "qwen3-tts-1.7b/config.json" in need and "qwen3-tts-1.7b/vocab.json" in need
    assert not any(d.startswith("chatterbox") for d in need)
    assert "parakeet-tdt-0.6b-v2/encoder-model.onnx.data" in need
    small = Config(data_dir="x", tts={"engine": "qwen3-tts", "qwen3": {"model": "0.6b"}})
    assert models.tts_group_for(small) == "tts-qwen3-0.6b"
    assert "qwen3-tts-0.6b/model.safetensors" in {f["dest"] for f in models.files_for(small)}
    plain = Config(data_dir="x")
    assert models.tts_group_for(plain) == "tts-fp32"
    assert not any("qwen3" in f["dest"] for f in models.files_for(plain))


def test_lock_entries_are_pinned_to_the_configured_revisions():
    lock = models.load_lock()
    for size, name in (("1.7b", "qwen3-tts-1.7b"), ("0.6b", "qwen3-tts-0.6b")):
        rev = QWEN3_TTS_REVISIONS[size]
        assert lock["models"][name]["revision"] == rev
        assert "Apache-2.0" in lock["models"][name]["licence"]
        files = [f for f in lock["files"] if f["dest"].startswith(name + "/")]
        assert len(files) == 11
        assert all(f"/resolve/{rev}/" in f["url"] for f in files)
        assert all(f["url"].endswith(f["dest"][len(name) + 1 :]) for f in files)
        assert not any(f["dest"].endswith(s) for f in files for s in qwen3_tts.PICKLE_SUFFIXES)
        assert any(f["dest"].endswith("model.safetensors") and f["size"] > 10**9 for f in files)


def test_config_example_documents_the_qwen3_block():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config.example.yaml").read_text())
    from speechforge.config import Qwen3TtsConfig

    assert set(raw["tts"]["qwen3"]) == set(Qwen3TtsConfig.model_fields)


# ------------------------------------------------------------------ streaming
def test_stream_yields_int16_chunks_and_reports_stats():
    tts, be = adapter(pieces=3)
    v = parse_voice(voice_bytes(3.0, 16000, "Hello there."))
    pieces = list(tts.stream("Say this.", "v1", v, seed=7))
    assert len(pieces) == 3 and all(p.dtype == np.dtype("<i2") for p in pieces)
    assert be.stream_calls[0]["seed"] == 7
    assert be.stream_calls[0]["chunk_size"] == 8 and be.stream_calls[0]["language"] == "English"
    assert be.stream_calls[0]["transcript"] == "Hello there."
    assert be.stream_calls[0]["sampling"]["top_k"] == 50
    st = tts.last_stats
    assert st["chunks"] == 3 and st["tokens"] == 24 and st["finished"] is True
    assert st["audio_ms"] == round(3 * 8 * 1000 / 12) and st["ttfa_ms"] is not None
    assert st["first_chunk_ms"] == st["ttfa_ms"] and st["prefill_ms"] == 12  # rounded 12.5
    assert st["rtf"] is not None and st["rtf"] >= 0 and st["chunk_size"] == 8
    assert be.closed_generators == 1


def test_prompt_is_cached_per_voice_id_and_evicted_lru():
    tts, be = adapter(settings=Qwen3Settings(max_prepared=2))
    v = parse_voice(voice_bytes())
    list(tts.stream("One.", "a", v))
    list(tts.stream("Two.", "a", v))
    assert len(be.prompt_calls) == 1
    list(tts.stream("Three.", "b", v))
    list(tts.stream("Four.", "c", v))  # evicts "a"
    list(tts.stream("Five.", "a", v))
    assert len(be.prompt_calls) == 4
    assert be.prompt_calls[0][2] == "Hello there." and be.prompt_calls[0][3] is False


def test_empty_transcript_and_xvec_setting_select_speaker_embedding_mode():
    tts, be = adapter()
    v = parse_voice(voice_bytes(transcript=""))
    list(tts.stream("Hi.", "a", v))
    assert be.prompt_calls[-1][3] is True
    forced, be2 = adapter(settings=Qwen3Settings(xvec_only=True))
    list(forced.stream("Hi.", "a", parse_voice(voice_bytes(transcript="Spoken text."))))
    assert be2.prompt_calls[-1][3] is True and be2.stream_calls[-1]["transcript"] == ""


def test_stop_between_chunks_closes_the_backend_generator():
    tts, be = adapter(pieces=100)
    v = parse_voice(voice_bytes())
    stop = threading.Event()
    gen = tts.stream("A long sentence.", "a", v, stop=stop)
    first = next(gen)
    assert len(first) > 0
    stop.set()
    rest = list(gen)
    assert rest == [] and be.closed_generators == 1
    assert tts.last_stats["chunks"] == 1 and tts.last_stats["finished"] is False
    assert not tts._gen_lock.locked()


def test_consumer_leaving_releases_the_engine_lock():
    tts, be = adapter(pieces=100)
    v = parse_voice(voice_bytes())
    gen = tts.stream("A long sentence.", "a", v)
    next(gen)
    gen.close()
    assert be.closed_generators == 1 and not tts._gen_lock.locked()
    assert list(tts.stream("Next.", "a", v))  # the next utterance proceeds


def test_credit_gate_holds_the_next_chunk_and_counts_40ms_units():
    tts, be = adapter(pieces=3)
    v = parse_voice(voice_bytes())
    seen: list[int] = []
    allow = threading.Event()

    def gate(units: int) -> bool:
        seen.append(units)
        return units == 0 or allow.is_set()

    gen = tts.stream("Gate me.", "a", v, options={"gate": gate})
    first = next(gen)
    produced = len(first) // UNIT_SAMPLES
    got: list[np.ndarray] = []
    th = threading.Thread(target=lambda: got.extend(gen))
    th.start()
    time.sleep(0.05)
    assert got == [] and seen[-1] == produced  # waiting on credit with the right count
    allow.set()
    th.join(timeout=2)
    assert len(got) == 2 and be.closed_generators == 1


def test_gate_stall_ends_on_stop():
    tts, _ = adapter(pieces=3)
    v = parse_voice(voice_bytes())
    stop = threading.Event()
    gen = tts.stream("Gate me.", "a", v, stop=stop, options={"gate": lambda n: n == 0})
    next(gen)
    th = threading.Thread(target=lambda: list(gen))
    th.start()
    time.sleep(0.02)
    stop.set()
    th.join(timeout=2)
    assert not th.is_alive() and not tts._gen_lock.locked()


def test_on_lm_done_gets_the_audio_length_in_units_and_first_chunk_tokens_maps_to_steps():
    tts, be = adapter(pieces=2)
    v = parse_voice(voice_bytes())
    done: list[int] = []
    pieces = list(
        tts.stream("Hi.", "a", v, options={"on_lm_done": done.append, "first_chunk_tokens": 15})
    )
    assert be.stream_calls[0]["chunk_size"] == 7  # 15 x 40 ms = 600 ms -> 7 codec steps
    assert done == [sum(len(p) for p in pieces) // UNIT_SAMPLES]
    assert chunk_steps_for(None, 8) == 8 and chunk_steps_for(4, 8) == 2
    assert chunk_steps_for(100, 8) == 48


def test_on_lm_done_is_skipped_when_stopped():
    tts, _ = adapter(pieces=5)
    v = parse_voice(voice_bytes())
    stop = threading.Event()
    done: list[int] = []
    gen = tts.stream("Hi.", "a", v, stop=stop, options={"on_lm_done": done.append})
    next(gen)
    stop.set()
    list(gen)
    assert done == []


def test_empty_text_yields_nothing_and_touches_no_backend():
    tts, be = adapter()
    v = parse_voice(voice_bytes())
    assert list(tts.stream("   ", "a", v)) == []
    assert be.stream_calls == [] and tts.last_stats["audio_ms"] == 0
    assert len(tts.render("", v)) == 0


def test_render_warm_up_describe_close():
    tts, be = adapter()
    v = parse_voice(voice_bytes())
    out = tts.render("Whole line.", v, seed=3)
    assert out.dtype == np.dtype("<i2") and len(out) == RATE and be.render_calls[0]["seed"] == 3
    tts.warm_up()
    assert be.warmups == 1 and "_warmup" not in tts._prepared
    assert be.stream_calls[-1]["text"] == "Warming up the voice."
    d = tts.describe()
    assert d["name"] == "qwen3-tts" and d["model"].startswith("Qwen/Qwen3-TTS-12Hz-1.7B")
    assert d["revision"] == "abc" and d["dtype"] == "bf16" and d["backend"] == {"library": "fake"}
    tts.close()
    assert be.closed and tts.backend is None and tts._prepared == {}


def test_utterances_serialise_on_the_engine_lock():
    tts, _ = adapter(pieces=2, step_s=0.01)
    v = parse_voice(voice_bytes())
    order: list[str] = []

    def run(tag: str) -> None:
        for _ in tts.stream(tag, "a", v):
            order.append(tag)

    a, b = threading.Thread(target=run, args=("a",)), threading.Thread(target=run, args=("b",))
    a.start()
    b.start()
    a.join()
    b.join()
    assert order in (["a", "a", "b", "b"], ["b", "b", "a", "a"])


def test_speaker_wraps_the_adapter(config):
    from speechforge.pipeline import Speaker
    from tests.conftest import FakeStt

    tts, _ = adapter(pieces=2)
    sp = Speaker(tts, FakeStt())
    vb = sp.build_voice(clip(3.0), RATE)
    v = sp.parse_voice(vb)
    assert v.xvec_only
    assert len(list(sp.speak("Hello.", "x", v, first_chunk_tokens=15))) == 2
    assert sp.last_tts["chunks"] == 2 and sp.describe["tts"]["name"] == "qwen3-tts"
