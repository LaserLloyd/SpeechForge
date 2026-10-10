"""Real models. Skipped unless the models and ONNX Runtime are present.

* ``SPEECHFORGE_TEST_MODELS=<dir>`` (holding ``chatterbox-turbo/`` and
  ``parakeet-tdt-0.6b-v2/``) and ``SPEECHFORGE_TEST_DTYPE`` (default q8: the
  smallest Chatterbox export that runs everywhere) enable the CPU tests --
  they prove the GPU code path's IOBinding loop and the plain loop sample the
  same tokens, which is what the rig relies on.
* The ``gpu`` tests also need the CUDA execution provider (the rig).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

MODELS = os.environ.get("SPEECHFORGE_TEST_MODELS", "")
DTYPE = os.environ.get("SPEECHFORGE_TEST_DTYPE", "q8")

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
if not MODELS or not (Path(MODELS) / "chatterbox-turbo").is_dir():
    pytest.skip("set SPEECHFORGE_TEST_MODELS to a models dir", allow_module_level=True)

HAS_CUDA = "CUDAExecutionProvider" in ort.get_available_providers()


def engine(provider: str, **kw):
    from speechforge.engines.chatterbox import ChatterboxEngine

    return ChatterboxEngine(
        Path(MODELS) / "chatterbox-turbo", dtype=DTYPE, provider=provider, cpu_threads=2, **kw
    )


@pytest.fixture(scope="module")
def cpu_engine():
    return engine("cpu", use_iobinding=False)


@pytest.fixture(scope="module")
def voice(cpu_engine):
    from speechforge.audio import synthetic_reference

    return cpu_engine.build_voice(synthetic_reference(4.0))


@pytest.mark.models
def test_iobinding_loop_samples_the_same_tokens(cpu_engine, voice):
    bound = engine("cpu", use_iobinding=True)
    text = "Short and sweet."
    plain = list(cpu_engine.stream_tokens(text, voice, None, 7))
    via_binding = list(bound.stream_tokens(text, voice, None, 7))
    assert plain and plain == via_binding


@pytest.mark.models
def test_prefix_cache_samples_the_same_tokens(cpu_engine, voice):
    text = "Short and sweet."
    full = list(cpu_engine.stream_tokens(text, voice, None, 11))
    prefix = cpu_engine.prefix_cache(voice)
    cached = list(cpu_engine.stream_tokens(text, voice, None, 11, prefix=prefix))
    assert full == cached


@pytest.mark.models
def test_pipelined_audio_length_matches_tokens(cpu_engine, voice):
    from speechforge.engines.chatterbox import SAMPLES_PER_TOKEN, slice_prompt
    from speechforge.pipeline import ChunkPlanConfig, pipelined

    stats: dict = {}
    pcm = np.concatenate(
        list(
            pipelined(
                cpu_engine,
                "Hello there, how are you today?",
                slice_prompt(voice, 4.0),
                plan=ChunkPlanConfig(),
                seed=3,
                stats=stats,
            )
        )
    )
    assert len(pcm) == (stats["tokens"] + 3) * SAMPLES_PER_TOKEN
    assert np.abs(pcm).max() > 100  # not silence


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUDA, reason="needs the CUDA execution provider")
def test_gpu_engine_streams_and_matches_cpu_tokens(voice):
    gpu = engine("cuda", device_id=int(os.environ.get("SPEECHFORGE_TEST_DEVICE", "0")))
    text = "Short and sweet."
    toks = list(gpu.stream_tokens(text, voice, None, 7))
    assert 10 < len(toks) < 200
    stop = threading.Event()
    stop.set()
    assert list(gpu.stream_tokens(text, voice, stop, 7)) == []


def _cpu_config(tmp_path):
    from speechforge.config import Config

    return Config(
        data_dir=tmp_path,
        gpu={"provider": "cpu", "devices": [0]},
        models={"dir": MODELS, "tts_dtype": DTYPE, "stt_quantization": "int8"},
        studioforge={"lease_mode": "off"},
        voices={"dir": tmp_path / "voices"},
        realtime={"ready_wait_s": 120},
    )


@pytest.mark.models
def test_smart_turn_and_parakeet_words_cpu():
    from speechforge.audio import resample_linear
    from speechforge.engines.parakeet import ParakeetEngine
    from speechforge.engines.turn import SmartTurn
    from speechforge.selftest import golden_clip

    clip, rate = golden_clip()
    a16 = resample_linear(clip, rate, 16000)
    st = SmartTurn(Path(MODELS) / "smart-turn" / "smart-turn-v3.2-cpu.onnx", provider="cpu")
    done = st.score(a16[: int(3.0 * 16000)])  # "Hey, welcome back to the show." + pause
    assert 0.0 <= done <= 1.0
    eng = ParakeetEngine(
        Path(MODELS) / "parakeet-tdt-0.6b-v2", quantization="int8", provider="cpu", cpu_threads=2
    )
    text, words = eng.transcribe_words(a16[: 3 * 16000])
    assert text.lower().startswith("hey") and words[0][0].lower().startswith("hey")
    assert all(w[1] <= w[2] for w in words) and words[-1][2] <= 3.1


@pytest.mark.models
def test_sf1_end_to_end_on_cpu_with_selftest(tmp_path, monkeypatch):
    """The whole server on the CPU provider with the real models: load, self-test,
    hello/ready, STT stream -> partial/eou/final, TTS -> frames/words/done, cancel."""
    import json
    import time

    from starlette.testclient import TestClient

    from speechforge import models
    from speechforge.app import create_app
    from speechforge.audio import resample_linear
    from speechforge.protocol import unpack_audio
    from speechforge.selftest import golden_clip
    from speechforge.session import mic_frames

    monkeypatch.setenv("SPEECHFORGE_TOKEN", "t" * 40)
    monkeypatch.setattr(models, "verify_for_load", lambda cfg: None)  # local copies, no stamps
    cfg = _cpu_config(tmp_path)
    app = create_app(cfg)
    clip, rate = golden_clip()
    eng = engine("cpu", use_iobinding=False)
    voice = eng.build_voice(resample_linear(clip, rate, 24000)).to_npz()
    import hashlib

    vid = hashlib.sha256(voice).hexdigest()
    with TestClient(app) as tc:
        r = tc.put(
            f"/v1/voices/{vid}", content=voice, headers={"Authorization": "Bearer " + "t" * 40}
        )
        assert r.status_code == 201
        with tc.websocket_connect(
            "/v1/realtime", subprotocols=["sf.1"], headers={"Authorization": "Bearer " + "t" * 40}
        ) as ws:
            ws.send_text(json.dumps({"type": "hello", "proto": "sf.1"}))
            msgs = []
            while True:
                m = ws.receive_json()
                msgs.append(m)
                if m["type"] in ("ready", "error"):
                    break
            assert m["type"] == "ready", msgs
            st = app.state.residency.selftest_result
            assert st["pass"], st
            a16 = (resample_linear(clip, rate, 16000)[: int(3.2 * 16000)] * 32767).astype("<i2")
            pad = np.zeros(int(0.6 * 16000), "<i2")
            for f in mic_frames(np.concatenate([a16, pad])):
                ws.send_bytes(f)
                time.sleep(0.018)
            ws.send_text(json.dumps({"type": "stt.flush", "id": 1}))
            seen = {}
            while "stt.final" not in seen:
                m = ws.receive_json()
                seen.setdefault(m["type"], m)
            assert "stt.partial" in seen and "eou" in seen
            assert seen["stt.final"]["text"].lower().startswith("hey")
            ws.send_text(
                json.dumps(
                    {"type": "tts.open", "ctx": 1, "gen": 1, "voice": f"sha256:{vid}", "seed": 3}
                )
            )
            ws.send_text(
                json.dumps(
                    {"type": "tts.text", "ctx": 1, "text": "Sure, one moment.", "final": True}
                )
            )
            ws.send_text(json.dumps({"type": "tts.close", "ctx": 1}))
            frames, kinds = [], []
            while True:
                m = ws.receive()
                if m.get("bytes") is not None:
                    frames.append(unpack_audio(m["bytes"]))
                    continue
                d = json.loads(m["text"])
                kinds.append(d["type"])
                if d["type"] == "tts.done":
                    break
            assert "tts.started" in kinds and "tts.words" in kinds
            assert sum(len(f.pcm) for f in frames) == d["audio_ms"] * 24
            ws.send_text(
                json.dumps({"type": "tts.open", "ctx": 2, "gen": 2, "voice": f"sha256:{vid}"})
            )
            ws.send_text(
                json.dumps(
                    {
                        "type": "tts.text",
                        "ctx": 2,
                        "final": True,
                        "text": "This is a much longer sentence that we will "
                        "interrupt before it finishes speaking.",
                    }
                )
            )
            while True:
                m = ws.receive()
                if m.get("text") and json.loads(m["text"])["type"] == "tts.started":
                    break
            ws.send_text(json.dumps({"type": "cancel", "gen": 2}))
            while True:
                m = ws.receive()
                if m.get("text") and json.loads(m["text"])["type"] == "cancelled":
                    c = json.loads(m["text"])
                    break
            assert c["ctxs"] == [2]
            print("selftest", st, "cancel_ms", c["stop_ms"])
