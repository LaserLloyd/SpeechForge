"""drama3-tts: Fish Audio Drama 3 through audio.cpp, as an add-on engine (S33).

Everything here is offline: the server is a MockTransport, the spawn path is
tested through the config document it writes and the checks it makes before
launching anything, and the resampler is held to the whole-clip reference.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from speechforge.audio import TTS_SAMPLE_RATE, resample_poly
from speechforge.config import Config
from speechforge.engines import base, drama3_tts
from speechforge.engines.drama3_tts import (
    MODEL_FILE,
    MODEL_ID,
    NAME,
    VOICE_FORMAT,
    Drama3Tts,
    PolyphaseResampler,
    make_voice,
    parse_voice,
    server_config,
)
from speechforge.models import tts_groups


def clip(seconds: float = 1.0, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def voice():
    return parse_voice(make_voice(clip(), 24000, "A short reference sentence."))


def _config(tmp_path, **drama3):
    return Config(
        data_dir=tmp_path,
        tts={"engine": NAME,
             "drama3": {"backend": "http", "url": "http://127.0.0.1:8035", **drama3}},
        voices={"allow_reference_clip": True},
    )


# ----------------------------------------------------------------------------- resampler
@pytest.mark.parametrize("chunk", [88200, 1234, 7])
def test_streaming_resampler_equals_the_whole_clip(chunk):
    x = (np.sin(np.arange(44100 * 2) * 2 * np.pi * 440 / 44100) * 0.5).astype(np.float32)
    ref = resample_poly(x, 44100, 24000)
    rs = PolyphaseResampler(44100, 24000)
    parts = [rs.process(x[i : i + chunk]) for i in range(0, len(x), chunk)]
    parts.append(rs.flush())
    y = np.concatenate(parts)
    assert len(y) == len(ref) == 48000
    assert float(np.abs(y - ref).max()) == 0.0


def test_resampler_passes_ratio_one_and_refuses_junk():
    rs = PolyphaseResampler(24000, 24000)
    x = clip(0.1)
    assert np.array_equal(rs.process(x), x) and len(rs.flush()) == 0
    with pytest.raises(ValueError):
        PolyphaseResampler(0, 24000)
    assert len(PolyphaseResampler(44100, 24000).flush()) == 0  # nothing pushed


# ----------------------------------------------------------------------------- registry / config
def test_registered_as_an_add_on_engine_with_no_lock_files(tmp_path):
    base.builtin()
    assert NAME in base.TTS_ENGINES
    assert tts_groups(_config(tmp_path)) == []  # audio.cpp's model manager owns the files


def test_config_defaults_and_loopback_rule(tmp_path):
    cfg = _config(tmp_path).tts.drama3
    assert cfg.backend == "http" and cfg.port == 8035 and cfg.max_ref_s == 6.0
    assert cfg.output_rate == 24000 and cfg.cpu_expert_layers == 0
    with pytest.raises(ValidationError):
        _config(tmp_path, url="http://192.168.1.9:8035")
    with pytest.raises(ValidationError):
        _config(tmp_path, url="https://127.0.0.1:8035")


def test_d3_gate_refuses_without_the_owner_opt_in(tmp_path):
    config = Config(data_dir=tmp_path, tts={"engine": NAME, "drama3": {"backend": "http"}})
    with pytest.raises(RuntimeError, match="allow_reference_clip"):
        drama3_tts.load(config, 0)


def test_server_config_is_the_servers_own_shape(tmp_path):
    config = _config(tmp_path, cpu_expert_layers=4, context_length=4096, max_tokens=500, threads=6)
    doc = server_config(config, Path("C:/m/drama.gguf"), port=9000)
    assert doc["host"] == "127.0.0.1" and doc["port"] == 9000 and doc["backend"] == "cuda"
    assert doc["lazy_load"] is False and doc["idle_unload_ms"] == 0 and doc["ui"] is False
    (model,) = doc["models"]
    assert model["id"] == MODEL_ID and model["family"] == "fish_drama3"
    assert model["mode"] == "streaming" and model["task"] == "tts"
    assert model["session_options"] == {
        "fish_drama3.cpu_expert_layers": 4,
        "fish_drama3.context_length": 4096,
    }
    assert model["default_request_options"] == {"max_tokens": 500}
    assert doc["threads"] == 6
    json.dumps(doc)  # serialisable


# ----------------------------------------------------------------------------- the http backend
def test_http_backend_sends_the_clip_as_base64_and_streams_pcm(tmp_path, voice):
    import httpx

    config = _config(tmp_path, max_ref_s=1.0)
    voice = parse_voice(make_voice(clip(2.0), 24000, "A short reference sentence."))
    be = drama3_tts._Backend(config, 1)
    try:
        assert be.client.trust_env is False and be.client.follow_redirects is False
        assert be.spawn is False and be.process is None
        seen: list[dict] = []
        pcm = (np.arange(-5, 6, dtype="<i2") * 1000).tobytes() + b"\x07"

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append({"url": str(request.url), "body": json.loads(request.content)})
            return httpx.Response(200, stream=httpx.ByteStream(pcm))

        be.client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
        out = np.concatenate(list(be.synth(voice, "v1", "Hello.", seed=3)))
        assert len(out) == 11 and out.dtype == np.float32
        assert seen[0]["url"] == "http://127.0.0.1:8035/v1/audio/speech"
        body = seen[0]["body"]
        assert body["model"] == MODEL_ID and body["input"] == "Hello."
        assert body["response_format"] == "pcm" and body["stream_format"] == "audio"
        assert body["seed"] == 3 and body["max_tokens"] == 800
        assert body["reference_text"] == voice.transcript
        wav = base64.b64decode(body["voice_ref"]["data"])
        assert body["voice_ref"]["type"] == "base64" and wav[:4] == b"RIFF"
        # max_ref_s cut the 2 s clip to 1 s: 24000 samples * 2 bytes + the 44-byte header
        assert len(wav) == 44 + 24000 * 2

        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="server_busy")

        be.client = httpx.Client(transport=httpx.MockTransport(refuse), trust_env=False)
        with pytest.raises(RuntimeError, match="503"):
            list(be.synth(voice, "v1", "Hello.", seed=None))
    finally:
        be.close()


def test_adapter_streams_24k_pcm16_with_stats_and_gate(tmp_path, voice):
    import httpx

    config = _config(tmp_path)
    be = drama3_tts._Backend(config, 1)
    try:
        native = (np.sin(np.arange(44100) * 2 * np.pi * 300 / 44100) * 0.4).astype(np.float32)
        pcm = (native * 32767).astype("<i2").tobytes()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=httpx.ByteStream(pcm))

        be.client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
        tts = Drama3Tts(be, output_rate=24000)
        assert tts.sample_rate == 24000 and tts.voice_format == VOICE_FORMAT
        gates: list[int] = []

        def gate(count: int) -> bool:
            gates.append(count)
            return True

        done: list[int] = []
        pieces = list(tts.stream("One second of tone.", "v1", voice,
                                 options={"gate": gate, "on_lm_done": done.append}))
        out = np.concatenate(pieces)
        assert out.dtype == np.int16 and abs(len(out) - 24000) <= 1
        ref = resample_poly(native, 44100, 24000)
        assert float(np.abs(out.astype(np.float32) / 32768 - ref).max()) < 2e-4
        stats = tts.last_stats
        assert stats["ttfa_ms"] is not None and stats["audio_ms"] in (999, 1000)
        assert stats["tokens"] == 44100 // 1764 and done == [stats["tokens"]]
        assert stats["rtf"] is not None and stats["stopped"] is False
        assert gates and gates[0] == 0
        info = tts.describe()
        assert info["engine"] == NAME and info["attribution"] == "Built with Fish Audio"
        assert info["native_sample_rate"] == 44100 and info["backend"] == "http"
        assert tts.render("x", voice).dtype == np.int16
    finally:
        tts.close()


def test_stop_event_cuts_the_stream(tmp_path, voice):
    import threading

    import httpx

    be = drama3_tts._Backend(_config(tmp_path), 1)
    try:
        pcm = np.zeros(44100 * 3, "<i2").tobytes()
        be.client = httpx.Client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, stream=httpx.ByteStream(pcm))),
            trust_env=False)
        tts = Drama3Tts(be)
        stop = threading.Event()
        got = 0
        for piece in tts.stream("Long.", "v1", voice, stop=stop):
            got += len(piece)
            stop.set()
        assert got < 24000 * 3 and tts.last_stats["stopped"] is True
    finally:
        be.close()


def test_a_voice_needs_its_transcript(tmp_path, voice):
    """The server refuses `voice_ref` without `reference_text` (seen live 2026-10-15),
    so a transcript-less voice is refused at parse time, from-reference is refused
    with the way that works, and the warm-up uses the built-in voice."""
    import httpx

    from speechforge.errors import BadRequest

    with pytest.raises(BadRequest, match="transcript"):
        parse_voice(make_voice(clip(), 24000, ""))
    be = drama3_tts._Backend(_config(tmp_path), 1)
    try:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(200, stream=httpx.ByteStream(b"\x00\x00" * 4410))

        be.client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
        tts = Drama3Tts(be)
        with pytest.raises(BadRequest, match="transcript"):
            tts.build_voice(clip(5.0, TTS_SAMPLE_RATE), TTS_SAMPLE_RATE)
        tts.warm_up()
        assert "voice_ref" not in seen[-1] and "reference_text" not in seen[-1]
        assert seen[-1]["input"] == "Warming up the voice."
        bare = drama3_tts.DotsVoice(clip(), 24000, "")
        with pytest.raises(RuntimeError, match="transcript"):
            list(tts.stream("x", "bare", bare))
        list(tts.stream("y", "v1", voice))
        assert seen[-1]["reference_text"] == voice.transcript
    finally:
        be.close()


# ----------------------------------------------------------------------------- the spawn backend
def test_spawn_refuses_before_launching_when_the_install_is_missing(tmp_path, monkeypatch):
    config = Config(
        data_dir=tmp_path,
        tts={"engine": NAME, "drama3": {"backend": "spawn"}},
        voices={"allow_reference_clip": True},
    )
    launched: list = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: launched.append(a))
    with pytest.raises(RuntimeError, match="engine-setup"):
        drama3_tts._Backend(config, 0)
    exe = tmp_path / "engines" / "audio.cpp" / "build" / (
        "windows-cuda-release" if sys.platform == "win32" else "linux-cuda-release") / "bin"
    exe.mkdir(parents=True)
    exe_file = exe / ("audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server")
    exe_file.write_bytes(b"")
    with pytest.raises(RuntimeError, match="model not found"):
        drama3_tts._Backend(config, 0)
    model = tmp_path / "models" / "audiocpp" / "Drama-3-GGUF" / MODEL_FILE
    model.parent.mkdir(parents=True)
    model.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="expected"):
        drama3_tts._Backend(config, 0)
    assert launched == []


def test_spawn_writes_the_config_and_pins_the_card(tmp_path, monkeypatch):
    """A fake Popen that answers /health through the backend's own client."""
    import httpx

    config = Config(
        data_dir=tmp_path,
        tts={"engine": NAME, "drama3": {"backend": "spawn", "port": 8111}},
        voices={"allow_reference_clip": True},
    )
    exe_dir = tmp_path / "bin"
    exe_dir.mkdir()
    exe = exe_dir / "audiocpp_server.exe"
    exe.write_bytes(b"")
    model = tmp_path / "drama.gguf"
    model.write_bytes(b"x" * 16)
    monkeypatch.setattr(drama3_tts, "MODEL_SIZE", 16)  # the size check, without 24 GB
    config.tts.drama3.server_exe = str(exe)
    config.tts.drama3.model_path = str(model)
    seen: dict = {}

    class FakeProc:
        pid = 4242

        def __init__(self, argv, **kw):
            seen["argv"] = argv
            seen["env"] = kw["env"]
            self._rc = None

        def poll(self):
            return self._rc

        def terminate(self):
            self._rc = 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self._rc = -9

    monkeypatch.setattr(subprocess, "Popen", FakeProc)
    real_init = drama3_tts._Backend.__init__

    def init(self, config, device_id):
        # Install the mock transport before _spawn() polls /health.
        orig_spawn = self._spawn

        def spawn(cfg):
            self.client = httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok"})),
                trust_env=False)
            orig_spawn(cfg)

        self._spawn = spawn
        real_init(self, config, device_id)

    monkeypatch.setattr(drama3_tts._Backend, "__init__", init)
    be = drama3_tts._Backend(config, 3)
    try:
        assert seen["argv"][0] == str(exe) and seen["argv"][1] == "--config"
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert seen["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert "__PYVENV_LAUNCHER__" not in seen["env"]
        doc = json.loads(Path(seen["argv"][2]).read_text(encoding="utf-8"))
        assert doc["port"] == 8111 and doc["models"][0]["path"] == str(model)
        assert be.url == "http://127.0.0.1:8111"
        assert be.describe()["pid"] == 4242 and be.describe()["device"] == 3
    finally:
        be.close()
    assert be.process is None


# ----------------------------------------------------------------------------- setup
def test_setup_names_what_is_missing_before_touching_anything(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="git"):
        drama3_tts.setup(_config(tmp_path), log=lambda s: None)
    assert calls == []


def test_setup_drives_the_ports_own_tools_in_order(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(drama3_tts, "_vs_build_tools", lambda: True)
    runs: list[list[str]] = []

    class Done:
        stdout = drama3_tts.AUDIOCPP_COMMIT + "\n"

    def fake_run(argv, **kw):
        runs.append([str(a) for a in argv])
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1], ".git").mkdir(parents=True)
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = drama3_tts.setup(_config(tmp_path), log=lambda s: None)
    kinds = [r[0] for r in runs]
    assert kinds[0] == "git" and runs[0][1] == "clone"
    assert runs[0][3] == drama3_tts.AUDIOCPP_BRANCH
    assert any("model_manager_v2.py" in " ".join(r) and drama3_tts.MODEL_PACKAGE in r for r in runs)
    build = [r for r in runs
             if "build_windows.ps1" in " ".join(r) or "build_linux.sh" in " ".join(r)]
    assert build and all("fish_drama3" in r for r in build)
    assert out["commit"] == drama3_tts.AUDIOCPP_COMMIT
    assert out["model_exists"] is False and out["attribution"] == "Built with Fish Audio"
    assert Path(out["model_path"]).name == MODEL_FILE
