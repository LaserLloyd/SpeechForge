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
    make_voice_codes,
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
        "fish_drama3.prefix_cache_entries": 4,
    }
    assert model["default_request_options"] == {"max_tokens": 500}
    assert doc["threads"] == 6
    json.dumps(doc)  # serialisable
    # A stock server refuses the option, so `prefix_cache: false` keeps it out (S36).
    stock = server_config(_config(tmp_path, prefix_cache=False), Path("C:/m/drama.gguf"))
    assert "fish_drama3.prefix_cache_entries" not in stock["models"][0]["session_options"]
    off = server_config(_config(tmp_path, prefix_cache_entries=0), Path("C:/m/drama.gguf"))
    assert off["models"][0]["session_options"]["fish_drama3.prefix_cache_entries"] == 0




# ----------------------------------------------------------------------------- a fake server
CODES = [[(i * 7 + j) % 4096 for j in range(43)] for i in range(10)]   # 2 s at 21.5 fps


def fake_server(pcm: bytes, *, log: list | None = None, take_codes=None, take_wav: bytes = b""):
    """MockTransport handler: the speech route streams ``pcm``; the tasks route
    answers an encode-only request with a drama3_reference artifact and a plain
    take with audio + a drama3_take artifact. Every request body is appended to
    ``log`` as ``(path, body)``."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        if log is not None:
            log.append((path, body))
        if path == "/v1/audio/speech":
            return httpx.Response(200, stream=httpx.ByteStream(pcm))
        if path == "/v1/tasks/run":
            opts = (body.get("request") or {}).get("options") or {}
            if opts.get("encode_reference_only"):
                payload = {"text": opts.get("reference_text", ""), "vq_codes": CODES}
                return httpx.Response(200, json={"artifacts": [{
                    "id": "drama3_reference_0", "kind": "acoustic_tokens",
                    "payload": base64.b64encode(json.dumps(payload).encode()).decode(),
                    "meta": {"family": "fish_drama3", "schema": "drama3_reference_v1"}}]})
            payload = {"text": body["request"]["text"], "vq_codes": take_codes or CODES}
            return httpx.Response(200, json={
                "audio": base64.b64encode(take_wav).decode(), "sample_rate": 44100,
                "artifacts": [{
                    "id": "drama3_take", "kind": "acoustic_tokens",
                    "payload": base64.b64encode(json.dumps(payload).encode()).decode(),
                    "meta": {"family": "fish_drama3", "schema": "drama3_take_v1"}}]})
        return httpx.Response(404)

    return handler

# ----------------------------------------------------------------------------- the http backend
def test_http_backend_encodes_a_clip_once_and_sends_codes(tmp_path, voice):
    """A clip voice is encoded ONCE through the server's encode_reference_only
    task (a server-side path, removed after) and every request carries the
    codes in options.references -- never the audio (S34)."""
    import httpx

    config = _config(tmp_path, max_ref_s=1.0)
    voice = parse_voice(make_voice(clip(2.0), 24000, "A short reference sentence."))
    be = drama3_tts._Backend(config, 1)
    try:
        assert be.client.trust_env is False and be.client.follow_redirects is False
        assert be.spawn is False and be.process is None
        seen: list = []
        pcm = (np.arange(-5, 6, dtype="<i2") * 1000).tobytes() + b"\x07"
        be.client = httpx.Client(transport=httpx.MockTransport(fake_server(pcm, log=seen)),
                                 trust_env=False)
        out = np.concatenate(list(be.synth(voice, "v1", "Hello.", seed=3)))
        assert len(out) == 11 and out.dtype == np.float32
        paths = [path for path, _ in seen]
        assert paths == ["/v1/tasks/run", "/v1/audio/speech"]
        enc = seen[0][1]["request"]
        assert enc["options"] == {"reference_text": voice.transcript,
                                  "encode_reference_only": True}
        assert enc["voice_ref"].startswith(str(be.ref_dir)) and not Path(enc["voice_ref"]).exists()
        body = seen[1][1]
        assert body["model"] == MODEL_ID and body["input"] == "Hello."
        assert body["response_format"] == "pcm" and body["stream_format"] == "audio"
        assert body["seed"] == 3 and body["max_tokens"] == 800
        assert "voice_ref" not in body and "reference_text" not in body
        (ref,) = body["options"]["references"]
        assert ref["text"] == voice.transcript and ref["vq_codes"] == CODES
        # Second request: no encode, the cache answers.
        list(be.synth(voice, "v1", "Again.", seed=None))
        assert [path for path, _ in seen][2:] == ["/v1/audio/speech"]
        assert be.describe()["cached_voices"] == 1
        # A codes-only voice never touches the encode task.
        kept = parse_voice(make_voice_codes("Kept.", np.array(CODES)))
        list(be.synth(kept, "kept", "Line.", seed=None))
        assert seen[-1][0] == "/v1/audio/speech"
        assert seen[-1][1]["options"]["references"][0]["vq_codes"] == CODES

        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="server_busy")

        be.client = httpx.Client(transport=httpx.MockTransport(refuse), trust_env=False)
        with pytest.raises(RuntimeError, match="503"):
            list(be.synth(voice, "v1", "Hello.", seed=None))
    finally:
        be.close()


def test_codes_container_roundtrip_and_validation():
    data = make_voice_codes("Kept take.", np.array(CODES), clip(0.5), 24000)
    v = parse_voice(data)
    assert v.transcript == "Kept take." and v.vq_codes.shape == (10, 43)
    assert v.audio is not None and abs(v.seconds - 0.5) < 1e-6
    bare = parse_voice(make_voice_codes("No clip.", np.array(CODES)))
    assert bare.audio is None and bare.seconds == 0.0
    from speechforge.errors import BadRequest

    with pytest.raises(BadRequest, match="transcript"):
        parse_voice(make_voice_codes("", np.array(CODES)))
    with pytest.raises(BadRequest, match="frames"):
        make_voice_codes("x", np.zeros((9, 5), np.int32))
    with pytest.raises(BadRequest, match="integers"):
        make_voice_codes("x", np.full((10, 5), 4096, np.int32))
    with pytest.raises(BadRequest, match="at most"):
        make_voice_codes("x", np.zeros((10, 3000), np.int32))


def test_take_keeps_an_improvised_voice(tmp_path):
    """take(): one request on /v1/tasks/run with no reference; the take's own
    codes become a codes-only voice and the audio comes back at 24 kHz."""
    import httpx

    from speechforge.audio import wav_bytes

    native = (np.sin(np.arange(44100) * 2 * np.pi * 300 / 44100) * 0.4).astype(np.float32)
    wav = wav_bytes((native * 32767).astype("<i2"), 44100)
    seen: list = []
    be = drama3_tts._Backend(_config(tmp_path), 1)
    try:
        be.client = httpx.Client(
            transport=httpx.MockTransport(fake_server(b"", log=seen, take_wav=wav)),
            trust_env=False)
        tts = Drama3Tts(be)
        voice_bytes, preview = tts.take("Hello from a stranger.", seed=9)
        assert seen == [("/v1/tasks/run", {"model": MODEL_ID, "request": {
            "text": "Hello from a stranger.", "max_tokens": 800, "seed": 9}})]
        assert preview.dtype == np.int16 and abs(len(preview) - 24000) <= 1
        v = parse_voice(voice_bytes)
        assert v.transcript == "Hello from a stranger." and v.vq_codes.tolist() == CODES
        assert v.rate == 24000 and v.audio is not None and abs(v.seconds - 1.0) < 1e-3
        from speechforge.errors import BadRequest

        with pytest.raises(BadRequest):
            tts.take("   ")
        assert tts.describe()["keep_voice"] is True
    finally:
        be.close()


def test_adapter_streams_24k_pcm16_with_stats_and_gate(tmp_path, voice):
    import httpx

    config = _config(tmp_path)
    be = drama3_tts._Backend(config, 1)
    try:
        native = (np.sin(np.arange(44100) * 2 * np.pi * 300 / 44100) * 0.4).astype(np.float32)
        pcm = (native * 32767).astype("<i2").tobytes()

        be.client = httpx.Client(transport=httpx.MockTransport(fake_server(pcm)), trust_env=False)
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
        be.client = httpx.Client(transport=httpx.MockTransport(fake_server(pcm)), trust_env=False)
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
        seen: list = []
        be.client = httpx.Client(
            transport=httpx.MockTransport(fake_server(b"\x00\x00" * 4410, log=seen)),
            trust_env=False)
        tts = Drama3Tts(be)
        with pytest.raises(BadRequest, match="transcript"):
            tts.build_voice(clip(5.0, TTS_SAMPLE_RATE), TTS_SAMPLE_RATE)
        tts.warm_up()
        assert "options" not in seen[-1][1] and "voice_ref" not in seen[-1][1]
        assert seen[-1][1]["input"] == "Warming up the voice."
        bare = drama3_tts.Drama3Voice("", clip(), 24000, None)
        with pytest.raises(RuntimeError, match="transcript"):
            list(tts.stream("x", "bare", bare))
        list(tts.stream("y", "v1", voice))
        assert seen[-1][1]["options"]["references"][0]["text"] == voice.transcript
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
    envs: list[dict] = []

    class Done:
        stdout = drama3_tts.AUDIOCPP_COMMIT + "\n"

    def fake_run(argv, **kw):
        runs.append([str(a) for a in argv])
        envs.append(kw.get("env") or {})
        if argv[:2] == ["git", "clone"]:
            Path(argv[-1], ".git").mkdir(parents=True)
        if argv[:2] == ["git", "apply"]:
            # `--check --reverse` (already applied?) says no; `--check` says yes.
            done = Done()
            done.returncode = 1 if "--reverse" in argv else 0
            done.stderr = ""
            return done
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    toolkit = tmp_path / "cuda-13.1"
    (toolkit / "bin").mkdir(parents=True)
    out = drama3_tts.setup(_config(tmp_path), cuda_toolkit=toolkit, cuda_arch="120a-real",
                           log=lambda s: None)
    kinds = [r[0] for r in runs]
    assert kinds[0] == "git" and runs[0][1] == "clone"
    assert runs[0][3] == drama3_tts.AUDIOCPP_BRANCH
    assert any("model_manager_v2.py" in " ".join(r) and drama3_tts.MODEL_PACKAGE in r for r in runs)
    build = [(r, e) for r, e in zip(runs, envs, strict=True)
             if "build_windows.ps1" in " ".join(r) or "build_linux.sh" in " ".join(r)]
    assert build and all("fish_drama3" in r for r, _ in build)
    # S35: the build compiles with the named toolkit, through CUDACXX (CUDA_PATH is not enough).
    for argv, env in build:
        assert env["CUDAToolkit_ROOT"] == str(toolkit) and env["CUDA_PATH"] == str(toolkit)
        assert env["CUDACXX"].startswith(str(toolkit / "bin"))
        if "build_windows.ps1" in " ".join(argv):
            assert argv[argv.index("-CudaArchitectures") + 1] == "120a-real"
    assert out["cuda_toolkit"] == str(toolkit)
    assert out["commit"] == drama3_tts.AUDIOCPP_COMMIT
    # S36: the shipped prefix-cache patch is applied before the build.
    applies = [r for r in runs if r[:2] == ["git", "apply"] and "--check" not in r]
    assert applies and applies[0][2].endswith("audiocpp-drama3-prefix-cache.patch")
    assert out["prefix_cache_patch"] is True
    assert runs.index(applies[0]) < runs.index(build[0][0])


def test_shipped_patch_is_present_and_targets_the_port():
    text = drama3_tts.PREFIX_CACHE_PATCH.read_text(encoding="utf-8")
    assert text.count("diff --git") == 10
    assert "fish_drama3/backbone.cpp" in text and "prefix_cache_entries" in text


def test_setup_can_build_the_stock_port(tmp_path, monkeypatch):
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
    out = drama3_tts.setup(_config(tmp_path), prefix_cache=False, build=False,
                           install_model=False, log=lambda s: None)
    assert out["prefix_cache_patch"] is False
    assert not any(r[:2] == ["git", "apply"] for r in runs)
    assert out["model_exists"] is False and out["attribution"] == "Built with Fish Audio"
    assert Path(out["model_path"]).name == MODEL_FILE
