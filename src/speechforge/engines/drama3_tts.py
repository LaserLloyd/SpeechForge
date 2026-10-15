"""Fish Audio Drama 3 through audio.cpp (``tts.engine: drama3-tts``). Add-on engine.

Drama 3 (Fish Audio, 2026-10-13) is a 35B mixture-of-experts text-to-speech
model with 3B active parameters, 83 languages, inline direction tags
(``[laughs]``, ``[whispering]``) and reference-clip cloning. The official
runtime is SGLang-Omni on Linux with a 48 GB card; this engine instead drives
the community **audio.cpp** port (JarodMica/audio.cpp, branch
``drama-3-native``, Apache-2.0), a ggml runtime that runs a mixed Q4/Q8 GGUF
(23 GB) natively on Windows and Linux with CUDA, streams PCM, and clones from
a reference clip plus its transcript. Measured on this host (docs/BENCHMARKS.md,
2026-10-15, RTX 5090, Q4 experts): first audio 185-234 ms, RTF 0.34-0.40;
with a 12.6 s reference clip the first audio moves to ~0.9-1.0 s because the
server encodes the reference on every request, so ``max_ref_s`` keeps clips
short.

Nothing of the engine lives in this repository: ``speechforge engine-setup
drama3`` clones the port at a pinned commit into ``<data>/engines``, builds the
server with the port's own build script, and installs the GGUF with the port's
own model manager (``tools/model_manager_v2.py``). SpeechForge then spawns
``audiocpp_server`` on the leased card with a generated config, exactly the
server's documented shape, and talks to its ``POST /v1/audio/speech`` over
loopback. ``backend: http`` points at a server you run yourself instead.

Licences: the weights are under the Fish Audio Community License v1.0 (free for
research, personal and non-commercial use, and for commercial use below
USD 500,000 annual revenue; a hosted TTS/voice-cloning service for third
parties needs a separate licence; cloning an identifiable person needs their
consent; products using the model must display "Built with Fish Audio").
audio.cpp is Apache-2.0. Neither is shipped here. This engine clones from the
reference clip itself, so it needs ``voices.allow_reference_clip: true`` (D3).

Heavy work (httpx, subprocess) happens inside :func:`load` and :func:`setup`.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.audio import TTS_SAMPLE_RATE, _poly_filters, float_to_pcm16
from speechforge.engines.dots_tts import DotsVoice, make_voice
from speechforge.engines.dots_tts import parse_voice as _parse_ref
from speechforge.errors import BadRequest

NAME = "drama3-tts"
#: The voice container is the reference-clip-plus-transcript npz the dots engine
#: defined (``audio``, ``rate``, ``transcript``): one format for every engine that
#: clones from a clip, so a client builds the file once.
VOICE_FORMAT = "dotstts-ref"
NATIVE_RATE = 44100
TOKEN_MS = 40
MODEL_ID = "drama-3"

#: The port, pinned. ``engine-setup`` checks out exactly this commit.
AUDIOCPP_REPO = "https://github.com/JarodMica/audio.cpp.git"
AUDIOCPP_BRANCH = "drama-3-native"
AUDIOCPP_COMMIT = "4e6018fd018499d75e5070ba4eb38db4e48b9b59"
#: The model-manager package (audio.cpp ``model_specs``): Q4_K experts, Q8_0
#: backbone, FP16 audio tower and codec in one file, 22.9 GiB. Quality is the
#: closest of the four packages to the original; it fits a 32 GB card without
#: CPU offload (23.3 GB resident, measured).
MODEL_PACKAGE = "fish_drama3_q4_experts_q8_backbone"
MODEL_TARGET_DIR = "Drama-3-GGUF"
MODEL_FILE = "drama-3-backbone-8bit-experts-Q4_K_S.gguf"
MODEL_SIZE = 24596497888  # bytes; checked at spawn (a full hash is setup's job)
# sha256 of the package file, measured 2026-10-15 after the model-manager install.
MODEL_SHA256 = "05af716baf78e7c20863b9ccebb3ebdeca9607dd3ea7d0b72c35079e675b212b"
MODEL_LICENCE = (
    "Fish Audio Community License v1.0 (commercial terms at fish.audio above USD 500k revenue)"
)

__all__ = [
    "NAME",
    "VOICE_FORMAT",
    "Drama3Tts",
    "PolyphaseResampler",
    "load",
    "parse_voice",
    "make_voice",
    "setup",
    "server_config",
]


# ----------------------------------------------------------------------------- voices
NO_TRANSCRIPT = (
    "drama3-tts needs the reference clip's transcript: the server refuses a reference "
    "without `reference_text`. Build the voice with make_voice(audio, rate, transcript) "
    "(the dotstts-ref container) and PUT that; /v1/voices/from-reference carries no transcript."
)


def parse_voice(data: bytes) -> DotsVoice:
    """Validate ``dotstts-ref`` bytes for Drama 3: the clip AND its transcript."""
    voice = _parse_ref(data)
    if not voice.transcript:
        raise BadRequest(NO_TRANSCRIPT, code="bad_voice")
    return voice


# ----------------------------------------------------------------------------- resampling
class PolyphaseResampler:
    """Streaming :func:`speechforge.audio.resample_poly`: the same Kaiser
    windowed-sinc polyphase filter, applied chunk by chunk with the history
    carried between calls, so a sentence resampled piecewise equals the whole
    clip resampled at once (edge padding at both ends included). Ratio 1 passes.
    """

    def __init__(self, src: int, dst: int, *, zero_crossings: int = 32, beta: float = 8.6) -> None:
        if src <= 0 or dst <= 0:
            raise ValueError("sample rates must be positive")
        self.src, self.dst = int(src), int(dst)
        if self.src == self.dst:
            self.table = None
            return
        g = math.gcd(self.src, self.dst)
        self.up, self.down = self.dst // g, self.src // g
        self.table = _poly_filters(self.up, self.down, zero_crossings, beta)
        self.taps = self.table.shape[1]
        self.d0 = (self.taps - 1) // 2
        self._buf = np.zeros(0, np.float32)  # absolute input index of _buf[0] is _off
        self._off = 0
        self._received = 0  # input samples pushed so far
        self._m_next = 0  # next output index
        self._started = False

    def _produce(self, m_end: int) -> np.ndarray:
        """Outputs ``[_m_next, m_end)``; the caller guarantees the inputs are in ``_buf``."""
        if m_end <= self._m_next:
            return np.zeros(0, np.float32)
        m = np.arange(self._m_next, m_end, dtype=np.int64)
        base = (m * self.down) // self.up
        phase = ((m * self.down) % self.up).astype(np.int64)
        t = np.arange(self.taps, dtype=np.int64)
        idx = (base + self.d0 - self._off)[:, None] - t[None, :]
        out = np.einsum("mt,mt->m", self._buf[idx], self.table[phase]).astype(np.float32)
        self._m_next = m_end
        # Drop what no later output can need: inputs below base(m_end) - d0.
        keep_from = int(((m_end * self.down) // self.up) - self.d0)
        drop = keep_from - self._off
        if drop > 0:
            self._buf = self._buf[drop:]
            self._off += drop
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        if self.table is None or not len(x):
            return x
        if not self._started:
            # resample_poly pads the start with d0 copies of the first sample.
            self._buf = np.concatenate([np.full(self.d0, x[0], np.float32), x])
            self._off = -self.d0
            self._started = True
        else:
            self._buf = np.concatenate([self._buf, x])
        self._received += len(x)
        last = self._received - 1  # highest real input index available
        # out[m] needs inputs up to base(m) + d0  ->  base(m) <= last - d0.
        if last - self.d0 < 0:
            return np.zeros(0, np.float32)
        m_end = ((last - self.d0) * self.up) // self.down + 1
        n_out = int(round(self._received * self.up / self.down))
        return self._produce(min(m_end, n_out)) if m_end > self._m_next else np.zeros(0, np.float32)

    def flush(self) -> np.ndarray:
        if self.table is None or not self._started:
            return np.zeros(0, np.float32)
        tail = np.full(2 * self.d0 + 2, self._buf[-1], np.float32)
        self._buf = np.concatenate([self._buf, tail])
        n_out = max(1, int(round(self._received * self.up / self.down)))
        out = self._produce(n_out)
        self._buf = np.zeros(0, np.float32)
        self._started = False
        self._received = 0
        self._m_next = 0
        self._off = 0
        return out


# ----------------------------------------------------------------------------- the server
def _exe_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def default_root(config: Any) -> Path:
    """Where ``engine-setup`` puts the port: ``<data dir>/engines/audio.cpp``."""
    return Path(config.data_dir) / "engines" / "audio.cpp"


def default_server_exe(config: Any) -> Path:
    preset = "windows-cuda-release" if sys.platform == "win32" else "linux-cuda-release"
    return default_root(config) / "build" / preset / "bin" / _exe_name("audiocpp_server")


def default_model_path(config: Any) -> Path:
    return Path(config.models_dir) / "audiocpp" / MODEL_TARGET_DIR / MODEL_FILE


def resolve_paths(config: Any) -> tuple[Path, Path]:
    cfg = config.tts.drama3
    exe = Path(cfg.server_exe) if cfg.server_exe else default_server_exe(config)
    model = Path(cfg.model_path) if cfg.model_path else default_model_path(config)
    return exe, model


def server_config(config: Any, model_path: Path, *, port: int | None = None) -> dict[str, Any]:
    """The ``audiocpp_server --config`` document, in the server's own shape
    (``app/server/README.md``): one streaming ``fish_drama3`` model, loaded at
    start (SpeechForge owns residency: the lease decides when this process
    lives), no idle unload of its own, no UI."""
    cfg = config.tts.drama3
    return {
        "host": "127.0.0.1",
        "port": int(port or cfg.port),
        "backend": "cuda",
        "threads": int(cfg.threads),
        "lazy_load": False,
        "max_loaded_models": 1,
        "idle_unload_ms": 0,
        "ui": False,
        "ui_management": False,
        "models": [
            {
                "id": MODEL_ID,
                "family": "fish_drama3",
                "path": str(model_path),
                "task": "tts",
                "mode": "streaming",
                "session_options": {
                    "fish_drama3.cpu_expert_layers": int(cfg.cpu_expert_layers),
                    "fish_drama3.context_length": int(cfg.context_length),
                },
                "default_request_options": {"max_tokens": int(cfg.max_tokens)},
            }
        ],
    }


class _Backend:
    """``POST /v1/audio/speech`` on an audio.cpp server, spawned here on the
    leased card (``backend: spawn``) or already running (``backend: http``).

    The reference clip travels as a base64 WAV in ``voice_ref`` with its
    transcript in ``reference_text`` -- the server's documented cloning
    request -- so no file is written and the http backend works the same way.
    The request goes to the loopback URL the config validated and nowhere
    else: no proxy from the environment, no redirect off the box.
    """

    native_rate = NATIVE_RATE

    def __init__(self, config: Any, device_id: int) -> None:
        import httpx

        cfg = config.tts.drama3
        self.cfg = cfg
        self.spawn = cfg.backend == "spawn"
        self.url = f"http://127.0.0.1:{int(cfg.port)}" if self.spawn else cfg.url
        self.device_id = int(device_id)
        self.process: subprocess.Popen[bytes] | None = None
        self.config_path: Path | None = None
        self.model_path: Path | None = None
        self.client = httpx.Client(
            timeout=httpx.Timeout(cfg.http_timeout_s, read=cfg.http_timeout_s),
            trust_env=False,
            follow_redirects=False,
        )
        if self.spawn:
            self._spawn(config)

    # -- process -----------------------------------------------------------
    def _spawn(self, config: Any) -> None:
        exe, model = resolve_paths(config)
        if not exe.is_file():
            raise RuntimeError(
                f"tts.drama3: audiocpp_server not found at {exe}; run "
                f"`speechforge engine-setup drama3` (docs/ENGINES.md) or set tts.drama3.server_exe"
            )
        if not model.is_file():
            raise RuntimeError(
                f"tts.drama3: model not found at {model}; run "
                f"`speechforge engine-setup drama3` or set tts.drama3.model_path"
            )
        size = model.stat().st_size
        if size != MODEL_SIZE:
            raise RuntimeError(
                f"tts.drama3: {model} is {size} bytes, expected {MODEL_SIZE} "
                f"({MODEL_PACKAGE}); re-run `speechforge engine-setup drama3`"
            )
        self.model_path = model
        doc = server_config(config, model)
        self.config_path = Path(config.data_dir) / "drama3-server.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env.pop("__PYVENV_LAUNCHER__", None)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        log_dir = Path(config.data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(log_dir / "drama3-server.log", "ab")  # noqa: SIM115 - closed in close()
        self.process = subprocess.Popen(
            [str(exe), "--config", str(self.config_path)],
            cwd=str(exe.parent),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        deadline = time.monotonic() + float(self.cfg.startup_timeout_s)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"tts.drama3: audiocpp_server exited with {self.process.returncode} "
                    f"while loading; see {log_dir / 'drama3-server.log'}"
                )
            if self.healthy():
                return
            time.sleep(0.5)
        self.close()
        raise RuntimeError(
            f"tts.drama3: audiocpp_server did not answer /health within "
            f"{self.cfg.startup_timeout_s:.0f} s"
        )

    def healthy(self) -> bool:
        try:
            r = self.client.get(f"{self.url}/health", timeout=2.0)
        except Exception:  # noqa: BLE001 - not up yet
            return False
        if r.status_code != 200:
            return False
        try:
            return str(r.json().get("status")) == "ok"
        except Exception:  # noqa: BLE001 - not JSON
            return False

    # -- requests ----------------------------------------------------------
    def _body(self, voice: DotsVoice | None, text: str, seed: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": MODEL_ID,
            "input": text,
            "response_format": "pcm",
            "stream_format": "audio",
            "max_tokens": int(self.cfg.max_tokens),
        }
        if voice is not None:
            # The server refuses a reference without its transcript (checked at
            # parse time too; this is the belt for a voice built elsewhere).
            if not voice.transcript:
                raise RuntimeError(NO_TRANSCRIPT)
            clip = voice.audio
            limit = int(float(self.cfg.max_ref_s) * voice.rate)
            if len(clip) > limit:
                clip = clip[:limit]
            wav = DotsVoice(clip, voice.rate, voice.transcript).wav()
            body["voice_ref"] = {"type": "base64", "data": base64.b64encode(wav).decode("ascii")}
            body["reference_text"] = voice.transcript
        if seed is not None:
            body["seed"] = int(seed)
        return body

    def synth(
        self, voice: DotsVoice | None, voice_id: str, text: str, *, seed: int | None
    ) -> Iterator[np.ndarray]:
        """``voice=None`` speaks with the model's own built-in voice (warm-up)."""
        carry = b""
        with self.client.stream(
            "POST", f"{self.url}/v1/audio/speech", json=self._body(voice, text, seed)
        ) as resp:
            if resp.status_code != 200:
                detail = resp.read()[:300].decode("utf-8", "replace")
                raise RuntimeError(f"drama3 server answered {resp.status_code}: {detail}")
            for raw in resp.iter_bytes():
                buf = carry + raw
                n = len(buf) // 2 * 2
                carry = buf[n:]
                if n:
                    yield np.frombuffer(buf[:n], "<i2").astype(np.float32) / 32768.0

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.cfg.backend,
            "url": self.url,
            "model": MODEL_ID,
            "package": MODEL_PACKAGE,
        }
        if self.model_path is not None:
            out["model_path"] = str(self.model_path)
        if self.process is not None:
            out["pid"] = self.process.pid
            out["device"] = self.device_id
        return out

    def close(self) -> None:
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()
            self._log = None
        self.client.close()


# ----------------------------------------------------------------------------- the adapter
def _samples_per_token(rate: int) -> int:
    """Samples per 40 ms credit token at ``rate`` (1764 at 44.1 kHz)."""
    return max(1, rate * TOKEN_MS // 1000)


class Drama3Tts:
    name = NAME
    voice_format = VOICE_FORMAT

    def __init__(self, backend: _Backend, *, output_rate: int = TTS_SAMPLE_RATE) -> None:
        self.backend = backend
        self.sample_rate = int(output_rate)
        self._lock = threading.Lock()  # one request at a time on the server
        self._last: dict[str, Any] = {}
        PolyphaseResampler(backend.native_rate, self.sample_rate)  # validate early

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last)

    def parse_voice(self, data: bytes) -> DotsVoice:
        return parse_voice(data)

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        """``POST /v1/voices/from-reference`` carries no transcript, and Drama 3
        cannot clone without one: refused with the way that works."""
        raise BadRequest(NO_TRANSCRIPT, code="bad_voice")

    def stream(
        self,
        text: str,
        voice_id: str,
        voice: DotsVoice | None,
        *,
        stop: threading.Event | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[np.ndarray]:
        opts = options or {}
        gate: Callable[[int], bool] | None = opts.get("gate")
        on_done: Callable[[int], None] | None = opts.get("on_lm_done")
        feed = opts.get("text_feed")
        if feed is not None:
            # The server starts from complete text: speak the feed as one unit.
            text = text + "".join(feed)
        t0 = time.perf_counter()
        stats: dict[str, Any] = {
            "ttfa_ms": None,
            "audio_ms": 0,
            "chunks": 0,
            "rtf": None,
            "tokens": 0,
            "stopped": False,
        }
        rs = PolyphaseResampler(self.backend.native_rate, self.sample_rate)
        emitted = 0
        native_samples = 0
        steps = self.backend.synth(voice, voice_id, text, seed=seed)
        finished = False

        def emit(chunk: np.ndarray, *, resample: bool = True) -> np.ndarray | None:
            nonlocal emitted
            out = rs.process(chunk) if resample else chunk
            if not len(out):
                return None
            if stats["ttfa_ms"] is None:
                stats["ttfa_ms"] = round((time.perf_counter() - t0) * 1000)
            emitted += len(out)
            return float_to_pcm16(out)

        try:
            while True:
                if stop is not None and stop.is_set():
                    stats["stopped"] = True
                    break
                if gate is not None and not self._wait_gate(gate, native_samples, stop):
                    stats["stopped"] = True
                    break
                with self._lock:
                    try:
                        chunk = next(steps)
                    except StopIteration:
                        finished = True
                        break
                native_samples += int(np.size(chunk))
                stats["chunks"] += 1
                piece = emit(chunk)
                if piece is not None:
                    yield piece
            if finished and not stats["stopped"]:
                tail = emit(rs.flush(), resample=False)
                if tail is not None:
                    yield tail
                if on_done is not None:
                    on_done(native_samples // _samples_per_token(self.backend.native_rate))
        finally:
            steps.close()
            stats["tokens"] = native_samples // _samples_per_token(self.backend.native_rate)
            stats["audio_ms"] = round(emitted / self.sample_rate * 1000)
            total = time.perf_counter() - t0
            stats["total_ms"] = round(total * 1000)
            if stats["audio_ms"]:
                stats["rtf"] = round(total * 1000 / stats["audio_ms"], 3)
            self._last = stats

    def _wait_gate(
        self, gate: Callable[[int], bool], native_samples: int, stop: threading.Event | None
    ) -> bool:
        count = native_samples // _samples_per_token(self.backend.native_rate)
        while not gate(count):
            if stop is not None and stop.is_set():
                return False
            time.sleep(0.005)
        return True

    def render(self, text: str, voice: DotsVoice, *, seed: int | None = None) -> np.ndarray:
        pieces = list(self.stream(text, "_render", voice, seed=seed))
        if not pieces:
            return np.zeros(0, "<i2")
        return np.concatenate(pieces)

    def warm_up(self) -> None:
        """One unit on the built-in voice: fills the server's CUDA graphs and
        decoder shapes without a reference clip (none exists at load)."""
        list(self.stream("Warming up the voice.", "_warmup", None, seed=0))

    def describe(self) -> dict[str, Any]:
        out = {
            "name": "Fish Audio Drama 3 (audio.cpp)",
            "engine": NAME,
            "attribution": "Built with Fish Audio",
            "licence": MODEL_LICENCE,
            "native_sample_rate": self.backend.native_rate,
            "sample_rate": self.sample_rate,
            "voice_format": VOICE_FORMAT,
            "text_stream": False,
        }
        out.update(self.backend.describe())
        return out

    def close(self) -> None:
        with self._lock:
            self.backend.close()


# ----------------------------------------------------------------------------- loading
def _check_allowed(config: Any) -> None:
    if not config.voices.allow_reference_clip:
        raise RuntimeError(
            "tts.engine drama3-tts clones from the reference clip itself, so the clip (and "
            "its transcript) must be on the rig; voices.allow_reference_clip is false "
            "(decision D3: reference clips stay on the box). Set it to true only if you, "
            "the owner, accept that -- see docs/ENGINES.md."
        )


def load(config: Any, device_id: int) -> Drama3Tts:
    """Registry loader (``engines/base.py``): spawns or attaches to the server."""
    _check_allowed(config)
    backend = _Backend(config, device_id)
    return Drama3Tts(backend, output_rate=config.tts.drama3.output_rate)


# ----------------------------------------------------------------------------- setup
def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(argv: list[str], *, cwd: Path | None, log: Callable[[str], None]) -> None:
    log("$ " + " ".join(argv))
    subprocess.run(argv, cwd=str(cwd) if cwd else None, check=True)


def setup(
    config: Any,
    *,
    root: Path | None = None,
    build: bool = True,
    install_model: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """``speechforge engine-setup drama3``: the port's own setup, driven.

    1. ``git clone`` the pinned branch of audio.cpp into ``root`` (default
       ``<data>/engines/audio.cpp``) and check out the pinned commit.
    2. ``scripts/build_windows.ps1`` (or ``build_linux.sh``) for the CUDA
       preset with ``-ModelSet custom -Models fish_drama3``, targets
       ``audiocpp_server`` and ``audiocpp_cli``.
    3. ``tools/model_manager_v2.py install <package> --models-root
       <models>/audiocpp`` -- the port downloads and places the GGUF.

    Prints what is missing (git, cmake, nvcc, Visual Studio Build Tools)
    instead of guessing. Returns the paths to put in ``config.yaml``.
    """
    root = root or default_root(config)
    missing = [tool for tool in ("git", "cmake") if not _which(tool)]
    if build and not _which("nvcc"):
        missing.append("nvcc (CUDA toolkit)")
    if build and sys.platform == "win32" and not _vs_build_tools():
        missing.append("Visual Studio 2022 Build Tools (C++ desktop workload)")
    if missing:
        raise RuntimeError("engine-setup drama3 needs: " + ", ".join(missing))

    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", AUDIOCPP_BRANCH, AUDIOCPP_REPO, str(root)],
             cwd=None, log=log)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
                          capture_output=True, text=True).stdout.strip()
    if AUDIOCPP_COMMIT and head != AUDIOCPP_COMMIT:
        _run(["git", "fetch", "--depth", "1", "origin", AUDIOCPP_COMMIT], cwd=root, log=log)
        _run(["git", "checkout", "--quiet", AUDIOCPP_COMMIT], cwd=root, log=log)

    if build:
        if sys.platform == "win32":
            for target in ("audiocpp_server", "audiocpp_cli"):
                _run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                      str(root / "scripts" / "build_windows.ps1"),
                      "-Preset", "windows-cuda-release", "-ModelSet", "custom",
                      "-Models", "fish_drama3", "-Target", target], cwd=root, log=log)
        else:
            _run(["bash", str(root / "scripts" / "build_linux.sh"), "--preset",
                  "linux-cuda-release", "--models", "fish_drama3"], cwd=root, log=log)

    models_root = Path(config.models_dir) / "audiocpp"
    if install_model:
        models_root.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-I", str(root / "tools" / "model_manager_v2.py"), "install",
              MODEL_PACKAGE, "--models-root", str(models_root)], cwd=root, log=log)

    exe = default_server_exe(config) if root == default_root(config) else (
        root / "build" / ("windows-cuda-release" if sys.platform == "win32"
                          else "linux-cuda-release") / "bin" / _exe_name("audiocpp_server"))
    model = models_root / MODEL_TARGET_DIR / MODEL_FILE
    if model.is_file():
        log(f"verifying {model.name} (sha256, {MODEL_SIZE / 2**30:.1f} GiB) ...")
        digest = _sha256(model)
        if digest != MODEL_SHA256:
            raise RuntimeError(
                f"{model}: sha256 {digest[:12]}... does not match the pinned package "
                f"{MODEL_SHA256[:12]}...; delete it and run engine-setup again"
            )
        log("sha256 ok")
    out = {
        "root": str(root),
        "commit": head,
        "server_exe": str(exe),
        "server_exe_exists": exe.is_file(),
        "model_path": str(model),
        "model_exists": model.is_file(),
        "licence": MODEL_LICENCE,
        "attribution": "Built with Fish Audio",
    }
    log("")
    log("config.yaml:")
    log("  tts:")
    log(f"    engine: {NAME}")
    log("    drama3:")
    log("      backend: spawn")
    log(f"      server_exe: {exe}")
    log(f"      model_path: {model}")
    log("  voices:")
    log("    allow_reference_clip: true   # D3: this engine clones from the clip itself")
    return out


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _vs_build_tools() -> bool:
    roots = [
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
    ]
    for base in roots:
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            if (base / "Microsoft Visual Studio" / "2022" / edition / "VC" / "Auxiliary"
                    / "Build" / "vcvars64.bat").is_file():
                return True
    return False
