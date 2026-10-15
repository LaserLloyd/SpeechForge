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

Voices (S34). A voice is the reference's ten-codebook VQ grid plus its
transcript. A clip-plus-transcript file (``dotstts-ref``) is encoded ONCE per
process through the server's ``encode_reference_only`` task and the codes are
cached; a codes-only file (the same container with ``vq_codes``) needs no
encode at all. Every request sends ``options.references`` with the codes, so
the server never re-encodes and the body is ~14 KB instead of ~800 KB. What
remains is the backbone pre-filling the reference rows on every request, and
how much that costs depends on the CUDA toolkit the port was BUILT with
(S35): under CUDA 13.4 the router's top-k compiles into a per-row host loop
(~2.4 ms per row, ~55 ms per second of reference); under CUDA 13.1 the same
source pre-fills at ~0.15 ms per row, so a 12.6 s reference reaches first
audio in ~130 ms and a 6 s one in ~100 ms, and seeds reproduce.
``engine-setup`` builds with 13.1 when it finds it. ``take()`` keeps a voice
the model improvised (generate with no reference, store the take's codes).

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
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.audio import (
    TTS_SAMPLE_RATE,
    _poly_filters,
    float_to_pcm16,
    read_wav,
    resample_poly,
)
from speechforge.engines.dots_tts import (
    MAX_INFLATED_BYTES,
    MAX_VOICE_BYTES,
    DotsVoice,
    make_voice,
)
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
#: The per-voice prompt-prefix cache for the port (S36), shipped as a patch
#: against ``AUDIOCPP_COMMIT`` and applied by ``engine-setup``. SpeechForge's
#: own code (MIT) on top of audio.cpp (Apache-2.0); offered upstream.
PREFIX_CACHE_PATCH = Path(__file__).with_name("patches") / "audiocpp-drama3-prefix-cache.patch"

MODEL_LICENCE = (
    "Fish Audio Community License v1.0 (commercial terms at fish.audio above USD 500k revenue)"
)

#: The VQ grid: ten codebooks (one semantic, nine residual), ~21.5 frames per
#: second of audio, every code below 4096.
CODEBOOKS = 10
CODE_LIMIT = 4096
MAX_CODE_FRAMES = 2048  # ~95 s; the server's default context is 2048 rows in all
#: Encoded references kept per process (clip-voice ids -> codes).
CODES_CACHE = 64

__all__ = [
    "NAME",
    "VOICE_FORMAT",
    "Drama3Tts",
    "Drama3Voice",
    "PolyphaseResampler",
    "load",
    "parse_voice",
    "make_voice",
    "make_voice_codes",
    "setup",
    "server_config",
]


# ----------------------------------------------------------------------------- voices
NO_TRANSCRIPT = (
    "drama3-tts needs the reference clip's transcript: the server refuses a reference "
    "without `reference_text`. Build the voice with make_voice(audio, rate, transcript) "
    "(the dotstts-ref container) and PUT that; /v1/voices/from-reference carries no transcript."
)


@dataclass(frozen=True)
class Drama3Voice:
    """A Drama 3 voice: the transcript and either a clip (encoded once) or
    the reference's VQ codes (``vq_codes``, int32 ``(10, frames)``), or both."""

    transcript: str
    audio: np.ndarray | None  # float32 mono at ``rate``; None for a codes-only voice
    rate: int
    vq_codes: np.ndarray | None

    @property
    def seconds(self) -> float:
        if self.audio is None:
            return 0.0
        return len(self.audio) / self.rate

    def clip(self, max_s: float) -> DotsVoice:
        """The clip cut to ``max_s`` as the dots container's voice (for ``wav()``)."""
        assert self.audio is not None
        limit = int(float(max_s) * self.rate)
        audio = self.audio[:limit] if len(self.audio) > limit else self.audio
        return DotsVoice(audio, self.rate, self.transcript)


def make_voice_codes(
    transcript: str,
    vq_codes: np.ndarray,
    audio: np.ndarray | None = None,
    rate: int = TTS_SAMPLE_RATE,
) -> bytes:
    """``dotstts-ref`` bytes carrying the reference's VQ codes (no encode at use).
    ``audio`` (a preview, or the clip the codes came from) is optional."""
    codes = np.ascontiguousarray(np.asarray(vq_codes, np.int32))
    _check_codes(codes)
    buf = io.BytesIO()
    arrays: dict[str, Any] = {
        "format": np.array(VOICE_FORMAT),
        "transcript": np.array(str(transcript or "")),
        "vq_codes": codes,
        "rate": np.array(int(rate), np.int64),
    }
    if audio is not None and len(audio):
        arr = np.asarray(audio)
        arrays["audio"] = np.ascontiguousarray(
            float_to_pcm16(arr) if arr.dtype.kind == "f" else arr, "<i2")
    else:
        arrays["audio"] = np.zeros(0, "<i2")
    np.savez(buf, **arrays)
    return buf.getvalue()


def _check_codes(codes: np.ndarray) -> None:
    if codes.ndim != 2 or codes.shape[0] != CODEBOOKS or codes.shape[1] < 1:
        raise BadRequest(
            f"vq_codes: expected ({CODEBOOKS}, frames), got {tuple(codes.shape)}",
            code="bad_voice")
    if codes.shape[1] > MAX_CODE_FRAMES:
        raise BadRequest(
            f"vq_codes: {codes.shape[1]} frames; at most {MAX_CODE_FRAMES}", code="bad_voice")
    if codes.dtype.kind not in "iu" or int(codes.min()) < 0 or int(codes.max()) >= CODE_LIMIT:
        raise BadRequest(f"vq_codes: integers in [0, {CODE_LIMIT}) expected", code="bad_voice")


def parse_voice(data: bytes) -> Drama3Voice:
    """Validate ``dotstts-ref`` bytes for Drama 3: a clip AND its transcript, or
    the transcript AND ``vq_codes`` (clip optional). Raises 400 ``bad_voice``."""
    if len(data) > MAX_VOICE_BYTES:
        raise BadRequest(f"voice file is larger than {MAX_VOICE_BYTES} bytes", code="bad_voice")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
        if total > MAX_INFLATED_BYTES:
            raise BadRequest("voice file inflates too large", code="bad_voice")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            names = set(z.files)
            has_codes = "vq_codes" in names
            codes = np.asarray(z["vq_codes"]) if has_codes else None
            transcript = str(np.asarray(z["transcript"])) if "transcript" in names else ""
            rate = int(np.asarray(z["rate"])) if "rate" in names else TTS_SAMPLE_RATE
            audio = np.asarray(z["audio"]) if "audio" in names else None
    except BadRequest:
        raise
    except Exception as exc:  # zipfile.BadZipFile, ValueError (pickled object), OSError
        raise BadRequest(f"not a valid voice npz ({type(exc).__name__})", code="bad_voice") from exc
    if not has_codes:
        ref = _parse_ref(data)
        if not ref.transcript:
            raise BadRequest(NO_TRANSCRIPT, code="bad_voice")
        return Drama3Voice(ref.transcript, ref.audio, ref.rate, None)
    if not transcript.strip():
        raise BadRequest(NO_TRANSCRIPT, code="bad_voice")
    assert codes is not None
    _check_codes(codes)
    clip: np.ndarray | None = None
    if audio is not None and np.size(audio):
        if audio.ndim != 1 or audio.dtype not in (np.dtype("<i2"), np.dtype("<f4")):
            raise BadRequest("audio: expected mono int16 or float32", code="bad_voice")
        clip = (audio.astype(np.float32) / 32768.0 if audio.dtype.kind == "i"
                else np.asarray(audio, np.float32))
    return Drama3Voice(transcript.strip(), clip, rate, np.ascontiguousarray(codes, np.int32))


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
                    # Only a patched build knows the option (S36); a stock one
                    # refuses it, so `prefix_cache: false` keeps it out.
                    **({"fish_drama3.prefix_cache_entries": int(cfg.prefix_cache_entries)}
                       if cfg.prefix_cache else {}),
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
        # Clip voices encoded once per process: voice id -> codes (S34).
        self._codes: OrderedDict[str, list[list[int]]] = OrderedDict()
        self._codes_lock = threading.Lock()
        # The encode task takes a server-side PATH: the clip is written here
        # for the call and removed after (loopback server, same machine).
        self.ref_dir = Path(config.data_dir) / "drama3-refs"
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

    # -- voices --------------------------------------------------------------
    def codes_for(self, voice: Drama3Voice, voice_id: str) -> list[list[int]]:
        """The reference's VQ codes: from the file, or encoded once and cached."""
        if voice.vq_codes is not None:
            return voice.vq_codes.tolist()
        if voice.audio is None:
            raise RuntimeError(NO_TRANSCRIPT)
        key = f"{voice_id}:{len(voice.audio)}:{voice.rate}:{float(self.cfg.max_ref_s)}"
        with self._codes_lock:
            hit = self._codes.get(key)
            if hit is not None:
                self._codes.move_to_end(key)
                return hit
        codes = self.encode_reference(voice.clip(self.cfg.max_ref_s), voice_id)
        with self._codes_lock:
            self._codes[key] = codes
            while len(self._codes) > CODES_CACHE:
                self._codes.popitem(last=False)
        return codes

    def encode_reference(self, clip: DotsVoice, voice_id: str) -> list[list[int]]:
        """The server's ``encode_reference_only`` task (``POST /v1/tasks/run``):
        the clip's VQ codes and nothing else, ~35-100 ms. The route takes a
        server-side path, so the clip is written next to the data dir for the
        call and deleted after."""
        if not clip.transcript:
            raise RuntimeError(NO_TRANSCRIPT)
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch for ch in voice_id if ch.isalnum() or ch in "-_")[:80] or "ref"
        path = self.ref_dir / f"{safe}-{os.getpid()}-{threading.get_ident()}.wav"
        path.write_bytes(clip.wav())
        try:
            body = {
                "model": MODEL_ID,
                "request": {
                    "text": "Prepare the reference.",
                    "voice_ref": str(path),
                    "options": {"reference_text": clip.transcript,
                                "encode_reference_only": True},
                },
            }
            r = self.client.post(f"{self.url}/v1/tasks/run", json=body)
            if r.status_code != 200:
                raise RuntimeError(
                    f"drama3 server refused the reference encode ({r.status_code}): "
                    f"{r.text[:300]}")
            for art in r.json().get("artifacts") or []:
                meta = art.get("meta") or {}
                if str(meta.get("schema") or "").startswith("drama3_reference"):
                    payload = json.loads(base64.b64decode(art["payload"]).decode("utf-8"))
                    codes = np.asarray(payload["vq_codes"], np.int32)
                    _check_codes(codes)
                    return codes.tolist()
            raise RuntimeError("drama3 server returned no drama3_reference artifact")
        finally:
            with contextlib.suppress(OSError):
                path.unlink()

    def take(self, text: str, seed: int | None) -> tuple[np.ndarray, int, list[list[int]]]:
        """One take in the model's own improvised voice (``POST /v1/tasks/run``,
        no reference): ``(float32 audio, rate, codes)``. The codes are the voice."""
        req: dict[str, Any] = {"text": text, "max_tokens": int(self.cfg.max_tokens)}
        if seed is not None:
            req["seed"] = int(seed)
        r = self.client.post(f"{self.url}/v1/tasks/run", json={"model": MODEL_ID, "request": req})
        if r.status_code != 200:
            raise RuntimeError(f"drama3 server refused the take ({r.status_code}): {r.text[:300]}")
        doc = r.json()
        audio, rate = read_wav(base64.b64decode(doc["audio"]))
        for art in doc.get("artifacts") or []:
            meta = art.get("meta") or {}
            if str(meta.get("schema") or "").startswith("drama3_take"):
                payload = json.loads(base64.b64decode(art["payload"]).decode("utf-8"))
                codes = np.asarray(payload["vq_codes"], np.int32)
                _check_codes(codes)
                return np.asarray(audio, np.float32), int(rate), codes.tolist()
        raise RuntimeError("drama3 server returned no drama3_take artifact")

    # -- requests ----------------------------------------------------------
    def _body(
        self, voice: Drama3Voice | None, voice_id: str, text: str, seed: int | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": MODEL_ID,
            "input": text,
            "response_format": "pcm",
            "stream_format": "audio",
            "max_tokens": int(self.cfg.max_tokens),
        }
        if voice is not None:
            if not voice.transcript:
                raise RuntimeError(NO_TRANSCRIPT)
            # Codes, never audio: the server skips its encoder and the body is
            # ~14 KB for a 12 s reference instead of ~800 KB.
            body["options"] = {"references": [
                {"text": voice.transcript, "vq_codes": self.codes_for(voice, voice_id)}]}
        if seed is not None:
            body["seed"] = int(seed)
        return body

    def synth(
        self, voice: Drama3Voice | None, voice_id: str, text: str, *, seed: int | None
    ) -> Iterator[np.ndarray]:
        """``voice=None`` speaks with the model's own improvised voice (warm-up)."""
        carry = b""
        with self.client.stream(
            "POST", f"{self.url}/v1/audio/speech", json=self._body(voice, voice_id, text, seed)
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
            "cached_voices": len(self._codes),
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

    def parse_voice(self, data: bytes) -> Drama3Voice:
        return parse_voice(data)

    def take(self, text: str, seed: int | None = None) -> tuple[bytes, np.ndarray]:
        """Keep this voice: one take in an improvised voice, returned as
        ``(voice bytes, preview pcm16 at sample_rate)``. The voice file carries
        the take's codes (no encode on use) and the preview as its clip."""
        line = (text or "").strip()
        if not line:
            raise BadRequest("text is required", code="bad_request")
        with self._lock:
            audio, rate, codes = self.backend.take(line, seed)
        preview = (resample_poly(audio, rate, self.sample_rate)
                   if rate != self.sample_rate else audio)
        voice = make_voice_codes(line, np.asarray(codes, np.int32), preview, self.sample_rate)
        return voice, float_to_pcm16(preview)

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        """``POST /v1/voices/from-reference`` carries no transcript, and Drama 3
        cannot clone without one: refused with the way that works."""
        raise BadRequest(NO_TRANSCRIPT, code="bad_voice")

    def stream(
        self,
        text: str,
        voice_id: str,
        voice: Drama3Voice | None,
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

    def render(self, text: str, voice: Drama3Voice, *, seed: int | None = None) -> np.ndarray:
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
            "voice_codes": True,   # the container may carry vq_codes; clips are encoded once
            "keep_voice": True,    # take(): POST /v1/voices/surprise
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


def _run(argv: list[str], *, cwd: Path | None, log: Callable[[str], None],
         env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(argv))
    subprocess.run(argv, cwd=str(cwd) if cwd else None, check=True, env=env)


#: The toolkit the port is built with decides the pre-fill speed (S35): CUDA
#: 13.4's CUB has a top-k that the port calls once per row per layer from the
#: host (~2.4 ms per prompt row); CUDA 13.1's CUB has none, the op falls back
#: to one sort per layer, and pre-fill runs at ~0.15 ms per row.
PREFERRED_CUDA = "13.1"


def find_cuda_toolkit(version: str = PREFERRED_CUDA) -> Path | None:
    """The CUDA toolkit root for ``version`` on this machine, or None."""
    key = "CUDA_PATH_V" + version.replace(".", "_")
    if os.environ.get(key):
        root = Path(os.environ[key])
        if root.is_dir():
            return root
    candidates = []
    if sys.platform == "win32":
        base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        candidates.append(base / "NVIDIA GPU Computing Toolkit" / "CUDA" / f"v{version}")
    else:
        candidates += [Path(f"/usr/local/cuda-{version}"), Path(f"/opt/cuda-{version}")]
    for root in candidates:
        if (root / "bin").is_dir():
            return root
    return None


def build_env(cuda_toolkit: Path | None) -> dict[str, str]:
    """The environment for the port's build script: CMake picks the compiler
    from ``CUDACXX`` (``CUDA_PATH`` alone is not enough, measured)."""
    env = dict(os.environ)
    if cuda_toolkit is not None:
        nvcc = cuda_toolkit / "bin" / _exe_name("nvcc")
        env["CUDACXX"] = str(nvcc)
        env["CUDA_PATH"] = str(cuda_toolkit)
        env["CUDAToolkit_ROOT"] = str(cuda_toolkit)
        env["PATH"] = str(cuda_toolkit / "bin") + os.pathsep + env.get("PATH", "")
    return env


def setup(
    config: Any,
    *,
    root: Path | None = None,
    build: bool = True,
    install_model: bool = True,
    cuda_toolkit: Path | None = None,
    cuda_arch: str = "auto",
    prefix_cache: bool = True,
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

    The build uses CUDA 13.1 when it is installed (``cuda_toolkit`` names
    another root; see ``PREFERRED_CUDA``) and says so when it is not, and
    applies the shipped prefix-cache patch first (``prefix_cache=False`` builds
    the stock port; then set ``tts.drama3.prefix_cache: false``).
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

    patched = apply_prefix_cache_patch(root, log=log) if prefix_cache else False
    toolkit = cuda_toolkit or (find_cuda_toolkit() if build else None)
    if build:
        if toolkit is None:
            log(f"note: CUDA {PREFERRED_CUDA} not found; building with the default toolkit. "
                f"With CUDA 13.4 the port pre-fills at ~2.4 ms per prompt row instead of "
                f"~0.15 (S35): cloned voices will start ~0.5-0.9 s late.")
        else:
            log(f"building with CUDA toolkit {toolkit}")
        env = build_env(toolkit)
        if sys.platform == "win32":
            for target in ("audiocpp_server", "audiocpp_cli"):
                _run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                      str(root / "scripts" / "build_windows.ps1"),
                      "-Preset", "windows-cuda-release", "-ModelSet", "custom",
                      "-Models", "fish_drama3", "-CudaArchitectures", str(cuda_arch),
                      "-Target", target], cwd=root, log=log, env=env)
        else:
            _run(["bash", str(root / "scripts" / "build_linux.sh"), "--preset",
                  "linux-cuda-release", "--models", "fish_drama3"], cwd=root, log=log,
                 env=env)

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
        "cuda_toolkit": str(toolkit) if toolkit else "",
        "prefix_cache_patch": patched,
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
    log(f"      prefix_cache: {'true' if patched else 'false'}")
    log("  voices:")
    log("    allow_reference_clip: true   # D3: this engine clones from the clip itself")
    return out


def apply_prefix_cache_patch(root: Path, *, log: Callable[[str], None] = print) -> bool:
    """``git apply`` the shipped prefix-cache patch to the checkout at ``root``;
    True when the tree carries it afterwards (already applied counts)."""
    patch = PREFIX_CACHE_PATCH
    if not patch.is_file():
        log(f"note: {patch.name} is not in this install; building the stock port")
        return False
    applied = subprocess.run(["git", "apply", "--check", "--reverse", str(patch)],
                             cwd=str(root), capture_output=True)
    if applied.returncode == 0:
        log("prefix-cache patch already applied")
        return True
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=str(root),
                           capture_output=True, text=True)
    if check.returncode != 0:
        log(f"note: the prefix-cache patch does not apply to this checkout "
            f"({check.stderr.strip()[:200]}); building the stock port")
        return False
    _run(["git", "apply", str(patch)], cwd=root, log=log)
    return True


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
