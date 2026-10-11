"""NVIDIA Nemotron Speech Streaming EN 0.6B (NVIDIA Open Model License; sherpa-onnx
export by the sherpa-onnx maintainers) through sherpa-onnx (Apache-2.0): the
cache-aware streaming A/B challenger for Parakeet's rolling partials (SPEC §5.1).

The encoder is a cache-aware FastConformer: every 160 ms chunk of audio is
encoded once against the kept cache (chunk size fixed at export; this engine
pins the 160 ms exports, which match ``realtime.partial_ms``), the RNN-T
decoder emits tokens as they become available, and token frame times give the
word timings. One stream per session (:class:`NemotronStream`); ``transcribe``
runs a whole clip through a fresh stream so the HTTP routes, the self-test and
``Speaker.transcribe_words`` need nothing new.

Two backends behind one interface (docs/ENGINES.md):

* ``sherpa-onnx`` -- in this process. Fine with the CPU provider, and with the
  CUDA provider on Linux (sherpa-onnx's CUDA 13 wheels there bundle ORT 1.28.2).
* ``subprocess`` -- :mod:`speechforge.engines.nemotron_worker` in a child
  interpreter (``stt.nemotron.python``), the only safe shape on Windows with
  ``gpu.provider: cuda``: sherpa-onnx's Windows GPU wheel carries its own
  CUDA 12 ONNX Runtime whose DLL names collide with this process's CUDA 13
  ``onnxruntime-gpu``. ``backend: auto`` picks this one there.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.config import NEMOTRON_INT8_REVISION, NEMOTRON_REVISION
from speechforge.engines.nemotron_worker import MAX_FEED_SAMPLES

log = logging.getLogger("speechforge.nemotron")

MODEL_DIR = "nemotron-speech-streaming-en-0.6b"  # under models_dir; also the lock key
LOCK_KEY = MODEL_DIR
SAMPLE_RATE = 16000
CHUNK_DIR = {160: "160ms"}


class NemotronConfigError(ValueError):
    pass


def model_files(chunk_ms: int, quantization: str | None) -> list[str]:
    """Relative paths under ``models_dir/MODEL_DIR`` (what models.lock lists)."""
    sub = CHUNK_DIR[chunk_ms]
    if quantization == "int8":
        parts = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"]
    else:
        parts = ["encoder.onnx", "encoder.data", "decoder.onnx", "joiner.onnx"]
    return ["tokens.txt", *(f"{sub}/{p}" for p in parts)]


@dataclass(frozen=True)
class NemotronOptions:
    backend: str  # "sherpa-onnx" | "subprocess" (auto already resolved)
    chunk_ms: int
    quantization: str | None
    provider: str  # "cuda" | "cpu"
    device_id: int
    num_threads: int
    decoding: str
    tail_pad_ms: int
    python: str
    worker_timeout_s: float

    @property
    def revision(self) -> str:
        return NEMOTRON_INT8_REVISION if self.quantization == "int8" else NEMOTRON_REVISION


def resolve_options(
    cfg: Any, *, provider: str, device_id: int, platform: str = sys.platform
) -> NemotronOptions:
    """Validate ``stt.nemotron`` against the provider and the platform.

    * ``auto`` -> ``subprocess`` on Windows + CUDA, else in-process.
    * In-process + CUDA on Windows is refused (the DLL clash is certain, not a
      risk: same-named ``onnxruntime*.dll``/``cudnn64_9.dll`` from two builds).
    * ``python`` only means something for ``subprocess``.
    * ``int8`` + CUDA is allowed but pointless (S4: no fast CUDA kernels for
      ``MatMulInteger``); the load logs a warning.
    """
    if provider not in ("cuda", "cpu"):
        raise NemotronConfigError(f"stt.nemotron: unsupported provider {provider!r}")
    if cfg.chunk_ms not in CHUNK_DIR:
        raise NemotronConfigError(
            f"stt.nemotron.chunk_ms: {cfg.chunk_ms} is not pinned in models.lock "
            f"(have {sorted(CHUNK_DIR)})"
        )
    backend = cfg.backend
    windows_cuda = platform == "win32" and provider == "cuda"
    if backend == "auto":
        backend = "subprocess" if windows_cuda else "sherpa-onnx"
    elif backend == "sherpa-onnx" and windows_cuda:
        raise NemotronConfigError(
            "stt.nemotron.backend: sherpa-onnx in-process with gpu.provider cuda on "
            "Windows would load sherpa-onnx's CUDA 12 ONNX Runtime beside this "
            "process's CUDA 13 onnxruntime-gpu (same DLL names); use backend: "
            "subprocess (with stt.nemotron.python pointing at a venv that holds the "
            "sherpa-onnx CUDA wheel) or gpu.provider: cpu"
        )
    if cfg.python and backend != "subprocess":
        raise NemotronConfigError(
            "stt.nemotron.python is only used by backend: subprocess (set backend: "
            "subprocess, or drop stt.nemotron.python)"
        )
    if cfg.python and not Path(cfg.python).exists():
        raise NemotronConfigError(f"stt.nemotron.python: {cfg.python!r} does not exist")
    return NemotronOptions(
        backend=backend,
        chunk_ms=int(cfg.chunk_ms),
        quantization=cfg.quantization,
        provider=provider,
        device_id=int(device_id),
        num_threads=int(cfg.num_threads),
        decoding=str(cfg.decoding),
        tail_pad_ms=int(cfg.tail_pad_ms),
        python=str(cfg.python or ""),
        worker_timeout_s=float(cfg.worker_timeout_s),
    )


def recognizer_kwargs(model_dir: Path, opts: NemotronOptions) -> dict[str, Any]:
    files = model_files(opts.chunk_ms, opts.quantization)
    names = {
        Path(f).name.split(".")[0]: str(model_dir / f)
        for f in files
        if f != "tokens.txt" and not f.endswith(".data")  # encoder.data rides along
    }
    return {
        "tokens": str(model_dir / "tokens.txt"),
        "encoder": names["encoder"],
        "decoder": names["decoder"],
        "joiner": names["joiner"],
        "provider": opts.provider,
        "device": opts.device_id,
        "num_threads": opts.num_threads,
        "decoding": opts.decoding,
        "tail_pad_ms": opts.tail_pad_ms,
    }


# ------------------------------------------------------------------ backends
class InProcessBackend:
    """:class:`~speechforge.engines.nemotron_worker.SherpaRecognizer` under a lock."""

    name = "sherpa-onnx"

    def __init__(self, kwargs: dict[str, Any]) -> None:
        from speechforge.engines.nemotron_worker import SherpaRecognizer

        self._rec = SherpaRecognizer(**kwargs)
        self._lock = threading.Lock()

    def create(self) -> int:
        with self._lock:
            return self._rec.create()

    def feed(self, sid: int, audio: np.ndarray) -> str:
        with self._lock:
            return self._rec.feed(sid, audio)

    def finalize(self, sid: int) -> tuple[str, list[tuple[str, float, float]]]:
        with self._lock:
            return self._rec.finalize(sid)

    def reset(self, sid: int) -> None:
        with self._lock:
            self._rec.reset(sid)

    def close_stream(self, sid: int) -> None:
        with self._lock:
            self._rec.close_stream(sid)

    def describe(self) -> dict[str, Any]:
        with self._lock:
            return self._rec.describe()

    def close(self) -> None:
        with self._lock:
            self._rec.close()


class WorkerError(RuntimeError):
    pass


class SubprocessBackend:
    """The same calls over pipes to ``nemotron_worker.py`` in another interpreter.

    One request in flight at a time (a lock); a reader thread hands lines back
    through a queue so a hung child is a :class:`WorkerError` after
    ``worker_timeout_s`` (and is killed), never a hung STT lane."""

    name = "subprocess"
    WORKER = Path(__file__).with_name("nemotron_worker.py")

    def __init__(self, kwargs: dict[str, Any], *, python: str, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self.python = python or sys.executable
        self._lock = threading.Lock()
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        self._proc = subprocess.Popen(  # noqa: S603 - our own worker, fixed argv
            [self.python, "-I", str(self.WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # the worker's log goes to ours
        )
        self._reader = threading.Thread(target=self._pump, name="sf-nemotron-pipe", daemon=True)
        self._reader.start()
        try:
            self.loaded = self._call({"op": "load", **kwargs})["describe"]
        except BaseException:
            self.close()
            raise

    def _pump(self) -> None:
        out = self._proc.stdout
        assert out is not None
        for line in out:
            self._lines.put(line)
        self._lines.put(None)

    def _call(self, cmd: dict[str, Any], audio: np.ndarray | None = None) -> dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                raise WorkerError(f"nemotron worker exited with {self._proc.returncode}")
            stdin = self._proc.stdin
            assert stdin is not None
            try:
                stdin.write((json.dumps(cmd) + "\n").encode("utf-8"))
                if audio is not None and len(audio):
                    stdin.write(np.ascontiguousarray(audio, dtype="<f4").tobytes())
                stdin.flush()
            except (OSError, ValueError) as exc:
                raise WorkerError(f"nemotron worker pipe closed: {exc}") from None
            try:
                line = self._lines.get(timeout=self.timeout_s)
            except queue.Empty:
                self._kill()
                raise WorkerError(
                    f"nemotron worker did not answer {cmd.get('op')} within {self.timeout_s} s"
                ) from None
            if line is None:
                raise WorkerError(f"nemotron worker exited with {self._proc.poll()}")
            reply = json.loads(line.decode("utf-8"))
            if "error" in reply:
                raise WorkerError(f"{reply.get('type', 'Error')}: {reply['error']}")
            return reply

    def create(self) -> int:
        return int(self._call({"op": "create"})["sid"])

    def feed(self, sid: int, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype="<f4")
        if len(audio) > MAX_FEED_SAMPLES:
            # Refused here, before anything is written: the child would exit on
            # it (protocol error) and take every open stream with it.
            raise WorkerError(f"feed of {len(audio)} samples exceeds {MAX_FEED_SAMPLES}")
        return str(self._call({"op": "feed", "sid": sid, "n": int(len(audio))}, audio)["text"])

    def finalize(self, sid: int) -> tuple[str, list[tuple[str, float, float]]]:
        r = self._call({"op": "finalize", "sid": sid})
        return str(r["text"]), [(str(w), float(s), float(e)) for w, s, e in r.get("words", [])]

    def reset(self, sid: int) -> None:
        self._call({"op": "reset", "sid": sid})

    def close_stream(self, sid: int) -> None:
        self._call({"op": "close_stream", "sid": sid})

    def describe(self) -> dict[str, Any]:
        return dict(self.loaded)

    def _kill(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5)

    def close(self) -> None:
        try:
            if self._proc.poll() is None:
                self._call({"op": "close"})
                self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - closing never fails an unload
            pass
        finally:
            self._kill()


# ------------------------------------------------------------------ adapters
class NemotronStream:
    """:class:`~speechforge.engines.base.SttStreamHandle` over one backend stream."""

    def __init__(self, backend: Any) -> None:
        self._b = backend
        self._sid: int | None = backend.create()
        self.fed_samples = 0

    def _id(self) -> int:
        if self._sid is None:
            raise RuntimeError("stream closed")
        return self._sid

    def feed(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        self.fed_samples += len(audio)
        return self._b.feed(self._id(), audio)

    def finalize(self) -> tuple[str, list[tuple[str, float, float]]]:
        return self._b.finalize(self._id())

    def reset(self) -> None:
        self.fed_samples = 0
        self._b.reset(self._id())

    def close(self) -> None:
        if self._sid is not None:
            sid, self._sid = self._sid, None
            self._b.close_stream(sid)


class NemotronStt:
    """:class:`~speechforge.engines.base.StreamingSttAdapter` (and so also an
    ``SttAdapter``) over an in-process or subprocess sherpa-onnx backend."""

    name = "nemotron"
    sample_rate = SAMPLE_RATE

    def __init__(self, backend: Any, opts: NemotronOptions) -> None:
        self.backend = backend
        self.opts = opts

    def create_stream(self) -> NemotronStream:
        return NemotronStream(self.backend)

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
        """Whole clip through a fresh stream. Under 0.1 s is silence by definition."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < SAMPLE_RATE // 10:
            return "", []
        s = self.create_stream()
        try:
            s.feed(audio)
            return s.finalize()
        finally:
            s.close()

    def transcribe(self, audio: np.ndarray) -> str:
        return self.transcribe_words(audio)[0]

    def warm_up(self) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, np.float32))

    def describe(self) -> dict[str, Any]:
        o = self.opts
        return {
            "name": MODEL_DIR,
            "weights": "int8" if o.quantization == "int8" else "fp32",
            "provider": o.provider,
            "backend": o.backend,
            "chunk_ms": o.chunk_ms,
            "device": o.device_id,
            "revision": o.revision[:8],
            "streaming": True,
            "lock_key": LOCK_KEY,
        }

    def close(self) -> None:
        self.backend.close()
        self.backend = None


def load(config: Any, device_id: int) -> NemotronStt:
    cfg = config.stt.nemotron
    opts = resolve_options(cfg, provider=config.gpu.provider, device_id=device_id)
    model_dir = Path(config.models_dir) / MODEL_DIR
    files = model_files(opts.chunk_ms, opts.quantization)
    missing = [f for f in files if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Nemotron files missing in {model_dir}: {', '.join(missing)} "
            f"(run `speechforge fetch-models`)"
        )
    if opts.quantization == "int8" and opts.provider == "cuda":
        log.warning("stt.nemotron.quantization int8 on CUDA: no fast int8 kernels (S4)")
    kwargs = recognizer_kwargs(model_dir, opts)
    backend: Any
    if opts.backend == "subprocess":
        backend = SubprocessBackend(kwargs, python=opts.python, timeout_s=opts.worker_timeout_s)
    else:
        backend = InProcessBackend(kwargs)
    log.info(
        "nemotron loaded: backend=%s provider=%s device=%s chunk_ms=%s %s",
        opts.backend,
        opts.provider,
        opts.device_id,
        opts.chunk_ms,
        backend.describe(),
    )
    return NemotronStt(backend, opts)
