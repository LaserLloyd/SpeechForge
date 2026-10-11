""":class:`~speechforge.engines.base.TtsAdapter` for dots.tts-mf (phase-2 bake-off).

dots.tts (Apache-2.0 code and weights) is a 2B continuous autoregressive TTS
model: a text LLM, a flow-matching acoustic head over a 48 kHz AudioVAE and a
BigVGAN-style vocoder. It clones a voice from a **reference clip plus its
transcript** at synthesis time ("continuation cloning"; the clip alone gives the
weaker x-vector-only cloning), so the clip itself has to be on the rig. That is
decision D3: this engine loads only with ``voices.allow_reference_clip: true``.

Voice format ``dotstts-ref``: an ``.npz`` (``allow_pickle=False``) with ``audio``
(mono int16 or float32, 0.5-20 s), ``rate`` (Hz) and an optional ``transcript``
(<= 1000 chars). :func:`make_voice` builds one, :func:`parse_voice` validates one.

Two backends behind one small protocol (:class:`Backend`), chosen by
``tts.dots.backend``:

* ``native`` -- the ``dots_tts`` PyTorch runtime in this process on the leased
  card (``torch.cuda.device(device_id)`` around every call, so the ONNX engines
  and this one agree on the ordinal). The reference clip never touches the disk:
  the runtime's prompt-audio loader is pointed at the in-memory tensor.
* ``http`` -- an SGLang Omni ``POST /v1/audio/speech`` server (Linux / WSL2 on
  this machine) over loopback only; ``stream: true`` + ``response_format: pcm``
  gives s16le 48 kHz mono as it is generated.

Streaming: the model yields audio every patch (160 ms at 48 kHz; the vocoder
merges later patches four at a time), so first audio does not wait for the
sentence. ``stop`` is checked between patches, the credit ``gate`` before each
one; the GPU lock is held per patch, so two contexts interleave. The output is
decimated to ``tts.dots.output_rate`` (24 kHz by default: the sf.1 frame rate).

Incremental text (``tts.dots.double_streaming``, native backend, the
``dots.tts-mf-2steps-stts`` artifact): ``stream(..., options={"text_feed": it})``
pulls further text pieces from ``it`` (an iterator of ``str``; exhaustion ends
the sentence) and pushes their tokens into one double-streaming session while
audio comes back. Without a session-capable backend the pieces are buffered and
spoken as one unit.

Heavy imports (torch, dots_tts, httpx) happen inside :func:`load` only.
"""

from __future__ import annotations

import base64
import io
import itertools
import os
import pickle
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from speechforge.audio import TTS_SAMPLE_RATE, float_to_pcm16, wav_bytes
from speechforge.errors import BadRequest

NAME = "dots-tts"
VOICE_FORMAT = "dotstts-ref"
NATIVE_RATE = 48000
#: One latent patch = 4 frames x 1920 samples (the vocoder's upsample product).
PATCH_SAMPLES = 7680
#: The sf.1 credit unit (tts_ctx.TOKEN_MS): the gate counts 40 ms "tokens".
TOKEN_MS = 40

HF_REPO = "dots-studio/dots.tts-mf"
HF_REVISION = "c28105adc8228143392b4e346994ff613ee48a06"
HF_STTS_REPO = "dots-studio/dots.tts-mf-2steps-stts"
HF_STTS_REVISION = "9e7be6817ceda3fd6474af4d089ca6227918ccec"
MODEL_DIR = "dots-tts-mf"
STTS_MODEL_DIR = "dots-tts-mf-2steps-stts"
PACKAGE_VERSION = "0.3.1"  # the dots.tts release the private-API hooks were read against

MIN_REF_SECONDS = 0.5
MAX_REF_SECONDS = 20.0
MIN_RATE = 8000
MAX_RATE = 96000
MAX_TRANSCRIPT_CHARS = 1000
MAX_VOICE_BYTES = 8 * 1024 * 1024
#: Inflated npz members: 20 s x 96 kHz x float32 = 7.7 MB, plus slack.
MAX_INFLATED_BYTES = 16 * 1024 * 1024

#: Globals a trusted ``latent_stats.pt`` may name (a dict of two numpy arrays,
#: or the torch tensor equivalent). Anything else is refused before torch.load.
LATENT_STATS_GLOBALS = frozenset(
    {
        "numpy.core.multiarray _reconstruct",
        "numpy._core.multiarray _reconstruct",
        "numpy.core.multiarray scalar",
        "numpy._core.multiarray scalar",
        "numpy ndarray",
        "numpy dtype",
        "_codecs encode",
        "collections OrderedDict",
        "torch._utils _rebuild_tensor_v2",
        "torch FloatStorage",
        "torch DoubleStorage",
        "torch HalfStorage",
        "torch BFloat16Storage",
    }
)
MAX_LATENT_STATS_BYTES = 1 << 20


# ----------------------------------------------------------------------------- voices
@dataclass(frozen=True)
class DotsVoice:
    """A validated reference: mono float32 ``audio`` at ``rate`` and its transcript."""

    audio: np.ndarray
    rate: int
    transcript: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.rate

    def to_npz(self) -> bytes:
        return make_voice(self.audio, self.rate, self.transcript)

    def wav(self) -> bytes:
        return wav_bytes(float_to_pcm16(self.audio), self.rate)


def make_voice(audio: np.ndarray, rate: int, transcript: str = "") -> bytes:
    """``dotstts-ref`` bytes from a mono clip (float32 in [-1, 1] or int16)."""
    arr = np.asarray(audio)
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    if arr.dtype.kind == "f":
        arr = float_to_pcm16(arr)
    buf = io.BytesIO()
    np.savez(
        buf,
        format=np.array(VOICE_FORMAT),
        audio=np.ascontiguousarray(arr, "<i2"),
        rate=np.array(int(rate), np.int64),
        transcript=np.array(str(transcript or "")),
    )
    return buf.getvalue()


def parse_voice(data: bytes) -> DotsVoice:
    """Validate ``dotstts-ref`` bytes. Raises 400 ``bad_voice``."""
    if len(data) > MAX_VOICE_BYTES:
        raise BadRequest(f"voice file is larger than {MAX_VOICE_BYTES} bytes", code="bad_voice")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
        if total > MAX_INFLATED_BYTES:
            raise BadRequest("voice file inflates too large", code="bad_voice")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            names = set(z.files)
            if "audio" not in names or "rate" not in names:
                raise BadRequest(
                    "voice file lacks 'audio' / 'rate' (format dotstts-ref: a reference "
                    "clip and its transcript, see docs/ENGINES.md)",
                    code="bad_voice",
                )
            fmt = z["format"] if "format" in names else None
            audio = np.asarray(z["audio"])
            rate_arr = np.asarray(z["rate"])
            transcript_arr = np.asarray(z["transcript"]) if "transcript" in names else None
    except BadRequest:
        raise
    except Exception as exc:  # zipfile.BadZipFile, ValueError (pickled object), OSError
        raise BadRequest(f"not a valid voice npz ({type(exc).__name__})", code="bad_voice") from exc

    if fmt is not None and (
        fmt.ndim != 0 or fmt.dtype.kind not in "US" or str(fmt) != VOICE_FORMAT
    ):
        raise BadRequest(f"voice file format is not {VOICE_FORMAT!r}", code="bad_voice")
    if rate_arr.ndim != 0 or rate_arr.dtype.kind not in "iu":
        raise BadRequest("rate: expected one integer (Hz)", code="bad_voice")
    rate = int(rate_arr)
    if not MIN_RATE <= rate <= MAX_RATE:
        raise BadRequest(f"rate: {rate} Hz is outside {MIN_RATE}-{MAX_RATE}", code="bad_voice")
    if audio.ndim == 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 1 or audio.dtype not in (np.dtype("<i2"), np.dtype("<f4")):
        raise BadRequest(
            f"audio: expected mono int16 or float32, got shape {audio.shape} / {audio.dtype}",
            code="bad_voice",
        )
    seconds = len(audio) / rate
    if not MIN_REF_SECONDS <= seconds <= MAX_REF_SECONDS:
        raise BadRequest(
            f"audio: {seconds:.2f} s; a reference must be {MIN_REF_SECONDS}-{MAX_REF_SECONDS} s",
            code="bad_voice",
        )
    if audio.dtype.kind == "f":
        if not np.isfinite(audio).all():
            raise BadRequest("audio: non-finite samples", code="bad_voice")
        if float(np.max(np.abs(audio))) > 1.0001:
            raise BadRequest("audio: float samples must be within [-1, 1]", code="bad_voice")
        pcm = np.asarray(audio, np.float32)
    else:
        pcm = audio.astype(np.float32) / 32768.0
    transcript = ""
    if transcript_arr is not None:
        if transcript_arr.ndim != 0 or transcript_arr.dtype.kind not in "US":
            raise BadRequest("transcript: expected one string", code="bad_voice")
        transcript = str(transcript_arr).strip()
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            raise BadRequest(
                f"transcript: longer than {MAX_TRANSCRIPT_CHARS} characters", code="bad_voice"
            )
        if any(ord(c) < 32 and c not in "\t\n\r" for c in transcript):
            raise BadRequest("transcript: control characters", code="bad_voice")
    return DotsVoice(np.ascontiguousarray(pcm, np.float32), rate, transcript)


# ----------------------------------------------------------------------------- safety
def check_latent_stats(path: Path) -> None:
    """Refuse a ``latent_stats.pt`` that names anything but numpy/torch array
    globals. dots_tts loads this one file with ``torch.load(weights_only=False)``
    (every weight is safetensors); models.lock pins its sha256 and this scan is
    the second lock, run before the runtime touches it (SPEC 8.5). The scan is a
    restricted unpickle: every global resolves to an inert stub or the file is
    refused, so no library code runs on the file's say-so."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"{path.name} is missing")
    if path.stat().st_size > MAX_LATENT_STATS_BYTES:
        raise RuntimeError(f"{path.name} is implausibly large for latent statistics")
    data = path.read_bytes()
    pickles: list[bytes] = []
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".pkl"):
                    if info.file_size > MAX_LATENT_STATS_BYTES:
                        raise RuntimeError(f"{path.name}: {info.filename} too large")
                    pickles.append(zf.read(info))
        if not pickles:
            raise RuntimeError(f"{path.name}: no pickle member in the archive")
    else:
        pickles.append(data)
    for blob in pickles:
        _scan_pickle(blob, path.name)


class _Stub:
    """Stands in for every allowed global: callable, buildable, inert."""

    def __call__(self, *_args: Any, **_kwargs: Any) -> _Stub:
        return _Stub()

    def __setstate__(self, _state: Any) -> None:
        pass


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, blob: bytes, name: str) -> None:
        super().__init__(io.BytesIO(blob))
        self.name = name

    def find_class(self, module: str, name: str) -> Any:
        ref = f"{module} {name}"
        if ref not in LATENT_STATS_GLOBALS:
            raise RuntimeError(f"{self.name}: pickle names {ref!r}; refusing to load it")
        return _Stub()

    def persistent_load(self, _pid: Any) -> Any:
        return _Stub()


def _scan_pickle(blob: bytes, name: str) -> None:
    try:
        _RestrictedUnpickler(blob, name).load()
    except RuntimeError:
        raise
    except Exception as exc:  # truncated, corrupt, or a shape the stubs cannot stand in for
        raise RuntimeError(f"{name}: unreadable pickle ({type(exc).__name__})") from exc


# ----------------------------------------------------------------------------- backends
class TextSession(Protocol):
    """One double-streaming utterance: push text, get audio as it is ready."""

    def push(self, text: str) -> Iterable[np.ndarray]: ...  # float32 mono @ native rate

    def finish(self) -> Iterable[np.ndarray]: ...

    def close(self) -> None: ...


class Backend(Protocol):
    native_rate: int
    text_stream: bool  # open_session() works

    def synth(
        self, voice: DotsVoice, voice_id: str, text: str, *, seed: int | None
    ) -> Iterator[np.ndarray]: ...  # float32 mono chunks at native_rate, as generated

    def open_session(self, voice: DotsVoice, voice_id: str, *, seed: int | None) -> TextSession: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class Decimator:
    """Streaming 48 kHz -> 24 kHz: a 63-tap Kaiser-windowed half-band low-pass,
    then every second sample, with the FIR tail and the phase carried between
    chunks so a sentence decimated piecewise equals the whole. Ratio 1 passes."""

    TAPS = 63

    def __init__(self, src: int, dst: int) -> None:
        if src == dst:
            self.ratio = 1
        elif src == 2 * dst:
            self.ratio = 2
        else:
            raise ValueError(f"unsupported rate change {src} -> {dst}")
        n = self.TAPS
        m = (n - 1) / 2
        k = np.arange(n) - m
        fc = 0.25  # cycles per input sample: half of the output Nyquist
        h = 2 * fc * np.sinc(2 * fc * k) * np.kaiser(n, 8.0)
        self._h = (h / h.sum()).astype(np.float32)
        self._carry = np.zeros(0, np.float32)
        self._phase = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1)
        if self.ratio == 1:
            return x
        buf = np.concatenate([self._carry, x]) if len(self._carry) else x
        if len(buf) < self.TAPS:
            self._carry = buf
            return np.zeros(0, np.float32)
        y = np.convolve(buf, self._h, mode="valid")
        out = y[self._phase :: 2]
        self._phase = (self._phase - len(y)) % 2
        self._carry = buf[-(self.TAPS - 1) :]
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        if self.ratio == 1 or not len(self._carry):
            return np.zeros(0, np.float32)
        out = self.process(np.zeros(self.TAPS - 1, np.float32))
        self._carry = np.zeros(0, np.float32)
        return out


# ----------------------------------------------------------------------------- adapter
class DotsTts:
    name = NAME
    voice_format = VOICE_FORMAT

    def __init__(
        self, backend: Backend, *, output_rate: int = TTS_SAMPLE_RATE, revision: str = HF_REVISION
    ) -> None:
        self.backend = backend
        self.sample_rate = int(output_rate)
        self.revision = revision
        self._lock = threading.Lock()  # one GPU step at a time; held per patch
        self._last: dict[str, Any] = {}
        Decimator(backend.native_rate, self.sample_rate)  # validate the pair early

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last)

    def parse_voice(self, data: bytes) -> DotsVoice:
        return parse_voice(data)

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        """From a clip alone (``POST /v1/voices/from-reference`` carries no
        transcript): x-vector-only cloning. Clients that have the transcript
        build the file themselves with :func:`make_voice`."""
        ref = np.asarray(reference, np.float32).reshape(-1)
        if len(ref) / rate > MAX_REF_SECONDS:
            ref = ref[: int(MAX_REF_SECONDS * rate)]
        return make_voice(ref, int(rate), "")

    def _steps(
        self,
        text: str,
        voice_id: str,
        voice: DotsVoice,
        seed: int | None,
        feed: Iterator[str] | None,
    ) -> Iterator[np.ndarray]:
        """Native-rate float32 chunks, one GPU step per ``next``."""
        if feed is None:
            yield from self.backend.synth(voice, voice_id, text, seed=seed)
            return
        if not self.backend.text_stream:
            # No session on this backend: buffer the feed and speak it as one unit.
            whole = text + "".join(feed)
            yield from self.backend.synth(voice, voice_id, whole, seed=seed)
            return
        session = self.backend.open_session(voice, voice_id, seed=seed)
        try:
            for piece in itertools.chain([text], feed):  # lazily: the feed arrives over time
                if piece:
                    yield from session.push(piece)
            yield from session.finish()
        finally:
            session.close()

    def stream(
        self,
        text: str,
        voice_id: str,
        voice: DotsVoice,
        *,
        stop: threading.Event | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[np.ndarray]:
        opts = options or {}
        gate: Callable[[int], bool] | None = opts.get("gate")
        on_done: Callable[[int], None] | None = opts.get("on_lm_done")
        feed = opts.get("text_feed")
        feed_it = iter(feed) if feed is not None else None
        t0 = time.perf_counter()
        stats: dict[str, Any] = {
            "ttfa_ms": None,
            "audio_ms": 0,
            "chunks": 0,
            "rtf": None,
            "tokens": 0,
            "stopped": False,
        }
        dec = Decimator(self.backend.native_rate, self.sample_rate)
        emitted = 0
        native_samples = 0
        steps = self._steps(text, voice_id, voice, seed, feed_it)
        finished = False

        def emit(chunk: np.ndarray, *, decimate: bool = True) -> np.ndarray | None:
            nonlocal emitted
            out = dec.process(chunk) if decimate else chunk
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
                tail = emit(dec.flush(), decimate=False)
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
        """Block (5 ms polls) while the credit gate is closed; False on stop."""
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
        from speechforge.audio import synthetic_reference

        voice = parse_voice(make_voice(synthetic_reference(3.0), TTS_SAMPLE_RATE, ""))
        list(self.stream("Warming up the voice.", "_warmup", voice, seed=0))

    def describe(self) -> dict[str, Any]:
        out = {
            "name": "dots.tts-mf",
            "engine": NAME,
            "revision": self.revision,
            "native_sample_rate": self.backend.native_rate,
            "sample_rate": self.sample_rate,
            "voice_format": VOICE_FORMAT,
            "text_stream": bool(self.backend.text_stream),
        }
        out.update(self.backend.describe())
        return out

    def close(self) -> None:
        with self._lock:
            self.backend.close()


def _samples_per_token(rate: int) -> int:
    """Samples per 40 ms credit token at ``rate`` (1920 at 48 kHz)."""
    return max(1, rate * TOKEN_MS // 1000)


# ----------------------------------------------------------------------------- loading
def _check_allowed(config: Any) -> None:
    if not config.voices.allow_reference_clip:
        raise RuntimeError(
            "tts.engine dots-tts clones from the reference clip itself, so the clip (and its "
            "transcript) must be on the rig; voices.allow_reference_clip is false (decision D3: "
            "reference clips stay on the box). Set it to true only if you, the owner, accept "
            "that -- see docs/ENGINES.md."
        )


def _normalizer_shim(allow: bool) -> bool:
    """dots_tts imports WeTextProcessing (``tn``) at import time; pynini has no
    Windows wheels. Install a stub whose normalisers refuse to be used, so the
    runtime imports; ``normalize_text`` must stay off. Returns True if shimmed."""
    import importlib.util
    import sys
    import types

    if importlib.util.find_spec("tn") is not None:
        return False
    if not allow:
        raise RuntimeError(
            "WeTextProcessing (package 'tn') is not installed and "
            "tts.dots.allow_missing_text_normalizer is false"
        )

    class _Unavailable:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        def normalize(self, _text: str) -> str:
            raise RuntimeError("text normalisation is unavailable: WeTextProcessing not installed")

    pkg = types.ModuleType("tn")
    pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["tn"] = pkg
    for lang in ("chinese", "english"):
        sub = types.ModuleType(f"tn.{lang}")
        sub.__path__ = []  # type: ignore[attr-defined]
        mod = types.ModuleType(f"tn.{lang}.normalizer")
        mod.Normalizer = _Unavailable  # type: ignore[attr-defined]
        sys.modules[f"tn.{lang}"] = sub
        sys.modules[f"tn.{lang}.normalizer"] = mod
    return True


class _NativeBackend:
    """The ``dots_tts`` runtime in-process. Private hooks used, read at dots.tts
    0.3.1: ``DotsTtsRuntime._load_prompt_audio`` (replaced so the clip stays in
    memory) and ``runtime.model.tokenizer`` (double streaming)."""

    native_rate = NATIVE_RATE

    def __init__(self, config: Any, device_id: int) -> None:
        cfg = config.tts.dots
        self.cfg = cfg
        self.device_id = int(device_id)
        self.text_stream = bool(cfg.double_streaming)
        self.shimmed = _normalizer_shim(cfg.allow_missing_text_normalizer)
        if cfg.normalize_text and self.shimmed:
            raise RuntimeError("tts.dots.normalize_text needs WeTextProcessing installed")
        # The weights are local and lock-verified; the hub is never consulted
        # while serving (SPEC 8.5). setdefault: an operator who set it wins.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch

        if config.gpu.provider == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("torch has no CUDA: install the torch extra for the rig")
        self.torch = torch
        model_dir = config.models_dir / (STTS_MODEL_DIR if cfg.double_streaming else MODEL_DIR)
        if not model_dir.is_dir():
            raise RuntimeError(f"missing {model_dir.name} (run `speechforge fetch-models`)")
        check_latent_stats(model_dir / "latent_stats.pt")
        if cfg.double_streaming:
            from dots_tts.runtime_double_streaming import DotsTtsRuntimeDoubleStreaming as Runtime
        else:
            from dots_tts.runtime import DotsTtsRuntime as Runtime
        precision = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[cfg.precision]
        with self._device():
            self.runtime = Runtime.from_pretrained(
                str(model_dir),
                precision=precision,
                optimize=cfg.optimize,
                max_generate_length=cfg.max_generate_length,
                warmup_on_optimize=cfg.warmup_on_optimize,
            )
        self.native_rate = int(self.runtime.sample_rate)
        self._prompts: dict[str, Any] = {}
        self._prompt_order: list[str] = []
        self._orig_load = self.runtime._load_prompt_audio
        self.runtime._load_prompt_audio = self._prompt_audio  # type: ignore[method-assign]
        self.model_dir = model_dir

    def _device(self) -> Any:
        if self.torch.cuda.is_available():
            return self.torch.cuda.device(self.device_id)
        import contextlib

        return contextlib.nullcontext()

    # -- the clip stays in memory: the runtime asks us for "sf-voice://<id>"
    def _register(self, voice: DotsVoice, voice_id: str) -> str:
        key = f"sf-voice://{voice_id}"
        if key not in self._prompts:
            from dots_tts.utils.audio import high_quality_resample

            wav = self.torch.from_numpy(np.ascontiguousarray(voice.audio, np.float32)).unsqueeze(0)
            if voice.rate != self.native_rate:
                wav = high_quality_resample(wav, orig_sr=voice.rate, target_sr=self.native_rate)
            self._prompts[key] = wav
            self._prompt_order.append(key)
            while len(self._prompt_order) > 16:
                self._prompts.pop(self._prompt_order.pop(0), None)
        return key

    def _prompt_audio(self, path: str) -> Any:
        wav = self._prompts.get(path)
        if wav is None:
            raise RuntimeError("dots-tts: prompt audio is only served from memory")
        return wav.to(self.runtime.device)

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self.torch.manual_seed(int(seed))

    def _sampling(self) -> dict[str, Any]:
        if self.cfg.double_streaming:  # fixed-step artifacts lock these; pass the defaults
            return {}
        return {"num_steps": self.cfg.num_steps, "guidance_scale": self.cfg.guidance_scale}

    def synth(
        self, voice: DotsVoice, voice_id: str, text: str, *, seed: int | None
    ) -> Iterator[np.ndarray]:
        if self.cfg.double_streaming:
            session = self.open_session(voice, voice_id, seed=seed)
            try:
                yield from session.push(text)
                yield from session.finish()
            finally:
                session.close()
            return
        key = self._register(voice, voice_id)
        with self._device(), self.torch.inference_mode():
            self._seed(seed)
            gen = self.runtime.generate_stream(
                text=text,
                prompt_audio_path=key,
                prompt_text=voice.transcript or None,
                speaker_scale=self.cfg.speaker_scale,
                language=self.cfg.language,
                normalize_text=self.cfg.normalize_text,
                **self._sampling(),
            )
        try:
            while True:
                with self._device(), self.torch.inference_mode():
                    try:
                        chunk = next(gen)
                    except StopIteration:
                        return
                yield chunk.detach().float().cpu().numpy().reshape(-1)
        finally:
            gen.close()

    def open_session(self, voice: DotsVoice, voice_id: str, *, seed: int | None) -> TextSession:
        if not self.cfg.double_streaming:
            raise RuntimeError("double streaming needs tts.dots.double_streaming: true")
        key = self._register(voice, voice_id)
        with self._device(), self.torch.inference_mode():
            self._seed(seed)
            sess = self.runtime.start_double_streaming(
                prompt_audio_path=key,
                prompt_text=voice.transcript or None,
                speaker_scale=self.cfg.speaker_scale,
                eos_threshold=self.cfg.eos_threshold,
            )
        return _NativeSession(self, sess)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "native",
            "device": f"cuda:{self.device_id}" if self.torch.cuda.is_available() else "cpu",
            "precision": self.cfg.precision,
            "num_steps": None if self.cfg.double_streaming else self.cfg.num_steps,
            "optimize": self.cfg.optimize,
            "double_streaming": self.cfg.double_streaming,
            "text_normalizer": "shim" if self.shimmed else "WeTextProcessing",
            "model_dir": self.model_dir.name,
        }

    def close(self) -> None:
        self._prompts.clear()
        self.runtime = None
        if self.torch.cuda.is_available():
            with self._device():
                self.torch.cuda.empty_cache()


class _NativeSession:
    def __init__(self, backend: _NativeBackend, sess: Any) -> None:
        self.b = backend
        self.sess = sess
        self.tok = backend.runtime.model.tokenizer

    def push(self, text: str) -> Iterator[np.ndarray]:
        ids = self.tok.encode(text, add_special_tokens=False)
        for tid in ids:
            with self.b._device(), self.b.torch.inference_mode():
                chunk = self.sess.push_text_token(int(tid))
            if chunk is not None:
                yield chunk.detach().float().cpu().numpy().reshape(-1)

    def finish(self) -> Iterator[np.ndarray]:
        gen = self.sess.finish_text()
        try:
            while True:
                with self.b._device(), self.b.torch.inference_mode():
                    try:
                        chunk = next(gen)
                    except StopIteration:
                        return
                yield chunk.detach().float().cpu().numpy().reshape(-1)
        finally:
            gen.close()

    def close(self) -> None:
        self.sess = None


class _HttpBackend:
    """SGLang Omni's ``POST /v1/audio/speech`` (loopback). The clip travels as a
    ``data:`` URL in ``references[0].audio_path`` with its transcript; the reply
    is raw s16le 48 kHz mono, yielded per HTTP chunk."""

    native_rate = NATIVE_RATE
    text_stream = False  # Omni starts a request from complete text (its docs)

    def __init__(self, config: Any) -> None:
        import httpx

        cfg = config.tts.dots
        self.cfg = cfg
        self.url = cfg.url
        self.model = cfg.http_model or HF_REPO
        # The body carries the reference clip (D3): the request goes to the
        # loopback URL the config validated and nowhere else -- no HTTP(S)_PROXY
        # from the environment, no redirect off the box.
        self.client = httpx.Client(
            timeout=httpx.Timeout(cfg.http_timeout_s, read=cfg.http_timeout_s),
            trust_env=False,
            follow_redirects=False,
        )

    def _body(self, voice: DotsVoice, text: str, seed: int | None) -> dict[str, Any]:
        clip = "data:audio/wav;base64," + base64.b64encode(voice.wav()).decode("ascii")
        body: dict[str, Any] = {
            "model": self.model,
            "input": text,
            "references": [{"audio_path": clip, "text": voice.transcript}],
            "stream": True,
            "response_format": "pcm",
            "stage_params": {
                "latent_engine": {
                    "speaker_scale": self.cfg.speaker_scale,
                    "eos_threshold": self.cfg.eos_threshold,
                }
            },
        }
        if seed is not None:
            body["seed"] = int(seed)
        if self.cfg.language:
            body["language"] = self.cfg.language
        return body

    def synth(
        self, voice: DotsVoice, voice_id: str, text: str, *, seed: int | None
    ) -> Iterator[np.ndarray]:
        carry = b""
        with self.client.stream(
            "POST", f"{self.url}/v1/audio/speech", json=self._body(voice, text, seed)
        ) as resp:
            if resp.status_code != 200:
                detail = resp.read()[:300].decode("utf-8", "replace")
                raise RuntimeError(f"dots-tts server answered {resp.status_code}: {detail}")
            for raw in resp.iter_bytes():
                buf = carry + raw
                n = len(buf) // 2 * 2
                carry = buf[n:]
                if n:
                    yield np.frombuffer(buf[:n], "<i2").astype(np.float32) / 32768.0

    def open_session(self, voice: DotsVoice, voice_id: str, *, seed: int | None) -> TextSession:
        raise RuntimeError("the http backend has no incremental-text session")

    def describe(self) -> dict[str, Any]:
        return {"backend": "http", "url": self.url, "model": self.model}

    def close(self) -> None:
        self.client.close()


def _make_backend(config: Any, device_id: int) -> Backend:
    if config.tts.dots.backend == "http":
        return _HttpBackend(config)
    return _NativeBackend(config, device_id)


def load(config: Any, device_id: int) -> DotsTts:
    """Registry loader (``engines/base.py``). Refuses without the owner's D3
    opt-in; imports torch / dots_tts / httpx only here."""
    _check_allowed(config)
    cfg = config.tts.dots
    if cfg.double_streaming and cfg.backend == "http":
        raise RuntimeError("tts.dots.double_streaming needs the native backend")
    backend = _make_backend(config, device_id)
    revision = HF_STTS_REVISION if cfg.double_streaming else HF_REVISION
    return DotsTts(backend, output_rate=cfg.output_rate, revision=revision)
