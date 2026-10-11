""":class:`~speechforge.engines.base.TtsAdapter` for Qwen3-TTS (Alibaba Qwen,
Apache-2.0) through faster-qwen3-tts (MIT): PyTorch CUDA graphs, streaming.

Phase-2 bake-off candidate (SPEC §7.1, §12; docs/ENGINES.md has the research
and the install). What differs from the Chatterbox adapter:

* **The voice is the reference clip.** Qwen3-TTS clones *in context*: the
  language model reads the reference clip's speech codes and its transcript
  before the text to speak. There are no portable encoder tensors, so the
  voice format ``qwen3tts-ref`` is a small npz carrying the clip (16-bit mono,
  1-20 s) and its UTF-8 transcript. That puts a recording of a person on the
  rig, which is exactly what decision D3 keeps off it by default: :func:`load`
  refuses unless ``voices.allow_reference_clip`` is true. Choosing this engine
  *is* the owner's D3 decision.
* **One utterance at a time.** faster-qwen3-tts keeps the KV cache in static
  buffers captured into CUDA graphs; two sentences cannot interleave per step
  as Chatterbox's do (S12). ``stream`` holds one lock for the whole utterance,
  including a credit stall, so concurrent contexts queue per sentence.
* **Codec steps, not 40 ms tokens.** The 12 Hz codec emits one step per 83 ms.
  The server's credit gate and ``on_lm_done`` count 40 ms units, so this
  adapter reports audio *emitted so far* in those units (960 samples each).
* **Heavy imports stay inside the loader.** torch and faster_qwen3_tts are
  imported in :class:`FasterQwen3Backend` only; the module, the voice parser
  and the tests need numpy alone. The backend is an injectable protocol, which
  is how the tests cover the adapter without a GPU.
"""

from __future__ import annotations

import contextlib
import gc
import io
import os
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from speechforge.audio import TTS_SAMPLE_RATE, float_to_pcm16
from speechforge.errors import BadRequest, ConfigError

NAME = "qwen3-tts"
VOICE_FORMAT = "qwen3tts-ref"
VOICE_FORMAT_VERSION = 1
#: Hugging Face repo per ``tts.qwen3.model``; revisions pinned in config.py and models.lock.
MODEL_IDS = {"1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
#: Codec steps per second of audio (the "12Hz" in the model name).
CODEC_HZ = 12
#: The server counts TTS progress in 40 ms "tokens" (tts_ctx.TOKEN_MS); the
#: credit gate and ``on_lm_done`` get audio emitted so far in that unit.
UNIT_SAMPLES = TTS_SAMPLE_RATE * 40 // 1000
MIN_REF_S, MAX_REF_S = 1.0, 20.0
MIN_REF_RATE, MAX_REF_RATE = 8000, 48000
MAX_TRANSCRIPT_CHARS = 1000
MAX_VOICE_BYTES = 4 * 1024 * 1024
MAX_CHUNK_STEPS = 48
TORCH_DTYPES = {"bf16": "bfloat16", "fp16": "float16"}
#: Weight files the loader refuses to sit next to (SPEC §8.5: safetensors only).
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")


# --------------------------------------------------------------------------- voices
@dataclass(frozen=True)
class RefVoice:
    """A parsed ``qwen3tts-ref`` voice: the clip as float32 mono at ``rate``
    plus its transcript. An empty transcript means speaker-embedding-only
    cloning (``x_vector_only_mode``), which needs no text but matches less."""

    audio: np.ndarray
    rate: int
    transcript: str

    @property
    def seconds(self) -> float:
        return len(self.audio) / self.rate

    @property
    def xvec_only(self) -> bool:
        return not self.transcript.strip()

    @property
    def nbytes(self) -> int:
        return int(self.audio.nbytes)


def build_voice_bytes(audio: np.ndarray, rate: int, transcript: str = "") -> bytes:
    """``qwen3tts-ref`` bytes from a float32 (or int16) mono clip at ``rate``.
    Stored as 16-bit PCM so a 20 s clip stays under the 2 MB voice upload cap."""
    arr = np.asarray(audio)
    if arr.ndim != 1:
        raise BadRequest("reference clip must be mono (one dimension)", code="bad_voice")
    pcm = arr.astype("<i2") if arr.dtype.kind in "iu" else float_to_pcm16(arr)
    buf = io.BytesIO()
    np.savez(
        buf,
        format=np.array(VOICE_FORMAT),
        version=np.array(VOICE_FORMAT_VERSION, dtype=np.int64),
        audio=pcm,
        rate=np.array(int(rate), dtype=np.int64),
        transcript=np.array(str(transcript)),
    )
    return buf.getvalue()


def _text_of(arr: np.ndarray, what: str) -> str:
    if arr.dtype.kind not in "US" or arr.size != 1:
        raise BadRequest(
            f"{what}: expected one text value, got {arr.dtype} {arr.shape}", code="bad_voice"
        )
    value = arr.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_voice(data: bytes) -> RefVoice:
    """Validate ``qwen3tts-ref`` bytes. Raises 400 ``bad_voice``.

    Checks: size, zip inflation, the three keys, mono int16/float32 audio of
    1-20 s at 8-48 kHz, finite samples, transcript <= 1000 chars. The clip is
    kept at its own rate; the model resamples once when the prompt is built."""
    if len(data) > MAX_VOICE_BYTES:
        raise BadRequest(f"voice file is larger than {MAX_VOICE_BYTES} bytes", code="bad_voice")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
        if total > 4 * MAX_VOICE_BYTES:
            raise BadRequest("voice file inflates too large", code="bad_voice")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            names = set(z.files)
            for key in ("audio", "rate", "transcript"):
                if key not in names:
                    raise BadRequest(f"voice file lacks {key!r} ({VOICE_FORMAT})", code="bad_voice")
            if "format" in names and _text_of(np.asarray(z["format"]), "format") != VOICE_FORMAT:
                raise BadRequest(f"voice file is not {VOICE_FORMAT}", code="bad_voice")
            audio = np.asarray(z["audio"])
            rate_arr = np.asarray(z["rate"])
            transcript = _text_of(np.asarray(z["transcript"]), "transcript")
    except BadRequest:
        raise
    except Exception as exc:  # zipfile.BadZipFile, ValueError (pickled object), OSError
        raise BadRequest(
            f"not a valid voice file ({type(exc).__name__})", code="bad_voice"
        ) from exc
    if rate_arr.dtype.kind not in "iu" or rate_arr.size != 1:
        raise BadRequest(f"rate: expected an integer, got {rate_arr.dtype}", code="bad_voice")
    rate = int(rate_arr.reshape(-1)[0])
    if not MIN_REF_RATE <= rate <= MAX_REF_RATE:
        raise BadRequest(
            f"rate: {rate} Hz is outside {MIN_REF_RATE}-{MAX_REF_RATE}", code="bad_voice"
        )
    if audio.ndim != 1:
        raise BadRequest(f"audio: expected mono, got shape {audio.shape}", code="bad_voice")
    if audio.dtype == np.int16:
        samples = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.float32:
        if not np.isfinite(audio).all():
            raise BadRequest("audio: non-finite samples", code="bad_voice")
        if np.abs(audio).max(initial=0.0) > 1.5:
            raise BadRequest("audio: float samples must be in [-1, 1]", code="bad_voice")
        samples = np.clip(audio, -1.0, 1.0)
    else:
        raise BadRequest(f"audio: dtype {audio.dtype} is not int16 or float32", code="bad_voice")
    seconds = len(samples) / rate
    if not MIN_REF_S <= seconds <= MAX_REF_S:
        raise BadRequest(
            f"reference clip is {seconds:.1f} s; use {MIN_REF_S:g}-{MAX_REF_S:g} s",
            code="bad_voice",
        )
    transcript = transcript.strip()
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise BadRequest(
            f"transcript is longer than {MAX_TRANSCRIPT_CHARS} characters", code="bad_voice"
        )
    if any(ord(c) < 32 and c not in "\n\t" for c in transcript):
        raise BadRequest("transcript contains control characters", code="bad_voice")
    return RefVoice(np.ascontiguousarray(samples, np.float32), rate, transcript)


# --------------------------------------------------------------------------- backend
class Qwen3Backend(Protocol):
    """What the adapter needs from the model library. :class:`FasterQwen3Backend`
    is the real one; the tests inject a fake."""

    sample_rate: int

    def clone_prompt(
        self, audio: np.ndarray, rate: int, transcript: str, *, xvec_only: bool
    ) -> Any: ...

    def stream(
        self,
        text: str,
        *,
        language: str,
        prompt: Any,
        transcript: str,
        chunk_size: int,
        seed: int | None,
        sampling: dict[str, Any],
    ) -> Iterator[tuple[np.ndarray, int, dict[str, Any]]]: ...  # (float32 piece, rate, timing)

    def render(
        self,
        text: str,
        *,
        language: str,
        prompt: Any,
        transcript: str,
        seed: int | None,
        sampling: dict[str, Any],
    ) -> np.ndarray: ...  # float32 at sample_rate

    def warmup(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class Qwen3Settings:
    """The ``tts.qwen3`` knobs the adapter reads per utterance."""

    language: str = "English"
    chunk_size: int = 8
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    xvec_only: bool = False
    max_prepared: int = 8

    @classmethod
    def from_config(cls, q: Any) -> Qwen3Settings:
        return cls(
            language=q.language,
            chunk_size=q.chunk_size,
            max_new_tokens=q.max_new_tokens,
            temperature=q.temperature,
            top_k=q.top_k,
            top_p=q.top_p,
            repetition_penalty=q.repetition_penalty,
            xvec_only=q.xvec_only,
        )

    def sampling(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }


def chunk_steps_for(first_chunk_tokens: int | None, default: int) -> int:
    """Map the protocol's ``first_chunk_tokens`` (40 ms Chatterbox tokens) onto
    codec steps (83 ms). The backend has one chunk size per utterance, so the
    request's first-chunk wish sets every chunk of that utterance."""
    if not first_chunk_tokens:
        return default
    steps = round(int(first_chunk_tokens) * 40 * CODEC_HZ / 1000)
    return max(1, min(MAX_CHUNK_STEPS, steps))


def check_safetensors_only(model_dir: Path) -> None:
    """SPEC §8.5: the model directory must hold safetensors weights and nothing
    that torch would unpickle. models.lock already pins exactly these files;
    this is the belt to its braces (a stray ``pytorch_model.bin`` would be
    picked up by transformers if the safetensors file were missing)."""
    model_dir = Path(model_dir)
    for rel in ("model.safetensors", "speech_tokenizer/model.safetensors"):
        if not (model_dir / rel).is_file():
            raise FileNotFoundError(
                f"Qwen3-TTS file missing: {model_dir / rel} (run `speechforge fetch-models`)"
            )
    stray = [p for p in model_dir.rglob("*") if p.suffix.lower() in PICKLE_SUFFIXES]
    if stray:
        names = ", ".join(sorted(p.relative_to(model_dir).as_posix() for p in stray)[:4])
        raise RuntimeError(f"refusing to load Qwen3-TTS: pickle-format files present ({names})")


class FasterQwen3Backend:
    """faster-qwen3-tts on one CUDA device. Every heavy import happens here."""

    def __init__(
        self,
        model_dir: Path,
        *,
        device_id: int,
        dtype: str = "bf16",
        attn: str = "sdpa",
        max_seq_len: int = 2048,
        append_silence: bool = True,
    ) -> None:
        # The model files are local and verified; the hub must never be consulted
        # while serving (§8.5). setdefault: an operator who set it wins.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        check_safetensors_only(model_dir)
        import faster_qwen3_tts
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch has no CUDA device (install the `tts-torch` extra's cu130 wheels)"
            )
        self._torch = torch
        self.device_id = int(device_id)
        self.device = f"cuda:{self.device_id}"
        # faster-qwen3-tts synchronises and allocates on the *current* device in
        # places; make the leased card current before anything is built.
        torch.cuda.set_device(self.device_id)
        self.dtype = dtype
        self.attn = attn
        self.append_silence = append_silence
        self.version = getattr(faster_qwen3_tts, "__version__", "?")
        self.model = FasterQwen3TTS.from_pretrained(
            str(model_dir),
            device=self.device,
            dtype=TORCH_DTYPES[dtype],
            attn_implementation=attn,
            max_seq_len=int(max_seq_len),
        )
        self.sample_rate = int(self.model.sample_rate)

    def clone_prompt(
        self, audio: np.ndarray, rate: int, transcript: str, *, xvec_only: bool
    ) -> Any:
        wav = np.asarray(audio, np.float32)
        if self.append_silence and not xvec_only:
            # faster-qwen3-tts does this for file inputs: the prompt then ends on
            # silence, so the reference's last phoneme does not bleed into the
            # first generated one. We pass arrays, so we do it here.
            wav = np.concatenate([wav, np.zeros(int(0.5 * rate), np.float32)])
        with self._torch.inference_mode():
            return self.model.model.create_voice_clone_prompt(
                ref_audio=(wav, int(rate)),
                ref_text="" if xvec_only else transcript,
                x_vector_only_mode=xvec_only,
            )

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self._torch.manual_seed(int(seed))
            self._torch.cuda.manual_seed_all(int(seed))

    def stream(
        self,
        text: str,
        *,
        language: str,
        prompt: Any,
        transcript: str,
        chunk_size: int,
        seed: int | None,
        sampling: dict[str, Any],
    ) -> Iterator[tuple[np.ndarray, int, dict[str, Any]]]:
        self._seed(seed)
        return self.model.generate_voice_clone_streaming(
            text,
            language,
            ref_text=transcript,
            voice_clone_prompt=prompt,
            chunk_size=int(chunk_size),
            **sampling,
        )

    def render(
        self,
        text: str,
        *,
        language: str,
        prompt: Any,
        transcript: str,
        seed: int | None,
        sampling: dict[str, Any],
    ) -> np.ndarray:
        self._seed(seed)
        wavs, _sr = self.model.generate_voice_clone(
            text, language, ref_text=transcript, voice_clone_prompt=prompt, **sampling
        )
        return np.asarray(wavs[0], np.float32).reshape(-1)

    def warmup(self) -> None:
        self.model.warmup()

    def describe(self) -> dict[str, Any]:
        torch = self._torch
        out: dict[str, Any] = {
            "library": "faster-qwen3-tts",
            "version": self.version,
            "cuda_graphs": True,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "attn": self.attn,
        }
        try:
            out["gpu"] = torch.cuda.get_device_name(self.device_id)
            major, minor = torch.cuda.get_device_capability(self.device_id)
            out["sm"] = f"sm_{major}{minor}"
        except Exception:  # noqa: BLE001 - descriptive only
            pass
        return out

    def close(self) -> None:
        self.model = None
        gc.collect()
        with contextlib.suppress(Exception):  # closing never fails an unload
            self._torch.cuda.empty_cache()


# --------------------------------------------------------------------------- adapter
def _wait_gate(gate: Any, units: int, stop: threading.Event | None) -> bool:
    """Block while the credit gate is closed (5 ms polls, like the pipeline);
    False when ``stop`` was set meanwhile."""
    if gate is not None:
        while not gate(units):
            if stop is not None and stop.is_set():
                return False
            time.sleep(0.005)
    return not (stop is not None and stop.is_set())


class Qwen3Tts:
    name = NAME
    sample_rate = TTS_SAMPLE_RATE
    voice_format = VOICE_FORMAT

    def __init__(
        self,
        backend: Qwen3Backend,
        *,
        settings: Qwen3Settings | None = None,
        model_id: str = MODEL_IDS["1.7b"],
        revision: str = "",
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> None:
        if int(backend.sample_rate) != TTS_SAMPLE_RATE:
            raise RuntimeError(
                f"Qwen3-TTS backend renders at {backend.sample_rate} Hz; the server streams "
                f"{TTS_SAMPLE_RATE} Hz frames"
            )
        self.backend: Any = backend
        self.settings = settings or Qwen3Settings()
        self.model_id, self.revision, self.device, self.dtype = model_id, revision, device, dtype
        self._prepared: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        #: One utterance at a time: the backend's CUDA-graph buffers are shared.
        self._gen_lock = threading.Lock()
        self._last: dict[str, Any] = {}

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last)

    # ------------------------------------------------------------------ voices
    def parse_voice(self, data: bytes) -> RefVoice:
        return parse_voice(data)

    def build_voice(self, reference: np.ndarray, rate: int, transcript: str = "") -> bytes:
        """A voice from a clip. Without a transcript the voice is speaker-embedding
        only (what the self-test and ``/v1/voices/from-reference`` get); a clip
        longer than the cap is then cut to it. With a transcript the clip must
        fit as it is, since the text has to match what is spoken."""
        clip = np.asarray(reference, np.float32).reshape(-1)
        limit = int(MAX_REF_S * rate)
        if len(clip) > limit:
            if transcript.strip():
                raise BadRequest(
                    f"reference clip is {len(clip) / rate:.1f} s; use at most {MAX_REF_S:g} s",
                    code="bad_voice",
                )
            clip = clip[:limit]
        return build_voice_bytes(clip, rate, transcript)

    def _prompt(self, vid: str, voice: RefVoice) -> Any:
        with self._lock:
            got = self._prepared.get(vid)
            if got is not None:
                self._prepared.move_to_end(vid)
                return got
        xvec = self.settings.xvec_only or voice.xvec_only
        prompt = self.backend.clone_prompt(
            voice.audio, voice.rate, voice.transcript, xvec_only=xvec
        )
        with self._lock:
            self._prepared[vid] = prompt
            while len(self._prepared) > self.settings.max_prepared:
                self._prepared.popitem(last=False)
        return prompt

    def _transcript_for(self, voice: RefVoice) -> str:
        return "" if self.settings.xvec_only else voice.transcript

    # ------------------------------------------------------------------ speech
    def stream(
        self,
        text: str,
        voice_id: str,
        voice: RefVoice,
        *,
        stop: threading.Event | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[np.ndarray]:
        opts = options or {}
        gate = opts.get("gate")
        on_done = opts.get("on_lm_done")
        chunk = chunk_steps_for(opts.get("first_chunk_tokens"), self.settings.chunk_size)
        stats: dict[str, Any] = {
            "ttfa_ms": None,
            "first_chunk_ms": None,
            "prefill_ms": None,
            "chunks": 0,
            "tokens": 0,
            "audio_ms": 0,
            "chunk_size": chunk,
        }
        t0 = time.perf_counter()
        emitted = 0
        finished = False
        text = (text or "").strip()
        try:
            if not text:
                finished = True
                return
            prompt = self._prompt(voice_id, voice)
            with self._gen_lock:
                if stop is not None and stop.is_set():
                    return
                gen = self.backend.stream(
                    text,
                    language=self.settings.language,
                    prompt=prompt,
                    transcript=self._transcript_for(voice),
                    chunk_size=chunk,
                    seed=seed,
                    sampling=self.settings.sampling(),
                )
                try:
                    while True:
                        # Credit / pause: wait before asking for the next chunk, so
                        # no GPU work runs past the credit (the gate counts 40 ms units).
                        if not _wait_gate(gate, emitted // UNIT_SAMPLES, stop):
                            return
                        try:
                            piece, rate, timing = next(gen)
                        except StopIteration:
                            finished = True
                            break
                        if timing:
                            stats["tokens"] = int(timing.get("total_steps_so_far", stats["tokens"]))
                            if stats["prefill_ms"] is None and timing.get("prefill_ms"):
                                stats["prefill_ms"] = round(float(timing["prefill_ms"]))
                        if stop is not None and stop.is_set():
                            return
                        if int(rate) != TTS_SAMPLE_RATE:
                            raise RuntimeError(f"backend streamed {rate} Hz audio")
                        pcm = float_to_pcm16(np.asarray(piece, np.float32).reshape(-1))
                        if len(pcm) == 0:
                            continue
                        if stats["first_chunk_ms"] is None:
                            ms = round((time.perf_counter() - t0) * 1000)
                            stats["first_chunk_ms"] = stats["ttfa_ms"] = ms
                        stats["chunks"] += 1
                        emitted += len(pcm)
                        yield pcm
                finally:
                    gen.close()
            if finished and on_done is not None and not (stop is not None and stop.is_set()):
                on_done(emitted // UNIT_SAMPLES)
        finally:
            stats["audio_ms"] = round(emitted / TTS_SAMPLE_RATE * 1000)
            stats["total_ms"] = round((time.perf_counter() - t0) * 1000)
            stats["rtf"] = (
                round(stats["total_ms"] / stats["audio_ms"], 3) if stats["audio_ms"] else None
            )
            stats["finished"] = finished
            self._last = stats

    def render(self, text: str, voice: RefVoice, *, seed: int | None = None) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, "<i2")
        # Not cached by id: render() has no voice_id; the prompt costs one
        # encoder pass, which a whole-utterance render dwarfs.
        xvec = self.settings.xvec_only or voice.xvec_only
        prompt = self.backend.clone_prompt(
            voice.audio, voice.rate, voice.transcript, xvec_only=xvec
        )
        with self._gen_lock:
            wav = self.backend.render(
                text,
                language=self.settings.language,
                prompt=prompt,
                transcript=self._transcript_for(voice),
                seed=seed,
                sampling=self.settings.sampling(),
            )
        return float_to_pcm16(np.asarray(wav, np.float32).reshape(-1))

    def warm_up(self) -> None:
        """Capture the CUDA graphs and speak one short line with a synthetic
        (non-human) reference, so the first real sentence pays no first-time cost."""
        from speechforge.audio import synthetic_reference

        self.backend.warmup()
        voice = parse_voice(self.build_voice(synthetic_reference(), TTS_SAMPLE_RATE))
        list(self.stream("Warming up the voice.", "_warmup", voice, seed=0))
        with self._lock:
            self._prepared.pop("_warmup", None)

    def describe(self) -> dict[str, Any]:
        s = self.settings
        return {
            "name": NAME,
            "model": self.model_id,
            "revision": self.revision,
            "dtype": self.dtype,
            "device": self.device,
            "voice_format": VOICE_FORMAT,
            "language": s.language,
            "chunk_size": s.chunk_size,
            "xvec_only": s.xvec_only,
            "backend": self.backend.describe() if self.backend is not None else None,
        }

    def close(self) -> None:
        with self._lock:
            self._prepared.clear()
        backend, self.backend = self.backend, None
        if backend is not None:
            backend.close()


def load(config: Any, device_id: int) -> Qwen3Tts:
    """The registry loader. Refuses without the owner's D3 decision; imports
    torch only past that point."""
    from speechforge.config import QWEN3_TTS_REVISIONS

    if not config.voices.allow_reference_clip:
        raise ConfigError(
            "tts.engine 'qwen3-tts' clones from a reference clip and its transcript kept "
            "on this machine (voice format qwen3tts-ref). That is decision D3: set "
            "voices.allow_reference_clip: true to make it, or keep tts.engine: chatterbox."
        )
    q = config.tts.qwen3
    model_id = MODEL_IDS[q.model]
    model_dir = Path(config.models_dir) / f"qwen3-tts-{q.model}"
    backend = FasterQwen3Backend(
        model_dir,
        device_id=device_id,
        dtype=q.dtype,
        attn=q.attn,
        max_seq_len=q.max_seq_len,
        append_silence=q.append_silence,
    )
    try:
        return Qwen3Tts(
            backend,
            settings=Qwen3Settings.from_config(q),
            model_id=model_id,
            revision=QWEN3_TTS_REVISIONS[q.model],
            device=f"cuda:{int(device_id)}",
            dtype=q.dtype,
        )
    except BaseException:
        backend.close()
        raise
