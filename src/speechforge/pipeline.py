"""Streaming synthesis: first audio from the first chunk, not the first sentence.

A port of the DisPatch voice add-on's ``PipelinedChatterbox`` (same author,
MIT): an LM thread streams speech tokens; as soon as ``first_chunk_tokens``
(+ lookahead) exist a decode is submitted, then every ``chunk_tokens`` more,
each re-decoding ``left_context`` already-spoken tokens and crossfading the
seam over 20 ms (the decoder is stochastic: butted chunks click). On a GPU the
decoder costs tens of milliseconds, so the first chunk can be small (15 tokens
= 0.6 s of audio) without risking an underrun.

:func:`pipelined` is the streaming helper for token-LM + decoder TTS models
(Chatterbox today). :class:`Speaker` is what the HTTP layer talks to: one STT
adapter and one TTS adapter (``engines/base.py``), loaded together onto one
card by :func:`load_speaker`, faked in the tests.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from speechforge.audio import crossfade, float_to_pcm16
from speechforge.voices import VoiceTensors

CROSSFADE_SAMPLES = 24000 * 20 // 1000
TRAILING_SILENCE = 3

# Long-lived worker threads (measured on the rig, DECISIONS S18): ONNX Runtime's
# CUDA provider builds a per-thread context (cuBLAS/cuDNN handles) the first
# time a thread runs a session, 300-450 ms a time. A thread per sentence paid
# that on every first chunk, so TTFA was bimodal (50 ms or 450 ms for the same
# decode). The LM producers and the decoder keep their threads across sentences.
_LM_POOL: ThreadPoolExecutor | None = None
_DEC_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
LM_WORKERS = 8  # >= realtime.max_sessions x max_contexts (3 x 2) plus warm-up/HTTP


def _pools() -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
    global _LM_POOL, _DEC_POOL
    with _POOL_LOCK:
        if _LM_POOL is None:
            _LM_POOL = ThreadPoolExecutor(max_workers=LM_WORKERS, thread_name_prefix="sf-tts-lm")
        if _DEC_POOL is None:
            _DEC_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sf-tts-dec")
        return _LM_POOL, _DEC_POOL


def decode_pool() -> ThreadPoolExecutor:
    """The decoder's long-lived single worker (warm-ups run on it too)."""
    return _pools()[1]


def warm_lm_workers(fn: Callable[[], Any], *, barrier_timeout_s: float = 30.0) -> int:
    """Run ``fn`` once on every LM worker thread (all held at a barrier so each
    lands on a distinct thread): the per-thread CUDA context is then built at
    load, never on a caller's first token. Returns how many threads ran it.

    A worker busy with a sentence cannot reach the barrier; when it times out
    the others still run ``fn`` on whatever thread they hold rather than failing
    the load, so a warm-up under load degrades to fewer warmed threads."""
    pool = _pools()[0]
    barrier = threading.Barrier(LM_WORKERS)

    def task() -> None:
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait(timeout=barrier_timeout_s)
        fn()

    futures = [pool.submit(task) for _ in range(LM_WORKERS)]
    return sum(1 for f in futures if f.result(timeout=120.0) is None)


@dataclass(frozen=True)
class ChunkPlanConfig:
    first: int = 15
    chunk: int = 25
    left: int = 8
    lookahead: int = 3
    prompt_s: float = 4.0
    #: Rate-aware chunking (DECISIONS S21): after the first chunk, the next chunk
    #: is sized so that generating it (at the token rate measured so far, plus one
    #: decode) finishes ``margin_ms`` before the audio already emitted runs out.
    #: A fixed 25-token second chunk behind a 10-token first chunk gapped on
    #: every sentence once two sessions shared the LM (26 ms/token).
    adaptive: bool = True
    margin_ms: int = 100
    decode_ms: int = 60
    min_chunk: int = 6


class Plan:
    """Chunk schedule over a growing token list. A chunk EMITS [emit_start,
    emit_end) and DECODES [emit_start - left, emit_end + look)."""

    def __init__(self, first: int, chunk: int, left: int, lookahead: int) -> None:
        self.first, self.chunk, self.left = max(1, first), max(1, chunk), max(0, left)
        self.lookahead = max(0, lookahead)
        self.decoded_upto = 0
        self.n_chunks = 0

    def next_chunk(self, n_tokens: int, done: bool) -> tuple[int, int, int, int] | None:
        """(start, end incl. lookahead, left_context, lookahead) or None."""
        size = self.first if self.n_chunks == 0 else self.chunk
        avail = n_tokens - self.decoded_upto
        if avail <= 0:
            return None
        if avail >= size + self.lookahead:
            emit_end, look = self.decoded_upto + size, self.lookahead
        elif done:
            emit_end, look = n_tokens, 0
        else:
            return None
        left = min(self.left, self.decoded_upto)
        start = self.decoded_upto - left
        self.decoded_upto = emit_end
        self.n_chunks += 1
        return start, emit_end + look, left, look


class TtsEngine(Protocol):
    def stream_tokens(
        self,
        text: str,
        voice: VoiceTensors,
        stop: threading.Event | None = ...,
        seed: int | None = ...,
        *,
        prefix: Any = ...,
    ) -> Iterator[int]: ...

    def decode_chunk(
        self,
        tokens: list[int],
        voice: VoiceTensors,
        *,
        left_context: int = ...,
        trailing_silence: int = ...,
        keep_silence: bool = ...,
        context_tail: int = ...,
        right_context: int = ...,
    ) -> np.ndarray: ...


def pipelined(
    eng: TtsEngine,
    text: str,
    voice: VoiceTensors,
    *,
    plan: ChunkPlanConfig,
    prefix: Any = None,
    stop: threading.Event | None = None,
    seed: int | None = None,
    stats: dict[str, Any] | None = None,
    gate: Callable[[int], bool] | None = None,
    on_lm_done: Callable[[int], None] | None = None,
) -> Iterator[np.ndarray]:
    """Yield int16 24 kHz pieces as chunks finish (sizes vary; concatenation is
    the sentence). ``stop`` is honoured per token and per chunk.

    ``gate(tokens_so_far)`` is the backpressure (sf.1 ``credit``, ``tts.pause``):
    while it returns False the LM thread waits before its next step, so no GPU
    work runs past the credit. ``on_lm_done(n_tokens)`` fires once the language
    model finished normally -- the sentence's audio length is then known
    (``n_tokens`` x 40 ms), before the decoder has produced it."""
    stats = stats if stats is not None else {}
    halt = _Either(stop)  # also set when the consumer goes away (generator closed)
    stop = halt  # type: ignore[assignment]
    lm_stop = _Gated(halt, gate) if gate is not None else halt
    t0 = time.perf_counter()
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    tokens: list[int] = []
    lm_error: list[BaseException] = []
    stats.update(first_token_ms=None, first_chunk_ms=None, chunks=0, tokens=0, audio_ms=0)

    def lm() -> None:
        n = 0
        try:
            for n, tok in enumerate(
                eng.stream_tokens(text, voice, lm_stop, seed, prefix=prefix), 1
            ):
                if n == 1:
                    stats["first_token_ms"] = round((time.perf_counter() - t0) * 1000)
                if isinstance(lm_stop, _Gated):
                    lm_stop.count = n
                events.put(("tok", (tok, time.perf_counter())))
                if halt.is_set():
                    break
            if on_lm_done is not None and not halt.is_set():
                on_lm_done(n)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the consumer
            lm_error.append(exc)
        finally:
            events.put(("done", None))

    lm_pool, pool = _pools()
    th = lm_pool.submit(lm)
    sched = Plan(plan.first, plan.chunk, plan.left, plan.lookahead)
    pending: list[Future[np.ndarray]] = []
    held: np.ndarray | None = None
    emitted = 0
    done = False
    tok_times: list[float] = []
    t_first_emit: float | None = None
    stats["chunk_tokens"] = []
    try:
        while True:
            if stop is not None and stop.is_set():
                break
            try:
                kind, item = events.get(timeout=0.002 if pending else 0.05)
                while True:
                    if kind == "tok":
                        tokens.append(item[0])
                        tok_times.append(item[1])
                    else:
                        done = True
                    kind, item = events.get_nowait()
            except queue.Empty:
                pass
            if lm_error:
                raise lm_error[0]
            if not pending:
                if plan.adaptive and sched.n_chunks and t_first_emit is not None:
                    sched.chunk = _adaptive_chunk(
                        plan, tok_times, len(tokens) - sched.decoded_upto, emitted, t_first_emit
                    )
                nxt = sched.next_chunk(len(tokens), done)
                if nxt is not None:
                    s, e, lc, look = nxt
                    stats["chunk_tokens"].append(e - look - s - lc)
                    last = done and e >= len(tokens) and look == 0
                    pending.append(
                        pool.submit(
                            eng.decode_chunk,
                            tokens[s:e],
                            voice,
                            left_context=lc,
                            trailing_silence=TRAILING_SILENCE,
                            keep_silence=last,
                            context_tail=CROSSFADE_SAMPLES if lc else 0,
                            right_context=look,
                        )
                    )
            while pending and pending[0].done():
                wav = pending.pop(0).result()
                stats["chunks"] += 1
                if stop is not None and stop.is_set():
                    break
                if len(wav) == 0:
                    continue
                if held is not None:
                    n = min(len(held), len(wav))
                    out = np.concatenate([crossfade(held[-n:], wav[:n]), wav[n:]])
                else:
                    out = wav
                finished = done and sched.decoded_upto >= len(tokens) and not pending
                if not finished:
                    held, out = out[-CROSSFADE_SAMPLES:], out[:-CROSSFADE_SAMPLES]
                else:
                    held = None
                if len(out):
                    if stats["first_chunk_ms"] is None:
                        stats["first_chunk_ms"] = round((time.perf_counter() - t0) * 1000)
                        t_first_emit = time.perf_counter()
                    emitted += len(out)
                    yield float_to_pcm16(out)
            if done and sched.decoded_upto >= len(tokens) and not pending:
                break
        if held is not None and len(held) and not (stop is not None and stop.is_set()):
            emitted += len(held)
            yield float_to_pcm16(held)
    finally:
        halt.halt()
        # Wait for a decode already running: "GPU idle for that gen" (the cancel
        # stop_ms) must include it, not just the LM thread noticing the stop.
        # The pools outlive the sentence (see _pools), so wait on the futures.
        for fut in pending:
            if not fut.cancel():
                with contextlib.suppress(BaseException):
                    fut.result(timeout=5.0)
        with contextlib.suppress(BaseException):
            th.result(timeout=5.0)
        stats["tokens"] = len(tokens)
        stats["audio_ms"] = round(emitted / 24000 * 1000)
        stats["total_ms"] = round((time.perf_counter() - t0) * 1000)


def _adaptive_chunk(
    plan: ChunkPlanConfig, tok_times: list[float], avail: int, emitted: int, t_first_emit: float
) -> int:
    """Tokens for the next chunk: as many as the emitted audio can cover.

    The chunk is ready after its missing tokens arrive (``avail`` are already here) at
    the measured step time, plus one decode; the audio runs out ``emitted`` samples
    after the first emission. Clamped to ``[min_chunk, chunk]``."""
    if len(tok_times) < 3:
        return plan.chunk
    recent = tok_times[-21:]
    step_s = float(np.median(np.diff(recent)))
    if step_s <= 0:
        return plan.chunk
    runout = t_first_emit + emitted / 24000
    budget_s = runout - time.perf_counter() - (plan.margin_ms + plan.decode_ms) / 1000
    fit = int(budget_s / step_s) if budget_s > 0 else 0
    size = (avail - plan.lookahead) + fit
    return max(plan.min_chunk, min(plan.chunk, size))


class _Either:
    """``is_set()`` when the caller's event is set or the pipeline halted itself."""

    def __init__(self, outer: threading.Event | None) -> None:
        self._outer = outer
        self._own = threading.Event()

    def is_set(self) -> bool:
        return self._own.is_set() or (self._outer is not None and self._outer.is_set())

    def halt(self) -> None:
        self._own.set()


class _Gated:
    """The LM's ``stop``: ``is_set()`` is checked before every step; while the
    gate is closed it waits there (5 ms polls), so a paused or credit-stalled
    context holds no GPU time. A halt ends the wait at once."""

    def __init__(self, halt: _Either, gate: Callable[[int], bool]) -> None:
        self._halt = halt
        self._gate = gate
        self.count = 0

    def is_set(self) -> bool:
        while not self._halt.is_set():
            if self._gate(self.count):
                return False
            time.sleep(0.005)
        return True


class Speaker:
    """Everything loaded onto one card: STT, TTS and end-of-turn adapters, plus
    the (CPU) VAD. Thread-safe as far as the adapters are (each serialises its
    own model calls)."""

    def __init__(
        self,
        tts: Any,
        stt: Any,
        *,
        eou: Any = None,
        vad: Any = None,
        shared_arena: bool = False,
    ) -> None:
        self.tts = tts
        self.stt = stt
        self.eou = eou
        self.vad = vad
        #: Every CUDA session uses the one environment arena (S24); /ready reports it.
        self.shared_arena = bool(shared_arena)
        self.last_stt: dict[str, Any] = {}
        self.selftest: dict[str, Any] = {}

    @property
    def describe(self) -> dict[str, Any]:
        out = {"tts": self.tts.describe(), "stt": self.stt.describe()}
        if self.eou is not None:
            out["eou"] = self.eou.describe()
        if self.vad is not None:
            out["vad"] = {"name": getattr(self.vad, "name", "vad")}
        return out

    @property
    def last_tts(self) -> dict[str, Any]:
        return self.tts.last_stats

    def parse_voice(self, data: bytes) -> Any:
        return self.tts.parse_voice(data)

    def speak(
        self,
        text: str,
        vid: str,
        voice: Any,
        *,
        stop: Any = None,
        seed: int | None = None,
        first_chunk_tokens: int | None = None,
        gate: Callable[[int], bool] | None = None,
        on_lm_done: Callable[[int], None] | None = None,
    ) -> Iterator[np.ndarray]:
        opts: dict[str, Any] = {}
        if first_chunk_tokens:
            opts["first_chunk_tokens"] = first_chunk_tokens
        if gate is not None:
            opts["gate"] = gate
        if on_lm_done is not None:
            opts["on_lm_done"] = on_lm_done
        yield from self.tts.stream(text, vid, voice, stop=stop, seed=seed, options=opts or None)

    def render(self, text: str, voice: Any, *, seed: int | None) -> np.ndarray:
        return self.tts.render(text, voice, seed=seed)

    def transcribe(self, audio: np.ndarray) -> str:
        t0 = time.perf_counter()
        text = self.stt.transcribe(audio)
        self.last_stt = {
            "audio_ms": round(len(audio) / self.stt.sample_rate * 1000),
            "cost_ms": round((time.perf_counter() - t0) * 1000),
        }
        return text

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
        fn = getattr(self.stt, "transcribe_words", None)
        if fn is None:
            return self.transcribe(audio), []
        return fn(audio)

    def eou_score(self, audio: np.ndarray) -> float | None:
        return None if self.eou is None else float(self.eou.score(audio))

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        return self.tts.build_voice(reference, rate)

    def warm_up(self) -> dict[str, Any]:
        """Exercise every session once so the arenas reach their working size
        before the first real request (S5)."""
        t0 = time.perf_counter()
        self.tts.warm_up()
        t1 = time.perf_counter()
        self.stt.warm_up()
        t2 = time.perf_counter()
        if self.eou is not None:
            self.eou.score(np.zeros(16000, np.float32))
        return {"tts_ms": round((t1 - t0) * 1000), "stt_ms": round((t2 - t1) * 1000)}

    def close(self) -> None:
        for part in (self.tts, self.stt, self.eou):
            if part is not None:
                with contextlib.suppress(Exception):  # closing never fails an unload
                    part.close()


def load_speaker(config: Any, device_id: int) -> Speaker:
    """Build the configured adapters on ``device_id`` (blocking: seconds).
    Every model file is verified against models.lock first (size + mtime
    stamp; a full sha256 when the stamp is older than a week)."""
    import os

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    import onnxruntime as ort

    from speechforge import models
    from speechforge.engines import arena, base
    from speechforge.engines.turn import EnergyVad, SileroVad, SmartTurn

    models.verify_for_load(config)
    shared = False
    if config.gpu.provider == "cuda":
        preload = getattr(ort, "preload_dlls", None)
        if callable(preload):
            # The pip CUDA/cuDNN wheels (the `gpu` extra): no system toolkit needed.
            preload()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "onnxruntime has no CUDAExecutionProvider: install the `gpu` "
                "extra (onnxruntime-gpu), not onnxruntime"
            )
        check_card_name(device_id, config.gpu.expect_name)
        # Before the first session: the one arena every session then shares (S24).
        shared = arena.ensure_shared_arena(config.gpu, device_id)
    base.builtin()
    try:
        tts_loader = base.TTS_ENGINES[config.tts.engine]
        stt_loader = base.STT_ENGINES[config.stt.engine]
    except KeyError as exc:
        raise RuntimeError(f"unknown engine {exc.args[0]!r}") from None
    parts: list[Any] = []
    try:
        tts = tts_loader(config, device_id)
        parts.append(tts)
        stt = stt_loader(config, device_id)
        parts.append(stt)
        eou = None
        if config.eou.enabled:
            eou = SmartTurn(
                models.path_of(config, "smart-turn"),
                provider=config.gpu.provider,
                device_id=device_id,
                shared_arena=shared,
            )
            parts.append(eou)
        vad_path = models.path_of(config, "silero-vad")
        vad = SileroVad(vad_path) if vad_path.is_file() else EnergyVad()
    except BaseException:
        for part in parts:
            with contextlib.suppress(Exception):
                part.close()
        raise
    return Speaker(tts, stt, eou=eou, vad=vad, shared_arena=shared)


def check_card_name(device_id: int, expect: str) -> None:
    """NVML name of ``device_id`` must contain ``expect`` (PCI-bus numbering)."""
    if not expect:
        return
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError("gpu.expect_name is set but nvidia-ml-py is not installed") from exc
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_id))
        name = pynvml.nvmlDeviceGetName(handle)
        name = name.decode() if isinstance(name, bytes) else str(name)
    finally:
        pynvml.nvmlShutdown()
    if expect.lower() not in name.lower():
        raise RuntimeError(
            f"CUDA {device_id} is {name!r}, expected a card matching "
            f"{expect!r} (gpu.expect_name); refusing to load"
        )
