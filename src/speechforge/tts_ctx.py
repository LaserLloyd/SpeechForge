"""One TTS context (``tts.open`` ... ``tts.done``) and its producer thread
(SPEC-speechforge §4.3, §4.4, §5.4, §5.5).

* **Units.** Text accumulates; a unit is cut when the buffer ends a sentence or
  clause (``. ! ? ; :``), on ``tts.text{final:true}``, ``tts.flush`` and
  ``tts.close``. Phase 1 renders unit by unit (the box splitter already sends
  clauses); a unit is one pipelined Chatterbox call, so its audio is exactly
  ``(tokens + 3) x 960`` samples and its length is known as soon as the LM
  finishes -- which is when ``tts.words`` goes out, ahead of the audio.
* **Frames.** 40 ms (960 samples @ 24 kHz) binary frames, ``pts`` = samples
  since the ctx's first sample, ``seq`` per ctx; the ctx's last frame may be
  shorter.
* **Credit / backpressure.** The LM waits before each step while the audio
  already generated (tokens x 40 ms) exceeds ``played_ms + ahead_ms``.
  ``played_ms`` is the box's report (``credit.played_ms``), else real-time
  playback since the first frame, minus paused time. ``tts.pause`` freezes it.
* **Cancel.** The stop event is checked before every LM step and every chunk:
  the thread is gone within one decode step; ``stop_ms`` is measured to the
  thread's exit.
* **Words** are *estimates*: Chatterbox has no aligner, so each unit's duration
  is split over its words in proportion to their length (+ a beat for
  punctuation). Good enough to cut "what was heard" at a word boundary; an
  engine with alignment replaces it per adapter.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from speechforge.metrics import METRICS, RATIO_BUCKETS

RATE = 24000
FRAME = 960
TOKEN_MS = 40
_UNIT_END = re.compile(r"[.!?;:][\"')\]]*\s*$")
_WORD = re.compile(r"\S+")


def estimate_words(text: str, start_ms: int, dur_ms: int) -> list[list[Any]]:
    toks = _WORD.findall(text)
    if not toks or dur_ms <= 0:
        return []
    weights = [
        max(1, len(re.sub(r"\W", "", t))) + (2 if re.search(r"[.,;:!?]$", t) else 0) for t in toks
    ]
    total = float(sum(weights))
    out: list[list[Any]] = []
    t = 0.0
    for tok, wgt in zip(toks, weights, strict=True):
        d = dur_ms * wgt / total
        word = re.sub(r"^[^\w']+|[^\w']+$", "", tok) or tok
        out.append([word, round(start_ms + t), round(start_ms + t + d)])
        t += d
    return out


class TtsContext:
    def __init__(
        self,
        *,
        ctx: int,
        gen: int,
        voice_id: str,
        voice: Any,
        speaker: Any,
        seed: int | None,
        first_chunk_tokens: int | None,
        turn_id: Any,
        speculative: bool,
        ahead_ms: int,
        max_ahead_ms: int,
        emit_audio: Callable[[bytes], None],
        emit: Callable[[dict[str, Any]], None],
        pack: Callable[[int, int, int, int, np.ndarray], bytes],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ctx, self.gen = ctx, gen
        self.voice_id, self.voice = voice_id, voice
        self.speaker = speaker
        self.seed = seed
        self.first_chunk_tokens = first_chunk_tokens
        self.turn_id = turn_id
        self.speculative = speculative
        self.ahead_ms = ahead_ms
        self.max_ahead_ms = max_ahead_ms
        self._emit_audio, self._emit, self._pack = emit_audio, emit, pack
        self._clock = clock
        self._cv = threading.Condition()
        self._buffer = ""
        self._units: list[tuple[str, Any]] = []  # (text, flush_id or None)
        self._closed = False
        self.stop = threading.Event()
        self.paused = False
        self._paused_since: float | None = None
        self._paused_total = 0.0
        self.played_ms: float | None = None
        self.text_chars = 0
        self.t_first_text: float | None = None
        self.t_first_frame: float | None = None
        self.tokens_total = 0
        self.unit_tokens = 0
        self.samples = 0
        self.seq = 0
        self._carry = np.zeros(0, "<i2")
        self.render_s = 0.0
        self.gated_s = 0.0
        self._stalled = False
        self.done = False
        self.cancelled = False
        self.thread: threading.Thread | None = None
        self.exited = threading.Event()

    # ----------------------------------------------------------- from the session
    def add_text(self, text: str, final: bool) -> None:
        with self._cv:
            if self.t_first_text is None and text.strip():
                self.t_first_text = self._clock()
            self.text_chars += len(text)
            self._buffer += text
            if final or _UNIT_END.search(self._buffer):
                self._cut(None)
            self._cv.notify_all()

    def flush(self, fid: Any) -> None:
        with self._cv:
            self._cut(fid, force=True)
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._cut(None)
            self._closed = True
            self._cv.notify_all()

    def _cut(self, fid: Any, force: bool = False) -> None:
        text = self._buffer.strip()
        self._buffer = ""
        if text:
            self._units.append((text, fid))
        elif fid is not None and force:
            self._units.append(("", fid))

    def set_credit(self, ahead_ms: float, played_ms: float | None) -> None:
        with self._cv:
            self.ahead_ms = int(max(0, min(self.max_ahead_ms, ahead_ms)))
            if played_ms is not None:
                self.played_ms = max(0.0, float(played_ms))
            self._cv.notify_all()

    def pause(self) -> None:
        with self._cv:
            if not self.paused:
                self.paused = True
                self._paused_since = self._clock()

    def resume(self) -> None:
        with self._cv:
            if self.paused:
                self.paused = False
                if self._paused_since is not None:
                    self._paused_total += self._clock() - self._paused_since
                self._paused_since = None
                self._cv.notify_all()

    def cancel(self) -> None:
        self.cancelled = True
        self.stop.set()
        with self._cv:
            self._cv.notify_all()

    # ----------------------------------------------------------- credit
    def assumed_played_ms(self) -> float:
        if self.played_ms is not None:
            return self.played_ms
        if self.t_first_frame is None:
            return 0.0
        paused = self._paused_total + (
            self._clock() - self._paused_since if self._paused_since is not None else 0.0
        )
        return max(0.0, (self._clock() - self.t_first_frame - paused) * 1000)

    def may_run(self, unit_count: int = 0) -> bool:
        if self.paused:
            return False
        produced = (self.tokens_total + unit_count) * TOKEN_MS
        return produced < self.assumed_played_ms() + self.ahead_ms

    def _gate(self, unit_count: int) -> bool:
        ok = self.may_run(unit_count)
        if not ok:
            if not self._stalled and not self.paused:
                METRICS.inc("tts_credit_stall_total")
            self.gated_s += 0.005
        self._stalled = not ok
        return ok

    # ----------------------------------------------------------- producer thread
    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name=f"sf-tts-ctx{self.ctx}", daemon=True)
        self.thread.start()

    def _next_unit(self) -> tuple[str, Any] | None:
        with self._cv:
            while not self._units and not self._closed and not self.stop.is_set():
                self._cv.wait(0.1)
            if self.stop.is_set() or not self._units:
                return None
            return self._units.pop(0)

    def _frames(self, pcm: np.ndarray, flush: bool) -> None:
        buf = np.concatenate([self._carry, pcm]) if len(self._carry) else pcm
        n_full = len(buf) // FRAME * FRAME
        for i in range(0, n_full, FRAME):
            self._send(buf[i : i + FRAME])
        self._carry = buf[n_full:]
        if flush and len(self._carry):
            rest, self._carry = self._carry, np.zeros(0, "<i2")
            self._send(rest)

    def _send(self, frame: np.ndarray) -> None:
        if self.stop.is_set():
            return
        if self.t_first_frame is None:
            self.t_first_frame = self._clock()
            ttfa = round((self.t_first_frame - (self.t_first_text or self.t_first_frame)) * 1000)
            METRICS.observe("tts_ttfa_ms", ttfa)
            self._emit(
                {"type": "tts.started", "ctx": self.ctx, "ttfa_ms": ttfa, "turn_id": self.turn_id}
            )
        self._emit_audio(self._pack(self.ctx, self.gen, self.seq, self.samples, frame))
        self.seq += 1
        self.samples += len(frame)

    def _run(self) -> None:
        first = True
        try:
            while not self.stop.is_set():
                unit = self._next_unit()
                if unit is None:
                    break
                text, fid = unit
                if text:
                    self._render(text, first)
                    first = False
                if fid is not None and not self.stop.is_set():
                    self._frames(np.zeros(0, "<i2"), flush=True)
                    self._emit(
                        {
                            "type": "tts.flushed",
                            "ctx": self.ctx,
                            "id": fid,
                            "audio_ms": round(self.samples * 1000 / RATE),
                        }
                    )
            if not self.stop.is_set():
                self._frames(np.zeros(0, "<i2"), flush=True)
                audio_ms = round(self.samples * 1000 / RATE)
                rtf = round(self.render_s * 1000 / audio_ms, 3) if audio_ms else None
                if rtf is not None:
                    METRICS.observe("tts_rtf", rtf, RATIO_BUCKETS)
                self._emit(
                    {
                        "type": "tts.done",
                        "ctx": self.ctx,
                        "audio_ms": audio_ms,
                        "rtf": rtf,
                        "gpu_ms": round((self.render_s - self.gated_s) * 1000),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - reported to the box, the session lives on
            self._emit(
                {
                    "type": "error",
                    "code": "engine_failed",
                    "ctx": self.ctx,
                    "fatal": False,
                    "message": type(exc).__name__,
                }
            )
        finally:
            self.done = True
            self.exited.set()

    def _render(self, text: str, first: bool) -> None:
        start_ms = round((self.samples + len(self._carry)) * 1000 / RATE)
        t0 = time.perf_counter()
        n_words: list[int] = []

        def on_lm_done(n: int) -> None:
            n_words.append(n)
            dur = (n + 3) * TOKEN_MS
            words = estimate_words(text, start_ms, n * TOKEN_MS)
            if words and not self.stop.is_set():
                self._emit(
                    {
                        "type": "tts.words",
                        "ctx": self.ctx,
                        "words": words,
                        "unit_ms": dur,
                        "estimated": True,
                    }
                )

        base = self.tokens_total

        def gate(count: int) -> bool:
            self.tokens_total = base + count
            return self._gate(0)

        for piece in self.speaker.speak(
            text,
            self.voice_id,
            self.voice,
            stop=self.stop,
            seed=self.seed,
            first_chunk_tokens=self.first_chunk_tokens if first else None,
            gate=gate,
            on_lm_done=on_lm_done,
        ):
            if self.stop.is_set():
                break
            self._frames(np.asarray(piece, "<i2"), flush=False)
        self.tokens_total = base + (n_words[0] if n_words else 0) + 3
        self.render_s += time.perf_counter() - t0
