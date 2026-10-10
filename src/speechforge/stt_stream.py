"""Rolling streaming STT on the rig, per session (SPEC-speechforge §5.1, §5.2).

Mic audio (16 kHz PCM16, 20 ms frames, ``pts`` on the session's mic clock)
arrives continuously. This object is pure bookkeeping -- no model calls; the
session runs the jobs it hands out on the STT lane:

* **VAD** (Silero, CPU, 512-sample windows) marks speech and pauses; the box's
  own Silero stays authoritative, the rig's is informational (``stt.vad``) and
  drives the two rig-side triggers below.
* **Partials** every ``partial_ms`` while there is new audio in an utterance
  that contained speech: re-decode the open tail (<= ``max_tail_s``). A longer
  tail is first *committed* at its last >= 300 ms pause (or cut hard), so a
  partial never re-decodes more than ``max_tail_s``. ``stable_words`` is v3's
  rule: words agreeing with the previous partial, minus 2.
* **EOU**: once per pause of ``min_pause_ms`` after speech, Smart Turn on the
  last 8 s up to the pause start.
* **Final** (``stt.flush``): decode the tail with word timings, join the
  committed segments, report ``upto`` (mic clock) and start a new utterance
  from there. Audio that arrived after ``upto`` stays for the next one.

All times on the wire are in the mic sample clock (16 kHz) except ``lag_ms``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

RATE = 16000
VAD_WINDOW = 512
SPEECH_ON = 0.5
SPEECH_OFF = 0.35
COMMIT_PAUSE_S = 0.3
MAX_GAP_FILL_S = 5.0
_PUNCT_END = re.compile(r"[.?!][\"')\]]*\s*$")


def words_of(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def agree(prev: list[str], cur: list[str]) -> int:
    n = 0
    for a, b in zip(prev, cur, strict=False):
        if a != b:
            break
        n += 1
    return n


@dataclass
class Word:
    w: str
    start: int  # mic samples
    end: int


@dataclass
class PartialJob:
    tail: np.ndarray
    tail_start: int
    upto: int
    commit: np.ndarray | None = None
    commit_start: int = 0
    commit_end: int = 0


@dataclass
class FinalJob:
    tail: np.ndarray
    tail_start: int
    upto: int
    fid: Any = None


@dataclass
class EouJob:
    audio: np.ndarray
    at: int
    pause_ms: int


@dataclass
class RollingStt:
    partial_ms: int = 160
    min_pause_ms: int = 150
    max_tail_s: float = 8.0
    max_utterance_s: float = 60.0
    vad: Any = None  # a per-session VAD state: callable(window float32) -> prob
    clock: Any = time.monotonic

    def __post_init__(self) -> None:
        self.reset_all()

    # ------------------------------------------------------------------ state
    def reset_all(self) -> None:
        self.expected = 0  # next mic sample index we expect (session clock)
        self._chunks: list[np.ndarray] = []
        self._buf_start = 0  # mic index of the first buffered sample
        self._buf_len = 0
        self.utt_start = 0
        self.tail_start = 0
        self.committed: list[tuple[str, list[Word]]] = []
        self._vad_buf = np.zeros(0, np.float32)
        self._vad_pos = 0  # mic index of _vad_buf[0]
        self.speech = False
        self.had_speech = False
        self.speech_end: int | None = None  # where the current pause began
        self._pause_reported = False
        self._silences: list[tuple[int, int]] = []  # (start, end) closed pauses in mic idx
        self._arrivals: list[tuple[int, float]] = []  # (end sample, monotonic arrival)
        self._last_partial_at = 0.0
        self._last_partial_upto = -1
        self._prev_tail_words: list[str] = []
        self._last_partial_text: str | None = None
        self.dropped_samples = 0

    def reset_utterance(self) -> None:
        """``stt.reset``: drop the open utterance; keep the clock."""
        end = self._buf_start + self._buf_len
        self._trim_to(end)
        self.utt_start = self.tail_start = end
        self.committed = []
        self.had_speech = self.speech
        self.speech_end = None if self.speech else end
        self._pause_reported = not self.speech
        self._silences = []
        self._prev_tail_words = []
        self._last_partial_text = None
        self._last_partial_upto = -1

    @property
    def end(self) -> int:
        return self._buf_start + self._buf_len

    def audio(self, start: int, end: int) -> np.ndarray:
        """Buffered mic samples [start, end) as float32."""
        start = max(start, self._buf_start)
        end = min(end, self.end)
        if end <= start:
            return np.zeros(0, np.float32)
        allbuf = np.concatenate(self._chunks) if len(self._chunks) != 1 else self._chunks[0]
        self._chunks = [allbuf]
        return allbuf[start - self._buf_start : end - self._buf_start].copy()

    def _trim_to(self, start: int) -> None:
        if start <= self._buf_start:
            return
        allbuf = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)
        cut = min(start - self._buf_start, len(allbuf))
        self._chunks = [allbuf[cut:]]
        self._buf_start += cut
        self._buf_len = len(self._chunks[0])

    # ------------------------------------------------------------------ ingest
    def push(self, pts: int, pcm16: np.ndarray) -> list[dict[str, Any]]:
        """One mic frame. Returns wire events (``stt.vad``) and internal
        triggers (``{"type": "_pause", "at", "pause_ms"}``)."""
        x = pcm16.astype(np.float32) / 32768.0
        if pts > self.expected:  # a gap: zero-fill (bounded), keep the clock honest
            gap = pts - self.expected
            if gap > MAX_GAP_FILL_S * RATE:
                self.reset_all()
                self.expected = self._buf_start = self.utt_start = self.tail_start = pts
                self._vad_pos = pts
            else:
                x = np.concatenate([np.zeros(gap, np.float32), x])
        elif pts < self.expected:  # overlap (a resend): drop what we have
            skip = self.expected - pts
            if skip >= len(x):
                return []
            x = x[skip:]
        self._chunks.append(x)
        self._buf_len += len(x)
        self.expected = self.end
        self._arrivals.append((self.end, self.clock()))
        if len(self._arrivals) > 256:
            del self._arrivals[:128]
        events = self._run_vad(x)
        # bound memory: an utterance longer than max_utterance_s loses its oldest tail-less audio
        limit = int(self.max_utterance_s * RATE)
        if self.end - self.tail_start > limit:
            self.tail_start = self.end - limit
        self._trim_to(min(self.tail_start, self._vad_pos) - RATE)
        return events

    def _run_vad(self, x: np.ndarray) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        self._vad_buf = np.concatenate([self._vad_buf, x])
        while len(self._vad_buf) >= VAD_WINDOW:
            win = self._vad_buf[:VAD_WINDOW]
            self._vad_buf = self._vad_buf[VAD_WINDOW:]
            p = float(self.vad(win)) if self.vad is not None else 0.0
            at = self._vad_pos
            self._vad_pos += VAD_WINDOW
            if not self.speech and p >= SPEECH_ON:
                self.speech = True
                if self.speech_end is not None and self.had_speech:
                    self._silences.append((self.speech_end, at))
                self.had_speech = True
                self.speech_end = None
                self._pause_reported = False
                events.append({"type": "stt.vad", "speech": True, "at": at})
            elif self.speech and p < SPEECH_OFF:
                self.speech = False
                self.speech_end = at
                events.append({"type": "stt.vad", "speech": False, "at": at})
            if (
                not self.speech
                and self.had_speech
                and self.speech_end is not None
                and not self._pause_reported
            ):
                pause = self._vad_pos - self.speech_end
                if pause * 1000 >= self.min_pause_ms * RATE:
                    self._pause_reported = True
                    events.append(
                        {
                            "type": "_pause",
                            "at": self.speech_end,
                            "pause_ms": round(pause * 1000 / RATE),
                        }
                    )
        return events

    # ------------------------------------------------------------------ jobs
    def partial_due(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return (
            self.had_speech
            and self.end > self._last_partial_upto
            and self.end - self.tail_start >= RATE // 10
            and (now - self._last_partial_at) * 1000 >= self.partial_ms
        )

    def take_partial(self, now: float | None = None) -> PartialJob:
        self._last_partial_at = self.clock() if now is None else now
        upto = self.end
        self._last_partial_upto = upto
        commit = None
        cs = ce = 0
        if upto - self.tail_start > self.max_tail_s * RATE:
            ce = self._commit_boundary(upto)
            cs = self.tail_start
            commit = self.audio(cs, ce)
            self.tail_start = ce
            self._prev_tail_words = []
        return PartialJob(self.audio(self.tail_start, upto), self.tail_start, upto, commit, cs, ce)

    def _commit_boundary(self, upto: int) -> int:
        """End of the tail's last >= 300 ms pause, else a hard cut keeping 2 s."""
        for start, end in reversed(self._silences):
            if end - start >= COMMIT_PAUSE_S * RATE and self.tail_start < start < upto - RATE:
                return start + (end - start) // 2
        return max(self.tail_start + RATE, upto - 2 * RATE)

    def apply_commit(self, text: str, words: list[tuple[str, float, float]], start: int) -> None:
        self.committed.append(
            (text, [Word(w, start + int(s * RATE), start + int(e * RATE)) for w, s, e in words])
        )

    def apply_partial(
        self, job: PartialJob, tail_text: str, cost_ms: int, now: float | None = None
    ) -> dict[str, Any] | None:
        """The ``stt.partial`` frame, or None when the text did not change."""
        now = self.clock() if now is None else now
        committed = " ".join(t for t, _ in self.committed if t).strip()
        tail_text = (tail_text or "").strip()
        cur = words_of(tail_text)
        stable = max(0, agree(self._prev_tail_words, cur) - 2) if self._prev_tail_words else 0
        self._prev_tail_words = cur
        full = f"{committed} {tail_text}".strip()
        if full == self._last_partial_text:
            return None
        self._last_partial_text = full
        return {
            "type": "stt.partial",
            "committed": committed,
            "tail": tail_text,
            "stable_words": stable,
            "punct_final": bool(_PUNCT_END.search(tail_text)),
            "upto": job.upto,
            "lag_ms": self.lag_ms(job.upto, now),
            "ms": cost_ms,
        }

    def lag_ms(self, upto: int, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        for end, t in reversed(self._arrivals):
            if end <= upto:
                return max(0, round((now - t) * 1000))
        return 0

    def take_final(self, fid: Any = None) -> FinalJob:
        return FinalJob(self.audio(self.tail_start, self.end), self.tail_start, self.end, fid)

    def apply_final(
        self,
        job: FinalJob,
        tail_text: str,
        tail_words: list[tuple[str, float, float]],
        cost_ms: int,
    ) -> dict[str, Any]:
        texts = [t for t, _ in self.committed if t]
        words = [w for _, ws in self.committed for w in ws]
        if tail_text:
            texts.append(tail_text.strip())
        words += [
            Word(w, job.tail_start + int(s * RATE), job.tail_start + int(e * RATE))
            for w, s, e in tail_words
        ]
        frame: dict[str, Any] = {
            "type": "stt.final",
            "text": " ".join(texts).strip(),
            "words": [[w.w, w.start, w.end] for w in words],
            "upto": job.upto,
            "ms": cost_ms,
        }
        if job.fid is not None:
            frame["id"] = job.fid
        # The next utterance starts at upto; later audio stays.
        self.utt_start = self.tail_start = job.upto
        self.committed = []
        self._silences = [(s, e) for s, e in self._silences if s >= job.upto]
        self.had_speech = self.speech
        self._pause_reported = not self.speech
        if not self.speech:
            self.speech_end = max(job.upto, self.speech_end or job.upto)
        self._prev_tail_words = []
        self._last_partial_text = None
        self._last_partial_upto = job.upto
        self._trim_to(min(job.upto, self._vad_pos) - RATE)
        return frame

    def take_eou(self, at: int, pause_ms: int) -> EouJob:
        start = max(self.utt_start, at - 8 * RATE)
        return EouJob(self.audio(start, at), at, pause_ms)
