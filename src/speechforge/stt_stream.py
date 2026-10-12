"""Rolling streaming STT on the rig, per session (SPEC-speechforge §5.1, §5.2).

Mic audio (16 kHz PCM16, 20 ms frames, ``pts`` on the session's mic clock)
arrives continuously. This object is bookkeeping; the session runs the jobs it
hands out on the STT lane (:meth:`RollingStt.decode_partial` and
:meth:`RollingStt.decode_final` are the job bodies, the only places that
touch a model):

* **VAD** (Silero, CPU, 512-sample windows) marks speech and pauses; the box's
  own Silero stays authoritative, the rig's is informational (``stt.vad``) and
  drives the two rig-side triggers below.
* **Partials** every ``partial_ms`` while there is new audio in an utterance
  that contained speech: re-decode the open tail (<= ``max_tail_s``). A longer
  tail is first *committed* at its last >= 300 ms pause (or cut hard), so a
  partial never re-decodes more than ``max_tail_s``. ``stable_words`` is v3's
  rule: words agreeing with the previous partial, minus 2.
* **EOU**: once per pause of ``min_pause_ms`` (never below the server floors, longer
  mid-sentence) after speech, Smart Turn on the
  last 8 s up to the pause start.
* **Final** (``stt.flush``): decode the tail with word timings, join the
  committed segments, report ``upto`` (mic clock) and start a new utterance
  from there. Audio that arrived after ``upto`` stays for the next one.

**Streaming adapters.** With a :class:`~speechforge.engines.base.StreamingSttAdapter`
(``stream`` is its per-session handle) a partial *feeds* only the audio that
arrived since the last one and takes the model's running text; nothing is
re-decoded and nothing is committed (the model keeps its own context). The
final feeds the rest, finalizes, and resets the handle for the next
utterance, which starts at ``upto``. Word timings come from the handle,
relative to the utterance start; a streaming model without timings falls
back to one ``transcribe_words`` pass over the buffered utterance when the
adapter has it. Parakeet's rolling path is untouched (``stream is None``).

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
    tail: np.ndarray  # streaming: only the audio not yet fed
    tail_start: int  # streaming: the utterance (= stream) start
    upto: int
    commit: np.ndarray | None = None
    commit_start: int = 0
    commit_end: int = 0
    streaming: bool = False
    prev_fed: int = 0  # streaming: the fed mark before this job advanced it


@dataclass
class FinalJob:
    tail: np.ndarray  # streaming: only the audio not yet fed
    tail_start: int  # streaming: the utterance (= stream) start
    upto: int
    fid: Any = None
    streaming: bool = False


@dataclass
class EouJob:
    audio: np.ndarray
    at: int
    pause_ms: int


@dataclass
class RollingStt:
    partial_ms: int = 160
    min_pause_ms: int = 150
    #: Server floors (``EouConfig``): the pause event waits for ``max(min_pause_ms,
    #: min_silence_ms)`` after a partial ending a sentence, and for the longer
    #: ``unpunctuated_silence_ms`` otherwise. 0 = no floor.
    min_silence_ms: int = 0
    unpunctuated_silence_ms: int = 0
    max_tail_s: float = 8.0
    max_utterance_s: float = 60.0
    vad: Any = None  # a per-session VAD state: callable(window float32) -> prob
    clock: Any = time.monotonic
    #: An ``SttStreamHandle`` when the STT adapter streams; None = rolling re-decode.
    stream: Any = None

    def __post_init__(self) -> None:
        self.reset_all()

    # ------------------------------------------------------------------ state
    def reset_all(self) -> None:
        self.expected = 0  # next mic sample index we expect (session clock)
        self._fresh = True  # no audio since a (re)set: the next frame's pts is the clock origin
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
        self._fed = 0  # streaming: mic index up to which the handle has been fed
        self._stream_reset()

    @property
    def streaming(self) -> bool:
        return self.stream is not None

    def _stream_reset(self) -> None:
        if self.stream is not None:
            self.stream.reset()

    def close_stream(self) -> None:
        """The session is gone: free the handle (a subprocess backend holds
        state per stream)."""
        if self.stream is not None:
            fn = getattr(self.stream, "close", None)
            if callable(fn):
                fn()
            self.stream = None

    def reset_utterance(self) -> None:
        """``stt.reset``: drop the open utterance; keep the clock."""
        end = self._buf_start + self._buf_len
        self._trim_to(end)
        self.utt_start = self.tail_start = end
        self._fed = end
        self._stream_reset()
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
        if self._fresh:
            # First frame after a (re)set -- a new session or a resume after a detach. The
            # client's mic clock kept running meanwhile, so adopt its pts as the origin
            # instead of zero-filling up to MAX_GAP_FILL_S of silence from sample 0.
            self.expected = self._buf_start = self.utt_start = self.tail_start = pts
            self._vad_pos = self._fed = pts
        elif pts > self.expected:  # a gap: zero-fill (bounded), keep the clock honest
            gap = pts - self.expected
            if gap > MAX_GAP_FILL_S * RATE:
                self.reset_all()
                self.expected = self._buf_start = self.utt_start = self.tail_start = pts
                self._vad_pos = self._fed = pts
            else:
                x = np.concatenate([np.zeros(gap, np.float32), x])
        elif pts < self.expected:  # overlap (a resend): drop what we have
            skip = self.expected - pts
            if skip >= len(x):
                return []
            x = x[skip:]
        self._fresh = False
        self._chunks.append(x)
        self._buf_len += len(x)
        self.expected = self.end
        self._arrivals.append((self.end, self.clock()))
        if len(self._arrivals) > 256:
            del self._arrivals[:128]
        events = self._run_vad(x)
        # bound memory: an utterance longer than max_utterance_s loses its oldest tail-less audio
        # (streaming: the handle keeps its own context; only the buffer is cut, the
        # utterance start -- what word timings are relative to -- stays put)
        limit = int(self.max_utterance_s * RATE)
        if self.end - self.tail_start > limit and not self.streaming:
            self.tail_start = self.end - limit
        keep = self.tail_start if not self.streaming else max(self.tail_start, self.end - limit)
        self._trim_to(min(keep, self._vad_pos) - RATE)
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
                if pause * 1000 >= self.pause_trigger_ms() * RATE:
                    self._pause_reported = True
                    events.append(
                        {
                            "type": "_pause",
                            "at": self.speech_end,
                            "pause_ms": round(pause * 1000 / RATE),
                        }
                    )
        return events

    def pause_trigger_ms(self) -> int:
        """Silence needed before the pause event fires: the client's ``min_pause_ms``
        raised to the server floor, and to the longer unpunctuated floor while the
        latest partial does not end in ``. ? !`` (a pause inside a sentence)."""
        ms = max(self.min_pause_ms, self.min_silence_ms)
        text = self._last_partial_text
        if text is None or not _PUNCT_END.search(text):
            ms = max(ms, self.unpunctuated_silence_ms)
        return ms

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
        if self.streaming:
            fed = max(self._fed, self.tail_start)
            prev_fed, self._fed = self._fed, upto
            return PartialJob(
                self.audio(fed, upto), self.tail_start, upto, streaming=True, prev_fed=prev_fed
            )
        commit = None
        cs = ce = 0
        if upto - self.tail_start > self.max_tail_s * RATE:
            ce = self._commit_boundary(upto)
            cs = self.tail_start
            commit = self.audio(cs, ce)
            self.tail_start = ce
            self._prev_tail_words = []
        return PartialJob(self.audio(self.tail_start, upto), self.tail_start, upto, commit, cs, ce)

    def release_partial(self, job: PartialJob) -> None:
        """Undo ``take_partial`` for a job that never ran (withdrawn from the
        lane), so the audio it would have fed or committed is still pending
        for the next job -- the final."""
        if job.streaming:
            self._fed = job.prev_fed
        elif job.commit is not None:
            self.tail_start = job.commit_start

    def _commit_boundary(self, upto: int) -> int:
        """End of the tail's last >= 300 ms pause, else a hard cut keeping 2 s."""
        for start, end in reversed(self._silences):
            if end - start >= COMMIT_PAUSE_S * RATE and self.tail_start < start < upto - RATE:
                return start + (end - start) // 2
        return max(self.tail_start + RATE, upto - 2 * RATE)

    def decode_partial(
        self, job: PartialJob, speaker: Any
    ) -> tuple[str, list[tuple[str, float, float]] | None, str]:
        """The partial's model work (runs on the STT lane): ``(commit_text,
        commit_words | None, tail_text)``. Streaming: feed the new audio, take
        the running text."""
        if job.streaming:
            return "", None, (self.stream.feed(job.tail) or "")
        commit = None
        if job.commit is not None and len(job.commit):
            commit = speaker.transcribe_words(job.commit)
        return (
            commit[0] if commit else "",
            commit[1] if commit else None,
            speaker.transcribe(job.tail),
        )

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
        if self.streaming:
            fed = max(self._fed, self.tail_start)
            self._fed = self.end
            return FinalJob(self.audio(fed, self.end), self.tail_start, self.end, fid, True)
        return FinalJob(self.audio(self.tail_start, self.end), self.tail_start, self.end, fid)

    def decode_final(
        self, job: FinalJob, speaker: Any
    ) -> tuple[str, list[tuple[str, float, float]]]:
        """The final's model work (runs on the STT lane): tail text and
        ``(word, start_s, end_s)`` relative to ``job.tail_start``. Streaming:
        feed what is left, finalize; a model without word timings gets one
        ``transcribe_words`` pass over the buffered utterance when the adapter
        offers it (``stt.final.words`` is part of the frame)."""
        if not job.streaming:
            return speaker.transcribe_words(job.tail)
        if len(job.tail):
            self.stream.feed(job.tail)
        text, words = self.stream.finalize()
        text = (text or "").strip()
        if text and not words:
            fn = getattr(getattr(speaker, "stt", None), "transcribe_words", None)
            if callable(fn):
                start = max(job.tail_start, self._buf_start)
                audio = self.audio(start, job.upto)
                if len(audio) >= RATE // 10:
                    text2, words2 = fn(audio)
                    shift = (start - job.tail_start) / RATE
                    text = (text2 or "").strip() or text
                    words = [(w, s + shift, e + shift) for w, s, e in words2]
        return text, list(words)

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
        # The next utterance starts at upto; later audio stays (streaming: it is
        # fed to the fresh handle by the next partial).
        self.utt_start = self.tail_start = job.upto
        if self.streaming:
            self._fed = job.upto
            self._stream_reset()
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
