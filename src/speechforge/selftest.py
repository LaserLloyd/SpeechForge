"""The post-load self-test (SPEC-speechforge §7.4, §10.4): a session only gets
``ready`` after this passes, and the blue/green updater runs it on the canary.

1. **Golden STT**: the shipped synthetic clip (``assets/selftest-voice.wav``,
   not a real person) transcribes to :data:`GOLDEN_TEXT` within ``max_stt_wer``.
2. **TTS TTFA**: a voice built from the same clip speaks a 6-word line; time to
   the first audio piece <= ``max_ttfa_ms`` (enforced on CUDA only).
3. **Cancel**: a long line is stopped after its first piece; the producer is
   gone within ``max_cancel_ms`` (enforced on CUDA only).
4. **Round trip**: the 6-word line's audio transcribes back within
   ``max_roundtrip_wer`` (the synthesised-speech WER check, §11.3).

No audio or text leaves this function except the scores.
"""

from __future__ import annotations

import io
import re
import threading
import time
import wave
from importlib import resources
from typing import Any

import numpy as np

from speechforge.audio import resample_linear

GOLDEN_TEXT = (
    "Hey, welcome back to the show. Today we're digging into something I've been "
    "curious about for weeks. Honestly, it's one of those topics that sound simple "
    "but the more you look, the stranger it gets."
)
TTS_LINE = "Sure, I can help with that."
LONG_LINE = (
    "This is a much longer sentence that keeps going so that the cancel check has "
    "something to interrupt while it is still being spoken aloud."
)


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)


def golden_clip() -> tuple[np.ndarray, int]:
    data = resources.files("speechforge").joinpath("assets/selftest-voice.wav").read_bytes()
    with wave.open(io.BytesIO(data)) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    return pcm, rate


def run(speaker: Any, cfg: Any, *, enforce_timing: bool) -> dict[str, Any]:
    clip, rate = golden_clip()
    out: dict[str, Any] = {"pass": False, "failed": []}
    # 1. golden STT
    stt_wer = wer(GOLDEN_TEXT, speaker.transcribe(resample_linear(clip, rate, 16000)))
    out["stt_wer"] = round(stt_wer, 3)
    if stt_wer > cfg.max_stt_wer:
        out["failed"].append("stt_wer")
    # 2. TTS TTFA (voice built from the synthetic clip)
    vbytes = speaker.build_voice(clip, rate)
    voice = speaker.parse_voice(vbytes)
    list(speaker.speak("Warm.", "_selftest", voice, seed=1))  # prepare the voice prefix
    t0 = time.perf_counter()
    ttfa = None
    pieces = []
    for piece in speaker.speak(TTS_LINE, "_selftest", voice, seed=1):
        if ttfa is None:
            ttfa = (time.perf_counter() - t0) * 1000
        pieces.append(piece)
    out["ttfa_ms"] = round(ttfa or 0.0, 1)
    if ttfa is None or (enforce_timing and ttfa > cfg.max_ttfa_ms):
        out["failed"].append("ttfa")
    # 3. cancel
    stop = threading.Event()
    gen = speaker.speak(LONG_LINE, "_selftest", voice, stop=stop, seed=2)
    next(gen, None)
    t1 = time.perf_counter()
    stop.set()
    for _ in gen:
        pass
    cancel_ms = (time.perf_counter() - t1) * 1000
    out["cancel_ms"] = round(cancel_ms, 1)
    if enforce_timing and cancel_ms > cfg.max_cancel_ms:
        out["failed"].append("cancel")
    # 4. round trip
    audio = np.concatenate(pieces).astype(np.float32) / 32768 if pieces else np.zeros(0)
    rt = wer(TTS_LINE, speaker.transcribe(resample_linear(audio, 24000, 16000)))
    out["roundtrip_wer"] = round(rt, 3)
    if rt > cfg.max_roundtrip_wer:
        out["failed"].append("roundtrip_wer")
    out["timing_enforced"] = enforce_timing
    out["pass"] = not out["failed"]
    return out
