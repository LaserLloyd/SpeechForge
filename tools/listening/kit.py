"""Shared pieces of the listening-test kit: lines, WAV I/O, loudness, labels,
the bearer token and the retrying HTTP call.

Everything here is numpy + the standard library (plus ``httpx`` for the two
scripts that talk to a running server). No audio or text is ever logged.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import struct
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_LINES = HERE / "lines.txt"
TOKEN_ENV = "SPEECHFORGE_TOKEN"
LABELS = "ABCD"
#: The default blind-set level: -23 dBFS RMS leaves a 10-20 dB crest factor for speech
#: before the -1 dBFS peak guard has to step in.
TARGET_RMS_DBFS = -23.0
PEAK_DBFS = -1.0


# ------------------------------------------------------------------ lines
def read_lines(path: str | os.PathLike[str] | None) -> list[str]:
    """Non-empty, non-comment lines of ``path`` (default ``lines.txt``), in order."""
    text = Path(path or DEFAULT_LINES).read_text(encoding="utf-8")
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    if not out:
        raise ValueError("no lines to render")
    return out


def line_name(n: int) -> str:
    """``01``, ``02``, ... the per-line directory/file stem (1-based)."""
    return f"{n:02d}"


# ------------------------------------------------------------------ WAV
def read_wav(path: str | os.PathLike[str]) -> tuple[np.ndarray, int]:
    """Mono float32 in [-1, 1] and the sample rate of a 16-bit PCM WAV."""
    return wav_to_float(Path(path).read_bytes())


def wav_to_float(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("WAV must be 16-bit PCM")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    return pcm, rate


def write_wav(path: str | os.PathLike[str], audio: np.ndarray, rate: int) -> None:
    pcm16 = (np.clip(np.asarray(audio, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def streamed_wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    """The PCM bytes and the rate of a WAV whose sizes may be the streaming
    placeholder (0xFFFFFFFF, ``streaming_wav_header``), which ``wave`` cannot read."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    pos, rate, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 :]
        if cid == b"fmt ":
            fmt, channels, rate, _br, _ba, bits = struct.unpack("<HHIIHH", body[:16])
            if fmt != 1 or channels != 1 or bits != 16:
                raise ValueError("expected mono 16-bit PCM")
        elif cid == b"data":
            pcm = body if size == 0xFFFFFFFF else body[:size]
            break
        if size == 0xFFFFFFFF:
            raise ValueError("unsized chunk before data")
        pos += 8 + size + (size & 1)
    if rate is None or pcm is None:
        raise ValueError("WAV without fmt or data chunk")
    return pcm[: len(pcm) // 2 * 2], rate


def duration_s(audio: np.ndarray, rate: int) -> float:
    return round(len(audio) / rate, 3) if rate else 0.0


# ------------------------------------------------------------------ loudness
def dbfs(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


def rms(audio: np.ndarray) -> float:
    a = np.asarray(audio, np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def normalise_rms(
    audio: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS, peak_dbfs: float = PEAK_DBFS
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale ``audio`` so its RMS sits at ``target_dbfs``, unless that would push the
    peak above ``peak_dbfs`` -- then the peak is the limit (the guard). Pure gain:
    no compression, no clipping, so timbre is untouched and the only thing the
    rater cannot use to tell engines apart is how loud they came out.

    Returns the scaled audio and the numbers (input/output RMS and peak, gain in dB)."""
    a = np.asarray(audio, np.float32)
    in_rms, in_peak = rms(a), float(np.max(np.abs(a))) if a.size else 0.0
    if in_rms <= 0 or in_peak <= 0:
        return a.copy(), {"gain_db": 0.0, "in_rms_dbfs": -math.inf, "out_rms_dbfs": -math.inf}
    gain = 10 ** (target_dbfs / 20) / in_rms
    peak_limit = 10 ** (peak_dbfs / 20)
    guarded = False
    if in_peak * gain > peak_limit:
        gain = peak_limit / in_peak
        guarded = True
    out = (a * gain).astype(np.float32)
    return out, {
        "gain_db": round(20 * math.log10(gain), 2),
        "in_rms_dbfs": round(dbfs(in_rms), 2),
        "out_rms_dbfs": round(dbfs(rms(out)), 2),
        "in_peak_dbfs": round(dbfs(in_peak), 2),
        "out_peak_dbfs": round(dbfs(float(np.max(np.abs(out)))), 2),
        "peak_guard": float(guarded),
    }


# ------------------------------------------------------------------ labels
def labels_for(count: int) -> list[str]:
    if not 2 <= count <= len(LABELS):
        raise ValueError(f"a blind set takes 2-{len(LABELS)} engines, got {count}")
    return list(LABELS[:count])


def shuffled_labels(rng: np.random.Generator, engines: list[str]) -> dict[str, str]:
    """``{label: engine}`` for one line: the engines in a random order under A/B/C/D."""
    order = list(engines)
    rng.shuffle(order)
    return dict(zip(labels_for(len(engines)), order, strict=True))


# ------------------------------------------------------------------ token + HTTP
def is_loopback_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").strip("[]").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")


def token_from_env() -> str:
    """The bearer token, from the environment only. Never a flag, never printed."""
    return os.environ.get(TOKEN_ENV, "").strip()


def auth_headers(url: str) -> dict[str, str]:
    token = token_from_env()
    if token:
        return {"Authorization": f"Bearer {token}"}
    if is_loopback_url(url):
        return {}
    raise SystemExit(
        f"{TOKEN_ENV} is not set and {url} is not loopback (set it in the environment)"
    )


def retry_after_s(headers: Any, default: float = 2.0, cap: float = 60.0) -> float:
    raw = headers.get("Retry-After") if headers is not None else None
    try:
        v = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        v = default
    return min(max(v, 0.1), cap)


def with_retries(
    call: Callable[[], Any],
    *,
    retries: int = 40,
    is_retryable: Callable[[Any], bool],
    headers_of: Callable[[Any], Any],
    sleep: Callable[[float], None] = time.sleep,
    on_wait: Callable[[float, int], None] | None = None,
) -> Any:
    """Run ``call`` until its result is not retryable (a 503 with ``Retry-After``:
    the server is loading, busy or cooling down -- DECISIONS S2 says come back later)."""
    for attempt in range(retries + 1):
        result = call()
        if not is_retryable(result) or attempt == retries:
            return result
        wait = retry_after_s(headers_of(result))
        if on_wait:
            on_wait(wait, attempt + 1)
        sleep(wait)
    return result  # pragma: no cover - the loop always returns


def error_message(body: bytes) -> str:
    try:
        err = json.loads(body.decode("utf-8", "replace")).get("error", {})
        return f"[{err.get('code', '?')}] {err.get('message', '')}".strip()
    except (ValueError, AttributeError):
        return body[:200].decode("utf-8", "replace")


# ------------------------------------------------------------------ misc
NICK_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def safe_nick(nick: str) -> str | None:
    """A rater's nickname as a file stem, or None when it is not one."""
    n = (nick or "").strip()
    return n if NICK_RE.match(n) else None


def json_dump(path: str | os.PathLike[str], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def json_load(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
