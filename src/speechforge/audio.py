"""PCM and WAV helpers. Standard library + numpy only (no soundfile on the rig)."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

from speechforge.errors import BadRequest

TTS_SAMPLE_RATE = 24000
STT_SAMPLE_RATE = 16000


def pcm16_to_float(raw: bytes) -> np.ndarray:
    if len(raw) % 2:
        raise BadRequest(
            "raw PCM must be 16-bit little-endian: an even number of bytes", code="bad_audio"
        )
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(audio, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 and the sample rate from a 16-bit PCM WAV. Stereo is averaged."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        raise BadRequest(
            f"not a readable WAV file ({type(exc).__name__})", code="bad_audio"
        ) from exc
    if width != 2:
        raise BadRequest("WAV must be 16-bit PCM", code="bad_audio")
    if channels < 1:
        raise BadRequest("WAV has no channels", code="bad_audio")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
    return pcm, rate


def wav_bytes(pcm16: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(pcm16, "<i2").tobytes())
    return buf.getvalue()


def streaming_wav_header(rate: int) -> bytes:
    """A WAV header for a stream of unknown length (sizes 0xFFFFFFFF, as ffmpeg
    and browsers accept for a live stream)."""
    byte_rate = rate * 2
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, byte_rate, 2, 16)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


def resample_linear(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear-interpolation resampling, for the convenience routes (a WAV upload
    at another rate). The low-latency raw route takes 16 kHz and never resamples."""
    if src == dst or len(audio) == 0:
        return np.asarray(audio, np.float32)
    n = max(1, int(round(len(audio) * dst / src)))
    x = np.linspace(0.0, len(audio) - 1, n, dtype=np.float64)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Equal-power crossfade of two same-length float32 pieces (cos out, sin in).
    The pipelined decoder's chunks are independent noise draws; butted, they click."""
    n = min(len(tail), len(head))
    if n <= 0:
        return np.zeros(0, np.float32)
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    return (tail[:n] * np.cos(t) + head[:n] * np.sin(t)).astype(np.float32)


def synthetic_reference(seconds: float = 3.0, rate: int = TTS_SAMPLE_RATE) -> np.ndarray:
    """A deterministic voiced-ish signal for warm-up: a gliding harmonic tone with
    a little noise. It builds a valid (if odd) voice and allocates every arena."""
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    f0 = 120.0 + 40.0 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    rng = np.random.default_rng(0)
    sig = sig + 0.02 * rng.standard_normal(n)
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)
