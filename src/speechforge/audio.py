"""PCM and WAV helpers. Standard library + numpy only (no soundfile on the rig)."""

from __future__ import annotations

import io
import math
import struct
import wave
from typing import Any

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
    at another rate). The low-latency raw route takes 16 kHz and never resamples.

    It does no anti-alias filtering, so content above the lower Nyquist folds back into
    the band. Fine for a throwaway STT feed; use :func:`resample_poly` wherever the
    result is kept or conditions a voice."""
    if src == dst or len(audio) == 0:
        return np.asarray(audio, np.float32)
    n = max(1, int(round(len(audio) * dst / src)))
    x = np.linspace(0.0, len(audio) - 1, n, dtype=np.float64)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def _poly_filters(up: int, down: int, zero_crossings: int, beta: float) -> np.ndarray:
    """Per-phase FIR taps for resampling by ``up/down``: shape ``(up, T)``, float32.

    One Kaiser-windowed sinc, designed at the up-sampled rate with its cutoff at the
    LOWER of the two Nyquist frequencies, then cut into ``up`` polyphase branches so
    the zero-stuffed signal is never built. The cutoff sits half a transition width
    below that Nyquist so the stop band starts there; the width follows from the
    Kaiser design rule for the window's attenuation."""
    m = max(up, down)
    half = zero_crossings * m
    # Kaiser's empirical beta -> attenuation (dB), good for beta >= 3.395.
    atten = beta / 0.1102 + 8.7
    width = (atten - 7.95) / (14.36 * 2 * half)  # transition width, cycles/sample
    fc = 0.5 / m - width / 2.0
    k = np.arange(-half, half + 1, dtype=np.float64)
    h = 2.0 * fc * np.sinc(2.0 * fc * k) * np.kaiser(2 * half + 1, beta)
    h *= up / h.sum()  # unity gain at DC once the up-stuffing's 1/up is paid back
    d0 = -(-half // up)  # ceil(half / up)
    taps = 2 * d0 + 1
    table = np.zeros((up, taps), np.float64)
    for p in range(up):
        idx = p + half + (np.arange(taps) - d0) * up
        ok = (idx >= 0) & (idx <= 2 * half)
        table[p, ok] = h[idx[ok]]
    return table.astype(np.float32)


def resample_poly(
    audio: np.ndarray,
    src: int,
    dst: int,
    *,
    zero_crossings: int = 32,
    beta: float = 8.6,
) -> np.ndarray:
    """Band-limited resampling by the rational factor ``dst/src`` (polyphase
    windowed-sinc FIR, pure numpy).

    Why not :func:`resample_linear`: linear interpolation is a ~6 dB/octave low-pass at
    best and does not filter before decimating, so on a 44.1 or 48 kHz reference the
    energy above 12 kHz folds down into the speech band as inaudible-on-playback but
    very measurable grit, which the speech encoder then reads as part of the voice. Here
    the cutoff is the lower Nyquist, the Kaiser window (``beta`` 8.6, about 80 dB)
    holds the stop band, and ``zero_crossings`` (32 per side, so 64 taps on the
    narrower rate) keeps the transition band under ~16 % of Nyquist. The ends are
    padded with the edge sample so a DC offset does not ring.

    Cost is ``len(out) * taps_per_phase`` multiply-adds, done in blocks, a fraction of
    a second for a 30 s clip. The result is float32 and ``round(len*dst/src)`` long."""
    audio = np.asarray(audio, np.float32)
    if src == dst or len(audio) == 0:
        return audio
    if src <= 0 or dst <= 0:
        raise ValueError("sample rates must be positive")
    g = math.gcd(int(src), int(dst))
    up, down = int(dst) // g, int(src) // g
    table = _poly_filters(up, down, zero_crossings, beta)
    taps = table.shape[1]
    d0 = (taps - 1) // 2
    n = len(audio)
    n_out = max(1, int(round(n * up / down)))
    xp = np.pad(audio, (d0, 2 * d0 + 2), mode="edge")
    out = np.empty(n_out, np.float32)
    t = np.arange(taps)
    block = 16384
    for m0 in range(0, n_out, block):
        m = np.arange(m0, min(n_out, m0 + block), dtype=np.int64) * down
        base, phase = m // up, m % up
        idx = (base + 2 * d0)[:, None] - t[None, :]
        out[m0 : m0 + len(m)] = np.einsum("mt,mt->m", xp[idx], table[phase])
    return out


REFERENCE_FRAME_MS = 20
#: A clip shorter than this after trimming gives the encoder too little to work with.
REFERENCE_MIN_S = 3.0
REFERENCE_QUIET_DBFS = -35.0
REFERENCE_LEAD_WARN_MS = 500.0
REFERENCE_CLIP_SAMPLES = 3
_FULL_SCALE = 0.999


def _db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 1e-9 else -180.0


def _warn(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def prepare_reference(
    audio24k: np.ndarray,
    *,
    trim_db: float = -40.0,
    margin_ms: float = 50.0,
    target_dbfs: float = -20.0,
    peak_dbfs: float = -1.0,
    dc: bool = True,
    clipped_samples: int | None = None,
    rate: int = TTS_SAMPLE_RATE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean a reference clip deterministically before the speech encoder sees it.

    A cloned voice is the encoder's view of this clip, so whatever is in it (a lead-in
    of room tone, a DC offset, a level 30 dB under a neighbour's) becomes part of the
    voice. Three plain operations, in order:

    1. ``dc``: subtract the mean (a sound-card offset wastes headroom and biases the
       level estimate).
    2. Trim leading and trailing silence. 20 ms frames; a frame is speech when its RMS
       is within ``trim_db`` of the clip's own loudest frame (relative, so a quiet
       recording trims as sensibly as a hot one). ``margin_ms`` of the original is kept
       either side so the first consonant and the last decay survive. Pauses INSIDE
       the clip are left alone: no gate, no noise reduction (either would mangle the
       timbre the encoder is trying to read).
    3. One gain so the trimmed clip's RMS is ``target_dbfs``, but never beyond what puts
       its peak at ``peak_dbfs``: a clip with big transients is peak-limited rather
       than RMS-matched. A pure gain, so nothing is clipped or compressed.

    The clip is not cut to a length (the encoder decides what it uses). ``clipped_samples``
    lets a caller that counted full-scale samples at the original rate (before a
    resample smears them) pass that count in; otherwise it is counted here.

    Returns ``(audio, stats)``. ``stats`` holds ``trimmed_lead_ms``, ``trimmed_tail_ms``,
    ``gain_db``, ``duration_s``, ``peak_dbfs_in`` (whole input), ``rms_dbfs_in`` (the
    trimmed region, i.e. the speech), ``clipped_samples`` and ``warnings``: a list of
    ``{"code", "message"}`` for ``short_clip``, ``quiet_clip``, ``clipping``,
    ``long_lead_silence`` or ``silent_clip``."""
    x = np.asarray(audio24k, np.float32)
    warnings: list[dict[str, str]] = []
    peak_in = float(np.max(np.abs(x))) if len(x) else 0.0
    clipped = (
        int(clipped_samples)
        if clipped_samples is not None
        else int(np.count_nonzero(np.abs(x) >= _FULL_SCALE))
    )
    stats: dict[str, Any] = {
        "trimmed_lead_ms": 0.0,
        "trimmed_tail_ms": 0.0,
        "gain_db": 0.0,
        "duration_s": round(len(x) / rate, 3),
        "peak_dbfs_in": round(_db(peak_in), 1),
        "rms_dbfs_in": -180.0,
        "clipped_samples": clipped,
        "warnings": warnings,
    }
    if len(x) == 0 or peak_in <= 1e-6:
        warnings.append(_warn("silent_clip", "the clip is silent"))
        return x.copy(), stats
    y = x - float(np.mean(x, dtype=np.float64)) if dc else x.copy()

    frame = max(1, int(rate * REFERENCE_FRAME_MS / 1000))
    n_frames = len(y) // frame
    if n_frames >= 1:
        rms = np.sqrt(
            np.mean(y[: n_frames * frame].astype(np.float64).reshape(n_frames, frame) ** 2, axis=1)
        )
        gate = float(rms.max()) * 10.0 ** (trim_db / 20.0)
        live = np.nonzero(rms >= gate)[0]
        margin = int(rate * margin_ms / 1000)
        start = max(0, int(live[0]) * frame - margin)
        end = min(len(y), (int(live[-1]) + 1) * frame + margin)
        # a tail shorter than one frame is not scored; it is kept if the last frame is live
        if int(live[-1]) == n_frames - 1:
            end = len(y)
        y = y[start:end]
        stats["trimmed_lead_ms"] = round(start * 1000.0 / rate, 1)
        stats["trimmed_tail_ms"] = round((len(x) - end) * 1000.0 / rate, 1)

    rms_in = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(y)))
    rms_db = _db(rms_in)
    gain_db = target_dbfs - rms_db
    gain_db = min(gain_db, peak_dbfs - _db(peak))
    y = (y * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32)
    stats["gain_db"] = round(gain_db, 1)
    stats["rms_dbfs_in"] = round(rms_db, 1)
    stats["duration_s"] = round(len(y) / rate, 3)

    if len(y) / rate < REFERENCE_MIN_S:
        warnings.append(
            _warn("short_clip", f"only {len(y) / rate:.1f} s of speech; 6-15 s works best")
        )
    if rms_db < REFERENCE_QUIET_DBFS:
        warnings.append(
            _warn(
                "quiet_clip", f"speech level {rms_db:.0f} dBFS is very low; noisy or too far away"
            )
        )
    if clipped >= REFERENCE_CLIP_SAMPLES:
        warnings.append(_warn("clipping", f"{clipped} samples at full scale; record lower"))
    if stats["trimmed_lead_ms"] > REFERENCE_LEAD_WARN_MS:
        warnings.append(
            _warn(
                "long_lead_silence",
                f"{stats['trimmed_lead_ms']:.0f} ms of silence trimmed from the start; "
                "start the clip at speech",
            )
        )
    return y, stats


def reference_clip(
    audio: np.ndarray,
    rate: int,
    *,
    prepare: bool = True,
    target_dbfs: float = -20.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Samples at ``rate`` -> the 24 kHz clip to build a voice from, plus the ``clip``
    stats. The one helper the HTTP route and the CPU bench voice share: band-limited
    resample, then :func:`prepare_reference` (or, with ``prepare=False``, the resampled
    samples as they are and only ``{"duration_s", "prepared": False}``)."""
    audio = np.asarray(audio, np.float32)
    clipped = int(np.count_nonzero(np.abs(audio) >= _FULL_SCALE))
    y = resample_poly(audio, rate, TTS_SAMPLE_RATE)
    if not prepare:
        return y, {"duration_s": round(len(y) / TTS_SAMPLE_RATE, 3), "prepared": False}
    y, stats = prepare_reference(y, target_dbfs=target_dbfs, clipped_samples=clipped)
    stats["prepared"] = True
    return y, stats


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
