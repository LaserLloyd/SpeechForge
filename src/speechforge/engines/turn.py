"""End of turn and voice activity on the rig (SPEC-speechforge §5.2, §7.1).

* :class:`SmartTurn` -- pipecat Smart Turn v3.2, **GPU build** (BSD-2), on the
  CUDA EP: P(the speaker has finished) from the last 8 s. Run once per pause
  of ``eou.min_pause_ms`` (150 ms). The box's TurnPolicy decides; this only
  scores.
* :class:`SileroVad` -- Silero VAD v5 (MIT) on the ORT **CPU** provider, one
  512-sample window at a time, state per session. Informational on the wire
  (``stt.vad``); internally it is what finds the pauses that trigger ``eou``
  and gates the partial cadence.
* :class:`EnergyVad` -- an RMS gate for tests and for a missing Silero file.

``whisper_features`` is the add-on's numpy port of WhisperFeatureExtractor
(chunk_length=8) as Smart Turn calls it (same author, MIT), so the rig and the
box's CPU Smart Turn see identical inputs.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
VAD_WINDOW = 512  # Silero's native window at 16 kHz (32 ms)


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    f = np.asarray(f, np.float64)
    mel = f / (200.0 / 3)
    log_region = f >= 1000.0
    return np.where(
        log_region, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), mel
    )


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, np.float64)
    f = m * (200.0 / 3)
    log_region = m >= 15.0
    return np.where(log_region, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), f)


def mel_filters(n_mels: int = 80, n_fft: int = 400, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Slaney-scale, slaney-normalised filterbank, shape (n_freqs, n_mels)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_freqs)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    diff = np.diff(hz_pts)
    slopes = hz_pts[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    return fb * enorm[None, :]


_MEL: np.ndarray | None = None


def whisper_features(audio: np.ndarray, seconds: int = 8) -> np.ndarray:
    """(1, 80, seconds*100) float32: last ``seconds`` s, left-padded, normalised, log-mel."""
    global _MEL
    if _MEL is None:
        _MEL = mel_filters()
    n = seconds * SAMPLE_RATE
    x = np.asarray(audio, np.float64)[-n:]
    if len(x) < n:
        x = np.pad(x, (n - len(x), 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    n_fft, hop = 400, 160
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = np.hanning(n_fft + 1)[:-1]
    spec = np.fft.rfft(xp[idx] * window, axis=1)
    power = (np.abs(spec) ** 2)[:-1]
    mel = power @ _MEL
    log_spec = np.log10(np.maximum(mel, 1e-10)).T
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)[None, :, :]


class SmartTurn:
    name = "smart-turn-v3.2"

    def __init__(
        self,
        path: Path,
        *,
        provider: str = "cuda",
        device_id: int = 0,
        mem_limit_mb: int = 1024,
        shared_arena: bool = False,
    ) -> None:
        import onnxruntime as ort

        from speechforge.engines import arena
        from speechforge.engines.chatterbox import providers_for

        so = ort.SessionOptions()
        so.log_severity_level = 3
        if provider == "cpu":
            so.intra_op_num_threads = 1
        self.shared_arena = bool(shared_arena) and provider == "cuda"
        arena.configure_session_options(so, self.shared_arena)
        self._sess = ort.InferenceSession(
            str(path), so, providers=providers_for(provider, device_id, mem_limit_mb)
        )
        self.provider = provider
        self._lock = threading.Lock()

    def score(self, audio: np.ndarray) -> float:
        feats = whisper_features(audio)
        with self._lock:
            out = self._sess.run(None, {"input_features": feats})
        return float(np.asarray(out[0]).reshape(-1)[0])

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "provider": self.provider}

    def close(self) -> None:
        self._sess = None


class _SileroState:
    CONTEXT = 64

    def __init__(self, sess: Any) -> None:
        self._sess = sess
        self._state = np.zeros((2, 1, 128), np.float32)
        self._context = np.zeros(self.CONTEXT, np.float32)

    def __call__(self, window: np.ndarray) -> float:
        x = np.concatenate([self._context, window.astype(np.float32)])[None, :]
        out, self._state = self._sess.run(
            None, {"input": x, "state": self._state, "sr": np.array(SAMPLE_RATE, np.int64)}
        )
        self._context = window[-self.CONTEXT :].astype(np.float32)
        return float(out[0][0])


class SileroVad:
    name = "silero-vad-v5"

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self._sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def stream(self) -> _SileroState:
        return _SileroState(self._sess)


class _EnergyState:
    def __init__(self, threshold_db: float) -> None:
        self.threshold_db = threshold_db

    def __call__(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(window), dtype=np.float64)) + 1e-12)
        return 1.0 if 20 * np.log10(rms) > self.threshold_db else 0.0


class EnergyVad:
    """RMS gate: a test double and the no-model fallback. Not for noisy audio."""

    name = "energy"

    def __init__(self, threshold_db: float = -40.0) -> None:
        self.threshold_db = threshold_db

    def stream(self) -> _EnergyState:
        return _EnergyState(self.threshold_db)
