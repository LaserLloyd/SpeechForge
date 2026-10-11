"""NVIDIA Parakeet TDT 0.6B v2 (CC-BY-4.0; ONNX export by istupakov, CC-BY-4.0)
through onnx-asr (MIT), on a CUDA card.

On the GPU the fp32 export is the right one: the int8 export's
``MatMulInteger``/``DynamicQuantizeLinear`` nodes have no fast CUDA kernels and
would bounce between host and device (S4). The model is whole-utterance; the
client's "tail mode" (SPEC §4.2 of the add-on) sends only the open tail at each
pause, so a call is 0.5-5 s of audio.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.engines.chatterbox import providers_for

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v2"


def model_files(quantization: str | None) -> list[str]:
    if quantization == "int8":
        enc, dec = ["encoder-model.int8.onnx"], ["decoder_joint-model.int8.onnx"]
    else:
        enc, dec = ["encoder-model.onnx", "encoder-model.onnx.data"], ["decoder_joint-model.onnx"]
    return ["config.json", "vocab.txt", "nemo128.onnx", *enc, *dec]


class ParakeetEngine:
    sample_rate = 16000

    def __init__(
        self,
        model_dir: Path,
        *,
        quantization: str | None = None,
        provider: str = "cuda",
        device_id: int = 0,
        mem_limit_mb: int = 6144,
        cpu_threads: int = 4,
    ) -> None:
        import onnx_asr
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        missing = [f for f in model_files(quantization) if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Parakeet files missing in {self.model_dir}: {', '.join(missing)} "
                f"(run `speechforge fetch-models`)"
            )
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if provider == "cpu":
            so.intra_op_num_threads = max(1, int(cpu_threads))
        self._model: Any = onnx_asr.load_model(
            MODEL_NAME,
            str(self.model_dir),
            quantization=quantization,
            sess_options=so,
            providers=providers_for(provider, device_id, mem_limit_mb),
        )
        self._timed: Any = self._model.with_timestamps()
        self.provider = provider
        self._lock = threading.Lock()

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
        """Text plus ``(word, start_s, end_s)`` from the TDT token timestamps
        (a token starting with a space starts a word; a word ends where the
        next one starts, the last one 80 ms after its last token)."""
        if audio.size < 1600:
            return "", []
        with self._lock:
            res = self._timed.recognize(
                np.ascontiguousarray(audio, np.float32), sample_rate=self.sample_rate
            )
        return (res.text or "").strip(), words_from_tokens(
            list(res.tokens or []), list(res.timestamps or [])
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz float32 mono -> text. Under 0.1 s is silence by definition."""
        if audio.size < 1600:
            return ""
        with self._lock:
            text = self._model.recognize(
                np.ascontiguousarray(audio, np.float32), sample_rate=self.sample_rate
            )
        return (text or "").strip()


def words_from_tokens(tokens: list[str], times: list[float]) -> list[tuple[str, float, float]]:
    words: list[list[Any]] = []
    for tok, t in zip(tokens, times, strict=False):
        if not words or tok.startswith(" "):
            text = tok.strip()
            if text:
                words.append([text, float(t), float(t)])
            elif words:
                words[-1][2] = float(t)
        else:
            words[-1][0] += tok
            words[-1][2] = float(t)
    out: list[tuple[str, float, float]] = []
    for i, (w, start, last) in enumerate(words):
        end = words[i + 1][1] if i + 1 < len(words) else last + 0.08
        out.append((w, round(start, 3), round(max(end, start), 3)))
    return out


class ParakeetStt:
    """:class:`~speechforge.engines.base.SttAdapter` over :class:`ParakeetEngine`."""

    name = "parakeet"
    sample_rate = 16000

    def __init__(self, engine: ParakeetEngine, *, quantization: str | None) -> None:
        self.engine = engine
        self.quantization = quantization

    def transcribe(self, audio: np.ndarray) -> str:
        return self.engine.transcribe(audio)

    def transcribe_words(self, audio: np.ndarray) -> tuple[str, list[tuple[str, float, float]]]:
        return self.engine.transcribe_words(audio)

    def warm_up(self) -> None:
        # One short window and the longest rolling-partial window (realtime.max_tail_s
        # is 8 s): the arena then holds its working set before the first session.
        self.engine.transcribe(np.zeros(16000, np.float32))
        self.engine.transcribe(np.zeros(8 * 16000, np.float32))

    def describe(self) -> dict[str, Any]:
        return {
            "name": "parakeet-tdt-0.6b-v2",
            "weights": "int8" if self.quantization == "int8" else "fp32",
            "provider": self.engine.provider,
        }

    def close(self) -> None:
        self.engine = None  # type: ignore[assignment]


def load(config: Any, device_id: int) -> ParakeetStt:
    q = config.models.stt_quantization
    eng = ParakeetEngine(
        config.models_dir / "parakeet-tdt-0.6b-v2",
        quantization=q,
        provider=config.gpu.provider,
        device_id=device_id,
        mem_limit_mb=config.gpu.mem_limit_mb,
    )
    return ParakeetStt(eng, quantization=q)
