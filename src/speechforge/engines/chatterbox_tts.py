""":class:`~speechforge.engines.base.TtsAdapter` for Chatterbox-Turbo.

Voice format ``chatterbox-npz``: the four speech-encoder tensors
(:mod:`speechforge.voices`). Per voice it keeps the decoder's sliced prompt and
the device-resident LM prefix (a handful, LRU), and streams through
:func:`speechforge.pipeline.pipelined`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np

from speechforge.audio import TTS_SAMPLE_RATE, float_to_pcm16, resample_linear
from speechforge.engines.chatterbox import ChatterboxEngine, slice_prompt
from speechforge.pipeline import ChunkPlanConfig, pipelined
from speechforge.voices import VoiceTensors, parse_npz


class ChatterboxTts:
    name = "chatterbox"
    sample_rate = TTS_SAMPLE_RATE
    voice_format = "chatterbox-npz"

    def __init__(self, engine: Any, *, plan: ChunkPlanConfig, max_prepared: int = 8) -> None:
        self.engine = engine
        self.plan = plan
        self._prepared: OrderedDict[str, tuple[VoiceTensors, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_prepared = max_prepared
        self._last: dict[str, Any] = {}

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last)

    def parse_voice(self, data: bytes) -> VoiceTensors:
        return parse_npz(data)

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        if rate != TTS_SAMPLE_RATE:
            reference = resample_linear(reference, rate, TTS_SAMPLE_RATE)
        return self.engine.build_voice(np.asarray(reference, np.float32)).to_npz()

    def _prepare(self, vid: str, voice: VoiceTensors) -> tuple[VoiceTensors, Any]:
        with self._lock:
            got = self._prepared.get(vid)
            if got is not None:
                self._prepared.move_to_end(vid)
                return got
        sliced = slice_prompt(voice, self.plan.prompt_s)
        fn = getattr(self.engine, "prefix_cache", None)
        prefix = fn(voice) if callable(fn) else None
        with self._lock:
            self._prepared[vid] = (sliced, prefix)
            while len(self._prepared) > self._max_prepared:
                self._prepared.popitem(last=False)
        return sliced, prefix

    def stream(
        self,
        text: str,
        voice_id: str,
        voice: VoiceTensors,
        *,
        stop: threading.Event | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[np.ndarray]:
        sliced, prefix = self._prepare(voice_id, voice)
        plan = self.plan
        first = (options or {}).get("first_chunk_tokens")
        if first:
            plan = ChunkPlanConfig(int(first), plan.chunk, plan.left, plan.lookahead, plan.prompt_s)
        stats: dict[str, Any] = {}
        try:
            # The LM reads the full conditioning (cond_emb is never sliced); the
            # decoder reads the sliced prompt.
            yield from pipelined(
                self.engine,
                text,
                sliced,
                plan=plan,
                prefix=prefix,
                stop=stop,
                seed=seed,
                stats=stats,
                gate=(options or {}).get("gate"),
                on_lm_done=(options or {}).get("on_lm_done"),
            )
        finally:
            self._last = stats

    def render(self, text: str, voice: VoiceTensors, *, seed: int | None = None) -> np.ndarray:
        return float_to_pcm16(self.engine.synthesize(text, voice, seed=seed))

    def warm_up(self) -> None:
        from speechforge.audio import synthetic_reference

        voice = parse_npz(self.build_voice(synthetic_reference(), TTS_SAMPLE_RATE))
        list(self.stream("Warming up the voice.", "_warmup", voice, seed=0))
        with self._lock:
            self._prepared.pop("_warmup", None)

    def describe(self) -> dict[str, Any]:
        eng = self.engine
        return {
            "name": "chatterbox-turbo",
            "weights": getattr(eng, "dtype", "?"),
            "provider": getattr(eng, "provider", "?"),
            "model_tag": eng.model_tag() if hasattr(eng, "model_tag") else None,
            "first_chunk_tokens": self.plan.first,
            "prompt_s": self.plan.prompt_s,
        }

    def close(self) -> None:
        with self._lock:
            self._prepared.clear()
        self.engine = None


def load(config: Any, device_id: int) -> ChatterboxTts:
    t = config.tts
    eng = ChatterboxEngine(
        config.models_dir / "chatterbox-turbo",
        dtype=config.models.tts_dtype,
        provider=config.gpu.provider,
        device_id=device_id,
        mem_limit_mb=config.gpu.mem_limit_mb,
        temperature=t.temperature,
        top_k=t.top_k,
        top_p=t.top_p,
        repetition_penalty=t.repetition_penalty,
    )
    plan = ChunkPlanConfig(
        t.first_chunk_tokens, t.chunk_tokens, t.left_context, t.lookahead_tokens, t.prompt_s
    )
    return ChatterboxTts(eng, plan=plan)
