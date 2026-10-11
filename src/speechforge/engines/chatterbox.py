"""Chatterbox-Turbo (Resemble AI, MIT) through ONNX Runtime, on a CUDA card.

The algorithm is the DisPatch voice add-on's CPU engine (``dispatch_voice.engine``,
same author, MIT) -- same pinned ONNX export, same sampling, same token
arithmetic -- so a voice profile and a seed mean the same thing on both sides.
What changes on a GPU (S4/S5):

* **The KV cache stays on the card.** The CPU engine hands ``present.*`` back
  as numpy and feeds it in again: on a GPU that would copy 48 tensors
  (~40 MB at 200 positions) device->host->device on every token. Here the
  language-model step runs through an ``IOBinding``: ``present.*`` are bound to
  device memory and become the next step's ``past_key_values.*`` as
  ``OrtValue`` s, never touching the host. Only the logits row (6563 floats)
  and the next token's embedding (1024 floats) cross the bus per token.
* **The voice prefix's KV cache** (``prefix_cache``) is also device-resident,
  computed once per voice; ORT never writes an input, so it is shared by every
  sentence in that voice.
* **Inputs are cast to what each session declares**, so the fp16 export (and a
  q4f16 one) take the same float32 voice tensors as the fp32 one.
* **No silent CPU fallback.** With ``provider="cuda"`` every session must
  report the CUDA provider first, or loading fails -- a rig that quietly ran on
  its CPU would be slower than the client's own engine and look healthy.

Watermarking: the Perth watermark Resemble's PyTorch package applies needs
torch; like the add-on's pipelined path, this engine does not apply it.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.metrics import METRICS
from speechforge.voices import VoiceTensors

START_SPEECH, STOP_SPEECH, SILENCE = 6561, 6562, 4299
NUM_KV_HEADS, HEAD_DIM = 16, 64
TOKENS_PER_SECOND = 25
SAMPLE_RATE = 24000
#: 960 samples of 24 kHz audio per speech token (measured by the add-on).
SAMPLES_PER_TOKEN = SAMPLE_RATE // TOKENS_PER_SECOND
#: Decoder sequence lengths (speech tokens incl. context and trailing silence)
#: are padded up to one of these with SILENCE tokens whose audio is dropped.
#: Measured on the host (DECISIONS S18): the CUDA provider pays 150-700 ms the
#: first time it sees a shape AND ~200 ms on every switch between shapes it has
#: already seen, while the decode itself costs ~55 ms whether the input is 8 or
#: 40 tokens. So every chunk the plan can produce (first 15+3, middle 8+25+3,
#: last <= 38, plus 3 silence) is padded to ONE shape; 64 only catches an
#: oversized chunk. Warm-up decodes each bucket once.
DECODE_BUCKETS: tuple[int, ...] = (40, 64)
COMPONENTS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")
DTYPES = ("fp32", "fp16", "q8", "q4", "q4f16")

_NP_TYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def filename(component: str, dtype: str) -> str:
    suffix = "" if dtype == "fp32" else "_quantized" if dtype == "q8" else f"_{dtype}"
    return f"{component}{suffix}.onnx"


def model_files(dtype: str) -> list[str]:
    """Every file the engine needs for ``dtype``, relative to the model dir."""
    files = ["tokenizer.json", "config.json"]
    for c in COMPONENTS:
        fn = filename(c, dtype)
        files += [f"onnx/{fn}", f"onnx/{fn}_data"]
    return files


def normalise_text(text: str) -> str:
    """Tidy spacing, capitalise, end on punctuation (the add-on's punc_norm)."""
    t = text or ""
    for a, b in (
        ("…", "..."),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
        ("—", ", "),
        ("–", ", "),
    ):
        t = t.replace(a, b)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    t = re.sub(r",{2,}", ",", t).strip(" ,")
    if not t:
        return t
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?,;:\"'" and not t.endswith("]"):
        t += "."
    return t


@dataclass
class PrefixKV:
    """The LM's KV cache after the voice's conditioning prefix: ``OrtValue`` s on
    the device (or numpy on the CPU path), keyed by ``past_key_values.*`` name."""

    length: int
    kv: dict[str, Any]


def slice_prompt(voice: VoiceTensors, prompt_s: float | None) -> VoiceTensors:
    """Keep the first ``prompt_s`` s of the decoder's reference prompt
    (``prompt_token`` 25/s, ``speaker_features`` 50/s); ``cond_emb`` untouched."""
    if prompt_s is None:
        return voice
    n_tok = round(float(prompt_s) * TOKENS_PER_SECOND)
    if n_tok <= 0 or n_tok >= voice.prompt_token.shape[1]:
        return voice
    feat = voice.speaker_features[:, : n_tok * 2, :]
    return VoiceTensors(
        voice.cond_emb,
        voice.prompt_token[:, :n_tok],
        voice.speaker_embeddings,
        np.ascontiguousarray(feat),
    )


class ChatterboxEngine:
    """Thread-safe: one LM *step* at a time (``_lm_lock``), one decode at a
    time (``_dec_lock``) -- the two overlap, which is the pipeline (S5)."""

    sample_rate = SAMPLE_RATE
    #: Class-level so fakes built without __init__ (tests) get the default too.
    decode_buckets: tuple[int, ...] = DECODE_BUCKETS

    def __init__(
        self,
        model_dir: Path,
        *,
        dtype: str = "fp32",
        provider: str = "cuda",
        device_id: int = 0,
        mem_limit_mb: int = 6144,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        cpu_threads: int = 4,
        use_iobinding: bool | None = None,
    ) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.provider = provider
        self.device_id = int(device_id)
        missing = [f for f in model_files(dtype) if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Chatterbox files missing in {self.model_dir}: {', '.join(missing[:4])}"
                f"{' ...' if len(missing) > 4 else ''} (run `speechforge fetch-models`)"
            )
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.repetition_penalty = repetition_penalty
        self._device_type = "cuda" if provider == "cuda" else "cpu"
        self._iobinding = (provider == "cuda") if use_iobinding is None else bool(use_iobinding)
        self._s: dict[str, Any] = {}
        for c in COMPONENTS:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3
            if provider == "cpu":
                so.intra_op_num_threads = max(1, int(cpu_threads))
            sess = ort.InferenceSession(
                str(self.model_dir / "onnx" / filename(c, dtype)),
                so,
                providers=providers_for(provider, device_id, mem_limit_mb),
            )
            if provider == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(
                    f"{c}: ONNX Runtime did not place the session on CUDA "
                    f"(got {sess.get_providers()}); refusing to serve from the CPU"
                )
            self._s[c] = sess
        lm = self._s["language_model"]
        self._kv_inputs = [i for i in lm.get_inputs() if "past_key_values" in i.name]
        self._kv_names = [i.name for i in self._kv_inputs]
        self._present_names = [o.name for o in lm.get_outputs()[1:]]
        if len(self._present_names) != len(self._kv_names):
            raise RuntimeError("language_model: present/past key-value counts differ")
        self._kv_dtype = _NP_TYPES.get(self._kv_inputs[0].type, np.float32)
        self._in_types = {
            c: {i.name: _NP_TYPES.get(i.type, np.float32) for i in s.get_inputs()}
            for c, s in self._s.items()
        }
        self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self._lm_lock = threading.Lock()
        self._dec_lock = threading.Lock()
        self._enc_lock = threading.Lock()
        self.last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------ identity #
    def model_tag(self) -> str:
        from speechforge.config import CHATTERBOX_REVISION

        return hashlib.sha256(f"{CHATTERBOX_REVISION}:{self.dtype}".encode()).hexdigest()[:8]

    def _cast(self, component: str, name: str, arr: np.ndarray) -> np.ndarray:
        want = self._in_types[component].get(name)
        if want is None or arr.dtype == want:
            return arr
        return arr.astype(want)

    def _run(self, component: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        cast = {k: self._cast(component, k, v) for k, v in feeds.items()}
        outs = self._s[component].run(None, cast)
        return [np.asarray(o) for o in outs]

    # ------------------------------------------------------------ voices #
    def build_voice(self, reference: np.ndarray) -> VoiceTensors:
        """24 kHz float32 mono reference clip -> voice tensors."""
        with self._enc_lock:
            out = self._run("speech_encoder", {"audio_values": reference[None].astype(np.float32)})
        cond, prompt, emb, feat = out
        return VoiceTensors(
            cond.astype(np.float32),
            prompt.astype(np.int64),
            emb.astype(np.float32),
            feat.astype(np.float32),
        )

    def _empty_kv(self) -> dict[str, np.ndarray]:
        return {n: np.zeros([1, NUM_KV_HEADS, 0, HEAD_DIM], self._kv_dtype) for n in self._kv_names}

    # ------------------------------------------------------------ LM step #
    def _lm_step(
        self, emb: np.ndarray, mask: np.ndarray, pos: np.ndarray, pkv: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """One language-model call -> (last-position logits float32, next pkv)."""
        sess = self._s["language_model"]
        emb = self._cast("language_model", "inputs_embeds", emb)
        if not self._iobinding:
            feeds = {"inputs_embeds": emb, "attention_mask": mask, "position_ids": pos}
            feeds.update({k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in pkv.items()})
            logits, *present = sess.run(None, feeds)
            return (
                np.asarray(logits)[0, -1, :].astype(np.float32),
                dict(zip(self._kv_names, present, strict=True)),
            )
        binding = sess.io_binding()
        binding.bind_cpu_input("inputs_embeds", np.ascontiguousarray(emb))
        binding.bind_cpu_input("attention_mask", np.ascontiguousarray(mask))
        binding.bind_cpu_input("position_ids", np.ascontiguousarray(pos))
        for name in self._kv_names:
            value = pkv[name]
            if isinstance(value, np.ndarray):
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
            else:
                binding.bind_ortvalue_input(name, value)
        binding.bind_output("logits", "cpu")
        for name in self._present_names:
            binding.bind_output(name, self._device_type, self.device_id)
        sess.run_with_iobinding(binding)
        outs = binding.get_outputs()
        logits = outs[0].numpy()
        nxt = dict(zip(self._kv_names, outs[1:], strict=True))
        return logits[0, -1, :].astype(np.float32), nxt

    def warm_thread(self) -> None:
        """One minimal LM step on the calling thread: builds that thread's CUDA
        provider context for the language model and the embedding session."""
        emb = self._run("embed_tokens", {"input_ids": np.array([[SILENCE]], np.int64)})[0]
        with self._lm_lock:
            self._lm_step(
                emb, np.ones((1, 1), np.int64), np.zeros((1, 1), np.int64), self._empty_kv()
            )

    def prefix_cache(self, voice: VoiceTensors) -> PrefixKV:
        """Read the voice's conditioning prefix once; keep its KV cache resident."""
        n = int(voice.cond_emb.shape[1])
        with self._lm_lock:
            _logits, present = self._lm_step(
                voice.cond_emb,
                np.ones((1, n), np.int64),
                np.arange(n, dtype=np.int64)[None],
                self._empty_kv(),
            )
        return PrefixKV(length=n, kv=present)

    def stream_tokens(
        self,
        text: str,
        voice: VoiceTensors,
        stop: threading.Event | None = None,
        seed: int | None = None,
        *,
        prefix: PrefixKV | None = None,
    ) -> Iterator[int]:
        """The LM loop as a generator of speech tokens (< START_SPEECH)."""
        text = normalise_text(text)
        if not text:
            return
        ids = np.array([self._tok.encode(text).ids], dtype=np.int64)
        words = len(text.split()) + 3 * text.count("[")
        max_new = min(1500, 25 + 20 * words)
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        n_out = 0
        t_prefill: float | None = None
        with self._lm_lock:
            text_emb = self._run("embed_tokens", {"input_ids": ids})[0]
        if prefix is not None and prefix.length == int(voice.cond_emb.shape[1]):
            emb = text_emb
            pkv: dict[str, Any] = dict(prefix.kv)
            start = prefix.length
        else:
            emb = np.concatenate([voice.cond_emb.astype(text_emb.dtype), text_emb], axis=1)
            pkv = dict(self._empty_kv())
            start = 0
        mask = np.ones((1, start + emb.shape[1]), np.int64)
        pos = np.arange(start, start + emb.shape[1], dtype=np.int64)[None]
        seen = np.zeros(STOP_SPEECH + 1, bool)
        seen[START_SPEECH] = True
        for _ in range(max_new):
            if stop is not None and stop.is_set():
                return
            # One step at a time under the lock (not the whole sentence): two
            # sessions' sentences interleave per token instead of queueing (S12).
            t_step = time.perf_counter()
            with self._lm_lock:
                row, pkv = self._lm_step(emb, mask, pos, pkv)
            METRICS.observe("tts_step_ms", (time.perf_counter() - t_step) * 1000)
            if t_prefill is None:
                t_prefill = time.perf_counter() - t0
            idx = np.nonzero(seen[: row.shape[0]])[0]
            s = row[idx]
            row[idx] = np.where(s < 0, s * self.repetition_penalty, s / self.repetition_penalty)
            nxt = self._sample(row, rng)
            if nxt == STOP_SPEECH:
                break
            if nxt < seen.shape[0]:
                seen[nxt] = True
            if nxt < START_SPEECH:
                n_out += 1
                yield nxt
            with self._lm_lock:
                emb = self._run("embed_tokens", {"input_ids": np.array([[nxt]], np.int64)})[0]
            mask = np.concatenate([mask, np.ones((1, 1), np.int64)], axis=1)
            pos = pos[:, -1:] + 1
        self.last_stats = {
            "lm_s": round(time.perf_counter() - t0, 3),
            "tokens": n_out,
            "prefill_s": round(t_prefill or 0.0, 3),
            "prefix": prefix is not None,
        }

    def _sample(self, row: np.ndarray, rng: np.random.Generator) -> int:
        if self.temperature <= 0:
            return int(np.argmax(row))
        row = row / self.temperature
        k = min(self.top_k, row.shape[0])
        idx = np.argpartition(row, -k)[-k:]
        logits = row[idx]
        order = np.argsort(logits)[::-1]
        idx, logits = idx[order], logits[order]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        keep = np.searchsorted(np.cumsum(p), self.top_p) + 1
        idx, p = idx[:keep], p[:keep] / p[:keep].sum()
        return int(rng.choice(idx, p=p))

    # ------------------------------------------------------------ decoder #
    def decode_chunk(
        self,
        tokens: Sequence[int],
        voice: VoiceTensors,
        *,
        left_context: int = 8,
        trailing_silence: int = 3,
        keep_silence: bool = False,
        context_tail: int = 0,
        right_context: int = 0,
    ) -> np.ndarray:
        """Decode speech tokens to float32 24 kHz (``voice`` already sliced).

        ``left_context`` leading tokens are re-decoded context whose audio is
        dropped except the last ``context_tail`` samples (the seam crossfade);
        ``right_context`` trailing tokens are lookahead whose audio is dropped;
        ``trailing_silence`` SILENCE tokens close the sequence."""
        toks = [int(t) for t in tokens]
        if not toks:
            return np.zeros(0, np.float32)
        left = max(0, min(int(left_context), len(toks)))
        right = max(0, min(int(right_context), len(toks) - left))
        trailing = max(0, int(trailing_silence))
        pad = bucket_padding(len(toks) + trailing, self.decode_buckets)
        seq = np.concatenate(
            [
                voice.prompt_token,
                np.array([toks], np.int64),
                np.full((1, trailing + pad), SILENCE, np.int64),
            ],
            axis=1,
        )
        with self._dec_lock:
            wav = self._run(
                "conditional_decoder",
                {
                    "speech_tokens": seq,
                    "speaker_embeddings": voice.speaker_embeddings,
                    "speaker_features": voice.speaker_features,
                },
            )[0].squeeze(0)
        wav = np.asarray(wav, np.float32)
        per = SAMPLES_PER_TOKEN
        start = max(0, left * per - max(0, int(context_tail))) if left else 0
        if keep_silence and not right:
            end = len(wav) - pad * per  # the bucket padding is never audio
        else:
            end = max(start, len(wav) - (trailing + pad + right) * per)
        return wav[start:end]

    def warm_decoder(self, voice: VoiceTensors) -> list[int]:
        """Decode one sequence per bucket so serving never meets a new shape."""
        done = []
        for size in self.decode_buckets:
            n = max(1, size - TRAILING_SILENCE_TOKENS)
            self.decode_chunk([SILENCE] * n, voice, left_context=0)
            done.append(size)
        return done

    def synthesize(
        self,
        text: str,
        voice: VoiceTensors,
        *,
        seed: int | None = None,
        stop: threading.Event | None = None,
    ) -> np.ndarray:
        """Whole sentence, full reference prompt (renders for caching)."""
        tokens = list(self.stream_tokens(text, voice, stop, seed))
        if stop is not None and stop.is_set():
            return np.zeros(0, np.float32)
        return self.decode_chunk(tokens, voice, left_context=0, keep_silence=True)


TRAILING_SILENCE_TOKENS = 3  # what the pipeline appends (pipeline.TRAILING_SILENCE)


def bucket_padding(length: int, buckets: Sequence[int]) -> int:
    """How many SILENCE tokens pad a decoder sequence of ``length`` speech tokens
    (context + emitted + lookahead + trailing silence) up to the next bucket.
    Beyond the largest bucket nothing is padded."""
    for b in buckets:
        if length <= b:
            return b - length
    return 0


def providers_for(provider: str, device_id: int, mem_limit_mb: int) -> list[Any]:
    """The ONNX Runtime provider list. CUDA options (S5, S22): the arena extends in
    power-of-two chunks, so the many distinct KV-cache and window sizes a
    conversation produces reuse the same blocks (measured: a fixed footprint
    after warm-up, where exact-size extension kept growing for 2.9 GB); capped;
    heuristic cuDNN algorithm search (exhaustive search re-tunes per new shape);
    copies on the default stream so IOBinding outputs are ready when read."""
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    opts = {
        "device_id": str(int(device_id)),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": str(int(mem_limit_mb) * 1024 * 1024),
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
    }
    return [("CUDAExecutionProvider", opts)]
