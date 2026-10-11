# Engines

The models behind SpeechForge sit behind two small adapter interfaces
(`src/speechforge/engines/base.py`, DECISIONS S10): one `SttAdapter` and one `TtsAdapter`,
chosen by name in `config.yaml` (`stt.engine`, `tts.engine`). Everything else -- the `sf.1`
session, the HTTP utilities, residency, leases, the voice store -- talks to those two objects
and never to a model library. This file documents each engine that ships: what it is, what it
costs, how its voices work, what was verified before it was integrated, and what remains open.

The shipped defaults are **Chatterbox-Turbo** (TTS) and **Parakeet TDT 0.6B v2** (STT), both
on ONNX Runtime (DECISIONS S4, SPEC §7.1). The phase-2 bake-off (SPEC §12) adds optional
PyTorch engines behind the same interface; they are off unless configured, and their
dependencies are extras that the default install never pulls.

Confidence tags as in the spec: **[P]** published by the model or runtime authors, **[V]**
vendor or third-party claim, **[M]** measured here, **[U]** unverified.

---

## Qwen3-TTS (`tts.engine: qwen3-tts`) -- phase-2 candidate

**One line.** Alibaba's Qwen3-TTS-12Hz-1.7B-Base (or 0.6B-Base), weights Apache-2.0, run with
`faster-qwen3-tts` (MIT; PyTorch CUDA graphs, chunked streaming) on the leased card. It
clones a voice *in context* from a reference clip and its transcript, which puts a recording of
a person on the rig. **That is decision D3**: the adapter refuses to load unless
`voices.allow_reference_clip: true`, and the owner choosing this engine *is* the D3 decision.

### Status

Implemented and unit-tested with a fake backend on a CPU (`tests/test_qwen3_tts.py`). **Not
yet run on a GPU**: the lead runs `scripts/gpu_smoke_tts.py` (below) on the host. Every number
in this section is [P] or [V] until that happens.

### Research findings (web, accessed 2026-10-11)

**The model.**

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, pinned at revision `fd4b254389122332181a7c3db7f27e918eec64e3`
  (last modified 2026-01-23) [P]. Files: `model.safetensors` 3,857,413,744 bytes,
  `speech_tokenizer/model.safetensors` 682,293,092 bytes, plus config/tokenizer JSON and the
  BPE `vocab.json`/`merges.txt`; **no pickle-format weights of any kind** [P]
  (https://huggingface.co/api/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/fd4b254389122332181a7c3db7f27e918eec64e3?recursive=true&expand=true).
  SPEC §8.5 ("PyTorch engines use safetensors only") is satisfied by the files themselves, and
  the loader additionally refuses a model directory that contains any `.bin/.pt/.pth/.ckpt/.pkl`.
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, revision `5d83992436eae1d760afd27aff78a71d676296fc`
  (2026-01-29): `model.safetensors` 1,829,344,272 bytes; the speech tokenizer is byte-identical
  to the 1.7B one (same sha256) [P].
- Output is 24 kHz (`speech_tokenizer/config.json`: `output_sample_rate: 24000`,
  `decode_upsample_rate: 1920` = 12.5 codec steps per second, the "12Hz" of the name) [P]. The
  speaker encoder also takes 24 kHz (`speaker_encoder_config.sample_rate`) [P]. This matches
  the server's 24 kHz / 40 ms frame format, so no resampling is needed on the way out.
- Languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
  Italian [P] (model card). The model takes a `language` string per utterance (`"English"`,
  ..., or `"Auto"`); `tts.qwen3.language` sets it.
- Voice cloning inputs [P] (https://github.com/QwenLM/Qwen3-TTS, "Base" usage):
  `create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)` where `ref_audio`
  may be a path, URL, base64 or an `(np.ndarray, sample_rate)` tuple, and `ref_text` is the
  clip's transcript. Default is **ICL mode** (the reference codes and transcript go into the
  model's context; `ref_text` required). `x_vector_only_mode=True` uses only the speaker
  embedding and needs no transcript, "but cloning quality may be reduced" [P]. The model card
  recommends a 3-second clip [P]; this adapter accepts 1-20 s.
- Streaming [P/V]: the model card claims a "dual-track hybrid streaming" design with end-to-end
  first-packet latency "as low as 97 ms" [V, vendor measurement, hardware unstated]. The
  official `qwen-tts` package exposes **no** streaming generator (0.1.1, 2026-02-06); streaming
  in Python comes from `faster-qwen3-tts`.

**Licence verdict.**

- The model cards for both repos declare `license: apache-2.0` in their metadata and the README
  front matter [P] (`https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/raw/fd4b254389122332181a7c3db7f27e918eec64e3/README.md`,
  first lines: `license: apache-2.0`). **The Hugging Face repos carry no `LICENSE` file**
  (`.../raw/main/LICENSE` returns "Entry not found", checked 2026-10-11). The code repository
  https://github.com/QwenLM/Qwen3-TTS carries the full Apache License 2.0 text
  (https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/LICENSE; GitHub reports
  `spdx_id: Apache-2.0`) [P]. Verdict: **Apache-2.0, permitted** -- with the note that the
  weights' licence is asserted by the card metadata rather than by a licence file in the weights
  repository. `THIRD_PARTY_NOTICES.md` carries the attribution.
- `faster-qwen3-tts`: MIT (`pyproject.toml` `license = "MIT"`, GitHub `spdx_id: MIT`) [P].
- `qwen-tts-hf` (the `qwen_tts` package `faster-qwen3-tts` depends on; see below): Apache-2.0
  (`License-Expression: Apache-2.0` in the wheel metadata) [P].

**Which Python package.**

| | official `qwen-tts` 0.1.1 (2026-02-06) | `faster-qwen3-tts` 0.5.4 (2026-10-05) |
|---|---|---|
| Streaming | none in the Python API | `generate_voice_clone_streaming(text, language, ref_audio, ref_text, chunk_size=12, ..., voice_clone_prompt=...)` yields `(float32 audio, sr, timing)` per `chunk_size` codec steps [P] |
| Speed | plain HF `generate` | CUDA graphs for the talker and the code predictor; "6-10x" [V] |
| Transformers | pins `transformers==4.57.3` | `transformers>=5.15.1,<6` via `qwen-tts-hf` (an Apache-2.0 compatibility build of the same `qwen_tts` package for Transformers 5, built from QwenLM/Qwen3-TTS PR #360) [P] |
| Voice prompt reuse | `create_voice_clone_prompt` -> `voice_clone_prompt=` | same, accepted by both streaming and non-streaming calls; the adapter caches one prompt per voice id (LRU 8) |
| Windows | no notes | `setup_windows.bat`, `WINDOWS_SETUP_GUIDE.md`; RTX 4060 on Windows in the README's table [P] |
| Measured TTFA (vendor table, `chunk_size=8`) | -- | RTX 4090: 0.6B 156 ms / 1.7B 174 ms, RTF 4.8 / 4.2; RTX 4060 (Windows): 0.6B 413 ms, RTF 2.26; T4: 901 ms [V] |

Chosen: **`faster-qwen3-tts`**. It is the only maintained Python path with streaming, it is MIT,
and its Windows table shows the Torch path working on Windows [V]. Its README says not to
install the upstream `qwen-tts` in the same environment (both provide `qwen_tts`) [P]; the
`tts-torch` extra does not.

Dependency hazards found in the `qwen-tts-hf` wheel (inspected 2026-10-11) [M]:

- `import qwen_tts` imports its 25 Hz tokenizer package eagerly, which imports **`torchaudio`,
  `onnxruntime` and `sox`** at module import even though the 12 Hz models never use them.
  Consequences: (1) `torchaudio` must match `torch` -- hence the matched pin below; (2) the
  CPU `onnxruntime` wheel it requires would unpack into the same `onnxruntime/` directory as
  `onnxruntime-gpu` from the `gpu` extra and could remove the CUDA provider -- hence the
  `override-dependencies` entry in `pyproject.toml` that turns every requirement on
  `onnxruntime` into "the CPU wheel, only with the `cpu` extra" (uv 0.7 has no per-package
  override, so the project's own `cpu` extra is routed through the same rule; `uv.lock` shows
  the `cpu` extra keeping onnxruntime and `qwen-tts-hf` losing the edge; `onnxruntime-gpu`
  satisfies `import onnxruntime`); (3) `sox` is the pysox binding, which
  imports without the SoX binary (it logs a warning) [U -- verify on the host: `python -c
  "import qwen_tts"`]. `gradio` (demo only) is dropped the same way so it cannot pin
  fastapi/starlette for the server.
- `accelerate==1.12.0` is pinned by `qwen-tts-hf`; harmless.

**PyTorch on native Windows, CUDA 13.0, RTX 3090 (sm_86) and RTX 5090 (sm_120).**

- PyTorch publishes CUDA 13.0 wheels at `https://download.pytorch.org/whl/cu130`; for Windows
  `cp312` the index holds `torch` 2.9.0 ... 2.14.1 `+cu130` win_amd64 wheels, and `torchaudio`
  2.9.0 ... **2.11.0** `+cu130` (torchaudio's last release is 2.11.0, 2026-03-23) [P]
  (index listing, 2026-10-11). PyPI's Windows `torch` wheels are CPU-only, so the index is
  required, and `pyproject.toml` declares it as an `explicit` uv index used only by `torch` and
  `torchaudio` when the `tts-torch` extra is selected (the default install and the `gpu` extra
  never touch it).
- From PyTorch 2.11 (2026-01) CUDA 13 is the default CUDA build; the 2.11 release notes [P]
  (https://pytorch.org/blog/pytorch-2-11-release/). Blackwell (sm_120) has native kernels in
  PyTorch's cu128 builds since 2.7 (faster-qwen3-tts's README: "RTX 50xx / Blackwell GPUs need
  CUDA 12.8 PyTorch wheels (PyTorch 2.7+)") [V]; the cu130 builds are a superset in arch terms
  and a Windows user reports `torch 2.11.0+cu130` working on a 5090 [V, blog]. sm_86 has been
  in every CUDA 11.1+ build. **Verify on the host** with
  `python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"` -- the list
  must contain `sm_86` and `sm_120`; a wheel without the card's arch fails only at the first
  kernel launch, not at `torch.cuda.is_available()`.
- Pinned pair: **`torch==2.11.0` + `torchaudio==2.11.0`** (`+cu130`, resolved by `uv lock`). It
  is the newest pair with cu130 win_amd64 wheels for *both* packages; a newer torch with an
  older torchaudio risks a torchaudio import failure (see hazards). The ComfyUI venv on the same
  rig runs a newer torch; the two are separate environments and do not interact.
- `faster-qwen3-tts` requires `torch>=2.5.1` ("CUDA-graph capture is unreliable on
  torch<=2.5.0") [P]. Attention: `sdpa` (default here) needs nothing extra; `flash_attention_2`
  needs a flash-attn wheel built for the exact torch/CUDA/Python triple (third-party Windows
  builds exist, e.g. for `cu130torch2.10-cp312-win_amd64` [V]) -- not pinned, opt-in via
  `tts.qwen3.attn`.
- Feasibility verdict: **native Windows + cu130 on the 3090 is expected to work** (torch and
  faster-qwen3-tts both document Windows; sm_86 is universally built) [V]; **the 5090 should
  work** with the same wheels (sm_120 built into cu130 wheels) [V/U until measured]. The known
  Windows-specific risk is CUDA-graph capture under WDDM, which faster-qwen3-tts's RTX 4060
  Windows row suggests is fine [V]. VRAM [E]: 1.7B bf16 ~3.6 GB weights + 0.7 GB tokenizer +
  static KV cache for `max_seq_len` 2048 + CUDA graphs; budget 6-7 GB, within the SPEC §7.2
  bake-off allowance on the 24 GB card.

### How it is wired

- **Voice format `qwen3tts-ref`** (`engines/qwen3_tts.py::parse_voice`): an npz with
  `audio` (int16 or float32 mono, 1-20 s, 8-48 kHz), `rate` (integer), `transcript` (UTF-8,
  <= 1000 chars; required key, may be empty), optional `format`/`version`. Loaded with
  `allow_pickle=False`, zip-size-checked, dtype/shape/range/finite-checked. The store is
  unchanged: the id is the sha256 of the bytes, `PUT /v1/voices/<sha256>` as before. A 20 s
  int16 clip at 24 kHz is ~0.96 MB, under the 2 MB upload cap. An **empty transcript selects
  x-vector-only cloning** (no text needed, lower similarity); that is what the self-test and
  `POST /v1/voices/from-reference` produce, since neither has a transcript. Build one with
  `scripts/gpu_smoke_tts.py --build-voice clip.wav --transcript "..." --voice out.npz`.
- **Streaming**: `stream()` pulls `faster-qwen3-tts`'s generator chunk by chunk, converts to
  int16 24 kHz, checks `stop` before every chunk and closes the generator on stop or when the
  consumer leaves. The credit gate (`options["gate"]`) is consulted before each chunk with the
  audio emitted so far in 40 ms units (the server counts Chatterbox tokens; this engine's
  codec steps are 83 ms). `first_chunk_tokens` (40 ms tokens) is mapped to codec steps and sets
  the chunk size for that utterance (15 tokens -> 7 steps ~ 0.58 s), since the backend has one
  chunk size per call. `on_lm_done` fires at the end with the length in 40 ms units, so
  `tts.words` arrives after the audio for this engine (estimates, as with Chatterbox).
- **Concurrency**: faster-qwen3-tts holds its KV cache in static buffers captured into CUDA
  graphs, so **one utterance at a time** per process (`Qwen3Tts._gen_lock`). Two sessions'
  sentences queue whole, not per step as Chatterbox's do (S12); a credit stall inside a
  sentence holds the engine for its duration. Acceptable for the bake-off; a per-step
  interleave would need the library to expose its decode step.
- **`last_stats`**: `ttfa_ms` (= `first_chunk_ms`), `prefill_ms`, `tokens` (codec steps),
  `chunks`, `audio_ms`, `total_ms`, `rtf` (wall / audio), `chunk_size`, `finished`.
- **`warm_up()`** captures the CUDA graphs and speaks one line with the synthetic (non-human)
  reference; **`describe()`** reports model id, revision, dtype, device and the backend's
  torch/CUDA/GPU facts; **`close()`** drops the model and empties the CUDA cache.
- **models.lock**: groups `tts-qwen3` (1.7B) and `tts-qwen3-0.6b`, dests `qwen3-tts-1.7b/...`
  and `qwen3-tts-0.6b/...`; `models.tts_group_for(config)` picks the group from
  `tts.engine` + `tts.qwen3.model`, so `speechforge fetch-models` / `verify-models` and the
  load-time verification work unchanged. Hashes: LFS sha256 from the tree API
  (`expand=true`); the small JSON/BPE files were downloaded and hashed (they have no LFS entry).
- **Config** (`tts.qwen3`, `config.example.yaml`): `model` 1.7b|0.6b, `dtype` bf16|fp16,
  `attn` sdpa|flash_attention_2, `language`, `chunk_size` (8), `max_seq_len` (2048),
  `max_new_tokens` (1024 ~ 85 s), sampling (`temperature` 0.9, `top_k` 50, `top_p` 1.0,
  `repetition_penalty` 1.05 -- the library's defaults), `xvec_only`, `append_silence`. Unknown
  keys are errors, as everywhere.
- **No network while serving**: the backend sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  (unless already set) and loads from the verified local directory only.

### Install and verify on the GPU host

```
uv sync --extra gpu --extra tts-torch                # adds torch 2.11.0+cu130, faster-qwen3-tts
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
uv run python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"   # CUDA EP must still be there
uv run python -c "import qwen_tts, faster_qwen3_tts; print(faster_qwen3_tts.__version__)"
# config.yaml: tts.engine: qwen3-tts, voices.allow_reference_clip: true  -> then
uv run speechforge fetch-models                       # ~4.5 GB for 1.7B
uv run python scripts/gpu_smoke_tts.py --engine qwen3-tts --device 3 --models-dir <data>/models \
    --build-voice <clip.wav> --transcript "<what the clip says>" --voice qwen-voice.npz \
    --text "Sure, I can help with that." --out qwen.wav
```

The smoke script prints load time, warm-up time, per-utterance TTFA / RTF / codec steps and
writes the WAV; `--qwen3-model 0.6b`, `--qwen3-chunk-size 4`, `--first-chunk-tokens 15` are the
knobs to sweep. Compare with `--engine chatterbox --voice <add-on npz>` on the same card.

### Open questions

1. TTFA and RTF on the 3090 and the 5090 under WDDM (nothing published for either card).
2. Whether `import qwen_tts` succeeds on the host without a SoX binary (pysox import) [U].
3. CUDA-graph capture with `max_seq_len` 2048 and a 20 s ICL reference (~250 codec steps +
   text): the library raises "Input is too long" past `max_seq_len`; 2048 leaves ~1700 steps
   for text and audio, i.e. ~2 minutes -- adequate, but the self-test's long line should
   confirm it.
4. Perceived quality of x-vector-only voices (the self-test path) versus ICL with a transcript;
   the family blind test (SPEC §12 phase 2) decides, and D2 decides whether a voice that does
   not match the CPU fallback is acceptable.
5. The weights repository has no LICENSE file; the licence is the card's metadata plus the code
   repository's Apache-2.0 text. Fine for use; note it if the engine is ever recommended in
   public documentation.

---

## dots-tts (dots.tts-mf) -- TTS candidate, phase 2

Adapter: `src/speechforge/engines/dots_tts.py`; name `dots-tts`; voice format `dotstts-ref`;
config `tts.engine: dots-tts` + `tts.dots.*`; model groups `tts-dots` / `tts-dots-stts`;
tests `tests/test_dots_tts.py`; GPU smoke `scripts/gpu_smoke_dots_tts.py`.

### What it is

dots.tts is a 2B-parameter *fully continuous* autoregressive TTS model from RedNote's hi-lab:
a semantic text encoder, a Qwen2.5-1.5B-sized LLM and an autoregressive flow-matching acoustic
head over a 48 kHz AudioVAE with a BigVGAN-style causal vocoder; there are no discrete speech
tokens anywhere **[V]**. `dots.tts-mf` is the MeanFlow-distilled student of `dots.tts-soar`
(the highest-similarity checkpoint), tuned for 4 solver steps and low latency **[V]**.
Technical report: arXiv 2606.07080 **[V]**.

### Research findings (accessed 2026-10-11)

| Item | Finding | Tag | Source |
| --- | --- | --- | --- |
| Hugging Face repo id | `dots-studio/dots.tts-mf`. The id in SPEC §7.1, `rednote-hilab/dots.tts-mf`, now **redirects** to it (the API answers with `id: dots-studio/dots.tts-mf`). The GitHub repo moved the same way: `rednote-hilab/dots.tts` -> `studio-dots-ai/dots.tts`. | [P] | https://huggingface.co/api/models/rednote-hilab/dots.tts-mf ; https://api.github.com/repos/rednote-hilab/dots.tts |
| Pinned revision | `c28105adc8228143392b4e346994ff613ee48a06` (last modified 2026-08-11, the current head of `main` on 2026-10-11) | [P] | https://huggingface.co/api/models/dots-studio/dots.tts-mf/revision/c28105adc8228143392b4e346994ff613ee48a06 |
| Double-streaming artifact | `dots-studio/dots.tts-mf-2steps-stts` @ `9e7be6817ceda3fd6474af4d089ca6227918ccec` (2026-08-11); it ships its own `SHA256SUMS`, which match the LFS hashes | [P] | https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts |
| Code licence | Apache-2.0; `LICENSE` reads "Copyright 2026 dots.tts Team, RedNote" | [P] | https://github.com/studio-dots-ai/dots.tts/blob/main/LICENSE |
| Weights licence | Apache-2.0: the model card's `license: apache-2.0` and the README's "dots.tts code and released checkpoints are licensed under Apache-2.0" | [P] card metadata / [V] README sentence | https://huggingface.co/dots-studio/dots.tts-mf ; https://github.com/studio-dots-ai/dots.tts#-license |
| Weights format | `model.safetensors` (4.40 GB), `vocoder.safetensors` (724 MB), `speaker_encoder.safetensors` (29 MB), all loaded with `safetensors.torch.load_file`. **One exception:** `latent_stats.pt` (3.2 KB) is a torch zip-pickle holding a dict of two numpy `(1, 128)` float32 arrays (`mean`, `var`) and is read with `torch.load(weights_only=False)` in `models/dots_tts/core.py` (`IOHelper.__init__`). | [P] | https://github.com/studio-dots-ai/dots.tts/blob/main/src/dots_tts/models/dots_tts/model.py ; `core.py` |
| Package | PyPI `dots.tts` 0.3.1 (2026-08-12), pure-Python wheel, `requires-python >=3.10,<3.13`; classifiers list **Linux and macOS only**; deps `torch>=2.8`, `torchaudio>=2.8`, `transformers>=4.57`, `safetensors>=0.8.0rc0`, `librosa`, `soundfile`, `torchdiffeq`, `einops`, `loguru`, `langcodes[data]`, `lingua-language-detector`, `WeTextProcessing`, `gradio` (the demo; unused by the runtime), `numpy>=2.2.6`, `pydantic`, `PyYAML`, `tqdm`, `huggingface-hub`. No flash-attn or Triton requirement. `constraints/recommended.txt` pins torch 2.8.0. | [P] | https://pypi.org/pypi/dots.tts/json ; https://github.com/studio-dots-ai/dots.tts/blob/main/pyproject.toml |
| Windows, native | **Not supported upstream** (classifiers, no Windows note anywhere in the README). The one hard blocker found is `WeTextProcessing` -> `pynini>=2.1.6`: PyPI `pynini` 2.1.7 ships **manylinux wheels only** and builds from source otherwise; `dots_tts/utils/text.py` imports `tn.chinese.normalizer` at import time and `runtime.py` imports that module, so the runtime cannot import without it. The normalisers are only *used* when `normalize_text=True`. Everything else has Windows wheels: official `torch` cu130 `cp312 win_amd64` wheels exist for 2.9.0-2.14.1 and `torchaudio` cu130 for 2.9-2.11 on download.pytorch.org. | [P] | https://pypi.org/pypi/pynini/json ; https://download.pytorch.org/whl/cu130/torch/ ; https://download.pytorch.org/whl/cu130/torchaudio/ |
| sm_86 / sm_120 | PyTorch's official wheels built with CUDA 12.8+ carry Blackwell (sm_120) kernels since 2.7 (PyTorch staff on the forum); cu130 wheels therefore should as well. Not measured here. | [P] for cu128 / [U] for cu130 | https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099 |
| `optimize=True` | `torch.compile` of the LLM/DiT/vocoder steps with a ~3 min warm-up on an H800 (`warmup_on_optimize`); needs Triton, which has no official Windows build (community `triton-windows` wheels exist). Default **off** here; eager mode is the Windows baseline. | [V] | README "Efficiency" |
| Audio out, streaming | `DotsTtsRuntime.generate_stream()` yields `torch.Tensor (1, samples)` **per latent patch** as it is decoded: one patch = `patch_size` 4 x hop 1920 = 7680 samples = **160 ms at 48 kHz**; after the first few patches the vocoder merges `vocoder_merge_steps` (4) patches per chunk. Output sample rate 48 000 (`config.json: vocoder.sample_rate`). | [P] | `runtime.py`, `config.json` |
| Text in, incremental | The plain runtime takes the **complete** text of an utterance per call. Incremental text ("double streaming") exists in `DotsTtsRuntimeDoubleStreaming` on the `-2steps-stts` artifact: `start_double_streaming(prompt_audio_path, prompt_text)` -> `session.push_text_token(id)` returns an audio chunk or `None`; `session.finish_text()` flushes the tail. SGLang Omni's server does **not** expose same-request incremental text (its WebSocket buffers and starts a request per clause). | [P] code / [V] Omni note | `runtime_double_streaming.py`; README "SGLang Omni Usage" |
| Voice cloning | Reference clip **plus its exact transcript** = "continuation cloning", best speaker similarity; clip alone = x-vector-only cloning (weaker); no clip = random voice. About 10 s of natural speech recommended, longer does not help; the speaker encoder reads at most `xvec_max_audio_seconds` 10 s. The runtime API takes a **file path** (`prompt_audio_path`, loaded with librosa, trimmed at 30 dB, resampled to 48 kHz). | [V] README / [P] `config.json`, `runtime._load_prompt_audio` | README "Voice cloning" |
| Seeds | No per-request seed in the Python runtime; the CLI calls `seed_everything()` (global `torch.manual_seed`). The Omni server accepts a top-level `seed`. | [P] | `cli.py`, cookbook |
| Latency / RTF | Vendor, single H800, torch 2.8 + CUDA 12.8, `--optimize`, bf16: `dots.tts-mf` first chunk **p50 204 ms** (voice cloning) / 68 ms (text only), RTF 0.15 / 0.13. | [V] | README "Steady-State Latency" |
| VRAM | Vendor peak allocation, same setup: 5.65 GB (< 10 s of audio), 6.53 GB (< 20 s), 7.86 GB (< 40 s); SPEC §7.1's 5.7-7.9 GB is this table. On a 3090/5090 under eager mode: **[U]**, the smoke script prints it. | [V] | README "Peak memory" |
| Server path | SGLang Omni serves `POST /v1/audio/speech` (Linux; CUDA graphs, continuous batching). Request: `model`, `input`, `references: [{audio_path, text}]` (exactly one; `audio_path` may be a local path, file/data/HTTP URL), `response_format` (`wav`|`mp3`|`flac`|`opus`|`aac`|`pcm`), `stream`, `seed`, `language`, `stage_params.latent_engine.{speaker_scale, eos_threshold, num_steps, ode_method, max_generate_length, normalize_text}`. `stream: true` + `pcm` = raw **s16le mono 48 kHz**, chunked. There is no `voice` preset field; unknown top-level keys are dropped silently. | [V] | https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html |
| Other runtimes | `audio.cpp` (ggml; CPU/CUDA/Vulkan/Metal, CLI and server, no Python) lists dots.tts support -- a possible Windows-native path not pursued here. vLLM: not mentioned by upstream. | [V] | README "Ecosystem" |

**Licence verdict: PASS.** Code and weights are Apache-2.0 (above). Attribution is in
`THIRD_PARTY_NOTICES.md`; the README's acceptable-use note (consent for reference voices, mark
synthetic audio) is repeated there.

**Windows-native verdict: [U], plausible, lead to confirm.** Nothing in the runtime is
Linux-specific; the install is. The expected recipe is `pip install dots.tts==0.3.1 --no-deps`
plus its dependencies *except* WeTextProcessing, on the cu130 torch the `tts-torch` extra
installs; the adapter then shims the missing `tn` package (`allow_missing_text_normalizer`,
default true) so the runtime imports with text normalisation unavailable (`normalize_text`
must stay false; SpeechForge sends clause-sized plain text anyway). If that fails on the rig,
the `http` backend below is the fallback (an SGLang Omni server in WSL2 on the same machine,
loopback only). This agent did not install WSL or touch a GPU.

### How the adapter maps dots.tts onto `TtsAdapter`

* **Decision D3 gate.** The engine needs the reference clip and its transcript on the rig at
  synthesis time, which SPEC §8.4 and DECISIONS D3 keep off the rig by default. `load()` refuses
  with a clear message unless `voices.allow_reference_clip: true`. Setting it is the owner's
  decision; choosing `dots-tts` *is* taking D3's "yes". The clip never reaches the disk on the
  native backend: the runtime's private `_load_prompt_audio` is pointed at the in-memory tensor
  (keyed `sf-voice://<voice id>`). On the `http` backend the clip travels as a `data:` URL inside
  the loopback request; the Omni server may cache it -- that is the owner's WSL2 instance.
* **Voice format `dotstts-ref`** (`PUT /v1/voices/<sha256>` with `.npz`, `allow_pickle=False`):
  `audio` mono int16 (or float32 in [-1, 1]), `rate` (8-96 kHz), optional `transcript`
  (<= 1000 chars, no control characters), optional `format: "dotstts-ref"`. Validated: byte
  size (<= 8 MB raw, <= 16 MB inflated), duration 0.5-20 s, dtype/shape, finiteness, range.
  `make_voice(audio, rate, transcript)` writes one. `POST /v1/voices/from-reference` (D3-gated,
  no transcript in the request) builds an x-vector-only voice; clients that have the transcript
  should build the file themselves. Note `voices.max_bytes` (2 MB default) applies at the HTTP
  layer: 20 s of int16 at 48 kHz is 1.9 MB.
* **`sample_rate`.** The model speaks at 48 kHz (`describe().native_sample_rate`); the adapter's
  `sample_rate` is `tts.dots.output_rate`, **24 000 by default** so the sf.1 frame math
  (`tts_ctx.RATE`, 40 ms = 960 samples) and the CPU-fallback timbre comparison stay unchanged.
  The 2:1 decimation is a streaming 63-tap Kaiser half-band FIR (`Decimator`; piecewise output
  equals whole-sentence output, 20 kHz content is -40 dB+ before folding). `48000` passes the
  model's audio through untouched (then the realtime path's 24 kHz constants would need work).
* **`stream()`** yields int16 chunks as the model produces patches. The GPU lock is held **per
  patch** (`next()` on the runtime's generator), so two contexts interleave patch by patch
  (S12's shape). `stop` is checked before every patch and the runtime generator is closed on
  stop. `seed` -> `torch.manual_seed` once per stream (global RNG: two interleaved streams are
  not individually reproducible). `options["gate"]` is the sf.1 credit gate, called with the
  count of **40 ms credit tokens** produced so far (`tts_ctx` does `tokens x 40 ms`; one patch
  = 4), polled every 5 ms while closed; `options["on_lm_done"]` fires once at the end with the
  same count (there is no separate LM phase, so `tts.words` estimates arrive with the last
  audio). `last_stats`: `ttfa_ms`, `audio_ms`, `rtf`, `chunks`, `tokens`, `total_ms`, `stopped`.
* **Incremental text:** `options["text_feed"]` = an iterator of further text pieces (the
  `text` argument is the first piece; exhaustion ends the utterance). With
  `tts.dots.double_streaming: true` (native backend, `tts-dots-stts` group) the adapter opens one
  double-streaming session, tokenises each piece with the model's tokenizer and pushes the ids;
  audio comes back as the model decides it has enough text. `describe().text_stream` says
  whether the backend supports it; without a session-capable backend (http, or
  `double_streaming` off) the pieces are buffered and spoken as one unit. No current caller
  feeds text this way (tts_ctx cuts clause units); it is the hook for SPEC §5.4's
  `caps.text_stream`.
* **`render()`** concatenates the stream; **`warm_up()`** speaks one line from the synthetic
  reference (x-vector-only); **`close()`** drops the runtime and empties the CUDA cache.
* **Device.** The runtime binds to `torch.device("cuda")` without an index, so every call runs
  under `torch.cuda.device(device_id)`; the ONNX engines and this one therefore agree on the
  PCI-bus ordinal from `gpu.devices`.
* **`latent_stats.pt` (SPEC §8.5).** The one non-safetensors file is sha256-pinned in
  `models.lock` *and* scanned before load by a restricted unpickler (`check_latent_stats`): every
  global in the pickle must be one of the numpy/torch array-rebuild names, each resolving to an
  inert stub, or the load is refused. No library code runs on the file's say-so.

### Configuration (`tts.dots`, defaults in `config.example.yaml`)

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `native` | `native`: `dots_tts` in-process on the leased card. `http`: SGLang Omni `/v1/audio/speech` over loopback. |
| `url` | `http://127.0.0.1:8000` | http backend; **must be loopback** (validated). |
| `http_model`, `http_timeout_s` | `""`, 30 | request `model` (default the pinned repo id); per-request timeout. |
| `precision` | `bf16` | `bf16` / `fp16` / `fp32` (fp32 sets `matmul_precision high`). |
| `num_steps` | 4 | MeanFlow solver steps; `null` = the artifact default (**required** for `-2steps`/`-stts`, which lock their sampling). |
| `guidance_scale` | 1.2 | No effect on `dots.tts-mf` (CFG is fused); `null` for fixed-step artifacts. |
| `speaker_scale`, `eos_threshold` | 1.5, 0.8 | Upstream defaults. |
| `optimize`, `warmup_on_optimize` | false, true | `torch.compile` path (Triton). |
| `max_generate_length` | 500 | Longest utterance in 160 ms patches (80 s). |
| `output_rate` | 24000 | 24000 (decimated, sf.1) or 48000 (native). |
| `double_streaming` | false | Incremental text on `dots.tts-mf-2steps-stts` (`tts-dots-stts` group), native only. |
| `language` | null | `null`, `auto_detect`, or a language code passed to the runtime. |
| `normalize_text` | false | Upstream WeTextProcessing normalisation (needs the real package). |
| `allow_missing_text_normalizer` | true | Shim a missing `tn` package so the runtime imports on Windows. |

Model files: group `tts-dots` = the 12 files of `dots.tts-mf` (4.44 GB) into
`<models>/dots-tts-mf/`; `tts-dots-stts` = the same layout of the stts artifact into
`<models>/dots-tts-mf-2steps-stts/`. `speechforge fetch-models` / `verify-models` follow
`tts.engine` (`models.tts_groups`); the `http` backend needs no local files.

### Dependencies

The adapter shares the `tts-torch` extra (torch + torchaudio, cu130) that the Qwen3-TTS branch
adds; on top of it the dots.tts stack is:

```
dots.tts==0.3.1  (install with --no-deps on Windows: its WeTextProcessing dependency has no wheel)
transformers>=4.57  safetensors>=0.8.0rc0  librosa>=0.11  soundfile>=0.13.1  torchdiffeq  einops
loguru  langcodes[data]  lingua-language-detector  tqdm  huggingface-hub  numpy>=2.2.6
```

(`gradio`, `WeTextProcessing` are declared by the package but not needed by the runtime;
`pydantic`, `PyYAML`, `huggingface-hub` are already in SpeechForge's environment.)

### GPU verification (the lead runs this; nothing here was run on a GPU)

```
# 1. environment (rig, after `uv sync --extra gpu --extra tts-torch`)
uv pip install --python .venv "dots.tts==0.3.1" --no-deps
uv pip install --python .venv "transformers>=4.57" "safetensors>=0.8.0rc0" "librosa>=0.11" \
    "soundfile>=0.13.1" torchdiffeq einops loguru "langcodes[data]" lingua-language-detector tqdm
uv run python -c "import torch, dots_tts; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"

# 2. model files (4.44 GB) with tts.engine: dots-tts in config.yaml (or a scratch data dir)
uv run speechforge fetch-models && uv run speechforge verify-models

# 3. smoke: eager bf16 on the 3090 (CUDA 3), then the 5090 (CUDA 0/1), then --optimize
uv run python scripts/gpu_smoke_dots_tts.py --device 3 --voice ref.wav --transcript "<exact words>" \
    --text "Sure, I can help with that." --out dots-3090.wav --runs 3
uv run python scripts/gpu_smoke_dots_tts.py --device 0 --voice ref.wav --transcript "<exact words>" \
    --text "Sure, I can help with that." --out dots-5090.wav --runs 3
uv run python scripts/gpu_smoke_dots_tts.py --device 3 --voice ref.wav --transcript "<exact words>" \
    --text "Sure, I can help with that." --out dots-opt.wav --optimize --runs 3   # needs Triton
```

The script prints per run: `ttfa` (adapter, first decimated chunk), `audio`, `total`, `rtf`,
`chunks`, `peak vram`. Compare with the phase-1 gate (TTFA <= 250 ms on CUDA) and the vendor
table above. If `import dots_tts` fails on Windows, note the missing module: that is the
Windows-native verdict. Fallback: SGLang Omni in WSL2 (`sgl-omni serve --config
examples/configs/dots_tts.yaml --port 8000`, loopback) and `tts.dots.backend: http`.

### Open questions

1. Does `dots_tts` import and run on Windows with cu130 torch + the `tn` shim (the only known
   blocker is pynini)? `lingua-language-detector` and `librosa`/`numba` wheels for 3.12 on
   Windows are expected but unverified here.
2. Eager-mode RTF/TTFA on sm_86 and sm_120 vs the vendor's compiled H800 numbers; whether
   `--optimize` (Triton on Windows) is needed to beat Chatterbox's TTFA.
3. The runtime takes `numpy>=2.2.6` while SpeechForge pins `numpy>=1.26`; check the resolved
   version does not break the ONNX stack.
4. The `http` backend's `references[0].text` with an empty transcript (x-vector-only) is
   untested against a live Omni server; Omni may require the transcript.
5. Private hooks (`_load_prompt_audio`, `model.tokenizer`) were read at dots.tts 0.3.1; a
   package upgrade needs a re-read.
6. Whether the owner takes D3's "yes" at all (SPEC §14): without it this engine cannot load.

---

## Nemotron Speech Streaming EN 0.6B (STT, `stt.engine: nemotron`)

**Status (2026-10-11):** implemented and unit-tested on CPU fakes; **not yet run with the real
model on any machine** (the agent that wrote it had no GPU and downloaded no weights). The
first GPU run is the operator's (commands at the end). The engine is **off by default**
(`stt.engine: parakeet`), and its weights are under the **NVIDIA Open Model License**, which the
owner approves personally before it is enabled (see *Licence*).

### What it is

NVIDIA's cache-aware streaming FastConformer (24 layers, 600 M parameters) with an RNN-T
decoder, English, punctuation and capitalisation [V]. Each 80 ms frame of audio is encoded
once against a kept cache (no overlapping re-computation), in chunks whose size is chosen at
export time: `att_context_size` `[70, 0]` = 80 ms, `[70, 1]` = 160 ms, `[70, 6]` = 560 ms,
`[70, 13]` = 1120 ms (the NeMo default) [V]. Published WER (average over the Open ASR
datasets, no punctuation/case): 6.93 % at 1.12 s, 7.07 % at 0.56 s, **7.67 % at 0.16 s**, 8.43 %
at 0.08 s [V] -- versus Parakeet TDT 0.6B v2's 6.43 % on the leaderboard (SPEC §0 row 3 [P]). So
at the 160 ms chunk this engine pays about 1.2 WER points for true streaming; SPEC §5.1's rule
stands: pick by measured partial lag and stability, not by the brochure.

- Model card: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b (released
  2026-03-13 as "v2603"; the January 2026 checkpoint lives on branch
  `nemotron-speech-streaming-jan2026`) [V, read 2026-10-11].
- NVIDIA's ASR blog (2026-01-05): https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents

### The ONNX export and the runtime

NVIDIA ships NeMo/safetensors and a GGUF; the ONNX export comes from the **sherpa-onnx**
project (Apache-2.0), whose maintainers export the transducer branch (encoder + decoder +
joiner + `tokens.txt`) with `scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py`
([source](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py)).
Support landed in sherpa-onnx PR #3044 (merged 2026-01-14) [P]; the export script writes the
chunk size into the encoder's metadata (`chunk_size_ms`, `window_size`, `chunk_shift`,
`feat_dim: 128`, `subsampling_factor: 8`, `model_type: EncDecHybridRNNTCTCBPEModel`) and loops over
all four chunk sizes [P, read 2026-10-11].

Pinned here (Hugging Face, publisher `csukuangfj2`, exported 2026-04-25 -- after the March
model refresh, so from the v2603 checkpoint [E: the export script loads
`nvidia/nemotron-speech-streaming-en-0.6b` main; the repo README states only "chunk size is
160ms"]):

| Repo | Revision | Files (group) |
| --- | --- | --- |
| [`sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-2026-04-25`](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-2026-04-25) | `9660703a1289d5b5c0c6470ced80eb52096fc740` | `encoder.onnx` 42 MB + `encoder.data` 2.44 GB, `decoder.onnx` 29 MB, `joiner.onnx` 6.9 MB, `tokens.txt` (`stt-nemotron`, fp32, the default) |
| [`...-160ms-int8-2026-04-25`](https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25) | `237e551abd7a411ef92d3595454d9f6ab5fe7d6c` | `encoder.int8.onnx` 653 MB, `decoder.int8.onnx` 7.3 MB, `joiner.int8.onnx` 1.7 MB (`stt-nemotron-int8`, `stt.nemotron.quantization: int8`, CPU) |

Sizes and sha256 in `models.lock` come from the tree API
(`https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=true`, `lfs.oid`);
`tokens.txt` is a plain git blob there (no LFS hash), so it was downloaded (9 KB) and hashed.
The 80 / 560 / 1120 ms exports exist under the same publisher and date; adding one is a lock
group plus the `chunk_ms` Literal in `config.py`. The older `csukuangfj/...-2026-01-14` repos
are the January checkpoint at the 1120 ms default chunk -- not what live partials want.

**fp32 on the GPU, int8 for CPU.** Same reasoning as DECISIONS S4: the int8 export is
`quantize_dynamic` (`MatMulInteger`/`DynamicQuantizeLinear`), which has no fast CUDA kernels.
VRAM for fp32 is about 2.5 GB of weights plus caches [E]; SPEC §7.1's "≈ 1.5 GB" assumed fp16/int8.

**onnx-asr does not cover it.** onnx-asr 0.12.0 (2026-07-15) lists no Nemotron model and has
no streaming API (its README: "streaming is not yet supported" for T-one; nothing cache-aware)
[P, read 2026-10-11]. So the project's own ONNX Runtime cannot run this model directly without
writing the cache-aware feeding loop (feature extraction, cache tensors, greedy RNN-T) --
possible later (it would remove sherpa-onnx and the DLL question below entirely), not done now.

### sherpa-onnx on Windows with CUDA: the verdict

- **PyPI** `sherpa-onnx` 1.13.8 (2026-09-10) ships `win_amd64` wheels for CPython 3.12 (and
  3.7-3.14) -- **CPU only**. Every wheel bundles its own ONNX Runtime
  (`onnxruntime.dll` on Windows) [P].
- **CUDA wheels** are not on PyPI; they come from k2-fsa's find-links index
  https://k2-fsa.github.io/sherpa/onnx/cuda.html as local versions. For Windows x64 / cp312 the
  index has, at 1.13.8: `sherpa_onnx-1.13.8+cuda-cp312-cp312-win_amd64.whl` (**CUDA 11.8**) and
  `sherpa_onnx-1.13.8+cuda12.cudnn9-cp312-cp312-win_amd64.whl` (**CUDA 12.x + cuDNN 9**). The
  **CUDA 13** builds (`+cuda13.cudnn9.onnxruntime1.28.2`) exist for **Linux x86_64 only**
  [P, index read 2026-10-11]. Install docs: https://k2-fsa.github.io/sherpa/onnx/python/install.html
- The Windows GPU wheel is built against the prebuilt
  `onnxruntime-win-x64-gpu_cuda12-1.28.2.zip` (`cmake/onnxruntime-win-x64-gpu.cmake`:
  "Requires cuda 12.x, cudnn 9.x"), and copies `onnxruntime.dll`,
  `onnxruntime_providers_shared.dll` and `onnxruntime_providers_cuda.dll` into the package [P].
  It expects `cudart64_12`, `cublas64_12`, `cublasLt64_12`, `cufft64_11`, `cudnn64_9` (the CUDA 12
  build of cuDNN 9) on the DLL search path -- a CUDA 12 toolkit, or the `nvidia-*-cu12` pip wheels.

**DLL-clash analysis.** This server runs `onnxruntime-gpu[cuda,cudnn] >= 1.28, < 1.32`, the
**CUDA 13** build, whose `preload_dlls()` loads `onnxruntime.dll`, `onnxruntime_providers_cuda.dll`
and the CUDA 13 runtime (`cudart64_13.dll`, `cublas64_13.dll`, `cudnn64_9.dll` built for CUDA 13)
from the `nvidia-*` wheels. Windows resolves an implicit import by **module base name against
the modules already loaded in the process**: once `onnxruntime.dll` (CUDA 13 build) is in,
sherpa-onnx's pybind extension gets *that* `onnxruntime.dll`, not its bundled CUDA 12 one --
a different ORT build with a different C ABI surface and provider bridge -- and
`onnxruntime_providers_cuda.dll` / `cudnn64_9.dll` collide the same way (same file names,
different CUDA majors). The failure mode is an access violation or a silent CPU fallback, not
a clean error, and it is certain rather than probable: the names are identical. **Verdict:
sherpa-onnx with CUDA cannot share a Windows process with this server's ONNX Runtime.** [E,
from the published wheel contents; the crash itself was not reproduced here -- no GPU host.]

The design around it: `stt.nemotron.backend`

| backend | where sherpa-onnx runs | when |
| --- | --- | --- |
| `auto` (default) | `subprocess` on Windows with `gpu.provider: cuda`; `sherpa-onnx` otherwise | the safe choice everywhere |
| `sherpa-onnx` | in this process (`engines/nemotron.py::InProcessBackend`) | CPU provider; CUDA on Linux (the CUDA 13 wheel bundles ORT 1.28.2 and uses the same `nvidia-*` runtime wheels -- still two ORT copies in one process, but with distinct `.so` names resolved per path, so **[U]** until tried) |
| `subprocess` | `engines/nemotron_worker.py` in a child interpreter over pipes, one JSON line per call plus raw float32 audio | Windows + CUDA (refused in-process: `NemotronConfigError`) |

The worker is standalone (stdlib + numpy + sherpa-onnx; no SpeechForge import), so
`stt.nemotron.python` can point at a **separate venv** holding the `+cuda12.cudnn9` wheel and
the CUDA 12 runtime wheels; on Windows it adds every `site-packages/nvidia/*/bin` of that venv
to its own DLL search path before importing sherpa-onnx. The parent kills a worker that does
not answer within `worker_timeout_s` and the session gets `error{engine_failed}` rather than a
hung STT lane. One stream per session lives in the child (`create` / `feed` / `finalize` /
`reset` / `close_stream`). The pipe adds one process hop per partial: at 160 ms cadence and
~26 KB of audio per feed that is well under a millisecond [E].

`sherpa-onnx` not being on PyPI with CUDA also means the `stt-nemotron` extra can only pin the
CPU wheel (`sherpa-onnx>=1.13.0,<1.14`; 1.13.0 of 2026-04-28 is the first release after the
pinned exports). The GPU wheel is installed by hand into the worker venv -- commands below.

### How it is wired

- `engines/base.py`: `SttStreamHandle` (`feed` / `finalize` / `reset`) and `StreamingSttAdapter`
  (`create_stream` + the `SttAdapter` methods) Protocols; `is_streaming_stt()`; registered as
  `nemotron`.
- `stt_stream.py`: `RollingStt(stream=handle)`. A partial feeds only the audio that arrived since
  the last one (`PartialJob.streaming`) and takes the model's running text; nothing is
  re-decoded, nothing is committed (the model keeps its context; `stable_words` and
  `punct_final` are computed as before). `stt.flush` feeds the rest, `finalize`s (tail padding
  `tail_pad_ms` so the last chunk flushes), resets the handle, and the next utterance starts at
  `upto`. Word timings come from the RNN-T token frame times (80 ms frames; a `▁` token starts a
  word; same grouping rule as Parakeet's) relative to the utterance start, so `stt.final.words`
  keeps its meaning. A streaming model without timings falls back to one `transcribe_words`
  pass over the buffered utterance when the adapter has it, else sends `words: []`.
  With `stream=None` every code path is the old one (the existing tests prove it).
- `session.py`: `start_stt` opens one handle per session when the loaded STT adapter streams;
  `ready.engines.stt.sha8` now follows the adapter's lock key (Parakeet unchanged).
- `models.py`: `stt_group()` picks `stt-nemotron[-int8]` when `stt.engine: nemotron`, so
  `fetch-models`, `verify-models`, `doctor` and the load-time verification all follow.
- Config (`stt.nemotron`, all optional): `backend`, `chunk_ms` (160), `quantization`
  (`null` | `int8`), `num_threads`, `decoding` (`greedy_search` | `modified_beam_search`),
  `tail_pad_ms` (480), `python`, `worker_timeout_s`. Unknown keys are errors.
- `describe()` -> `name`, `weights`, `provider`, `backend`, `chunk_ms`, `device`, `revision`,
  `streaming: true` (and the internal `lock_key`, stripped before the wire).

### Licence (owner approval required)

The weights are under the **NVIDIA Open Model License Agreement**
(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/,
"last modified 2025-10-24" on the page read 2026-10-11) [P]. It is **not** a permissive OSI
licence, so under this project's rule (SPEC §7.1, D7: "the owner approves every non-permissive
licence personally") it is flagged here, in `THIRD_PARTY_NOTICES.md` and in `models.lock`.

What the text says (read 2026-10-11; not legal advice): commercial use, redistribution and
derivative models are permitted, royalty-free; outputs are not derivatives and NVIDIA claims no
rights in them; redistribution must carry the notice **"Licensed by NVIDIA Corporation under
the NVIDIA Open Model License"** and a copy of the agreement; the licence **terminates
automatically** if a safety or security guardrail of the model is bypassed without a
substantially similar one, or if the licensee sues NVIDIA claiming the model infringes; use
must be consistent with NVIDIA's Trustworthy AI terms; NVIDIA may update the agreement (comply
or stop using). SpeechForge never redistributes the weights -- the operator fetches them -- so
the terms bind the installation, and the `THIRD_PARTY_NOTICES.md` entry carries the notice.
sherpa-onnx itself and its ONNX Runtime are Apache-2.0 / MIT; the CUDA runtime wheels carry
NVIDIA's redistributable runtime terms, the same as the `gpu` extra's.

### Running it on the GPU host (operator)

Nothing below was run by the agent. Pick a card nothing else uses (the smoke script loads the
STT adapter alone, without a lease).

```powershell
# 1. Fetch and verify the weights (2.5 GB for fp32) into the normal data dir.
uv run speechforge fetch-models --group stt-nemotron
uv run speechforge verify-models

# 2. The worker venv: sherpa-onnx's CUDA 12 wheel + the CUDA 12 runtime wheels, apart from
#    the server's venv (CUDA 13). Python 3.12.
uv venv .venv-sherpa --python 3.12
uv pip install --python .venv-sherpa\Scripts\python.exe numpy `
    "sherpa-onnx==1.13.8+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html `
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 "nvidia-cudnn-cu12>=9,<10"
.venv-sherpa\Scripts\python.exe -c "import sherpa_onnx; print(sherpa_onnx.__version__)"

# 3. Smoke test: partial cadence, final text, word timings (CPU first, then CUDA).
uv run --extra gpu --extra stt-nemotron python scripts\gpu_smoke_stt.py --engine nemotron `
    --provider cpu --wav src\speechforge\assets\selftest-voice.wav
uv run --extra gpu --extra stt-nemotron python scripts\gpu_smoke_stt.py --engine nemotron `
    --device 3 --backend subprocess --python .venv-sherpa\Scripts\python.exe `
    --wav src\speechforge\assets\selftest-voice.wav --realtime

# 4. Same clip through Parakeet for the A/B (rolling re-decode; prints the same cadence line).
uv run --extra gpu python scripts\gpu_smoke_stt.py --engine parakeet --device 3 `
    --wav src\speechforge\assets\selftest-voice.wav --realtime

# 5. Serve with it (config.yaml): stt.engine: nemotron, stt.nemotron.python: <abs path to
#    .venv-sherpa\Scripts\python.exe>; then `speechforge bench` and the §11.3 gates.
```

Expected on the smoke run: one `partial` line whenever the text changes (the model emits a
chunk of text every 160 ms of audio, so lag should sit near 160-250 ms with `--realtime`),
then `final`, `words` and a `cadence` summary (`p90_cost` is the per-feed decode cost; the
§11.3 gate is partial lag p90 <= 250 ms).

### Open questions

1. **sherpa-onnx CUDA on Windows with this export** [U] -- SPEC §14's open item is still open:
   the GPU wheel exists for cp312/win_amd64 (CUDA 12 + cuDNN 9), but nobody has run Nemotron
   through it on a 3090 here. First run = step 3 above. If it fails in the worker, the log line
   `nemotron worker: DLL dirs [...]` shows which runtime directories were found.
2. **CUDA 12 runtime next to a CUDA 13 driver stack** -- the R580 driver runs CUDA 12 user-mode
   libraries fine (CUDA is backward compatible per driver) [P], but two CUDA runtimes on one card
   from two processes is an untested combination for this rig.
3. **Export checkpoint** [E]: the 2026-04-25 exports were made from `main` after the 2026-03-13
   refresh, so they should be v2603; a WER check against the card's 7.67 % (160 ms) on the local
   corpus confirms it.
4. **`tail_pad_ms`** 480 is a guess between "enough right context" and final latency; sherpa's
   example pads 660 ms. Tune from the smoke output (final `cost`).
5. **In-process CUDA on Linux** (two ORT copies, distinct `.so` names) [U] -- not relevant to
   the Windows rig, left as `sherpa-onnx` for `auto` there.
6. **Dropping sherpa-onnx altogether** by implementing the cache-aware loop on the project's
   own ONNX Runtime (the export's metadata documents the cache shapes). Removes the DLL
   question and the second venv; a day of work plus parity testing against sherpa's output.
