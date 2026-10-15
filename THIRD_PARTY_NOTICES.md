# Third-party notices

SpeechForge's own code is MIT (see `LICENSE`). It downloads and runs models and
libraries that carry their own licences. None of the model weights are in this
repository; `speechforge fetch-models` fetches them from their publishers at
pinned revisions.

## Models

### Chatterbox-Turbo (ONNX export)

- Source: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX (pinned revision
  in `src/speechforge/config.py`)
- Upstream: https://github.com/resemble-ai/chatterbox
- Licence: MIT, Copyright (c) Resemble AI.
- Note: Resemble's PyTorch package embeds a Perth audio watermark in its output.
  SpeechForge runs the ONNX export without torch and does not apply that watermark
  (the same as the DisPatch voice add-on's streaming engine). If you publish audio
  made with a cloned voice, say that it is synthetic.

### Parakeet TDT 0.6B v2

- Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- ONNX conversion used: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx
  (pinned revision in `src/speechforge/config.py`)
- Licence: Creative Commons Attribution 4.0 International (CC-BY-4.0),
  https://creativecommons.org/licenses/by/4.0/
- Attribution: "Contains NVIDIA Parakeet TDT 0.6B v2 by NVIDIA, licensed CC-BY-4.0;
  ONNX conversion by istupakov, CC-BY-4.0. The weights are used unmodified."

### Nemotron Speech Streaming EN 0.6B (optional STT engine, `stt.engine: nemotron`)

- Source: https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b (NVIDIA, released
  2026-03-13; Jan 2026 checkpoint on its `nemotron-speech-streaming-jan2026` branch)
- ONNX export used: the sherpa-onnx maintainers' 160 ms-chunk exports,
  https://huggingface.co/csukuangfj2/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-2026-04-25
  (fp32) and `...-160ms-int8-2026-04-25` (int8); pinned revisions in `src/speechforge/config.py`
  and per-file hashes in `src/speechforge/models.lock`. Only the transducer branch is exported.
- Licence: **NVIDIA Open Model License Agreement** (not an OSI licence),
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
  (version last modified 2025-10-24 at the time of writing). It permits commercial use,
  redistribution and derivatives. Conditions to be aware of: the attribution notice below
  must travel with any redistribution of the weights; the rights end automatically if a
  safety/security guardrail of the model is bypassed without a substantially similar one,
  or if the licensee sues NVIDIA alleging the model infringes; use must follow NVIDIA's
  Trustworthy AI terms; NVIDIA may update the agreement. SpeechForge does not redistribute
  the weights (they are fetched from Hugging Face by the operator), so these terms bind the
  operator who installs the engine, not the users of this repository. The owner approves this
  licence explicitly before the engine is enabled on a deployment.
- Required notice (Section 3.1 of the agreement):
  "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
- Runtime: sherpa-onnx (https://github.com/k2-fsa/sherpa-onnx, Apache-2.0), the `stt-nemotron`
  extra; its wheels bundle their own copy of ONNX Runtime (MIT) and, for the GPU builds, link
  NVIDIA's CUDA 12 / cuDNN 9 runtime libraries (NVIDIA software licence agreements).

### Smart Turn v3.2 (end of turn)

- Source: https://huggingface.co/pipecat-ai/smart-turn-v3 (`smart-turn-v3.2-gpu.onnx`; pinned in
  `src/speechforge/models.lock`)
- Licence: BSD-2-Clause, Copyright (c) Daily / pipecat-ai.

### Silero VAD v5

- Source: https://github.com/snakers4/silero-vad (`silero_vad.onnx`, pinned in `models.lock`)
- Licence: MIT, Copyright (c) Silero Team.

### WeSpeaker ECAPA-TDNN speaker embedding (evaluation only, optional)

- Source: https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512
  (`voxceleb_ECAPA512.onnx`, pinned in `models.lock` under group `eval`; fetched only by
  `speechforge fetch-models --group eval`)
- Upstream: https://github.com/wenet-e2e/wespeaker
- Licence: Apache-2.0, Copyright (c) the WeSpeaker authors.
- Used by `tools/listening/similarity.py` for the speaker-similarity check of SPEC §11.3; the
  server never loads it.

### Qwen3-TTS-12Hz-1.7B-Base and -0.6B-Base (phase-2 candidate, optional)

- Source: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base and
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base (revisions pinned in
  `src/speechforge/config.py` and `models.lock`; groups `tts-qwen3`, `tts-qwen3-0.6b`)
- Upstream: https://github.com/QwenLM/Qwen3-TTS (report: https://arxiv.org/abs/2601.15621)
- Licence: Apache License 2.0, Copyright (c) Alibaba Cloud / Qwen team. The model cards
  declare `license: apache-2.0`; the code repository carries the LICENSE text.
- Attribution: "Contains Qwen3-TTS by the Qwen team, Alibaba Cloud, licensed Apache-2.0.
  The weights are used unmodified (safetensors)."
- Note: this engine clones from a reference clip and its transcript held on the server
  (voice format `qwen3tts-ref`), which decision D3 keeps off by default. It is only loaded
  when `tts.engine: qwen3-tts` and `voices.allow_reference_clip: true` are both set.

### dots.tts-mf (phase-2 TTS candidate; `tts.engine: dots-tts`)

- Source: https://huggingface.co/dots-studio/dots.tts-mf (formerly `rednote-hilab/dots.tts-mf`;
  pinned revision in `src/speechforge/models.lock`, group `tts-dots`), and
  https://huggingface.co/dots-studio/dots.tts-mf-2steps-stts (group `tts-dots-stts`)
- Upstream code: https://github.com/studio-dots-ai/dots.tts (formerly `rednote-hilab/dots.tts`),
  PyPI package `dots.tts`
- Licence: Apache-2.0 for the code and the released checkpoints, Copyright 2026 dots.tts Team,
  RedNote. https://www.apache.org/licenses/LICENSE-2.0
- Attribution: "Contains dots.tts-mf by the dots.tts Team (RedNote), licensed Apache-2.0. The
  weights are used unmodified." Not installed or fetched by default: the engine loads only with
  the owner's `voices.allow_reference_clip: true` (decision D3) and `tts.engine: dots-tts`.
- Note: the authors ask that reference voices be used with consent and that AI-generated audio
  be clearly marked; dots.tts applies no watermark. If you publish audio made with a cloned
  voice, say that it is synthetic.

### Fish Audio Drama 3 through audio.cpp (add-on engine; `tts.engine: drama3-tts`)

- Weights: https://huggingface.co/fishaudio/drama-3 (Fish Audio), run here as the community
  GGUF package https://huggingface.co/Jmica/Drama-3-GGUF (a derivative work under the same
  licence). Not in `models.lock`: `speechforge engine-setup drama3` installs it with the
  port's model manager and verifies the pinned sha256.
- Licence: **Fish Audio Community License v1.0** (2026-10-13). Free for research, personal
  and non-commercial use and for commercial use under USD 500,000 annual revenue; a hosted
  text-to-speech or voice-cloning service for third parties needs a separate licence
  (contact on fish.audio); no cloning of an identifiable person without consent; no watermark
  removal; no training of other speech models on its outputs. Required attribution:
  **Built with Fish Audio** (the engine's `describe()` carries it; display it on any
  product that uses the engine).
- Runtime: https://github.com/JarodMica/audio.cpp, branch `drama-3-native`, Apache-2.0,
  Copyright 2026 ShugoAI LLC. Built from source outside the repository by `engine-setup`,
  which first applies `src/speechforge/engines/patches/audiocpp-drama3-prefix-cache.patch`
  (SpeechForge's own change to that Apache-2.0 code, under this repository's MIT licence;
  `--no-prefix-cache` builds the stock port).
- Nothing is fetched or installed by default; the engine loads only with the owner's
  `voices.allow_reference_clip: true` (decision D3) and `tts.engine: drama3-tts`.

### Self-test clip (shipped)

- `src/speechforge/assets/selftest-voice.wav`: synthetic speech made with Kokoro-82M (voice
  `am_michael`; weights Apache-2.0, https://huggingface.co/hexgrad/Kokoro-82M). Not a recording of
  a person. Taken from the DisPatch voice add-on's stock clip (MIT, same author).

## Libraries (installed by pip/uv, not vendored)

| Library | Licence |
| --- | --- |
| ONNX Runtime (`onnxruntime-gpu`) | MIT |
| NVIDIA CUDA / cuDNN runtime wheels (`nvidia-*`, pulled by `onnxruntime-gpu[cuda,cudnn]`) | NVIDIA software licence agreements (redistributable runtime) |
| onnx-asr | MIT |
| sherpa-onnx (`stt-nemotron` extra; bundles ONNX Runtime 1.28.2) | Apache-2.0 |
| websockets | BSD-3-Clause |
| tokenizers, huggingface-hub | Apache-2.0 |
| FastAPI, Starlette, Pydantic, httpx, uvicorn | MIT / BSD-3-Clause |
| NumPy | BSD-3-Clause |
| PyYAML | MIT |
| nvidia-ml-py | BSD-3-Clause |
| faster-qwen3-tts (`tts-torch` extra only) | MIT, Copyright (c) Andres Marafioti |
| qwen-tts-hf (the `qwen_tts` package, a Transformers-5 build of QwenLM/Qwen3-TTS; `tts-torch` only) | Apache-2.0, Alibaba Cloud / Qwen team |
| PyTorch, torchaudio (`tts-torch` only, CUDA 13.0 wheels) | BSD-3-Clause; bundled NVIDIA CUDA runtime libraries under NVIDIA's licence |
| transformers, accelerate (`tts-torch` only) | Apache-2.0 |
| librosa, soundfile, einops, pysox (`tts-torch` only) | ISC / BSD-3-Clause / MIT / BSD-3-Clause |
| dots.tts (only with `tts.engine: dots-tts`) | Apache-2.0 |
| PyTorch, torchaudio (phase-2 torch engines) | BSD-3-Clause |
| transformers, safetensors, einops, langcodes, lingua-language-detector | Apache-2.0 |
| librosa, soundfile, loguru, torchdiffeq, tqdm | ISC / BSD-3-Clause / MIT / MIT / MIT+MPL-2.0 |

## Code lineage

`src/speechforge/engines/chatterbox.py`, `src/speechforge/pipeline.py` and the Whisper
log-mel features in `src/speechforge/engines/turn.py` port code from the DisPatch voice add-on
(https://github.com/LaserLloyd/DisPatch_Voice, MIT, same author).
