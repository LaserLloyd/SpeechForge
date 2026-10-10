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

### Smart Turn v3.2 (end of turn)

- Source: https://huggingface.co/pipecat-ai/smart-turn-v3 (`smart-turn-v3.2-gpu.onnx`; pinned in
  `src/speechforge/models.lock`)
- Licence: BSD-2-Clause, Copyright (c) Daily / pipecat-ai.

### Silero VAD v5

- Source: https://github.com/snakers4/silero-vad (`silero_vad.onnx`, pinned in `models.lock`)
- Licence: MIT, Copyright (c) Silero Team.

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
| websockets | BSD-3-Clause |
| tokenizers, huggingface-hub | Apache-2.0 |
| FastAPI, Starlette, Pydantic, httpx, uvicorn | MIT / BSD-3-Clause |
| NumPy | BSD-3-Clause |
| PyYAML | MIT |
| nvidia-ml-py | BSD-3-Clause |

## Code lineage

`src/speechforge/engines/chatterbox.py`, `src/speechforge/pipeline.py` and the Whisper
log-mel features in `src/speechforge/engines/turn.py` port code from the DisPatch voice add-on
(https://github.com/LaserLloyd/DisPatch_Voice, MIT, same author).
