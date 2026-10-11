# SpeechForge

A GPU speech server for realtime voice chat: **streaming speech-to-text** (NVIDIA Parakeet TDT
0.6B v2, rolling partials every 160 ms, finals with word timings), **semantic end-of-turn**
(Smart Turn v3.2 at every pause) and **streaming text-to-speech with cloned voices**
(Resemble AI's Chatterbox-Turbo), all on ONNX Runtime's CUDA execution provider, over one
persistent WebSocket per conversation (`sf.1`).

It runs on a multi-GPU Windows box that also runs [StudioForge](https://github.com/LaserLloyd/StudioForge)
(the LLM server) without fighting it: it takes a StudioForge **GPU lease** on one card while a
voice chat is open, never displaces the chat model, leaves when a better-class tenant asks,
and gives the card back after 15 idle minutes. It is a separate program on purpose
(DECISIONS S1).

**Status:** 0.2.0, alpha. Unit-tested with CPU fakes (protocol, sessions, leases, residency,
limits, auth) and end to end with the real models on the CPU provider; not yet run on a GPU.

## How a client uses it

The first client is the DisPatch voice add-on. The rule both sides keep: **the box decides,
the rig computes**, and **voice never waits on the rig** -- the client runs its own CPU
engines until `ready`, and falls back within a second if pongs stop.

```
box -> rig   hello{proto:"sf.1"}                       rig -> box  ready{engines, gpu, limits, lease}
box -> rig   20 ms mic frames (16 kHz PCM16)           rig -> box  stt.vad, stt.partial (160 ms), eou{p}
box -> rig   stt.flush{id}                             rig -> box  stt.final{text, words, upto}
box -> rig   tts.open{ctx, gen, voice}, tts.text ...   rig -> box  tts.started, tts.words, 40 ms audio frames, tts.done
box -> rig   cancel{gen} / credit / tts.pause          rig -> box  cancelled{stop_ms}
box -> rig   ping every 250 ms                         rig -> box  pong{gpu_util, vram_free_mib}
```

Protocol: [`docs/PROTOCOL.md`](docs/PROTOCOL.md). HTTP utilities (health, ready, metrics,
voice storage, OpenAI-compatible one-shot routes): [`docs/API.md`](docs/API.md).

## Install and operate (Windows host)

One elevated run of [`install.ps1`](install.ps1) installs the newest `v*` release tag straight
from this repository (`-Source <git URL | bundle | clone path>` for a fork or an offline bundle;
`-Tag` to pin one): driver/OS checks, uv, `releases\<tag>` + `current` layout, locked
environment, hash-verified models, token shown once, firewall rule scoped to the box
(`-BoxAddress`; the whole tailnet range when omitted), mini dumps, an at-logon **tray** (an "SF"
icon with a status dot, like the other Forges on the host; DECISIONS S23), `doctor`.
Re-running it is safe. Updates are blue/green from the same source with a canary self-test and
automatic rollback. A plain checkout with `SPEECHFORGE_DATA_DIR` set works too (the developer
layout). Details: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

| Command | What |
| --- | --- |
| `speechforge serve` / `supervise` | the server / the restart loop the launcher runs |
| `speechforge tray` | system-tray controller (`tray` extra): adopts a running server or launches `supervise`; status dot, restart, stop, logs, start at logon |
| `speechforge doctor [--full]` | driver R580+, CUDA, ORT window, cards (sm), models.lock, bind, token, firewall |
| `speechforge fetch-models` / `verify-models` | download / re-check every file against `models.lock` |
| `speechforge token new\|rotate\|fingerprint` | 256-bit bearer token; rotation keeps the old one 5 minutes |
| `speechforge bench --url ws://host:1236/v1/realtime` | the phase-1 acceptance gates, PASS/FAIL |
| `speechforge update --check\|--apply\|--rollback [--source URL\|PATH] [--tag T]` | blue/green release switch from a git remote or a bundle |

Requirements: Windows 11, NVIDIA driver R580+, Python 3.12 via uv,
`onnxruntime-gpu[cuda,cudnn] >=1.28,<1.32` (CUDA 13 + cuDNN 9 as pip wheels; no CUDA toolkit).
~6-8 GB of VRAM on the leased card.

## Development

```
uv sync --extra dev
uv run pytest -q                       # CPU fakes only: no ONNX Runtime, no models
SPEECHFORGE_TEST_MODELS=<dir> uv run --extra cpu pytest tests/test_real_models.py
sh scripts/install-hooks.sh            # scrub hooks: no private data in commits
```

Models sit behind small adapters (`engines/base.py`, `engines/turn.py`); transport
(`realtime.py`, `session.py`), residency/leases (`residency.py`, `lease.py`) and the engines
are separate layers. Design log: [`DECISIONS.md`](DECISIONS.md).

## Licences

SpeechForge is MIT (`LICENSE`). Models are fetched at pinned revisions, not shipped:
Chatterbox-Turbo (MIT, Resemble AI); Parakeet TDT 0.6B v2 (CC-BY-4.0, NVIDIA; ONNX conversion
by istupakov, CC-BY-4.0); Smart Turn v3.2 (BSD-2); Silero VAD (MIT). The shipped self-test clip
is synthetic Kokoro-82M speech (Apache-2.0 weights). Attribution:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
