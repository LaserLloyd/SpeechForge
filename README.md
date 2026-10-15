# SpeechForge

A GPU speech server for realtime voice chat: **streaming speech-to-text** (NVIDIA Parakeet TDT
0.6B v2, rolling partials every 160 ms, finals with word timings), **semantic end-of-turn**
(Smart Turn v3.2, consulted after a silence floor the client cannot lower) and **streaming
text-to-speech with cloned voices** (Resemble AI's Chatterbox-Turbo), all on ONNX Runtime's CUDA
execution provider, over one persistent WebSocket per conversation (`sf.1`).

It runs on a multi-GPU Windows box that also runs [StudioForge](https://github.com/LaserLloyd/StudioForge)
(the LLM server) without fighting it: it takes a StudioForge **GPU lease** on one card while a
voice chat is open, never displaces the chat model, leaves when a better-class tenant asks, and
gives the card back after a quiet spell. A server that loaded without a lease (StudioForge
away) keeps asking for one and gives way the moment it is refused. It is a separate program on
purpose (DECISIONS S1).

**Status:** 0.3.0, alpha, in daily use. It has served the
[DisPatch](https://github.com/LaserLloyd/DisPatch_Chat) voice chat
([DisPatch Voice](https://github.com/LaserLloyd/DisPatch_Voice) 1.1.0 and later) from an RTX
3090 since 2026-10-11. Unit-tested with CPU fakes (protocol, sessions, leases, residency,
limits, auth, engines behind fakes: 365 tests), end to end with the real models on the CPU
provider, and benchmarked on the card ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

Measured on the host (fp16 Chatterbox, one session, over a WireGuard tailnet):

| What | Value |
| --- | --- |
| first TTS audio after the text arrives | p50 ~200 ms, p90 200-500 ms run to run |
| TTS real-time factor | 0.24-0.26 |
| `stt.flush` to `stt.final`, 8-15 s utterances under continuous partials | p50 90 ms, p99 170 ms |
| end of turn after the last word (700 ms floor + Smart Turn) | p50 ~0.8 s |
| `cancel` to GPU idle | < 10 ms |
| round-trip word error rate (TTS then STT) | 2.5 % |
| cold load of both models | 18-25 s (warm on boot is a config switch) |

## How a client uses it

The first client is the DisPatch voice add-on. The rule both sides keep: **the box decides,
the rig computes**, and **voice never waits on the rig** -- the client runs its own CPU engines
until `ready`, and falls back within a second if pongs stop.

```
box -> rig   hello{proto:"sf.1"}                       rig -> box  ready{engines, gpu, limits, lease}
box -> rig   20 ms mic frames (16 kHz PCM16)           rig -> box  stt.vad, stt.partial (160 ms), eou{p}
box -> rig   stt.flush{id}                             rig -> box  stt.final{text, words, upto}
box -> rig   tts.open{ctx, gen, voice}, tts.text ...   rig -> box  tts.started, tts.words, 40 ms audio frames, tts.done
box -> rig   cancel{gen} / credit / tts.pause          rig -> box  cancelled{stop_ms}
box -> rig   ping every 250 ms                         rig -> box  pong{gpu_util, vram_free_mib}
```

What the server promises a client, and what it never does:

* `eou{p}` is a score, not a decision. It fires once per pause, after at least
  `eou.min_silence_ms` (700 ms; 1000 ms when the running partial does not end a sentence);
  the client's `stt.config` can only raise that floor.
* A `played_ms` credit report is held until the next report; a client re-sends it every 250 ms
  while a reply is open. `limits.max_text` (2000 characters) is per context, one active and one
  queued context per session, and a longer reply continues in a fresh context.
* Lease states a client will see in `ready` and `state`: `held`, `unleased` (serviceable),
  `vacating{deadline_s}` (finish the reply, open no new one), `lost`. A vacate waits only for
  audio in progress, never for an idle open context.
* Voices are cloned from the four Chatterbox conditioning tensors (`PUT /v1/voices/<sha256>`),
  so a client that also runs Chatterbox on its CPU keeps one timbre on both. Reference clips are
  accepted only when `voices.allow_reference_clip` is on, and are cleaned before the encoder
  sees them.
* It never ends a turn, never calls an LLM and never stores audio or text.

Protocol: [`docs/PROTOCOL.md`](docs/PROTOCOL.md). HTTP utilities (health, ready, metrics,
voice storage, OpenAI-compatible one-shot routes): [`docs/API.md`](docs/API.md). Client-side
behaviour (deadlines, failover, when a turn starts on the rig):
[DisPatch Voice's rig guide](https://github.com/LaserLloyd/DisPatch_Voice/blob/main/docs/voice-chat/rig.md).
Reference client code: `src/speechforge/client.py` (`RigClient`), `bench.py` (a complete
session: voice upload, speak, cancel timing, mic streaming, a two-session soak).

## Install and operate (Windows host)

One elevated run of [`install.ps1`](install.ps1) installs the newest `v*` release tag straight
from this repository (`-Source <git URL | bundle | clone path>` for a fork or an offline bundle;
`-Tag` to pin one): driver/OS checks, uv, `releases\<tag>` + `current` layout, locked
environment, hash-verified models, token shown once, firewall rule scoped to the box
(`-BoxAddress`; the whole tailnet range when omitted), mini dumps, an at-logon **tray** (an "SF"
icon with a status dot, like the other Forges on the host; DECISIONS S25), `doctor`.
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
~7 GB of VRAM on the leased card (fp16 Chatterbox; `models.tts_dtype`).

Exposure: the server binds loopback and the host's tailnet address only (never `0.0.0.0`),
every route but `/health` needs the bearer token in an `Authorization` header, a browser
`Origin` is refused, and the token is never logged. The security review of the tree is
[`docs/SECURITY-REVIEW.md`](docs/SECURITY-REVIEW.md).

Engines beyond the default pair (Qwen3-TTS, dots.tts, Nemotron streaming STT) and what each
measured: [`docs/ENGINES.md`](docs/ENGINES.md). A blind listening test for choosing between
voices: [`docs/LISTENING-TEST.md`](docs/LISTENING-TEST.md).

## Development

```
uv sync --extra dev
uv run pytest -q                       # CPU fakes only: no ONNX Runtime, no models
uv run ruff check .
SPEECHFORGE_TEST_MODELS=<dir> uv run --extra cpu pytest tests/test_real_models.py
sh scripts/install-hooks.sh            # scrub hooks: no private data in commits
python scripts/scrub_check.py          # the same scan by hand (CI runs it on every push)
```

Models sit behind small adapters (`engines/base.py`, `engines/turn.py`); transport
(`realtime.py`, `session.py`, `tts_ctx.py`, `stt_stream.py`), residency/leases (`residency.py`,
`lease.py`) and the engines are separate layers. Design log: [`DECISIONS.md`](DECISIONS.md);
the specification the first build was written to: [`docs/SPEC.md`](docs/SPEC.md).

No private data ships: hostnames, addresses, tokens, voices, reference clips and listening-test
renders are all git-ignored or scanned for (`scripts/scrub_check.py` fails the commit, the
push and the CI run). A fork adds its own identifiers to the git-ignored
`scripts/scrub-rules.local.txt`.

## Licences

SpeechForge is MIT (`LICENSE`). Models are fetched at pinned revisions, not shipped:
Chatterbox-Turbo (MIT, Resemble AI); Parakeet TDT 0.6B v2 (CC-BY-4.0, NVIDIA; ONNX conversion
by istupakov, CC-BY-4.0); Smart Turn v3.2 (BSD-2); Silero VAD (MIT). The shipped self-test clip
is synthetic Kokoro-82M speech (Apache-2.0 weights). Attribution:
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
