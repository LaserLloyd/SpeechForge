# Decisions

Running log of the decisions behind SpeechForge, with the reasoning and the evidence. Newest
appended. Numbered `S<n>` so they never collide with StudioForge's `D<n>`, which several of
these cite.

---

## S1 -- Its own program, not a StudioForge feature

**Decision.** Speech runs in a separate process tree with its own repository, port (`:1236`),
config, logs and supervisor. It talks to StudioForge only through StudioForge's public GPU lease
API.

**Why.** StudioForge's first decision (D1) is "never implement inference": it is a manager of
`llama-server` children, and its crash-isolation story ("a crash takes down one model, never the
gateway") depends on that. Speech is inference -- an ONNX Runtime with CUDA contexts, cuDNN and
a ~4 GB footprint -- and putting it in the gateway process would make a speech fault a gateway
fault and couple two release cycles. A pinned image-rendering workload already proved the tenant shape on this
rig: a separate service that leases a card from StudioForge, renders, and gives the card back.

**Cost.** One more thing to install, update and watch on the rig. Mitigated by the same
launcher/logon-task/update pattern StudioForge uses (S9).

## S2 -- A request never waits for the GPU side

**Decision.** Any request that finds the models not loaded (loading, refused, cooling down,
draining) is answered **at once** with `503` and a `Retry-After`, and kicks the load so the next
request finds them ready. `POST /v1/speech/warm` does the same without a payload. Concurrency
caps answer `503 busy` with `Retry-After: 1` rather than queueing.

**Why.** The first client is a voice conversation with a local CPU fallback. Its latency budget
(DisPatch voice SPEC §4.9) has no room for a queue on another machine: a hedged client that waits
even half a second on a busy rig has already lost what the rig would have saved. "Fail fast,
say when to come back" lets the client decide, and keeps the server simple: no request ever
holds a model load open.

## S3 -- One card, leased while in use, class 2, never forced

**Decision.** Before loading, SpeechForge `POST`s a StudioForge lease (D43) for the first card in
`gpu.devices` that is granted (default `[3]`), with `holder: speechforge`, `priority: 2`, a
`vacate_url` + per-process `vacate_token` (D56), `idle_ttl_s = idle_unload_s + 300`, and never
`force`. It touches the lease (at most every 30 s, only after activity) while serving, releases
it when it unloads after `idle_unload_s` (900 s) of quiet, and releases any lease under its holder
name at boot. A vacate request drains in-flight work (bounded, 15 s), unloads, releases and stays
off the card for `vacate_cooldown_s`.

**Why class 2.** StudioForge measures an idle resident against the *claim's* class (D56 item 6):
a class-2 grant may displace an idle tier-2/3 resident and is refused by an idle tier-1 one. Voice
is a person waiting, so it outranks background work and renders (class 3) -- but the chat model
(tier 1) is also a person waiting, and voice must never evict it. Class 1 would. A busy resident
is never interrupted at any class (D36): the grant answers 503, SpeechForge cools down for
StudioForge's `retry_after_s` and the client stays local.

**Why a whole card, and why no StudioForge change.** A D43 lease is exclusive: nothing else is
planned onto the card while it stands, which is more than ~4 GB of speech needs on a 24 GB card.
The alternative -- a "share" lease that reserves N MiB instead of a card -- would need a new
planner rule in StudioForge. It was considered and not built: (a) the lease is held only while
voice is in use plus the idle window, (b) the preferred card (CUDA 3, an RTX 3090) is the least
used on this rig (the least busy at design time; the LLMs live on the 5090s and an image-rendering
workload is pinned to CUDA 2), and (c) without a lease the load is a check-then-act race against a JIT LLM
load -- a race that has already corrupted an image render on this rig. If CUDA 3 becomes contested, the
share lease is the StudioForge change to make; nothing here would need to change but one field.

**Why fail open.** StudioForge unreachable means its planner is not JIT-loading anything onto the
card, so loading unleased is safe (`on_unreachable: proceed`, the image-rendering tenant's rule); `refuse` exists
for an operator who disagrees. A StudioForge without the lease API (404/405) is treated the same.

## S4 -- The same ONNX models as the client, on an RTX 3090, ONNX Runtime CUDA 13 build

**Decision.** Chatterbox-Turbo ONNX at the add-on's pinned revision (fp32 by default) and
Parakeet TDT 0.6B v2's **fp32** ONNX export, both through `onnxruntime-gpu[cuda,cudnn] >= 1.28, < 1.32` (1.28 fixed `preload_dlls()` for the CUDA 13 wheel layout on Windows)
on CUDA 3 (a 3090, sm_86).

**Why the same models.** A voice profile is the speech encoder's tensors, and the add-on already
has them; the same export means the rig and the CPU fallback speak with one timbre, so a sentence
that falls back mid-conversation does not change voice. Measured on the CPU provider: the GPU
engine's IOBinding loop samples **token-for-token the same speech tokens** as the add-on's engine
for the same seed and the same files (2/2 sentences, 85 and 62 tokens), and the voice tensors are
identical.

**Why not PyTorch Chatterbox.** Possibly faster for the autoregressive LM (no published
head-to-head exists), but it costs a 2.5 GB torch install with its own CUDA wheels, loses
parity with the client's tokens and voice files, and needs a cu128+ build for the 5090s. Worth a
measured experiment later behind the same adapter interface (S10), not the starting point.

**Why a 3090.** These are small, latency-bound, batch-1 models (a 350M GPT-2-medium LM stepping
one token at a time, a conditional decoder over ~1-4 s of audio, a 0.6B encoder). They are
bounded by kernel launches, not FLOPs, so a 5090 buys little; sm_86 has had native kernels in
every ONNX Runtime CUDA build, while sm_120 has a known gap (1.23.0 shipped without usable code
for it). And the 5090s are where the large LLMs fit; the 3090 on CUDA 3 is the card StudioForge
uses least. `gpu.devices` takes a preference list if that changes.

**Why fp32 first.** The int8 ONNX exports use `MatMulInteger`/`DynamicQuantizeLinear`, which
have no fast CUDA kernels; fp16 halves VRAM but its numerics for this vocoder are unverified.
24 GB makes fp32 affordable; `speechforge bench` compares fp16 on the rig before anyone switches.

## S5 -- The KV cache never leaves the card; the footprint is fixed at warm-up

**Decision.** The language-model step runs through an ONNX Runtime `IOBinding`: `present.*` are
bound to device memory and fed back as the next step's `past_key_values.*` as `OrtValue`s; only
the logits row (6563 floats) and the next token's embedding (1024 floats) cross the bus per
token. The voice prefix's KV cache is computed once per voice and stays resident. CUDA provider
options: `arena_extend_strategy: kSameAsRequested`, `gpu_mem_limit` per session
(`gpu.mem_limit_mb`), `cudnn_conv_algo_search: HEURISTIC`, `do_copy_in_default_stream: 1`.
Every load ends with a warm-up (a synthetic voice, one sentence, one transcription).

**Why.** The CPU engine round-trips the KV cache through numpy each token; on a GPU that is ~40
MB device->host->device per token at 200 positions -- more time than the step itself. A fixed
footprint matters to the neighbours: StudioForge plans with measured free VRAM, so a tenant whose
arena grows mid-conversation would invalidate the plan it was granted. Exhaustive cuDNN search
re-tunes for every new shape, and every sentence is a new shape.

**Guard.** With `provider: cuda`, every Chatterbox session must report the CUDA provider first or
the load fails: a rig silently serving from its CPU is slower than the client and looks healthy.

## S6 -- Voices are content-addressed tensors, sent once

**Decision.** `PUT /v1/voices/<sha256>` stores the four speech-encoder tensors as `.npz`; the id
is the sha256 of the bytes (a mismatch is a 400). `HEAD` answers "do you have it". Synthesis
names the voice by id. Both naming schemes (the add-on's `cond_emb`... and the encoder's
`audio_features`...) are accepted. `allow_pickle=False`, zip-size check, shape/dtype/finite/range
validation. 256 files on disk, 32 parsed in memory, LRU. `POST /v1/voices/from-reference` builds
one from a clip with the server's own encoder.

**Why.** The client knows the id before uploading, so a voice is sent once per server lifetime and
a restart costs one 404 and one upload. Tensors rather than a clip keep the parity of S4 and
skip the encoder per sentence.

## S7 -- The speech stream: committed only when there is audio

**Decision.** `POST /v1/audio/speech` waits (up to 30 s) for the first decoded chunk before it
returns the `200`. A failure before any audio is a JSON error with a proper status; a failure
after audio started aborts the chunked response, so the client sees a transport error rather
than a clean but truncated sentence. A client that disconnects stops the LM within a token.

**Why.** The status line goes out with the first byte, and there is nothing to send before the
first audio anyway, so waiting costs nothing and turns "200 then garbage" into a status a
fallback can act on. The add-on's `FallbackTTS` rule is the same: fail over before audio, end the
sentence after it.

## S8 -- Configuration, data and secrets

**Decision.** `SPEECHFORGE_DATA_DIR` (else `<repo>/data`) holds `config.yaml`, `speechforge.env`,
`models/`, `voices/`, `logs/`. Secrets are names in the env file -- `SPEECHFORGE_TOKEN`,
`SPEECHFORGE_SF_KEY` -- never in YAML, never logged (`check` says "set"/"NOT SET"; `token
fingerprint` prints 8 hex chars). Only `SPEECHFORGE_*` names are read from the file, and the
process environment wins. A non-loopback bind without a token is refused at start (exit 78).
Unknown config keys are an error.

## S9 -- Supervision, delivery, rollback

**Decision.** `speechforge supervise` (what the launchers run) restarts the server when it exits
(backoff 2 s doubling to 60 s, reset after 10 min up) or when `/health` fails three probes in a row;
stops it gracefully (CTRL_BREAK on Windows, SIGINT elsewhere, then terminate, then kill) so the
lease is released on the way out; and does not restart a configuration refusal (exit 78). A
per-user logon task (`SpeechForge Autostart.bat`) starts it. Code arrives as a **git bundle**
(`Update SpeechForge.bat <bundle>`: verify, fast-forward pull, `uv pip install -e .[gpu]`, fetch
missing models); rollback is `git reset --hard <previous commit>` (the updater prints it) and a
restart. No public remote until the owner decides to publish.

## S10 -- Models behind two adapters

**Decision.** The HTTP layer, residency and leases see one `SttAdapter` and one `TtsAdapter`
(`engines/base.py`), chosen by name (`stt.engine`, `tts.engine`) from a registry; voice parsing is
the TTS adapter's. Chatterbox and Parakeet are the two built-ins.

**Why.** The engine choice and the client protocol (request/response today; a persistent
bidirectional session is likely next) are expected to move independently of the lease and
residency machinery, which is the part that has to be right for the rig's other tenants.

---

## S11 -- The realtime path is one WebSocket per voice chat (sf.1)

**Decision.** `GET /v1/realtime`, sub-protocol `sf.1` (`docs/PROTOCOL.md`): mic audio in
continuously, partials/end-of-turn/finals/word timings/TTS audio out, cancel by generation,
credit, pings, all on one socket. The per-request HTTP routes stay as utilities only.

**Why.** Per-request HTTP cannot stream partials while the user talks, accept text
incrementally, cancel cleanly, or find a dead rig in under a second (the design spec's
audit gap 1). Every realtime voice stack runs a persistent bidirectional stream.

**Choices the spec left open, made here:** `cancel{gen}` stops every context with gen **<=**
the given one (the barge-in generation itself included); `credit` carries an optional
`played_ms` and otherwise assumes real-time playback from the first frame; `tts.pause` /
`tts.resume` are the false-interruption verbs; `voice.put` over the socket is chunked
(64 KiB messages cannot carry a 600-800 KB conditionals file); an unknown frame is an error,
never a close; 50 errors in 10 s close with 1008.

## S12 -- Two lanes in one process; TTS interleaves per LM step

**Decision.** One uvicorn worker, models loaded once. STT work (end-of-turn first, then
finals, then partials) runs on one priority-ordered worker thread; each TTS context has its
own producer thread, and the Chatterbox LM lock is held **per decode step**, not per sentence,
so two sessions' sentences interleave token by token instead of queueing a whole sentence.
Credit stalls and pauses wait *before* an LM step, so a stalled context holds no GPU time.

**Not done (yet).** Separate CUDA streams with priorities and CUDA graphs for the LM step
(§4.4 of the spec): ONNX Runtime's Python API exposes neither per-run stream priority nor a
graph that tolerates a growing KV cache without a fixed-shape re-export. Measure first
(`bench`); the per-step lock already gives the round-robin.

## S13 -- models.lock: every file pinned by sha256

**Decision.** `src/speechforge/models.lock` lists URL (with the pinned revision), size and
sha256 for every file (generated from the Hugging Face tree API at the pinned revisions).
Downloads stream to `.partial`, hash on the fly, rename only on a match. Each load checks size
and a stamp (`data/state/model-stamps/`, never inside the models directory, which may be
shared or read-only); a stamp older than 7 days (or none) means a full re-hash. The lock is
part of a release, so a model change goes through the blue/green gate like code.

## S14 -- Ready means self-tested

**Decision.** After loading and warming up, the server runs a self-test before any session
gets `ready`: the shipped synthetic clip transcribes within 15 % WER of its known text; a voice
built from it speaks a 6-word line with TTFA <= 250 ms (enforced on CUDA); a cancel stops
within 50 ms (enforced on CUDA); the spoken line transcribes back within 25 % WER. A failure
unloads, releases the lease and cools down. The clip is Kokoro-generated speech, not a person,
so building it into conditionals on the rig does not touch D3.

Measured on this box's CPU provider (q8 Chatterbox, int8 Parakeet) during development: golden
STT WER 0.0, round trip 0.0. A first version used "The quick brown fox jumps today." and failed
on "to day" (one split word = 33 % WER on six words); the line is now "Sure, I can help with
that."

## S15 -- Exposure: tailnet + loopback, explicit binds, rotating tokens

**Decision.** `server.hosts` is a list of addresses; loopback and tailnet (100.64.0.0/10,
fd7a:115c:a1e0::/48) are allowed, `0.0.0.0`/`::` never, anything else only with TLS; any
non-loopback bind needs a token. `/metrics` is loopback **and** token; the vacate callback is
loopback-only; a WebSocket upgrade carrying `Origin` is refused; upgrades are rate-limited per
address. `token rotate` keeps the old token valid for 5 minutes and the server re-reads the env
file on change. The reference-clip route is off by default (D3). `install.ps1` adds a
firewall rule scoped to the box's tailnet address and sets mini dumps without heap.

## S16 -- StudioForge unreachable: load only on a clearly free card

**Decision.** Replaces S3's plain fail-open: with StudioForge unreachable, SpeechForge loads
unleased only if NVML shows >= 8 GiB free on the card and no `llama-server` process holds
memory there; otherwise it refuses and cools down.

## S17 -- Blue/green releases, canary-gated

**Decision.** `releases\<tag>` + a `current` junction; `update --apply` builds the new release
beside the running one (never `git pull` in place), gates it on a loopback canary (`:1237`,
leases off, same card) over sf.1, drains, switches, restarts, and switches back if the switched
server does not reach `ready` within 2 minutes. The supervisor starts the child from
`current\.venv`, stops restarting after 5 crashes in 10 minutes and then answers `/health` with
`crashloop` itself. Supersedes S9's bundle pull.

**Source (added when the repository went public).** A release is a `v*` tag, newest by version
order. `install.ps1 -Source` and `update --source` take a git URL (default: the public
repository, else the running release's `origin`) or a local bundle/clone path -- the bundle
stays as the offline route, nothing else changes between the two: the tag is checked with
`git ls-remote --tags`, then cloned shallow at that tag. A plain checkout with
`SPEECHFORGE_DATA_DIR` set (the developer layout) is supported beside the releases layout;
`doctor` names the layout it sees and does not look for a junction in a checkout.

## S18 -- Long-lived worker threads and one decoder shape

**Decision.** `pipelined()` runs the language-model producer and the decoder on
long-lived thread pools that outlive the sentence; warm-up runs one LM step on every
LM worker and one decode per bucket on the decoder's thread; and every decoder input
pads to a single shape (`DECODE_BUCKETS = (40, 64)`, SILENCE tokens whose audio is
dropped).

**Why (measured on the host, 2026-10-11; `docs/BENCHMARKS.md`).** ONNX Runtime's CUDA
provider builds a per-thread context (cuBLAS/cuDNN handles) the first time a thread
runs a session: 300-450 ms. A thread per sentence paid that on every first chunk, so
TTFA was bimodal (the same 18-token decode took 50 ms or 450 ms). A new decoder input
shape costs 150-700 ms the first time, and switching between two *known* shapes still
costs ~200 ms per switch, while the decode itself costs ~55 ms whether the input holds
8 or 40 tokens. Together these took the in-process first audio from p50 662 / p90 691 ms
to p50 221 / p90 233 ms (fp16, first chunk 10) and the RTF from 0.67 to 0.30.

**Cost.** Up to 32 padding tokens are decoded and discarded per chunk (no measurable
time, since the decoder's cost is flat in this range). The pools are module-level and
sized for `realtime.max_sessions x max_contexts` plus warm-up.

## S19 -- fp16 Chatterbox on the host; the repo default stays fp32 for now

**Decision.** The host's config sets `models.tts_dtype: fp16` and
`tts.first_chunk_tokens: 10`. The repository default remains fp32 (S4) until the
speaker-similarity check of SPEC §11.3 has run with the owner's voice.

**Why.** The LM step is 11-13 ms per token at fp16 and 28-30 ms at fp32 on the same
card (the fp32 path also pays ~4 ms of IOBinding overhead per step); the RTX 5090 is no
faster than the 3090 at fp32, so this is launch-bound work and only the smaller graph
helps. The self-test's WER gates hold at fp16 (golden STT 2.9 %, round trip 0 %) and the
harness round trip reads 2.5 %. What has not been measured is timbre: S4 asked for a
golden-audio and similarity check before fp16 becomes the default, and that needs the
owner's clip (tools/listening, `similarity.py`).

## S20 -- The harness measures from the wire, not the nominal clock

**Decision.** `RigClient.send_mic(realtime=True)` paces frames against a deadline
schedule and returns the wall time at which the frame carrying a given sample left;
`bench` measures end-of-turn latency from that moment.

**Why.** `asyncio.sleep(0.02)` lasts ~31 ms on Windows' 15.6 ms timer, so the "real-time"
stream ran at two thirds of real time and the first EOU measurement read 6.9 s while the
server's `eou` arrived 125 ms after the last speech frame. The gate now reads 171 ms.

## S21 -- Chunks are sized to the measured token rate

**Decision.** After the first chunk, `pipelined()` sizes the next chunk so that generating
its missing tokens at the median step time measured so far, plus one decode, finishes
`margin_ms` (100) before the audio already emitted runs out; clamped to
`[min_chunk (6), chunk_tokens (25)]`. `ChunkPlanConfig.adaptive` turns it off.

**Why.** The 30-minute two-session soak counted one underrun per sentence per session: the
10-token first chunk is 400 ms of audio and the fixed second chunk (25 + 3 lookahead tokens)
takes ~780 ms when two sessions interleave on the LM at 26 ms/token (S12). A fixed schedule
is right for exactly one token rate; the real rate depends on how many contexts share the
card. With one session (13 ms/token) the schedule still reaches 25-token chunks.

## S22 -- A fixed footprint: power-of-two arenas and warm-up to the working set

**Decision.** `arena_extend_strategy: kNextPowerOfTwo` for every CUDA session; warm-up also
synthesises one long line and transcribes one 8 s window (`realtime.max_tail_s`).

**Why (measured, 2026-10-11).** With `kSameAsRequested` the TTS arenas grew 2.9 GB over the
first 30 sentences after load and the STT arena 374 MiB over its first 120 varied windows,
then both plateaued: a plateau, not a leak, but it showed up as "VRAM drift" while serving
(336 MiB over the soak; 1.6 GB since load) and it is the neighbours' free memory that
StudioForge plans with (S5). With power-of-two extension the same runs grew 54 MiB (TTS) and
14 MiB (STT) and the total card use was 7.1 GB instead of 8.5 GB, at the same speed. The two
extra warm-up items make the first long sentence and the longest partial window pay their
allocations at load rather than in a conversation.

## S23 -- Owner decisions after the first host build (2026-10-11)

* **D3: reference clips may be stored on the host for the phase-2 bake-off.** The host's
  config sets `voices.allow_reference_clip: true`; the clip lives in the user-only voices
  folder (never indexed, never backed up, deleted with the profile) and never leaves the
  machine. The repository default stays `false`.
* **Nemotron Speech Streaming is approved for private use only** (NVIDIA Open Model License):
  it stays off by default and is never recommended in public documents.
* **fp16 Chatterbox stays a host setting** until the speaker-similarity check of SPEC §11.3
  has run with the owner's voice (S19).
* **Residency on the host:** `warm_on_boot: true` and `idle_unload_s: 3600`, so the cold load
  is paid at logon and a quiet hour keeps the card; a tier-1 (chat model) lease still vacates
  it within the drain budget (StudioForge D56).
* **Trial order for phase 2:** Qwen3-TTS, then dots.tts, then Nemotron; each measured with
  its GPU smoke script before any wider wiring.
* **The long-term voice (D2) is decided by the family listening test**, not by latency alone.

## S25 -- A tray, like the other Forges

**Decision.** `speechforge tray` (extra `tray`: pystray + Pillow, imported lazily) puts an "SF"
icon with a status dot in the notification area beside the other Forges' trays. It is a
controller, not a second supervisor: it **adopts** a server whose `/health` already answers,
else launches `speechforge supervise` as a window-less child with the same data dir; it polls
`/health` (2 s, no auth) and `/ready` (5 s, token from the env file, never shown or logged);
restart and stop are file requests the supervisor honours (`state/restart.request`, new
`state/stop.request` -- a console signal cannot reach a child that has no console), and the
stop applies only to a supervisor the tray launched. `install.ps1` points the Startup entry at
`SpeechForge Tray.bat`; `Start SpeechForge.bat` (console) is unchanged.

**Why.** The host already shows ClawChat, ClawForge and StudioForge as icons; a console
window for the fourth service was the odd one out, and "is it ready, is anything connected,
did the lease go" should be a glance, not a `curl`. Adopt-or-launch keeps one supervisor per
port whichever launcher started it.

## S24 -- One arena for every session (measured: off by default)

**Decision.** With `gpu.shared_arena: true` (the default) and the CUDA provider, `load_speaker`
registers one ONNX Runtime arena for the card before the first session is created --
`create_and_register_allocator_v2("CUDAExecutionProvider", OrtMemoryInfo("Cuda", ARENA, device,
DEFAULT), {device_id}, OrtArenaCfg{...})` -- and every session (Chatterbox's four, Parakeet's
through onnx-asr's `sess_options`, Smart Turn) sets `session.use_env_allocators = 1`. The arena's
first chunk is `gpu.arena_initial_mb` (default and ceiling 2047 MiB), it extends power-of-two (S22)
and is capped at twice `gpu.mem_limit_mb`. Registration is once per device per process; a refusal
logs one warning and the load continues with per-session arenas, which every session also keeps
in its provider options as the floor. `/ready` reports `gpu.shared_arena`.

**Why.** The cold load is 33.7 s against a 10 s target (`docs/BENCHMARKS.md`, "Open"): ~9 s of
session creation and ~20 s of warm-up that is believed to be six arenas each growing to their
working set. One arena with a large first chunk makes that ramp one allocation at load and lets
the sessions share free blocks instead of each holding its own high-water mark.

**What the runtime allows (checked against onnxruntime-gpu 1.31.0).** The plain
`create_and_register_allocator(OrtMemoryInfo, OrtArenaCfg)` refuses CUDA memory ("Only CPU devices
are supported. Please call CreateAndRegisterAllocatorV2() for other device"), so the CUDA path is
the v2 call with the provider name and a provider-options mapping. `OrtArenaCfg.initial_chunk_size_bytes`
is a 32-bit int: 2047 MiB is the largest first chunk the wheel accepts (2048 and 4096 MiB fail at
construction), hence the bound. There is no Python call to unregister an environment allocator.

**To be measured (`scripts/gpu_cold_load.py`, both `--shared` and `--no-shared`, fp16, the serving
card).** Cold load: session creation and warm-up time separately, against 9 s + 20 s. Footprint:
NVML used after warm-up (7.2 GB today) and whether it stays fixed over a soak. First audio: TTFA
p50/p90 and RTF through `speechforge bench` must be unchanged (190 / 226 ms, 0.23). Unload:
whether the card's used memory returns to its baseline after `close()` -- an environment arena
outlives the sessions, so if it keeps its chunks the default must become `false` for a tenant
that gives the card back (S3), or the unload must shrink it. The decision stands as "on by default"
only if the load gets faster and the footprint and TTFA hold.

**Measured 2026-10-11 (RTX 5090, fp16, `scripts/gpu_cold_load.py`, two runs each).** Shared arena:
sessions 12.4 / 12.9 s, warm-up 32.8 / 15.3 s, footprint after warm-up 6.7 GB, **after close 6.3 GB
(5.5 GB held)**. Per-session arenas: sessions 14.7 / 11.7 s, warm-up 35.9 / 21.1 s, footprint
7.9 GB, after close 1.4 GB. The shared arena saves ~15 % of warm-up and 1.2 GB while serving, but
ONNX Runtime cannot unregister an environment allocator, so the memory stays on the card after the
models unload -- which is exactly when a lease-returning tenant (S3) must give the card back.
**Default `gpu.shared_arena: false`**; the code stays behind the flag for a deployment that never
unloads (or that exits the process to unload). The cold-load answer remains the Open list in
docs/BENCHMARKS.md.

## S26 -- Smart Turn waits out a mid-sentence pause (2026-10-12)

**Decision.** The server owns the silence floor before Smart Turn is consulted. `eou.min_silence_ms`
(700 ms) is the least silence after which an `eou` verdict is produced, and the client's
`stt.config eou.min_pause_ms` is only a request: the effective trigger is `max(requested, floor)`,
so a client can raise it but never lower it. When the latest partial does not end in `. ? !` the
pause is a candidate mid-sentence breath and the longer `eou.unpunctuated_silence_ms` (1000 ms)
applies instead. The check re-evaluates on every VAD window, so a partial that catches up with a
full stop during the pause releases it early; the verdict is still sent once per pause and a pause
cut short by speech sends nothing. The `eou` and `stt.final` frames and the `_pause` event are
unchanged (`pause_ms` is the measured pause when the event fires); only when the event fires moves.

**Why.** Measured live on 2026-10-12 by the add-on: on medium and long utterances (8 s and up) a
natural pause inside a sentence got a high Smart Turn `p` at the 150 ms default, the box took it as
the end of the turn, and the model started answering half a question. The probe
(`scripts/eot_probe.py`, docs/BENCHMARKS.md) then showed why: Smart Turn is scored once, at the
start of the pause, so a 400 ms breath and a 1000 ms one get the same verdict (73 % of inserted
mid-sentence pauses scored `p >= 0.5`, 37 % `p >= 0.9`), and Parakeet puts a full stop at nearly
every clause boundary (89 of 90 mid-sentence partials ended in `. ? !`), so punctuation alone
separates almost nothing. The silence itself is the lever: a verdict waits until the pause has
outlasted a breath. 700 ms clears every 400 ms pause and about half the 700 ms ones in the probe
set and costs about 65 ms plus the floor on a clean end of turn (under a second, which is what the
fast path needs); 1000 ms covers the unpunctuated case the same way. Both are configuration, so the
host can trade latency for cut-offs once real speech has been measured. Keeping the floor on the
server means no client configuration can bring the false end of turn back.

## S27 -- A vacate waits for audio in progress, not for an open context

**Decision.** `TtsContext.producing` is True only while the context has work in hand: a unit
taken from the queue and not yet fully handled (rendered and flushed), queued units, buffered
text that a flush or close will cut into a unit, or a close whose `tts.done` is still to be
sent. `SessionManager.before_unload` waits (bounded by `drain_timeout_s`) only while some
context is `not done and producing`. An open context with nothing to say no longer holds the
drain; nothing else about the drain changed (a context still running when the bound expires is
stopped by the unload path, as before).

**Why.** A voice chat keeps a context open between sentences. Its producer thread polls for the
next unit, so `done` stayed False and every vacate cost the full 15 s of the bound for nothing,
while the owner's other tenant waited for the card. The unit is marked busy under the same lock
that pops it, so there is no instant at which a unit in flight looks idle.

## S28 -- A final does not wait behind its own partials

**Decision.** On `stt.flush` the session cancels its partial if the STT lane has not started
it, and holds off new partials until the final has been sent. The lane job carries a `Ticket`
that the worker claims before running; `Ticket.cancel()` wins only while nobody has, so a job
is either withdrawn whole or runs whole (a partial already decoding is left to finish, one
decode, ~85 ms). A withdrawn job is undone with `RollingStt.release_partial` -- the streaming
fed mark, or the tail start a commit had moved -- so the final decodes that audio itself and
the text is the same as without the withdrawal. `Session._final_pending` counts flushes in
flight; `on_audio` does not spawn a partial while it is non-zero, and a partial task spawned
just before the flush returns without taking a job. The cancelled future's `CancelledError` is
swallowed in `_partial` (no warning) and `_partial_pending` is still cleared. `stt_final_ms` is
unchanged.

**Why.** The lane is one worker with a priority queue. A final outranks a partial but cannot
pre-empt one, so on a long utterance with continuous partials the final could sit behind a
decode that was already queued for audio the final is about to cover anyway. The final is what
the box waits on; a partial that is stale the moment the flush arrives is worth nothing.

## S29 -- An unleased server asks for the lease again, and gives way when refused

**Decision.** A server that loaded without a lease because StudioForge was unreachable
(`on_unreachable: proceed_if_free`) retries the lease every `studioforge.relink_s` (default 30 s,
5-600) from `Residency.tick()`, on the card it is loaded on, with the same class-2 claim and vacate
callback as at load. Granted: it holds the lease from then on (`state{lease:"held"}`, touched like
any other). Refused with `lease_conflict` (the card is leased by or promised to an equal or better
class): it gives way the way a vacate does (`vacating`, bounded drain, unload with `gpu_vacated`,
the vacate cooldown), no lease id or token needed. Refused with anything else (`lease_vacating`,
`model_busy`: StudioForge's own transient codes): it keeps serving and asks again. Still
unreachable: nothing changes and the next try is `relink_s` later. A stale lease under our own
holder is released before a refusal is believed. `lease_mode: off` never retries and never gives
way.

**Why.** An unleased server was invisible to StudioForge, so a vacate request could never reach it
and a priority-1 load had to wait for our idle unload or fail against a card we still held. The
voice client treats `unleased` as serviceable on the assumption that the server still gives way to
such a load; this makes that true instead of hopeful. The cost is a bounded delay (one retry
interval after StudioForge returns) and, when StudioForge refuses, a reload for the voice client's
CPU fallback window; both are cheaper than holding a card another tenant has been promised.

## S30 -- The server cleans a reference clip before the encoder sees it

**Decision.** `POST /v1/voices/from-reference` (and the CPU bench voice) now run the upload
through `audio.reference_clip`: a band-limited polyphase resample to 24 kHz
(`resample_poly`, Kaiser-windowed sinc, cutoff at the lower Nyquist, about 90 dB of alias
rejection; `resample_linear` stays on the realtime paths), then `prepare_reference`: DC removal,
leading/trailing silence trimmed by a 20 ms short-time RMS gate 40 dB under the clip's own
loudest frame (50 ms margin kept), and one gain to -20 dBFS RMS capped so the peak never passes
-1 dBFS. The 3-30 s check applies to the trimmed clip. The response carries a `clip` object
(trim, gain, levels, `warnings`: `short_clip`, `quiet_clip`, `clipping`, `long_lead_silence`,
`silent_clip`). `voices.prepare_reference` (default true) turns the cleaning off (the resample
stays); `voices.reference_target_dbfs` (default -20, range -30..-10) sets the level.
`ChatterboxTts.build_voice` still takes raw audio and, at another rate, resamples with
`resample_poly`; the route prepares, the engine does not.

**Why.** A cloned voice is the encoder's view of the clip. A lead-in of room tone, a DC offset,
a level far from the training range or a 48 kHz recording folded down by an aliasing resampler
all land in the conditioning, and the owner hears them as uneven cadence and blips in every
sentence afterwards. Cleaning is cheap, deterministic and identical for every client, so it
belongs on the server rather than in whatever uploads the clip. Warnings tell the owner what to
re-record instead of silently shipping a bad voice.
Not done, on purpose: noise reduction and a gate inside speech (either alters the timbre the
encoder is reading, and a clean recording does not need them); compression or limiting beyond
the peak cap (the gain is a pure scale, so the dynamics are the speaker's); cutting to a fixed
length (the engine knows what it uses; Qwen3 and dots cut their own); loudness (LUFS)
weighting or spectral EQ (no evidence they help, and they cannot be undone); refusing a clip on
a warning (the voice is still built, and the 3-30 s limit is the only hard gate).

## S31 -- Voice quality: the decoder reads the whole 10 s prompt; units are sentences

**Decision.** `tts.prompt_s` defaults to 10 s (was 4): the decoder is conditioned on everything
the profile carries (the add-on's profiles are exactly 10 s; `cond_emb` for the LM was never
sliced). A TTS unit is cut at a sentence end (`. ! ?`) and no longer at `;` or `:`; a full stop
right after a digit with nothing following, after a common abbreviation or after an initial waits
for more text (`tts_ctx.unit_ready`). `final`, `tts.flush` and `tts.close` cut as before. Left
context, lookahead, the 20 ms seam crossfade, the 40-token decoder bucket, fp16 and the sampling
knobs stay as they were.

**Why.** Measured on 2026-10-12 (docs/BENCHMARKS.md, "Voice quality: chunking, precision and
prompt"): 13 variants x 6 lines x 2 seeds on an idle RTX 5090 under a probe lease, with
objective seam and similarity metrics, after the owner reported uneven cadence and blips.
The seams are not the blips: the envelope dip at a chunk seam is within 0.2 dB (median) of the
same instant in a whole-sentence render, the spectral jump at seams is below that at random
positions, and neither a 50 ms crossfade nor a phase-aligned one changes any metric. fp16 and
fp32 are indistinguishable (ECAPA cosine 0.91 between them, the same as two seeds of either).
The 10 s prompt costs nothing (decode 74 vs 75 ms per chunk, TTFA 254 vs 244 ms, inside noise)
and raises speaker similarity to the whole render from 0.909 to 0.934 against a decoder-noise
ceiling of 0.941. A longer left context (16 / lookahead 6) forces the 64-token decoder bucket,
and every switch between the 40 and 64 shapes costs a 300-530 ms re-shape: 9 of 12 sentences
then had a playout gap. What does move cadence is structural: every unit is an independent LM
call with its own prosody and a 120 ms silent tail, so cutting at `;` and `:` (and at a decimal
point or "Dr.") produced fragments that read as uneven pacing. The only other cadence risk the
measurement found was the LM slowing from 8 to 46-65 ms per token while the decoder ran, in
the probe's first renders on a cold engine; the live host warms every LM worker at load and its
`tts_step_ms` histogram shows 98 % of steps under 20 ms, so it is recorded here, not acted on.

## S33 -- Drama 3 is an add-on engine built from the port's own tooling (2026-10-15)

**Context.** Fish Audio released Drama 3 on 2026-10-13: a 35B-A3B MoE TTS model with
direction tags and clip cloning, officially served by SGLang-Omni on Linux with 48 GB of
VRAM. A community audio.cpp port (`drama-3-native`) runs a 23 GB mixed-quant GGUF natively on
Windows with CUDA and streams. Measured on the host (BENCHMARKS): first audio 185-234 ms, RTF
0.35 on a 5090 -- the Chatterbox class -- with cloned voices at ~1 s because the server
re-encodes the reference every request.

**Decision.**

- *Nothing vendored.* The port (Apache-2.0) and the weights (Fish Audio Community License
  v1.0: free below USD 500k revenue, no hosted service for third parties, consent for
  cloning, "Built with Fish Audio") stay out of the repository. `speechforge engine-setup
  drama3` clones the port at a pinned commit, builds it with its own `build_windows.ps1` /
  `build_linux.sh`, and installs the package with its own `model_manager_v2.py`, then
  sha256-verifies the file against the pin. `models.lock` does not list it (`tts_groups`
  answers `[]`); the port's model manager is the source of truth for its files.
- *The server is the server's.* `backend: spawn` writes the documented config shape and
  launches `audiocpp_server` on the leased card; `backend: http` attaches to one the operator
  runs. The adapter speaks the documented `/v1/audio/speech` streaming request and nothing
  private to the port.
- *One clip-voice container.* `dotstts-ref` (clip + transcript) is reused rather than a
  fourth format; `max_ref_s` (6 s) bounds the per-request encode.
- *Exact resampling.* 44.1 kHz to 24 kHz through `PolyphaseResampler`, a streaming form of
  `audio.resample_poly` proven sample-for-sample equal to the whole-clip result.
- *Not the live engine.* Chatterbox stays on the 3090; Drama 3 needs a 32 GB card or
  `cpu_expert_layers`. The gates, the listening test and a reference-code cache are the
  next steps before a switch.

## S34 -- A Drama 3 voice is its codes; clips are encoded once, takes are kept (2026-10-15)

**Context.** Cloned voices on the Drama 3 engine paid ~0.9 s before first audio. Three agents
established why: the port's codec encode is cached and cheap; the cost is the backbone
pre-filling every reference row again on every request (~2.4 ms per row, ~55 ms per second
of clip), and the official runtime does the same (prefix caching off). The model has no
built-in or seed-stable voice: with no reference it improvises one per take.

**Decision.**

- *The voice object is the VQ grid plus the transcript.* The `dotstts-ref` container gains
  an optional `vq_codes` int32 `(10, frames)` array; a clip file is encoded once per process
  through the server's `encode_reference_only` task (a server-side path, written for the
  call and removed) and cached per voice id; a codes file needs no encode. Every request
  carries `options.references: [{text, vq_codes}]`, never audio: the encoder is skipped and
  the body shrinks from ~800 KB to ~14 KB. The transcript stays mandatory.
- *Keep this voice.* `take()` on the engine and `POST /v1/voices/surprise` on the server:
  one take in an improvised voice through the port's task route, the take's own codes stored
  as a codes-only voice with a 24 kHz preview, the id returned. A kept 2-3 s take speaks in
  ~300-350 ms, which is the practical way to a fast cloned-sounding voice today.
- *The latency knob is reference length,* so `max_ref_s` stays at 6 s (~450 ms) and the
  docs state the 110 ms + 55 ms/s model instead of promising ~200 ms.
- *Not done here:* the per-voice backbone prefix cache in the port (saved state after the
  reference rows, suffix-only pre-fill) is the one change that reaches ~200 ms with long
  references; it is a C++ patch to audio.cpp, documented as the lead in ENGINES.md.

## S35 -- Build the port with CUDA 13.1: the toolkit's top-k was the per-row cost (2026-10-15)

**Context.** S34 attributed the cloned-voice latency to the backbone pre-filling every
reference row at ~2.4 ms per row. A profile (`nsys`) showed the GPU idle most of that time:
under CUDA 13.4 the port's MoE router (`ggml_top_k`) compiles to CUB's top-k, which the port
calls once per row per layer from the host -- 62.6k launches for one 371-row pre-fill.

**Decision.** `engine-setup` builds the port with the CUDA 13.1 toolkit when it is installed
(`CUDACXX` + `CUDA_PATH` + `CUDAToolkit_ROOT`; CMake picks the compiler from `CUDACXX`, the
others are not enough), says so when it is not, and accepts `--cuda-toolkit` / `--cuda-arch`.
Same source, no patch: pre-fill 0.15 ms per row, a 12.6 s reference to first audio in
~130 ms, 6 s in ~100 ms, seeds reproducible. The host's config points at the 13.1 build.
`max_ref_s` stays at 6 s (the knee is now similarity, not latency; raise it when the
listening test says so). The change to offer the port upstream is `ggml_argsort_top_k` at
`backbone.cpp:413`, which would make every toolkit equally fast; the per-voice prefix cache
of S34 is now worth ~60-90 ms and was not written.

## S36 -- The per-voice prefix cache ships as a patch to the port (2026-10-15)

**Context.** After S35 a repeated cloned request still re-pre-filled its reference rows
(136-152 ms to first audio for 12.6 s). An agent wrote the prefix cache sketched in S34:
split the prompt at the end of the reference blocks, keep the backbone and audio-tower state
after that prefix on the device per voice, pre-fill only the suffix on a hit. Measured: 79 ms
at the server, 95-107 ms through the adapter, byte-identical audio cold vs cached, ~85 MiB
per voice, ~50 ms more on a voice's first request.

**Decision.** Ship it as `engines/patches/audiocpp-drama3-prefix-cache.patch` (SpeechForge's
own code, MIT, on top of the Apache-2.0 port, pinned to `AUDIOCPP_COMMIT`); `engine-setup`
applies it with `git apply` before building (`--no-prefix-cache` for the stock port), the
generated server config carries `fish_drama3.prefix_cache_entries` only when
`tts.drama3.prefix_cache` is true (a stock server refuses the option), and the host runs the
patched 13.1 build. Offer the patch upstream together with the `ggml_argsort_top_k` change;
when the port carries both, the shipped patch and the toolkit preference go away. Not done:
a single-pass pre-fill that snapshots mid-way (would remove the cold penalty), a memory
budget instead of an entry count, snapshots across restarts.

## S37 -- from-reference builds a Drama 3 voice: the rig transcribes the clip (2026-10-15)

**Context.** A Drama 3 voice needs the clip's words, and the clips that exist -- the DisPatch
voice add-on's `reference.wav` per profile, the owner's recordings -- come without a transcript.
The add-on uploads Chatterbox conditionals today, which carry neither the waveform nor the words,
so nothing it sends can become a Drama 3 voice.

**Decision.** `POST /v1/voices/from-reference` does the whole job for an engine that
`needs_transcript`: the caller may send `X-Reference-Transcript`; otherwise the rig transcribes
the prepared clip with the STT it has loaded anyway (Parakeet, resampled to its 16 kHz), then the
engine cuts the clip to `max_ref_s`, encodes it once and stores codes + transcript + clip. The
reply says which (`transcript_source`). A clip with no recognisable words is `bad_audio`. On the
box side the add-on needs one change to use it: when the rig's `ready.engines.tts.voice_format`
is not `chatterbox-npz`, POST the profile's `reference.wav` to from-reference (under the owner's
D3 opt-in, already on for this rig) instead of PUTting `voice.npz`, and remember the returned id.
