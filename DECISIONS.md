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
