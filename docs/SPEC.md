# SpeechForge: design spec

**A GPU speech server on the rig that makes DisPatch voice chat feel like talking to a person.**
Independent review and design, 2026-10-10. Status: design. Nothing here has been built, installed or measured on the rig.

Inputs: the original change log proposing the rig offload ("the changelog"); the voice chat v3 design notes (`SPEC.md` and `CONTRACTS.md`); StudioForge `DECISIONS.md` D43/D56; the first SpeechForge cut (read only, no commits yet); live checks from the box (`sfctl status`, curl RTTs) on 2026-10-10; and four web research passes (STT and turn-taking, TTS, speech-to-speech, runtimes), with full notes kept in the add-on's design notes.

Confidence tags: **[M]** measured on this box or rig. **[P]** published by a third party. **[V]** vendor claim. **[E]** my estimate. **[U]** unverified and flagged.

---

## 0. The answer in one screen

| # | Decision | Why (the number that decided it) |
|---|---|---|
| 1 | **One persistent WebSocket session per voice chat** between the box add-on and SpeechForge (`/v1/realtime`, protocol `sf.1`). Audio streams in continuously and audio streams out. STT partials and finals, end-of-turn scores, word timings, cancel by generation, credits and pings all travel on that one socket. The per-request HTTP routes stay only as utilities. | Per-request HTTP cannot stream partials, accept text incrementally, cancel cleanly or detect a dead rig in under 1 s. Every realtime voice stack runs a persistent bidirectional stream per session: OpenAI Realtime, Kyutai Unmute, Deepgram, ElevenLabs, Cartesia [P]. |
| 2 | **Use the GPU on the real bottleneck, the end-of-turn decision, not just the tail transcript.** The rig transcribes continuously and runs Smart Turn v3.2 (GPU build) at every pause of 150 ms or more. The box's TurnPolicy stays the authority. | Today's decision lands **0.55–0.92 s** after speech end [M, T4 harness], because it waits for a CPU tail transcript. On a GPU the tail costs about 10–30 ms [E] and Smart Turn 1–2 ms [P], so a confident turn can end at **≈ 0.25–0.35 s**. |
| 3 | **STT: Parakeet TDT 0.6B v2 stays.** On the GPU it re-decodes the open tail every 160 ms ("rolling partials") and decodes the full utterance for the final transcript. **NVIDIA Nemotron Speech Streaming EN 0.6B** (true cache-aware streaming, 80–1120 ms chunks) is the A/B challenger for partials. | Parakeet v2 is still the best English model at its size on the re-cleaned HF Open ASR Leaderboard: 6.43 % average WER, 11.42 % on the noisy set, RTFx 6,148 (results file updated 2026-10-09) [P]. Nemotron: 6.89 / 12.54 %, and it streams [P]. The rolling re-decode needs no new model and no new licence. |
| 4 | **TTS phase 1: Chatterbox-Turbo on the GPU (fp32, ONNX).** It is the same model and the same voice conditionals as the box's CPU fallback, so the voice doesn't change when the rig drops out. **Phase 2: a bake-off.** Qwen3-TTS-12Hz-1.7B-Base (Apache-2.0) against dots.tts-mf (Apache-2.0) against Chatterbox-on-GPU, using the owner's voice and blind family listening. | Chatterbox-Turbo is no longer the best open model on published cloning numbers. **Qwen3-TTS** has first audio at 156–174 ms on an RTX 4090 with native Windows support [P] and English WER 1.24 [P]. **dots.tts** has the best open speaker similarity (76–77) and accepts streamed LLM text [P]. But Chatterbox is MIT, already integrated, has laugh/sigh tags, and is the only one with **voice parity with the CPU fallback**. |
| 5 | **Card: GPU 3 (RTX 3090), a whole-card StudioForge lease, class 2, with a vacate URL**, held while a voice chat is open plus 15 idle minutes. No co-tenancy with llama.cpp on the same card. | GPU 3 was the least-used card at design time, and the other 3090 is held by a pinned image-rendering workload [M]. Windows has no MPS. Two processes time-slice a card under WDDM, and stream priority does not cross processes [P], so a co-tenant LLM would add jitter to every TTS step. |
| 6 | **Network: the tailnet address (`<rig-tailnet-ip>`), not the LAN.** | Health over the tailnet measured **7–14 ms total, 200 OK** [M]. The LAN path to `:1234` timed out **10/10** [M], while another rig service on the LAN answered, so the rig's firewall is port-scoped. Plain HTTP on the LAN would carry the bearer token and family audio unencrypted; the tailnet is WireGuard. |
| 7 | **Native Windows, Python 3.12, `onnxruntime-gpu[cuda,cudnn]>=1.28,<1.32`, driver R580 or newer**, started at logon like StudioForge (not a Session-0 service), with blue/green venvs for rollback. | The ORT 1.23.0 sm_120 bug was fixed in 1.23.2 [P]. 1.27+ is the CUDA 13 build, and 1.28 fixed the CUDA 13 DLL preload on Windows and a Blackwell kernel [P]. Latest is 1.31.0 (2026-10-09) [P]. WSL2 brings no latency gain [P]. |
| 8 | **Targets (advisor bot, cloned voice, Wi-Fi, rig healthy):** speech end to first reply audio **p50 ≤ 1.6 s, p90 ≤ 2.2 s** in phase 1, and **p50 ≤ 1.1 s** in phase 3. First sound (ack or preamble) **≤ 0.7 s**. Barge-in heard as quiet in **≤ 200 ms** (plus Bluetooth latency). | Today **3.50 s** median in power-saver, ≈ 2.8 s with the performance hold [M]. The LLM's first sentence, about 1.2–1.3 s after submit [M], is now the dominant term, so a GPU alone cannot reach "sub-second". Speculation and early first-clause TTS are what close the gap (§5). |
| 9 | **Full-duplex speech-to-speech is optional (phase 4) and a separate mode.** The leading open candidates are **NVIDIA PersonaPlex-7B** (Moshi-based, 0.17 s turn-taking, NVIDIA Open Model License) and **MiniCPM-o 4.5** (Apache-2.0 per its card [U]). Each runs via WSL2 on a 5090 and has no tools or memory. | None of these models can speak the bots' own words verbatim, and none has native Windows support [P]. OpenAI's own newest voice product (GPT-Live, 2026-07/09) puts a fast voice front-end on top of a delegated reasoning backend [P]. That is the cascade we are building. |

---

## 1. Audit of the 2026-10-10 changelog

The changelog gets the big call right: a separate rig process that is a lease tenant like the rig's pinned image-rendering workload, with the CPU always kept as a fallback. It is weak on everything that makes a voice chat feel live. Every gap is listed below, most important first, followed by what this spec changes.

### 1.1 Gaps, ranked

| # | Gap | Concretely | What this spec does |
|---|---|---|---|
| 1 | **Protocol: per-request HTTP** | `POST /v1/audio/transcriptions/pcm` per tail and a chunked `POST /v1/audio/speech` per sentence. That gives no partials while the user talks, no incremental text into TTS, and no cancel except dropping the TCP stream. A dead rig is found through a 5 s cached `/health` plus 3 failures, which is up to about 15 s of slow turns. | §4: one WebSocket session per voice chat, cancel by `gen`, 250 ms pings, dead in ≤ 1 s. |
| 2 | **Latency table arithmetic and baseline** | "Total on the rig ≈ 1.3–1.8 s", but the rows' own minimums sum to **1.46 s** (0.3 + 0.03 + 1.0 + 0.12 + 0.01) and the maximums to 2.85 s. The "CPU today ≈ 2.4–3.0 s" baseline is lower than the measured **3.50 s** (power-saver, Chatterbox) and ≈ 2.8 s (performance hold) [M, T4 harness]. | §3 rebuilds the budget from measured stages. |
| 3 | **The GPU is aimed at the wrong stage** | The changelog says end-of-turn "stays local, unchanged". But the decision is slow (0.55–0.92 s) precisely because it waits for the CPU tail transcript (`at_pause()` 0.25–0.52 s) before the policy may end the turn. Speeding up only the final transcript saves about 0.3 s; moving partials and Smart Turn to the GPU saves about 0.5 s more. | §5.1–5.2. |
| 4 | **Barge-in and cancellation on the rig undefined** | No generation id, no "stop within N ms" promise, no flush, no pause/resume of a reply under a false interruption, and **no word timestamps** from TTS, which v3's heard-text tracking needs to cut at what was actually heard. | `cancel {gen}` with ≤ 50 ms GPU stop, `tts.words` timings, credit-based lead (§4.4). |
| 5 | **Network path and transport security** | The default `rig.url` is `http://<lan-ip>:<port>`. From the box, LAN `:1234` timed out 10/10 while another rig port answered [M], so the port needs a firewall change nobody listed. On the LAN the bearer token and family audio go in cleartext. "Accepted only from LAN or tailnet" is not a mechanism: there is no bind address and no firewall scope. | Tailnet address by default. Bind to the tailscale IP and loopback, with a firewall rule scoped to the box (§8). |
| 6 | **Audio formats and framing unspecified** | "PCM16 at 24 kHz or Opus" with no frame size, endianness, header, sample clock, resampling rule, or how a chunk maps to a sentence or generation. | §4.3: fixed binary header, 16 kHz in / 24 kHz out, 20 ms in, 40 ms out, sample offsets. |
| 7 | **Hedging can be slower than CPU-only** | STT: a 250 ms deadline, *then* the CPU tail (0.3–0.7 s), so a timed-out turn costs up to about 0.95 s for the tail. TTS: a 400 ms deadline, then the CPU first chunk (0.7–1.3 s), so up to about 1.7 s. Each turn pays the deadline again until health marks the rig down. | Session-level health from pings and partial lag. The fallback decision is made **before** the turn needs it, with a per-reply sticky engine (§6). |
| 8 | **Model provenance** | No revisions, file hashes or export provenance in the changelog. The first cut does pin HF revisions (`d21799bd…`, `0bbb45a3…`), which is good, but not per-file sha256. "fp16 replacing q4" is unvalidated: S3Gen and HiFiGAN fp16 numerics can buzz or clip [E]. | `models.lock` with a sha256 per file, verified at load. fp16 is admitted per graph only after a golden-audio check (§7.1). |
| 9 | **Windows/CUDA specifics missing** | No driver minimum, no ORT version window (1.23.0/1.23.1 have no usable sm_120 code; onnx-asr excludes 1.24.1, 1.25.*, 1.26.0), no CUDA 13 wheel notes, no Windows 10 driver end-of-life (NVIDIA's last Win10 Game Ready driver is this month [P]), and nothing on WDDM launch overhead (fixed with CUDA graphs) or service vs logon session. | §10. |
| 10 | **Concurrency and backpressure contradictory** | The changelog says one STT and one TTS at a time, with a 300 ms queue, then 503. The first cut has `max_streams: 3` and `stt.max_concurrent: 4`. Neither limits how far TTS runs ahead of playback, so an interrupted 40-second answer burns GPU on audio nobody hears, and TCP is the only backpressure. | Per-session credit (default 3 s lead). A global cap of 3 sessions and 1 active TTS context per session (§4.4, §8). |
| 11 | **Privacy claim inconsistent with the first cut** | The changelog says "your voice clip never leaves the box", but the first cut ships `POST /v1/voices/from-reference`, which takes a 3–30 s WAV. Voice conditionals are biometric-grade data stored at rest on Windows, and nothing covers crash dumps, backups, Search indexing or OneDrive. | §8.4. The reference clip goes to the rig only if the owner allows it (decision D3). |
| 12 | **Observability thin** | Timings in logs and Voice lab medians only. No per-turn trace id across box and rig, no metrics endpoint, no GPU step p99, no VRAM drift, no lease-state history. | §9. |
| 13 | **Tests have no acceptance numbers** | Only "a latency script reports first-chunk times". No WER gate, no soak, no co-tenancy, no chaos tests (vacate mid-sentence, rig killed mid-stream, network partition), no two-session test. | §11 thresholds. |
| 14 | **Lease details** | The changelog doesn't say which class to claim (the first cut uses 2, correctly) or what happens when a voice chat opens while the card is leased by a benchmark (benchmarking tooling never vacates, D56 item 5). `on_unreachable: proceed` fails open, which is right for an image-rendering tenant but needs a VRAM check first. | §7.3. |
| 15 | **Update and rollback** | "git pull + `uv sync`, roll back to the previous tag if `/health` fails". `/health` proves the process started, not that STT transcribes or TTS speaks. A pulled tag with re-downloaded models has no atomic switch. | Blue/green venvs plus a synthetic self-test gate (§10.4). |
| 16 | **Voice consistency on fallback** | fp16 GPU Chatterbox and q4 CPU Chatterbox differ audibly [E]. A phase-2 engine change would make the fallback sound like a different person. | Engine stickiness per reply and per session. The parity rule is §6.3. |
| 17 | **Install path is a blocker, not a footnote** | The box has no SSH to the rig, so first install needs the owner. | §10.2: a single `install.ps1`, then self-update. |
| 18 | **No browser-leg echo note** | Chrome's echo cancellation may not reference Web Audio (AudioWorklet) playback [U], so the bot can hear itself. This is outside SpeechForge, but it caps how aggressive barge-in can be. | §5.6 and decision D9. |

### 1.2 What the changelog got right (kept)

- SpeechForge is a separate process: not inside StudioForge, a lease tenant like any other rig workload, MIT, author "Laser Lloyd".
- "Voice never waits on the rig." The CPU path is always installed. `503` with `Retry-After` is never a hang.
- The lease lifecycle: warm on chat open, touch while busy, idle unload, and D56 vacate.
- Voices are content-addressed by sha256, idempotent `PUT`, `HEAD` to skip the upload.
- No transcripts or reply text in logs. Audio stays in memory.
- DisPatch itself needs no change; it all lives in the add-on.

---

## 2. Goals, non-goals, numeric targets

**Goals**
1. A turn feels like talking to a person: a reply starts within about a breath, you can cut it off, and it resumes politely when you didn't mean to.
2. The brain stays the existing bots: the primary assistant model and local tool-using agents. SpeechForge only hears and speaks.
3. The box runs cool. While the rig serves, the box does VAD, session logic and Opus, and nothing heavier.
4. Voice chat never gets worse because the rig exists. With the rig gone it behaves exactly like v3 on the CPU.

**Non-goals (this spec):** a public endpoint; multi-tenant hosting; languages other than English (the UI says "English only"); replacing the phone↔box transport (WebRTC is noted as a DisPatch-side future item); making SpeechForge part of StudioForge.

**Targets.** "Phase 1" is what must hold to ship; "phase 3" is the stretch. All are measured by the harness in §11 on the live box against the rig.

| Metric | Today (measured) | Phase 1 target | Phase 3 target |
|---|---|---|---|
| Speech end → first reply audio, advisor, cloned voice, Wi-Fi | 3.50 s p50 (power-saver); ≈ 2.8 s (hold) | **p50 ≤ 1.6 s, p90 ≤ 2.2 s** | **p50 ≤ 1.1 s, p90 ≤ 1.6 s** |
| Same, gateway agent (3–15 s think) | first sound 0.6 s ack | first sound **≤ 0.7 s**; spoken tool narration as today | same |
| Speech end → end-of-turn decision, confident turns | 0.62–0.92 s p50 | **p50 ≤ 0.35 s, p90 ≤ 0.55 s** | p50 ≤ 0.30 s |
| False end-of-turn (cut-offs) | not tracked | **≤ 4 %** of turns on the family corpus | ≤ 3 % |
| STT partial cadence / lag behind audio | ≤ 2/s, tail lag 0.25–0.5 s | every **160 ms**; lag **p90 ≤ 250 ms** | lag p90 ≤ 200 ms |
| Final transcript after the decision | `final()` ≤ 300 ms | **p90 ≤ 60 ms** | same |
| Final WER, clean family speech / car noise | 0.0 on Opus test clips | **≤ 6 % / ≤ 12 %** on the §11 corpus | — |
| TTS: text at rig → first audio out | 1.27 s (2 GHz) / 0.69 s (hold) | **p50 ≤ 150 ms, p90 ≤ 250 ms** (3090, warm) | p50 ≤ 100 ms |
| TTS real-time factor | ≈ 0.5–1.0 | **≤ 0.25** (≥ 4× real time) | ≤ 0.15 |
| Barge-in: onset → reply inaudible at the phone | ≤ 150–200 ms + BT | **≤ 200 ms + BT output latency** (phone duck, unchanged) | same |
| Barge-in: `cancel` received → rig GPU idle for that gen | n/a | **≤ 50 ms** | ≤ 30 ms |
| Onset → word-confirmed interruption verdict | 0.45–0.9 s | **≤ 0.45 s** (streaming partials) | ≤ 0.35 s |
| Rig failure → session on CPU | up to ~15 s of slow turns | **≤ 1.0 s** detection, no turn waits longer than its deadline | same |
| Cold bring-up (chat open → models ready) | n/a | **≤ 10 s** (overlaps the first sentence) | ≤ 6 s (TRT/graph caches) |
| Box CPU while the rig serves | 46–100 % of a core per call | **≤ 15 % of one core** per session (VAD + Opus + hub) | same |
| Underruns over a 30-min soak | 0–1 per run | **0** | 0 |

Why these and not "sub-second": the measured LLM leg (decision → first sentence ready) is **1.2–1.3 s** [M]. A human gap is 0–200 ms [P, Stivers 2009], and production cascades aim for 500–800 ms voice-to-voice with 0.8–1.2 s acceptable [P]. Phase 1 hits "acceptable" on speech; the stretch needs the LLM's first clause sooner (speculation on stable partials, first-clause TTS, a measured time-to-first-token for the assistant model) and is reported, not promised.

---

## 3. Latency budget

Advisor bot, cloned voice, Wi-Fi, rig healthy and warm. Phase-1 values are [E] built from [M]/[P] parts and are replaced by harness numbers before ship.

| Stage | Today, CPU (hold) [M] | Phase 1 rig | How |
|---|---|---|---|
| Phone → box audio | ≈ 50 ms | ≈ 50 ms | unchanged (Opus, 100 ms messages; 20 ms in a later client change) |
| Box → rig hop | — | 4–10 ms | tailnet direct; persistent socket, no handshake per turn |
| Partial covering the last word | 0.17–0.31 s (`at_pause`) | ≤ 0.2 s after the word | rolling Parakeet re-decode every 160 ms |
| End-of-turn decision (confident) | 0.45–0.6 s | **0.25–0.35 s** | Smart Turn GPU at a 150 ms pause, rig-side; the box policy decides on `eou` + partial |
| Speculative LLM start | at the decision | **at the first `eou ≥ 0.5` on a stable partial (≈ 0.2 s)** | existing Speculator; earlier and cheaper signal |
| LLM first clause (≤ 48 chars) | ≈ 1.2–1.3 s after submit (sentence) | ≈ 0.9–1.2 s [E] after start | first-clause cut (existing), streaming tokens |
| TTS first audio | 0.6–0.9 s | **0.10–0.25 s** | GPU Chatterbox, 15-token first chunk, voice-prefix KV cache (T4 already did this on CPU: prefill 531 → 98 ms) |
| Rig → box → phone | ≈ 50 ms | ≈ 60 ms | 40 ms frames; box Opus encode 0.3 ms/frame [M] |
| **Total p50** | **≈ 2.8 s** | **≈ 1.35–1.65 s** | |
| Phase 3 additions | | **−0.2 to −0.5 s** | first-clause TTS during the hold, rendered speculatively and released on commit; LLM start on the 150 ms pause; incremental text into TTS; a 5090 if decode-step bound |

**First sound** comes from the existing ack pool when the first-audio ETA is over 1.2 s. With the rig the ETA often drops under that gate, so acks fire less. That is intended: a real reply beats a filler.

---

## 4. Architecture and the streaming session protocol

### 4.1 Picture

```
PHONE (PWA)  ──/ws/voice v3 (Opus up/down, say frames)──►  BOX the DisPatch box (DisPatch + DisPatch_Voice add-on)
                                                           │  session hub (TurnPolicy, Speculator, acks, heard, transport)
                                                           │  engines:  rig (primary)  |  cpu (fallback: Parakeet int8,
                                                           │            Smart Turn CPU, Chatterbox q4, Kokoro plugin)
                                                           │  Silero VAD stays here (gate, duck, barge-in onset, ~0.2 ms/frame)
                                                           ▼
                        wss/ws over tailnet <rig-tailnet-ip>:1236  /v1/realtime  (sf.1, one socket per voice chat)
                                                           ▼
RIG (Windows 11, native)  SpeechForge (one process, single uvicorn worker)
   session mux ─┬─ STT lane  (CUDA stream A): Parakeet TDT v2 rolling partials + finals   [Nemotron streaming: A/B]
                ├─ EOU lane  (stream A): Smart Turn v3.2 GPU on every ≥150 ms pause
                ├─ TTS lane  (CUDA stream B): Chatterbox-Turbo LM (CUDA graph, KV cache) → S3Gen → HiFiGAN, 40 ms frames
                └─ residency: StudioForge lease (class 2, vacate URL) on GPU 3, warm/idle/vacate, self-test
   /metrics (Prometheus text, no content)  /health  /ready   HTTP utilities (OpenAI-compatible, not the realtime path)
StudioForge :1234 (lease book D43/D56)   pinned image rendering (GPU 2)   llama.cpp models (GPUs 0/1 mostly)
```

Ownership rule: **the box decides, the rig computes.** The rig never decides that a turn is over, never calls an LLM, never stores text. It emits signals (`partial`, `vad`, `eou`, `final`, `words`) and audio. The add-on's TurnPolicy, Speculator, barge-in classifier and heard tracking stay exactly where v3 put them. The rig engine is an adapter behind the existing `StreamingSTT` and `PipelinedChatterbox`-shaped seams (C-§B.3), so the hub barely changes.

### 4.2 Why WebSocket and not gRPC, WebRTC or HTTP

| Option | For | Against | Verdict |
|---|---|---|---|
| HTTP per request (changelog) | trivial, OpenAI-compatible | no partials, no incremental text, coarse cancel, a handshake per call, health by polling | **utility only** |
| **WebSocket, binary + JSON frames** | full duplex, one auth at upgrade, the same stack as the add-on (uvicorn/websockets), the pattern Kyutai/Unmute, Deepgram, ElevenLabs and Cartesia use [P] | TCP head-of-line blocking (irrelevant on a 5 ms tailnet hop) | **chosen** |
| gRPC bidi | typed schemas, flow control (Riva/NIM use it [P]) | awkward with asyncio on Windows, an extra codegen toolchain, nothing else on the box speaks it | no |
| WebRTC box↔rig | jitter buffer, packet-loss concealment | a server-to-server LAN hop doesn't need media processing; heavy | no (it **is** recommended for phone↔box later, §5.6) |

Encoding: control messages are JSON text frames (small, debuggable, schema-checked); audio is binary frames with a fixed header. msgpack was considered and rejected, because it saves nothing that matters at these rates.

### 4.3 Wire protocol `sf.1`

**Connect.** `GET /v1/realtime` upgraded to WebSocket, `Authorization: Bearer <SPEECHFORGE_TOKEN>`, `Sec-WebSocket-Protocol: sf.1`. The server refuses any request carrying an `Origin` header (no browser may connect), refuses a missing or wrong token with 401 before upgrade, and refuses beyond `max_sessions` with 503 `Retry-After`. Max message size is 64 KiB. Compression is off for binary frames.

**Binary frames** (little-endian):

```
offset size  field
0      1     magic 0x53 ('S')
1      1     kind: 0x01 mic PCM in · 0x02 TTS audio out
2      2     ctx   (u16; 0 for mic)            — TTS context id
4      2     gen   (u16)                        — box generation; mic = 0
6      4     seq   (u32)                        — per-ctx frame counter
10     4     pts   (u32)                        — first-sample index: mic @16 kHz from session start; TTS @24 kHz from ctx start
14     2     n     (u16) sample count
16     2n    PCM16 mono
```

- Mic in: 16 kHz mono PCM16, **20 ms frames (320 samples)**, about 256 kbps. That is trivial on the tailnet; Opus on this hop costs CPU on both ends for nothing. The box resamples to 16 kHz once, as today.
- TTS out: 24 kHz mono PCM16, **40 ms frames (960 samples)**. Phase 2 engines with other native rates (dots.tts is 48 kHz) are resampled on the rig, so the wire format never changes.
- `pts` is the shared clock. `stt.final.upto`, `eou.at` and word times are in the mic sample clock, which is v3's `upto` clock, so heard and replay logic is unchanged.

**Control frames, box → rig**

| Frame | Fields | Rules |
|---|---|---|
| `hello` | `proto:"sf.1"`, `client:{name,version}`, `resume?:{session, token}`, `caps:{opus_out:false}` | first frame; the reply is `ready` |
| `stt.config` | `lang:"en"`, `partial_ms:160`, `eou:{enabled, min_pause_ms:150}`, `hotwords?:[≤64 strings]` | re-sendable |
| `stt.flush` | `id` | force the rig to decode now (end of turn or tail): it answers `stt.final{id}`; the "flush trick" so no right-context waits [P, Kyutai] |
| `stt.reset` | — | drop the open utterance (merge/undo paths) |
| `tts.open` | `ctx`, `gen`, `voice:"sha256:…"`, `engine?`, `style?:{tags:bool, exaggeration?, speed:1.0}`, `seed?`, `first_chunk_tokens?`, `turn_id`, `speculative:bool` | at most 1 active + 1 queued ctx per session; `speculative:true` audio is produced but **held on the box** until commit |
| `tts.text` | `ctx`, `text`, `final:bool` | append text. Phase 1 renders per clause/sentence as the box's splitter emits them; phase 2 engines that accept token streams take smaller deltas |
| `tts.flush` | `ctx`, `id` | render everything buffered now → `tts.flushed{id}` |
| `tts.close` | `ctx` | no more text; the rig emits `tts.done` when the audio is out |
| `cancel` | `gen` or `ctx` | **idempotent.** Stop every ctx with `gen <= given` (the given generation and every older one), or that ctx, drop queued audio and text, answer `cancelled{gen, stop_ms}`. GPU work for it ends within one decode step (≤ 50 ms) |
| `credit` | `ctx`, `ahead_ms` | how much unplayed audio the box will accept beyond what the phone has played (default 3000). The rig pauses LM/decoder work past it. This is the backpressure, and it bounds wasted GPU on an interrupt |
| `voice.put` / `voice.has` / `voice.delete` | `id`, payload (conditionals) | same semantics as the HTTP routes; usually done out of band |
| `ping` | `t` | every 250 ms; also carries `box_load` |
| `bye` | `reason` | clean close |

**Control frames, rig → box**

| Frame | Fields |
|---|---|
| `ready` | `session`, `resume_token`, `proto:"sf.1"`, `engines:{stt, eou, tts}` with model ids + sha8, `gpu:{index, name, sm}`, `limits:{max_text, ahead_ms_max, sessions}`, `lease:{state, id}` |
| `stt.vad` | `speech:true\|false`, `at` (mic clock): the rig's own VAD, informational; the box's Silero is authoritative for duck and barge-in |
| `stt.partial` | `committed`, `tail`, `stable_words`, `punct_final`, `upto`, `lag_ms` (≤ 1 per `partial_ms`; only when text changed) |
| `eou` | `p` (Smart Turn), `at`, `pause_ms`, `ms` (compute) at every pause ≥ `min_pause_ms`, once per pause |
| `stt.final` | `id?`, `text`, `words:[[w, start, end]]`, `upto`, `ms` |
| `tts.started` | `ctx`, `ttfa_ms` (text-in → first frame) |
| `tts.words` | `ctx`, `words:[[w, start_ms, end_ms]]` relative to the ctx audio start: drives heard-text cuts (§5.5) and captions |
| `tts.flushed` / `tts.done` | `ctx`, `audio_ms`, `rtf`, `gpu_ms` |
| `cancelled` | `gen`, `stop_ms` |
| `state` | `lease: held\|vacating\|lost`, `deadline_s?`; `vacating` = finish the current sentence, take no new ctx |
| `error` | `code` (`bad_frame`, `unknown_voice`, `busy`, `too_long`, `engine_failed`, `vacated`, `internal`), `ctx?`, `fatal:bool`, `retry_after_s?` |
| `pong` | `t`, `srv_t`, `gpu_util`, `vram_free_mib`, `queue` |

**Keepalive and liveness.** Pings run every 250 ms. The box declares the rig dead after **3 missed pongs, or 1 s**, or after partial lag over 800 ms for 1 s, and fails over (§6). The rig closes a session after 30 s without any frame.

**Resume.** The rig keeps **no conversational state worth resuming**: every utterance and reply can be recreated from the box's ring buffer and splitter. `resume` only re-attaches within 10 s, keeping the same lease and warm caches. After a reconnect the box does not replay anything to the rig: it moves the open utterance or reply to its CPU path and re-attaches the socket, and the mic clock simply continues. That design choice removes a whole class of split-brain bugs.

**Versioning.** `sf.1` is frozen once shipped. Additive fields are allowed; a meaning change is `sf.2`, offered alongside. The add-on pins `sf.1` in a contract test; the rig rejects unknown frame types with `error{bad_frame}` and never closes the session for them.

### 4.4 Scheduling inside SpeechForge

- **One process and one worker** (Windows forces Proactor; multiple uvicorn workers would duplicate models). Models load once; sessions share them.
- **Two CUDA streams:** STT+EOU on stream A at high priority, TTS on stream B. Stream priority works inside one process [P], which is the reason both lanes live in one process.
- **Work order per GPU tick:** cancels → EOU → STT partial/final → TTS for the oldest ctx with credit. With 2 sessions, TTS round-robins per chunk.
- **Bounded queues:** text ≤ 2,000 chars per ctx, ≤ 2 ctx per session, mic ingest ≤ 1.25× real time (excess dropped and counted), audio out bounded by `credit`.
- **CUDA graphs** for the Chatterbox LM decode step (fixed-shape KV cache, IOBinding). Under WDDM the per-launch overhead dominates a 350M-parameter decode at batch 1 [P]. S3Gen/HiFiGAN run in length buckets (25/33/50 tokens).

---

## 5. How the cascade reaches human feel

Most of this is already designed in v3. The rig changes how fast each piece can run, and a few defaults flip because GPU time is cheap.

### 5.1 Hearing: streaming partials
The rig re-decodes the open tail every 160 ms with Parakeet on the GPU. The cost is about 10–30 ms per call for a tail of 10 s or less [E]; Parakeet's RTFx is 6,148 at batch [P], so batch-1 latency must be measured. That gives the session live partials with a lag of about 0.2 s, versus 0.25–0.5 s per pause today. `stable_words` uses v3's rule (agreement minus 2 words; churn never exceeded 2 words in 44 steps [M]). The A/B challenger is Nemotron Speech Streaming EN 0.6B through sherpa-onnx (Windows wheels since 1.13.8 [P]; CUDA on Windows [U]) at 160 ms chunks. It decodes each audio frame once (less GPU), but its English WER is 0.5 points worse and 1.1 points worse on noise [P]. Pick by measured lag and partial stability, not by brochure.

### 5.2 Knowing when you're done: semantic end-of-turn
- Smart Turn v3.2 (BSD-2, 8M parameters; 94.7 % accuracy on the GPU build, 3 % false positives; 1–2 ms on a GPU [P]) runs on the rig at every pause of 150 ms or more, on the last 8 s.
- The box's existing decision table (SPEC §4.3) is unchanged. Only its inputs arrive sooner:
  - `p ≥ 0.9` ends the turn.
  - `0.5–0.9` with terminal punctuation ends it.
  - Otherwise the policy holds for a learned time, or listens.
- The learned per-device hold (EMA of the user's resume gaps) is unchanged.
- Rejected models:
  - The LiveKit turn detector: its licence ties it to LiveKit Agents [P].
  - Vogent: gated, with platform restrictions [P].
  - TEN: a 7.6B text model [P].
- Namo (Apache-2.0, text-only, 135 MB) is a measured-only candidate for a second vote.

### 5.3 Starting before you finish: speculation
- **LLM.** The Speculator starts on the first `eou ≥ 0.5` with a stable partial of at least 4 words. That is earlier than today because the partial no longer waits for a CPU tail. The existing caps stay: 2 per turn, 3 s apart, advisor only by default, never gateway agents (tools and memory writes make a shadow run unsafe).
- **TTS, new default with the rig: `rt.preemptive_tts: on`.** The first clause of a speculative run is rendered with `tts.open{speculative:true}` and held on the box. On commit it plays at once; on discard it is dropped.
  - On the CPU this cost about 1 s of decoder per waste, so v3 left it off. On the GPU a wasted clause costs about 100 ms of an otherwise idle card.
  - Expected saving: about 0.15–0.25 s on the turns where the hold row applies [E].
- **Endpoint anticipation.** A June 2026 paper reports −505 ms on Unmute by predicting end-of-turn early [P]. Our speculation on the 150 ms pause is the cheap version of the same idea.

### 5.4 Speaking: incremental, phrase-level TTS
- The box splitter's first-clause cut (48 chars) stays. Phase 1 sends **clauses** as `tts.text`.
- The rig's first chunk is 15 tokens (≈ 0.6 s of audio). The voice-prefix KV cache makes prefill about 50–100 ms on the CPU [M], so far less on the GPU.
- Prosody across sentences: one `seed` per reply, and each ctx carries the whole reply. The rig keeps the previous sentence's last tokens as left context where the engine supports it.
- Phase 2 engines that accept streamed text take word-level deltas: dots.tts double-streaming [P]; Qwen3-TTS through vLLM-Omni [P], which is Linux-only, so WSL2. Engines declare `caps.text_stream` in `ready`.
- Expressiveness:
  - Chatterbox tags (`[laugh]`, `[chuckle]`, `[sigh]`) pass through only for tag-capable voices (existing filter).
  - `style.exaggeration` passes through where the engine has it.
  - The persona tells the LLM it may use tags sparingly. Overuse sounds fake.
  - The reference clip matters as much as the model: record the owner *talking*, not reading [P].

### 5.5 Being interrupted, and resuming
- The layers are v3's:
  - The phone ducks optimistically within 200 ms.
  - The server ducks.
  - A word-confirmed verdict cancels or resumes.
- Faster partials make the verdict arrive about 0.3 s sooner.
- On `barge_in` the box sends `cancel{gen}`. The rig stops in ≤ 50 ms, and the credit window means at most 3 s of audio was ever produced ahead.
- **Heard-text cut** uses `tts.words` timings mapped to `heard.played_ms`: word-accurate instead of v3's linear time map. The source offset maps back to the stored markdown at a word boundary (existing `source_offset`).
- **False interruption:** the reply is *paused*, not cancelled. The rig holds the ctx (credit frozen) for `false_interruption_timeout` (2.0 s, same as v3 and LiveKit [P]). `resume{gen, rewind}` continues; past the timeout, `cancel`.

### 5.6 Backchannels, acks, fillers
- Acks: the v3 pool and selector are unchanged. With lower ETAs they fire less, which is the right outcome.
- Backchannels: still opt-in per bot (off for the advisor). The rig adds nothing new until the Pixel echo test passes.
- **Echo risk to verify (D9):** the phone plays through an AudioWorklet over a WebSocket. Browser echo cancellation may not reference that output [U], which is why v3 needs `echoBias` and the echo classifier.
- A WebRTC phone leg would get platform echo cancellation, a jitter buffer and packet-loss concealment [P]. It is the biggest remaining "feel" lever after this spec, and it is DisPatch-side work.

### 5.7 Phase 3: a preamble lane (optional, default off)
GPT-Live's design, and ConvFill [P], let a small fast model start talking while the big brain thinks. For DisPatch this means a 1–3B model on the rig producing a contextual 3–8 word opener ("Okay, so for Saturday —") that the main reply then continues.

This is risky: two brains can disagree, and the opener goes into the persona's mouth. It ships only behind a flag, only for the advisor, only if blind tests prefer it over the ack pool. The opener is fed to the main LLM as a prefix it must continue.

---

## 6. Box-side engine, fallback and hedging

### 6.1 Engine selection per session
At voice-chat open the add-on opens the `sf.1` socket (a `warm` replacement) in parallel with the CPU engines' normal warm. Engine state per session: `rig` (socket `ready` and lease `held`), else `cpu`. Partials come from whichever is active. The box keeps a **10 s mic ring**, so a switch replays the open utterance into the CPU STT with no lost words.

### 6.2 Failover rules
| Trigger | Action | Cost to the user |
|---|---|---|
| socket refused / 503 / lease not held at open | `cpu` for the session; retry the rig after `Retry-After`, then between turns | none vs v3 |
| 3 missed pongs or 1 s silence | mark the rig down; the current utterance re-transcribes from the ring on CPU; TTS ctx re-renders on CPU from the first unplayed sentence (sentence boundary, never mid-word) | ≈ 1 s gap once |
| `partial lag > 800 ms` for 1 s | treat as down for this turn | as above |
| `tts.started` not within 400 ms of `tts.text` | render *that sentence* on CPU; stay on CPU for the rest of the reply | ≈ +0.7 s for that reply |
| `state: vacating` | finish the current ctx on the rig, open the next on CPU | maybe +0.7 s on the next reply |
| 401 | rig `misconfigured`, no retries until settings change; Voice lab says so | CPU speed |

A health probe between turns (not a polled cache) re-promotes the rig; no turn ever *starts* on a rig that last failed under 30 s ago.

### 6.3 Voice parity rule
One reply is always one engine. A session's TTS engine changes only between replies. With Chatterbox on both sides (phase 1) the voice is the same model, but precision differs, so the harness's speaker-similarity check must show GPU vs CPU within 0.05 SIM. If phase 2 adopts a different rig engine, the fallback will sound like a different rendering of the owner. the owner decides whether that's acceptable (D2).

### 6.4 Power and heat
While the rig serves, the performance-profile hold is **not taken** (as the changelog says). It is taken on the first CPU failover in a session and released when the rig re-promotes.

---

## 7. Models, licences, VRAM

### 7.1 Model table

| Role | Phase | Model | Code licence | Weights licence | Runtime | Precision | VRAM [E] |
|---|---|---|---|---|---|---|---|
| STT partials + final | 1 | NVIDIA Parakeet TDT 0.6B v2 (ONNX, `istupakov/…-onnx` @ `0bbb45a3…`) | onnx-asr MIT | **CC-BY-4.0** (attribution) | ORT CUDA EP | fp32 (fp16 only after a WER gate) | 2.5 GB fp32 / 1.3 fp16 |
| STT partials (A/B) | 1–2 | NVIDIA Nemotron Speech Streaming EN 0.6B | sherpa-onnx Apache-2.0 | **NVIDIA Open Model License** (commercial OK [P]) | sherpa-onnx CUDA | fp16/int8 | ≈ 1.5 GB |
| End of turn | 1 | pipecat Smart Turn v3.2 (GPU build) | BSD-2 | BSD-2 | ORT CUDA | fp32 | 0.05 GB |
| VAD (rig, informational) | 1 | Silero VAD v5 | MIT | MIT | ORT CPU | — | 0 |
| TTS | 1 | Resemble Chatterbox-Turbo ONNX (`ResembleAI/chatterbox-turbo-ONNX` @ `d21799bd…`) | MIT | **MIT** (Perth watermark in the PyTorch pipeline; check the ONNX export does the same [U]) | ORT CUDA + CUDA graphs | fp32 reference; fp16 per graph after a golden check | 2–3 GB |
| TTS candidate | 2 | Qwen3-TTS-12Hz-1.7B-Base (or 0.6B) via faster-qwen3-tts (MIT, PyTorch CUDA graphs, Windows scripts) | Apache-2.0 / MIT | **Apache-2.0** | PyTorch 2.14 cu130 | bf16 | ≈ 5–7 GB |
| TTS candidate | 2 | rednote dots.tts-mf 2B | Apache-2.0 | **Apache-2.0** | PyTorch (Windows [U]; SGLang Linux) | bf16 | 5.7–7.9 GB [P] |
| CPU fallback (box) | — | Parakeet v2 int8, Smart Turn CPU, Chatterbox q4, Kokoro (optional plugin, see below) | as v3 | as v3 | ORT CPU | — | — |
| S2S mode | 4 | NVIDIA PersonaPlex-7B / MiniCPM-o 4.5 | — | NVIDIA Open Model License / Apache-2.0 per card [U] | WSL2 | bf16/int4 | 16–19 GB on a 5090 |

**Kokoro and the GPL.** The Kokoro-82M *weights* are Apache-2.0. Only the optional box-side Kokoro plugin is GPL, because of its phonemizer dependency; it is a separate add-on package, installed by choice. Nothing GPL is in this repo, and SpeechForge neither imports nor ships it.

**Excluded by licence (weights):**
- Non-commercial: F5-TTS (CC-BY-NC), Spark-TTS (NC-SA), Llasa (NC), Fish Audio S2 Pro, Breeze TTS 2, Higgs Audio v3, Voxtral TTS, moshika-rl-seamless.
- Custom licence: IndexTTS2 (bilibili licence).
- Llama licence: Orpheus.

A family app is private use, so these are *legally* usable here. But they can never ship in the public DisPatch_Voice docs as a recommendation, and the owner's rule prefers permissive licences, so they're listed rather than used (D7).

**Excluded by capability:**
- Kyutai TTS 1.6B: its voice-embedding model is withheld, so it can't clone the owner [P].
- VibeVoice-Realtime: no custom voices, plus an audible disclaimer [P].
- Kokoro, Magpie, Maya1: no cloning.
- LiveKit turn detector: licence.

**Provenance:** `models.lock` lists repo, revision, file, size and sha256 for every file. Download verifies before rename. Load re-verifies size+mtime and, weekly, the full hash. The first cut pins revisions only; add the hashes. `THIRD_PARTY_NOTICES.md` carries the CC-BY attribution for Parakeet and the NVIDIA Open Model License notice for Nemotron if it's adopted.

### 7.2 VRAM plan per card

| Card | Today | With SpeechForge phase 1 | Phase 2/4 |
|---|---|---|---|
| GPU 0, 5090 32 GB | llama.cpp (on demand) | unchanged | possible SpeechForge TTS if the 3090 is decode-bound (measured, D4) |
| GPU 1, 5090 32 GB | llama.cpp (on demand) | unchanged | S2S mode spike (WSL2) leases it while in use |
| GPU 2, 3090 24 GB | a pinned image-rendering workload | unchanged | — |
| **GPU 3, 3090 24 GB** | idle mostly | **SpeechForge: ≈ 6–8 GB** (Parakeet fp32 2.5 + Chatterbox 2–3 + Smart Turn + 2 CUDA contexts/arenas ≈ 1–1.5 + headroom); `mem_limit_mb` 6144 per session arena as in the first cut | + one phase-2 TTS (≈ 6 GB) during the bake-off only |

### 7.3 StudioForge lease integration
- `POST /api/leases {devices:[3], holder:"speechforge", priority:2, idle_ttl_s: idle_unload_s+300, vacate_url, vacate_token}`, as the first cut does. This works with the existing D43/D56 API and **needs no StudioForge change** for phase 1.
- Class 2: it may displace an idle tier-2/3 resident and never an idle tier-1 one (the chat model). A tier-1 lease may ask it to vacate.
- On vacate: stop taking new contexts, finish the current one, `state: vacating` to every session, unload and release (D56). The `vacate_cooldown_s` of 180 s matches StudioForge's window.
- **Leased by a benchmark** (benchmarking tooling, which never vacates): GPU 3 is unavailable. Try `devices` preference `[3, 2?]`; never GPU 2 while the image-rendering workload holds it. Then the session runs on the CPU and the Voice lab says "Rig busy (benchmark)".
- **StudioForge unreachable:** proceed only if NVML shows ≥ 8 GiB free on the card and no llama-server process owns memory there; otherwise refuse. This is stricter than the first cut's plain `proceed`.
- `reserve_gib` co-tenancy (changelog §4.2) is **not** needed for phase 1. Keep it in reserve for the day GPU 3 is wanted by an LLM; even then, compute co-tenancy hurts TTS jitter, so measure first (§11 co-tenant test).
- `sfctl status` already lists leases by holder; the panel shows `speechforge` like any other holder.

### 7.4 Warm, cold, idle
- **Warm** on voice-chat open (and optionally when the Voice tab opens). Load ≤ 10 s, overlapping the user's first sentence.
- A self-test runs after load: a 2 s golden clip transcribes to the expected words, and a 6-word TTS renders with TTFA under threshold. The session gets `ready` only after it passes.
- **Idle unload** 15 min after the last session closes (`idle_unload_s: 900`, as in the first cut). Lease TTL is idle + 300 s as StudioForge's safety net.
- **Caches** that survive restarts: `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE` for JIT PTX (in case the wheels lack native sm_120 or sm_86 code [U]), and TensorRT RTX engine caches if that execution provider is adopted later.

---

## 8. Hardening

### 8.1 Network exposure
- **Bind:** `<rig-tailnet-ip>:1236` (tailscale) + `127.0.0.1:1236`. Never `0.0.0.0`. The first cut refuses a non-loopback bind without a token, which is good; add an explicit allowlist of bind addresses.
- **Windows Firewall:** inbound TCP 1236 allowed only from the box's tailnet IP (and `<lan-ip>` only if D1 picks the LAN, and then only with TLS). Profile: Private. All other sources are blocked.
- **Tailscale ACL** (if the owner uses ACLs): `the DisPatch box → the rig:1236` only.

### 8.2 Authentication
- A 256-bit random bearer token, constant-time compare, sent only in the upgrade header (never a query string, which lands in logs). Box file `~/.config/local-chat/speechforge.env` (0600, `SPEECHFORGE_TOKEN`); rig `data\speechforge.env` with an ACL for the owner's account only.
- Rotation: `speechforge token rotate` prints nothing and writes the file. The add-on picks it up on the next session; old tokens stay valid for a 5-minute overlap.
- Requests with an `Origin` header are refused, so a malicious web page in a browser on either machine can't drive the socket.
- The vacate route accepts only the StudioForge-minted lease token (D56) and only from loopback.

### 8.3 Input limits and denial of service
- Sessions ≤ 3 (global), plus 1 admin socket.
- Messages ≤ 64 KiB. Mic ≤ 1.25× real time (excess dropped). Text ≤ 2,000 chars per ctx and ≤ 20,000 per session per minute.
- ≤ 2 contexts per session, ≤ 30 `voice.put` per hour, voice blobs ≤ 2 MB.
- Unknown frames → `error`, counted. 50 errors in 10 s → close 1008.
- Upgrade attempts per source IP ≤ 10/min.
- Per-session GPU time is accounted; a session over 60 % of GPU time for 30 s is deprioritised (TTS round-robin).
- Every queue is bounded; nothing grows with input size.

### 8.4 Privacy
- **No audio and no text at rest. Ever.** No request bodies in logs, no transcripts, no reply text, no word timings.
- Crash safety:
  - Windows Error Reporting LocalDumps is set to **mini dumps with no heap** for `python.exe` under SpeechForge (registry key, set by the installer).
  - The faulthandler prints stacks only.
- **Voice conditionals** are biometric-grade data:
  - Stored in `%LOCALAPPDATA%\SpeechForge\voices\`, with an ACL for the owner's user only.
  - The folder is excluded from Windows Search indexing and must not be inside OneDrive.
  - It is not included in any backup by default.
  - Deleted when the box deletes the profile.
  - `voice.list` exists for audit.
- **The reference clip stays on the box** (Chatterbox conditionals are computed there). The first cut's `POST /v1/voices/from-reference` is **disabled by default**; enabling it is the owner's decision D3 (phase-2 engines such as Qwen3-TTS in-context cloning and dots.tts need the clip plus its transcript on the rig).
- Debug capture (recording a session for a bug) exists only as an explicit CLI flag. It writes to an encrypted folder, auto-deletes after 24 h, and is announced in the Voice lab.

### 8.5 Process isolation
- SpeechForge runs as the owner's user (not LocalSystem), in its own venv. No `shell=True`, fixed-argv subprocesses only.
- The process opens no outbound connections except StudioForge (loopback) and model downloads (HF, only during `fetch` and never while serving).
- Models load with `onnxruntime` only; no pickle and no `torch.load` of untrusted files. Phase-2 PyTorch engines use safetensors only.

---

## 9. Observability

**Per-turn trace.** The box assigns `turn_id` (already in v3 metrics) and passes it in `tts.open` and in a `mark` on the mic stream. The rig stamps every event with it and with the mic clock. One Voice lab row per turn then shows:

```
speech_end → partial_last (ms) → eou (p, ms) → decision → spec_start → llm_first_clause → tts.open → ttfa → first_audio_phone
```

**Rig metrics** (`GET /metrics`, Prometheus text; loopback and token only; no content):

| Family | Metrics |
|---|---|
| STT | `stt_partial_lag_ms` (hist), `stt_decode_ms` (hist), `stt_final_ms`, `stt_ingest_drop_total` |
| EOU | `eou_ms` (hist), `eou_p` (hist), `eou_calls_total` |
| TTS | `tts_ttfa_ms` (hist), `tts_rtf` (hist), `tts_step_ms` (p50/p99), `tts_cancel_stop_ms` (hist), `tts_credit_stall_total`, `tts_wasted_ms_total` (rendered then cancelled) |
| Sessions | `sessions_open`, `session_errors_total{code}`, `ping_rtt_ms` |
| GPU | `gpu_util`, `vram_used_mib`, `vram_drift_mib` (since load), `gpu_temp_c`, `gpu_power_w` (NVML) |
| Residency | `lease_state`, `lease_events_total{grant,refuse,vacate,lost}`, `load_ms`, `selftest_pass` |

**Health:**
- `GET /health` (liveness, no auth, returns `{ok, version, api:1}` only).
- `GET /ready` (auth: models loaded, lease, self-test result, engine shas).
- `speechforge doctor`: driver, CUDA, ORT providers, sm of each card, models.lock verify, StudioForge reachable, token present, firewall rule present, bind address.

**Logs:** JSON lines, rotated at 10 MB × 5, with `turn_id`, durations, codes and sizes only. The token and vacate token are redacted at the formatter (as StudioForge D6 does).

**Box side:** the Voice lab gains the rig line (state, GPU, p50/p90 of TTFA and partial lag, share of turns on the rig, last fallback reason). A box-side health check can probe `speechforge /ready`.

---

## 10. Windows deployment

### 10.1 Platform
- **Windows 11 required for ongoing driver updates.** NVIDIA's last Windows 10 Game Ready driver is this month [P]. Check the rig's version first.
- **Driver:** R580 minimum for CUDA 13.x; current is 616.64 (2026-09-09) [P].
- **Python 3.12** via uv.
- `onnxruntime-gpu[cuda,cudnn]>=1.28,<1.32`, exact version and hashes in `uv.lock`. The `[cuda,cudnn]` extras pull NVIDIA's CUDA 13 runtime wheels, so **no system CUDA toolkit** [P]. Verify `get_available_providers()` and a real sm_86 and sm_120 run in `doctor`.
- `onnx-asr` 0.12.x. Its own exclusions (1.24.1, 1.25.*, 1.26.0) are respected by the range.
- WSL2 is **only** for the phase-4 S2S spike and any Linux-only phase-2 serving stack (vLLM-Omni, SGLang): Ubuntu 24.04, mirrored networking, its own port. It is never the core path [P].

### 10.2 Install (one manual step by the owner)
`install.ps1` from a release zip or a `git clone`:
1. Checks the Windows version, driver and free disk.
2. Installs uv if missing.
3. Creates `C:\SpeechForge\{releases\<tag>\, data\}` and a `current` junction.
4. Runs `uv sync --locked --extra gpu`.
5. Fetches the models and verifies their hashes.
6. Generates the token and shows it **once** with a "copy to the box" instruction, so it never reaches a log.
7. Adds the firewall rule and the WER LocalDumps key.
8. Adds the Startup-folder launcher and runs `doctor`.

The box then needs only the env file, `voice.yaml rig.*` and a restart of DisPatch so the add-on reads them.

### 10.3 Running
- Launched **at logon** by a Startup-folder `.bat` that runs `supervise.py` (the first cut's), mirroring how StudioForge runs.
- CUDA from a Session-0 service is reported to work (Ollama under NSSM) but is unverified here [P/U]. WinSW is the documented option if the owner wants it running without a logon.
- The watchdog restarts on crash with backoff 1 → 30 s and stops after 5 crashes in 10 min (`state: crashloop`, visible in `/health` via a sidecar file).

### 10.4 Update and rollback (blue/green)
1. `speechforge update --check` fetches tags.
2. `--apply` builds `releases\<new>\` (venv + models by lock; shared model files are hard-linked by hash).
3. It starts the new release on a **canary port** (1237, loopback) and runs the self-test: golden STT clip at the expected WER, TTS TTFA ≤ 250 ms, cancel ≤ 50 ms, protocol contract test.
4. Only then does it drain sessions (≤ 15 s), switch the `current` junction and restart.
5. A failed self-test leaves `current` untouched.
6. A failed post-switch health check within 2 min switches back automatically.
7. The two most recent releases are kept.
8. Never `git pull` in place.

Model upgrades follow the same path, because the lock is part of the release.

---

## 11. Tests and benchmark harness

### 11.1 Layers
| Layer | Where | What |
|---|---|---|
| Unit (CPU, fakes) | rig repo CI + box | frame codec (header, endianness, wrap), session state machine, cancel idempotence, credit/backpressure, vacate drain, auth (no Origin, bad token, rotation overlap), limits, lock verify |
| Contract | both repos | golden `sf.1` transcripts (JSON + binary fixtures) replayed against the fake rig (add-on) and the real server with fake engines (rig). `api`/`proto` pinned |
| GPU | the rig | real models on GPU 3: WER on the corpus, TTFA, RTF, cancel stop, CUDA-graph parity (graph vs eager output identical within 1e-3) |
| Chaos | the rig + box | kill the process mid-reply; drop the network for 2 s; vacate mid-sentence; lease lost; image rendering saturating GPU 2; llama-server load on GPU 0/1; 2 sessions at once |
| Soak | the rig | 30 min continuous conversation replay ×2 sessions: zero underruns, VRAM drift ≤ 100 MB, p99 step stable |
| End to end | phone + box + rig | 20 spoken turns per bot class, Pixel over Wi-Fi and LTE, Bluetooth car kit: first-audio p50/p90, barge-in, false-interrupt resume |
| Listening | family | blind A/B (phase 2 bake-off): 20 lines × engines, the owner's voice; MOS-style 1–5 plus "which sounds more like the owner" |

### 11.2 The corpus
- 60 utterances recorded by the family on the box (consent; stored on the box only, never on the rig or in a repo).
- Clean, car (engine and road), kitchen, and Bluetooth mic.
- Names of family members, bots and places.
- 20 "unfinished" utterances ("…maybe we could") to score false end-of-turn.
- Plus the existing v3 clips (`addendum.wav` etc.).

### 11.3 Acceptance thresholds (phase 1 ship gate)
| Test | Threshold |
|---|---|
| Final WER, clean / noisy / names | ≤ 6 % / ≤ 12 % / ≥ 90 % names correct (with hotwords) |
| Partial lag p90 | ≤ 250 ms |
| EOU decision after speech end (confident turns) p50 / p90 | ≤ 350 / ≤ 550 ms |
| False end-of-turn on the unfinished set | ≤ 4 % |
| TTS TTFA p50 / p90 (warm, 3090, 1 session) | ≤ 150 / ≤ 250 ms |
| TTS TTFA p90 with 2 sessions | ≤ 320 ms |
| RTF | ≤ 0.25 |
| `cancel` → GPU idle for that gen | ≤ 50 ms p99 |
| Rig killed → session on CPU | ≤ 1.0 s, no duplicate words, no lost utterance |
| Speaker similarity GPU vs CPU Chatterbox (same voice) | Δ ≤ 0.05 (ECAPA cosine) |
| Synthesised-speech WER (TTS → Parakeet) | ≤ 3 % |
| End to end, advisor, Wi-Fi | first audio p50 ≤ 1.6 s, p90 ≤ 2.2 s |
| Box CPU per session while on the rig | ≤ 15 % of one core |
| Soak | 0 underruns, VRAM drift ≤ 100 MB, 0 session errors |
| Co-tenant (image rendering busy on GPU 2, LLM on 0/1) | TTFA p90 within +20 % of idle |

---

## 12. Phased roadmap

| Phase | Ships | Effort [E] | Exit criterion |
|---|---|---|---|
| **0. Measure** (this week) | The first cut's HTTP server installed on the rig as a measuring tool: Parakeet + Chatterbox on GPU 3. `bench` prints TTFA/RTF/decode p99 for 3090 vs 5090, fp32 vs fp16, graphs on/off. Settle the [U] items in §14. | 1 day incl. the owner's install | measured numbers replace every [E] in §3 |
| **1. Realtime session** | `sf.1` WebSocket; rolling Parakeet partials + finals; Smart Turn GPU; Chatterbox GPU with word timings, cancel, credit; lease class 2 + vacate; add-on `rig` engine behind the existing seams; failover; Voice lab rig line; metrics; blue/green update; tests to §11.3 | 6–8 days (rig 4, add-on 3, live wiring 1) | §11.3 green on the live box |
| **2. Better voice and hearing** | TTS bake-off (Qwen3-TTS 1.7B/0.6B, dots.tts-mf, Chatterbox GPU) + family blind test; Nemotron streaming A/B; fp16/graph tuning; text-streaming for engines that take it; 5090 placement if decode-bound | 4–6 days | a winner chosen by the owner (D2); no regression in §11.3 |
| **3. Human-feel extras** | preemptive first-clause TTS during holds (default on with the rig), LLM start on the 150 ms pause, backchannels for opted-in bots after the Pixel echo test, preamble lane experiment (flagged), WebRTC phone leg proposal (DisPatch) | 4–6 days + DisPatch work | p50 ≤ 1.1 s on the advisor |
| **4. Optional full-duplex mode** | WSL2 spike: PersonaPlex-7B (or MiniCPM-o 4.5) on a 5090 as a "casual chat" mode with a cloned or chosen voice, no tools; a text side-channel from the bot (MoshiRAG/GPT-Live delegation pattern [P]) | 3–5 days spike | the owner judges the feel; a licence and safety review before any family use |

What ships first is phase 1. Phases 2–4 are independent and can be dropped.

---

## 13. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The LLM's first token stays ≈ 1.2 s, so the stretch target is missed | high | medium | report it honestly; speculation + preemptive TTS; measure the assistant model's time-to-first-token; preamble lane (phase 3) |
| ORT Windows wheels lack native sm_86/sm_120 code → long JIT on first load | medium | low | persistent CUDA cache; the self-test warms it; measure in phase 0 |
| WDDM launch overhead makes the Chatterbox LM step slow at batch 1 | medium | medium | CUDA graphs + IOBinding; TensorRT RTX EP as an A/B [P: Ampere+, JIT < 30 s] |
| GPU 3 taken by a benchmark lease when the owner wants to talk | low | low | CPU fallback; Voice lab says why; the owner can stop the benchmark |
| Echo through the phone speaker triggers barge-in | medium | high in the car | v3 echo classifier + `echoBias`; Pixel test; WebRTC leg later |
| Rolling re-decode partials churn more than expected | low | medium | Nemotron streaming A/B is already planned |
| fp16 artefacts in S3Gen/HiFiGAN | medium | medium | fp32 default; fp16 per graph only after golden audio and SIM checks |
| Voice conditionals leak from the rig | low | high | ACL, no indexing/OneDrive, no backups, deletion sync, no clip on the rig by default |
| Phase-2 engine sounds better but fallback voice mismatch is jarring | medium | medium | D2: either accept it or use the winner only for non-the owner bot voices |
| Two GPU-hungry tenants (image rendering, llama) on the same PSU/thermals throttle the rig | low | low | GPU temperature/power in metrics |
| Third-party claims in §15 move (fast field) | high | low | everything is re-measured in phase 0/2; pins in `models.lock` |

---

## 14. Decisions (defaults chosen; the owner can change any)

| # | Decision | Default | Alternatives |
|---|---|---|---|
| D1 | Box↔rig network path | **Tailnet IP <rig-tailnet-ip>:1236**, firewall scoped to the box | LAN <lan-ip> with TLS + firewall rule |
| D2 | TTS engine long-term | **Chatterbox GPU in phase 1**; switch only if the phase-2 blind test prefers another *and* the owner accepts a different-sounding CPU fallback | use the winner only for bot voices, keep Chatterbox for the owner's |
| D3 | May the reference clip live on the rig? | **No** (conditionals only) | yes, required for Qwen3-TTS in-context cloning and dots.tts |
| D4 | Card | **GPU 3 (3090)**, whole-card lease while active + 15 min idle | a 5090 if decode-bound; `reserve_gib` co-tenancy |
| D5 | Install | **One manual `install.ps1` run by the owner**, then self-update | SpeechForge as a StudioForge-managed process |
| D6 | Preemptive TTS during holds | **On** when the rig serves | off |
| D7 | Non-commercial weights (Fish S2 Pro, Breeze 2, Higgs v3) for private use | **Not used** | allow in a private-only lane, never in public docs |
| D8 | Full-duplex mode | **Phase-4 spike only**, PersonaPlex first | skip |
| D9 | Phone leg | **WebSocket (v3) stays**; a WebRTC proposal comes after the Pixel echo test | start the WebRTC work now |
| D10 | Family test corpus | **Record 60 utterances on the box, with consent**; never leaves the box | synthetic only |
| D11 | Repo visibility | SpeechForge stays local (unchanged from the changelog) | public under MIT after a pre-release review |
| D12 | Running without logon | **At-logon launcher** (like StudioForge) | WinSW service after a CUDA-in-service smoke test |

**Open [U] items for phase 0:**
- Native sm_86/sm_120 code in the ORT 1.31 Windows wheels.
- sherpa-onnx CUDA on Windows with Nemotron.
- Batch-1 Parakeet decode time on a 3090.
- Chatterbox ONNX watermarking.
- CUDA from a service session.
- PyAV/libopus on Windows (only if Opus ever goes rig-side).
- Faster-qwen3-tts on sm_86.
- Whether dots.tts runs natively on Windows.
- MiniCPM-o 4.5's actual LICENSE file.

---

## 15. Sources (accessed 2026-10-10 unless dated otherwise)

Full tables and per-claim sources are in the add-on's design notes (one research pass each for STT and turn-taking, TTS, speech-to-speech, and runtimes).

**Speech to text and turn-taking**
- HF Open ASR Leaderboard results (`english_short_latest.csv`, updated 2026-10-09): https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results
- Parakeet TDT 0.6B v2 (2025-05-01): https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
- Nemotron Speech Streaming EN 0.6B (2026-03-13): https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- NVIDIA Nemotron ASR blog (2026-01-05): https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
- sherpa-onnx Nemotron support: https://github.com/k2-fsa/sherpa-onnx/issues/3664
- onnx-asr (version exclusions, commit d9f1071, 2026-06-22): https://github.com/istupakov/onnx-asr
- Smart Turn v3.2 (2026-01-07): https://huggingface.co/pipecat-ai/smart-turn-v3 and https://www.daily.co/blog/smart-turn-v3-2-handling-noisy-environments-and-short-responses/
- LiveKit turn detector (2026-06-17) and its licence: https://livekit.com/blog/solving-end-of-turn-detection · https://huggingface.co/livekit/turn-detector/raw/main/LICENSE
- TurnBench (2026-08): https://arxiv.org/abs/2608.25218 · Voice-Light (2026-09): https://arxiv.org/abs/2609.20995
- Kyutai STT / delayed streams: https://kyutai.org/stt · https://github.com/kyutai-labs/delayed-streams-modeling

**Text to speech**
- Qwen3-TTS (2026-01-22): https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base · report https://arxiv.org/abs/2601.15621
- faster-qwen3-tts (MIT; RTX 4090 TTFA 156/174 ms; Windows scripts): https://github.com/andimarafioti/faster-qwen3-tts
- dots.tts (2026-06): https://github.com/rednote-hilab/dots.tts · https://arxiv.org/abs/2606.07080
- Chatterbox-Turbo: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX · https://www.resemble.ai/learn/models/chatterbox-turbo
- Artificial Analysis speech arena (Sep 2026 data): https://artificialanalysis.ai/text-to-speech/leaderboard
- Kyutai TTS / Pocket TTS: https://kyutai.org/next/tts · https://github.com/kyutai-labs/pocket-tts
- Fish Audio S2 Pro (2026-03-09): https://huggingface.co/fishaudio/s2-pro
- Higgs Audio v3 (2026-06-04): https://www.lmsys.org/blog/2026-06-04-higgs-audio-v3-tts
- Voxtral TTS (2026-03-26): https://huggingface.co/mistralai/Voxtral-4B-TTS-2603
- VibeVoice-Realtime: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B

**Realtime feel and speech-to-speech**
- OpenAI GPT-Live delegation (`gpt-live-1`, events verified 2026-10-10): https://developers.openai.com/api/docs/guides/live-delegation
- OpenAI realtime turn-taking: https://developers.openai.com/api/docs/guides/realtime-vad · https://developers.openai.com/api/docs/guides/realtime-conversations
- gpt-realtime-2.1 (2026-07-06): https://community.openai.com/t/new-realtime-models-on-the-api-gpt-realtime-2-1-and-gpt-realtime-2-1-mini/1385896
- Gemini Live (`gemini-3.8-live`, 2026-09-15): https://ai.google.dev/gemini-api/docs/changelog · https://ai.google.dev/gemini-api/docs/live-guide · https://ai.google.dev/gemini-api/docs/live-tools
- Kyutai Unmute: https://github.com/kyutai-labs/unmute
- Endpoint anticipation (−505 ms, 2026-06): https://arxiv.org/abs/2606.13450
- Pipecat interruptions: https://docs.pipecat.ai/pipecat/fundamentals/interruptions
- LiveKit turn handling: https://docs.livekit.io/reference/agents/turn-handling-options
- ConvFill (2025-11): https://arxiv.org/html/2511.07397v1
- Stivers et al. 2009, PNAS 106:10587 (turn gaps 0–200 ms): https://emcawiki.net/Stivers-etal2009
- Production latency budgets: https://cerebrium.ai/resources/pipecat-voice-agent-gpu-deployment
- PersonaPlex-7B (2026-01): https://huggingface.co/nvidia/personaplex-7b-v1

**Transport**
- https://bloggeek.me/voice-ai/ (2026)
- https://webrtc.ventures/2026/04/webtransport-is-now-baseline-what-it-means-for-real-time-media/
- Streaming TTS socket APIs: https://docs.cartesia.ai/api-reference/tts/websocket · https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-multi-stream-input · https://developers.deepgram.com/docs/flux-tts/client-messages

**Runtimes and platform**
- ORT sm_120 fix (PR #26230, merged 2025-10-07, shipped 1.23.2): https://github.com/microsoft/onnxruntime/pull/26230
- ORT CUDA execution provider (CUDA 13 default from 1.27): https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- onnxruntime-gpu 1.31.0 (2026-10-09): https://pypi.org/project/onnxruntime-gpu/
- CUDA 13.4 U1 and driver minimums: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- Riva ASR NIM on WSL2: https://docs.nvidia.com/nim/riva/asr/latest/getting-started-wsl.html
- NIM ASR support matrix (2026-10-07): https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html

**Local, measured** [M]
- the add-on's T4 harness notes (harness: 3.50 s first audio power-saver; decision 0.62–0.92 s; TTS first chunk 1267/690 ms; prefill 531 → 98 ms)
- the voice chat v3 `SPEC.md` §0.2 and §4.9
- round-trip checks from the box to the rig (tailnet 7–14 ms; LAN :1234 10/10 timeouts)
