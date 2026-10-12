# Benchmarks

Measured on the GPU host on 2026-10-11 against the targets in `docs/SPEC.md` §2 and
§11.3. Everything here is **[M]** (measured on this host) unless marked otherwise.
Numbers are from the shipped harness (`speechforge bench`, over the `sf.1` socket on
loopback) and from in-process probes of the Chatterbox engine (same code paths, no
socket). The probes are not part of the repo; the harness is.

## Host

| Item | Value |
| --- | --- |
| OS / driver | Windows 11 (build 26200), NVIDIA driver 616.92, CUDA 13.4 runtime via the pip wheels |
| Runtime | Python 3.12.10, onnxruntime-gpu 1.31.0 (CUDA execution provider, cuDNN 9.27 wheels), fp32 Parakeet, Smart Turn GPU build |
| Serving card | RTX 3090 (CUDA 3, sm_86), taken through a StudioForge class-2 lease; the lease unloads an idle tier-3 resident on that card (an embedding model), as DECISIONS S3 anticipates |
| Comparison card | RTX 5090 (CUDA 1, sm_120), idle, leased for the probe only |
| Co-tenants during the runs | the LLM server idle on the other cards; ComfyUI idle on CUDA 2 |

## Bring-up

| Metric | Target | Measured | Verdict |
| --- | --- | --- | --- |
| Cold load (4 ORT sessions + STT + Smart Turn) | <= 10 s | **18.6 s** before S22; **33.7 s** with S22's warm-up to the working set (Chatterbox session creation alone 8.5-9.5 s) | FAIL, see "Open" |
| Warm-up | - | 4.0 s TTS + 0.16 s STT (short line, buckets, 8 LM workers); 19.4 s + 1.0 s once the long line and the 8 s window are included (S22): the arena growth that used to happen while serving | - |
| Self-test | pass | STT WER 0.029, TTFA 198 ms, cancel 0.2 ms, round-trip WER 0.0 | PASS |
| Ready for a client once loaded (`ready_s`) | - | 0.03 s | - |
| Footprint on the card after warm-up | fixed (S5) | **7.2 GB**, unchanged after 568 soak turns (was 6.9 GB growing to 8.5 GB) | PASS |

The first live load of the day, before the fixes below, warmed up in 10.0 s and failed
its own self-test on TTFA (841 ms), which unloaded the card and started the cooldown.
That failure is what the rest of this document chases.

## Where the time went (TTS)

### Language-model step (one token, batch 1, warm, KV on the card)

| Card / dtype | ORT run | Python bind + sample + embed | Step total |
| --- | --- | --- | --- |
| 3090 fp32 | 25.7 ms p50 (34.5 p90) | ~4.8 ms | **30.5 ms** |
| 5090 fp32 | 27.9 ms p50 | - | **27.9 ms** |
| 3090 fp16 | 11.1 ms p50 (15.2 p90) | ~0.6 ms | **11.7 ms** (13 ms in the pipeline) |

The 5090 is no faster than the 3090 at fp32, so the step is launch- and host-bound,
not compute-bound, exactly as DECISIONS S4 predicted. fp16 still halves it (fewer,
fused kernels in the fp16 graph). `enable_mem_pattern` on/off and
`arena_extend_strategy` made no measurable difference. The fp16 export also removes a
4 ms IOBinding overhead the fp32 path pays per step.

### Decoder (S3Gen + HiFiGAN, one chunk)

| Case | Cost |
| --- | --- |
| Warm, same input shape as the previous call | **54-62 ms** (fp16, heuristic cuDNN algo); 65-77 ms with `cudnn_conv_algo_search: DEFAULT`; fp32 ~84 ms |
| Input length 8 vs 40 tokens, same shape class | no difference (fixed overhead dominates) |
| First call of a never-seen shape | **300-700 ms** |
| Switch between two shapes both seen before | **~200 ms** (heuristic) / ~80 ms (DEFAULT) on every switch |
| First call on a new thread (any session) | **300-450 ms**: the CUDA provider builds a per-thread context (cuBLAS/cuDNN handles) |

### Pipeline, in process (9 sentences, 45-96 tokens each, after a warm-up line)

| Configuration | TTFA p50 / p90 (ms) | RTF p50 |
| --- | --- | --- |
| 0.2.0 as shipped, fp32 | 601 / 837 | 0.86 |
| 0.2.0 as shipped, fp16 | 662 / 691 | 0.67 |
| + long-lived LM/decoder threads | 284 / 664 | 0.54 |
| + every LM worker warmed at load | 278 / 651 | 0.61 |
| + one decoder shape for every chunk (`DECODE_BUCKETS = (40, 64)`), fp16, first chunk 15 | **264 / 270** | **0.30** |
| same, first chunk 10 | **221 / 233** | **0.30** |
| same, fp32, first chunk 15 | 536 / 552 | 0.68 |

What the three fixes are (DECISIONS S18): `pipelined()` used to start a new LM thread
and a new decoder executor per sentence, so every first chunk could pay the per-thread
CUDA context; every sentence's last chunk was a new decoder shape, and even switching
between known shapes cost ~200 ms, so all chunks now pad to one shape. With those in
place the time left is the LM: 13 tokens (10 + 3 lookahead) x 13 ms + one 55 ms decode
is ~225 ms, which is what the probe shows.

## Phase-1 gates (`speechforge bench -n 20`, over sf.1 on loopback, fp16, first chunk 10)

Two columns: after the thread/shape fixes (S18-S20) and after the chunk-rate and footprint
fixes (S21-S22), which is what the host runs.

| Gate | Threshold | S18-S20 | S21-S22 (deployed) | Verdict |
| --- | --- | --- | --- | --- |
| TTS TTFA p50 (1 session) | <= 150 ms | 252 ms | **190 ms** | FAIL |
| TTS TTFA p90 (1 session) | <= 250 ms | 269 ms | **226 ms** | PASS |
| TTS TTFA p90, 2 sessions | <= 320 ms | 492 ms | **386 ms** | FAIL |
| RTF p50 | <= 0.25 | 0.37 | **0.23** | PASS |
| cancel -> GPU idle for that gen, p99 | <= 50 ms | 19 ms | 7 ms | PASS |
| Partial lag p90 | <= 250 ms | 109 ms | 94 ms | PASS |
| EOU decision after speech end p50 / p90 | <= 350 / 550 ms | 171 ms | 166 ms | PASS |
| Synthesised-speech WER (TTS -> STT) | <= 3 % | 2.5 % | 2.5 % | PASS |
| Golden clip WER | (self-test <= 15 %) | 2.9 % | 2.9 % | PASS |
| Ping RTT p50 (loopback) | - | 0.6 ms | 0.6 ms | - |
| Soak, 2 sessions: underruns | 0 | 2027 over 1015 turns (30 min) | **0 over 568 turns (10 min)** | PASS |
| Soak: VRAM drift | <= 100 MiB | 336 MiB | **0 MiB** | PASS |
| Soak: session errors | 0 | 0 | 0 | PASS |
| Final WER clean / noisy, names, false EOU | §11.3 | not measured: needs the family corpus, which lives on the box | | - |
| End-to-end, box CPU, speaker similarity, failover, co-tenant | §11.3 | box-side harness | | - |

The power-of-two arena (S22) also sped up the LM loop itself (RTF 0.37 -> 0.23, TTFA
252 -> 190 ms): exact-size extension had been calling into the allocator for new chunk
sizes during the token loop.

The underruns were one per sentence per session: a 10-token first chunk is 400 ms of audio
and the fixed 25-token second chunk took ~780 ms to generate with two sessions sharing the LM.
S21 sizes each chunk to the measured token rate; the soak then ran gap-free.

The over-the-wire TTFA is ~30 ms above the in-process probe: the session layer, the
WebSocket and the 40 ms output frame. The first measurement of the EOU gate read 6.9 s;
that was the harness pacing its "real-time" mic stream with `asyncio.sleep(0.02)` per
frame, which lasts ~31 ms on Windows, so the stream ran at two thirds of real time and
the nominal speech-end clock was seconds early. The sender now paces against deadlines
and the harness measures from the frame that actually carried the last speech sample.

With two sessions the LM lock interleaves the two sentences token by token (S12), so
each sees half the step rate: 13 tokens x 26 ms + decode is ~490 ms. Meeting the
320 ms two-session gate needs a batched LM step (both sessions in one call), which the
current ONNX export and loop do not do.

## What this means for the spec

* **The 3090 is the right card** for this workload: the 5090 is no faster for a
  batch-1 token loop (S4).
* **fp16 Chatterbox is adopted on the host** (config `models.tts_dtype: fp16`, DECISIONS
  S19): 2.3x faster LM step, the self-test's WER gates hold (STT 2.9 %, round trip 0 %).
  The repo default stays fp32 until the speaker-similarity check of §11.3 has run with
  the owner's voice (the listening kit's `similarity.py`).
* **The 150 ms p50 and the 320 ms two-session targets are out of reach for this runtime**:
  the ONNX LM step is launch-bound at ~11-13 ms/token and the decoder has a 55 ms floor;
  with two sessions each sees half the step rate. Within ONNX Runtime the remaining levers
  are a smaller first chunk and zero lookahead on the first chunk (each worth a step per
  token; seams need the listening test to judge), a batched two-session LM step, and a
  TensorRT execution-provider trial. The phase-2 engines (Qwen3-TTS with CUDA graphs,
  dots.tts) are the designed path to sub-200 ms first audio and are implemented behind
  the same adapter on their own branches.

## Open

* Cold load 33.7 s vs the 10 s target: ~9 s of session creation (graph optimisation of
  four ONNX sessions) plus ~20 s of warm-up that is mostly the arenas growing to their
  working set. Candidates, in order: one shared GPU arena for all sessions with a large
  initial chunk (`create_and_register_allocator` + `session.use_env_allocators`), so the
  allocation ramp is one call at load and the per-session arenas disappear;
  `optimized_model_filepath` caches; loading STT and TTS sessions in parallel threads.
  The shared arena is implemented behind `gpu.shared_arena` (DECISIONS S24) and measured
  with `scripts/gpu_cold_load.py`; its numbers go here when they exist.
  On the host `warm_on_boot: true` and a longer `idle_unload_s` hide the cost, at the
  price of holding the lease longer (a tier-1 load still vacates it, D56).
* The corpus gates and every box-side gate.
* A TensorRT EP trial for the LM step, and a batched two-session LM step.

## How to reproduce

```
speechforge bench --url ws://127.0.0.1:1236/v1/realtime --cpu-voice <models dir> --cpu-dtype fp16 -n 20 --out bench.json
speechforge bench ... --soak 1800            # the 30-minute two-session soak
speechforge bench ... --corpus <dir>         # the family corpus (on the box only)
```

## End-of-turn and final latency under long utterances (2026-10-12)

**Baseline ("before") numbers for two planned fixes**: an end-of-turn silence floor (do not
act on an `eou` until the pause has lasted long enough) and pre-empting the final
(decoding it before `stt.flush` so the flush costs nothing). Everything here is **[M]**,
measured over `sf.1` on loopback against the live server, with `scripts/eot_probe.py`
(a pure client: it changes nothing on the server).

### Method

* **Clips.** Windows SAPI speech (`System.Speech`, three voices, 16 kHz mono), clause
  fragments trimmed and assembled in numpy with digital-zero-plus-noise gaps (about -60 dBFS,
  a quiet room), so every pause position is known to the sample. Three scripts per length
  (the "8 s" clips are 6.8-11.2 s long and the "15 s" clips 14.5-19.3 s, counting a 0.6 s
  lead and the inserted pauses), plus
  three 2-3 s sanity clips. Each long script exists as: a control with no pause (the clean
  end of turn), and variants with **one or two mid-sentence pauses of 400, 700 and 1000 ms**
  that split a clause, so the text before the pause is unpunctuated. Five repetitions
  of each of the 15 clip classes (the repetition number rotates through the scripts), 75
  runs, 90 inserted pauses, about 17 minutes.
* **Run.** One reused session; `stt.config` with defaults (`eou.enabled`, `min_pause_ms:
  150`, `partial_ms: 160`); the clip streamed in real time (20 ms frames, deadline-paced),
  1.5 s of silent tail, then `stt.flush` -> `stt.final`. Every `eou`, `stt.partial`,
  `stt.vad` and the `stt.final` are recorded with arrival times and mic-sample positions,
  so a policy can be re-scored offline (`scripts/eot_probe.py --summarise`).
* **Caveat: synthetic speech.** Smart Turn scores prosody. SAPI voices are not human, and
  they differ among themselves. The first run used question-form endings and Smart Turn gave
  the true end of the Zira and Mark scripts `p` of about 0.01-0.03 (21 of 35 clips of each
  length had no `eou >= 0.5` at the true end); the scripts were then reworded as statements
  and the run repeated, which is the run reported here. Three scripts (the 15 s Zira script
  and both Mark scripts) still draw a low or erratic `p` at the true end, so read the true-end figures per script
  (the "by script" table) and read the false-end-of-turn rates as a property of this clip
  set, not of speakers in general. No claim is made that a person would be scored the same.
* **Thresholds.** SPEC §5.2: `p >= 0.9` ends the turn, `0.5-0.9` ends it with terminal
  punctuation. Reported at 0.5, 0.7 and 0.9. The server scores Smart Turn **once per pause**,
  when the pause reaches 150 ms (every `eou` here carried `pause_ms` 160), so a longer
  pause does not re-score: the 400, 700 and 1000 ms variants differ only in how long the
  false decision stands.

### Host and server build

| Item | Value |
| --- | --- |
| Server | SpeechForge `/health` version 0.2.0 (`api` 1), source `main` at `bb8fb39` |
| Engines | Parakeet TDT 0.6B v2 (fp32, CUDA), Smart Turn v3.2 (GPU build), Chatterbox Turbo (fp16) |
| Card | RTX 3090 (CUDA 3, sm_86) |
| Client | `scripts/eot_probe.py` on the same host, loopback `sf.1`; one session, 0 `busy` or other errors, 0 disconnects |

### Before: results

Clip runs scored: 75 (0 failed), inserted pauses: 90.

### False end-of-turn at inserted mid-sentence pauses

A pause counts as a false end of turn when an `eou` with `p >= threshold` fired at it.

| utterance | pause ms | pauses | p >= 0.5 | p >= 0.7 | p >= 0.9 | no eou |
| --- | --- | --- | --- | --- | --- | --- |
| 8 s | 400 | 15 | 14/15 (93%) | 13/15 (87%) | 8/15 (53%) | 0 |
| 8 s | 700 | 15 | 13/15 (87%) | 12/15 (80%) | 8/15 (53%) | 0 |
| 8 s | 1000 | 15 | 14/15 (93%) | 12/15 (80%) | 8/15 (53%) | 0 |
| 8 s | all | 45 | 41/45 (91%) | 37/45 (82%) | 24/45 (53%) | 0 |
| 15 s | 400 | 15 | 7/15 (47%) | 5/15 (33%) | 3/15 (20%) | 0 |
| 15 s | 700 | 15 | 9/15 (60%) | 7/15 (47%) | 4/15 (27%) | 0 |
| 15 s | 1000 | 15 | 9/15 (60%) | 6/15 (40%) | 2/15 (13%) | 0 |
| 15 s | all | 45 | 25/45 (56%) | 18/45 (40%) | 9/45 (20%) | 0 |
| all | 400 | 30 | 21/30 (70%) | 18/30 (60%) | 11/30 (37%) | 0 |
| all | 700 | 30 | 22/30 (73%) | 19/30 (63%) | 12/30 (40%) | 0 |
| all | 1000 | 30 | 23/30 (77%) | 18/30 (60%) | 10/30 (33%) | 0 |
| all | all | 90 | 66/90 (73%) | 55/90 (61%) | 33/90 (37%) | 0 |

### Distribution of `p`

Mid-sentence pauses (an `eou` that fired there) against the true end of the utterance and against pauses inside speech that were not inserted.

| where | n | no eou | p10 | p25 | p50 | p75 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mid-sentence 400 ms | 30 | 0 | 0.02 | 0.49 | 0.77 | 0.98 | 0.99 | 0.99 |
| mid-sentence 700 ms | 30 | 0 | 0.02 | 0.42 | 0.77 | 0.98 | 0.99 | 0.99 |
| mid-sentence 1000 ms | 30 | 0 | 0.02 | 0.55 | 0.78 | 0.98 | 0.99 | 0.99 |
| mid-sentence, all | 90 | 0 | 0.02 | 0.49 | 0.77 | 0.98 | 0.99 | 0.99 |
| true end, short, clean | 5 | 0 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| true end, 8 s, clean | 5 | 0 | 0.83 | 0.90 | 0.93 | 0.98 | 0.98 | 0.98 |
| true end, 8 s, after pauses | 30 | 0 | 0.07 | 0.93 | 0.96 | 0.99 | 0.99 | 0.99 |
| true end, 15 s, clean | 5 | 0 | 0.01 | 0.01 | 0.11 | 0.75 | 0.85 | 0.92 |
| true end, 15 s, after pauses | 30 | 0 | 0.01 | 0.10 | 0.43 | 0.84 | 0.97 | 0.98 |
| other pauses inside speech | 0 | 0 | - | - | - | - | - | - |

### `p` by script (synthetic voices differ)

Median (min-max) of `p` at the inserted pauses and at the true end, over every rep and pause variant of the script. A low true-end `p` is a property of the synthetic clip as Smart Turn hears it, not of the pipeline.

| script | voice | runs | pauses | p at pauses | p at true end |
| --- | --- | --- | --- | --- | --- |
| s15a | david | 14 | 18 | 0.02 (0.01-0.93) | 0.89 (0.72-0.98) |
| s15b | zira | 14 | 18 | 0.54 (0.20-0.80) | 0.01 (0.01-0.44) |
| s15c | mark | 7 | 9 | 0.99 (0.76-0.99) | 0.38 (0.04-0.84) |
| s8a | david | 14 | 18 | 0.80 (0.76-0.98) | 0.99 (0.78-0.99) |
| s8b | zira | 14 | 18 | 0.98 (0.30-0.98) | 0.96 (0.90-0.98) |
| s8c | mark | 7 | 9 | 0.99 (0.33-0.99) | 0.08 (0.01-0.90) |
| sh1 | david | 2 | 0 | - | 0.99 (0.99-0.99) |
| sh2 | zira | 2 | 0 | - | 0.99 (0.99-0.99) |
| sh3 | mark | 1 | 0 | - | 0.99 (0.99-0.99) |

### `p` against the punctuation of the latest partial

The newest `stt.partial` received before each `eou`: does its text end in `.`, `?` or `!`? (The inserted pauses follow unpunctuated clauses; the recogniser may still add punctuation.)

| where | n | p50 | p >= 0.5 | p >= 0.7 | p >= 0.9 |
| --- | --- | --- | --- | --- | --- |
| mid-sentence pause, punctuated | 89 | 0.78 | 65/89 (73%) | 54/89 (61%) | 33/89 (37%) |
| mid-sentence pause, unpunctuated | 1 | 0.76 | 1/1 (100%) | 1/1 (100%) | 0/1 (0%) |
| true end, punctuated | 74 | 0.91 | 50/74 (68%) | 49/74 (66%) | 38/74 (51%) |
| true end, unpunctuated | 1 | 0.99 | 1/1 (100%) | 1/1 (100%) | 1/1 (100%) |

### End-of-turn latency (true speech end to first qualifying `eou`)

Wall time from the moment the frame holding the last speech sample left the client to the arrival of the `eou`. `miss` = no `eou` with `p >= threshold` at the end.

| threshold | utterance | n | miss | p50 ms | p90 ms | max ms |
| --- | --- | --- | --- | --- | --- | --- |
| any eou | short | 5 | 0 | 270 | 281 | 285 |
| any eou | 8 s | 35 | 0 | 266 | 341 | 386 |
| any eou | 15 s | 35 | 0 | 257 | 327 | 364 |
| p >= 0.5 | short | 5 | 0 | 270 | 281 | 285 |
| p >= 0.5 | 8 s | 35 | 5 | 265 | 338 | 355 |
| p >= 0.5 | 15 s | 35 | 19 | 291 | 335 | 364 |
| p >= 0.7 | short | 5 | 0 | 270 | 281 | 285 |
| p >= 0.7 | 8 s | 35 | 6 | 266 | 338 | 355 |
| p >= 0.7 | 15 s | 35 | 19 | 291 | 335 | 364 |
| p >= 0.9 | short | 5 | 0 | 270 | 281 | 285 |
| p >= 0.9 | 8 s | 35 | 8 | 266 | 339 | 355 |
| p >= 0.9 | 15 s | 35 | 28 | 294 | 335 | 336 |

Per clip class at `p >= 0.5`:

| class | n | miss | p50 ms | p90 ms | max ms |
| --- | --- | --- | --- | --- | --- |
| short/0 | 5 | 0 | 270 | 281 | 285 |
| 15/0 | 5 | 3 | 269 | 285 | 289 |
| 15/1x1000 | 5 | 3 | 268 | 289 | 294 |
| 15/1x400 | 5 | 2 | 281 | 295 | 298 |
| 15/1x700 | 5 | 2 | 320 | 355 | 364 |
| 15/2x1000 | 5 | 3 | 313 | 328 | 332 |
| 15/2x400 | 5 | 3 | 335 | 336 | 336 |
| 15/2x700 | 5 | 3 | 269 | 277 | 280 |
| 8/0 | 5 | 0 | 292 | 315 | 329 |
| 8/1x1000 | 5 | 1 | 290 | 310 | 312 |
| 8/1x400 | 5 | 1 | 239 | 323 | 355 |
| 8/1x700 | 5 | 1 | 282 | 330 | 343 |
| 8/2x1000 | 5 | 0 | 235 | 250 | 250 |
| 8/2x400 | 5 | 1 | 296 | 345 | 348 |
| 8/2x700 | 5 | 1 | 290 | 313 | 316 |

### `stt.final` cost, flush wall time and partials

`ms` is the server's figure in the `stt.final` frame; wall is `stt.flush` sent to `stt.final` received, 1.5 s after the speech end. Partials are per utterance.

| utterance | n | ms p50 | ms p90 | ms p99 | ms max | wall p50 | wall p90 | wall p99 | wall max | partials/utt | lag p50 | lag p90 | lag p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short | 5 | 86 | 117 | 129 | 130 | 87 | 117 | 129 | 130 | 6 | 78 | 94 | 136 |
| 8 s | 35 | 96 | 168 | 174 | 175 | 100 | 169 | 175 | 176 | 33 | 79 | 109 | 172 |
| 15 s | 35 | 97 | 153 | 190 | 205 | 100 | 154 | 191 | 206 | 65 | 93 | 109 | 172 |

### Final text against the script (word overlap)

| utterance | n | mean % | min % | runs under 80 % |
| --- | --- | --- | --- | --- |
| short | 5 | 100 | 100 | 0 |
| 8 s | 35 | 100 | 96 | 0 |
| 15 s | 35 | 100 | 97 | 0 |

Reproduce (token read from the environment or an env file, never printed):

```
python scripts/eot_probe.py --out run --reps 5 --token-file <speechforge.env>
python scripts/eot_probe.py --summarise run/records.jsonl
```

### What the baseline says

* **A pause is not a turn end, and `p` cannot tell them apart.** 73 % of the inserted
  mid-sentence pauses drew `p >= 0.5` and 37 % drew `p >= 0.9`, at 400 ms exactly as at
  1000 ms (the score is taken once, at 150 ms). The median `p` at a mid-sentence pause (0.77)
  sits close to the median at the true end of the control utterances (0.93 for 8 s). Where a
  clause stops on a function word ("agreed on", "teams") Smart Turn does score low (p10 0.02);
  where it stops on a complete clause it scores 0.8-0.99.
* **Terminal punctuation does not help here.** The newest partial before the `eou` ended in
  `.`, `?` or `!` for 89 of the 90 mid-sentence pauses: Parakeet punctuates the clause
  boundary of any pause, however the speaker meant to continue.
* **Latency of the decision.** The true-end `eou` arrives 270 ms (p50) and 340 ms (p90) after
  the last speech sample leaves the client: the rig stamps the pause start about 65 ms
  after that sample, the pause must reach 150 ms, and the Smart Turn call takes 19 ms (p50)
  to 95 ms (p90) on the shared lane; VAD windowing and transport make up the rest.
  That is inside the SPEC target (p50 <= 350 ms, p90 <= 550 ms), so the headroom for a
  silence floor is the difference between those targets and the floor.
* **`stt.final` costs about 100 ms at the flush, whatever the length**: the `ms` field
  (p50 96-97 ms for 8 s and 15 s utterances) matches the flush-to-final wall time within a
  few ms. The tail (p90 150-170 ms, p99 175-190 ms) is the GPU lane shared with partials
  and Smart Turn. Partials arrive steadily (33 per 8 s utterance, 65 per 15 s) with `lag_ms`
  p50 79-93 ms, p90 109 ms.
* **The recogniser is sound on these clips** (word overlap 96-100 %), so the figures above are
  not an artefact of broken audio.

### After

Same harness, same clip set and reps (75 runs, 90 inserted pauses), against the build with
the S26 silence floors (`eou.min_silence_ms` 700, `eou.unpunctuated_silence_ms` 1000) and the
S28 final pre-emption, 17:49 on 2026-10-12, one session, no errors.

#### False end-of-turn at inserted mid-sentence pauses

| utterance | pause ms | pauses | p >= 0.5 | p >= 0.7 | p >= 0.9 | no eou |
| --- | --- | --- | --- | --- | --- | --- |
| 8 s | 400 | 15 | 0/15 (0%) | 0/15 (0%) | 0/15 (0%) | 15 |
| 8 s | 700 | 15 | 0/15 (0%) | 0/15 (0%) | 0/15 (0%) | 15 |
| 8 s | 1000 | 15 | 14/15 (93%) | 12/15 (80%) | 8/15 (53%) | 0 |
| 15 s | 400 | 15 | 0/15 (0%) | 0/15 (0%) | 0/15 (0%) | 15 |
| 15 s | 700 | 15 | 1/15 (7%) | 1/15 (7%) | 1/15 (7%) | 14 |
| 15 s | 1000 | 15 | 9/15 (60%) | 6/15 (40%) | 2/15 (13%) | 0 |
| all | 400 | 30 | 0/30 (0%) | 0/30 (0%) | 0/30 (0%) | 30 |
| all | 700 | 30 | 1/30 (3%) | 1/30 (3%) | 1/30 (3%) | 29 |
| all | 1000 | 30 | 23/30 (77%) | 18/30 (60%) | 10/30 (33%) | 0 |
| all | all | 90 | 24/90 (27%) | 19/90 (21%) | 11/90 (12%) | 59 |

Before: 73 % / 61 % / 37 % over all 90 pauses. A pause shorter than the floor now produces no
verdict at all (speech resumes first); a pause that outlasts the floor is scored exactly as
before, so the 1000 ms rows are unchanged and the floor is what decides. The one 700 ms pause
that was scored measured just over the floor after VAD rounding.

#### End-of-turn latency (true speech end to first `eou`)

| threshold | utterance | n | miss | p50 ms | p90 ms | max ms |
| --- | --- | --- | --- | --- | --- | --- |
| any eou | short | 5 | 0 | 817 | 835 | 840 |
| any eou | 8 s | 35 | 0 | 793 | 875 | 956 |
| any eou | 15 s | 35 | 0 | 799 | 854 | 883 |
| p >= 0.9 | 8 s | 35 | 7 | 791 | 873 | 944 |
| p >= 0.9 | 15 s | 35 | 27 | 802 | 833 | 838 |

Before: p50 257-270 ms, p90 327-341 ms. The cost of the floor is the floor: about 65 ms to
stamp the pause start, 700 ms of silence, the Smart Turn call and transport, so a clean end of
turn is decided about 0.8 s after the last word (p90 0.88 s, max 0.96 s). That is outside the
SPEC §2 target (p50 <= 350 ms) and inside the brief's bound ("well under a second" is not met,
"under a second" is). The misses (`p` below threshold at the true end) are the same synthetic
clips as before; see "`p` by script".

#### `stt.final` cost under continuous partials

| utterance | n | `ms` p50 / p90 / p99 | flush->final wall p50 / p90 / p99 | partials/utt | lag p50 / p90 |
| --- | --- | --- | --- | --- | --- |
| 8 s | 35 | 89 / 136 / 155 (was 96 / 168 / 174) | 90 / 137 / 156 | 32 | 79 / 94 |
| 15 s | 35 | 90 / 139 / 168 (was 97 / 153 / 190) | 92 / 140 / 169 | 64 | 93 / 109 |

The withdrawn partial takes 15-20 ms off the p90 and p99; the median is the decode itself. The
client's final deadline can be set from the p99: 168 + 10 = about 180 ms, with the 350 ms it
uses today leaving ample room. Word overlap with the scripts stayed at 96-100 %.

#### What to tune

Both floors are host configuration (`config.yaml` `eou:`). Lowering `min_silence_ms` buys back
latency one-for-one and lets pauses of that length through again; this clip set only says
where 400, 700 and 1000 ms land, not where human hesitation sits between them. The add-on's
false-EOT gate (ADDON-ENGINE §7) should be run against this table before the floor is moved.

## Voice quality: chunking, precision and prompt (2026-10-12)

The owner reported uneven cadence and blips in the cloned voice. Thirteen variants of the
streaming TTS path were rendered from the same profile on an idle RTX 5090 under a probe lease
(the live card was not touched): 6 lines x 2 seeds each, the token list banked per seed and
replayed at the live LM pace so decoder-only variants decode identical tokens. Metrics: envelope
dip at each chunk seam against the median envelope around it (and at random control positions);
the mean absolute log-mel difference 20 ms before and after a seam; WeSpeaker ECAPA cosine
similarity between renders (no reference clip exists on the host, so similarity is
render-to-render); decode ms per chunk; simulated playout gaps. Full tables and WAVs live with
the private run data; the figures that matter:

| Variant | Seam dip, median (control) | Spectral jump, seam (control) | Similarity to the whole render, same tokens | Decode p50 ms | TTFA p50 ms |
| --- | --- | --- | --- | --- | --- |
| A live (fp16, first 10, chunk 25, left 8, look 3, prompt 4 s, 20 ms fade) | -3.4 dB (-2.0) | 1.76 (1.92) | 0.909 | 75 | 244 |
| A repeated (decoder-noise floor) | -6.6 dB (-1.9) | 1.78 (1.93) | 0.904 | 71 | 264 |
| B whole sentence, one decode | - (-2.0) | - (1.96) | 1 by definition; 0.906 between seeds | 92 (one) | - |
| D live, fp32 | -4.8 dB (-2.1) | 2.20 (2.00) | 0.902 | 85 | 440 |
| E prompt 10 s | -2.9 dB (-2.5) | 1.86 (1.96) | **0.934** | 74 | 254 |
| F left 16 / lookahead 6 (64 bucket) | -3.7 dB (-2.5) | 1.87 (1.92) | 0.909 | 82 (p95 485) | 300 |
| G 50 ms crossfade | -2.6 dB (-2.5) | 1.86 (1.93) | 0.910 | 75 | 254 |
| H phase-aligned seams | -3.3 dB (-2.5) | 1.86 (1.94) | 0.907 | 74 | 254 |
| I temperature 0.6, repetition 1.05 | -6.5 dB (-2.5) | 2.04 (1.91) | - (different tokens) | 76 | 253 |
| J E+F+G+H | -3.3 dB (-2.7) | 2.07 (1.96) | 0.938 | 84 (p95 529) | 294 |

Readings:

* **Seams are not the blips.** The seam dip in the live configuration is within 0.2 dB (median)
  of the whole render at the same instants; only 1 of 59 seams was 10 dB deeper. The spectral jump
  at seams is *below* the jump at random positions. The decoder's own noise (A vs A repeated,
  identical tokens) moves these statistics more than any variant does.
* **Crossfade changes nothing.** 50 ms and phase alignment are inside noise; the chunks are
  essentially uncorrelated at the seam (cross-correlation 0.05), so a lag search finds nothing.
* **fp16 = fp32** on every metric (similarity 0.912 between B and C against 0.901-0.906 between
  seeds of either). fp16's LM is not bit-reproducible (the same seed gave 146-148 tokens);
  fp32's is. fp32 costs 2x LM time (TTFA 440 vs 244 ms).
* **prompt 10 s is free and better**: similarity 0.909 -> 0.934 (ceiling 0.941), decode and
  TTFA unchanged. It is now the default (S31).
* **The 64-token bucket is not worth it**: every switch between the 40 and 64 shapes costs a
  300-530 ms re-shape; F gapped in 9 of 12 simulated playouts, J in 4 of 12.
* **Calm sampling** (temperature 0.6, repetition 1.05) gives 6.5 % more tokens and more silence
  tokens per sentence; not adopted without listening.
* **LM stall**: in the probe's first renders the LM ran 46-65 ms per token beside the decoder
  (7.6 ms alone), the planner fell to 6-token chunks and the simulated client heard ~12 gaps per
  sentence. It stopped after the engine had rendered ~7 lines, i.e. a cold-engine effect the live
  host's warm-up covers (its `tts_step_ms` histogram: 98 % of steps under 20 ms, max 100 ms).
  Recorded for the next time cadence is questioned.

Live bench after the restart that evening (`speechforge bench -n 10` with the owner's profile,
before S31): TTFA p50 191 / p90 392 ms, RTF p50 0.25, cancel p99 8.8 ms, partial lag p90 94 ms,
round-trip WER 2.5 %, end-of-turn 690 ms (the S26 floor).

After S31 went live (10 s prompt, sentence units; three `bench -n 10` runs with warm shapes):
TTFA p50 201-203 ms (the longer decoder prompt costs about 10 ms on the 3090), p90 205-511 ms
(run-to-run; the server-side `tts_ttfa_ms` histogram over 90 contexts puts p50 under 250 ms),
RTF p50 0.24-0.26, cancel p99 7 ms, partial lag p90 94 ms, round-trip WER 2.5 %.
