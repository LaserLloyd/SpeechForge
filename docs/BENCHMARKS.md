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
