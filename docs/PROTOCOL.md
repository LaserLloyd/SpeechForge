# sf.1 -- the SpeechForge realtime session protocol

One WebSocket per voice chat between the client (the DisPatch voice add-on on the box) and
SpeechForge. Audio streams in continuously and audio streams out; partials, end-of-turn
scores, finals, word timings, cancels, credits and pings share the socket. Frozen once
shipped: fields may be added; a change of meaning is `sf.2`, offered alongside. The server
answers an unknown frame type with `error{bad_frame}` and never closes the session for it.

**Ownership: the box decides, the rig computes.** SpeechForge never ends a turn, never calls
an LLM, never stores text or audio.

## Connect

`GET /v1/realtime`, upgraded, with

* `Authorization: Bearer <SPEECHFORGE_TOKEN>` (header only; never a query string),
* `Sec-WebSocket-Protocol: sf.1`,
* **no `Origin` header** (no browser may connect).

Refusals before the upgrade are plain HTTP with the usual error JSON:

| Status | When | Client action |
| --- | --- | --- |
| 401 | missing/wrong token | stop; "misconfigured" until settings change |
| 403 | an `Origin` header | -- |
| 400 | `sf.1` not offered | -- |
| 429 | > 10 upgrades/min from one address | back off 60 s |
| 503 + `Retry-After` | no session slot (3), or the models are cooling down (lease refused, vacated, failed load) | CPU path; retry after `Retry-After`, then between turns |

Max message 64 KiB; no compression. The first frame must be `hello` (within 5 s). The server
then waits for the models (up to `realtime.ready_wait_s`, 20 s; it sends `state{models:
"loading"}` first) and answers `ready`, or `error{not_ready, fatal, retry_after_s}` and closes.
**The client runs on its CPU until `ready` arrives.**

## Binary frames (little-endian)

```
offset size field
0      1    magic 0x53 ('S')
1      1    kind: 0x01 mic PCM in | 0x02 TTS audio out
2      2    ctx   u16   TTS context id (0 for mic)
4      2    gen   u16   box generation (0 for mic)
6      4    seq   u32   per-ctx frame counter (wraps)
10     4    pts   u32   first-sample index: mic @16 kHz from session start; TTS @24 kHz from ctx start
14     2    n     u16   sample count
16     2n   PCM16 mono
```

* Mic in: 16 kHz, 20 ms (320 samples). `pts` gaps are zero-filled (<= 5 s; larger resets the
  stream), overlaps dropped. Ingest is capped at 1.25x real time (1 s burst); excess is dropped
  and counted (`stt_ingest_drop_total`).
* TTS out: 24 kHz, 40 ms (960 samples); a context's last frame may be shorter. `pts` and
  `seq` are per context.
* All STT/EOU times on the wire (`at`, `upto`, word start/end) are **mic samples** (16 kHz).

## Box -> rig

| Frame | Fields | Notes |
| --- | --- | --- |
| `hello` | `proto:"sf.1"`, `client:{name,version}`, `resume?:{session, token}`, `caps` | first frame |
| `stt.config` | `lang:"en"`, `partial_ms` (160), `eou:{enabled, min_pause_ms:150}`, `hotwords?` (<= 64) | re-sendable. `eou.min_pause_ms` is a request: the server's configured silence floors (`eou.min_silence_ms`, `eou.unpunctuated_silence_ms`) win, so the client can raise the trigger but never lower it |
| `stt.flush` | `id?` | decode now -> `stt.final{id}`; the next utterance starts at its `upto` |
| `stt.reset` | -- | drop the open utterance |
| `mark` | `turn_id` | stamps later `eou` frames |
| `tts.open` | `ctx`, `gen`, `voice:"sha256:<hex>"`, `seed?`, `first_chunk_tokens?`, `turn_id?`, `speculative?`, `style?` | 1 active + 1 queued per session (a third is `busy`); the queued ctx starts when the active one ends (`tts.done`, an error, or a cancel) |
| `tts.text` | `ctx`, `text`, `final?` | a unit is rendered when the buffer ends `. ! ? ; :`, on `final:true`, `tts.flush` or `tts.close`. At most `limits.max_text` (2000) characters per ctx in total (`too_long` past it); open a second ctx for more |
| `tts.flush` | `ctx`, `id` | render the buffer now -> `tts.flushed{id}` |
| `tts.close` | `ctx` | no more text -> `tts.done` after the audio |
| `tts.pause` / `tts.resume` | `ctx` | false interruption: production stops at the next step while paused; the default played clock freezes |
| `cancel` | `gen` **or** `ctx` | idempotent. `gen`: every ctx with gen **<= gen**; `ctx`: that ctx. -> `cancelled{gen?, ctx?, ctxs, stop_ms}` after the producer and any in-flight decode have stopped |
| `credit` | `ctx`, `ahead_ms`, `played_ms?` | production stops once the audio generated for the ctx reaches `played_ms + ahead_ms` (default 3000 ms, max 10000, clamped). See [Credit](#credit-and-backpressure): a reported `played_ms` is **not** advanced by the rig |
| `voice.put` | `id`, `part`, `parts`, `data_b64` | chunked (<= 64 KiB per message, <= 2 MB total, 30/hour); usually `PUT /v1/voices/<id>` instead |
| `voice.has` / `voice.delete` / `voice.list` | `id` | -> `voice.has{present}` / `voice.deleted` / `voice.list{ids}` |
| `ping` | `t` | every 250 ms |
| `bye` | `reason?` | clean close (no resume) |

## Rig -> box

| Frame | Fields |
| --- | --- |
| `ready` | `session`, `resume_token`, `proto`, `version`, `engines:{stt, tts, eou}` (name, weights, provider, `sha8`; tts `caps:{text_stream:false, tags:true, words:"estimated"}`), `gpu:{index, name, sm}`, `limits:{max_text, ahead_ms_max, sessions, contexts, max_message}`, `lease:{state, id}` |
| `state` | `lease: held \| unleased \| vacating \| lost`, `deadline_s?`, `models?` -- `vacating`: finish the current context, open no new one; see below |
| `stt.vad` | `speech`, `at` -- the rig's Silero, informational |
| `stt.partial` | `committed`, `tail`, `stable_words`, `punct_final`, `upto`, `lag_ms`, `ms` -- at most one per `partial_ms`, only when the text changed |
| `eou` | `p`, `at` (pause start), `pause_ms`, `ms`, `turn_id` -- once per pause, once the server's silence floor is reached (`eou.min_silence_ms` after a partial that ends a sentence, the longer `eou.unpunctuated_silence_ms` when the running partial does not; never below the client's `min_pause_ms`); cancelled if speech resumes first |
| `stt.final` | `id?`, `text`, `words:[[w, start, end]]`, `upto`, `ms` |
| `tts.started` | `ctx`, `ttfa_ms` (first text in -> first frame), `turn_id` |
| `tts.words` | `ctx`, `words:[[w, start_ms, end_ms]]` (ctx audio clock), `unit_ms`, `estimated:true` -- sent when a unit's length is known, before its audio |
| `tts.flushed` | `ctx`, `id`, `audio_ms` |
| `tts.done` | `ctx`, `audio_ms`, `rtf`, `gpu_ms` |
| `cancelled` | `gen?`, `ctx?`, `ctxs`, `stop_ms` |
| `error` | `code`, `fatal`, `ctx?`, `message?`, `retry_after_s?` |
| `pong` | `t` (echoed), `srv_t`, `queue`, `gpu_util`, `vram_free_mib`, `vram_used_mib`, `gpu_temp_c?`, `gpu_power_w?` |

Error codes: `bad_frame`, `unknown_voice`, `busy`, `too_long`, `rate_limited`, `engine_failed`,
`not_ready`, `vacated`, `internal`. 50 non-fatal `bad_frame`, `too_long`, `busy`, `unknown_voice`
and `rate_limited` errors within 10 s close the socket with 1008 (`error flood`); the other codes,
`engine_failed` included, are sent but do not count toward that limit.

## Credit and backpressure

Before every LM step a ctx's producer checks that the audio it has generated (40 ms per token,
counting the unit about to be rendered) is still below `assumed_played + ahead_ms`; if not it
waits (a credit stall) until a `credit` frame, the clock or a `tts.resume` lets it go on. A paused
ctx never runs. `ahead_ms` defaults to 3000 and is clamped to 0..`limits.ahead_ms_max` (10000).
The lookahead is therefore about 3 s of audio per ctx by default; that is also the most that is
wasted on a `cancel`.

`assumed_played` is the rig's guess at how much of the ctx the phone has played:

* **No `played_ms` ever sent for the ctx:** real time since the ctx's first audio frame (0 until
  then), minus the time spent paused. `tts.pause` freezes that clock and `tts.resume` restarts it.
* **A `credit` with `played_ms`:** the rig takes that number as is (negative -> 0) and **holds it
  until the next report**. It does not advance in real time between reports, and a later `credit`
  without `played_ms` keeps the last reported value. A client that reports must therefore keep
  reporting (each `credit` with a fresh `played_ms`), or production stalls once the generated
  audio reaches the last report plus `ahead_ms`. Pausing does not change the held figure, but a
  paused ctx still produces nothing.

A `credit` for an unknown or finished ctx is ignored.

## Lease states

`ready.lease.state` and `state{lease}` carry one of:

* `held` -- the models are loaded under a StudioForge lease (class 2, with a vacate callback).
  When StudioForge needs the card back it calls the vacate callback: the rig sends
  `state{lease:"vacating", deadline_s}`, lets the open contexts finish (bounded by the drain
  timeout), refuses new `tts.open` with `vacated`, then unloads; every session gets a fatal
  `vacated` error and a 1001 close. The lease is touched while sessions are open; if StudioForge
  forgot it and a new one cannot be had, the rig sends `state{lease:"lost"}` and unloads (no `vacating` grace; sessions get the same fatal `vacated` error).
* `unleased` -- the models are loaded **without** a lease, either because StudioForge was
  unreachable at load time and NVML showed a clearly free card (`on_unreachable:
  proceed_if_free`: at least `unreachable_min_free_gib` free and no llama-server holding memory
  there), or because `lease_mode` is `off`. It is fully serviceable.
  * Unreachable at load: the rig asks StudioForge for the lease again every
    `studioforge.relink_s` (default 30 s) while the models stay loaded. When it is granted the
    rig sends `state{lease:"held"}` and from then on behaves as `held` above. When StudioForge
    is back but refuses with `lease_conflict` (the card is leased by, or promised to, an equal or
    better class) the rig gives way exactly as for a vacate: `state{lease:"vacating",
    deadline_s}`, the open contexts finish (bounded), new `tts.open` is refused with `vacated`,
    then every session gets the fatal `vacated` error and a 1001 close. A transient refusal
    (`lease_vacating`: the card is being cleared for us; `model_busy`: a resident is mid-request)
    changes nothing; the rig keeps serving and asks again at the next interval. While StudioForge
    stays unreachable nothing changes either.
    So this kind of `unleased` still gives way to a priority-1 load, at most about one retry
    interval after StudioForge is reachable.
  * `lease_mode: off`: there is no StudioForge to ask. The rig never leases and never gives way;
    it keeps the card until it unloads (idle or restart).
* `vacating`, `lost` -- as above, transient.
* `none` -- only on the early `state` frame while the models are still loading.

## Liveness and resume

* The box pings every 250 ms and declares the rig dead after 3 missed pongs or 1 s of
  silence (or partial lag > 800 ms for 1 s) -> CPU path.
* The rig closes a session after 30 s without any frame.
* A dropped socket keeps its slot for 10 s (`realtime.resume_window_s`). The rig stops every
  TTS ctx of the detached session (queued ones too) and drops its open utterance; nothing else
  is kept. `hello{resume:{session, token}}` within the window re-attaches the same session and
  `ready` carries a **new** `resume_token`. A refused resume (unknown, expired or wrong token)
  starts a new session (non-fatal `error`, then `ready`).
* The rig keeps **no conversational state**, and the client does not replay anything: it does not
  re-send mic audio from `stt.final.upto` and does not re-open TTS. It moves the open utterance
  or the reply it was playing to its own CPU path and only re-attaches the socket.
* The mic clock continues across the gap: the client's `pts` kept running while the socket was
  down, and the rig takes the first frame after the resume as the new clock origin. It does not
  zero-fill the missing time (a gap of up to 5 s *within* a live stream is zero-filled; a longer
  one restarts the clock).

## Word timings

`stt.final.words` come from Parakeet TDT's token timestamps (mic samples). `tts.words` are
**estimates** for Chatterbox (no aligner): a unit's duration (tokens x 40 ms) split over its
words by length, punctuation adding a beat -- enough to cut "what was heard" at a word boundary.
An engine with alignment would send exact ones and drop `estimated`.
