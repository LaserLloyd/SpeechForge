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
| `tts.open` | `ctx`, `gen`, `voice:"sha256:<hex>"`, `seed?`, `first_chunk_tokens?`, `turn_id?`, `speculative?`, `style?` | 1 active + 1 queued per session; a queued ctx starts when the active one is done |
| `tts.text` | `ctx`, `text`, `final?` | a unit is rendered when the buffer ends `. ! ? ; :`, on `final:true`, `tts.flush` or `tts.close` |
| `tts.flush` | `ctx`, `id` | render the buffer now -> `tts.flushed{id}` |
| `tts.close` | `ctx` | no more text -> `tts.done` after the audio |
| `tts.pause` / `tts.resume` | `ctx` | false interruption: production stops at the next step while paused; the default played clock freezes |
| `cancel` | `gen` **or** `ctx` | idempotent. `gen`: every ctx with gen **<= gen**; `ctx`: that ctx. -> `cancelled{gen?, ctx?, ctxs, stop_ms}` after the producer and any in-flight decode have stopped |
| `credit` | `ctx`, `ahead_ms`, `played_ms?` | production stops while generated audio > `played_ms + ahead_ms` (default 3000, max 10000). Without `played_ms` the rig assumes real-time playback from the ctx's first frame |
| `voice.put` | `id`, `part`, `parts`, `data_b64` | chunked (<= 64 KiB per message, <= 2 MB total, 30/hour); usually `PUT /v1/voices/<id>` instead |
| `voice.has` / `voice.delete` / `voice.list` | `id` | -> `voice.has{present}` / `voice.deleted` / `voice.list{ids}` |
| `ping` | `t` | every 250 ms |
| `bye` | `reason?` | clean close (no resume) |

## Rig -> box

| Frame | Fields |
| --- | --- |
| `ready` | `session`, `resume_token`, `proto`, `version`, `engines:{stt, tts, eou}` (name, weights, provider, `sha8`; tts `caps:{text_stream:false, tags:true, words:"estimated"}`), `gpu:{index, name, sm}`, `limits:{max_text, ahead_ms_max, sessions, contexts, max_message}`, `lease:{state, id}` |
| `state` | `lease: held \| unleased \| vacating \| lost`, `deadline_s?`, `models?` -- `vacating`: finish the current context, open no new one |
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
`not_ready`, `vacated`, `internal`. 50 errors in 10 s close the socket with 1008.

## Liveness and resume

* The box pings every 250 ms and declares the rig dead after 3 missed pongs or 1 s of
  silence (or partial lag > 800 ms for 1 s) -> CPU path.
* The rig closes a session after 30 s without any frame.
* A dropped socket keeps its slot for 10 s. `hello{resume:{session, token}}` re-attaches
  (new `resume_token` in `ready`). The rig keeps **no conversational state**: on resume the box
  re-sends mic audio from the last `stt.final.upto` (its 10 s ring) and re-opens TTS from the
  first unplayed sentence. A refused resume starts a new session (non-fatal `error`, then
  `ready`).

## Word timings

`stt.final.words` come from Parakeet TDT's token timestamps (mic samples). `tts.words` are
**estimates** for Chatterbox (no aligner): a unit's duration (tokens x 40 ms) split over its
words by length, punctuation adding a beat -- enough to cut "what was heard" at a word boundary.
An engine with alignment would send exact ones and drop `estimated`.
