# SpeechForge HTTP routes

The realtime path is the sf.1 WebSocket (`docs/PROTOCOL.md`). The HTTP routes are
utilities: health, readiness, metrics, voice storage, and OpenAI-compatible one-shot
transcription and speech for tools and tests.

Every route except `GET /health` needs `Authorization: Bearer <token>` (the env file's
`SPEECHFORGE_TOKEN`, or the previous one during a 5-minute rotation overlap). With no token
configured only loopback peers are served.

## Errors

`{"error": {"message", "type", "code", "retry_after_s"?}}`; every 503 carries `Retry-After`.
Codes: `not_ready` (loading; a load was kicked), `gpu_unavailable` (lease refused /
StudioForge unreachable and the card not clearly free), `gpu_vacated`, `load_failed`,
`unloading`, `busy`, `timeout`, `voice_not_found`, `bad_voice`, `voice_hash_mismatch`,
`bad_audio`, `invalid_request`, `invalid_token`, `forbidden`, `reference_clip_disabled`,
`too_large`, `synthesis_failed`.

## Routes

| Route | Auth | What |
| --- | --- | --- |
| `GET /health` | none | `{"ok": true, "version", "api": 1}` (+ `"crashloop": true`); while the supervisor is in a crash loop it answers `{"ok": false, "state": "crashloop"}` itself |
| `GET /ready` | token | 200 when the models are loaded and self-tested, else 503 + `Retry-After`; body: state, device, gpu (name, sm, `shared_arena`), lease, sessions, counters, `selftest`, `engines`, `engine_shas`, `load_ms`, `voices` |
| `GET /metrics` | loopback **and** token | Prometheus text (§9 of the spec): `stt_*`, `eou_*`, `tts_*`, `sessions_open`, `session_errors_total`, `gpu_util`, `vram_used_mib`, `vram_drift_mib`, `gpu_temp_c`, `gpu_power_w`, `lease_state`, `lease_events_total`, `load_ms`, `selftest_pass`. No content ever |
| `WS /v1/realtime` | token | sf.1 |
| `POST /v1/audio/transcriptions/raw?sample_rate=16000` | token | raw s16le 16 kHz mono (<= 30 s) -> `{text, audio_ms, cost_ms}` |
| `POST /v1/audio/transcriptions` | token | OpenAI multipart (`file` = 16-bit WAV, any rate), `response_format` json/text/verbose_json |
| `POST /v1/audio/speech` | token | OpenAI JSON `{input, voice:"sha256:<id>", response_format:"pcm"\|"wav", stream, seed, first_chunk_tokens, render}` -> chunked PCM s16le 24 kHz; the 200 is sent with the first audio |
| `PUT /v1/voices/<sha256>` | token | voice conditionals (`.npz`, <= 2 MB); the id must be the body's sha256 |
| `HEAD`/`GET /v1/voices/<sha256>` | token | 200 / 404 |
| `GET /v1/voices` | token | stored ids (audit) |
| `DELETE /v1/voices/<sha256>` | token | `{deleted}` |
| `POST /v1/voices/from-reference` | token | **403 `reference_clip_disabled`** unless `voices.allow_reference_clip` (decision D3: reference clips stay on the box) |
| `POST /v1/voices/surprise` | token | keep an improvised voice (S34): `{"text", "seed"}` -> 201 `{id, text, audio_ms, preview_wav_base64}`; **403 `reference_clip_disabled`** unless `voices.allow_reference_clip`; **501 `unsupported`** on an engine without improvised voices (only `drama3-tts` has them) |
| `POST /internal/vacate` | loopback + StudioForge's lease token | StudioForge D56 callback; 202 at once, drains after |

## Voice conditionals

The four Chatterbox speech-encoder tensors as an `.npz` -- `cond_emb` (1, N, 1024) float32,
`prompt_token` (1, T) int64, `speaker_embeddings` (1, 192), `speaker_features` (1, F, 80) --
or the encoder's own names (`audio_features`, `audio_tokens`). Loaded with
`allow_pickle=False`, zip-size-checked, shape/dtype/range/finite validated. Stored in
`voices.dir` (Windows default `%LOCALAPPDATA%\SpeechForge\voices`, user-only ACL, not indexed).

## Improvised voices (`POST /v1/voices/surprise`)

For an engine that improvises a voice when given no reference (Drama 3): the body
`{"text": "<1-500 characters>", "seed": <int or null>}` is spoken in a new voice, the take's
own codes are stored as a voice file (`dotstts-ref` with `vq_codes`, no clip), and the reply
is `201 {id: "sha256:<id>", text, audio_ms, preview_wav_base64}` (the preview at the engine's
rate, 24 kHz). Speak the text you keep with `voice: "sha256:<id>"`. Off unless
`voices.allow_reference_clip` (the opt-in the clip-cloning engines need, decision D3); an
engine without the feature answers `501 unsupported`. See `docs/ENGINES.md`, "Voices (S34)".

## Reference clips (`POST /v1/voices/from-reference`)

Off unless `voices.allow_reference_clip: true` (decision D3: the clip stays on the box). Body: a
16-bit PCM WAV, any rate, mono or stereo (averaged). The server cleans the clip itself before
the speech encoder sees it (decision S30); `voices.prepare_reference: false` skips steps 2-4.

1. Resample to 24 kHz with a band-limited polyphase filter (about 90 dB of alias rejection).
2. Remove the DC offset.
3. Trim leading and trailing silence: 20 ms frames, a frame is silent when its RMS is more than
   40 dB under the clip's loudest frame; 50 ms is kept either side. Pauses inside the clip stay.
4. Apply one gain so the speech sits at `voices.reference_target_dbfs` RMS (default -20, range
   -30..-10), never past a -1 dBFS peak (a clip with big transients is peak-limited instead).

Nothing else: no noise reduction, no gate inside speech, no compression, no cut to a fixed length.
The 3-30 s limit applies to the trimmed clip (`400 bad_audio` otherwise).

`201` returns `{"id": "sha256:...", "clip": {...}}`:

| `clip` field | Meaning |
|---|---|
| `duration_s` | length of what the encoder was given |
| `trimmed_lead_ms`, `trimmed_tail_ms` | silence removed (margin already kept) |
| `gain_db` | gain applied (negative for a hot clip) |
| `peak_dbfs_in`, `rms_dbfs_in` | level of the upload (peak: whole clip; RMS: the speech after trimming) |
| `clipped_samples` | samples at full scale in the upload |
| `prepared` | `false` when `voices.prepare_reference` is off (only `duration_s` and `prepared` are then present) |
| `warnings` | list of `{code, message}`; the voice is built anyway |

Warning codes: `short_clip` (under 3 s of speech after trimming), `quiet_clip` (speech under
-35 dBFS: too far from the mic, or noisy), `clipping` (3 or more samples at full scale),
`long_lead_silence` (more than 500 ms trimmed from the start), `silent_clip`.

A good clip: 6-15 s of clean, continuous speech by one speaker in the voice and style you want to
hear, starting right at the first word, no music or background voices, recorded at 24 kHz or
above (a lower rate caps the top of the voice). The encoder reads the clip as it is: room, noise
and delivery all become part of the voice.
