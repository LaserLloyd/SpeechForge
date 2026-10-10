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
| `GET /ready` | token | 200 when the models are loaded and self-tested, else 503 + `Retry-After`; body: state, device, gpu (name, sm), lease, sessions, counters, `selftest`, `engines`, `engine_shas`, `load_ms`, `voices` |
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
| `POST /internal/vacate` | loopback + StudioForge's lease token | StudioForge D56 callback; 202 at once, drains after |

## Voice conditionals

The four Chatterbox speech-encoder tensors as an `.npz` -- `cond_emb` (1, N, 1024) float32,
`prompt_token` (1, T) int64, `speaker_embeddings` (1, 192), `speaker_features` (1, F, 80) --
or the encoder's own names (`audio_features`, `audio_tokens`). Loaded with
`allow_pickle=False`, zip-size-checked, shape/dtype/range/finite validated. Stored in
`voices.dir` (Windows default `%LOCALAPPDATA%\SpeechForge\voices`, user-only ACL, not indexed).
