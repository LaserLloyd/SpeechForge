# Security review: the 2026-10-11 integration

- **Date:** 2026-10-11
- **Branch reviewed:** `integration` at the merge of the seven feature branches
  (`main...integration`, 52 files, +11,104 / -225). Fixes are on `review/security`, which
  starts at that merge.
- **Rules applied:** `docs/SPEC.md` section 8 (network exposure, authentication, limits,
  privacy, process isolation) and `DECISIONS.md` S6 (voices as content-addressed tensors,
  `allow_pickle=False`, zip-size and shape checks), S8 (secrets in the env file only, never
  logged), S15 (tailnet + loopback binds, token, no `Origin`, loopback-only vacate) and S16
  (load only on a clearly free card), plus the D3 rule that reference clips stay off the rig
  unless the owner says otherwise.

## Method

Code review with the diff and the touched pre-existing surfaces open; no GPU, no model
download, no server started on a fixed port. For every finding that was a bug or a hardening
gap, a fix with a test went onto `review/security` as its own commit; `uv run pytest -q`
and `uv run ruff check src tests tools` stay green. The scrub hooks (`scripts/hooks`,
installed by `scripts/install-hooks.sh`) ran on every commit; the whole tree and every commit
tree also pass `scripts/scrub_check.py` (generic rules; the private-identifier rules file is
not present on the reviewing machine, so names and hostnames were checked by hand with
grep instead).

Surfaces covered:

1. New engine adapters: `engines/qwen3_tts.py`, `engines/dots_tts.py`, `engines/nemotron.py`,
   `engines/nemotron_worker.py` (voice-format parsers, the D3 refusal, the dots `http`
   backend, the Nemotron child process and its pipe protocol, outbound connections).
2. Install and update: `install.ps1`, `update.py`, `__main__.py`, the launchers.
3. The listening kit: `tools/listening/*`.
4. Pipeline and speed work: `pipeline.py`, `engines/chatterbox.py`, `engines/chatterbox_tts.py`,
   `client.py`, `bench.py`.
5. Shared surfaces: `models.lock`, `THIRD_PARTY_NOTICES.md`, `config.py`, the scrub hooks,
   `.gitignore`.
6. Public safety of the tree: grep for hostnames, user names, local paths, tailnet addresses,
   tokens and personal names.

## Findings

Severity: **High** = data leaves the machine or code runs that should not; **Medium** = a
limit or an isolation rule is missing on a reachable path; **Low** = defence in depth,
robustness, or operator-error exposure; **Info** = verified, nothing to change.

| # | Severity | Where | Finding | Status |
|---|---|---|---|---|
| 1 | Medium | `engines/dots_tts.py` `_HttpBackend.__init__` | The `http` backend's httpx client used the defaults: `trust_env=True` honours `HTTP_PROXY`/`HTTPS_PROXY` from the environment, so the request carrying the reference clip (a recording of a person, D3) could have been routed to a proxy instead of the loopback URL the config validated. The lease client already sets `trust_env=False, follow_redirects=False`; this one did not. | Fixed `ad76b1e` |
| 2 | Low | `engines/dots_tts.py` `_NativeBackend.__init__` | SPEC 8.5 says no outbound connection while serving; the Qwen3-TTS backend pins `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` before importing the runtime, the dots native backend did not. | Fixed `ad76b1e` |
| 3 | Low | `engines/nemotron_worker.py` `serve`, `engines/nemotron.py` `SubprocessBackend.feed` | The pipe protocol's `feed` frame carried an unbounded sample count that the child allocated blindly (`_read_exact(src, 4 * n)`); a negative or non-integer `n` was also accepted. Both ends now hold it to `0..MAX_FEED_SAMPLES` (10 minutes of 16 kHz audio); the parent refuses before writing so one bad frame cannot end every open stream. The parent is our own process, so this is a bound, not a fix for an exploitable path. | Fixed `2f56e8d` |
| 4 | Medium | `.gitignore` | The listening kit's ratings were ignored one directory level deep only (`*/ratings/`): a blind set made anywhere else inside the checkout would have committed `ratings/<nick>.jsonl` (the family's nicknames, answers and comments). `manifest.json`, `key.json` and every `.wav` were already covered. | Fixed `e76be3d` |
| 5 | Medium | `scripts/scrub_check.py` | The fail-closed gate behind every commit and push knew nothing about this project's data: a `.npz` voice conditional (biometric-grade, SPEC 8.4), a `.wav` other than the synthetic self-test clip, `ratings/*.jsonl`, `key.json`, and `speechforge.env` (the `.env` rule only matched the bare name, so `<name>.env` passed). All five are now refused when git is carrying them; the selftest stages one of each plus the allowed neighbours. | Fixed `e76be3d` |
| 6 | Low | `pipeline.py` `warm_lm_workers` | The warm-up held all eight LM workers at a barrier. A worker busy with a sentence cannot reach it, so the barrier broke after 30 s and the load failed with `BrokenBarrierError`. Reachable only if a sentence runs during a (re)load, which residency does not allow today, but a load path that can fail on timing is a crash-loop waiting for a future caller. A broken barrier now means fewer distinct threads warmed, never a failed load. | Fixed `e97047f` |
| 7 | Low | `tools/listening/serve.py` `make_handler` | `--owner-label` was spliced into the page's inline `<script>` through `json.dumps`, which does not escape `<`: a label containing `</script>` ended the script. Operator-supplied, so self-inflicted, but the page is served to other people on the home network. `<` and `>` are now JSON-escaped. | Fixed `ad7ff9c` |
| 8 | Low | `tests/test_units.py` stamp test | Flaky in a full run: the verifier's fast path trusts size + `mtime_ns`, and the test rewrote the file inside one filesystem timestamp tick. The test moves the mtime on explicitly now. Not a product bug: the fast path is the documented design (SPEC 7.1), and a full re-hash runs every 7 days. | Fixed `bbe139c` |
| 9 | Low | `install.ps1` step 3 | A re-run without `-DataDir` after one with it left `<Root>\local-env.bat` pointing the launchers at the old data folder while the installer wrote config and token to `<Root>\data`. The script now says so; it does not delete a file the operator may have edited. | Fixed `ad03819` |
| 10 | Info | `engines/qwen3_tts.py` `parse_voice`, `engines/dots_tts.py` `parse_voice` | Both voice formats: raw size cap (4 MB / 8 MB, and the route caps the body at `voices.max_bytes` = 2 MB first), declared-inflation cap before `np.load`, `allow_pickle=False` (a pickled member is a `ValueError`, mapped to 400), one-dimensional int16/float32 only, finite and in range, 1-20 s (dots: 0.5-20 s) at 8-48 kHz (dots: to 96 kHz), transcript at most 1000 chars with no control characters. `zipfile` enforces the declared member size on read, so the inflation check is sound. No `np.fromfile`, no archive member names are used as paths. | Verified |
| 11 | Info | `engines/qwen3_tts.py` `check_safetensors_only`, `engines/dots_tts.py` `check_latent_stats` | Qwen3-TTS refuses to load beside any `.bin/.pt/.pth/.ckpt/.pkl/.pickle` and requires the two safetensors files. dots.tts's runtime loads one `latent_stats.pt` with `torch.load(weights_only=False)` (third-party code); before that the file is sha256-pinned in `models.lock`, size-capped at 1 MiB and walked by a restricted unpickler whose `find_class` admits only numpy/torch array globals and whose `persistent_load` returns an inert stub, so no library code runs on the file's say-so. `NEWOBJ` with arguments fails on the stub and is refused. | Verified |
| 12 | Info | `engines/qwen3_tts.py` `load`, `engines/dots_tts.py` `_check_allowed` | Both engines refuse to load unless `voices.allow_reference_clip` is true, before importing torch; `POST /v1/voices/from-reference` is 403 under the same flag. Qwen3 raises `ConfigError`, dots raises `RuntimeError`; residency treats both as a load failure and cools down, so a misconfigured engine is a repeating cooldown rather than exit 78. Cosmetic. | Verified |
| 13 | Info | `engines/nemotron.py` `SubprocessBackend` | Fixed argv (`[python, "-I", worker]`), no `shell=True`, `stt.nemotron.python` must exist and is only honoured for the subprocess backend; one request in flight; a hung child is killed after `worker_timeout_s` and every later call is a `WorkerError`, never a hung STT lane. The child inherits the server's environment (our own code). `add_cuda_dll_dirs` only walks `sys.path` entries that contain `nvidia/` or `sherpa_onnx/lib`. | Verified |
| 14 | Info | `update.py`, `install.ps1` | Tags are read with one `git ls-remote --tags`, filtered to `v*` and checked for membership before `git clone --quiet --branch <tag> --depth 1 <source> <target>`: the tag cannot start with `-`, a URL source must match a scheme (`^[a-z][a-z0-9+.-]*://` or `user@host:`), a local source must exist (bundles are `git bundle verify`'d). `--depth 1` clones exactly the tag's commit; each release is its own clone, so rollback is unaffected. `default_source` prefers the release's `origin` only if it is a URL or a path that still exists. | Verified |
| 15 | Info | `install.ps1` steps 7-8, `__main__.py` | The token is printed once by `token new` and otherwise only as an 8-hex `token fingerprint`; a re-run keeps an existing token; the env file gets an owner-only ACL. The firewall rule is `TCP <port>`, local = the host's tailnet address, remote = `-BoxAddress` or, when omitted, the whole `100.64.0.0/10` range with a printed warning (the bearer token then stands alone). `-BoxAddress` is validated as IPv4 or CIDR. Writes outside the install root: the Startup-folder launcher (documented, `-SkipStartup` to opt out) and the WER LocalDumps registry key (mini dumps, no heap, SPEC 8.4). | Verified |
| 16 | Info | `tools/listening/serve.py` | Audio routes match `^/audio/(\d{2})/([A-D])\.wav$` and the file must be named in the manifest; nicknames are `[A-Za-z0-9_-]{1,32}`; every rating is one `json.dumps` line (newlines in comments are escaped); body cap 64 KiB; `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, a CSP with no external source; loopback by default, a non-loopback non-tailnet bind prints a plain warning; refuses to serve a set directory containing `key.json`. | Verified |
| 17 | Info | `tools/listening/kit.py`, `render_set.py`, `wer.py`, `similarity.py` | The bearer token comes from `SPEECHFORGE_TOKEN` in the environment only, is never a flag and never printed; a non-loopback URL without it is an error. Nothing leaves the machine except requests to the `--url` the operator names; `similarity.py` reads the owner's clip locally and writes cosine numbers only. `render.json` / `wer.json` hold the public line texts and timings, no audio. | Verified |
| 18 | Info | `pipeline.py`, `engines/chatterbox.py` | The long-lived pools are bounded (`LM_WORKERS` = 8, one decoder thread) and module-level, so they cannot grow with input. Cancel still waits for a running decode and the LM future (5 s each). `bucket_padding` returns 0 at and beyond the last bucket; `decode_chunk` trims `pad * SAMPLES_PER_TOKEN` in both the keep-silence and the regular path (tested at 21, 40, 41, 999). No audio or text is logged. | Verified |
| 19 | Low | `pipeline.py` `LM_WORKERS` | `LM_WORKERS` is a constant (8) while the config allows `realtime.max_sessions` (<= 16) x `max_contexts` (<= 4) plus `tts.max_http_streams` (<= 8). Above eight simultaneous sentences the ninth LM producer queues behind the others; eight contexts held paused by their clients would starve it until one resumes. Every client is the owner's, and the defaults (3 x 2 + 1 = 7) fit. | Open: size the pool from the config, or document the ceiling next to `max_sessions` |
| 20 | Info | `models.lock` | Every new file (WeSpeaker, Qwen3-TTS 1.7B and 0.6B, dots.tts-mf and -stts, Nemotron fp32 and int8) carries `size` and a 64-hex `sha256`; every URL pins a 40-hex revision that matches the `models` table; `dest` paths are relative with no `..`. `download` streams to `<dest>.partial`, verifies size and hash, then renames; a mismatch deletes the partial. | Verified |
| 21 | Info | `THIRD_PARTY_NOTICES.md` | Qwen3-TTS: Apache-2.0 (model cards and repository). dots.tts-mf: Apache-2.0 for code and checkpoints. WeSpeaker ECAPA: Apache-2.0, evaluation only. Parakeet: CC-BY-4.0 with attribution (pre-existing). Nemotron: NVIDIA Open Model License, correctly described as not an OSI licence, with the required notice text and the conditions (attribution, guardrail clause, Trustworthy AI terms, NVIDIA may update it) and flagged for the owner's explicit approval before the engine is enabled. | Verified; approval is the owner's (see below) |
| 22 | Info | `config.py` | `extra="forbid"` on every section, so unknown keys are still errors (tested for `tts.dots`, `tts.qwen3`, `stt.nemotron`). New numeric fields are bounded; `tts.dots.url` must be `http://` on a loopback host; `tts.dots.language` <= 32 chars; `stt.nemotron.chunk_ms` is a `Literal[160]` tied to what `models.lock` pins. Unbounded strings: `tts.qwen3.language`, `tts.dots.http_model`, `stt.nemotron.python` (operator-controlled, passed to the model or used as a path that must exist). | Verified |
| 23 | Info | whole tree | No tailnet node addresses (only the `100.64.0.0/10` range and test fixtures under `tests/`), no `C:\Users\<name>` or `/home/<name>` paths, no MagicDNS names, no tokens, no e-mail addresses. The public GitHub handle and the MIT copyright name are the author's deliberate public identifiers. `uv.lock` records the package as a relative path. | Verified |

## Residual risk for the owner

These are decisions or steps this review cannot take.

1. **Nemotron weights are under the NVIDIA Open Model License**, not an OSI licence. The
   notice is in `THIRD_PARTY_NOTICES.md`; enabling `stt.engine: nemotron` on a deployment is
   the owner's explicit acceptance of its terms (attribution with any redistribution, the
   guardrail and litigation clauses, NVIDIA's Trustworthy AI terms, and that NVIDIA may update
   the agreement). The repository does not redistribute the weights.
2. **Decision D3 is still the owner's.** Both phase-2 TTS engines need the reference clip and
   its transcript on the rig and refuse to load without `voices.allow_reference_clip: true`.
   Turning it on puts a recording of a person into the voice store (owner-only ACL, not
   indexed) and, with `tts.dots.backend: http`, sends it over loopback to a server the owner
   runs in WSL2, which may cache it. The default stays off.
3. **The firewall rule needs an elevated shell** (`install.ps1`, or the rule by hand with
   `-SkipFirewall`). Without `-BoxAddress` the rule admits the whole tailnet range and the
   bearer token is the only barrier; narrow it to the client box's address.
4. **Release tags are not signed.** The installer and updater trust HTTPS to the public
   repository (or a bundle the owner carried over) and pin by tag name; a tag moved upstream
   would be installed. Signed tags or pinning by commit would close this.
5. **Dependency trust for the torch engines.** `dots.tts` calls `torch.load(weights_only=False)`
   on one file; this project pins its hash and scans its pickle, but the package itself and
   the PyTorch cu130 wheels come from their indexes as pinned in `uv.lock`. Review the lock
   when it changes.
6. **The scrub hooks' private-identifier rules live only on the owner's machine**
   (`scripts/scrub-rules.local.txt`, git-ignored by design). A clone without it, including the
   reviewing worktree, scans with the generic rules only; the hooks print this every time.
7. **`LM_WORKERS` ceiling** (finding 19): raise `realtime.max_sessions`, `max_contexts` or
   `tts.max_http_streams` past eight concurrent sentences only after sizing the pool.
8. **`local-env.bat` is written in ASCII** (`install.ps1`); a `-DataDir` with non-ASCII
   characters would be mangled for the launchers. Use an ASCII path or set
   `SPEECHFORGE_DATA_DIR` in the environment.

## What was not verified

- Nothing was run on a GPU, no model was downloaded, no live server was contacted, and
  `install.ps1` was not executed: its logic was read, not exercised.
- The third-party runtimes (`faster_qwen3_tts`, `dots_tts`, `sherpa_onnx`) were reviewed at
  the boundary this project controls (what is passed in, what is loaded, where the output
  goes), not line by line.
