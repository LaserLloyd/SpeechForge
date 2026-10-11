# SpeechForge runbook

Layout after `install.ps1`: `C:\SpeechForge\releases\<tag>\` (a clone + its `.venv`),
`C:\SpeechForge\current` (junction to the running release), `C:\SpeechForge\data\`
(config, `speechforge.env`, models, logs, state). Voice conditionals:
`%LOCALAPPDATA%\SpeechForge\voices`. `PY` below is `C:\SpeechForge\current\.venv\Scripts\python.exe`.

## First install (one manual step on the host)

1. Prerequisites: Windows 11, NVIDIA driver R580 or newer, Tailscale running, git, ~20 GB free.
2. Get `install.ps1` onto the host (download it from the repository, or copy it along with a
   `speechforge.bundle` for an offline install).
3. Elevated PowerShell:
   `powershell -ExecutionPolicy Bypass -File install.ps1 -BoxAddress <box tailnet IP>`
   installs the newest `v*` tag from `https://github.com/LaserLloyd/SpeechForge.git`
   (`git ls-remote --tags`, version order, then `git clone --branch <tag> --depth 1`).
   Options: `-Source <git URL | bundle | clone path>` (offline or a fork; `-Bundle` still
   works), `-Tag v0.2.0`, `-Root` (default `C:\SpeechForge`), `-DataDir` (default
   `<Root>\data`; elsewhere is remembered in `<Root>\local-env.bat`), `-Device`, `-ExpectName`,
   `-Port`, `-SkipFirewall`, `-SkipStartup`.
   It checks Windows/driver/disk/git/tailscale, installs uv, creates the layout, runs
   `uv sync --locked --extra gpu`, writes `config.yaml` (binds 127.0.0.1 + the host's tailnet
   address, `gpu.devices: [3]`, `expect_name: "3090"`), downloads and hash-verifies the models
   (~6 GB), creates the voices folder (user-only ACL, not indexed), **prints the token once**,
   adds the firewall rule (TCP 1236, local = tailnet address, remote = the box only) and the
   mini-dump key, installs the Startup-folder launcher and runs `doctor`.
   Without `-BoxAddress` the rule admits the whole tailnet range `100.64.0.0/10`: every device
   on the tailnet (including nodes shared into it) can reach the port and only the bearer token
   stops them. Narrow it later by re-running with `-BoxAddress`.
   Re-running the installer is safe: an existing `releases\<tag>` is reused and said so,
   `config.yaml` and the token are left untouched (the token's fingerprint is printed instead),
   the firewall rule and the Startup launcher are rewritten.
4. Copy the printed token to the box (`SPEECHFORGE_TOKEN=` in the add-on's env file). Never
   paste it into a chat or a commit.
5. Start it: `C:\SpeechForge\current\launchers\Start SpeechForge.bat` (or log off and on).
6. From the box: `curl http://<host tailnet IP>:1236/health` -> `{"ok":true,...}`.

## Run from a checkout (developer layout)

A plain `git clone` works without `releases\<tag>` + `current`: the data folder is wherever
`SPEECHFORGE_DATA_DIR` points (default `<repo>\data`), and `doctor` reports
`layout checkout ...` instead of looking for a junction. Keep the data outside the repo so a
checkout, reset or worktree never touches models, token or logs.

```
set SPEECHFORGE_DATA_DIR=D:\SpeechForge\data      (or copy launchers\local-env.example.bat
                                                   to <repo>\local-env.bat and set it there)
uv sync --extra gpu
uv run speechforge doctor
uv run speechforge supervise                      (or launchers\Start SpeechForge.bat)
```

The first start needs the same one-time pieces as the installer: `config.yaml` in the data
dir, `uv run speechforge fetch-models`, `uv run speechforge token new`, and a firewall rule
if the box connects over the tailnet. New code arrives with `git fetch` + `git checkout <tag>`
and a fresh `uv sync` -- `speechforge update` is for the installed layout only
(`update --check` still answers from a checkout). The two layouts must not run at once: they
share the port and the StudioForge lease.

## Measuring (phase 0 / the ship gate)

* On the host: `PY -m pytest tests/test_real_models.py -m "gpu or models"` with
  `SPEECHFORGE_TEST_MODELS=C:\SpeechForge\data\models`, `SPEECHFORGE_TEST_DTYPE=fp32`,
  `SPEECHFORGE_TEST_DEVICE=3`.
* From the box (the speechforge checkout, CPU extra, `SPEECHFORGE_TOKEN` exported):
  `speechforge bench --url ws://<host>:1236/v1/realtime --cpu-voice <local models dir> -n 20 --out bench.json`
  prints every §11.3 gate PASS/FAIL. Add `--corpus <dir>` for the family corpus (WER, names,
  false end-of-turn) and `--soak 1800` for the 30-minute two-session soak.
* fp16 trial: `PY -m speechforge fetch-models --group tts-fp16`, `models.tts_dtype: fp16`,
  restart, bench again, and listen; keep it only if the gates and the similarity check hold.

## Everyday

* `GET /health` (no auth), `GET /ready` (token), `GET /metrics` (on the host, token).
* `sfctl status` shows a `speechforge` lease on CUDA 3 while a voice chat is open and for
  15 minutes after the last one closes.
* Logs: `C:\SpeechForge\data\logs\speechforge.jsonl` (JSON lines, 10 MB x 5, no content).
* `PY -m speechforge doctor [--full]` -- everything that would stop a start.
* Token rotation: `PY -m speechforge token rotate` (prints nothing; the old token stays valid
  for 5 minutes). Read the new one from `data\speechforge.env` and update the box.

## Update (blue/green) and rollback

`C:\SpeechForge\current\launchers\Update SpeechForge.bat [source] [tag]`, or
`PY -m speechforge update --check` then `--apply [--source URL|PATH] [--tag T]`. The source
defaults to the running release's `origin` (the public repository after a default install, the
bundle after an offline one) and falls back to the GitHub URL; `--bundle` is the old name of
`--source`. `--check` lists the `v*` tags at the source (one `git ls-remote`, nothing fetched).
`--apply`:

1. checks the tag exists at the source, clones it into `releases\<tag>`
   (`git clone --branch <tag> --depth 1`; never a `git pull` in place) and runs
   `uv sync --locked --extra gpu`;
2. fetches any model file the new lock adds (shared files are hash-checked, not re-fetched);
3. starts the new release as a canary on `127.0.0.1:1237` (no lease; it shares the card) and
   gates it over sf.1: `ready` (= self-test passed), a TTS round trip with TTFA <= 250 ms, a
   cancel <= 50 ms;
4. waits for live sessions to drain (<= 15 s), switches `current`, asks the supervisor to
   restart (`data\state\restart.request`);
5. watches the switched server for 2 minutes and **switches back by itself** on failure;
6. keeps the two newest releases.

A failed canary leaves `current` untouched. Manual rollback:
`Update SpeechForge.bat rollback` (switches `current` to the previous release and restarts).
The supervisor process itself picks up a new release at the next logon.

To take SpeechForge out of service: close its window and delete
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SpeechForge.bat`. The box falls back
to its CPU on the first refused connection; a leftover lease expires on StudioForge's idle TTL
(20 min) or is released at the next start.

## Troubleshooting

| Symptom | Look at | Fix |
| --- | --- | --- |
| upgrade 503 `gpu_unavailable` | `/ready` -> `cooldown.refusals` | a tier-1 model or another lease (a benchmark) is on CUDA 3; wait, stop the benchmark, or add a card to `gpu.devices` (never a card another tenant has pinned) |
| `load_failed` | `/ready` -> `last_error`, `selftest` | "not placed on CUDA": the CUDA wheels did not load (re-run `uv sync`); "expected a card matching": bus renumbered; "self-test failed: ttfa": the card is busy or slower than expected -- bench it |
| `/health` says `crashloop` | the console, `logs\speechforge.jsonl` | 5 crashes in 10 min; fix the cause, restart the launcher |
| exit 78 | console | configuration refused (bind rules, unknown key) |
| box keeps falling back | Voice lab rig line, `pong` RTT | Tailscale relaying (DERP)? `tailscale ping <host>`; partial lag > 800 ms trips the fallback |
| a stale lease | `sfctl status` | released at the next start or by its TTL |
