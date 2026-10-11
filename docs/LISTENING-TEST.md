# The family listening test

The phase-2 TTS bake-off (SPEC §11.1 "Listening", §12) is decided by ears, not by a
benchmark: the family hears the same lines in the owner's cloned voice from each engine
(Chatterbox-Turbo, Qwen3-TTS, dots.tts), blind, and says how natural each sounds and which
one sounds most like the owner. This document is the protocol; the tools are in
`tools/listening/` and are run from the repository root with
`uv run python -m tools.listening.<step> ...`.

## Consent and what is recorded

- **Ask first.** The voice being cloned is a family member's; the engines need their
  reference clip (or conditionals built from it) to render the set. They agree to the clip
  being used for this test, and the raters agree to their answers being kept.
- **What a rater gives:** a nickname they choose, one 1-5 rating per sample per line, one
  "most like" pick per line, an optional free-text comment, and the time they spent on the
  line. Nothing else: no IP address, no browser details, no account. The server does not log
  requests.
- **What is kept, and where.** Everything lives in a directory the lead chooses on the host
  (the examples below use `tools/listening/out/`, which is git-ignored along with every
  `*.wav`, `*/ratings/`, `key.json`, `render.json`, `manifest.json` and `report.md`):
  - `renders/<engine>/NN.wav` + `render.json`: the synthetic audio and its timings;
  - `set/NN/<label>.wav` + `manifest.json`: the blind set the raters hear (no engine names);
  - `set/ratings/<nickname>.jsonl`: the answers;
  - `key/key.json`: label -> engine per line. Kept **outside** the set directory so the rating
    server cannot serve it; the lead does not open it until the ratings are in;
  - `set/report.md`: the scored result, which names engines.
- **None of it enters the repository.** The reference clip stays on the box (D3, D10); run
  `similarity.py` where the clip is rather than copying it. Delete the renders and the set
  when the decision is recorded; keep `report.md` with the owner's notes if you like, outside
  the tree.
- The rating page runs on the home network. On a tailnet address the traffic is encrypted
  by the tailnet; on a plain LAN address it is not, and anyone on that network can open the
  page and submit answers. The server says so when it starts. Loopback is the default.

## The five steps

The bearer token is read from `SPEECHFORGE_TOKEN` in the environment by the two scripts
that talk to the server (`render_set`, `wer`). It is never a flag and never printed. On
the host, `set SPEECHFORGE_TOKEN=<token>` (cmd) or `$env:SPEECHFORGE_TOKEN = "<token>"`
(PowerShell) in the shell you run them from, or source the data directory's
`speechforge.env`.

### 1. Render the lines, one engine at a time

The server speaks with the engine its config names (`tts.engine`). Start it with one
engine, render, switch, render again:

```
uv run python -m tools.listening.render_set --url http://127.0.0.1:1236 --engine chatterbox \
    --voice sha256:<voice-id> --out tools/listening/out/renders
uv run python -m tools.listening.render_set --url http://127.0.0.1:1236 --engine qwen3-tts \
    --voice sha256:<voice-id> --out tools/listening/out/renders
uv run python -m tools.listening.render_set --url http://127.0.0.1:1236 --engine dots-tts \
    --voice sha256:<voice-id> --out tools/listening/out/renders
```

`--engine` names the output directory (and is sent as the request's `model`); `--voice` is
whatever voice id that server's engine accepts. Twenty lines from `tools/listening/lines.txt`
(`--lines FILE` for another set; lines are numbered in file order). A `503` is waited out
per its `Retry-After`: the first request usually finds the models cold. Each line's time
to first audio (headers in, which the server sends with the first audio) and wall time land
in `render.json`. Use a seed (`--seed`, default 1000; line *n* uses seed+*n*) so a re-render is
comparable.

### 2. Build the blind set

```
uv run python -m tools.listening.make_set \
    --engines tools/listening/out/renders/chatterbox tools/listening/out/renders/qwen3-tts \
              tools/listening/out/renders/dots-tts \
    --set tools/listening/out/set --key tools/listening/out/key --seed 7
```

Per line the engines are shuffled under A/B/C (a new shuffle per line), loudness-matched
(RMS to -23 dBFS with a -1 dBFS peak guard; a plain gain, so timbre and artefacts are
untouched) and copied to `set/NN/<label>.wav`. The manifest carries the line texts, labels
and durations only. Same seed and engine order -> the same set. Lines that any engine failed
to render are left out of the set.

### 3. Serve the rating page

```
uv run python -m tools.listening.serve --set tools/listening/out/set --owner-label "Dad"
uv run python -m tools.listening.serve --set tools/listening/out/set --host <tailnet-address> --port 8181
```

It prints the URL to share. One page, inline CSS/JS, no CDN, nothing fetched from outside.
The rater types a nickname; for each line the samples appear in a random order with a play
button each (keys `1`-`4`), a 1-5 "how natural" scale per sample (`q w e r t` for the
selected one), the "which sounds most like <owner>" choice (`m`), a comment box, and Next
(`Enter`). Answers are appended to `set/ratings/<nickname>.jsonl` as they are given, so a
rater can close the page and come back to the next unanswered line. Headphones, a quiet
room, and no talking about the samples until everyone is done.

### 4. Score

```
uv run python -m tools.listening.score --set tools/listening/out/set --key tools/listening/out/key \
    --render-json chatterbox=tools/listening/out/renders/chatterbox/render.json \
    --render-json qwen3-tts=tools/listening/out/renders/qwen3-tts/render.json \
    --render-json dots-tts=tools/listening/out/renders/dots-tts/render.json
```

Writes `set/report.md` (and prints it): per engine the MOS mean with a 95 % bootstrap
interval, the "most like the owner" win rate with its interval, and, from the render
files, TTFA p50/p90 and real-time factor; then a per-line table, a per-rater table (each
rater's agreement with the others' majority pick, pairwise agreement) and the comments.

### 5. Objective checks (optional, SPEC §11.3)

Synthesised-speech WER through the server's own Parakeet (gate: <= 3 %):

```
uv run python -m tools.listening.wer --url http://127.0.0.1:1236 \
    --engines tools/listening/out/renders/chatterbox tools/listening/out/renders/qwen3-tts
```

Speaker similarity to the owner's reference clip (ECAPA cosine; SPEC asks for a delta
<= 0.05 between the GPU and CPU Chatterbox, and the same number tells how close a
candidate gets). The model is WeSpeaker's ECAPA-TDNN (Apache-2.0), pinned in
`models.lock` under group `eval`; it runs on the CPU and the server never loads it:

```
uv run speechforge fetch-models --group eval
uv run --extra cpu python -m tools.listening.similarity \
    --onnx <models-dir>/wespeaker-ecapa/voxceleb_ECAPA512.onnx --reference <owner-clip.wav> \
    --engines tools/listening/out/renders/chatterbox tools/listening/out/renders/qwen3-tts
```

Run it where the clip lives (the box, D3); it writes `similarity.json` next to each engine's
renders and prints the per-engine mean cosine and the delta against Chatterbox.

## Reading the report

- **The "most like" win rate decides D2.** It is the question the family can answer that no
  metric can: which engine is the owner. MOS says how natural each engine is on its own and
  explains a preference (an engine can be the most natural and still not sound like them).
- **Intervals first, means second.** With a family-sized panel (3-5 raters x 20 lines = 60-100
  answers) two engines a few points apart will overlap. Overlap means "not separated by this
  test", which is a valid result: the cheaper or faster engine wins a tie.
- **Per-line table:** a line where every rater flipped (the tongue-twister, the `[laugh]`
  tag, the numbers) shows a specific weakness; read the comments for that line.
- **Per-rater table:** a rater far from the others' majority is not wrong; they may hear
  something the rest do not. Agreement below ~50 % across the board means the engines are
  genuinely close, or the room was noisy.
- **TTFA / RTF** come from `render_set` with a warm server and one request at a time. They
  are the cost side of the decision (SPEC §2 targets: TTFA p50 <= 150 ms, RTF <= 0.25) and
  carry over to the `bench` gates in §11.3 only after the winner is wired into the realtime
  path.
- **WER** above 3 % on an engine is a ship blocker for that engine whatever the ears said;
  **similarity** delta beyond 0.05 against Chatterbox says the candidate's clone has drifted
  from the reference even if the family liked it.

## Recording the decision

The owner reads `report.md`, listens to anything in doubt (the set is still on disk), and
writes the outcome as a new entry in `DECISIONS.md` (`S<n>`, the next number) referencing
D2: which engine is the long-term TTS, whether the CPU fallback stays Chatterbox (D2's
accepted mismatch) or the winner is used for bot voices only, the numbers that decided it
(win rate with interval, MOS, TTFA, WER, similarity), how many raters and lines, and the
date. The report file is quoted, not committed. The lead does not write that entry; the
owner does.
