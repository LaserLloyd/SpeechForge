"""End-of-turn and final-latency probe for a RUNNING SpeechForge server.

Drives ``ws://<rig>:1236/v1/realtime`` as an sf.1 client only (it changes nothing on the
server) and measures what a long utterance with mid-sentence pauses does to the
end-of-turn score (``eou.p``) and to the final transcript:

* **false end-of-turn**: how often an inserted mid-sentence pause (400 / 700 / 1000 ms,
  one or two per utterance, the text before it unpunctuated) draws an ``eou`` with
  ``p >= threshold`` (reported at 0.5, 0.7 and 0.9, SPEC §5.2);
* **eot latency**: true speech end -> first ``eou`` with ``p >= threshold``;
* **stt.final**: the frame's ``ms`` and the wall time ``stt.flush`` -> ``stt.final``,
  with the number of partials and their ``lag_ms``;
* a word-overlap check of the final against the script, so a broken clip is noticed.

The clips are synthesised with Windows SAPI (``System.Speech``, no model, no download):
clause fragments are rendered at 16 kHz mono, trimmed, and assembled in numpy with
digital-zero-plus-faint-noise gaps (about -60 dBFS, a quiet room), so the positions of
the pauses are known to the sample. Synthetic speech is not a human: Smart Turn may
score it differently. Fragment wavs are cached in ``--out``; nothing is committed.

Usage (from a checkout; the token is read at run time and never printed)::

    SPEECHFORGE_TOKEN=... python scripts/eot_probe.py --out run1
    python scripts/eot_probe.py --token-file path/to/speechforge.env --out run1 --reps 5
    python scripts/eot_probe.py --out run1 --only 8            # 8 s clips only
    python scripts/eot_probe.py --out run1 --plan              # show the plan, run nothing
    python scripts/eot_probe.py --summarise run1/records.jsonl # re-score saved records

``--out`` receives ``records.jsonl`` (one JSON object per clip run: every ``eou``,
``stt.partial``, ``stt.vad`` and the ``stt.final``, with arrival times and the mic-sample
positions of the inserted pauses and of the true speech end, enough to re-evaluate a
policy offline), ``meta.json`` and ``summary.md``. ``--token-file`` accepts an env file
holding ``SPEECHFORGE_TOKEN=...``. A ``busy`` or refused session is retried for a while:
another client (the owner's voice client) may hold the slot.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import wave
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RATE = 16000
LEAD_S = 0.6
TAIL_S = 1.5
NOISE_STD = 0.001  # about -60 dBFS
SPEECH_PEAK = 0.6
TRIM_MARGIN_S = 0.02
PAUSES_MS = (400, 700, 1000)
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9)
BUDGET_S = 25 * 60

VOICES = {
    "david": "Microsoft David Desktop",
    "zira": "Microsoft Zira Desktop",
    "mark": "Microsoft Mark",
}


@dataclass(frozen=True)
class Script:
    sid: str
    length: str  # "short" | "8" | "15"
    voice: str
    rate: int
    clauses: tuple[str, ...]  # only the last carries terminal punctuation


SCRIPTS: tuple[Script, ...] = (
    Script(
        "s8a",
        "8",
        "david",
        0,
        (
            (
                "I was wondering if you could explain how the new scheduling feature "
                "works for our team"
            ),
            "and whether it can handle recurring events",
            "across several different time zones?",
        ),
    ),
    Script(
        "s8b",
        "8",
        "zira",
        0,
        (
            "I need to set up the backup system on my laptop",
            "so that it runs every night",
            "and sends me a message if anything fails.",
        ),
    ),
    Script(
        "s8c",
        "8",
        "mark",
        0,
        (
            "Last week I asked about changing the delivery address on my order",
            "and I still have not heard back",
            "so please check where things stand.",
        ),
    ),
    Script(
        "s15a",
        "15",
        "david",
        0,
        (
            (
                "I have been thinking carefully about how we should organize the whole "
                "project for the next quarter given the budget that we agreed on"
            ),
            (
                "because there are several tasks that depend on each other and some of "
                "them need input from the other teams"
            ),
            (
                "and which ones can start right away while the design review is still "
                "going on and which ones have to wait?"
            ),
        ),
    ),
    Script(
        "s15b",
        "15",
        "zira",
        0,
        (
            (
                "When I plug in the printer in the back room it shows up in the list of "
                "devices for a few seconds"
            ),
            (
                "and then it disappears and the computer says that the driver is not "
                "installed even though I installed it twice already"
            ),
            "so I would like to know what to try first before I call the shop.",
        ),
    ),
    Script(
        "s15c",
        "15",
        "mark",
        0,
        (
            (
                "My family is planning a trip to the coast in the spring and we want to "
                "find a quiet town to stay in for a week"
            ),
            "with a good bakery and a few easy walking trails and a small museum",
            (
                "not too far from the train station so please suggest which town "
                "we should look at first."
            ),
        ),
    ),
    Script("sh1", "short", "david", 0, ("What time is it right now?",)),
    Script("sh2", "short", "zira", 0, ("Please turn on the kitchen lights.",)),
    Script("sh3", "short", "mark", 0, ("Thank you that was very helpful.",)),
)


@dataclass(frozen=True)
class Variant:
    length: str
    npause: int
    pause_ms: int

    @property
    def key(self) -> str:
        return f"{self.length}|{self.npause}x{self.pause_ms}" if self.npause else f"{self.length}|0"


def variants(only: str | None) -> list[Variant]:
    out: list[Variant] = []
    for length in ("short", "8", "15"):
        if only and only != length:
            continue
        out.append(Variant(length, 0, 0))
        if length != "short":
            out += [Variant(length, n, p) for n in (1, 2) for p in PAUSES_MS]
    return out


def fragments_for(script: Script, npause: int) -> list[str]:
    c = script.clauses
    if npause == 0 or len(c) == 1:
        return [" ".join(c)]
    if npause == 1:
        return [c[0], " ".join(c[1:])]
    return list(c)


# ---------------------------------------------------------------- SAPI synthesis

_PS = r"""
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 -LiteralPath '@@MANIFEST@@' | ConvertFrom-Json
$bits = [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen
$mono = [System.Speech.AudioFormat.AudioChannel]::Mono
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, $bits, $mono)
foreach ($it in $items) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  try { $s.SelectVoice([string]$it.voice) } catch { }
  $s.Rate = [int]$it.rate
  $s.SetOutputToWaveFile([string]$it.path, $fmt)
  $s.Speak([string]$it.text)
  $s.Dispose()
}
"""


def frag_path(cache: Path, voice: str, rate: int, text: str) -> Path:
    h = hashlib.sha1(f"{VOICES[voice]}|{rate}|{text}".encode()).hexdigest()[:12]
    return cache / f"frag-{h}.wav"


def synthesise_missing(cache: Path, needed: set[tuple[str, int, str]]) -> None:
    todo = [(v, r, t) for v, r, t in sorted(needed) if not frag_path(cache, v, r, t).exists()]
    if not todo:
        return
    manifest = [
        {"path": str(frag_path(cache, v, r, t)), "voice": VOICES[v], "rate": r, "text": t}
        for v, r, t in todo
    ]
    mpath = cache / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    ps = _PS.replace("@@MANIFEST@@", str(mpath).replace("'", "''"))
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:
        sys.exit("SAPI synthesis needs Windows PowerShell (System.Speech)")
    enc = base64.b64encode(ps.encode("utf-16-le")).decode()
    subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-EncodedCommand", enc], check=True, timeout=600
    )


def read_fragment(path: Path) -> np.ndarray:
    """A SAPI wav as float32 mono at 16 kHz, trimmed to the speech plus a small margin."""
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if rate != RATE or ch != 1 or width != 2:
        raise RuntimeError(f"unexpected SAPI format {rate} Hz / {ch} ch / {width * 8} bit")
    a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    idx = np.flatnonzero(np.abs(a) > 0.004)
    if idx.size == 0:
        raise RuntimeError(f"silent fragment {path.name}")
    m = int(TRIM_MARGIN_S * RATE)
    a = a[max(0, idx[0] - m) : idx[-1] + m]
    return a * (SPEECH_PEAK / float(np.max(np.abs(a))))


def noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal(n) * NOISE_STD).astype(np.float32)


@dataclass
class Clip:
    cid: str
    variant: Variant
    script: Script
    pcm: np.ndarray  # int16, lead + speech + gaps (no tail)
    speech_end: int  # sample, clip-relative
    pauses: list[tuple[int, int]]  # (start sample, length samples), clip-relative
    text: str


def build_clip(cache: Path, script: Script, v: Variant) -> Clip:
    cid = f"{script.sid}-{v.key.split('|', 1)[1]}"
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    parts = [noise(int(LEAD_S * RATE), rng)]
    pos = len(parts[0])
    pauses: list[tuple[int, int]] = []
    frags = fragments_for(script, v.npause)
    for i, text in enumerate(frags):
        a = read_fragment(frag_path(cache, script.voice, script.rate, text))
        parts.append(a)
        pos += len(a)
        if i < len(frags) - 1:
            gap = int(v.pause_ms * RATE / 1000)
            parts.append(noise(gap, rng))
            pauses.append((pos, gap))
            pos += gap
    pcm = np.concatenate(parts)
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
    return Clip(cid, v, script, pcm16, pos, pauses, " ".join(script.clauses))


def build_plan(cache: Path, only: str | None) -> dict[str, list[Clip]]:
    """Variant key -> one clip per script of that length (reps rotate through them)."""
    vs = variants(only)
    needed = {
        (s.voice, s.rate, t)
        for v in vs
        for s in SCRIPTS
        if s.length == v.length
        for t in fragments_for(s, v.npause)
    }
    synthesise_missing(cache, needed)
    return {v.key: [build_clip(cache, s, v) for s in SCRIPTS if s.length == v.length] for v in vs}


def estimate_s(plan: dict[str, list[Clip]], reps: int) -> float:
    total = 0.0
    for clips in plan.values():
        for r in range(reps):
            c = clips[r % len(clips)]
            total += len(c.pcm) / RATE + TAIL_S + 0.4
    return total


# ---------------------------------------------------------------- server access


def load_token(path: str | None) -> str:
    if path:
        text = Path(path).read_text(encoding="utf-8").strip()
        m = re.search(r"^\s*(?:export\s+)?SPEECHFORGE_TOKEN\s*=\s*(.+?)\s*$", text, re.M)
        tok = m.group(1) if m else (text if "\n" not in text and "=" not in text else "")
        return tok.strip().strip("'\"")
    return os.environ.get("SPEECHFORGE_TOKEN", "")


def health(url: str) -> dict[str, Any]:
    base = re.sub(r"^ws", "http", url).split("/v1/")[0]
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:  # noqa: S310
            return json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


class Link:
    """One reusable session; reconnects (and waits out ``busy``) when it drops."""

    def __init__(self, url: str, token: str, wait_s: float = 900.0) -> None:
        self.url, self.token, self.wait_s = url, token, wait_s
        self.c: Any = None
        self.ready: dict[str, Any] = {}
        self.connects = 0

    def alive(self) -> bool:
        return self.c is not None and self.c.queues["_closed"].empty()

    async def ensure(self) -> Any:
        from speechforge.client import RigClient

        if self.alive():
            return self.c
        await self.drop()
        deadline = time.monotonic() + self.wait_s
        while True:
            c = RigClient(self.url, self.token)
            try:
                self.ready = await c.connect(ready_timeout=30)
            except Exception as exc:  # noqa: BLE001
                await c.close()
                msg = str(exc)
                print(f"  connect failed ({type(exc).__name__}: {msg[:80]}); retrying", flush=True)
                if time.monotonic() > deadline:
                    raise
                await asyncio.sleep(10)
                continue
            self.c = c
            self.connects += 1
            await c.send(
                {
                    "type": "stt.config",
                    "lang": "en",
                    "partial_ms": 160,
                    "eou": {"enabled": True, "min_pause_ms": 150},
                }
            )
            return c

    async def drop(self) -> None:
        if self.c is not None:
            with contextlib.suppress(Exception):
                await self.c.close()
        self.c = None


# ---------------------------------------------------------------- one clip run

STREAM_KINDS = ("stt.partial", "eou", "stt.final", "stt.vad", "error", "pong")


async def run_clip(link: Link, clip: Clip, rep: int) -> dict[str, Any]:
    c = await link.ensure()
    for k in STREAM_KINDS:
        c.drain(k)
    c.rtts.clear()
    fid = f"p{rep}-{clip.cid}"
    start = c.mic_pts
    tail = noise(int(TAIL_S * RATE), np.random.default_rng(rep + 1))
    pcm = np.concatenate([clip.pcm, (np.clip(tail, -1, 1) * 32767).astype("<i2")])
    t0 = time.perf_counter()
    t_end = await c.send_mic(pcm, realtime=True, mark_pts=start + clip.speech_end)
    if t_end is None:
        t_end = t0 + clip.speech_end / RATE
    t_flush = time.perf_counter()
    await c.send({"type": "stt.flush", "id": fid})
    final: dict[str, Any] | None = None
    while final is None or final.get("id") != fid:
        final = await c.next("stt.final", timeout=15)
    ms = lambda t: round((t - t0) * 1000, 1)  # noqa: E731
    errors = c.drain("error")
    rec = {
        "clip": clip.cid,
        "variant": clip.variant.key,
        "length": clip.variant.length,
        "npause": clip.variant.npause,
        "pause_ms": clip.variant.pause_ms,
        "script": clip.script.sid,
        "voice": clip.script.voice,
        "rep": rep,
        "text": clip.text,
        "mic_start": start,
        "speech_end": start + clip.speech_end,
        "pauses": [{"at": start + s, "len": n} for s, n in clip.pauses],
        "t_speech_end_ms": ms(t_end),
        "t_flush_ms": ms(t_flush),
        "eou": [
            {
                "p": e["p"],
                "at": e["at"],
                "pause_ms": e.get("pause_ms"),
                "ms": e.get("ms"),
                "t_ms": ms(e["_t"]),
            }
            for e in c.drain("eou")
        ],
        "partials": [
            {
                "text": " ".join(x for x in (p.get("committed", ""), p.get("tail", "")) if x),
                "lag_ms": p.get("lag_ms"),
                "ms": p.get("ms"),
                "upto": p.get("upto"),
                "stable_words": p.get("stable_words"),
                "punct_final": p.get("punct_final"),
                "t_ms": ms(p["_t"]),
            }
            for p in c.drain("stt.partial")
        ],
        "vad": [
            {"speech": v.get("speech"), "at": v.get("at"), "t_ms": ms(v["_t"])}
            for v in c.drain("stt.vad")
        ],
        "final": {
            "text": final.get("text", ""),
            "ms": final.get("ms"),
            "upto": final.get("upto"),
            "n_words": len(final.get("words") or []),
            "wall_ms": round((final["_t"] - t_flush) * 1000, 1),
            "t_ms": ms(final["_t"]),
        },
        "errors": [{"code": e.get("code"), "fatal": e.get("fatal")} for e in errors],
        "rtt_max_ms": round(max(c.rtts), 1) if c.rtts else None,
    }
    c.drain("pong")
    return rec


# ---------------------------------------------------------------- scoring


_PUNCT_END = re.compile(r"[.?!][\"')\]]*\s*$")


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def overlap(ref: str, hyp: str) -> float:
    r, h = Counter(norm_words(ref)), Counter(norm_words(hyp))
    return sum((r & h).values()) / max(1, sum(r.values()))


def classify(rec: dict[str, Any]) -> dict[str, Any]:
    """Assign each eou to an inserted pause, the true end, or 'other' (a pause inside speech)."""
    pauses = rec["pauses"]
    end = rec["speech_end"]
    per_pause: list[list[dict[str, Any]]] = [[] for _ in pauses]
    at_end: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for e in sorted(rec["eou"], key=lambda e: e["t_ms"]):
        if e["at"] < rec["mic_start"]:
            continue
        for i, p in enumerate(pauses):
            if p["at"] - 3200 <= e["at"] <= p["at"] + p["len"]:
                per_pause[i].append(e)
                break
        else:
            (at_end if e["at"] >= end - 8000 else other).append(e)
    return {"pause": per_pause, "end": at_end, "other": other}


def pct(vals: list[float], q: float) -> float | None:
    return float(np.percentile(vals, q)) if vals else None


def f1(x: float | None, nd: int = 0) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def lab(ln: str) -> str:
    return ln if ln == "short" else f"{ln} s"


def rate_cell(hits: int, n: int) -> str:
    return "-" if n == 0 else f"{hits}/{n} ({100 * hits / n:.0f}%)"


def summarise(records: list[dict[str, Any]], thresholds: tuple[float, ...]) -> str:
    ok = [r for r in records if "final" in r]
    cls = [(r, classify(r)) for r in ok]
    lens = [x for x in ("short", "8", "15") if any(r["length"] == x for r in ok)]
    out: list[str] = []
    n_err = sum(1 for r in records if "final" not in r)
    out.append(
        f"Clip runs scored: {len(ok)} ({n_err} failed), "
        f"inserted pauses: {sum(len(r['pauses']) for r in ok)}.\n"
    )

    # 1. false end of turn at inserted pauses
    out.append("### False end-of-turn at inserted mid-sentence pauses\n")
    out.append(
        "A pause counts as a false end of turn when an `eou` with `p >= threshold` fired at it.\n"
    )
    rows: list[list[str]] = []

    def pause_stats(sel: Any) -> tuple[int, list[float | None]]:
        ps: list[float | None] = []
        for r, c in cls:
            for i, evs in enumerate(c["pause"]):
                if sel(r, i):
                    ps.append(max((e["p"] for e in evs), default=None))
        return len(ps), ps

    for ln in [x for x in lens if x != "short"]:
        for pm in (*PAUSES_MS, 0):
            sel = (
                (lambda r, i, ln=ln, pm=pm: r["length"] == ln and r["pause_ms"] == pm)
                if pm
                else (lambda r, i, ln=ln: r["length"] == ln)
            )
            n, ps = pause_stats(sel)
            rows.append(
                [lab(ln), str(pm) if pm else "all", str(n)]
                + [rate_cell(sum(1 for p in ps if p is not None and p >= t), n) for t in thresholds]
                + [str(sum(1 for p in ps if p is None))]
            )
    for pm in PAUSES_MS:
        n, ps = pause_stats(lambda r, i, pm=pm: r["pause_ms"] == pm)
        rows.append(
            ["all", str(pm), str(n)]
            + [rate_cell(sum(1 for p in ps if p is not None and p >= t), n) for t in thresholds]
            + [str(sum(1 for p in ps if p is None))]
        )
    n, ps = pause_stats(lambda r, i: True)
    rows.append(
        ["all", "all", str(n)]
        + [rate_cell(sum(1 for p in ps if p is not None and p >= t), n) for t in thresholds]
        + [str(sum(1 for p in ps if p is None))]
    )
    out.append(
        md_table(
            ["utterance", "pause ms", "pauses"] + [f"p >= {t:g}" for t in thresholds] + ["no eou"],
            rows,
        )
    )

    # 2. distribution of p
    out.append("\n### Distribution of `p`\n")
    out.append(
        "Mid-sentence pauses (an `eou` that fired there) against the true end of the utterance "
        "and against pauses inside speech that were not inserted.\n"
    )
    drows: list[list[str]] = []

    def dist_row(label: str, vals: list[float], misses: int = 0) -> None:
        drows.append(
            [label, str(len(vals)), str(misses)]
            + [f2(pct(vals, q)) for q in (10, 25, 50, 75, 90)]
            + [f2(max(vals) if vals else None)]
        )

    def f2(x: float | None) -> str:
        return f1(x, 2)

    for pm in PAUSES_MS:
        vals = [
            e["p"] for r, c in cls if r["pause_ms"] == pm for evs in c["pause"] for e in evs[:1]
        ]
        npaus = sum(len(r["pauses"]) for r, _ in cls if r["pause_ms"] == pm)
        dist_row(f"mid-sentence {pm} ms", vals, npaus - len(vals))
    vals = [e["p"] for r, c in cls for evs in c["pause"] for e in evs[:1]]
    dist_row("mid-sentence, all", vals, sum(len(r["pauses"]) for r, _ in cls) - len(vals))
    for ln in lens:
        for label, sel in (
            (
                f"true end, {lab(ln)}, clean",
                lambda r, ln=ln: r["length"] == ln and r["npause"] == 0,
            ),
            (
                f"true end, {lab(ln)}, after pauses",
                lambda r, ln=ln: r["length"] == ln and r["npause"],
            ),
        ):
            sub = [(r, c) for r, c in cls if sel(r)]
            if not sub:
                continue
            vals = [c["end"][0]["p"] for _, c in sub if c["end"]]
            dist_row(label, vals, sum(1 for _, c in sub if not c["end"]))
    dist_row("other pauses inside speech", [e["p"] for _, c in cls for e in c["other"]])
    out.append(md_table(["where", "n", "no eou", "p10", "p25", "p50", "p75", "p90", "max"], drows))

    # 2b. p by script (the synthetic voices differ a lot)
    out.append("\n### `p` by script (synthetic voices differ)\n")
    out.append(
        "Median (min-max) of `p` at the inserted pauses and at the true end, over every rep "
        "and pause variant of the script. A low true-end `p` is a property of the synthetic "
        "clip as Smart Turn hears it, not of the pipeline.\n"
    )
    srows: list[list[str]] = []
    for sid in sorted({r["script"] for r in ok}):
        sub = [(r, c) for r, c in cls if r["script"] == sid]
        mids = [evs[0]["p"] for _, c in sub for evs in c["pause"] if evs]
        ends = [c["end"][0]["p"] for _, c in sub if c["end"]]

        def span(v: list[float]) -> str:
            return f"{statistics.median(v):.2f} ({min(v):.2f}-{max(v):.2f})" if v else "-"

        srows.append(
            [sid, sub[0][0]["voice"], str(len(sub)), str(len(mids)), span(mids), span(ends)]
        )
    out.append(
        md_table(["script", "voice", "runs", "pauses", "p at pauses", "p at true end"], srows)
    )

    # 2c. does the text at the eou end in terminal punctuation?
    out.append("\n### `p` against the punctuation of the latest partial\n")
    out.append(
        "The newest `stt.partial` received before each `eou`: does its text end in `.`, `?` or "
        "`!`? (The inserted pauses follow unpunctuated clauses; the recogniser may still add "
        "punctuation.)\n"
    )

    def punct_at(r: dict[str, Any], e: dict[str, Any]) -> bool:
        prev = [p for p in r["partials"] if p["t_ms"] <= e["t_ms"]]
        return bool(prev) and bool(_PUNCT_END.search(prev[-1]["text"]))

    groups: dict[str, list[tuple[float, bool]]] = {}
    for r, c in cls:
        for evs in c["pause"]:
            for e in evs[:1]:
                k = "mid-sentence pause" + (", punctuated" if punct_at(r, e) else ", unpunctuated")
                groups.setdefault(k, []).append((e["p"], True))
        for e in c["end"][:1]:
            k = "true end" + (", punctuated" if punct_at(r, e) else ", unpunctuated")
            groups.setdefault(k, []).append((e["p"], True))
    prows: list[list[str]] = []
    for k in sorted(groups):
        ps = [p for p, _ in groups[k]]
        prows.append(
            [k, str(len(ps)), f1(pct(ps, 50), 2)]
            + [rate_cell(sum(1 for p in ps if p >= t), len(ps)) for t in thresholds]
        )
    out.append(md_table(["where", "n", "p50"] + [f"p >= {t:g}" for t in thresholds], prows))

    # 3. eot latency
    out.append("\n### End-of-turn latency (true speech end to first qualifying `eou`)\n")
    out.append(
        "Wall time from the moment the frame holding the last speech sample left the client to "
        "the arrival of the `eou`. `miss` = no `eou` with `p >= threshold` at the end.\n"
    )
    lrows: list[list[str]] = []
    for t in (0.0, *thresholds):
        for ln in lens:
            sub = [c for r, c in cls if r["length"] == ln]
            ts_end = {id(c): r["t_speech_end_ms"] for r, c in cls if r["length"] == ln}
            lat, miss = [], 0
            for c in sub:
                ev = next((e for e in c["end"] if e["p"] >= t), None)
                if ev is None:
                    miss += 1
                else:
                    lat.append(ev["t_ms"] - ts_end[id(c)])
            lrows.append(
                ["any eou" if t == 0 else f"p >= {t:g}", lab(ln), str(len(sub)), str(miss)]
                + [f1(pct(lat, q)) for q in (50, 90)]
                + [f1(max(lat) if lat else None)]
            )
    out.append(
        md_table(["threshold", "utterance", "n", "miss", "p50 ms", "p90 ms", "max ms"], lrows)
    )
    head = thresholds[0]
    out.append(f"\nPer clip class at `p >= {head:g}`:\n")
    crows: list[list[str]] = []
    for key in sorted({r["variant"] for r in ok}, key=lambda k: (k.split("|")[0] != "short", k)):
        sub = [(r, c) for r, c in cls if r["variant"] == key]
        lat, miss = [], 0
        for r, c in sub:
            ev = next((e for e in c["end"] if e["p"] >= head), None)
            if ev is None:
                miss += 1
            else:
                lat.append(ev["t_ms"] - r["t_speech_end_ms"])
        crows.append(
            [key.replace("|", "/"), str(len(sub)), str(miss)]
            + [f1(pct(lat, q)) for q in (50, 90)]
            + [f1(max(lat) if lat else None)]
        )
    out.append(md_table(["class", "n", "miss", "p50 ms", "p90 ms", "max ms"], crows))

    # 4. stt.final and partials
    out.append("\n### `stt.final` cost, flush wall time and partials\n")
    out.append(
        "`ms` is the server's figure in the `stt.final` frame; wall is `stt.flush` sent to "
        "`stt.final` received, 1.5 s after the speech end. Partials are per utterance.\n"
    )
    frows: list[list[str]] = []
    for ln in lens:
        sub = [r for r in ok if r["length"] == ln]
        fm = [r["final"]["ms"] for r in sub if r["final"]["ms"] is not None]
        fw = [r["final"]["wall_ms"] for r in sub]
        npart = [len(r["partials"]) for r in sub]
        lag = [p["lag_ms"] for r in sub for p in r["partials"] if p["lag_ms"] is not None]
        frows.append(
            [lab(ln), str(len(sub))]
            + [f1(pct(fm, q)) for q in (50, 90, 99)]
            + [f1(max(fm) if fm else None)]
            + [f1(pct(fw, q)) for q in (50, 90, 99)]
            + [f1(max(fw) if fw else None)]
            + [f1(statistics.median(npart) if npart else None)]
            + [f1(pct(lag, q)) for q in (50, 90, 99)]
        )
    out.append(
        md_table(
            [
                "utterance",
                "n",
                "ms p50",
                "ms p90",
                "ms p99",
                "ms max",
                "wall p50",
                "wall p90",
                "wall p99",
                "wall max",
                "partials/utt",
                "lag p50",
                "lag p90",
                "lag p99",
            ],
            frows,
        )
    )

    # 5. text sanity
    out.append("\n### Final text against the script (word overlap)\n")
    trows: list[list[str]] = []
    for ln in lens:
        ov = [overlap(r["text"], r["final"]["text"]) * 100 for r in ok if r["length"] == ln]
        trows.append(
            [
                lab(ln),
                str(len(ov)),
                f1(statistics.mean(ov) if ov else None),
                f1(min(ov) if ov else None),
                str(sum(1 for x in ov if x < 80)),
            ]
        )
    out.append(md_table(["utterance", "n", "mean %", "min %", "runs under 80 %"], trows))
    bad = [r for r in ok if overlap(r["text"], r["final"]["text"]) < 0.8]
    if bad:
        out.append("\nLowest-overlap runs (final text as heard):\n")
        for r in sorted(bad, key=lambda r: overlap(r["text"], r["final"]["text"]))[:5]:
            out.append(f"* `{r['clip']}` rep {r['rep']}: {r['final']['text']!r}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main


async def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(out, args.only)
    est = estimate_s(plan, args.reps)
    print(f"{len(plan)} clip classes, {args.reps} reps, estimated {est / 60:.1f} min", flush=True)
    for key, clips in plan.items():
        secs = ", ".join(f"{len(c.pcm) / RATE:.1f}" for c in clips)
        print(f"  {key:10s} speech+pauses (s) per script: {secs}", flush=True)
    if args.plan:
        return
    if est > BUDGET_S:
        print(f"warning: estimate exceeds the {BUDGET_S // 60} min budget", flush=True)
    token = load_token(args.token_file)
    h = health(args.url)
    if not h.get("ok"):
        sys.exit(f"health check failed ({h.get('error', 'not ok')})")
    link = Link(args.url, token)
    rec_path = out / "records.jsonl"
    started = time.time()
    done = 0
    for rep in range(args.reps):
        for key, clips in plan.items():
            clip = clips[rep % len(clips)]
            rec: dict[str, Any] | None = None
            for attempt in (1, 2):
                try:
                    rec = await run_clip(link, clip, rep)
                    break
                except Exception as exc:  # noqa: BLE001 - reconnect and retry once
                    print(f"  {clip.cid} rep {rep}: {type(exc).__name__}", flush=True)
                    await link.drop()
                    rec = {
                        "clip": clip.cid,
                        "variant": key,
                        "rep": rep,
                        "error": type(exc).__name__,
                        "attempt": attempt,
                    }
                    with rec_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    await asyncio.sleep(3)
            if rec is not None and "final" in rec:
                with rec_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            done += 1
            if done % 10 == 0:
                print(
                    f"  {done}/{len(plan) * args.reps} clips, "
                    f"{(time.time() - started) / 60:.1f} min",
                    flush=True,
                )
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "health": h,
        "engines": link.ready.get("engines"),
        "gpu": link.ready.get("gpu"),
        "limits": link.ready.get("limits"),
        "reps": args.reps,
        "connects": link.connects,
        "elapsed_s": round(time.time() - started),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    await link.drop()
    report(rec_path, out, tuple(args.threshold or DEFAULT_THRESHOLDS))


def report(rec_path: Path, out: Path | None, thresholds: tuple[float, ...]) -> None:
    records = [json.loads(line) for line in rec_path.read_text(encoding="utf-8").splitlines()]
    md = summarise(records, thresholds)
    print(md)
    target = (out or rec_path.parent) / "summary.md"
    target.write_text(md, encoding="utf-8")
    print(f"summary written to {target.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--url", default="ws://127.0.0.1:1236/v1/realtime")
    ap.add_argument("--token-file", default=None, help="env file with SPEECHFORGE_TOKEN=...")
    ap.add_argument("--out", default="eot-probe", help="output directory")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", choices=("short", "8", "15"), default=None)
    ap.add_argument(
        "--threshold",
        type=float,
        action="append",
        help="p threshold to report (repeatable; default 0.5 0.7 0.9; the first is the headline)",
    )
    ap.add_argument("--plan", action="store_true", help="build the clips, print the plan, stop")
    ap.add_argument("--summarise", metavar="RECORDS", help="re-score an existing records.jsonl")
    args = ap.parse_args()
    if args.summarise:
        report(Path(args.summarise), None, tuple(args.threshold or DEFAULT_THRESHOLDS))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
