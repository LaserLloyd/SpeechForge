"""Step 5a: the synthesised-speech WER check (SPEC §11.3: TTS -> STT <= 3 %).

    uv run python -m tools.listening.wer --url http://127.0.0.1:1236 \\
        --engines <renders>/chatterbox <renders>/qwen3-tts ...

Every ``NN.wav`` in each render directory is transcribed by the RUNNING
SpeechForge (``POST /v1/audio/transcriptions``, token from ``SPEECHFORGE_TOKEN``)
and scored against the matching line of ``lines.txt``. Prints a per-engine table
and writes ``<dir>/wer.json`` (per-line WER and the transcript) next to the audio.

Normalisation is the server's own (``selftest.words``: lower-case, ``[a-z0-9']+``
tokens) preceded by Chatterbox's typographic tidy (``normalise_text``: curly
quotes, ellipses and dashes to ASCII), plus two things a rendered line needs that
a dictated one does not: bracketed performance tags (``[laugh]``) are dropped,
because an engine may voice or ignore them, and small numbers are spelled out on
both sides so ``7:15`` and ``seven fifteen`` agree.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import httpx

from speechforge.engines.chatterbox import normalise_text
from speechforge.selftest import words
from tools.listening import kit

ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]  # fmt: skip
TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    5: "fifth",
    8: "eighth",
    9: "ninth",
    12: "twelfth",
}
TAG_RE = re.compile(r"\[[^\]]{1,24}\]|<[^>]{1,24}>")
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
ORDINAL_RE = re.compile(r"\b(\d{1,3})(st|nd|rd|th)\b")


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + (f" {ONES[o]}" if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{ONES[h]} hundred" + (f" {number_words(r)}" if r else "")
    return str(n)


def ordinal_words(n: int) -> str:
    if n in ORDINALS:
        return ORDINALS[n]
    if n < 20:
        return ONES[n] + "th"
    t, o = divmod(n, 10)
    if o == 0 and n < 100:
        return TENS[t][:-1] + "ieth"
    if n < 100:
        return f"{TENS[t]} {ordinal_words(o)}"
    return number_words(n) + "th"


def _time(m: re.Match) -> str:
    h, mm = int(m.group(1)), int(m.group(2))
    if mm == 0:
        return f"{number_words(h)} o'clock"
    if mm < 10:
        return f"{number_words(h)} oh {number_words(mm)}"
    return f"{number_words(h)} {number_words(mm)}"


def normalise_for_wer(text: str) -> list[str]:
    t = TAG_RE.sub(" ", text or "")
    t = normalise_text(t)
    t = TIME_RE.sub(_time, t)
    t = ORDINAL_RE.sub(lambda m: ordinal_words(int(m.group(1))), t)
    t = re.sub(r"\b\d{1,3}\b", lambda m: number_words(int(m.group(0))), t)
    return words(t)


def edit_distance(r: list[str], h: list[str]) -> int:
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)]


def wer(ref: str, hyp: str) -> float:
    r, h = normalise_for_wer(ref), normalise_for_wer(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


def transcribe(
    client: httpx.Client, url: str, headers: dict[str, str], wav: bytes, *, retries: int, sleep=None
) -> str:
    def call():
        return client.post(
            f"{url.rstrip('/')}/v1/audio/transcriptions",
            headers=headers,
            files={"file": ("line.wav", wav, "audio/wav")},
            data={"response_format": "json"},
        )

    kwargs = {"sleep": sleep} if sleep else {}
    r = kit.with_retries(
        call,
        retries=retries,
        is_retryable=lambda r: r.status_code == 503,
        headers_of=lambda r: r.headers,
        on_wait=lambda w, k: print(
            f"  server not ready (503), retry {k} in {w:.0f} s", file=sys.stderr
        ),
        **kwargs,
    )
    if r.status_code != 200:
        raise SystemExit(
            f"transcription failed: HTTP {r.status_code} {kit.error_message(r.content)}"
        )
    return r.json().get("text", "")


def score_dir(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    d: Path,
    lines: list[str],
    *,
    retries: int,
) -> dict:
    rows = []
    for n, text in enumerate(lines, 1):
        p = d / f"{kit.line_name(n)}.wav"
        if not p.exists():
            continue
        hyp = transcribe(client, url, headers, p.read_bytes(), retries=retries)
        rows.append({"n": n, "wer": round(wer(text, hyp), 4), "transcript": hyp})
    mean = sum(r["wer"] for r in rows) / len(rows) if rows else None
    return {
        "engine": d.name,
        "lines": rows,
        "wer_mean": round(mean, 4) if mean is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True)
    ap.add_argument("--engines", nargs="+", required=True, type=Path, metavar="DIR")
    ap.add_argument("--lines", type=Path, default=None)
    ap.add_argument("--retries", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.03, help="SPEC 11.3 gate (3 %%)")
    args = ap.parse_args(argv)
    lines = kit.read_lines(args.lines)
    headers = kit.auth_headers(args.url)
    ok = True
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        print(f"{'engine':16} {'WER':>7}  lines")
        for d in args.engines:
            res = score_dir(client, args.url, headers, d, lines, retries=args.retries)
            kit.json_dump(d / "wer.json", res)
            m = res["wer_mean"]
            flag = "" if m is None or m <= args.threshold else f"  FAIL (> {args.threshold:.0%})"
            ok &= not flag
            worst = sorted(res["lines"], key=lambda r: -r["wer"])[:3]
            print(f"{d.name:16} {m if m is None else f'{m:.1%}':>7}  {len(res['lines'])}{flag}")
            for w in worst:
                if w["wer"] > 0:
                    print(f"    line {kit.line_name(w['n'])}: {w['wer']:.0%}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
