"""Step 1: render every line through a RUNNING SpeechForge, one engine at a time.

    uv run python -m tools.listening.render_set --url http://127.0.0.1:1236 \\
        --engine chatterbox --voice sha256:<id> --out <renders>

writes ``<renders>/<engine>/01.wav`` ... and ``<renders>/<engine>/render.json`` with
the time to first audio and the wall time of every line. The server decides which
engine speaks (``tts.engine`` in its config); ``--engine`` names the output
directory and is sent as the request's ``model`` so a server that routes on it
can. Run it once per engine, restarting or reconfiguring the server in between,
then feed the directories to ``make_set``.

The bearer token comes from ``SPEECHFORGE_TOKEN`` in the environment only (never a
flag, never printed). A ``503`` is retried after its ``Retry-After`` -- the first
request usually finds the models cold and kicks the load (DECISIONS S2).

Timing: the request streams (``stream: true``, ``response_format: wav``); the 200
is sent with the first audio (S7), so the time to the response headers is the
time to first audio as a client sees it, and the time to the last byte is the
wall time. The streamed WAV header carries placeholder sizes and is rewritten
on save so ordinary tools can read the files.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from tools.listening import kit


def speak_once(
    client: httpx.Client, url: str, body: dict, headers: dict[str, str]
) -> tuple[int, dict, bytes, float, float]:
    """One POST; returns (status, headers, body, ttfa_s, wall_s)."""
    t0 = time.perf_counter()
    with client.stream(
        "POST", f"{url.rstrip('/')}/v1/audio/speech", json=body, headers=headers
    ) as r:
        t_first = time.perf_counter()
        chunks = []
        for chunk in r.iter_bytes():
            if chunk:
                if len(chunks) == 0:
                    t_first = time.perf_counter()
                chunks.append(chunk)
        t_end = time.perf_counter()
        return r.status_code, dict(r.headers), b"".join(chunks), t_first - t0, t_end - t0


def render_all(
    url: str,
    engine: str,
    voice: str,
    lines: list[str],
    out_dir: Path,
    *,
    seed: int | None,
    retries: int,
    timeout_s: float,
    client: httpx.Client | None = None,
    sleep=time.sleep,
) -> dict:
    headers = kit.auth_headers(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))
    try:
        info: dict = {"version": None, "engines": None}
        try:
            ready = client.get(f"{url.rstrip('/')}/ready", headers=headers)
            if ready.status_code == 200:
                data = ready.json()
                info = {"version": data.get("version"), "engines": data.get("engines")}
        except httpx.HTTPError:
            pass
        results = []
        for n, text in enumerate(lines, 1):
            body = {
                "model": engine,
                "input": text,
                "voice": voice,
                "response_format": "wav",
                "stream": True,
            }
            if seed is not None:
                body["seed"] = seed + n

            def call(body=body):
                return speak_once(client, url, body, headers)

            status, rh, data, ttfa, wall = kit.with_retries(
                call,
                retries=retries,
                is_retryable=lambda r: r[0] == 503,
                headers_of=lambda r: r[1],
                sleep=sleep,
                on_wait=lambda w, k, n=n: print(
                    f"  line {kit.line_name(n)}: server not ready (503), retry {k} in {w:.0f} s",
                    file=sys.stderr,
                ),
            )
            if status != 200:
                raise SystemExit(
                    f"line {kit.line_name(n)}: HTTP {status} {kit.error_message(data)}"
                )
            pcm, rate = kit.streamed_wav_to_pcm(data)
            audio = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0
            kit.write_wav(out_dir / f"{kit.line_name(n)}.wav", audio, rate)
            audio_s = kit.duration_s(audio, rate)
            results.append(
                {
                    "n": n,
                    "file": f"{kit.line_name(n)}.wav",
                    "text": text,
                    "ttfa_ms": round(ttfa * 1000, 1),
                    "wall_ms": round(wall * 1000, 1),
                    "audio_s": audio_s,
                    "rtf": round(wall / audio_s, 3) if audio_s else None,
                    "sample_rate": rate,
                }
            )
            print(
                f"  {kit.line_name(n)}  ttfa {ttfa * 1000:7.1f} ms  wall {wall * 1000:7.1f} ms"
                f"  audio {audio_s:5.2f} s",
                file=sys.stderr,
            )
    finally:
        if own:
            client.close()
    ttfas = [r["ttfa_ms"] for r in results]
    render = {
        "engine": engine,
        "voice": voice,
        "server": info,
        "seed": seed,
        "lines": results,
        "summary": {
            "n": len(results),
            "ttfa_p50_ms": round(float(np.percentile(ttfas, 50)), 1) if ttfas else None,
            "ttfa_p90_ms": round(float(np.percentile(ttfas, 90)), 1) if ttfas else None,
            "wall_mean_ms": round(float(np.mean([r["wall_ms"] for r in results])), 1)
            if results
            else None,
            "rtf_mean": round(
                float(np.mean([r["rtf"] for r in results if r["rtf"] is not None])), 3
            )
            if any(r["rtf"] is not None for r in results)
            else None,
        },
    }
    kit.json_dump(out_dir / "render.json", render)
    return render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:1236")
    ap.add_argument(
        "--engine",
        required=True,
        help="name for the output directory (chatterbox, qwen3-tts, dots-tts)",
    )
    ap.add_argument("--voice", required=True, help="voice id the server knows, e.g. sha256:<id>")
    ap.add_argument(
        "--out", required=True, type=Path, help="renders directory; <out>/<engine>/ is written"
    )
    ap.add_argument(
        "--lines", type=Path, default=None, help="lines file (default: tools/listening/lines.txt)"
    )
    ap.add_argument(
        "--seed", type=int, default=1000, help="base seed; line n uses seed+n (--seed -1: none)"
    )
    ap.add_argument("--retries", type=int, default=40, help="how many 503s to wait out per line")
    ap.add_argument(
        "--timeout", type=float, default=120.0, help="read timeout per request, seconds"
    )
    args = ap.parse_args(argv)
    lines = kit.read_lines(args.lines)
    seed = None if args.seed is not None and args.seed < 0 else args.seed
    print(
        f"rendering {len(lines)} lines as '{args.engine}' -> {args.out / args.engine}",
        file=sys.stderr,
    )
    render = render_all(
        args.url,
        args.engine,
        args.voice,
        lines,
        args.out / args.engine,
        seed=seed,
        retries=args.retries,
        timeout_s=args.timeout,
    )
    s = render["summary"]
    print(
        f"done: {s['n']} lines, ttfa p50 {s['ttfa_p50_ms']} ms, p90 {s['ttfa_p90_ms']} ms, "
        f"rtf {s['rtf_mean']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
