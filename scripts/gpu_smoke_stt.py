"""GPU smoke test for an STT engine: stream a WAV through the real adapter the way
a session does and print the partial cadence, the final text and the word
timings. For the operator on the GPU host; the unit suite never runs this.

    uv run --extra gpu --extra stt-nemotron python scripts/gpu_smoke_stt.py \\
        --engine nemotron --device 3 --wav clip.wav \\
        [--backend subprocess --python <venv>/Scripts/python.exe]
    uv run --extra gpu python scripts/gpu_smoke_stt.py --engine parakeet --device 3 --wav clip.wav

Reads the normal config (SPEECHFORGE_DATA_DIR / --data-dir) so the model files
are the ones `speechforge fetch-models` verified; command-line flags override
``gpu.provider``, ``stt.engine`` and the ``stt.nemotron`` knobs. Loads the STT
adapter ALONE (no TTS, no lease): run it only on a card nothing else is using.
Prints lines like

    partial  upto=  1.60s  lag=  14ms  cost=  9ms  "the quick brown"
    ...
    final    cost= 31ms  "The quick brown fox jumps today."
    words    The 0.000-0.320  quick 0.320-0.560 ...
    cadence  partials=18  mean_gap=160ms  p90_cost=12ms  rtf=0.07

Exit code 0 when a final was produced, 1 otherwise.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speechforge.config import load_config  # noqa: E402
from speechforge.engines import base  # noqa: E402
from speechforge.engines.base import is_streaming_stt  # noqa: E402


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: need 16-bit PCM, got {8 * width}-bit")
    pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    if rate != 16000:
        n = int(len(pcm) * 16000 / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype(
            np.float32
        )
    return pcm


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine", default="nemotron", help="stt.engine (nemotron | parakeet)")
    ap.add_argument("--device", type=int, default=None, help="CUDA ordinal (PCI-bus order)")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--provider", choices=["cuda", "cpu"], default=None)
    ap.add_argument(
        "--partial-ms", type=int, default=160, help="feed cadence (realtime.partial_ms)"
    )
    ap.add_argument("--realtime", action="store_true", help="pace the feed at wall-clock speed")
    ap.add_argument("--backend", default=None, help="stt.nemotron.backend")
    ap.add_argument("--python", default=None, help="stt.nemotron.python (subprocess backend)")
    ap.add_argument("--quantization", default=None, help="stt.nemotron.quantization (int8)")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()

    config = load_config(data_dir=args.data_dir)
    config.stt.engine = args.engine
    if args.provider:
        config.gpu.provider = args.provider
    nem = config.stt.nemotron
    if args.backend:
        nem.backend = args.backend
    if args.python:
        nem.python = args.python
    if args.quantization:
        nem.quantization = args.quantization
    device = args.device if args.device is not None else config.gpu.devices[0]

    audio = read_wav(args.wav)
    print(
        f"clip     {args.wav.name}: {len(audio) / 16000:.2f}s, engine={args.engine}, "
        f"provider={config.gpu.provider}, device={device}"
    )

    if config.gpu.provider == "cuda":
        import os

        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if args.engine == "parakeet":
            import onnxruntime as ort

            preload = getattr(ort, "preload_dlls", None)
            if callable(preload):
                preload()
    base.builtin()
    t0 = time.perf_counter()
    stt = base.STT_ENGINES[args.engine](config, device)
    print(f"loaded   {time.perf_counter() - t0:.1f}s  {stt.describe()}")
    t0 = time.perf_counter()
    stt.warm_up()
    print(f"warm-up  {(time.perf_counter() - t0) * 1000:.0f}ms")

    step = 16000 * args.partial_ms // 1000
    costs: list[float] = []
    gaps: list[float] = []
    last_text = None
    final_text = ""
    words: list[tuple[str, float, float]] = []
    wall0 = time.perf_counter()
    try:
        if is_streaming_stt(stt):
            print("mode     streaming (feed per partial, finalize at the end)")
            h = stt.create_stream()
            prev = None
            for i in range(0, len(audio), step):
                chunk = audio[i : i + step]
                if args.realtime:
                    due = wall0 + (i + len(chunk)) / 16000
                    while time.perf_counter() < due:
                        time.sleep(0.002)
                t1 = time.perf_counter()
                text = h.feed(chunk)
                cost = (time.perf_counter() - t1) * 1000
                costs.append(cost)
                if prev is not None:
                    gaps.append((t1 - prev) * 1000)
                prev = t1
                upto = (i + len(chunk)) / 16000
                lag = (time.perf_counter() - wall0 - upto) * 1000 if args.realtime else 0.0
                if text != last_text:
                    print(
                        f"partial  upto={upto:6.2f}s  lag={lag:5.0f}ms  "
                        f'cost={cost:4.0f}ms  "{text}"'
                    )
                    last_text = text
            t1 = time.perf_counter()
            final_text, words = h.finalize()
            fcost = (time.perf_counter() - t1) * 1000
            if hasattr(h, "close"):
                h.close()
        else:
            print("mode     rolling re-decode (transcribe the growing tail per partial)")
            prev = None
            for end in range(step, len(audio) + step, step):
                tail = audio[:end]
                if args.realtime:
                    due = wall0 + min(end, len(audio)) / 16000
                    while time.perf_counter() < due:
                        time.sleep(0.002)
                t1 = time.perf_counter()
                text = stt.transcribe(tail)
                cost = (time.perf_counter() - t1) * 1000
                costs.append(cost)
                if prev is not None:
                    gaps.append((t1 - prev) * 1000)
                prev = t1
                if text != last_text:
                    print(
                        f"partial  upto={min(end, len(audio)) / 16000:6.2f}s  "
                        f'cost={cost:4.0f}ms  "{text}"'
                    )
                    last_text = text
            t1 = time.perf_counter()
            fn = getattr(stt, "transcribe_words", None)
            final_text, words = fn(audio) if fn else (stt.transcribe(audio), [])
            fcost = (time.perf_counter() - t1) * 1000
        total = time.perf_counter() - wall0
        print(f'final    cost={fcost:4.0f}ms  "{final_text}"')
        if words:
            print("words    " + "  ".join(f"{w} {s:.3f}-{e:.3f}" for w, s, e in words))
        else:
            print("words    (none)")
        if costs:
            srt = sorted(costs)
            p90 = srt[min(len(srt) - 1, int(0.9 * len(srt)))]
            gap = statistics.mean(gaps) if gaps else 0.0
            print(
                f"cadence  partials={len(costs)}  mean_gap={gap:.0f}ms  "
                f"mean_cost={statistics.mean(costs):.0f}ms  "
                f"p90_cost={p90:.0f}ms  max_cost={srt[-1]:.0f}ms  "
                f"rtf={sum(costs) / 1000 / (len(audio) / 16000):.3f}  "
                f"wall={total:.2f}s"
            )
    finally:
        stt.close()
    return 0 if final_text else 1


if __name__ == "__main__":
    sys.exit(main())
