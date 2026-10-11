"""GPU smoke test for the dots-tts adapter: load, clone, speak, time it.

Run on the GPU host from the repo root, inside the environment that has the
torch extra and the dots.tts package (docs/ENGINES.md lists them)::

    uv run python scripts/gpu_smoke_dots_tts.py --device 3 --voice ref.wav \\
        --transcript "exact words spoken in ref.wav" \\
        --text "Sure, I can help with that." --out dots.wav

``--voice`` is either a ``dotstts-ref`` ``.npz`` (``make_voice``) or a 16-bit
WAV reference clip (0.5-20 s; ~10 s of natural talking works best) with
``--transcript`` (omit it for the weaker x-vector-only cloning). The script
builds a Config with ``voices.allow_reference_clip: true`` (D3 -- this is the
owner's decision, taken here for the measurement only), loads the engine on
``--device``, speaks ``--text`` ``--runs`` times (the first run pays the
warm-up) and prints TTFA / RTF / audio length / VRAM for each. It never
touches a StudioForge lease: take one by hand (or pick an idle card) first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", type=int, default=3, help="CUDA ordinal (PCI-bus order)")
    ap.add_argument(
        "--voice", required=True, help="dotstts-ref .npz, or a 16-bit WAV reference clip"
    )
    ap.add_argument("--transcript", default="", help="exact transcript of the WAV reference")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True, help="WAV written from the last run")
    ap.add_argument("--models-dir", default=None, help="default: <SPEECHFORGE_DATA_DIR>/models")
    ap.add_argument("--backend", choices=["native", "http"], default="native")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="http backend (loopback only)")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--num-steps", type=int, default=4)
    ap.add_argument("--optimize", action="store_true", help="torch.compile + warm-up at load")
    ap.add_argument("--output-rate", type=int, choices=[24000, 48000], default=24000)
    ap.add_argument("--double-streaming", action="store_true", help="the -2steps-stts artifact")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--verify", action="store_true", help="check the files against models.lock")
    ap.add_argument("--save-voice", default=None, help="also write the built .npz voice here")
    args = ap.parse_args(argv)

    from speechforge.audio import read_wav, wav_bytes
    from speechforge.config import Config, default_data_dir
    from speechforge.engines import dots_tts

    dots = {
        "backend": args.backend,
        "url": args.url,
        "precision": args.precision,
        "num_steps": None if args.double_streaming else args.num_steps,
        "guidance_scale": None if args.double_streaming else 1.2,
        "optimize": args.optimize,
        "output_rate": args.output_rate,
        "double_streaming": args.double_streaming,
    }
    config = Config(
        data_dir=default_data_dir(),
        models={"dir": args.models_dir} if args.models_dir else {},
        gpu={"provider": "cuda", "devices": [args.device]},
        tts={"engine": "dots-tts", "dots": dots},
        voices={"allow_reference_clip": True},
        selftest={"enabled": False},
    )

    path = Path(args.voice)
    data = path.read_bytes()
    if path.suffix.lower() == ".wav":
        audio, rate = read_wav(data)
        data = dots_tts.make_voice(audio, rate, args.transcript)
        if args.save_voice:
            Path(args.save_voice).write_bytes(data)
    voice = dots_tts.parse_voice(data)
    print(
        f"voice: {voice.seconds:.1f} s @ {voice.rate} Hz, transcript "
        f"{'yes' if voice.transcript else 'NO (x-vector-only cloning)'}"
    )

    if args.verify and args.backend == "native":
        from speechforge import models

        print("verify:", models.verify(config, models.tts_groups(config)))

    t0 = time.perf_counter()
    tts = dots_tts.load(config, args.device)
    print(f"loaded in {time.perf_counter() - t0:.1f} s: {tts.describe()}")

    vram = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(args.device)
            vram = lambda: torch.cuda.max_memory_allocated(args.device) / 2**30  # noqa: E731
    except ImportError:
        pass

    pcm = np.zeros(0, "<i2")
    for run in range(1, max(1, args.runs) + 1):
        pieces: list[np.ndarray] = []
        t0 = time.perf_counter()
        first = None
        for piece in tts.stream(args.text, "smoke", voice, seed=args.seed):
            if first is None:
                first = time.perf_counter()
            pieces.append(piece)
        total = time.perf_counter() - t0
        pcm = np.concatenate(pieces) if pieces else pcm
        st = tts.last_stats
        wall = round((first - t0) * 1000) if first else None
        peak = f", peak vram {vram():.2f} GiB" if vram else ""
        print(
            f"run {run}: ttfa {st['ttfa_ms']} ms (wall {wall} ms), audio {st['audio_ms']} ms, "
            f"total {round(total * 1000)} ms, rtf {st['rtf']}, chunks {st['chunks']}{peak}"
        )
    Path(args.out).write_bytes(wav_bytes(pcm, tts.sample_rate))
    print(f"wrote {args.out}: {len(pcm) / tts.sample_rate:.2f} s @ {tts.sample_rate} Hz")
    tts.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
