#!/usr/bin/env python3
"""Load one TTS adapter straight from the registry on one CUDA device, speak a
line, print TTFA / RTF and write a WAV. No server, no lease, no StudioForge:
this is the phase-2 bake-off's measuring stick, run by hand on the GPU host.

    uv run --extra gpu --extra tts-torch python scripts/gpu_smoke_tts.py \\
        --engine qwen3-tts --device 3 --models-dir <data>/models \\
        --voice voice.npz --text "Sure, I can help with that." --out out.wav

Make the voice first (the clip and its transcript stay on this machine, D3):

    ... --engine qwen3-tts --build-voice clip.wav --transcript "What is said in the clip." \\
        --voice voice.npz

`--engine chatterbox --voice voice.npz` takes the add-on's npz conditionals.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine", default="qwen3-tts", help="registry name: qwen3-tts | chatterbox")
    ap.add_argument("--device", type=int, default=0, help="CUDA ordinal, PCI-bus order")
    ap.add_argument(
        "--models-dir", type=Path, default=None, help="where fetch-models put the files"
    )
    ap.add_argument("--voice", type=Path, required=True, help="voice file in the engine's format")
    ap.add_argument("--text", default="Sure, I can help with that.")
    ap.add_argument("--out", type=Path, default=Path("smoke.wav"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--repeat", type=int, default=3, help="utterances after warm-up (TTFA p50 over them)"
    )
    ap.add_argument(
        "--first-chunk-tokens", type=int, default=None, help="protocol knob (40 ms tokens)"
    )
    ap.add_argument("--qwen3-model", default="1.7b", choices=["1.7b", "0.6b"])
    ap.add_argument("--qwen3-chunk-size", type=int, default=8)
    ap.add_argument("--qwen3-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--qwen3-language", default="English")
    ap.add_argument("--no-verify", action="store_true", help="skip the models.lock size/sha check")
    ap.add_argument(
        "--build-voice", type=Path, default=None, help="16-bit WAV reference clip -> --voice"
    )
    ap.add_argument(
        "--transcript", default="", help="what the clip says (qwen3-tts; empty = x-vector only)"
    )
    args = ap.parse_args(argv)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from speechforge import models
    from speechforge.audio import TTS_SAMPLE_RATE, read_wav, wav_bytes
    from speechforge.config import Config
    from speechforge.engines import base

    config = Config(
        data_dir=Path(os.environ.get("SPEECHFORGE_DATA_DIR", "data")),
        models={"dir": args.models_dir} if args.models_dir else {},
        gpu={"devices": [args.device]},
        voices={"allow_reference_clip": True},  # the smoke test IS the owner's local run
        tts={
            "engine": args.engine,
            "qwen3": {
                "model": args.qwen3_model,
                "chunk_size": args.qwen3_chunk_size,
                "dtype": args.qwen3_dtype,
                "language": args.qwen3_language,
            },
        },
    )
    base.builtin()
    try:
        loader = base.TTS_ENGINES[args.engine]
    except KeyError:
        print(f"unknown engine {args.engine!r}; known: {sorted(base.TTS_ENGINES)}", file=sys.stderr)
        return 2

    if args.build_voice is not None:
        if args.engine != "qwen3-tts":
            print(
                "--build-voice is for qwen3-tts; Chatterbox voices come from the add-on",
                file=sys.stderr,
            )
            return 2
        from speechforge.engines.qwen3_tts import build_voice_bytes, parse_voice

        pcm, rate = read_wav(args.build_voice.read_bytes())
        data = build_voice_bytes(pcm, rate, args.transcript)
        voice = parse_voice(data)  # validates length, rate, transcript
        args.voice.write_bytes(data)
        print(
            f"voice written: {args.voice} ({len(data)} bytes, {voice.seconds:.1f} s @ {rate} Hz, "
            f"{'x-vector only' if voice.xvec_only else 'in-context'}; "
            f"id sha256:{hashlib.sha256(data).hexdigest()})"
        )

    if not args.no_verify:
        group = models.tts_group_for(config)
        missing = models.missing(config, [group])
        if missing:
            print(
                f"missing or wrong-size model files for group {group}: {missing[:4]}",
                file=sys.stderr,
            )
            print(
                "run: speechforge fetch-models (with tts.engine set in config.yaml)",
                file=sys.stderr,
            )
            return 3
        t = time.perf_counter()
        models.verify(config, [group])
        print(f"models.lock verify ({group}): ok in {time.perf_counter() - t:.1f} s")

    print(f"loading {args.engine} on CUDA {args.device} ...")
    t0 = time.perf_counter()
    tts = loader(config, args.device)
    print(f"loaded in {time.perf_counter() - t0:.1f} s: {tts.describe()}")
    voice = tts.parse_voice(args.voice.read_bytes())
    t1 = time.perf_counter()
    tts.warm_up()
    print(f"warm-up {time.perf_counter() - t1:.1f} s; stats {tts.last_stats}")

    opts = {"first_chunk_tokens": args.first_chunk_tokens} if args.first_chunk_tokens else None
    ttfas: list[float] = []
    pieces = []
    for i in range(max(1, args.repeat)):
        t2 = time.perf_counter()
        ttfa = None
        pieces = []
        for piece in tts.stream(args.text, "smoke", voice, seed=args.seed + i, options=opts):
            if ttfa is None:
                ttfa = (time.perf_counter() - t2) * 1000
            pieces.append(piece)
        total_ms = (time.perf_counter() - t2) * 1000
        audio_ms = sum(len(p) for p in pieces) / TTS_SAMPLE_RATE * 1000
        ttfas.append(ttfa or float("nan"))
        st = tts.last_stats
        print(
            f"[{i + 1}] ttfa {ttfa:.0f} ms, audio {audio_ms:.0f} ms, wall {total_ms:.0f} ms, "
            f"rtf {total_ms / audio_ms if audio_ms else float('nan'):.3f}, "
            f"tokens {st.get('tokens')}, chunks {st.get('chunks')}, "
            f"prefill {st.get('prefill_ms')} ms"
        )
    ttfas.sort()
    print(f"ttfa p50 {ttfas[len(ttfas) // 2]:.0f} ms over {len(ttfas)} utterance(s)")
    if pieces:
        import numpy as np

        args.out.write_bytes(wav_bytes(np.concatenate(pieces), TTS_SAMPLE_RATE))
        print(f"wrote {args.out}")
    tts.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
