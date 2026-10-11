#!/usr/bin/env python3
"""Measure a cold load the way residency does it, with or without the shared
GPU arena (DECISIONS S24), and print what changed on the card.

No server, no lease, no StudioForge: run it by hand on the GPU host, on a card
you hold a lease for. It builds the same ``Speaker`` ``residency`` builds
(``load_speaker`` then ``warm_up``), so its load and warm-up times are what
``/ready`` would report as ``load_ms`` minus the self-test.

    uv run --extra gpu python scripts/gpu_cold_load.py --device 3 --models-dir <data>/models \\
        --dtype fp16 --shared
    uv run --extra gpu python scripts/gpu_cold_load.py --device 3 --models-dir <data>/models \\
        --dtype fp16 --no-shared

Each line of output is one fact; the last line is one JSON object with all of
them, for the benchmarks table. ``--runs 2`` loads, unloads and loads again in
one process: the second load exercises the idempotent registration and the
``used_after_close_mib`` of the first shows whether the shared arena gives the
card back on unload (it cannot be unregistered from Python).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _used_mib(device: int) -> int | None:
    from speechforge.residency import nvml_live

    return nvml_live(device).get("vram_used_mib")


def _probe_ttfa(speaker: Any, text: str) -> dict[str, Any]:
    """One short line in a synthetic voice: first-audio and total time."""
    from speechforge.audio import TTS_SAMPLE_RATE, synthetic_reference

    voice = speaker.parse_voice(speaker.build_voice(synthetic_reference(), TTS_SAMPLE_RATE))
    t0 = time.perf_counter()
    ttfa = None
    samples = 0
    for piece in speaker.speak(text, "_probe", voice, seed=1):
        if ttfa is None:
            ttfa = (time.perf_counter() - t0) * 1000
        samples += len(piece)
    wall = (time.perf_counter() - t0) * 1000
    audio = samples / TTS_SAMPLE_RATE * 1000
    return {
        "ttfa_ms": round(ttfa or float("nan")),
        "audio_ms": round(audio),
        "rtf": round(wall / audio, 3) if audio else None,
    }


def one_run(config: Any, device: int, *, probe: bool, index: int) -> dict[str, Any]:
    from speechforge.engines import arena
    from speechforge.pipeline import load_speaker

    out: dict[str, Any] = {"run": index, "used_before_mib": _used_mib(device)}
    t0 = time.perf_counter()
    speaker = load_speaker(config, device)
    out["load_ms"] = round((time.perf_counter() - t0) * 1000)
    out["used_after_load_mib"] = _used_mib(device)
    print(f"[{index}] sessions created in {out['load_ms']} ms; {speaker.describe}")
    t1 = time.perf_counter()
    warm = speaker.warm_up()
    out["warmup_ms"] = round((time.perf_counter() - t1) * 1000)
    out["warmup"] = warm
    out["total_ms"] = out["load_ms"] + out["warmup_ms"]
    out["used_after_warmup_mib"] = _used_mib(device)
    out["shared_arena"] = bool(speaker.shared_arena)
    out["arena"] = arena.facts(device)
    print(
        f"[{index}] warm-up {out['warmup_ms']} ms ({warm}); load + warm-up {out['total_ms']} ms; "
        f"shared_arena={out['shared_arena']} {out['arena'] or ''}"
    )
    if probe:
        out["probe"] = _probe_ttfa(speaker, "Sure, I can help with that this afternoon.")
        print(f"[{index}] probe {out['probe']}")
    speaker.close()
    del speaker
    gc.collect()
    out["used_after_close_mib"] = _used_mib(device)
    before, after = out["used_before_mib"], out["used_after_close_mib"]
    if before is not None and after is not None:
        out["held_after_close_mib"] = after - before
    print(
        f"[{index}] VRAM used (MiB): before {before}, after load {out['used_after_load_mib']}, "
        f"after warm-up {out['used_after_warmup_mib']}, after close {after}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", type=int, required=True, help="CUDA ordinal, PCI-bus order")
    ap.add_argument(
        "--models-dir", type=Path, required=True, help="where fetch-models put the files"
    )
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "q4", "q4f16", "q8"])
    shared = ap.add_mutually_exclusive_group()
    shared.add_argument("--shared", dest="shared", action="store_true", default=True)
    shared.add_argument("--no-shared", dest="shared", action="store_false")
    ap.add_argument(
        "--arena-initial-mb", type=int, default=None, help="first chunk (default: the config's)"
    )
    ap.add_argument("--mem-limit-mb", type=int, default=None, help="gpu.mem_limit_mb")
    ap.add_argument("--no-eou", action="store_true", help="skip Smart Turn")
    ap.add_argument("--runs", type=int, default=1, help="load/unload cycles in this process")
    ap.add_argument("--probe", action="store_true", help="speak one line after warm-up (TTFA)")
    ap.add_argument("--json", type=Path, default=None, help="also write the facts here")
    args = ap.parse_args(argv)

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from speechforge.config import Config

    gpu: dict[str, Any] = {"devices": [args.device], "shared_arena": args.shared}
    if args.arena_initial_mb is not None:
        gpu["arena_initial_mb"] = args.arena_initial_mb
    if args.mem_limit_mb is not None:
        gpu["mem_limit_mb"] = args.mem_limit_mb
    config = Config(
        data_dir=Path(os.environ.get("SPEECHFORGE_DATA_DIR", "data")),
        models={"dir": args.models_dir, "tts_dtype": args.dtype},
        gpu=gpu,
        eou={"enabled": not args.no_eou},
        selftest={"enabled": False},
    )
    print(
        f"cold load on CUDA {args.device}: tts_dtype {args.dtype}, shared_arena {args.shared}, "
        f"arena_initial_mb {config.gpu.arena_initial_mb}, mem_limit_mb {config.gpu.mem_limit_mb}"
    )
    runs = [one_run(config, args.device, probe=args.probe, index=i + 1) for i in range(args.runs)]
    summary = {
        "device": args.device,
        "tts_dtype": args.dtype,
        "shared_arena_requested": args.shared,
        "arena_initial_mb": config.gpu.arena_initial_mb,
        "mem_limit_mb": config.gpu.mem_limit_mb,
        "runs": runs,
    }
    line = json.dumps(summary, sort_keys=True)
    print(line)
    if args.json is not None:
        args.json.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
