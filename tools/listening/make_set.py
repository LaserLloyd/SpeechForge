"""Step 2: build the blind set from 2-4 engine render directories.

    uv run python -m tools.listening.make_set --engines <renders>/chatterbox <renders>/qwen3-tts \\
        --set <blind-set> --key <key-dir> --seed 7

For every line the engines are shuffled under the labels A/B(/C/D) -- a fresh
shuffle per line, so "A" is not one engine all the way through -- loudness-matched
(RMS to a common level with a peak guard; a pure gain, nothing else touches the
audio) and copied to ``<set>/<NN>/<label>.wav``.

``<set>/manifest.json`` lists the lines, labels and durations and names NO
engine; it is what ``serve`` sends to the rater's browser. ``<key>/key.json``
maps label -> engine per line and must live outside ``<set>``, so nothing the
rating page can serve reveals it. ``score`` joins the two afterwards.

Deterministic for a given ``--seed`` and engine order, so a set can be rebuilt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tools.listening import kit


def engine_name(spec: str) -> tuple[str, Path]:
    """``name=DIR`` or ``DIR`` (the directory's basename is the engine name)."""
    if "=" in spec and not Path(spec).exists():
        name, _, d = spec.partition("=")
        return name.strip(), Path(d)
    p = Path(spec)
    return p.name, p


def build_set(
    engines: list[tuple[str, Path]],
    set_dir: Path,
    key_dir: Path,
    *,
    seed: int,
    lines: list[str] | None = None,
    target_dbfs: float = kit.TARGET_RMS_DBFS,
    peak_dbfs: float = kit.PEAK_DBFS,
) -> tuple[dict, dict]:
    names = [n for n, _ in engines]
    if len(set(names)) != len(names):
        raise ValueError(f"engine names must differ: {names}")
    kit.labels_for(len(engines))
    set_dir, key_dir = set_dir.resolve(), key_dir.resolve()
    if set_dir == key_dir or set_dir in key_dir.parents:
        raise ValueError("the key directory must not be inside the set directory")
    if lines is None:
        lines = kit.read_lines(None)
    # Only the lines every engine rendered make it into the set.
    present = [
        n
        for n in range(1, len(lines) + 1)
        if all((d / f"{kit.line_name(n)}.wav").exists() for _, d in engines)
    ]
    if not present:
        raise ValueError("no line was rendered by every engine")
    rng = np.random.default_rng(seed)
    manifest_lines, key_lines, levels = [], {}, {}
    for n in present:
        stem = kit.line_name(n)
        assignment = kit.shuffled_labels(rng, names)  # label -> engine
        by_engine = dict(engines)
        samples = {}
        for label, engine in sorted(assignment.items()):
            audio, rate = kit.read_wav(by_engine[engine] / f"{stem}.wav")
            out, stats = kit.normalise_rms(audio, target_dbfs, peak_dbfs)
            rel = f"{stem}/{label}.wav"
            kit.write_wav(set_dir / rel, out, rate)
            samples[label] = {"file": rel, "duration_s": kit.duration_s(out, rate)}
            levels.setdefault(stem, {})[label] = stats
        manifest_lines.append({"n": n, "id": stem, "text": lines[n - 1], "samples": samples})
        key_lines[stem] = assignment
    manifest = {
        "version": 1,
        "labels": kit.labels_for(len(engines)),
        "level_dbfs": target_dbfs,
        "lines": manifest_lines,
    }
    key = {
        "version": 1,
        "seed": seed,
        "engines": names,
        "lines": key_lines,
        "levels": levels,
    }
    kit.json_dump(set_dir / "manifest.json", manifest)
    (set_dir / "ratings").mkdir(exist_ok=True)
    kit.json_dump(key_dir / "key.json", key)
    return manifest, key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--engines",
        nargs="+",
        required=True,
        metavar="DIR",
        help="2-4 render directories from render_set (or name=DIR)",
    )
    ap.add_argument(
        "--set", required=True, type=Path, help="blind set directory (served to raters)"
    )
    ap.add_argument("--key", required=True, type=Path, help="key directory, outside --set")
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed (same seed = same set)")
    ap.add_argument("--lines", type=Path, default=None)
    ap.add_argument("--level-dbfs", type=float, default=kit.TARGET_RMS_DBFS, help="target RMS")
    ap.add_argument("--peak-dbfs", type=float, default=kit.PEAK_DBFS, help="peak guard")
    args = ap.parse_args(argv)
    engines = [engine_name(s) for s in args.engines]
    for name, d in engines:
        if not d.is_dir():
            raise SystemExit(f"{name}: {d} is not a directory")
    manifest, key = build_set(
        engines,
        args.set,
        args.key,
        seed=args.seed,
        lines=kit.read_lines(args.lines),
        target_dbfs=args.level_dbfs,
        peak_dbfs=args.peak_dbfs,
    )
    print(
        f"blind set: {len(manifest['lines'])} lines x {len(key['engines'])} engines "
        f"-> {args.set}  (key: {args.key / 'key.json'}; keep it away from the raters)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
