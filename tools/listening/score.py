"""Step 4: score the ratings against the key and write ``report.md``.

    uv run python -m tools.listening.score --set <blind-set> --key <key-dir> \\
        [--render-json chatterbox=<renders>/chatterbox/render.json ...] [--out report.md]

Per engine: MOS mean with a 95 % bootstrap confidence interval, the "sounds most
like the owner" win rate with its interval, and (with ``--render-json``) the
time-to-first-audio p50/p90 and real-time factor from ``render_set``. Then a
per-line table, a per-rater table (how often each rater agreed with the others'
majority pick, pairwise agreement) and the raters' comments.

A rater who answered a line twice (a resumed session that repeated it) counts
once: the last answer stands. The report names engines; it is the owner's input
for decision D2 and, like everything else here, stays out of the repository.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.listening import kit


def load_ratings(set_dir: Path) -> list[dict]:
    """Every rater's last answer per line, from ``<set>/ratings/*.jsonl``."""
    last: dict[tuple[str, str], dict] = {}
    for p in sorted((set_dir / "ratings").glob("*.jsonl")):
        for raw in p.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                r = json.loads(raw)
                last[(str(r["rater"]), str(r["line"]))] = r
            except (ValueError, KeyError, TypeError):
                continue
    return list(last.values())


def join(ratings: list[dict], key: dict) -> list[dict]:
    """Attach engine names: ``{rater, line, mos: {engine: 1-5}, pick: engine, comment}``."""
    out = []
    for r in ratings:
        labels = key["lines"].get(str(r["line"]))
        if not labels:
            continue
        mos = {labels[lbl]: int(v) for lbl, v in r.get("mos", {}).items() if lbl in labels}
        pick = labels.get(r.get("most_like"))
        if len(mos) != len(labels) or pick is None:
            continue
        out.append(
            {
                "rater": r["rater"],
                "line": str(r["line"]),
                "mos": mos,
                "pick": pick,
                "comment": r.get("comment", ""),
            }
        )
    return out


def bootstrap_ci(
    values: list[float], *, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean."""
    a = np.asarray(values, np.float64)
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (float(lo), float(hi))


def majority(picks: list[str]) -> str | None:
    """The plurality pick, or None on a tie."""
    if not picks:
        return None
    c = Counter(picks).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


def aggregate(joined: list[dict], key: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    engines = list(key["engines"])
    per_engine: dict[str, dict] = {}
    picks_all = [j["pick"] for j in joined]
    for k, e in enumerate(engines):
        mos = [j["mos"][e] for j in joined]
        wins = [1.0 if p == e else 0.0 for p in picks_all]
        per_engine[e] = {
            "n": len(mos),
            "mos_mean": round(statistics.fmean(mos), 3) if mos else None,
            "mos_ci": [round(x, 3) for x in bootstrap_ci(mos, n_boot=n_boot, seed=seed + k)],
            "mos_hist": {str(v): mos.count(v) for v in range(1, 6)},
            "pref_wins": int(sum(wins)),
            "pref_total": len(wins),
            "pref_rate": round(statistics.fmean(wins), 3) if wins else None,
            "pref_ci": [
                round(x, 3) for x in bootstrap_ci(wins, n_boot=n_boot, seed=seed + 100 + k)
            ],
        }
    # per line
    by_line: dict[str, list[dict]] = defaultdict(list)
    for j in joined:
        by_line[j["line"]].append(j)
    lines = {}
    for line_id in sorted(by_line):
        rows = by_line[line_id]
        lines[line_id] = {
            "n": len(rows),
            "mos": {e: round(statistics.fmean(r["mos"][e] for r in rows), 2) for e in engines},
            "picks": {e: sum(r["pick"] == e for r in rows) for e in engines},
            "majority": majority([r["pick"] for r in rows]),
        }
    # per rater
    raters = sorted({j["rater"] for j in joined})
    per_rater = {}
    for nick in raters:
        mine = [j for j in joined if j["rater"] == nick]
        agree = total = 0
        for j in mine:
            others = [r["pick"] for r in by_line[j["line"]] if r["rater"] != nick]
            # Leave-one-out when there are others; a lone rater has nobody to agree with.
            m = majority(others)
            if m is None:
                continue
            total += 1
            agree += j["pick"] == m
        per_rater[nick] = {
            "lines": len(mine),
            "mos_mean": {e: round(statistics.fmean(j["mos"][e] for j in mine), 2) for e in engines},
            "picks": {e: sum(j["pick"] == e for j in mine) for e in engines},
            "agreement": round(agree / total, 3) if total else None,
            "agreement_n": total,
        }
    pairwise = {}
    for a, b in itertools.combinations(raters, 2):
        pa = {j["line"]: j["pick"] for j in joined if j["rater"] == a}
        pb = {j["line"]: j["pick"] for j in joined if j["rater"] == b}
        shared = sorted(set(pa) & set(pb))
        if shared:
            pairwise[f"{a} / {b}"] = {
                "n": len(shared),
                "agreement": round(sum(pa[x] == pb[x] for x in shared) / len(shared), 3),
            }
    comments = [
        {"rater": j["rater"], "line": j["line"], "comment": j["comment"]}
        for j in joined
        if j.get("comment")
    ]
    ranked = sorted(
        engines,
        key=lambda e: (per_engine[e]["pref_rate"] or 0, per_engine[e]["mos_mean"] or 0),
        reverse=True,
    )
    return {
        "engines": per_engine,
        "ranked": ranked,
        "lines": lines,
        "raters": per_rater,
        "pairwise": pairwise,
        "comments": comments,
        "n_ratings": len(joined),
        "n_raters": len(raters),
        "n_lines": len(lines),
    }


def load_render(spec: str) -> tuple[str, dict]:
    name, _, path = spec.partition("=")
    if not path:
        path, name = name, Path(name).resolve().parent.name
    data = kit.json_load(path)
    return (name or data.get("engine", "?")), data


def _ci(ci: list[float], pct: bool = False) -> str:
    if not ci or any(x != x for x in ci):
        return "-"
    if pct:
        return f"{ci[0]:.0%} - {ci[1]:.0%}"
    return f"{ci[0]:.2f} - {ci[1]:.2f}"


def render_report(
    agg: dict, key: dict, manifest: dict | None, renders: dict[str, dict] | None = None
) -> str:
    engines = key["engines"]
    renders = renders or {}
    texts = {ln["id"]: ln["text"] for ln in (manifest or {}).get("lines", [])}
    out = ["# Listening test report", ""]
    out.append(
        f"{agg['n_ratings']} answers from {agg['n_raters']} rater(s) over {agg['n_lines']} "
        f"line(s); {len(engines)} engines, blind, loudness-matched (seed {key.get('seed')})."
    )
    out.append("")
    out.append("## Engines")
    out.append("")
    head = "| engine | MOS mean | 95 % CI | n | most like the owner | 95 % CI |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    if renders:
        head += " TTFA p50 / p90 | RTF |"
        sep += " --- | --- |"
    out += [head, sep]
    for e in agg["ranked"]:
        pe = agg["engines"][e]
        if pe["pref_rate"] is None:
            row = f"| {e} | - | - | 0 | 0/0 | - |"
        else:
            row = (
                f"| {e} | {pe['mos_mean']} | {_ci(pe['mos_ci'])} | {pe['n']} | "
                f"{pe['pref_wins']}/{pe['pref_total']} ({pe['pref_rate']:.0%}) | "
                f"{_ci(pe['pref_ci'], pct=True)} |"
            )
        if renders:
            s = renders.get(e, {}).get("summary", {})
            p50, p90, rtf = s.get("ttfa_p50_ms"), s.get("ttfa_p90_ms"), s.get("rtf_mean")
            fmt = lambda v, d: "-" if v is None else f"{float(v):.{d}f}"  # noqa: E731
            row += f" {fmt(p50, 0)} / {fmt(p90, 0)} ms | {fmt(rtf, 2)} |"
        out.append(row)
    out.append("")
    if len(agg["ranked"]) >= 2 and agg["n_ratings"]:
        a, b = agg["ranked"][0], agg["ranked"][1]
        ca, cb = agg["engines"][a]["pref_ci"], agg["engines"][b]["pref_ci"]
        overlap = ca[0] <= cb[1] and cb[0] <= ca[1]
        out.append(
            f"Most often picked as the owner's voice: **{a}** "
            f"({agg['engines'][a]['pref_rate']:.0%}). "
            + (
                f"Its interval overlaps {b}'s, so with this many answers the preference is "
                "not settled; more raters or lines would narrow it."
                if overlap
                else f"Its interval does not overlap {b}'s."
            )
        )
        out.append("")
    out.append(
        "MOS: 1 = robotic, 5 = indistinguishable from a person. Intervals are percentile "
        "bootstrap over answers (2000 resamples)."
    )
    out.append("")
    out.append("## Per line")
    out.append("")
    out.append(
        "| line | text | " + " | ".join(f"MOS {e}" for e in engines) + " | picks | majority |"
    )
    out.append("| --- | --- | " + " | ".join("---" for _ in engines) + " | --- | --- |")
    for line_id, row in agg["lines"].items():
        text = texts.get(line_id, "")
        text = (text[:57] + "...") if len(text) > 60 else text
        picks = ", ".join(f"{e} {n}" for e, n in row["picks"].items() if n)
        out.append(
            f"| {line_id} | {text} | "
            + " | ".join(str(row["mos"][e]) for e in engines)
            + f" | {picks} | {row['majority'] or 'tie'} |"
        )
    out.append("")
    out.append("## Raters")
    out.append("")
    out.append(
        "| rater | lines | "
        + " | ".join(f"MOS {e}" for e in engines)
        + " | picks | agrees with the others |"
    )
    out.append("| --- | --- | " + " | ".join("---" for _ in engines) + " | --- | --- |")
    for nick, r in agg["raters"].items():
        picks = ", ".join(f"{e} {n}" for e, n in r["picks"].items() if n)
        agree = f"{r['agreement']:.0%} of {r['agreement_n']}" if r["agreement"] is not None else "-"
        out.append(
            f"| {nick} | {r['lines']} | "
            + " | ".join(str(r["mos_mean"][e]) for e in engines)
            + f" | {picks} | {agree} |"
        )
    out.append("")
    if agg["pairwise"]:
        out.append('Pairwise agreement on the "most like" pick (shared lines):')
        out.append("")
        for pair, v in agg["pairwise"].items():
            out.append(f"- {pair}: {v['agreement']:.0%} of {v['n']}")
        out.append("")
    if agg["comments"]:
        out.append("## Comments")
        out.append("")
        for c in agg["comments"]:
            text = c["comment"].replace("\n", " ").replace("|", "/")
            out.append(f"- line {c['line']}, {c['rater']}: {text}")
        out.append("")
    out.append("## Reading it")
    out.append("")
    out.append(
        "- The preference question is the one that decides D2 (SPEC §14): the engine the family "
        "hears as the owner. MOS says how natural each one is on its own."
    )
    out.append(
        "- Overlapping intervals mean the test did not separate the engines; that is a result too."
    )
    out.append(
        "- TTFA/RTF come from `render_set` on the rig, not from the raters, and are one engine "
        "at a time with a warm server."
    )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", required=True, type=Path)
    ap.add_argument("--key", required=True, type=Path, help="key directory (or key.json)")
    ap.add_argument("--out", type=Path, default=None, help="report path (default <set>/report.md)")
    ap.add_argument(
        "--render-json",
        action="append",
        default=[],
        metavar="ENGINE=PATH",
        help="merge ttfa/wall from a render.json (repeatable)",
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the aggregate as JSON")
    args = ap.parse_args(argv)
    key_path = args.key if args.key.is_file() else args.key / "key.json"
    key = kit.json_load(key_path)
    manifest = (
        kit.json_load(args.set / "manifest.json") if (args.set / "manifest.json").exists() else None
    )
    joined = join(load_ratings(args.set), key)
    if not joined:
        print("no complete ratings found", file=sys.stderr)
    agg = aggregate(joined, key, n_boot=args.bootstrap, seed=args.seed)
    renders = dict(load_render(s) for s in args.render_json)
    report = render_report(agg, key, manifest, renders)
    out = args.out or (args.set / "report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    if args.json:
        kit.json_dump(args.json, agg)
    print(report)
    print(f"\nwritten: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
