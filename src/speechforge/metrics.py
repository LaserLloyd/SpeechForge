"""Prometheus text metrics, no content ever (SPEC-speechforge §9).

Small on purpose: counters, gauges and fixed-bucket histograms in one
registry, rendered as the text exposition format. Labels are bounded
(error codes, lease events). Nothing here may carry a transcript, a reply,
a voice id or a token.
"""

from __future__ import annotations

import bisect
import threading
from collections.abc import Callable

MS_BUCKETS = (5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 2000, 5000)
RATIO_BUCKETS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0, 2.0)
P_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


class _Hist:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.n = 0
        self.recent: list[float] = []

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, v)] += 1
        self.sum += v
        self.n += 1
        self.recent.append(v)
        if len(self.recent) > 512:
            del self.recent[:256]

    def quantile(self, q: float) -> float | None:
        if not self.recent:
            return None
        xs = sorted(self.recent)
        return xs[min(len(xs) - 1, int(q * len(xs)))]


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._gauges: dict[str, float] = {}
        self._gauge_fns: dict[str, Callable[[], float | None]] = {}
        self._hists: dict[str, _Hist] = {}
        self._help: dict[str, str] = {}

    def inc(self, name: str, n: float = 1, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + n

    def set(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def gauge_fn(self, name: str, fn: Callable[[], float | None]) -> None:
        self._gauge_fns[name] = fn

    def observe(self, name: str, value: float, buckets: tuple[float, ...] = MS_BUCKETS) -> None:
        with self._lock:
            h = self._hists.get(name)
            if h is None:
                h = self._hists[name] = _Hist(buckets)
            h.observe(float(value))

    def quantile(self, name: str, q: float) -> float | None:
        with self._lock:
            h = self._hists.get(name)
            return None if h is None else h.quantile(q)

    def counter(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters.get((name, tuple(sorted(labels.items()))), 0)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            hists = {k: (h.buckets, list(h.counts), h.sum, h.n) for k, h in self._hists.items()}
        for name, fn in self._gauge_fns.items():
            try:
                v = fn()
            except Exception:  # noqa: BLE001 - a gauge never breaks the scrape
                v = None
            if v is not None:
                gauges[name] = float(v)
        seen: set[str] = set()
        for (name, labels), v in sorted(counters.items()):
            full = f"speechforge_{name}"
            if full not in seen:
                lines.append(f"# TYPE {full} counter")
                seen.add(full)
            lab = ",".join(f'{k}="{_esc(val)}"' for k, val in labels)
            lines.append(f"{full}{{{lab}}} {v:g}" if lab else f"{full} {v:g}")
        for name, v in sorted(gauges.items()):
            full = f"speechforge_{name}"
            lines.append(f"# TYPE {full} gauge")
            lines.append(f"{full} {v:g}")
        for name, (buckets, counts, total, n) in sorted(hists.items()):
            full = f"speechforge_{name}"
            lines.append(f"# TYPE {full} histogram")
            acc = 0
            for b, c in zip(buckets, counts, strict=False):
                acc += c
                lines.append(f'{full}_bucket{{le="{b:g}"}} {acc}')
            lines.append(f'{full}_bucket{{le="+Inf"}} {n}')
            lines.append(f"{full}_sum {total:g}")
            lines.append(f"{full}_count {n}")
        return "\n".join(lines) + "\n"


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")[:64]


METRICS = Registry()
