"""The STT lane: one worker thread, priority order (SPEC-speechforge §4.4).

Work order on the speech-recognition side of the card: end-of-turn scores
first (a person is waiting on the decision), then finals, then partials. The
TTS lane is per context (``tts_ctx.py``); the Chatterbox engine interleaves
two contexts per decode step. One process, so stream priority and the single
CUDA context are shared (why both lanes live in one process).

Jobs are plain callables run on the worker; the caller awaits an
``asyncio.Future``. Callers coalesce their own partials (one pending per
session), so the queue stays bounded by sessions x 3.
"""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
from collections.abc import Callable
from typing import Any

PRIO_EOU = 0
PRIO_FINAL = 1
PRIO_PARTIAL = 2
_STOP = object()


class SttLane:
    def __init__(self, name: str = "sf-stt-lane") -> None:
        self._q: queue.PriorityQueue[tuple[int, int, Any]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, prio: int, fn: Callable[[], Any]) -> asyncio.Future[Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._q.put((prio, next(self._seq), (fn, fut, loop)))
        return fut

    def depth(self) -> int:
        return self._q.qsize()

    def close(self) -> None:
        self._q.put((-1, next(self._seq), _STOP))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            _prio, _n, item = self._q.get()
            if item is _STOP:
                return
            fn, fut, loop = item
            if fut.cancelled():
                continue
            try:
                result = fn()
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting coroutine
                loop.call_soon_threadsafe(_set_exc, fut, exc)
            else:
                loop.call_soon_threadsafe(_set_result, fut, result)


def _set_result(fut: asyncio.Future[Any], value: Any) -> None:
    if not fut.done():
        fut.set_result(value)


def _set_exc(fut: asyncio.Future[Any], exc: BaseException) -> None:
    if not fut.done():
        fut.set_exception(exc)
