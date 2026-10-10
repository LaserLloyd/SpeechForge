"""A small sf.1 client (the benchmark harness, the canary self-test, and the
reference the DisPatch add-on's ``RigClient`` follows).

Uses the ``websockets`` package (installed with ``uvicorn[standard]``).
Everything is async; frames are dispatched to per-type queues so a caller can
``await c.next("stt.final")`` while audio streams in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import defaultdict
from typing import Any

import numpy as np

from speechforge.protocol import KIND_TTS, PROTO, dumps, unpack_audio
from speechforge.session import mic_frames


class RigClient:
    def __init__(self, url: str, token: str = "", *, ping_s: float = 0.25) -> None:
        self.url = url
        self.token = token
        self.ping_s = ping_s
        self.ws: Any = None
        self.ready: dict[str, Any] = {}
        self.queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)
        self.audio: dict[int, list[tuple[float, Any]]] = defaultdict(list)  # ctx -> (t, frame)
        self.rtts: list[float] = []
        self._reader: asyncio.Task[None] | None = None
        self._pinger: asyncio.Task[None] | None = None
        self.mic_pts = 0

    async def connect(
        self, *, resume: dict[str, Any] | None = None, ready_timeout: float = 60.0
    ) -> dict[str, Any]:
        import websockets

        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.ws = await websockets.connect(
            self.url,
            subprotocols=[PROTO],
            additional_headers=headers,
            max_size=2**16,
            compression=None,
            open_timeout=5,
        )
        hello: dict[str, Any] = {
            "type": "hello",
            "proto": PROTO,
            "client": {"name": "speechforge-client", "version": "1"},
            "caps": {"opus_out": False},
        }
        if resume:
            hello["resume"] = resume
        await self.ws.send(dumps(hello))
        self._reader = asyncio.create_task(self._read())
        self.ready = await self.next("ready", timeout=ready_timeout)
        if self.ping_s:
            self._pinger = asyncio.create_task(self._ping())
        return self.ready

    async def _read(self) -> None:
        try:
            async for msg in self.ws:
                now = time.perf_counter()
                if isinstance(msg, bytes):
                    f = unpack_audio(msg)
                    if f.kind == KIND_TTS:
                        self.audio[f.ctx].append((now, f))
                    continue
                data = json.loads(msg)
                data["_t"] = now
                if data.get("type") == "pong":
                    self.rtts.append((time.perf_counter() - float(data["t"])) * 1000)
                if data.get("type") == "error" and data.get("fatal"):
                    await self.queues["ready"].put(data)
                await self.queues[data.get("type", "?")].put(data)
        except Exception:  # noqa: BLE001 - the socket closed
            pass
        finally:
            await self.queues["_closed"].put({"type": "_closed"})

    async def _ping(self) -> None:
        while True:
            await asyncio.sleep(self.ping_s)
            with contextlib.suppress(Exception):
                await self.send({"type": "ping", "t": time.perf_counter()})

    async def next(self, kind: str, timeout: float = 10.0) -> dict[str, Any]:
        msg = await asyncio.wait_for(self.queues[kind].get(), timeout)
        if kind == "ready" and msg.get("type") == "error":
            raise RuntimeError(f"rig refused: {msg.get('code')} {msg.get('message', '')}")
        return msg

    def drain(self, kind: str) -> list[dict[str, Any]]:
        q = self.queues[kind]
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    async def send(self, msg: dict[str, Any]) -> None:
        await self.ws.send(dumps(msg))

    async def send_mic(self, pcm16: np.ndarray, *, realtime: bool = False) -> None:
        for frame in mic_frames(np.asarray(pcm16, "<i2"), self.mic_pts):
            await self.ws.send(frame)
            if realtime:
                await asyncio.sleep(0.02)
        self.mic_pts += len(pcm16)

    def ctx_pcm(self, ctx: int) -> np.ndarray:
        frames = [f.pcm for _, f in self.audio.get(ctx, [])]
        return np.concatenate(frames) if frames else np.zeros(0, "<i2")

    async def close(self) -> None:
        for task in (self._pinger, self._reader):
            if task is not None:
                task.cancel()
        with contextlib.suppress(Exception):
            await self.send({"type": "bye", "reason": "done"})
        with contextlib.suppress(Exception):
            await self.ws.close()
