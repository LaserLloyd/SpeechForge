"""``GET /v1/realtime`` -- the sf.1 WebSocket (SPEC-speechforge §4.3, §8).

Before the upgrade (each refusal is an HTTP response, never a half-open socket):

* an ``Origin`` header -> 403 (no browser may drive the socket);
* more than ``upgrades_per_ip_per_min`` attempts from one address -> 429;
* missing or wrong ``Authorization: Bearer`` -> 401 (the token is never read
  from the query string);
* ``Sec-WebSocket-Protocol`` without ``sf.1`` -> 400;
* no free session slot -> 503 + Retry-After; the models cooling down -> 503 +
  Retry-After (the box stays on its CPU and retries later).

After the upgrade the first frame must be ``hello`` (within 5 s). The session
then waits for the models (``realtime.ready_wait_s``; the box runs on its CPU
meanwhile), sends ``ready``, and serves until ``bye``, a close, 30 s without any
frame, or an unload. A dropped socket keeps its slot for ``resume_window_s``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict, deque
from typing import Any

from starlette.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from speechforge.auth import TokenStore, bearer
from speechforge.config import is_loopback_host
from speechforge.errors import SpeechError
from speechforge.metrics import METRICS
from speechforge.protocol import PROTO, FrameError, check_control, dumps
from speechforge.session import Session, SessionManager

log = logging.getLogger("speechforge.realtime")
HELLO_TIMEOUT_S = 5.0


class UpgradeLimiter:
    def __init__(self, per_min: int) -> None:
        self.per_min = per_min
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        q = self._hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_min:
            return False
        q.append(now)
        return True


def _deny(status: int, code: str, message: str, retry_after: float | None = None) -> JSONResponse:
    err = SpeechError(message, code=code, status_code=status, retry_after_s=retry_after)
    return JSONResponse(err.to_payload(), status_code=status, headers=err.headers())


async def realtime_endpoint(
    ws: WebSocket, manager: SessionManager, tokens: TokenStore, limiter: UpgradeLimiter
) -> None:
    peer = ws.client.host if ws.client else ""
    deny: JSONResponse | None = None
    protos = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",")]
    if ws.headers.get("origin") is not None:
        deny = _deny(403, "forbidden", "browsers may not connect (Origin header present)")
    elif not limiter.allow(peer):
        deny = _deny(429, "rate_limited", "too many upgrade attempts", 60)
    elif tokens.configured and not tokens.check(bearer(ws.headers.get("authorization"))):
        deny = _deny(401, "invalid_token", "missing or wrong bearer token")
    elif not tokens.configured and not is_loopback_host(peer):
        deny = _deny(401, "invalid_token", "no token is configured: loopback only")
    elif PROTO not in protos:
        deny = _deny(400, "bad_frame", f"Sec-WebSocket-Protocol must offer {PROTO}")
    elif manager.capacity_left() <= 0:
        deny = _deny(503, "busy", "all session slots are in use", 5)
    else:
        refusal = manager.residency.refusal()
        if refusal is not None:
            deny = _deny(503, refusal.code, refusal.message, refusal.retry_after_s)
    if deny is not None:
        METRICS.inc("session_errors_total", code="upgrade_refused")
        await ws.send_denial_response(deny)
        return
    await ws.accept(subprotocol=PROTO)
    session: Session | None = None
    try:
        try:
            first = await asyncio.wait_for(ws.receive_text(), HELLO_TIMEOUT_S)
            hello = check_control(first)
            if hello["type"] != "hello" or hello.get("proto") != PROTO:
                raise FrameError("the first frame must be hello{proto:'sf.1'}")
        except (TimeoutError, FrameError, KeyError) as exc:
            await ws.send_text(
                dumps(
                    {
                        "type": "error",
                        "code": "bad_frame",
                        "fatal": True,
                        "message": str(exc)[:200] or "no hello",
                    }
                )
            )
            await ws.close(1002)
            return
        resume = hello.get("resume")
        if isinstance(resume, dict) and resume.get("session"):
            session = manager.resume(str(resume.get("session")), str(resume.get("token", "")))
            if session is None:
                await ws.send_text(
                    dumps(
                        {
                            "type": "error",
                            "code": "bad_frame",
                            "fatal": False,
                            "message": "resume refused; starting a new session",
                        }
                    )
                )
        if session is None:
            if manager.capacity_left() <= 0:
                await ws.send_text(
                    dumps({"type": "error", "code": "busy", "fatal": True, "retry_after_s": 5})
                )
                await ws.close(1013)
                return
            session = manager.open()

        async def send_text(t: str) -> None:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.send_text(t)

        async def send_bytes(b: bytes) -> None:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.send_bytes(b)

        async def close(code: int, reason: str) -> None:
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(code, reason[:120])

        session.attach(send_text, send_bytes, close)
        r = manager.residency
        if r.state != "ready":
            await session.send(
                {"type": "state", "lease": manager.lease_state(), "models": "loading"}
            )
        if not await r.wait_ready(manager.config.realtime.ready_wait_s):
            refusal = r.refusal()
            await session.error(
                "not_ready",
                refusal.message if refusal else "models not ready",
                fatal=True,
                retry_after_s=refusal.retry_after_s if refusal else 5,
            )
            manager.forget(session)
            await ws.close(1013)
            return
        if session.stt is None:
            session.start_stt()
        session.ready = True
        await session.send(session.ready_frame())
        await _serve(ws, session, manager)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("session failed")
        if session is not None:
            await session.error("internal", "server error", fatal=True)
            manager.forget(session)
    finally:
        if session is not None and not session.closed:
            session.detach()
        with contextlib.suppress(Exception):
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close(1000)


async def _serve(ws: WebSocket, session: Session, manager: SessionManager) -> None:
    idle = manager.config.realtime.idle_close_s
    while not session.closed:
        try:
            msg: dict[str, Any] = await asyncio.wait_for(ws.receive(), idle)
        except TimeoutError:
            await session.error("bad_frame", f"no frame for {idle:.0f} s", fatal=True)
            manager.forget(session)
            await session.shutdown(1001, "idle")
            return
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        if msg.get("bytes") is not None:
            await session.on_bytes(msg["bytes"])
        elif msg.get("text") is not None:
            await session.on_text(msg["text"])
