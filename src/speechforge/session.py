"""The ``sf.1`` session: one per voice chat (SPEC-speechforge §4.3-§4.4, §8.3).

Transport-agnostic: :class:`Session` gets a ``send_text`` / ``send_bytes`` /
``close`` triple and is fed with ``on_text`` / ``on_bytes``; ``realtime.py`` binds
it to a Starlette WebSocket. :class:`SessionManager` holds the global cap,
the resume slots, and the residency hooks (lease ``state`` broadcasts and the
vacate drain).

Ownership rule: the box decides, the rig computes. The session emits signals
(``stt.vad``, ``stt.partial``, ``eou``, ``stt.final``, ``tts.*``) and audio; it
never ends a turn, never calls an LLM, never stores text.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from speechforge import __version__
from speechforge.errors import SpeechError
from speechforge.lanes import PRIO_EOU, PRIO_FINAL, PRIO_PARTIAL, SttLane
from speechforge.metrics import METRICS, P_BUCKETS
from speechforge.protocol import (
    KIND_MIC,
    KIND_TTS,
    MAX_MESSAGE_BYTES,
    MIC_RATE,
    PROTO,
    FrameError,
    check_control,
    dumps,
    pack_audio,
    unpack_audio,
)
from speechforge.stt_stream import RollingStt
from speechforge.tts_ctx import TtsContext
from speechforge.voices import parse_voice_id, sha256_hex

log = logging.getLogger("speechforge.session")

SendText = Callable[[str], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]
CloseFn = Callable[[int, str], Awaitable[None]]


class Session:
    def __init__(self, manager: SessionManager, sid: str, resume_token: str) -> None:
        self.m = manager
        self.cfg = manager.config.realtime
        self.sid = sid
        self.resume_token = resume_token
        self._send_text: SendText | None = None
        self._send_bytes: SendBytes | None = None
        self._close: CloseFn | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready = False
        self.closed = False
        self.detached_at: float | None = None
        self.stt: RollingStt | None = None
        self.stt_config: dict[str, Any] = {
            "partial_ms": self.cfg.partial_ms,
            "eou": {"enabled": True, "min_pause_ms": self.cfg.min_pause_ms},
        }
        self.ctxs: dict[int, TtsContext] = {}
        self.queued: list[TtsContext] = []
        self._partial_pending = False
        self._final_lock = asyncio.Lock()
        self._errors: deque[float] = deque()
        self._text_window: deque[tuple[float, int]] = deque()
        self._ingest_allow = float(MIC_RATE)  # token bucket, samples
        self._ingest_t = time.monotonic()
        self.last_frame = time.monotonic()
        self.turn_id: Any = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._voice_parts: dict[str, list[str | None]] = {}

    # ------------------------------------------------------------ wiring
    def attach(self, send_text: SendText, send_bytes: SendBytes, close: CloseFn) -> None:
        self._send_text, self._send_bytes, self._close = send_text, send_bytes, close
        self.loop = asyncio.get_running_loop()
        self.detached_at = None
        self.last_frame = time.monotonic()

    def detach(self) -> None:
        """Socket gone. Stop all TTS; keep the slot for ``resume_window_s``."""
        self._send_text = self._send_bytes = self._close = None
        self.detached_at = time.monotonic()
        self.ready = False
        self._stop_all_tts()
        if self.stt is not None:
            self.stt.reset_all()

    async def send(self, msg: dict[str, Any]) -> None:
        fn = self._send_text
        if fn is None:
            return
        with contextlib.suppress(Exception):
            await fn(dumps(msg))

    async def send_bin(self, data: bytes) -> None:
        fn = self._send_bytes
        if fn is None:
            return
        with contextlib.suppress(Exception):
            await fn(data)

    def post(self, msg: dict[str, Any]) -> None:
        """Thread-safe send (TTS producer threads)."""
        if self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self._spawn, self.send(msg))

    def post_bin(self, data: bytes) -> None:
        if self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self._spawn, self.send_bin(data))

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def error(
        self,
        code: str,
        message: str = "",
        *,
        fatal: bool = False,
        ctx: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        METRICS.inc("session_errors_total", code=code)
        msg: dict[str, Any] = {"type": "error", "code": code, "fatal": fatal}
        if message:
            msg["message"] = message[:200]
        if ctx is not None:
            msg["ctx"] = ctx
        if retry_after_s is not None:
            msg["retry_after_s"] = int(max(1, round(retry_after_s)))
        await self.send(msg)
        if not fatal and code in ("bad_frame", "too_long", "busy", "unknown_voice", "rate_limited"):
            now = time.monotonic()
            self._errors.append(now)
            while self._errors and now - self._errors[0] > self.cfg.error_window_s:
                self._errors.popleft()
            if len(self._errors) >= self.cfg.error_flood:
                await self.shutdown(1008, "error flood")

    async def shutdown(self, code: int = 1000, reason: str = "") -> None:
        self._stop_all_tts()
        close = self._close
        self.closed = True
        if close is not None:
            with contextlib.suppress(Exception):
                await close(code, reason)

    # ------------------------------------------------------------ hello / ready
    def ready_frame(self) -> dict[str, Any]:
        sp = self.m.residency.speaker
        lease = self.m.residency.lease
        engines: dict[str, Any] = {}
        if sp is not None:
            d = sp.describe
            shas = self.m.engine_shas()
            engines = {
                "stt": {**d.get("stt", {}), "sha8": shas.get("parakeet-tdt-0.6b-v2")},
                "tts": {
                    **d.get("tts", {}),
                    "sha8": shas.get("chatterbox-turbo"),
                    "caps": {"text_stream": False, "tags": True, "words": "estimated"},
                },
            }
            if "eou" in d:
                engines["eou"] = {**d["eou"], "sha8": shas.get("smart-turn")}
        return {
            "type": "ready",
            "session": self.sid,
            "resume_token": self.resume_token,
            "proto": PROTO,
            "version": __version__,
            "engines": engines,
            "gpu": self.m.gpu_info(),
            "limits": {
                "max_text": self.cfg.max_text_per_ctx,
                "ahead_ms_max": self.cfg.max_ahead_ms,
                "sessions": self.m.config.realtime.max_sessions,
                "contexts": self.cfg.max_contexts,
                "max_message": MAX_MESSAGE_BYTES,
            },
            "lease": {"state": self.m.lease_state(), "id": lease.lease_id if lease else None},
        }

    def start_stt(self) -> None:
        sp = self.m.residency.speaker
        vad_state = sp.vad.stream() if sp is not None and sp.vad is not None else None
        eou = self.stt_config.get("eou") or {}
        self.stt = RollingStt(
            partial_ms=int(self.stt_config.get("partial_ms", self.cfg.partial_ms)),
            min_pause_ms=int(eou.get("min_pause_ms", self.cfg.min_pause_ms)),
            max_tail_s=self.cfg.max_tail_s,
            max_utterance_s=self.cfg.max_utterance_s,
            vad=vad_state,
        )

    # ------------------------------------------------------------ inbound
    async def on_bytes(self, data: bytes) -> None:
        self.last_frame = time.monotonic()
        if len(data) > MAX_MESSAGE_BYTES:
            await self.error("too_long", "binary frame over 64 KiB")
            return
        try:
            frame = unpack_audio(data)
        except FrameError as exc:
            await self.error("bad_frame", str(exc))
            return
        if frame.kind != KIND_MIC:
            await self.error("bad_frame", "only mic frames (kind 0x01) go box -> rig")
            return
        if not self.ready or self.stt is None:
            return
        n = len(frame.pcm)
        now = time.monotonic()
        rate = MIC_RATE * self.cfg.max_ingest_ratio
        self._ingest_allow = min(
            float(MIC_RATE), self._ingest_allow + (now - self._ingest_t) * rate
        )
        self._ingest_t = now
        if n > self._ingest_allow:
            METRICS.inc("stt_ingest_drop_total", n)
            self.stt.dropped_samples += n
            return
        self._ingest_allow -= n
        for ev in self.stt.push(frame.pts, frame.pcm):
            if ev["type"] == "_pause":
                if (
                    self.stt_config.get("eou", {}).get("enabled", True)
                    and self.m.residency.speaker is not None
                    and self.m.residency.speaker.eou is not None
                ):
                    self._spawn(self._eou(ev["at"], ev["pause_ms"]))
            else:
                await self.send(ev)
        if not self._partial_pending and self.stt.partial_due():
            self._partial_pending = True
            self._spawn(self._partial())

    async def on_text(self, text: str) -> None:
        self.last_frame = time.monotonic()
        if len(text.encode()) > MAX_MESSAGE_BYTES:
            await self.error("too_long", "control frame over 64 KiB")
            return
        try:
            msg = check_control(text)
        except FrameError as exc:
            await self.error("bad_frame", str(exc))
            return
        handler = getattr(self, "_on_" + msg["type"].replace(".", "_"), None)
        if handler is None:
            await self.error("bad_frame", f"{msg['type']} is not valid here")
            return
        try:
            await handler(msg)
        except SpeechError as exc:
            await self.error(exc.code, exc.message, ctx=msg.get("ctx"))

    # ------------------------------------------------------------ control handlers
    async def _on_hello(self, msg: dict[str, Any]) -> None:
        await self.error("bad_frame", "hello was already received")

    async def _on_ping(self, msg: dict[str, Any]) -> None:
        pong: dict[str, Any] = {
            "type": "pong",
            "t": msg["t"],
            "srv_t": round(time.time(), 3),
            "queue": self.m.lane.depth(),
        }
        pong.update(self.m.gpu_live())
        await self.send(pong)

    async def _on_bye(self, msg: dict[str, Any]) -> None:
        self.m.forget(self)
        await self.shutdown(1000, "bye")

    async def _on_mark(self, msg: dict[str, Any]) -> None:
        self.turn_id = msg.get("turn_id", self.turn_id)

    async def _on_stt_config(self, msg: dict[str, Any]) -> None:
        if msg.get("lang", "en") not in ("en", None):
            await self.error("bad_frame", "only lang 'en' is supported")
            return
        pm = msg.get("partial_ms")
        if isinstance(pm, int) and not isinstance(pm, bool):
            self.stt_config["partial_ms"] = max(80, min(2000, pm))
        eou = msg.get("eou")
        if isinstance(eou, dict):
            cur = self.stt_config["eou"]
            if isinstance(eou.get("enabled"), bool):
                cur["enabled"] = eou["enabled"]
            mp = eou.get("min_pause_ms")
            if isinstance(mp, int) and not isinstance(mp, bool):
                cur["min_pause_ms"] = max(60, min(2000, mp))
        hot = msg.get("hotwords")
        if hot is not None and (not isinstance(hot, list) or len(hot) > 64):
            await self.error("bad_frame", "hotwords: at most 64 strings")
            return
        if self.stt is not None:
            self.stt.partial_ms = int(self.stt_config["partial_ms"])
            self.stt.min_pause_ms = int(self.stt_config["eou"]["min_pause_ms"])

    async def _on_stt_reset(self, msg: dict[str, Any]) -> None:
        if self.stt is not None:
            self.stt.reset_utterance()

    async def _on_stt_flush(self, msg: dict[str, Any]) -> None:
        if self.stt is None:
            return
        async with self._final_lock:
            job = self.stt.take_final(msg.get("id"))
            sp = self.m.residency.speaker
            if sp is None:
                await self.error("not_ready", "models unloaded")
                return
            t0 = time.perf_counter()

            def run() -> tuple[str, list[tuple[str, float, float]]]:
                return sp.transcribe_words(job.tail)

            try:
                text, words = await self.m.lane.submit(PRIO_FINAL, run)
            except Exception as exc:  # noqa: BLE001
                await self.error("engine_failed", type(exc).__name__)
                return
            ms = round((time.perf_counter() - t0) * 1000)
            METRICS.observe("stt_final_ms", ms)
            await self.send(self.stt.apply_final(job, text, words, ms))

    async def _partial(self) -> None:
        try:
            stt, sp = self.stt, self.m.residency.speaker
            if stt is None or sp is None:
                return
            job = stt.take_partial()
            t0 = time.perf_counter()

            def run() -> tuple[str, list[tuple[str, float, float]] | None, str]:
                commit = None
                if job.commit is not None and len(job.commit):
                    commit = sp.transcribe_words(job.commit)
                return (
                    commit[0] if commit else "",
                    commit[1] if commit else None,
                    sp.transcribe(job.tail),
                )

            ctext, cwords, tail = await self.m.lane.submit(PRIO_PARTIAL, run)
            if self.stt is not stt:
                return
            if job.commit is not None and cwords is not None:
                stt.apply_commit(ctext, cwords, job.commit_start)
            ms = round((time.perf_counter() - t0) * 1000)
            METRICS.observe("stt_decode_ms", ms)
            frame = stt.apply_partial(job, tail, ms)
            if frame is not None:
                METRICS.observe("stt_partial_lag_ms", frame["lag_ms"])
                await self.send(frame)
        except Exception as exc:  # noqa: BLE001 - a failed partial is skipped, not fatal
            log.warning("partial failed: %s", type(exc).__name__)
        finally:
            self._partial_pending = False

    async def _eou(self, at: int, pause_ms: int) -> None:
        stt, sp = self.stt, self.m.residency.speaker
        if stt is None or sp is None:
            return
        job = stt.take_eou(at, pause_ms)
        t0 = time.perf_counter()
        try:
            p = await self.m.lane.submit(PRIO_EOU, lambda: sp.eou_score(job.audio))
        except Exception as exc:  # noqa: BLE001
            log.warning("eou failed: %s", type(exc).__name__)
            return
        if p is None:
            return
        ms = round((time.perf_counter() - t0) * 1000)
        METRICS.observe("eou_ms", ms)
        METRICS.observe("eou_p", p, P_BUCKETS)
        METRICS.inc("eou_calls_total")
        await self.send(
            {
                "type": "eou",
                "p": round(p, 4),
                "at": job.at,
                "pause_ms": job.pause_ms,
                "ms": ms,
                "turn_id": self.turn_id,
            }
        )

    # ------------------------------------------------------------ TTS
    def _active(self) -> list[TtsContext]:
        return [c for c in self.ctxs.values() if not c.done]

    async def _on_tts_open(self, msg: dict[str, Any]) -> None:
        ctx_id = msg["ctx"]
        if self.m.vacating:
            await self.error(
                "vacated",
                "the rig is giving the card back; open on the CPU",
                ctx=ctx_id,
                retry_after_s=self.m.config.studioforge.vacate_cooldown_s,
            )
            return
        if ctx_id in self.ctxs and not self.ctxs[ctx_id].done:
            await self.error("bad_frame", "ctx is already open", ctx=ctx_id)
            return
        if len(self._active()) >= self.cfg.max_contexts:
            await self.error("busy", "one active and one queued context per session", ctx=ctx_id)
            return
        vid = parse_voice_id(msg["voice"])
        try:
            voice = await asyncio.to_thread(self.m.voices.get, vid)
        except SpeechError:
            await self.error("unknown_voice", f"voice.put sha256:{vid} first", ctx=ctx_id)
            return
        sp = self.m.residency.speaker
        if sp is None:
            await self.error("not_ready", "models unloaded", ctx=ctx_id, fatal=False)
            return
        fct = msg.get("first_chunk_tokens")
        seed = msg.get("seed")
        tc = TtsContext(
            ctx=ctx_id,
            gen=msg["gen"],
            voice_id=vid,
            voice=voice,
            speaker=sp,
            seed=seed if isinstance(seed, int) and not isinstance(seed, bool) else None,
            first_chunk_tokens=fct if isinstance(fct, int) and 4 <= fct <= 100 else None,
            turn_id=msg.get("turn_id"),
            speculative=bool(msg.get("speculative")),
            ahead_ms=self.cfg.default_ahead_ms,
            max_ahead_ms=self.cfg.max_ahead_ms,
            emit_audio=self.post_bin,
            emit=self._ctx_event,
            pack=lambda c, g, s, p, pcm: pack_audio(KIND_TTS, c, g, s, p, pcm),
        )
        self.ctxs[ctx_id] = tc
        for old in [k for k, c in self.ctxs.items() if c.done and k != ctx_id]:
            self.ctxs.pop(old, None)
        if any(not c.done and c.thread is not None for c in self.ctxs.values() if c is not tc):
            self.queued.append(tc)
        else:
            tc.start()

    def _ctx_event(self, msg: dict[str, Any]) -> None:
        self.post(msg)
        if msg["type"] in ("tts.done", "error") and self.loop is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self._start_next)

    def _start_next(self) -> None:
        while self.queued:
            nxt = self.queued.pop(0)
            if not nxt.done and not nxt.cancelled:
                nxt.start()
                return

    def _ctx(self, msg: dict[str, Any]) -> TtsContext:
        tc = self.ctxs.get(msg["ctx"])
        if tc is None or tc.done:
            raise SpeechError("no such open ctx", code="bad_frame")
        return tc

    async def _on_tts_text(self, msg: dict[str, Any]) -> None:
        tc = self._ctx(msg)
        text = msg["text"]
        if tc.text_chars + len(text) > self.cfg.max_text_per_ctx:
            raise SpeechError(
                f"over {self.cfg.max_text_per_ctx} chars in this ctx", code="too_long"
            )
        now = time.monotonic()
        while self._text_window and now - self._text_window[0][0] > 60:
            self._text_window.popleft()
        if sum(n for _, n in self._text_window) + len(text) > self.cfg.max_text_per_min:
            raise SpeechError("over the per-minute text budget", code="rate_limited")
        self._text_window.append((now, len(text)))
        tc.add_text(text, bool(msg.get("final")))

    async def _on_tts_flush(self, msg: dict[str, Any]) -> None:
        self._ctx(msg).flush(msg.get("id"))

    async def _on_tts_close(self, msg: dict[str, Any]) -> None:
        self._ctx(msg).close()

    async def _on_tts_pause(self, msg: dict[str, Any]) -> None:
        self._ctx(msg).pause()

    async def _on_tts_resume(self, msg: dict[str, Any]) -> None:
        self._ctx(msg).resume()

    async def _on_credit(self, msg: dict[str, Any]) -> None:
        tc = self.ctxs.get(msg["ctx"])
        if tc is None:
            return
        played = msg.get("played_ms")
        tc.set_credit(
            float(msg["ahead_ms"]),
            float(played)
            if isinstance(played, (int, float)) and not isinstance(played, bool)
            else None,
        )

    async def _on_cancel(self, msg: dict[str, Any]) -> None:
        t0 = time.perf_counter()
        gen, ctx = msg.get("gen"), msg.get("ctx")
        hit = [
            c
            for c in self.ctxs.values()
            if (ctx is not None and c.ctx == ctx) or (gen is not None and c.gen <= gen)
        ]
        for c in hit:
            c.cancel()
        self.queued = [c for c in self.queued if c not in hit]
        running = [c for c in hit if c.thread is not None]
        if running:
            await asyncio.to_thread(lambda: [c.exited.wait(0.5) for c in running])
        stop_ms = round((time.perf_counter() - t0) * 1000, 1)
        METRICS.observe("tts_cancel_stop_ms", stop_ms)
        for c in hit:
            METRICS.inc(
                "tts_wasted_ms_total", max(0.0, c.tokens_total * 40 - c.assumed_played_ms())
            )
            self.ctxs.pop(c.ctx, None)
        self._start_next()
        out: dict[str, Any] = {
            "type": "cancelled",
            "stop_ms": stop_ms,
            "ctxs": sorted(c.ctx for c in hit),
        }
        if gen is not None:
            out["gen"] = gen
        if ctx is not None:
            out["ctx"] = ctx
        await self.send(out)

    def _stop_all_tts(self) -> None:
        for c in self.ctxs.values():
            c.cancel()
        self.ctxs.clear()
        self.queued.clear()

    # ------------------------------------------------------------ voices
    async def _on_voice_has(self, msg: dict[str, Any]) -> None:
        vid = parse_voice_id(msg["id"])
        await self.send(
            {"type": "voice.has", "id": f"sha256:{vid}", "present": self.m.voices.exists(vid)}
        )

    async def _on_voice_delete(self, msg: dict[str, Any]) -> None:
        vid = parse_voice_id(msg["id"])
        await self.send(
            {"type": "voice.deleted", "id": f"sha256:{vid}", "deleted": self.m.voices.delete(vid)}
        )

    async def _on_voice_list(self, msg: dict[str, Any]) -> None:
        await self.send({"type": "voice.list", "ids": [f"sha256:{v}" for v in self.m.voices.ids()]})

    async def _on_voice_put(self, msg: dict[str, Any]) -> None:
        """Chunked: ``{id, part, parts, data_b64}``; parts <= 64 KiB each."""
        vid = parse_voice_id(msg["id"])
        parts = msg.get("parts", 1)
        part = msg.get("part", 0)
        if not (
            isinstance(parts, int)
            and isinstance(part, int)
            and 1 <= parts <= 64
            and 0 <= part < parts
        ):
            raise SpeechError("voice.put: part/parts out of range", code="bad_frame")
        buf = self._voice_parts.setdefault(vid, [None] * parts)
        if len(buf) != parts:
            self._voice_parts.pop(vid, None)
            raise SpeechError("voice.put: parts changed mid-upload", code="bad_frame")
        buf[part] = msg["data_b64"]
        if any(p is None for p in buf):
            return
        self._voice_parts.pop(vid, None)
        if not self.m.voice_put_allowed():
            raise SpeechError("voice.put: over the hourly limit", code="rate_limited")
        try:
            data = base64.b64decode("".join(p for p in buf if p), validate=True)
        except ValueError as exc:
            raise SpeechError("voice.put: bad base64", code="bad_frame") from exc
        if len(data) > self.m.config.voices.max_bytes:
            raise SpeechError("voice blob over the size limit", code="too_long")
        if sha256_hex(data) != vid:
            raise SpeechError("voice.put: sha256 does not match id", code="bad_frame")
        _, created = await asyncio.to_thread(self.m.voices.put, vid, data)
        await self.send({"type": "voice.stored", "id": f"sha256:{vid}", "created": created})


class SessionManager:
    def __init__(self, config: Any, residency: Any, voices: Any) -> None:
        self.config = config
        self.residency = residency
        self.voices = voices
        self.lane = SttLane()
        self.sessions: dict[str, Session] = {}
        self.vacating = False
        self._voice_puts: deque[float] = deque()
        self._gpu_cache: tuple[float, dict[str, Any]] = (0.0, {})
        self._shas: dict[str, str] | None = None
        residency.add_listener(self)

    # ------------------------------------------------------------ registry
    def live(self) -> list[Session]:
        return [s for s in self.sessions.values() if not s.closed]

    def attached(self) -> list[Session]:
        return [s for s in self.live() if s.detached_at is None]

    def expire_detached(self) -> None:
        now = time.monotonic()
        for s in list(self.sessions.values()):
            if s.closed or (
                s.detached_at is not None
                and now - s.detached_at >= self.config.realtime.resume_window_s
            ):
                self.forget(s)

    def capacity_left(self) -> int:
        self.expire_detached()
        return self.config.realtime.max_sessions - len(self.live())

    def open(self) -> Session:
        sid = secrets.token_hex(8)
        s = Session(self, sid, secrets.token_urlsafe(24))
        self.sessions[sid] = s
        self.residency.session_opened()
        METRICS.set("sessions_open", len(self.live()))
        return s

    def resume(self, sid: str, token: str) -> Session | None:
        self.expire_detached()
        s = self.sessions.get(sid)
        if s is None or s.closed or s.detached_at is None:
            return None
        if not secrets.compare_digest(s.resume_token.encode(), str(token).encode()):
            return None
        s.resume_token = secrets.token_urlsafe(24)
        return s

    def forget(self, s: Session) -> None:
        if self.sessions.pop(s.sid, None) is not None:
            s._stop_all_tts()
            s.closed = True
            self.residency.session_closed()
        METRICS.set("sessions_open", len(self.live()))

    # ------------------------------------------------------------ limits
    def voice_put_allowed(self) -> bool:
        now = time.monotonic()
        while self._voice_puts and now - self._voice_puts[0] > 3600:
            self._voice_puts.popleft()
        if len(self._voice_puts) >= self.config.realtime.voice_puts_per_hour:
            return False
        self._voice_puts.append(now)
        return True

    # ------------------------------------------------------------ facts
    def engine_shas(self) -> dict[str, str]:
        if self._shas is None:
            try:
                from speechforge import models

                self._shas = models.engine_shas(self.config)
            except Exception:  # noqa: BLE001
                self._shas = {}
        return self._shas

    def lease_state(self) -> str:
        r = self.residency
        if self.vacating:
            return "vacating"
        if r.lease is not None:
            return "held"
        return "unleased" if r.state == "ready" else "none"

    def gpu_info(self) -> dict[str, Any]:
        dev = self.residency.device
        info: dict[str, Any] = {"index": dev}
        info.update(self.residency.gpu_facts())
        return info

    def gpu_live(self) -> dict[str, Any]:
        now = time.monotonic()
        t, cached = self._gpu_cache
        if now - t > 1.0:
            cached = self.residency.gpu_live()
            self._gpu_cache = (now, cached)
        return cached

    # ------------------------------------------------------------ residency hooks
    async def on_tick(self) -> None:
        self.expire_detached()

    async def on_lease_state(self, state: str, deadline_s: float | None = None) -> None:
        msg: dict[str, Any] = {"type": "state", "lease": state}
        if deadline_s is not None:
            msg["deadline_s"] = round(deadline_s)
        for s in self.attached():
            await s.send(msg)

    async def before_unload(self, reason: str, timeout_s: float) -> None:
        """Vacate (or any unload with sessions open): stop new contexts, let the
        current ones finish (bounded), then the residency unloads."""
        if not self.live():
            return
        self.vacating = True
        await self.on_lease_state("vacating", timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and any(
            not c.done for s in self.live() for c in s.ctxs.values()
        ):
            await asyncio.sleep(0.05)

    async def after_unload(self, code: str, retry_after_s: float) -> None:
        self.vacating = False
        for s in self.live():
            await s.error(
                code,
                "the speech models were unloaded; use the CPU path",
                fatal=True,
                retry_after_s=retry_after_s,
            )
            self.forget(s)
            await s.shutdown(1001, code)

    def close(self) -> None:
        for s in list(self.sessions.values()):
            self.forget(s)
        self.lane.close()


def mic_frames(pcm16: np.ndarray, start_pts: int = 0, frame: int = 320) -> list[bytes]:
    """Split PCM16 @16 kHz into sf.1 mic frames (clients, bench, tests)."""
    out = []
    for i in range(0, len(pcm16), frame):
        out.append(pack_audio(KIND_MIC, 0, 0, i // frame, start_pts + i, pcm16[i : i + frame]))
    return out
