"""The HTTP surface. The realtime path is the sf.1 WebSocket (``realtime.py``);
everything here is a utility (SPEC-speechforge §4.1, §9).

==========================================  ==========  ===========================================
Route                                       Auth        What
==========================================  ==========  ===========================================
GET  /health                                no          ``{ok, version, api:1}`` (+ ``crashloop``)
GET  /ready                                 token       models, lease, self-test, engine shas
GET  /metrics                               loopback+   Prometheus text, no content
                                            token
WS   /v1/realtime                           token       the sf.1 session (``realtime.py``)
POST /v1/audio/transcriptions               token       OpenAI multipart (WAV ``file``) -> text
POST /v1/audio/transcriptions/raw           token       raw s16le 16 kHz mono -> ``{text, ...}``
POST /v1/audio/speech                       token       OpenAI JSON -> chunked PCM s16le 24 kHz
PUT  /v1/voices/{sha256}                    token       store voice conditionals (npz <= 2 MB)
HEAD/GET /v1/voices/{sha256}                token       200 present / 404
GET  /v1/voices                             token       the stored ids (audit)
DELETE /v1/voices/{sha256}                  token       forget a voice
POST /v1/voices/from-reference              token       403 unless voices.allow_reference_clip (D3)
POST /internal/vacate                       loopback +  StudioForge D56 vacate callback
                                            lease token
==========================================  ==========  ===========================================

Auth is ``Authorization: Bearer <token>`` checked by :class:`~speechforge.auth.TokenStore`
(constant time, rotation overlap). With no token configured only loopback
peers are served (and ``validate_for_serving`` refuses a non-loopback bind).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from speechforge import __version__
from speechforge.audio import (
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    pcm16_to_float,
    read_wav,
    resample_linear,
    streaming_wav_header,
    wav_bytes,
)
from speechforge.auth import TokenStore, bearer
from speechforge.config import Config, is_loopback_host
from speechforge.errors import (
    BadRequest,
    Forbidden,
    NotFound,
    SpeechError,
    TooLarge,
    Unauthorized,
    Unavailable,
)
from speechforge.lease import LeaseClient
from speechforge.metrics import METRICS
from speechforge.pipeline import Speaker, load_speaker
from speechforge.realtime import UpgradeLimiter, realtime_endpoint
from speechforge.residency import Residency
from speechforge.session import SessionManager
from speechforge.voices import VoiceStore, parse_voice_id

log = logging.getLogger("speechforge.api")

#: How long the speech route waits for the FIRST audio before giving up with
#: a JSON error (the status is committed with the first byte of audio).
FIRST_AUDIO_TIMEOUT_S = 30.0
_END = object()


class SpeechRequest(BaseModel):
    model: str = "chatterbox-turbo"
    input: str = Field(..., min_length=1)
    #: ``sha256:<hex>`` of a voice uploaded to /v1/voices.
    voice: str
    response_format: Literal["pcm", "wav"] = "pcm"
    stream: bool = True
    seed: int | None = None
    #: Override the configured first chunk for this request (4-100 tokens).
    first_chunk_tokens: int | None = Field(None, ge=4, le=100)
    #: Whole-sentence render with the full prompt (for cached lines) instead of
    #: the pipelined stream. Implies stream=false.
    render: bool = False
    speed: float | None = None  # accepted for OpenAI compatibility; must be 1.0 or absent


def create_app(
    config: Config,
    *,
    speaker_factory: Callable[[int], Speaker] | None = None,
    lease_client: LeaseClient | None | Literal["auto"] = "auto",
    clock: Callable[[], float] = time.monotonic,
    selftest: Callable[[Speaker], dict[str, Any]] | None | Literal["auto"] = "auto",
    residency_kw: dict[str, Any] | None = None,
) -> FastAPI:
    if lease_client == "auto":
        lease_client = None
        if config.studioforge.lease_mode == "hold":
            lease_client = LeaseClient(
                config.studioforge.url,
                holder=config.studioforge.holder,
                priority=config.studioforge.priority,
                idle_ttl_s=config.lease_idle_ttl_s,
                vacate_url=config.vacate_url,
                vacate_token=secrets.token_urlsafe(32),
                timeout_s=config.studioforge.timeout_s,
                api_key=config.studioforge_key,
            )
    factory = speaker_factory or (lambda device: load_speaker(config, device))
    if selftest == "auto":
        from speechforge import selftest as st

        def selftest(sp: Speaker) -> dict[str, Any]:  # type: ignore[misc]
            return st.run(sp, config.selftest, enforce_timing=config.gpu.provider == "cuda")

    residency = Residency(
        config,
        factory=factory,
        lease=lease_client,
        clock=clock,
        selftest=selftest,
        **(residency_kw or {}),
    )
    from speechforge.engines import base as engines

    engines.builtin()
    parser = engines.VOICE_PARSERS.get(config.tts.engine)
    if parser is None:
        raise ValueError(f"unknown tts.engine {config.tts.engine!r}")
    voices = VoiceStore(config.voices_dir, parse=parser, max_files=config.voices.max_files)
    tts_slots = asyncio.Semaphore(config.tts.max_http_streams)
    stt_slots = asyncio.Semaphore(1)
    max_body = int(config.server.max_body_mb * 1024 * 1024)
    tokens = TokenStore(config.env_file)
    manager = SessionManager(config, residency, voices)
    limiter = UpgradeLimiter(config.server.upgrades_per_ip_per_min)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await residency.start()
        try:
            yield
        finally:
            manager.close()
            await residency.stop()

    app = FastAPI(
        title="SpeechForge",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config
    app.state.residency = residency
    app.state.voices = voices
    app.state.sessions = manager
    app.state.tokens = tokens

    # ------------------------------------------------------------ plumbing
    @app.exception_handler(SpeechError)
    async def _speech_error(_: Request, exc: SpeechError) -> JSONResponse:
        if exc.status_code >= 500 and exc.status_code != 503:
            log.error("request failed: %s (%s)", exc.code, exc.message)
        return JSONResponse(exc.to_payload(), status_code=exc.status_code, headers=exc.headers())

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        err = BadRequest(f"{where or 'request'}: {first.get('msg', 'invalid')}")
        return JSONResponse(err.to_payload(), status_code=400)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        err = SpeechError(f"internal error ({type(exc).__name__})")
        return JSONResponse(err.to_payload(), status_code=500)

    def authorize(request: Request) -> None:
        if tokens.configured:
            if tokens.check(bearer(request.headers.get("authorization"))):
                return
            raise Unauthorized("missing or wrong bearer token")
        peer = request.client.host if request.client else ""
        if not is_loopback_host(peer):
            raise Unauthorized("no token is configured, so only this machine may call")

    async def read_body(request: Request, limit: int) -> bytes:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise TooLarge(f"body is larger than {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                raise TooLarge(f"body is larger than {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------ health / ready / metrics
    crash_file = config.state_dir / "crashloop.json"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True, "version": __version__, "api": 1}
        if crash_file.exists():
            out["crashloop"] = True
        return out

    @app.get("/ready")
    async def ready(request: Request) -> JSONResponse:
        authorize(request)
        st = residency.status()
        body = {
            "ready": st["ready"],
            "version": __version__,
            **st,
            "engine_shas": manager.engine_shas(),
            "sessions_open": len(manager.live()),
            "voices": voices.count(),
        }
        return JSONResponse(
            body,
            status_code=200 if st["ready"] else 503,
            headers={} if st["ready"] else {"Retry-After": "3"},
        )

    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        peer = request.client.host if request.client else ""
        if not is_loopback_host(peer):
            raise Forbidden("metrics are served to this machine only")
        authorize(request)
        METRICS.set("sessions_open", len(manager.live()))
        METRICS.set("lease_state", {"held": 2, "unleased": 1}.get(manager.lease_state(), 0))
        return Response(METRICS.render(), media_type="text/plain; version=0.0.4")

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket) -> None:
        await realtime_endpoint(ws, manager, tokens, limiter)

    # ------------------------------------------------------------ STT
    async def _transcribe(audio: np.ndarray) -> dict[str, Any]:
        if len(audio) > config.stt.max_audio_s * STT_SAMPLE_RATE:
            raise TooLarge(f"audio is longer than stt.max_audio_s ({config.stt.max_audio_s} s)")
        speaker = residency.require()
        if stt_slots.locked():
            raise Unavailable(
                "speech-to-text is at its concurrency limit", code="busy", retry_after_s=1
            )
        async with stt_slots, residency.use():
            t0 = time.perf_counter()
            text = await asyncio.to_thread(speaker.transcribe, audio)
            cost = round((time.perf_counter() - t0) * 1000)
        return {
            "text": text,
            "audio_ms": round(len(audio) / STT_SAMPLE_RATE * 1000),
            "cost_ms": cost,
        }

    @app.post("/v1/audio/transcriptions/raw")
    async def transcribe_raw(request: Request, sample_rate: int = STT_SAMPLE_RATE) -> dict:
        authorize(request)
        if sample_rate != STT_SAMPLE_RATE:
            raise BadRequest(
                "raw audio must be 16000 Hz s16le mono (sample_rate=16000)", code="bad_audio"
            )
        limit = min(max_body, int(config.stt.max_audio_s * STT_SAMPLE_RATE * 2) + 1)
        raw = await read_body(request, limit)
        return await _transcribe(pcm16_to_float(raw))

    @app.post("/v1/audio/transcriptions", response_model=None)
    async def transcribe_openai(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("parakeet-tdt-0.6b-v2"),
        response_format: str = Form("json"),
        language: str | None = Form(None),
    ) -> Any:
        authorize(request)
        if language and not language.lower().startswith("en"):
            raise BadRequest("Parakeet TDT 0.6B v2 transcribes English only", code="bad_language")
        data = await file.read(max_body + 1)
        if len(data) > max_body:
            raise TooLarge(f"file is larger than {max_body} bytes")
        audio, rate = read_wav(data)
        if rate != STT_SAMPLE_RATE:
            audio = resample_linear(audio, rate, STT_SAMPLE_RATE)
        result = await _transcribe(audio)
        if response_format == "text":
            return Response(result["text"], media_type="text/plain")
        if response_format == "verbose_json":
            return {**result, "language": "en", "duration": result["audio_ms"] / 1000}
        return {"text": result["text"]}

    # ------------------------------------------------------------ TTS
    @app.post("/v1/audio/speech")
    async def speech(body: SpeechRequest, request: Request) -> Response:
        authorize(request)
        if body.speed not in (None, 1.0):
            raise BadRequest("speed other than 1.0 is not supported", code="bad_speed")
        text = body.input.strip()
        if len(text) > config.realtime.max_text_per_ctx:
            raise TooLarge(f"input is over {config.realtime.max_text_per_ctx} chars")
        vid = parse_voice_id(body.voice)
        voice = await asyncio.to_thread(voices.get, vid)  # 404 before anything else
        speaker = residency.require()
        if tts_slots.locked():
            raise Unavailable(
                "text-to-speech is at its concurrency limit", code="busy", retry_after_s=1
            )
        headers = {
            "X-Sample-Rate": str(TTS_SAMPLE_RATE),
            "X-Sample-Format": "s16le",
            "X-Channels": "1",
            "X-Voice": f"sha256:{vid}",
            "Cache-Control": "no-store",
        }
        media = "audio/wav" if body.response_format == "wav" else "audio/pcm"

        if body.render or not body.stream:
            async with tts_slots, residency.use():
                if body.render:
                    pcm = await asyncio.to_thread(speaker.render, text, voice, seed=body.seed)
                else:
                    pcm = await asyncio.to_thread(
                        lambda: np.concatenate(
                            list(
                                speaker.speak(
                                    text,
                                    vid,
                                    voice,
                                    seed=body.seed,
                                    first_chunk_tokens=body.first_chunk_tokens,
                                )
                            )
                            or [np.zeros(0, "<i2")]
                        )
                    )
            data = (
                wav_bytes(pcm, TTS_SAMPLE_RATE)
                if body.response_format == "wav"
                else pcm.astype("<i2").tobytes()
            )
            return Response(data, media_type=media, headers=headers)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()
        await tts_slots.acquire()
        use = residency.use()
        stop = await use.__aenter__()
        released = False

        async def release() -> None:
            nonlocal released
            if not released:
                released = True
                stop.set()
                tts_slots.release()
                await use.__aexit__(None, None, None)

        def put(item: Any) -> None:
            with contextlib.suppress(RuntimeError):  # loop closed at shutdown
                loop.call_soon_threadsafe(q.put_nowait, item)

        def produce() -> None:
            try:
                for piece in speaker.speak(
                    text,
                    vid,
                    voice,
                    stop=stop,
                    seed=body.seed,
                    first_chunk_tokens=body.first_chunk_tokens,
                ):
                    if stop.is_set():
                        break
                    put(piece.astype("<i2").tobytes())
            except BaseException as exc:  # noqa: BLE001 - handed to the response
                put(exc)
            finally:
                put(_END)

        threading.Thread(target=produce, name="sf-speech", daemon=True).start()
        try:
            first = await asyncio.wait_for(q.get(), FIRST_AUDIO_TIMEOUT_S)
        except TimeoutError:
            await release()
            raise Unavailable(
                "no audio within the first-audio deadline", code="timeout", retry_after_s=5
            ) from None
        if first is _END or isinstance(first, BaseException):
            await release()
            if isinstance(first, BaseException):
                log.error("synthesis failed before any audio: %s", type(first).__name__)
                raise SpeechError(
                    f"synthesis failed ({type(first).__name__})", code="synthesis_failed"
                )
            return Response(
                b""
                if body.response_format == "pcm"
                else wav_bytes(np.zeros(0, "<i2"), TTS_SAMPLE_RATE),
                media_type=media,
                headers=headers,
            )

        async def gen() -> AsyncIterator[bytes]:
            try:
                if body.response_format == "wav":
                    yield streaming_wav_header(TTS_SAMPLE_RATE)
                yield first
                while True:
                    item = await q.get()
                    if item is _END:
                        return
                    if isinstance(item, BaseException):
                        # Audio already went out: end the stream abnormally so the
                        # client sees an error, not a clean (truncated) sentence.
                        log.error("synthesis failed mid-stream: %s", type(item).__name__)
                        raise RuntimeError("synthesis failed mid-stream")
                    yield item
            finally:
                await release()

        return StreamingResponse(gen(), media_type=media, headers=headers)

    # ------------------------------------------------------------ voices
    @app.put("/v1/voices/{voice_id}")
    async def put_voice(voice_id: str, request: Request) -> JSONResponse:
        authorize(request)
        vid = parse_voice_id(voice_id)
        data = await read_body(request, min(max_body, config.voices.max_bytes))
        _, created = await asyncio.to_thread(voices.put, vid, data)
        return JSONResponse(
            {"id": f"sha256:{vid}", "created": created}, status_code=201 if created else 200
        )

    @app.api_route("/v1/voices/{voice_id}", methods=["GET", "HEAD"])
    async def has_voice(voice_id: str, request: Request) -> JSONResponse:
        authorize(request)
        vid = parse_voice_id(voice_id)
        if not voices.exists(vid):
            raise NotFound(f"voice sha256:{vid} is not on this server", code="voice_not_found")
        return JSONResponse({"id": f"sha256:{vid}", "present": True})

    @app.get("/v1/voices")
    async def list_voices(request: Request) -> dict[str, Any]:
        authorize(request)
        return {"ids": [f"sha256:{v}" for v in voices.ids()]}

    @app.delete("/v1/voices/{voice_id}")
    async def delete_voice(voice_id: str, request: Request) -> dict[str, Any]:
        authorize(request)
        vid = parse_voice_id(voice_id)
        return {"id": f"sha256:{vid}", "deleted": voices.delete(vid)}

    @app.post("/v1/voices/from-reference")
    async def voice_from_reference(request: Request) -> JSONResponse:
        """Body: a 16-bit WAV reference clip (3-30 s). Built with this server's
        own speech encoder; returns the id to speak with. Off by default: the
        reference clip stays on the box (decision D3)."""
        authorize(request)
        if not config.voices.allow_reference_clip:
            raise Forbidden(
                "reference clips stay on the box (voices.allow_reference_clip is "
                "off): upload conditionals with PUT /v1/voices/<sha256>",
                code="reference_clip_disabled",
            )
        data = await read_body(request, max_body)
        audio, rate = read_wav(data)
        if rate != TTS_SAMPLE_RATE:
            audio = resample_linear(audio, rate, TTS_SAMPLE_RATE)
        seconds = len(audio) / TTS_SAMPLE_RATE
        if not 3.0 <= seconds <= 30.0:
            raise BadRequest(f"reference clip is {seconds:.1f} s; use 3-30 s", code="bad_audio")
        speaker = residency.require()
        async with residency.use():
            data = await asyncio.to_thread(speaker.build_voice, audio, TTS_SAMPLE_RATE)
        vid = await asyncio.to_thread(voices.put_bytes, data)
        return JSONResponse({"id": f"sha256:{vid}"}, status_code=201)

    # ------------------------------------------------------------ vacate (D56)
    @app.post("/internal/vacate")
    async def vacate(request: Request) -> JSONResponse:
        peer = request.client.host if request.client else ""
        if not is_loopback_host(peer):
            raise Forbidden("the vacate callback is accepted from this machine only")
        token = request.headers.get("x-sf-vacate-token")
        try:
            payload = await request.json()
        except ValueError:
            payload = {}
        lease_id = str(
            (payload or {}).get("lease_id") or request.headers.get("x-sf-lease-id") or ""
        )
        result = await residency.vacate(lease_id, token)
        return JSONResponse(result, status_code=202)

    return app
