"""The ``sf.1`` wire protocol (SPEC-speechforge §4.3): frame codec and control-frame checks.

Binary frames, little-endian, 16-byte header then PCM16 mono::

    0  u8   magic 0x53 ('S')
    1  u8   kind  0x01 mic PCM in (16 kHz, 20 ms = 320 samples) | 0x02 TTS audio out (24 kHz, 40 ms)
    2  u16  ctx   TTS context id (0 for mic)
    4  u16  gen   box generation (0 for mic)
    6  u32  seq   per-ctx frame counter
    10 u32  pts   first-sample index: mic @16 kHz from session start; TTS @24 kHz from ctx start
    14 u16  n     sample count
    16 2n   PCM16

Control frames are JSON text objects with a ``type``. :func:`check_control`
validates the box->rig frames against :data:`CLIENT_FRAMES` (required fields
and their JSON types; extra fields are allowed -- sf.1 is additive). Unknown
types are ``bad_frame`` errors that never close the session.

Protocol notes where the spec left a choice (recorded in DECISIONS S11):

* ``cancel{gen}`` stops every ctx whose gen is **<= gen** (the given generation
  and every older one); ``cancel{ctx}`` stops that ctx.
* ``credit`` carries ``ahead_ms`` and, optionally, ``played_ms`` (audio of that
  ctx the phone has played). Without ``played_ms`` the rig assumes real-time
  playback from the ctx's first frame.
* ``tts.pause{ctx}`` / ``tts.resume{ctx}`` freeze and release a ctx (a false
  interruption): production stops at the next token while paused.
* ``voice.put`` over the socket is chunked (``part``/``parts``/``data_b64``,
  each message <= 64 KiB); the HTTP ``PUT /v1/voices/<id>`` is the usual path.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTO = "sf.1"
MAGIC = 0x53
KIND_MIC = 0x01
KIND_TTS = 0x02
HEADER = struct.Struct("<BBHHIIH")
HEADER_SIZE = HEADER.size  # 16
MIC_RATE = 16000
TTS_RATE = 24000
MIC_FRAME_SAMPLES = 320  # 20 ms
TTS_FRAME_SAMPLES = 960  # 40 ms
MAX_MESSAGE_BYTES = 64 * 1024
U16 = 0xFFFF
U32 = 0xFFFFFFFF

ERROR_CODES = (
    "bad_frame",
    "unknown_voice",
    "busy",
    "too_long",
    "engine_failed",
    "vacated",
    "internal",
    "not_ready",
    "rate_limited",
)


class FrameError(ValueError):
    """A frame the session rejects with ``error{bad_frame}``."""


@dataclass(frozen=True)
class AudioFrame:
    kind: int
    ctx: int
    gen: int
    seq: int
    pts: int
    pcm: np.ndarray  # int16


def pack_audio(kind: int, ctx: int, gen: int, seq: int, pts: int, pcm16: np.ndarray) -> bytes:
    pcm = np.asarray(pcm16, dtype="<i2")
    n = int(pcm.shape[0])
    if n > U16:
        raise FrameError("too many samples for one frame")
    return HEADER.pack(MAGIC, kind, ctx & U16, gen & U16, seq & U32, pts & U32, n) + pcm.tobytes()


def unpack_audio(buf: bytes) -> AudioFrame:
    if len(buf) < HEADER_SIZE:
        raise FrameError("binary frame shorter than the 16-byte header")
    magic, kind, ctx, gen, seq, pts, n = HEADER.unpack_from(buf)
    if magic != MAGIC:
        raise FrameError("bad magic")
    if kind not in (KIND_MIC, KIND_TTS):
        raise FrameError(f"unknown binary kind {kind}")
    if len(buf) != HEADER_SIZE + 2 * n:
        raise FrameError("sample count does not match the frame length")
    pcm = np.frombuffer(buf, dtype="<i2", offset=HEADER_SIZE, count=n)
    return AudioFrame(kind, ctx, gen, seq, pts, pcm)


# ---------------------------------------------------------------- control frames
_NUM = (int, float)
#: type -> {field: allowed JSON types} for REQUIRED fields.
CLIENT_FRAMES: dict[str, dict[str, tuple[type, ...]]] = {
    "hello": {"proto": (str,)},
    "stt.config": {},
    "stt.flush": {},
    "stt.reset": {},
    "mark": {},
    "tts.open": {"ctx": (int,), "gen": (int,), "voice": (str,)},
    "tts.text": {"ctx": (int,), "text": (str,)},
    "tts.flush": {"ctx": (int,)},
    "tts.close": {"ctx": (int,)},
    "tts.pause": {"ctx": (int,)},
    "tts.resume": {"ctx": (int,)},
    "cancel": {},
    "credit": {"ctx": (int,), "ahead_ms": _NUM},
    "voice.put": {"id": (str,), "data_b64": (str,)},
    "voice.has": {"id": (str,)},
    "voice.delete": {"id": (str,)},
    "voice.list": {},
    "ping": {"t": _NUM},
    "bye": {},
}


def check_control(text: str) -> dict[str, Any]:
    """Parse and validate one box->rig control frame. Raises :class:`FrameError`."""
    try:
        msg = json.loads(text)
    except ValueError as exc:
        raise FrameError("control frame is not JSON") from exc
    if not isinstance(msg, dict):
        raise FrameError("control frame must be a JSON object")
    kind = msg.get("type")
    if not isinstance(kind, str) or kind not in CLIENT_FRAMES:
        raise FrameError(f"unknown frame type {kind!r}")
    for field, types in CLIENT_FRAMES[kind].items():
        value = msg.get(field)
        if value is None or isinstance(value, bool) or not isinstance(value, types):
            raise FrameError(f"{kind}: field {field!r} missing or of the wrong type")
    for field in ("ctx", "gen"):
        value = msg.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= U16
        ):
            raise FrameError(f"{kind}: {field} must be an integer 0..65535")
    if kind == "cancel" and msg.get("gen") is None and msg.get("ctx") is None:
        raise FrameError("cancel needs gen or ctx")
    return msg


def dumps(msg: dict[str, Any]) -> str:
    return json.dumps(msg, separators=(",", ":"), allow_nan=False)
