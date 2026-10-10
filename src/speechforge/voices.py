"""Voice profiles: Chatterbox's four speech-encoder tensors, keyed by content hash.

A cloned voice is not a reference clip at synthesis time; it is what the
speech encoder made of one (``cond_emb`` for the language model,
``prompt_token`` + ``speaker_embeddings`` + ``speaker_features`` for the
decoder). The DisPatch voice add-on already keeps exactly these tensors per
profile (``voice.npz``), so the rig takes them as they are (S6):

* ``PUT /v1/voices/<sha256>`` with the npz bytes; the id IS the sha256 of the
  bytes, so a client knows the id before uploading, ``HEAD`` answers "do you
  have it?", and an upload can never be stored under the wrong name.
* The same tensors on both sides means the rig and the client's CPU fallback
  speak with one timbre; a hedged sentence that falls back does not change
  voice.

Both naming schemes are accepted: the add-on's ``VoiceTensors.save`` keys
(``cond_emb``, ``prompt_token``, ...) and the speech encoder's own output names
(``audio_features``, ``audio_tokens``, ...). ``allow_pickle=False`` always; every
array is shape- and size-checked before it is kept, because an npz is a zip
of arrays from the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
import threading
import time
import zipfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.errors import BadRequest, NotFound

HIDDEN = 1024
SPEAKER_DIM = 192
MEL_BINS = 80
#: Caps: a 30 s reference is 750 prompt tokens / 1500 feature frames; the
#: conditioning prefix the encoder emits is ~150 positions.
MAX_COND = 1024
MAX_PROMPT_TOKENS = 1500
MAX_FEATURE_FRAMES = 3000
MAX_NPZ_BYTES = 32 * 1024 * 1024

_ALIASES = {
    "cond_emb": ("cond_emb", "audio_features"),
    "prompt_token": ("prompt_token", "audio_tokens"),
    "speaker_embeddings": ("speaker_embeddings",),
    "speaker_features": ("speaker_features",),
}
_ID_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


@dataclass(frozen=True)
class VoiceTensors:
    cond_emb: np.ndarray  # (1, N, 1024) float32
    prompt_token: np.ndarray  # (1, T) int64
    speaker_embeddings: np.ndarray  # (1, 192) float32
    speaker_features: np.ndarray  # (1, F, 80) float32

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.arrays())

    def arrays(self) -> tuple[np.ndarray, ...]:
        return (self.cond_emb, self.prompt_token, self.speaker_embeddings, self.speaker_features)

    def to_npz(self) -> bytes:
        buf = io.BytesIO()
        np.savez(
            buf,
            cond_emb=self.cond_emb,
            prompt_token=self.prompt_token,
            speaker_embeddings=self.speaker_embeddings,
            speaker_features=self.speaker_features,
        )
        return buf.getvalue()


def parse_voice_id(text: str) -> str:
    """``sha256:<hex>`` or bare ``<hex>`` (lower-case) -> the hex. 400 otherwise."""
    m = _ID_RE.match((text or "").strip())
    if not m:
        raise BadRequest(
            "voice must be 'sha256:<64 lower-case hex>' (the sha256 of the npz)",
            code="bad_voice_id",
        )
    return m.group(1)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_npz(data: bytes) -> VoiceTensors:
    """Validate and load voice tensors from npz bytes. Raises 400 ``bad_voice``."""
    if len(data) > MAX_NPZ_BYTES:
        raise BadRequest(f"voice file is larger than {MAX_NPZ_BYTES} bytes", code="bad_voice")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
        if total > 4 * MAX_NPZ_BYTES:  # a zip bomb stops here, before numpy inflates it
            raise BadRequest("voice file inflates too large", code="bad_voice")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            names = set(z.files)
            got: dict[str, np.ndarray] = {}
            for key, aliases in _ALIASES.items():
                name = next((a for a in aliases if a in names), None)
                if name is None:
                    raise BadRequest(f"voice file lacks {key!r}", code="bad_voice")
                got[key] = np.asarray(z[name])
    except BadRequest:
        raise
    except Exception as exc:  # zipfile.BadZipFile, ValueError (pickled object), OSError
        raise BadRequest(f"not a valid voice npz ({type(exc).__name__})", code="bad_voice") from exc

    def check(name: str, arr: np.ndarray, ndim: int, kinds: str) -> None:
        if arr.ndim != ndim or arr.dtype.kind not in kinds or arr.shape[0] != 1:
            raise BadRequest(
                f"{name}: unexpected shape {arr.shape} / dtype {arr.dtype}", code="bad_voice"
            )
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise BadRequest(f"{name}: non-finite values", code="bad_voice")

    cond, prompt = got["cond_emb"], got["prompt_token"]
    emb, feat = got["speaker_embeddings"], got["speaker_features"]
    check("cond_emb", cond, 3, "f")
    check("prompt_token", prompt, 2, "iu")
    check("speaker_embeddings", emb, 2, "f")
    check("speaker_features", feat, 3, "f")
    if cond.shape[2] != HIDDEN or not 1 <= cond.shape[1] <= MAX_COND:
        raise BadRequest(
            f"cond_emb: shape {cond.shape} is not (1, 1..{MAX_COND}, {HIDDEN})", code="bad_voice"
        )
    if not 1 <= prompt.shape[1] <= MAX_PROMPT_TOKENS:
        raise BadRequest(f"prompt_token: {prompt.shape[1]} tokens", code="bad_voice")
    if prompt.min() < 0 or prompt.max() >= 6561:
        raise BadRequest("prompt_token: values outside the speech vocabulary", code="bad_voice")
    if emb.shape[1] != SPEAKER_DIM:
        raise BadRequest(f"speaker_embeddings: shape {emb.shape}", code="bad_voice")
    if feat.shape[2] != MEL_BINS or not 1 <= feat.shape[1] <= MAX_FEATURE_FRAMES:
        raise BadRequest(f"speaker_features: shape {feat.shape}", code="bad_voice")
    return VoiceTensors(
        np.ascontiguousarray(cond, np.float32),
        np.ascontiguousarray(prompt, np.int64),
        np.ascontiguousarray(emb, np.float32),
        np.ascontiguousarray(feat, np.float32),
    )


class VoiceStore:
    """Content-addressed npz files on disk plus a small in-memory LRU.

    ``max_files`` bounds the directory (oldest by mtime go first; a read
    refreshes mtime). Thread-safe: the synthesis threads read, the routes write.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = 256,
        max_memory: int = 32,
        parse: Callable[[bytes], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self._parse = parse or parse_npz
        self.max_files = max_files
        self.max_memory = max_memory
        self._mem: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def _path(self, vid: str) -> Path:
        return self.root / f"{vid}.npz"

    def exists(self, vid: str) -> bool:
        with self._lock:
            if vid in self._mem:
                return True
        return self._path(vid).is_file()

    def put(self, vid: str, data: bytes) -> tuple[Any, bool]:
        """Store ``data`` under ``vid``; returns (tensors, created). 400 when the
        bytes do not hash to ``vid`` or do not parse."""
        if sha256_hex(data) != vid:
            raise BadRequest(
                "the body's sha256 does not match the voice id in the URL",
                code="voice_hash_mismatch",
            )
        voice = self._parse(data)
        path = self._path(vid)
        created = not path.is_file()
        if created:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            self._prune()
        self._remember(vid, voice)
        return voice, created

    def put_bytes(self, data: bytes) -> str:
        vid = sha256_hex(data)
        self.put(vid, data)
        return vid

    def get(self, vid: str) -> Any:
        with self._lock:
            voice = self._mem.get(vid)
            if voice is not None:
                self._mem.move_to_end(vid)
                return voice
        path = self._path(vid)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise NotFound(
                f"voice sha256:{vid} is not on this server; PUT it to /v1/voices/{vid} and retry",
                code="voice_not_found",
            ) from None
        voice = self._parse(data)
        with contextlib.suppress(OSError):
            now = time.time()
            os.utime(path, (now, now))
        self._remember(vid, voice)
        return voice

    def delete(self, vid: str) -> bool:
        with self._lock:
            self._mem.pop(vid, None)
        try:
            self._path(vid).unlink()
            return True
        except FileNotFoundError:
            return False

    def ids(self) -> list[str]:
        try:
            return sorted(p.stem for p in self.root.glob("*.npz"))
        except OSError:
            return []

    def count(self) -> int:
        try:
            return sum(1 for _ in self.root.glob("*.npz"))
        except OSError:
            return 0

    def _remember(self, vid: str, voice: Any) -> None:
        with self._lock:
            self._mem[vid] = voice
            self._mem.move_to_end(vid)
            while len(self._mem) > self.max_memory:
                self._mem.popitem(last=False)

    def _prune(self) -> None:
        try:
            files = sorted(self.root.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for path in files[: max(0, len(files) - self.max_files)]:
            with contextlib.suppress(OSError):
                path.unlink()
            with self._lock:
                self._mem.pop(path.stem, None)
