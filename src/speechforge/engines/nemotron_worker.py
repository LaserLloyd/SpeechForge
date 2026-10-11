"""sherpa-onnx side of the Nemotron streaming engine, usable in-process or as a
child process.

Standalone on purpose: this file imports only the standard library, numpy and
(lazily) ``sherpa_onnx``, so the ``subprocess`` backend can run it with a
*different* interpreter -- the one whose venv holds sherpa-onnx's CUDA wheel
and its CUDA 12 runtime -- without that venv needing SpeechForge installed
(``python nemotron_worker.py``). ``engines/nemotron.py`` imports
:class:`SherpaRecognizer` from here for the in-process backend and talks to
:func:`serve` over pipes for the subprocess one.

Why a child process exists at all (docs/ENGINES.md): sherpa-onnx's GPU wheel
bundles its own ``onnxruntime.dll`` + ``onnxruntime_providers_cuda.dll`` built
for CUDA 12 / cuDNN 9, and this server's ``onnxruntime-gpu`` is the CUDA 13
build with its own copies of DLLs of the same names. Windows binds an already
loaded module by base name, so whichever loads second gets the other's
library; one process cannot hold both safely.

Pipe protocol (parent -> child on stdin, child -> parent on stdout, one JSON
object per line, UTF-8; ``feed`` is followed by ``n`` float32 little-endian
samples as raw bytes):

    {"op": "load", ...model paths and options...}      -> {"ok": true, "describe": {...}}
    {"op": "create"}                                   -> {"sid": 1}
    {"op": "feed", "sid": 1, "n": 1600} + bytes        -> {"text": "..."}
    {"op": "finalize", "sid": 1}                       -> {"text": "...", "words": [[w, s, e], ...]}
    {"op": "reset", "sid": 1}                          -> {"ok": true}
    {"op": "close_stream", "sid": 1}                   -> {"ok": true}
    {"op": "close"}                                    -> {"ok": true} and exit 0

Any failure answers ``{"error": "<message>", "type": "<ExceptionName>"}`` and
keeps serving; a protocol error (bad JSON, short read) exits 2.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

SAMPLE_RATE = 16000
FEATURE_DIM = 128  # the export's log-mel bins (encoder metadata feat_dim)
#: Longest ``feed`` the pipe accepts (10 minutes of 16 kHz audio, 38 MB): a frame
#: count beyond it is a protocol error, not a buffer the child tries to allocate.
MAX_FEED_SAMPLES = SAMPLE_RATE * 600
WORD_END_PAD_S = 0.08  # the last word ends one 80 ms frame after its last token


def words_from_sp_tokens(tokens: list[str], times: list[float]) -> list[tuple[str, float, float]]:
    """``(word, start_s, end_s)`` from sentencepiece tokens (``▁`` starts a
    word) and their frame times: a word ends where the next one starts, the
    last one 80 ms after its last token. Same rule as Parakeet's."""
    words: list[list[Any]] = []
    for tok, t in zip(tokens, times, strict=False):
        starts = tok.startswith("▁") or tok.startswith(" ")
        text = tok.replace("▁", " ")
        if not words or starts:
            text = text.strip()
            if text:
                words.append([text, float(t), float(t)])
            elif words:
                words[-1][2] = float(t)
        else:
            words[-1][0] += text
            words[-1][2] = float(t)
    out: list[tuple[str, float, float]] = []
    for i, (w, start, last) in enumerate(words):
        end = words[i + 1][1] if i + 1 < len(words) else last + WORD_END_PAD_S
        out.append((w, round(start, 3), round(max(end, start), 3)))
    return out


def add_cuda_dll_dirs() -> list[str]:
    """Windows only: put the ``nvidia-*-cu12`` pip wheels' ``bin`` directories
    (and sherpa-onnx's own ``lib``) on the DLL search path of *this* process,
    so the bundled CUDA 12 ONNX Runtime finds cudart/cublas/cudnn without a
    system CUDA toolkit. Returns the directories added (for the log)."""
    if sys.platform != "win32":
        return []
    added: list[str] = []
    roots: list[Path] = []
    for entry in sys.path:
        p = Path(entry) if entry else Path.cwd()
        if (p / "nvidia").is_dir():
            roots.append(p / "nvidia")
        if (p / "sherpa_onnx" / "lib").is_dir():
            roots.append(p / "sherpa_onnx" / "lib")
    for root in roots:
        dirs = [root] if root.name == "lib" else sorted(d / "bin" for d in root.iterdir())
        for d in dirs:
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                added.append(str(d))
    return added


class SherpaRecognizer:
    """One sherpa-onnx ``OnlineRecognizer`` and its streams, by small integer
    id. Not thread-safe: the caller serialises (the in-process backend holds
    a lock; the child process is single-threaded)."""

    def __init__(
        self,
        *,
        tokens: str,
        encoder: str,
        decoder: str,
        joiner: str,
        provider: str = "cpu",
        device: int = 0,
        num_threads: int = 2,
        decoding: str = "greedy_search",
        tail_pad_ms: int = 480,
    ) -> None:
        import sherpa_onnx  # heavy; the only non-stdlib import besides numpy

        for f in (tokens, encoder, decoder, joiner):
            if not Path(f).is_file():
                raise FileNotFoundError(f)
        self._rec: Any = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=int(num_threads),
            sample_rate=SAMPLE_RATE,
            feature_dim=FEATURE_DIM,
            decoding_method=decoding,
            enable_endpoint_detection=False,
            # model_type "" = read it from the encoder's metadata (the export
            # writes model_type=EncDecHybridRNNTCTCBPEModel), as the upstream
            # example command lines do
            provider=provider,
            device=int(device),
        )
        self.provider = provider
        self.device = int(device)
        self.tail_pad = int(SAMPLE_RATE * tail_pad_ms / 1000)
        self.version = str(getattr(sherpa_onnx, "__version__", "?"))
        self._streams: dict[int, Any] = {}
        self._next = 0

    # --------------------------------------------------------------- streams
    def create(self) -> int:
        self._next += 1
        self._streams[self._next] = self._rec.create_stream()
        return self._next

    def _stream(self, sid: int) -> Any:
        try:
            return self._streams[sid]
        except KeyError:
            raise KeyError(f"unknown stream {sid}") from None

    def _drain(self, s: Any) -> None:
        while self._rec.is_ready(s):
            self._rec.decode_stream(s)

    def feed(self, sid: int, audio: np.ndarray) -> str:
        s = self._stream(sid)
        if len(audio):
            s.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
            self._drain(s)
        return (self._rec.get_result(s) or "").strip()

    def finalize(self, sid: int) -> tuple[str, list[tuple[str, float, float]]]:
        s = self._stream(sid)
        if self.tail_pad:
            s.accept_waveform(SAMPLE_RATE, np.zeros(self.tail_pad, np.float32))
        s.input_finished()
        self._drain(s)
        text = (self._rec.get_result(s) or "").strip()
        words = words_from_sp_tokens(list(self._rec.tokens(s)), list(self._rec.timestamps(s)))
        return text, words

    def reset(self, sid: int) -> None:
        """A fresh stream under the same id: timestamps restart at 0 and no
        text carries over (``recognizer.reset`` keeps a segment offset)."""
        self._stream(sid)
        self._streams[sid] = self._rec.create_stream()

    def close_stream(self, sid: int) -> None:
        self._streams.pop(sid, None)

    def describe(self) -> dict[str, Any]:
        return {
            "sherpa_onnx": self.version,
            "provider": self.provider,
            "device": self.device,
            "streams": len(self._streams),
        }

    def close(self) -> None:
        self._streams.clear()
        self._rec = None


# ------------------------------------------------------------------ the child
def _read_exact(src: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = src.read(n - len(buf))
        if not chunk:
            raise EOFError("short read")
        buf += chunk
    return bytes(buf)


def serve(
    src: BinaryIO,
    dst: BinaryIO,
    make: Any = None,
    *,
    log: Any = None,
) -> int:
    """The child's loop: read a command line, answer one line. ``make`` builds
    the recognizer from the ``load`` command's fields (the tests inject a fake);
    the default is :class:`SherpaRecognizer`. Returns the exit code."""
    make = make or (lambda **kw: SherpaRecognizer(**kw))
    log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
    rec: Any = None

    def reply(obj: dict[str, Any]) -> None:
        dst.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        dst.flush()

    while True:
        line = src.readline()
        if not line:
            return 0
        try:
            cmd = json.loads(line.decode("utf-8"))
            op = cmd.get("op")
        except (ValueError, AttributeError):
            reply({"error": "bad command line", "type": "ProtocolError"})
            return 2
        audio: np.ndarray | None = None
        if op == "feed":
            try:
                n = int(cmd.get("n", 0))
            except (TypeError, ValueError):
                n = -1
            if not 0 <= n <= MAX_FEED_SAMPLES:
                reply({"error": f"feed of {n} samples is out of bounds", "type": "ProtocolError"})
                return 2
            try:
                audio = np.frombuffer(_read_exact(src, 4 * n), dtype="<f4") if n else np.zeros(0)
            except EOFError:
                reply({"error": "short audio read", "type": "ProtocolError"})
                return 2
        try:
            if op == "load":
                kw = {k: v for k, v in cmd.items() if k != "op"}
                rec = make(**kw)
                reply({"ok": True, "describe": rec.describe()})
            elif op == "close":
                if rec is not None:
                    rec.close()
                reply({"ok": True})
                return 0
            elif rec is None:
                reply({"error": "not loaded", "type": "RuntimeError"})
            elif op == "create":
                reply({"sid": rec.create()})
            elif op == "feed":
                reply({"text": rec.feed(int(cmd["sid"]), audio)})
            elif op == "finalize":
                text, words = rec.finalize(int(cmd["sid"]))
                reply({"text": text, "words": [list(w) for w in words]})
            elif op == "reset":
                rec.reset(int(cmd["sid"]))
                reply({"ok": True})
            elif op == "close_stream":
                rec.close_stream(int(cmd["sid"]))
                reply({"ok": True})
            elif op == "describe":
                reply({"describe": rec.describe()})
            else:
                reply({"error": f"unknown op {op!r}", "type": "ProtocolError"})
        except Exception as exc:  # noqa: BLE001 - every failure goes back over the pipe
            log(f"nemotron worker: {op} failed: {type(exc).__name__}: {exc}")
            reply({"error": str(exc)[:500], "type": type(exc).__name__})


def main() -> int:
    added = add_cuda_dll_dirs()
    if added:
        print(f"nemotron worker: DLL dirs {added}", file=sys.stderr, flush=True)
    return serve(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    sys.exit(main())
