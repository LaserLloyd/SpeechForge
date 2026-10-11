"""The adapter interfaces every model plugs in behind, and the registry.

The server (transport, residency, leases, voice store) never touches a model
API directly; it talks to one :class:`SttAdapter` and one :class:`TtsAdapter`,
picked by name (``stt.engine`` / ``tts.engine`` in config.yaml). Adding a
model is a module that implements one of these and registers a loader -- the
HTTP surface, the lease handling and the tests of both stay as they are.

Voices are opaque to everything but the TTS adapter: the store keeps bytes
keyed by their sha256 and asks the adapter to parse them (``parse_voice``) and,
for a reference clip, to make them (``build_voice``). A different TTS model
means a different voice format, not a different store.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SttAdapter(Protocol):
    name: str
    sample_rate: int  # what transcribe() takes (16 kHz for every current model)

    def transcribe(self, audio: np.ndarray) -> str: ...  # float32 mono -> text

    def warm_up(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@runtime_checkable
class SttStreamHandle(Protocol):
    """One open utterance on a cache-aware streaming model (one per session).

    ``feed`` takes float32 16 kHz mono frames in arrival order and returns the
    current partial text for everything fed since the last ``reset``;
    ``finalize`` flushes the model (tail padding, end of input) and returns
    the final text plus ``(word, start_s, end_s)`` timings relative to the
    first sample fed since the last ``reset`` -- ``[]`` when the model has
    none. After ``finalize`` the handle must be ``reset`` before more audio.
    """

    def feed(self, audio: np.ndarray) -> str: ...

    def finalize(self) -> tuple[str, list[tuple[str, float, float]]]: ...

    def reset(self) -> None: ...


@runtime_checkable
class StreamingSttAdapter(Protocol):
    """An :class:`SttAdapter` whose model decodes each frame once (cache-aware
    streaming). ``stt_stream`` feeds it at the ``partial_ms`` cadence instead
    of re-decoding the open tail; ``transcribe`` still works (one stream,
    fed whole, finalized) so every ``SttAdapter`` caller is unchanged."""

    name: str
    sample_rate: int

    def create_stream(self) -> SttStreamHandle: ...

    def transcribe(self, audio: np.ndarray) -> str: ...

    def warm_up(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def is_streaming_stt(adapter: Any) -> bool:
    """True when the loaded STT adapter streams (duck-typed: a fake with
    ``create_stream`` counts; a protocol isinstance check would also accept
    any object with the attribute, so this is the same test said plainly)."""
    return callable(getattr(adapter, "create_stream", None))


@runtime_checkable
class TtsAdapter(Protocol):
    name: str
    sample_rate: int  # what stream()/render() yield (int16 mono)
    voice_format: str  # e.g. "chatterbox-npz": what PUT /v1/voices accepts

    def parse_voice(self, data: bytes) -> Any: ...  # validated, ready object; 400 if bad

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes: ...  # clip -> voice bytes

    def stream(
        self,
        text: str,
        voice_id: str,
        voice: Any,
        *,
        stop: threading.Event | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Iterator[np.ndarray]: ...

    def render(self, text: str, voice: Any, *, seed: int | None = None) -> np.ndarray: ...

    def warm_up(self) -> None: ...

    def describe(self) -> dict[str, Any]: ...

    @property
    def last_stats(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


#: name -> loader(config, device_id) -> adapter. Loaders import their heavy
#: dependencies inside, so registering costs nothing.
STT_ENGINES: dict[str, Callable[[Any, int], SttAdapter]] = {}
TTS_ENGINES: dict[str, Callable[[Any, int], TtsAdapter]] = {}
#: name -> parse(bytes) -> voice object. Needs no model loaded, so a voice can
#: be uploaded (and validated) while the card is still being acquired.
VOICE_PARSERS: dict[str, Callable[[bytes], Any]] = {}


def register_stt(name: str, loader: Callable[[Any, int], SttAdapter]) -> None:
    STT_ENGINES[name] = loader


def register_tts(
    name: str, loader: Callable[[Any, int], TtsAdapter], parse_voice: Callable[[bytes], Any]
) -> None:
    TTS_ENGINES[name] = loader
    VOICE_PARSERS[name] = parse_voice


def builtin() -> None:
    """Register the engines that ship with SpeechForge."""
    from speechforge.engines import chatterbox_tts, dots_tts, nemotron, parakeet, qwen3_tts
    from speechforge.voices import parse_npz

    register_stt("parakeet", parakeet.load)
    register_stt("nemotron", nemotron.load)
    register_tts("chatterbox", chatterbox_tts.load, parse_npz)
    register_tts("qwen3-tts", qwen3_tts.load, qwen3_tts.parse_voice)
    register_tts("dots-tts", dots_tts.load, dots_tts.parse_voice)
