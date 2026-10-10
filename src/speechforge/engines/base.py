"""The two adapter interfaces every model plugs in behind, and the registry.

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
    from speechforge.engines import chatterbox_tts, parakeet
    from speechforge.voices import parse_npz

    register_stt("parakeet", parakeet.load)
    register_tts("chatterbox", chatterbox_tts.load, parse_npz)
