"""Fakes: the suite runs on any CPU with no ONNX Runtime and no models."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import httpx
import numpy as np
import pytest

from speechforge.config import Config
from speechforge.voices import VoiceTensors, parse_npz


def make_voice(seed: int = 0, cond: int = 20, prompt: int = 120) -> VoiceTensors:
    rng = np.random.default_rng(seed)
    return VoiceTensors(
        rng.standard_normal((1, cond, 1024)).astype(np.float32),
        rng.integers(0, 6561, (1, prompt)).astype(np.int64),
        rng.standard_normal((1, 192)).astype(np.float32),
        rng.standard_normal((1, prompt * 2, 80)).astype(np.float32),
    )


@pytest.fixture
def voice_bytes() -> bytes:
    return make_voice().to_npz()


class FakeTts:
    """Speaks ``chunks`` pieces of 960 samples (value i+1) per call, honouring
    ``stop``, the credit ``gate`` (per fake token: one piece = one token) and
    ``on_lm_done``. ``step_s`` paces each piece like a real decode step."""

    name = "fake"
    sample_rate = 24000
    voice_format = "chatterbox-npz"

    def __init__(
        self,
        *,
        chunks: int = 3,
        delay_s: float = 0.0,
        fail_at: int | None = None,
        step_s: float = 0.0,
    ):
        self.chunks, self.delay_s, self.fail_at = chunks, delay_s, fail_at
        self.step_s = step_s
        self.closed = False
        self.streams = 0
        self.texts: list[str] = []
        self.steps = 0
        self._last: dict[str, Any] = {}

    @property
    def last_stats(self) -> dict[str, Any]:
        return dict(self._last)

    def parse_voice(self, data: bytes) -> Any:
        return parse_npz(data)

    def build_voice(self, reference: np.ndarray, rate: int) -> bytes:
        return make_voice(int(len(reference)) % 7).to_npz()

    def stream(self, text, voice_id, voice, *, stop=None, seed=None, options=None):
        options = options or {}
        gate = options.get("gate")
        self.streams += 1
        self.texts.append(text)
        for i in range(self.chunks):
            if gate is not None:
                while not gate(i):
                    if stop is not None and stop.is_set():
                        return
                    time.sleep(0.002)
            if stop is not None and stop.is_set():
                return
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError("boom")
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.step_s:
                time.sleep(self.step_s)
            self.steps += 1
            yield np.full(960, i + 1, dtype="<i2")
        if options.get("on_lm_done") is not None:
            options["on_lm_done"](self.chunks)
        self._last = {"first_chunk_ms": 1, "chunks": self.chunks}

    def render(self, text, voice, *, seed=None):
        return np.full(2400, 7, dtype="<i2")

    def warm_up(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-tts"}

    def close(self) -> None:
        self.closed = True


class FakeStt:
    name = "fake"
    sample_rate = 16000

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        return f"heard {len(audio)} samples"

    def transcribe_words(self, audio: np.ndarray):
        self.calls += 1
        n = len(audio) / 16000
        return f"final {len(audio)} samples", [("final", 0.0, n / 2), ("samples", n / 2, n)]

    def warm_up(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-stt"}

    def close(self) -> None:
        self.closed = True


class FakeEou:
    def __init__(self, p: float = 0.93) -> None:
        self.p = p
        self.calls = 0

    def score(self, audio: np.ndarray) -> float:
        self.calls += 1
        return self.p

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-eou"}

    def close(self) -> None:
        pass


class FakeStudioForge:
    """The four lease routes, in memory, with switchable refusals."""

    def __init__(self) -> None:
        self.leases: dict[str, dict[str, Any]] = {}
        self.requests: list[tuple[str, str, Any]] = []
        self.refuse: tuple[int, str, int | None] | None = None  # status, code, retry_after
        self.down = False
        self._n = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        path = request.url.path
        if request.method == "GET" and path == "/api/leases":
            return httpx.Response(200, json={"leases": list(self.leases.values())})
        if request.method == "POST" and path == "/api/leases":
            if self.refuse:
                status, code, retry = self.refuse
                err: dict[str, Any] = {
                    "message": f"refused: {code}",
                    "code": code,
                    "type": "invalid_request_error",
                }
                headers = {}
                if retry:
                    err["studioforge"] = {"retry_after_s": retry}
                    headers["Retry-After"] = str(retry)
                return httpx.Response(status, json={"error": err}, headers=headers)
            for lease in self.leases.values():
                if set(lease["devices"]) & set(body["devices"]):
                    return httpx.Response(
                        409, json={"error": {"code": "lease_conflict", "message": "taken"}}
                    )
            self._n += 1
            lid = f"lease{self._n}"
            self.leases[lid] = {
                "id": lid,
                "devices": body["devices"],
                "holder": body["holder"],
                "priority": body["priority"],
                "idle_ttl_s": body["idle_ttl_s"],
                "state": "active",
            }
            return httpx.Response(200, json={**self.leases[lid], "vacate_url": body["vacate_url"]})
        if request.method == "DELETE" and path.startswith("/api/leases/"):
            lid = path.rsplit("/", 1)[1]
            if lid not in self.leases:
                return httpx.Response(404, json={"error": {"code": "lease_not_found"}})
            return httpx.Response(200, json={"released": self.leases.pop(lid)})
        if request.method == "POST" and path.endswith("/touch"):
            lid = path.split("/")[-2]
            if lid not in self.leases:
                return httpx.Response(404, json={"error": {"code": "lease_not_found"}})
            return httpx.Response(200, json=self.leases[lid])
        return httpx.Response(404)


@pytest.fixture
def studioforge() -> FakeStudioForge:
    return FakeStudioForge()


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        data_dir=tmp_path,
        gpu={"devices": [3, 1]},
        voices={"dir": tmp_path / "voices"},
        selftest={"enabled": False},
        realtime={"ready_wait_s": 5},
        residency={"idle_unload_s": 600, "drain_timeout_s": 1, "failure_cooldown_s": 60},
        studioforge={"refusal_cooldown_s": 30, "touch_interval_s": 30, "vacate_cooldown_s": 120},
    )
