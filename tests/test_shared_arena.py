"""The shared GPU arena (DECISIONS S24) against a fake ``onnxruntime``.

The suite has no ONNX Runtime; the fake records what the engines would have
asked of it: one allocator registration with the configured sizes, and the
``session.use_env_allocators`` entry on every session's options.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from speechforge.config import GpuConfig
from speechforge.engines import arena
from speechforge.pipeline import Speaker
from speechforge.residency import Residency
from tests.conftest import FakeStt, FakeTts

MIB = 1024 * 1024


class _Enum:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class FakeSessionOptions:
    def __init__(self) -> None:
        self.entries: dict[str, str] = {}
        self.graph_optimization_level = None
        self.log_severity_level = None
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


class _IO:
    def __init__(self, name: str, type_: str = "tensor(float)") -> None:
        self.name, self.type = name, type_


class FakeSession:
    def __init__(self, path: str, so: FakeSessionOptions, providers: list[Any]) -> None:
        self.path, self.so, self.providers = str(path), so, providers

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def get_inputs(self) -> list[_IO]:
        if "language_model" in self.path:
            return [
                _IO("inputs_embeds"),
                _IO("attention_mask", "tensor(int64)"),
                _IO("position_ids", "tensor(int64)"),
                _IO("past_key_values.0.key"),
                _IO("past_key_values.0.value"),
            ]
        return [_IO("x")]

    def get_outputs(self) -> list[_IO]:
        if "language_model" in self.path:
            return [_IO("logits"), _IO("present.0.key"), _IO("present.0.value")]
        return [_IO("y")]


def fake_ort(*, register_error: Exception | None = None, with_v2: bool = True) -> Any:
    mod = types.ModuleType("onnxruntime")
    mod.registrations = []  # type: ignore[attr-defined]
    mod.sessions = []  # type: ignore[attr-defined]

    class OrtAllocatorType:
        ORT_ARENA_ALLOCATOR = _Enum("ORT_ARENA_ALLOCATOR")
        ORT_DEVICE_ALLOCATOR = _Enum("ORT_DEVICE_ALLOCATOR")

    class OrtMemType:
        DEFAULT = _Enum("DEFAULT")
        CPU = _Enum("CPU")

    class OrtMemoryInfo:
        def __init__(self, name: str, alloc: Any, device_id: int, mem_type: Any) -> None:
            self.args = (name, alloc, int(device_id), mem_type)

    class OrtArenaCfg:
        def __init__(self, cfg: dict[str, int]) -> None:
            if cfg["initial_chunk_size_bytes"] > 2**31 - 1:
                raise RuntimeError("Unable to cast Python instance of type int to C++ type")
            self.cfg = dict(cfg)

    def create_and_register_allocator_v2(
        provider: str, mem_info: Any, options: Any, cfg: Any
    ) -> None:
        if register_error is not None:
            raise register_error
        mod.registrations.append((provider, mem_info.args, dict(options), cfg.cfg))  # type: ignore[attr-defined]

    def create_and_register_allocator(mem_info: Any, cfg: Any) -> None:
        raise RuntimeError(
            "Error when creating and registering allocator: Only CPU devices are supported. "
            "Please call CreateAndRegisterAllocatorV2() for other device."
        )

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = 99

    def InferenceSession(path: str, so: Any, providers: list[Any]) -> FakeSession:  # noqa: N802
        sess = FakeSession(path, so, providers)
        mod.sessions.append(sess)  # type: ignore[attr-defined]
        return sess

    mod.OrtAllocatorType = OrtAllocatorType  # type: ignore[attr-defined]
    mod.OrtMemType = OrtMemType  # type: ignore[attr-defined]
    mod.OrtMemoryInfo = OrtMemoryInfo  # type: ignore[attr-defined]
    mod.OrtArenaCfg = OrtArenaCfg  # type: ignore[attr-defined]
    mod.create_and_register_allocator = create_and_register_allocator  # type: ignore[attr-defined]
    if with_v2:
        mod.create_and_register_allocator_v2 = create_and_register_allocator_v2  # type: ignore[attr-defined]
    mod.SessionOptions = FakeSessionOptions  # type: ignore[attr-defined]
    mod.GraphOptimizationLevel = GraphOptimizationLevel  # type: ignore[attr-defined]
    mod.InferenceSession = InferenceSession  # type: ignore[attr-defined]
    mod.get_available_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]  # type: ignore[attr-defined]
    return mod


@pytest.fixture
def ort(monkeypatch):
    """A fresh fake runtime and an empty registry for every test."""
    mod = fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    monkeypatch.setattr(arena, "_REGISTERED", {})
    return mod


# ------------------------------------------------------------------ config
def test_gpu_config_defaults_and_bounds():
    gpu = GpuConfig()
    assert gpu.shared_arena is False and gpu.arena_initial_mb == arena.ARENA_INITIAL_MB_MAX
    assert GpuConfig(arena_initial_mb=64).arena_initial_mb == 64
    with pytest.raises(ValidationError):
        GpuConfig(arena_initial_mb=2048)  # the runtime's int32 field refuses 2 GiB
    with pytest.raises(ValidationError):
        GpuConfig(arena_initial_mb=4096)
    with pytest.raises(ValidationError):
        GpuConfig(arena_initial_mb=0)


def test_arena_cfg_sizes_follow_the_config():
    cfg = arena.arena_cfg_dict(1024, 6144)
    assert cfg["initial_chunk_size_bytes"] == 1024 * MIB
    assert cfg["max_mem"] == 2 * 6144 * MIB
    assert cfg["arena_extend_strategy"] == 1  # kNextPowerOfTwo (S22)
    assert arena.arena_cfg_dict(99999, 512)["initial_chunk_size_bytes"] == 2047 * MIB


# ------------------------------------------------------------ registration
def test_registers_once_per_device_with_the_configured_sizes(ort):
    gpu = GpuConfig(devices=[3], mem_limit_mb=6144, arena_initial_mb=2047, shared_arena=True)
    assert arena.ensure_shared_arena(gpu, 3) is True
    assert arena.ensure_shared_arena(gpu, 3) is True  # idempotent: a reload after an unload
    assert len(ort.registrations) == 1
    provider, mem_info, options, cfg = ort.registrations[0]
    assert provider == "CUDAExecutionProvider"
    assert mem_info == ("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 3, ort.OrtMemType.DEFAULT)
    assert options == {"device_id": "3"}
    assert cfg["initial_chunk_size_bytes"] == 2047 * MIB
    assert cfg["max_mem"] == 2 * 6144 * MIB
    assert cfg["arena_extend_strategy"] == 1
    assert arena.in_use(3) and not arena.in_use(1) and not arena.in_use(None)
    assert arena.facts(3) == {"initial_mb": 2047, "max_mem_mb": 12288}
    # Another card is another arena.
    assert arena.ensure_shared_arena(gpu, 1) is True
    assert [r[1][2] for r in ort.registrations] == [3, 1]


def test_off_or_cpu_registers_nothing(ort):
    assert arena.ensure_shared_arena(GpuConfig(shared_arena=False), 3) is False
    assert arena.ensure_shared_arena(GpuConfig(provider="cpu"), 3) is False
    assert ort.registrations == [] and not arena.in_use(3)


def test_a_refused_registration_falls_back_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(arena, "_REGISTERED", {})
    broken = fake_ort(register_error=RuntimeError("CUDA failure: device busy"))
    monkeypatch.setitem(sys.modules, "onnxruntime", broken)
    gpu = GpuConfig(devices=[3], shared_arena=True)
    with caplog.at_level(logging.WARNING, logger="speechforge.arena"):
        assert arena.ensure_shared_arena(gpu, 3) is False
    assert not arena.in_use(3)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "per-session arenas" in warnings[0].getMessage()
    assert "CUDA failure" in warnings[0].getMessage()
    # The next load tries again (and succeeds once the runtime does).
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort())
    assert arena.ensure_shared_arena(gpu, 3) is True


def test_a_wheel_without_v2_falls_back_and_names_the_api(monkeypatch, caplog):
    monkeypatch.setattr(arena, "_REGISTERED", {})
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort(with_v2=False))
    with caplog.at_level(logging.WARNING, logger="speechforge.arena"):
        assert arena.ensure_shared_arena(GpuConfig(shared_arena=True), 3) is False
    assert "create_and_register_allocator_v2" in caplog.text


def test_an_already_registered_arena_counts_as_ours(monkeypatch):
    monkeypatch.setattr(arena, "_REGISTERED", {})
    dup = RuntimeError("An allocator for this device has already been registered for sharing.")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort(register_error=dup))
    assert arena.ensure_shared_arena(GpuConfig(shared_arena=True), 3) is True
    assert arena.in_use(3)


# -------------------------------------------------------- session options
def test_configure_session_options_sets_the_entry_only_when_shared():
    so = FakeSessionOptions()
    arena.configure_session_options(so, False)
    assert so.entries == {}
    arena.configure_session_options(so, True)
    assert so.entries == {"session.use_env_allocators": "1"}


def _chatterbox_dir(tmp_path: Path, dtype: str) -> Path:
    from speechforge.engines.chatterbox import model_files

    root = tmp_path / "chatterbox-turbo"
    for rel in model_files(dtype):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


@pytest.fixture
def tokenizers(monkeypatch):
    mod = types.ModuleType("tokenizers")

    class Tokenizer:
        @staticmethod
        def from_file(path: str) -> Any:
            return object()

    mod.Tokenizer = Tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", mod)
    return mod


@pytest.mark.parametrize("shared", [True, False])
def test_chatterbox_marks_all_four_sessions(ort, tokenizers, tmp_path, shared):
    from speechforge.engines.chatterbox import COMPONENTS, ChatterboxEngine

    eng = ChatterboxEngine(_chatterbox_dir(tmp_path, "fp16"), dtype="fp16", shared_arena=shared)
    assert eng.shared_arena is shared
    assert len(ort.sessions) == len(COMPONENTS) == 4
    want = {"session.use_env_allocators": "1"} if shared else {}
    assert all(s.so.entries == want for s in ort.sessions)
    # The per-session CUDA options stay as the floor either way (S22).
    for s in ort.sessions:
        name, opts = s.providers[0]
        assert name == "CUDAExecutionProvider"
        assert opts["arena_extend_strategy"] == "kNextPowerOfTwo" and "gpu_mem_limit" in opts


def test_chatterbox_on_cpu_never_asks_for_env_allocators(ort, tokenizers, tmp_path):
    from speechforge.engines.chatterbox import ChatterboxEngine

    eng = ChatterboxEngine(_chatterbox_dir(tmp_path, "fp32"), provider="cpu", shared_arena=True)
    assert eng.shared_arena is False
    assert all(s.so.entries == {} for s in ort.sessions)


@pytest.mark.parametrize("shared", [True, False])
def test_parakeet_passes_the_entry_to_onnx_asr(ort, monkeypatch, tmp_path, shared):
    from speechforge.engines import parakeet

    seen: dict[str, Any] = {}
    onnx_asr = types.ModuleType("onnx_asr")

    class Model:
        def with_timestamps(self) -> Any:
            return self

    def load_model(name: str, path: str, *, quantization: Any, sess_options: Any, providers: Any):
        seen.update(sess_options=sess_options, providers=providers, name=name)
        return Model()

    onnx_asr.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", onnx_asr)
    root = tmp_path / "parakeet-tdt-0.6b-v2"
    root.mkdir()
    for rel in parakeet.model_files(None):
        (root / rel).write_bytes(b"")
    eng = parakeet.ParakeetEngine(root, device_id=3, shared_arena=shared)
    assert eng.shared_arena is shared
    want = {"session.use_env_allocators": "1"} if shared else {}
    assert seen["sess_options"].entries == want
    assert seen["providers"][0][0] == "CUDAExecutionProvider"


@pytest.mark.parametrize("shared", [True, False])
def test_smart_turn_session_options(ort, tmp_path, shared):
    from speechforge.engines.turn import SmartTurn

    path = tmp_path / "smart-turn-v3.2-gpu.onnx"
    path.write_bytes(b"")
    eou = SmartTurn(path, device_id=3, shared_arena=shared)
    assert eou.shared_arena is shared
    want = {"session.use_env_allocators": "1"} if shared else {}
    assert ort.sessions[-1].so.entries == want


def test_loaders_follow_the_registry(ort, tokenizers, monkeypatch, tmp_path, config):
    """``load_speaker`` registers first; the registry is what the loaders read."""
    from speechforge.engines import chatterbox_tts

    config.models.dir = tmp_path
    config.gpu.shared_arena = True
    _chatterbox_dir(tmp_path, "fp32")
    tts = chatterbox_tts.load(config, 3)
    assert tts.engine.shared_arena is False  # nothing registered for CUDA 3 yet
    assert arena.ensure_shared_arena(config.gpu, 3) is True
    tts = chatterbox_tts.load(config, 3)
    assert tts.engine.shared_arena is True
    assert all(s.so.entries == {"session.use_env_allocators": "1"} for s in ort.sessions[4:])


# --------------------------------------------------------------- /ready
async def test_ready_reports_the_shared_arena(config, clock):
    def factory(device: int) -> Speaker:
        return Speaker(FakeTts(), FakeStt(), shared_arena=True)

    res = Residency(
        config,
        factory=factory,
        lease=None,
        clock=clock,
        facts=lambda d: {"name": "Fake 3090", "sm": "sm_86"},
        live=lambda d: {},
    )
    assert res.status()["gpu"] == {"shared_arena": False}
    res.kick()
    for _ in range(200):
        if res._bringup is not None and res._bringup.done():
            break
        import asyncio

        await asyncio.sleep(0.01)
    assert res.state == "ready"
    assert res.status()["gpu"] == {"name": "Fake 3090", "sm": "sm_86", "shared_arena": True}
