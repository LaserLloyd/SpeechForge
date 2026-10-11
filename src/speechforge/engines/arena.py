"""One ONNX Runtime GPU arena for every CUDA session on the card (DECISIONS S24).

Each ``InferenceSession`` on the CUDA provider normally owns a BFC arena that
starts empty and grows as the model runs, so a cold load pays the allocation
ramp six times over (four Chatterbox sessions, Parakeet, Smart Turn) and the
warm-up spends its time growing arenas to their working set. ONNX Runtime can
instead hold one arena in its environment and hand it to every session that
asks for it:

* ``create_and_register_allocator_v2("CUDAExecutionProvider", mem_info,
  provider_options, arena_cfg)`` creates the arena once per device and
  registers it with the process-wide environment. The plain
  ``create_and_register_allocator`` refuses CUDA memory ("Only CPU devices
  are supported. Please call CreateAndRegisterAllocatorV2() for other
  device", checked against the 1.31 wheel), so v2 is the only Python path.
* ``SessionOptions.add_session_config_entry("session.use_env_allocators",
  "1")`` on every session makes it use the registered arena instead of its own.

Both are process-wide and there is no Python API to unregister, so the
registration is done once per device per process (:func:`ensure_shared_arena`
is idempotent) and a second load after an unload reuses it. A failed
registration logs a warning and the sessions keep their per-session arenas
(the S22 configuration), which is also what every session keeps in its
provider options as the floor.

Sizes. ``OrtArenaCfg.initial_chunk_size_bytes`` is a C ``int``: 2047 MiB is
the largest first chunk the wheel accepts (2048 MiB fails at construction), so
``gpu.arena_initial_mb`` is bounded there. ``max_mem`` is one cap for everything
on the card: ``gpu.mem_limit_mb`` is a per-session cap that no single session
approached, so the shared cap is twice it (the two model groups, TTS and
STT + EOU, could each have reached one), 12 GiB at the default, well inside a
24 GB card and still a bound on a runaway. Extension stays power-of-two (S22).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("speechforge.arena")

MIB = 1024 * 1024
#: ``OrtArenaCfg.initial_chunk_size_bytes`` is a 32-bit int in the C API.
ARENA_INITIAL_MB_MAX = 2047
#: The shared ``max_mem`` is this many times ``gpu.mem_limit_mb`` (module docstring).
MAX_MEM_FACTOR = 2
#: ``ArenaExtendStrategy::kNextPowerOfTwo`` (S22); ``0`` would be kSameAsRequested.
K_NEXT_POWER_OF_TWO = 1
USE_ENV_ALLOCATORS = "session.use_env_allocators"
PROVIDER = "CUDAExecutionProvider"
#: ``OrtMemoryInfo`` name the runtime maps to a CUDA device allocation.
MEM_NAME = "Cuda"

_LOCK = threading.Lock()
#: device ordinal -> facts about the arena registered for it in this process.
_REGISTERED: dict[int, dict[str, Any]] = {}


def shared_max_mem_bytes(mem_limit_mb: int) -> int:
    """The shared arena's cap: twice the per-session ``gpu.mem_limit_mb``."""
    return int(mem_limit_mb) * MAX_MEM_FACTOR * MIB


def arena_cfg_dict(initial_mb: int, mem_limit_mb: int) -> dict[str, int]:
    """The ``OrtArenaCfg`` as its dict form (``-1`` = the runtime's default:
    128 MiB dead bytes per chunk, 2 MiB first growth chunk, 1 GiB largest
    power-of-two extension)."""
    initial = min(int(initial_mb), ARENA_INITIAL_MB_MAX)
    return {
        "max_mem": shared_max_mem_bytes(mem_limit_mb),
        "arena_extend_strategy": K_NEXT_POWER_OF_TWO,
        "initial_chunk_size_bytes": initial * MIB,
        "max_dead_bytes_per_chunk": -1,
        "initial_growth_chunk_size_bytes": -1,
        "max_power_of_two_extend_bytes": -1,
    }


def ensure_shared_arena(gpu: Any, device_id: int) -> bool:
    """Register the shared arena for ``device_id`` once per process; True when
    the sessions created afterwards should use it.

    False, with one warning, when ``gpu.provider`` is not ``cuda``, when
    ``gpu.shared_arena`` is off, or when the runtime refuses (an old wheel, a
    CPU-only wheel, a CUDA error): the caller then creates per-session arenas.
    A registration the runtime reports as already present counts as ours.
    """
    if getattr(gpu, "provider", "cuda") != "cuda" or not getattr(gpu, "shared_arena", False):
        return False
    device = int(device_id)
    cfg = arena_cfg_dict(gpu.arena_initial_mb, gpu.mem_limit_mb)
    with _LOCK:
        if device in _REGISTERED:
            return True
        try:
            import onnxruntime as ort

            register = getattr(ort, "create_and_register_allocator_v2", None)
            if register is None:
                raise AttributeError(
                    "onnxruntime has no create_and_register_allocator_v2 "
                    "(the CUDA path; create_and_register_allocator is CPU-only)"
                )
            mem_info = ort.OrtMemoryInfo(
                MEM_NAME, ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, device, ort.OrtMemType.DEFAULT
            )
            arena_cfg = ort.OrtArenaCfg(cfg)
            try:
                register(PROVIDER, mem_info, {"device_id": str(device)}, arena_cfg)
            except Exception as exc:  # noqa: BLE001
                if "already" not in str(exc).lower():
                    raise
                log.info("shared GPU arena on CUDA %d was already registered; using it", device)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "shared GPU arena on CUDA %d not registered (%s: %s); "
                "sessions keep their per-session arenas",
                device,
                type(exc).__name__,
                str(exc)[:200],
            )
            return False
        _REGISTERED[device] = {
            "initial_mb": cfg["initial_chunk_size_bytes"] // MIB,
            "max_mem_mb": cfg["max_mem"] // MIB,
        }
        log.info(
            "shared GPU arena registered on CUDA %d: first chunk %d MiB, cap %d MiB, "
            "power-of-two extension",
            device,
            _REGISTERED[device]["initial_mb"],
            _REGISTERED[device]["max_mem_mb"],
        )
        return True


def in_use(device_id: int | None) -> bool:
    """True when this process registered a shared arena for ``device_id``."""
    return device_id is not None and int(device_id) in _REGISTERED


def facts(device_id: int | None) -> dict[str, Any]:
    """Sizes of the registered arena (empty when there is none)."""
    if device_id is None:
        return {}
    return dict(_REGISTERED.get(int(device_id), {}))


def configure_session_options(session_options: Any, shared: bool) -> None:
    """Make ``session_options`` use the environment's allocators when ``shared``
    (a no-op otherwise, so every engine can call it unconditionally)."""
    if shared:
        session_options.add_session_config_entry(USE_ENV_ALLOCATORS, "1")
