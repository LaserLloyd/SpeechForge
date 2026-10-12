"""When the models are on the card, and on whose say-so (SPEC-speechforge §7.3, §7.4).

States::

    unloaded -> acquiring (lease) -> loading (sessions, warm-up, self-test) -> ready
    ready -> draining (idle | vacate | lease lost | stop) -> unloaded | cooldown
    acquiring/loading -> cooldown (refused | load failed | self-test failed) -> unloaded

Promises:

* **Never wait for the GPU side.** HTTP utilities answer 503 + Retry-After at
  once when not ready and kick the bring-up; an sf.1 session waits (up to
  ``realtime.ready_wait_s``) while the box runs on its CPU.
* **Leases** (S3): class 2, vacate URL, never ``force``; the card from
  ``gpu.devices`` in order. StudioForge unreachable: load unleased only when
  NVML shows ``unreachable_min_free_gib`` free on the card and no llama-server
  holds memory there (``proceed_if_free``), else refuse.
* **Ready means tested.** Load -> warm-up -> self-test (golden STT clip, TTS
  TTFA, cancel stop, TTS->STT round trip); a failed self-test unloads and
  releases.
* **Sessions keep it resident.** No idle unload while a session is open; the
  idle clock starts when the last one closes. The lease is touched while
  sessions are open or requests ran.
* **Unleased is not exempt** (S29): a server that loaded without a lease because
  StudioForge was unreachable retries the lease every ``studioforge.relink_s``.
  Granted -> it holds one from then on (``held``). Refused with
  ``lease_conflict`` -> the card is committed to someone else, so it gives way
  exactly like a vacate would; any other refusal is transient and it asks again.
* **Vacate** (D56): the session manager announces ``state: vacating``, lets
  current contexts finish (bounded), then unload, release, cool down.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from speechforge.errors import Forbidden, Unavailable
from speechforge.lease import Granted, LeaseClient, Refused, Unreachable
from speechforge.metrics import METRICS
from speechforge.pipeline import Speaker

log = logging.getLogger("speechforge.residency")

LOADING_RETRY_S = 3.0
TICK_S = 5.0


def nvml_free_check(device: int, min_free_gib: float) -> tuple[bool, str]:
    """(ok, reason): enough free VRAM and no llama-server on ``device``."""
    try:
        import pynvml
    except ImportError:
        return False, "NVML unavailable (nvidia-ml-py not installed)"
    try:
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(int(device))
            free = pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**30
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                try:
                    name = pynvml.nvmlSystemGetProcessName(proc.pid)
                    name = name.decode() if isinstance(name, bytes) else str(name)
                except Exception:  # noqa: BLE001
                    name = ""
                if "llama-server" in name.lower():
                    return False, "a llama-server process holds memory on the card"
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        return False, f"NVML failed ({type(exc).__name__})"
    if free < min_free_gib:
        return False, f"only {free:.1f} GiB free"
    return True, f"{free:.1f} GiB free"


def nvml_facts(device: int | None) -> dict[str, Any]:
    if device is None:
        return {}
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(int(device))
            name = pynvml.nvmlDeviceGetName(h)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
            return {
                "name": name.decode() if isinstance(name, bytes) else str(name),
                "sm": f"sm_{major}{minor}",
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return {}


def nvml_live(device: int | None) -> dict[str, Any]:
    if device is None:
        return {}
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(int(device))
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out = {
                "gpu_util": int(util),
                "vram_free_mib": int(mem.free // 2**20),
                "vram_used_mib": int(mem.used // 2**20),
            }
            with contextlib.suppress(Exception):
                out["gpu_temp_c"] = int(pynvml.nvmlDeviceGetTemperature(h, 0))
            with contextlib.suppress(Exception):
                out["gpu_power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 1)
            return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001
        return {}


class Residency:
    def __init__(
        self,
        config: Any,
        *,
        factory: Callable[[int], Speaker],
        lease: LeaseClient | None,
        clock: Callable[[], float] = time.monotonic,
        selftest: Callable[[Speaker], dict[str, Any]] | None = None,
        free_check: Callable[[int, float], tuple[bool, str]] = nvml_free_check,
        facts: Callable[[int | None], dict[str, Any]] = nvml_facts,
        live: Callable[[int | None], dict[str, Any]] = nvml_live,
    ) -> None:
        self.config = config
        self._factory = factory
        self.lease_client = lease
        self._clock = clock
        self._selftest = selftest
        self._free_check = free_check
        self._facts_fn, self._live_fn = facts, live
        self._facts: dict[str, Any] = {}
        self.state = "unloaded"
        self.speaker: Speaker | None = None
        self.device: int | None = None
        self.lease: Granted | None = None
        self.unleased_reason: str | None = None
        self._cooldown_until = 0.0
        self._cooldown_code = ""
        self._cooldown_message = ""
        self.last_error: str | None = None
        self.last_refusals: list[dict[str, Any]] = []
        self.warmup: dict[str, Any] = {}
        self.selftest_result: dict[str, Any] = {}
        self.load_ms: int | None = None
        self.vram_at_load: int | None = None
        self._inflight = 0
        self._sessions = 0
        self._stops: set[threading.Event] = set()
        self._last_used = clock()
        self._last_touch = clock()
        self._last_relink = clock()
        self._relink_last_code: str | None = None
        self._active_since_touch = False
        self._bringup: asyncio.Task[None] | None = None
        self._unload_task: asyncio.Task[None] | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._vacate_pending = False
        self._ready_event: asyncio.Event | None = None
        self._listeners: list[Any] = []
        self.counters = {
            "loads": 0,
            "unloads": 0,
            "refused": 0,
            "vacates": 0,
            "load_failures": 0,
            "selftest_failures": 0,
        }

    # ------------------------------------------------------------------ hooks
    def add_listener(self, listener: Any) -> None:
        """An object with async ``on_lease_state``, ``before_unload``, ``after_unload``."""
        self._listeners.append(listener)

    async def _notify(self, method: str, *args: Any) -> None:
        for listener in self._listeners:
            fn = getattr(listener, method, None)
            if fn is not None:
                try:
                    await fn(*args)
                except Exception:  # noqa: BLE001
                    log.exception("listener %s failed", method)

    def session_opened(self) -> None:
        self._sessions += 1
        self._last_used = self._clock()
        self._active_since_touch = True

    def session_closed(self) -> None:
        self._sessions = max(0, self._sessions - 1)
        self._last_used = self._clock()

    def gpu_facts(self) -> dict[str, Any]:
        return dict(self._facts)

    def shared_arena(self) -> bool:
        """Whether the loaded sessions share one arena (S24); false until a
        speaker is loaded with one."""
        return bool(self.speaker is not None and getattr(self.speaker, "shared_arena", False))

    def gpu_live(self) -> dict[str, Any]:
        return self._live_fn(self.device) if self.device is not None else {}

    # ------------------------------------------------------------------ public
    def require(self) -> Speaker:
        """The loaded speaker, or an immediate 503 (and a bring-up kicked)."""
        now = self._clock()
        if self.state == "ready" and self.speaker is not None:
            return self.speaker
        if self.state == "draining":
            raise Unavailable(
                "the speech models are being unloaded",
                code="unloading",
                retry_after_s=max(LOADING_RETRY_S, self._cooldown_left(now)),
            )
        if self.state == "cooldown":
            if now < self._cooldown_until:
                raise Unavailable(
                    self._cooldown_message or "the GPU is not available",
                    code=self._cooldown_code or "gpu_unavailable",
                    retry_after_s=self._cooldown_left(now),
                )
            self.state = "unloaded"
        if self.state == "unloaded":
            self.kick()
        raise Unavailable(
            "the speech models are loading; use a local engine meanwhile",
            code="not_ready",
            retry_after_s=LOADING_RETRY_S,
        )

    def refusal(self) -> Unavailable | None:
        """What a new session should be told before upgrading, or None to proceed."""
        now = self._clock()
        if self.state == "cooldown" and now < self._cooldown_until:
            return Unavailable(
                self._cooldown_message or "the GPU is not available",
                code=self._cooldown_code or "gpu_unavailable",
                retry_after_s=self._cooldown_left(now),
            )
        if self.state == "draining":
            return Unavailable("unloading", code="unloading", retry_after_s=LOADING_RETRY_S)
        return None

    async def wait_ready(self, timeout_s: float) -> bool:
        if self.state == "ready":
            return True
        self.kick()
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            if self.state == "ready":
                return True
            if self.state in ("cooldown", "draining"):
                return False
            await asyncio.sleep(0.05)
        return self.state == "ready"

    def kick(self) -> bool:
        now = self._clock()
        if self.state == "cooldown" and now >= self._cooldown_until:
            self.state = "unloaded"
        if self.state != "unloaded":
            return False
        if self._bringup is not None and not self._bringup.done():
            return False
        self.state = "acquiring"
        self._bringup = asyncio.get_running_loop().create_task(self._bring_up())
        return True

    @contextlib.asynccontextmanager
    async def use(self) -> AsyncIterator[threading.Event]:
        stop = threading.Event()
        self._inflight += 1
        self._stops.add(stop)
        self._last_used = self._clock()
        self._active_since_touch = True
        try:
            yield stop
        finally:
            self._inflight -= 1
            self._stops.discard(stop)
            self._last_used = self._clock()

    async def start(self) -> None:
        if self.lease_client is not None:
            try:
                n = await self.lease_client.release_stale()
                if n:
                    log.info("released %d stale lease(s) from a previous run", n)
            except Exception:  # noqa: BLE001
                log.warning("could not check for stale leases", exc_info=True)
        if self.config.residency.warm_on_boot:
            self.kick()
        self._loop_task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        if self._bringup is not None and not self._bringup.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._bringup), 60)
        await self._unload("shutdown", cooldown_s=0)
        if self.lease_client is not None:
            await self.lease_client.aclose()

    async def vacate(self, lease_id: str, token: str | None) -> dict[str, Any]:
        client = self.lease_client
        if client is None or not client.token_matches(token):
            raise Forbidden("vacate token does not match", code="bad_vacate_token")
        self.counters["vacates"] += 1
        METRICS.inc("lease_events_total", event="vacate")
        held = self.lease.lease_id if self.lease is not None else None
        if held != lease_id:
            if lease_id:
                asyncio.get_running_loop().create_task(client.release(lease_id))
            return {"accepted": True, "state": self.state, "held": False}
        cooldown = self.config.studioforge.vacate_cooldown_s
        if self.state in ("acquiring", "loading"):
            self._vacate_pending = True
        else:
            self._schedule_unload(
                "vacated",
                cooldown_s=cooldown,
                code="gpu_vacated",
                message="StudioForge asked for the card back; use a local engine",
                announce=True,
            )
        log.info("vacate requested; draining", extra={"state": self.state})
        return {"accepted": True, "state": self.state, "held": True}

    async def tick(self) -> None:
        await self._notify("on_tick")  # the session manager expires detached sessions
        now = self._clock()
        if self.state != "ready":
            return
        idle_s = self.config.residency.idle_unload_s
        if (
            idle_s > 0
            and self._inflight == 0
            and self._sessions == 0
            and now - self._last_used >= idle_s
        ):
            log.info("idle for %.0f s; unloading and releasing the card", now - self._last_used)
            await self._unload("idle", cooldown_s=0)
            return
        if await self._maybe_relink(now):
            return
        await self._maybe_touch(now)
        live = self.gpu_live()
        if "vram_used_mib" in live:
            METRICS.set("vram_used_mib", live["vram_used_mib"])
            if self.vram_at_load is not None:
                METRICS.set("vram_drift_mib", live["vram_used_mib"] - self.vram_at_load)
            for key in ("gpu_util", "gpu_temp_c", "gpu_power_w"):
                if key in live:
                    METRICS.set(key, live[key])

    def status(self) -> dict[str, Any]:
        now = self._clock()
        idle_s = self.config.residency.idle_unload_s
        out: dict[str, Any] = {
            "state": self.state,
            "ready": self.state == "ready",
            "device": self.device,
            # The sf.1 ready frame carries gpu_facts() alone; the shared-arena flag
            # is for /ready and the harness.
            "gpu": {**self.gpu_facts(), "shared_arena": self.shared_arena()},
            "sessions": self._sessions,
            "inflight": self._inflight,
            "idle_s": round(now - self._last_used, 1),
            "idle_unload_in_s": (
                round(max(0.0, idle_s - (now - self._last_used)), 1)
                if idle_s > 0 and self.state == "ready" and not self._sessions
                else None
            ),
            "lease": dict(self.lease.view, id=self.lease.lease_id) if self.lease else None,
            "lease_mode": self.config.studioforge.lease_mode,
            "unleased_reason": self.unleased_reason,
            "counters": dict(self.counters),
            "last_error": self.last_error,
            "load_ms": self.load_ms,
            "warmup": dict(self.warmup),
            "selftest": dict(self.selftest_result),
        }
        if self.state == "cooldown":
            out["cooldown"] = {
                "code": self._cooldown_code,
                "message": self._cooldown_message,
                "retry_after_s": round(self._cooldown_left(now), 1),
                "refusals": list(self.last_refusals),
            }
        if self.speaker is not None:
            out["engines"] = dict(self.speaker.describe)
        return out

    # ------------------------------------------------------------------ internals
    def _cooldown_left(self, now: float) -> float:
        return max(1.0, self._cooldown_until - now)

    def _enter_cooldown(self, code: str, seconds: float, message: str) -> None:
        self.state = "cooldown"
        self._cooldown_code = code
        self._cooldown_message = message
        self._cooldown_until = self._clock() + max(0.0, seconds)

    async def _bring_up(self) -> None:
        cfg = self.config
        devices = list(cfg.gpu.devices) or [0]
        device: int | None = None
        lease: Granted | None = None
        self.unleased_reason = None
        self.last_refusals = []
        t_start = time.perf_counter()
        if self.lease_client is None:
            device = devices[0]
            self.unleased_reason = "lease_mode off"
        else:
            refusals: list[Refused] = []
            for dev in devices:
                outcome = await self.lease_client.acquire(dev)
                if isinstance(outcome, Granted):
                    device, lease = dev, outcome
                    METRICS.inc("lease_events_total", event="grant")
                    break
                if isinstance(outcome, Refused):
                    refusals.append(outcome)
                    METRICS.inc("lease_events_total", event="refuse")
                    continue
                assert isinstance(outcome, Unreachable)
                if cfg.studioforge.on_unreachable == "proceed_if_free":
                    ok, why = await asyncio.to_thread(
                        self._free_check, dev, cfg.studioforge.unreachable_min_free_gib
                    )
                    if ok:
                        device = dev
                        self.unleased_reason = f"StudioForge unreachable ({outcome.reason}); {why}"
                        log.warning(
                            "StudioForge unreachable (%s); loading on CUDA %d without a lease: %s",
                            outcome.reason,
                            dev,
                            why,
                        )
                        break
                    self._enter_cooldown(
                        "gpu_unavailable",
                        cfg.studioforge.refusal_cooldown_s,
                        f"StudioForge unreachable and the card is not clearly free ({why})",
                    )
                    return
                self._enter_cooldown(
                    "gpu_unavailable",
                    cfg.studioforge.refusal_cooldown_s,
                    f"StudioForge is unreachable ({outcome.reason})",
                )
                return
            if device is None:
                self.counters["refused"] += 1
                self.last_refusals = [
                    {
                        "device": r.device,
                        "status": r.status,
                        "code": r.code,
                        "retry_after_s": r.retry_after_s,
                        "message": r.message[:160],
                    }
                    for r in refusals
                ]
                waits = [r.retry_after_s for r in refusals if r.retry_after_s]
                wait = max(cfg.studioforge.refusal_cooldown_s, min(waits) if waits else 0.0)
                codes = ", ".join(sorted({r.code for r in refusals})) or "refused"
                log.info("lease refused (%s); retry in %.0f s", codes, wait)
                self._enter_cooldown(
                    "gpu_unavailable", wait, f"StudioForge refused the lease ({codes})"
                )
                return
        self.state = "loading"
        self.device, self.lease = device, lease
        await self._notify("on_lease_state", "held" if lease else "unleased")
        try:
            speaker = await asyncio.to_thread(self._factory, device)
            self.warmup = await asyncio.to_thread(speaker.warm_up)
            self._facts = await asyncio.to_thread(self._facts_fn, device)
            if self._selftest is not None and self.config.selftest.enabled:
                result = await asyncio.to_thread(self._selftest, speaker)
                self.selftest_result = result
                METRICS.set("selftest_pass", 1 if result.get("pass") else 0)
                if not result.get("pass"):
                    self.counters["selftest_failures"] += 1
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(speaker.close)
                    raise RuntimeError(f"self-test failed: {result.get('failed')}")
        except Exception as exc:  # noqa: BLE001
            self.counters["load_failures"] += 1
            self.last_error = f"{type(exc).__name__}: {str(exc)[:300]}"
            log.error("loading the speech models failed: %s", self.last_error)
            await self._release_lease()
            self.device = None
            self._enter_cooldown(
                "load_failed",
                cfg.residency.failure_cooldown_s,
                "the speech models failed to load or self-test",
            )
            return
        self.speaker = speaker
        self.state = "ready"
        self.counters["loads"] += 1
        self.last_error = None
        self.load_ms = round((time.perf_counter() - t_start) * 1000)
        METRICS.set("load_ms", self.load_ms)
        live = self.gpu_live()
        self.vram_at_load = live.get("vram_used_mib")
        self._last_used = self._last_touch = self._last_relink = self._clock()
        log.info("speech models ready", extra={"device": device, "ms": self.load_ms})
        if self._vacate_pending:
            self._vacate_pending = False
            self._schedule_unload(
                "vacated",
                cooldown_s=cfg.studioforge.vacate_cooldown_s,
                code="gpu_vacated",
                announce=True,
                message="StudioForge asked for the card back",
            )

    def _schedule_unload(
        self,
        reason: str,
        *,
        cooldown_s: float,
        code: str = "",
        message: str = "",
        announce: bool = False,
    ) -> None:
        if self._unload_task is not None and not self._unload_task.done():
            return
        self._unload_task = asyncio.get_running_loop().create_task(
            self._unload(
                reason, cooldown_s=cooldown_s, code=code, message=message, announce=announce
            )
        )

    async def _unload(
        self,
        reason: str,
        *,
        cooldown_s: float,
        code: str = "",
        message: str = "",
        announce: bool = False,
    ) -> None:
        if self.speaker is None and self.lease is None:
            if cooldown_s > 0:
                self._enter_cooldown(code or reason, cooldown_s, message or reason)
            return
        drain = self.config.residency.drain_timeout_s
        if announce:
            await self._notify("before_unload", reason, drain)
        self.state = "draining"
        self._cooldown_until = self._clock() + cooldown_s
        deadline = self._clock() + drain
        while self._inflight > 0 and self._clock() < deadline:
            await asyncio.sleep(0.05)
        if self._inflight > 0:
            log.warning(
                "%d request(s) still in flight after the drain window; stopping them",
                self._inflight,
            )
            for ev in list(self._stops):
                ev.set()
            await asyncio.sleep(0.2)
        wire = "vacated" if reason in ("vacated", "lease_lost") else "internal"
        await self._notify("after_unload", wire, max(cooldown_s, LOADING_RETRY_S))
        speaker, self.speaker = self.speaker, None
        if speaker is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(speaker.close)
            del speaker
            gc.collect()
        await self._release_lease()
        self.device = None
        self.counters["unloads"] += 1
        log.info("speech models unloaded (%s)", reason)
        if cooldown_s > 0:
            self._enter_cooldown(code or reason, cooldown_s, message or reason)
        else:
            self.state = "unloaded"

    async def _release_lease(self) -> None:
        lease, self.lease = self.lease, None
        if lease is not None and self.lease_client is not None:
            await self.lease_client.release(lease.lease_id)

    def _unloading(self) -> bool:
        return self._unload_task is not None and not self._unload_task.done()

    async def _maybe_relink(self, now: float) -> bool:
        """Unleased (StudioForge was unreachable at load): try for the lease again.

        True when the server is giving way (the caller stops ticking).
        """
        client = self.lease_client
        cfg = self.config.studioforge
        if (
            client is None
            or self.lease is not None
            or self.unleased_reason is None
            or self.device is None
            or self._unloading()
            or now - self._last_relink < cfg.relink_s
        ):
            return False
        self._last_relink = now
        device = self.device
        outcome = await client.acquire(device)
        # A lease under our own holder (persisted by StudioForge across its restart, left
        # by a crash of ours) reads as a conflict; we hold none here, so it is stale.
        if isinstance(outcome, Refused) and await client.release_stale():
            outcome = await client.acquire(device)
        if self.state != "ready" or self._unloading():
            # Something else took over while we were asking; hand a fresh grant back.
            if isinstance(outcome, Granted):
                await client.release(outcome.lease_id)
            return False
        if isinstance(outcome, Granted):
            self.lease = outcome
            self.unleased_reason = None
            self._last_touch = now
            METRICS.inc("lease_events_total", event="relink")
            log.info("StudioForge is reachable again; now holding a lease on CUDA %d", device)
            await self._notify("on_lease_state", "held")
            return False
        if isinstance(outcome, Refused) and outcome.code != "lease_conflict":
            # Transient: the card is being cleared for us (lease_vacating) or a resident
            # is mid-request (model_busy). Keep serving and ask again next interval.
            if outcome.code != self._relink_last_code:
                log.info("StudioForge refused the lease for now (%s); asking again", outcome.code)
            self._relink_last_code = outcome.code
            return False
        if isinstance(outcome, Refused):
            self.counters["vacates"] += 1
            METRICS.inc("lease_events_total", event="relink_refused")
            log.info(
                "StudioForge is back and the card is committed elsewhere (%s); giving it back",
                outcome.code,
                extra={"state": self.state},
            )
            self._schedule_unload(
                "vacated",
                cooldown_s=cfg.vacate_cooldown_s,
                code="gpu_vacated",
                message="StudioForge needs the card; use a local engine",
                announce=True,
            )
            return True
        log.debug("StudioForge still unreachable (%s); staying unleased", outcome.reason)
        return False

    async def _maybe_touch(self, now: float) -> None:
        if self.lease is None or self.lease_client is None:
            return
        cfg = self.config
        if now - self._last_touch < cfg.studioforge.touch_interval_s:
            return
        always = cfg.residency.idle_unload_s <= 0 or self._sessions > 0
        if not always and not self._active_since_touch and self._inflight == 0:
            return
        self._last_touch = now
        self._active_since_touch = False
        ok = await self.lease_client.touch(self.lease.lease_id)
        if ok is not False:
            return
        device = self.device if self.device is not None else cfg.gpu.devices[0]
        outcome = await self.lease_client.acquire(device)
        if isinstance(outcome, Granted):
            self.lease = outcome
            log.info("lease was gone; re-acquired on CUDA %d", device)
            return
        self.lease = None
        METRICS.inc("lease_events_total", event="lost")
        log.warning("lease was gone and could not be re-acquired; unloading")
        await self._notify("on_lease_state", "lost")
        self._schedule_unload(
            "lease_lost",
            cooldown_s=cfg.studioforge.refusal_cooldown_s,
            code="gpu_unavailable",
            message="the GPU lease was lost",
        )

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001
                log.exception("residency tick failed")
