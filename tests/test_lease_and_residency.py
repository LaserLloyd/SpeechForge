"""The lease client against a fake StudioForge, and the residency state machine."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from speechforge.errors import Forbidden, Unavailable
from speechforge.lease import Granted, LeaseClient, Refused, Unreachable
from speechforge.pipeline import Speaker
from speechforge.residency import Residency
from tests.conftest import FakeStt, FakeTts

TOKEN = "vacate-secret-token"


def client_for(sf, **kw) -> LeaseClient:
    return LeaseClient(
        "http://sf.test",
        holder="speechforge",
        priority=2,
        idle_ttl_s=900.0,
        vacate_url="http://127.0.0.1:1236/internal/vacate",
        vacate_token=TOKEN,
        transport=httpx.MockTransport(sf.handler),
        **kw,
    )


async def settle(res: Residency) -> None:
    for _ in range(200):
        task = res._bringup
        unload = res._unload_task
        if (task is None or task.done()) and (unload is None or unload.done()):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("residency did not settle")


# --------------------------------------------------------------------- client
async def test_grant_sends_class_2_vacate_url_and_never_force(studioforge):
    lc = client_for(studioforge)
    out = await lc.acquire(3)
    assert isinstance(out, Granted) and out.device == 3
    method, path, body = studioforge.requests[-1]
    assert (method, path) == ("POST", "/api/leases")
    assert body["priority"] == 2 and body["force"] is False and body["devices"] == [3]
    assert body["vacate_url"].endswith("/internal/vacate") and body["vacate_token"] == TOKEN
    assert "vacate_token" not in out.view and "vacate_url" not in out.view
    await lc.aclose()


async def test_refusal_carries_code_and_retry_after(studioforge):
    studioforge.refuse = (409, "lease_conflict", 120)
    lc = client_for(studioforge)
    out = await lc.acquire(3)
    assert isinstance(out, Refused)
    assert (out.status, out.code, out.retry_after_s) == (409, "lease_conflict", 120.0)
    await lc.aclose()


async def test_unreachable_is_a_value_not_an_exception(studioforge):
    studioforge.down = True
    lc = client_for(studioforge)
    out = await lc.acquire(3)
    assert isinstance(out, Unreachable) and out.reason == "ConnectError"
    assert await lc.touch("x") is None
    assert await lc.release("x") is False
    await lc.aclose()


async def test_release_stale_only_touches_our_holder(studioforge):
    studioforge.leases = {
        "a": {"id": "a", "devices": [3], "holder": "speechforge"},
        "b": {"id": "b", "devices": [2], "holder": "image-render"},
    }
    lc = client_for(studioforge)
    assert await lc.release_stale() == 1
    assert list(studioforge.leases) == ["b"]
    await lc.aclose()


def test_token_compare():
    lc = LeaseClient(
        "http://sf.test", holder="h", priority=2, idle_ttl_s=1, vacate_url="u", vacate_token=TOKEN
    )
    assert lc.token_matches(TOKEN)
    assert not lc.token_matches("nope") and not lc.token_matches(None) and not lc.token_matches("")


# --------------------------------------------------------------------- residency
def make(
    config,
    studioforge,
    clock,
    *,
    factory=None,
    lease=True,
    free=(True, "20.0 GiB free"),
    selftest=None,
):
    tts, stt = FakeTts(), FakeStt()
    made: list[int] = []

    def default_factory(device: int) -> Speaker:
        made.append(device)
        return Speaker(tts, stt)

    res = Residency(
        config,
        factory=factory or default_factory,
        lease=client_for(studioforge) if lease else None,
        clock=clock,
        free_check=lambda d, gib: free,
        facts=lambda d: {"name": "Fake 3090", "sm": "sm_86"},
        live=lambda d: {"vram_used_mib": 6000},
        selftest=selftest,
    )
    return res, made, tts


async def test_first_request_503s_then_ready_with_lease(config, studioforge, clock):
    res, made, _ = make(config, studioforge, clock)
    with pytest.raises(Unavailable) as exc:
        res.require()
    assert exc.value.code == "not_ready" and exc.value.retry_after_s
    await settle(res)
    assert res.state == "ready" and made == [3]
    assert res.require() is res.speaker
    assert list(studioforge.leases.values())[0]["devices"] == [3]


async def test_refused_on_first_card_takes_the_next(config, studioforge, clock):
    studioforge.leases = {"x": {"id": "x", "devices": [3], "holder": "benchmark"}}
    res, made, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    assert res.state == "ready" and made == [1]


async def test_refused_everywhere_cools_down_with_studioforge_advice(config, studioforge, clock):
    studioforge.refuse = (503, "model_busy", 45)
    res, made, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    assert res.state == "cooldown" and made == []
    with pytest.raises(Unavailable) as exc:
        res.require()
    assert exc.value.code == "gpu_unavailable" and exc.value.retry_after_s == pytest.approx(45)
    assert res.status()["cooldown"]["refusals"][0]["code"] == "model_busy"
    # No new ask inside the cooldown; one after it.
    n = len(studioforge.requests)
    with pytest.raises(Unavailable):
        res.require()
    assert len(studioforge.requests) == n
    studioforge.refuse = None
    clock.advance(46)
    with pytest.raises(Unavailable):
        res.require()
    await settle(res)
    assert res.state == "ready"


async def test_unreachable_proceeds_only_if_the_card_is_free(config, studioforge, clock):
    studioforge.down = True
    res, made, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    assert res.state == "ready" and res.lease is None and "unreachable" in res.unleased_reason


async def test_unreachable_and_card_busy_refuses(config, studioforge, clock):
    studioforge.down = True
    res, made, _ = make(
        config, studioforge, clock, free=(False, "a llama-server process holds memory on the card")
    )
    res.kick()
    await settle(res)
    assert res.state == "cooldown" and made == []
    assert "llama-server" in res.status()["cooldown"]["message"]


async def test_unreachable_refuse_mode_stays_off_the_card(config, studioforge, clock):
    config.studioforge.on_unreachable = "refuse"
    studioforge.down = True
    res, made, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    assert res.state == "cooldown" and made == []


async def test_failed_selftest_unloads_and_releases(config, studioforge, clock):
    config.selftest.enabled = True
    res, made, tts = make(
        config, studioforge, clock, selftest=lambda sp: {"pass": False, "failed": ["ttfa"]}
    )
    res.kick()
    await settle(res)
    assert res.state == "cooldown" and studioforge.leases == {} and tts.closed
    assert "self-test" in res.status()["last_error"]


async def test_sessions_hold_off_idle_unload_and_keep_touching(config, studioforge, clock):
    res, _, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    res.session_opened()
    clock.advance(10_000)
    await res.tick()
    assert res.state == "ready"
    assert [r for r in studioforge.requests if r[1].endswith("/touch")]
    res.session_closed()
    clock.advance(601)
    await res.tick()
    assert res.state == "unloaded"


async def test_load_failure_releases_the_lease(config, studioforge, clock):
    def boom(device: int) -> Speaker:
        raise RuntimeError("CUDA out of memory")

    res, _, _ = make(config, studioforge, clock, factory=boom)
    res.kick()
    await settle(res)
    assert res.state == "cooldown" and studioforge.leases == {}
    assert "out of memory" in res.status()["last_error"]
    with pytest.raises(Unavailable) as exc:
        res.require()
    assert exc.value.code == "load_failed"


async def test_idle_unload_releases_the_lease(config, studioforge, clock):
    res, _, tts = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    clock.advance(599)
    await res.tick()
    assert res.state == "ready"
    clock.advance(2)
    await res.tick()
    assert res.state == "unloaded" and studioforge.leases == {} and tts.closed


async def test_in_flight_request_holds_off_idle_unload(config, studioforge, clock):
    res, _, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    async with res.use():
        clock.advance(10_000)
        await res.tick()
        assert res.state == "ready"


async def test_touch_only_after_activity_and_reacquire_when_gone(config, studioforge, clock):
    res, _, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    touches = lambda: [r for r in studioforge.requests if r[1].endswith("/touch")]  # noqa: E731
    clock.advance(31)
    await res.tick()
    assert touches() == []  # quiet: let the idle clocks run
    async with res.use():
        pass
    clock.advance(31)
    await res.tick()
    assert len(touches()) == 1
    studioforge.leases.clear()  # an admin released it
    async with res.use():
        pass
    clock.advance(31)
    await res.tick()
    await settle(res)
    assert res.state == "ready" and len(studioforge.leases) == 1


async def test_vacate_drains_unloads_releases_and_cools_down(config, studioforge, clock):
    res, _, tts = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    lid = res.lease.lease_id
    with pytest.raises(Forbidden):
        await res.vacate(lid, "wrong")
    out = await res.vacate(lid, TOKEN)
    assert out["accepted"] and out["held"]
    await settle(res)
    assert res.state == "cooldown" and studioforge.leases == {} and tts.closed
    with pytest.raises(Unavailable) as exc:
        res.require()
    assert exc.value.code == "gpu_vacated" and exc.value.retry_after_s == pytest.approx(120)


async def test_vacate_during_load_unloads_right_after(config, studioforge, clock):
    gate = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow(device: int) -> Speaker:
        asyncio.run_coroutine_threadsafe(gate.wait(), loop).result(5)
        return Speaker(FakeTts(), FakeStt())

    res, _, _ = make(config, studioforge, clock, factory=slow)
    res.kick()
    for _ in range(100):
        if res.state == "loading":
            break
        await asyncio.sleep(0.01)
    await res.vacate(res.lease.lease_id, TOKEN)
    gate.set()
    await settle(res)
    await settle(res)
    assert res.state == "cooldown" and studioforge.leases == {}


async def test_vacate_for_a_lease_we_do_not_hold_releases_it(config, studioforge, clock):
    studioforge.leases = {"old": {"id": "old", "devices": [3], "holder": "speechforge"}}
    res, _, _ = make(config, studioforge, clock)
    out = await res.vacate("old", TOKEN)
    assert out["held"] is False
    await asyncio.sleep(0.05)
    assert studioforge.leases == {}


async def test_lease_mode_off_never_talks_to_studioforge(config, studioforge, clock):
    res, made, _ = make(config, studioforge, clock, lease=False)
    res.kick()
    await settle(res)
    assert res.state == "ready" and made == [3] and studioforge.requests == []


async def test_drain_stops_a_stuck_stream(config, studioforge, clock):
    res, _, _ = make(config, studioforge, clock)
    res.kick()
    await settle(res)
    clock_real = asyncio.get_running_loop().time

    async def stuck():
        async with res.use() as stop:
            while not stop.is_set():
                await asyncio.sleep(0.01)

    task = asyncio.create_task(stuck())
    await asyncio.sleep(0.02)
    # Use real time for the drain window: swap the clock for the loop's.
    res._clock = clock_real
    await res._unload("test", cooldown_s=0)
    await asyncio.wait_for(task, 2)
    assert res.state == "unloaded"


# --------------------------------------------------------------------- relink (S29)
class Told:
    """A session-manager stand-in that records what the residency announces."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def on_lease_state(self, state, deadline_s=None):
        self.events.append(("lease", state))

    async def before_unload(self, reason, timeout_s):
        self.events.append(("before_unload", reason))

    async def after_unload(self, code, retry_after_s):
        self.events.append(("after_unload", code))


async def unleased(config, studioforge, clock):
    """A ready server that loaded without a lease because StudioForge was down."""
    studioforge.down = True
    res, _, tts = make(config, studioforge, clock)
    told = Told()
    res.add_listener(told)
    res.kick()
    await settle(res)
    assert res.state == "ready" and res.lease is None
    told.events.clear()
    return res, told, tts


def lease_posts(studioforge) -> int:
    return len([r for r in studioforge.requests if (r[0], r[1]) == ("POST", "/api/leases")])


async def test_relink_grants_when_studioforge_returns(config, studioforge, clock):
    res, told, _ = await unleased(config, studioforge, clock)
    studioforge.down = False
    clock.advance(31)
    await res.tick()
    assert res.state == "ready" and res.lease is not None
    assert list(studioforge.leases) == [res.lease.lease_id]
    assert studioforge.leases[res.lease.lease_id]["devices"] == [3]
    assert told.events == [("lease", "held")]
    st = res.status()
    assert st["lease"]["id"] == res.lease.lease_id and st["unleased_reason"] is None
    # From here on it is an ordinary holder: touched, released on unload.
    async with res.use():
        pass
    clock.advance(31)
    await res.tick()
    assert [r for r in studioforge.requests if r[1].endswith("/touch")]
    await res._unload("test", cooldown_s=0)
    assert studioforge.leases == {}


async def test_relink_refusal_gives_way_like_a_vacate(config, studioforge, clock):
    res, told, tts = await unleased(config, studioforge, clock)
    studioforge.down = False
    studioforge.refuse = (409, "lease_conflict", None)
    clock.advance(31)
    await res.tick()
    await settle(res)
    assert told.events == [
        ("before_unload", "vacated"),
        ("after_unload", "vacated"),
    ]
    assert res.state == "cooldown" and tts.closed and studioforge.leases == {}
    assert res.counters["vacates"] == 1
    with pytest.raises(Unavailable) as exc:
        res.require()
    assert exc.value.code == "gpu_vacated" and exc.value.retry_after_s == pytest.approx(120)


async def test_relink_transient_refusal_keeps_serving_and_retries(config, studioforge, clock):
    res, told, tts = await unleased(config, studioforge, clock)
    studioforge.down = False
    for code in ("lease_vacating", "model_busy"):
        studioforge.refuse = (409 if code == "lease_vacating" else 503, code, 5)
        clock.advance(31)
        await res.tick()
        assert res.state == "ready" and res.lease is None and not tts.closed
        assert told.events == [] and res.counters["vacates"] == 0
    studioforge.refuse = None  # the card is clear now: granted on the next try
    clock.advance(31)
    await res.tick()
    assert res.state == "ready" and res.lease is not None
    assert told.events == [("lease", "held")]


async def test_relink_still_unreachable_waits_for_the_next_interval(config, studioforge, clock):
    res, told, _ = await unleased(config, studioforge, clock)
    reason = res.unleased_reason
    clock.advance(20)
    await res.tick()
    assert lease_posts(studioforge) == 0  # fake records nothing while down; interval not up
    clock.advance(11)
    await res.tick()  # 31 s: tries, still down
    assert res.state == "ready" and res.lease is None and res.unleased_reason == reason
    assert told.events == []
    # Back up, but only 10 s after the failed attempt: no second try yet.
    studioforge.down = False
    clock.advance(10)
    await res.tick()
    assert lease_posts(studioforge) == 0 and res.lease is None
    clock.advance(21)
    await res.tick()
    assert lease_posts(studioforge) == 1 and res.lease is not None


async def test_relink_interval_is_configurable(config, studioforge, clock):
    config.studioforge.relink_s = 5
    res, _, _ = await unleased(config, studioforge, clock)
    studioforge.down = False
    clock.advance(6)
    await res.tick()
    assert res.lease is not None


async def test_relink_clears_a_stale_lease_of_our_own_holder(config, studioforge, clock):
    res, told, _ = await unleased(config, studioforge, clock)
    studioforge.leases = {"old": {"id": "old", "devices": [3], "holder": "speechforge"}}
    studioforge.down = False
    clock.advance(31)
    await res.tick()
    assert res.lease is not None and res.lease.lease_id != "old"
    assert list(studioforge.leases) == [res.lease.lease_id]
    assert told.events == [("lease", "held")]


async def test_lease_mode_off_never_relinks(config, studioforge, clock):
    res, _, _ = make(config, studioforge, clock, lease=False)
    res.kick()
    await settle(res)
    assert res.unleased_reason == "lease_mode off"
    async with res.use():  # busy, so the idle clock cannot unload it first
        clock.advance(10_000)
        await res.tick()
        assert res.state == "ready" and res.lease is None and studioforge.requests == []
