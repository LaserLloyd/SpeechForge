"""StudioForge GPU leases (StudioForge D43/D53/D56/D61), the client side.

SpeechForge does not share StudioForge's process or its VRAM book; it is a
*tenant* of the rig, exactly like any other lease-holding workload. The lease is the
primitive that makes that safe: "while this lease stands nothing else is
planned onto those cards". Without it a JIT LLM load can be planned into the
very bytes the speech models are being loaded into -- a check-then-act race
no amount of polling free VRAM can close (it has corrupted an image render on
this rig before).

Rules this module keeps (S3):

* **Never ``force``.** A pinned or deliberately placed model is the
  operator's decision; a voice chat does not overrule it.
* **Class 2 by default.** StudioForge measures an idle resident against the
  claim's class: a class-2 grant may displace an idle tier-2/3 resident and
  is refused by an idle tier-1 one (the chat model) -- "voice never evicts the
  chat LLM". A busy resident is never interrupted (503 + retry).
* **Register a vacate URL.** A strictly better class (a tier-1 lease) may
  then ask us to leave (D56); :mod:`speechforge.residency` drains, unloads and
  releases. Equal or worse classes get a plain 409 and wait for our idle TTL.
* **Fail open when StudioForge is unreachable** (configurable): with the
  planner down nothing is being JIT-loaded onto the card.
* **Clean up after ourselves.** At boot this process holds nothing, so any
  lease still under our holder name (D61 persists leases across a StudioForge
  restart; a crash of ours leaves one for up to its idle TTL) is released.

Every call is bounded by ``timeout_s`` and never raises to the caller; the
outcome is a value. The vacate token never appears in a log line or a status.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("speechforge.lease")


@dataclass(frozen=True)
class Granted:
    lease_id: str
    device: int
    view: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Refused:
    """StudioForge said no. ``code`` is its machine code (``lease_conflict``,
    ``lease_vacating``, ``model_busy``, ...); ``retry_after_s`` its advice."""

    status: int
    code: str
    message: str
    retry_after_s: float | None
    device: int


@dataclass(frozen=True)
class Unreachable:
    reason: str  # an exception CLASS name or "http <status>", never a message
    device: int


LeaseOutcome = Granted | Refused | Unreachable


def _retry_after(response: httpx.Response, body: dict[str, Any]) -> float | None:
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass
    err = body.get("error") if isinstance(body.get("error"), dict) else {}
    details = err.get("studioforge") if isinstance(err.get("studioforge"), dict) else {}
    for src in (details, err, body):
        value = src.get("retry_after_s") if isinstance(src, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


class LeaseClient:
    def __init__(
        self,
        base_url: str,
        *,
        holder: str,
        priority: int,
        idle_ttl_s: float | None,
        vacate_url: str,
        vacate_token: str,
        timeout_s: float = 5.0,
        api_key: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.holder = holder
        self.priority = priority
        self.idle_ttl_s = idle_ttl_s
        self.vacate_url = vacate_url
        self._vacate_token = vacate_token
        self.timeout_s = timeout_s
        headers = {"X-SF-Client": holder}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": timeout_s,
            "headers": headers,
            # The lease body carries our vacate token: never via an env proxy.
            "trust_env": False,
            "follow_redirects": False,
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**kwargs)

    def token_matches(self, presented: str | None) -> bool:
        import hmac

        return bool(presented) and hmac.compare_digest(
            str(presented).encode(), self._vacate_token.encode()
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def acquire(self, device: int) -> LeaseOutcome:
        body = {
            "devices": [int(device)],
            "holder": self.holder,
            "reason": "speech: Parakeet STT + Chatterbox TTS (SpeechForge)",
            "idle_ttl_s": self.idle_ttl_s,
            "priority": self.priority,
            "vacate_url": self.vacate_url,
            "vacate_token": self._vacate_token,
            "force": False,
        }
        try:
            response = await self._client.post("/api/leases", json=body)
        except httpx.HTTPError as exc:
            return Unreachable(type(exc).__name__, device)
        data = _json(response)
        if response.status_code in (200, 201):
            lease_id = str(data.get("id") or "")
            if not lease_id:
                return Unreachable("grant without an id", device)
            return Granted(lease_id, device, _public(data))
        if response.status_code in (404, 405):
            # A StudioForge without the lease API (pre-D43) or not StudioForge at all.
            return Unreachable(f"http {response.status_code}", device)
        err = data.get("error") if isinstance(data.get("error"), dict) else {}
        return Refused(
            status=response.status_code,
            code=str(err.get("code") or f"http_{response.status_code}"),
            message=str(err.get("message") or "")[:300],
            retry_after_s=_retry_after(response, data),
            device=device,
        )

    async def touch(self, lease_id: str) -> bool | None:
        """True kept alive, False the lease is gone (404), None unknown."""
        try:
            response = await self._client.post(f"/api/leases/{lease_id}/touch")
        except httpx.HTTPError:
            return None
        if response.status_code == 404:
            return False
        return True if response.is_success else None

    async def release(self, lease_id: str) -> bool:
        try:
            response = await self._client.delete(f"/api/leases/{lease_id}")
        except httpx.HTTPError as exc:
            log.warning(
                "lease release failed (%s); StudioForge's idle TTL will reclaim it",
                type(exc).__name__,
            )
            return False
        if response.status_code == 404:
            return True  # already gone: the outcome we wanted
        if not response.is_success:
            log.warning(
                "lease release answered %s; StudioForge's idle TTL will reclaim it",
                response.status_code,
            )
        return response.is_success

    async def ours(self) -> list[dict[str, Any]]:
        try:
            response = await self._client.get("/api/leases")
        except httpx.HTTPError:
            return []
        if not response.is_success:
            return []
        rows = _json(response).get("leases") or []
        return [r for r in rows if isinstance(r, dict) and r.get("holder") == self.holder]

    async def release_stale(self) -> int:
        """At boot nothing is loaded, so every lease under our holder is stale."""
        released = 0
        for row in await self.ours():
            lease_id = str(row.get("id") or "")
            if lease_id and await self.release(lease_id):
                released += 1
                log.info("released stale lease %s on CUDA %s", lease_id, row.get("devices"))
        return released


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _public(view: dict[str, Any]) -> dict[str, Any]:
    """The lease fields worth showing in our status (never the vacate URL/token)."""
    keep = ("id", "devices", "holder", "priority", "idle_ttl_s", "state", "expires_at")
    return {k: view.get(k) for k in keep if k in view}
