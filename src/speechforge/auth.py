"""Bearer tokens with rotation (SPEC-speechforge §8.2).

The token lives in ``speechforge.env``. :class:`TokenStore` re-reads that file
whenever its mtime changes, so ``speechforge token rotate`` takes effect
without a restart: the new token is valid at once and the previous one for a
5-minute overlap (``SPEECHFORGE_TOKEN_PREVIOUS`` /
``SPEECHFORGE_TOKEN_PREVIOUS_UNTIL``), long enough for the add-on to pick up
the new one at its next session. Without a file, the process environment is
used (tests, a launcher that exports it). Comparison is constant-time; the
token is only ever read from the ``Authorization`` header, never a query
string (which lands in logs).
"""

from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from pathlib import Path

from speechforge.config import (
    TOKEN_ENV,
    TOKEN_PREV_ENV,
    TOKEN_PREV_UNTIL_ENV,
    parse_env_file,
)

OVERLAP_S = 300.0


class TokenStore:
    def __init__(self, env_file: Path, *, clock=time.time) -> None:  # type: ignore[no-untyped-def]
        self.env_file = Path(env_file)
        self._clock = clock
        self._lock = threading.Lock()
        self._mtime: int | None = -1
        self._current = ""
        self._previous = ""
        self._previous_until = 0.0

    def _refresh(self) -> None:
        try:
            mtime: int | None = self.env_file.stat().st_mtime_ns
        except OSError:
            mtime = None
        if mtime == self._mtime:
            return
        self._mtime = mtime
        values = parse_env_file(self.env_file) if mtime is not None else {}
        self._current = values.get(TOKEN_ENV, "") or os.environ.get(TOKEN_ENV, "")
        self._previous = values.get(TOKEN_PREV_ENV, "")
        try:
            self._previous_until = float(values.get(TOKEN_PREV_UNTIL_ENV, "0") or 0)
        except ValueError:
            self._previous_until = 0.0

    @property
    def configured(self) -> bool:
        with self._lock:
            self._refresh()
            return bool(self._current)

    def check(self, presented: str | None) -> bool:
        if not presented:
            return False
        with self._lock:
            self._refresh()
            cur, prev, until = self._current, self._previous, self._previous_until
        ok = bool(cur) and hmac.compare_digest(presented.encode(), cur.encode())
        if prev and self._clock() < until:
            ok = hmac.compare_digest(presented.encode(), prev.encode()) or ok
        return ok


def bearer(header: str | None) -> str | None:
    scheme, _, value = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def write_token(env_file: Path, *, rotate: bool) -> str:
    """Write a new 256-bit token. ``rotate`` keeps the old one valid for
    :data:`OVERLAP_S`. Returns the new token (the caller decides whether to
    show it -- ``rotate`` prints nothing)."""
    env_file = Path(env_file)
    values = parse_env_file(env_file)
    old = values.get(TOKEN_ENV, "")
    token = secrets.token_urlsafe(32)
    keep = []
    if env_file.exists():
        drop = (f"{TOKEN_ENV}=", f"{TOKEN_PREV_ENV}=", f"{TOKEN_PREV_UNTIL_ENV}=")
        keep = [
            ln
            for ln in env_file.read_text(encoding="utf-8").splitlines()
            if not ln.strip().removeprefix("export ").startswith(drop)
        ]
    keep.append(f"{TOKEN_ENV}={token}")
    if rotate and old:
        keep.append(f"{TOKEN_PREV_ENV}={old}")
        keep.append(f"{TOKEN_PREV_UNTIL_ENV}={time.time() + OVERLAP_S:.0f}")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = env_file.with_name(env_file.name + ".tmp")
    tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
    if os.name != "nt":
        tmp.chmod(0o600)
    os.replace(tmp, env_file)
    return token
