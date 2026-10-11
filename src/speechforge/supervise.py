"""Keep the server running (SPEC-speechforge §10.3).

``speechforge supervise`` is what the at-logon launcher runs. It starts
``<python> -m speechforge serve`` as a child (fixed argv, no shell) and:

* restarts it when it exits, with backoff 1, 2, 4 ... 30 s (reset after 10
  minutes up); after **5 crashes in 10 minutes** it stops restarting, writes
  ``<data>/state/crashloop.json`` and answers ``GET /health`` itself with
  ``{"ok": false, "state": "crashloop"}`` on loopback, so the box sees a
  crash loop instead of a refused connection;
* restarts it when ``/health`` fails 3 probes in a row (a wedged driver call);
* restarts it on request: ``<data>/state/restart.request`` (written by
  ``speechforge update --apply`` after it switched the ``current`` release,
  and by the tray's "Restart server");
* stops on request: ``<data>/state/stop.request`` (the tray's "Stop server":
  a console signal cannot reach a child that runs without a console);
* stops it gracefully on exit (CTRL_BREAK on Windows, SIGINT elsewhere; the
  child releases its StudioForge lease on the way out), then terminate, kill.

The child's interpreter is ``<root>/current/.venv`` when a blue/green layout
exists (``update.py``), so a switched release takes effect at the next start;
the supervisor itself is updated at the next logon.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("speechforge.supervise")

BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 30.0
CRASH_LIMIT = 5
CRASH_WINDOW_S = 600.0
STABLE_AFTER_S = 600.0
#: ``serve`` exits with this on a configuration it refuses (EX_CONFIG).
EXIT_CONFIG = 78


def _healthy(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local URL
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _stop(proc: subprocess.Popen[Any], grace_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def child_python(root: Path | None) -> str:
    if root is not None:
        for rel in ("current/.venv/Scripts/python.exe", "current/.venv/bin/python"):
            cand = root / rel
            if cand.exists():
                return str(cand)
    return sys.executable


class CrashTracker:
    def __init__(self, limit: int = CRASH_LIMIT, window_s: float = CRASH_WINDOW_S) -> None:
        self.limit, self.window_s = limit, window_s
        self.times: list[float] = []

    def crashed(self, now: float) -> bool:
        """Record a crash; True when the loop limit is reached."""
        self.times = [t for t in self.times if now - t <= self.window_s] + [now]
        return len(self.times) >= self.limit


def serve_crashloop(port: int, stop: threading.Event, stop_file: Path | None = None) -> None:
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            body = json.dumps({"ok": False, "state": "crashloop"}).encode()
            self.send_response(503 if self.path != "/health" else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: Any) -> None:
            pass

    with contextlib.suppress(OSError):
        srv = http.server.HTTPServer(("127.0.0.1", port), H)
        srv.timeout = 1
        while not stop.is_set():
            srv.handle_request()
            if stop_file is not None and stop_file.exists():
                with contextlib.suppress(OSError):
                    stop_file.unlink()
                stop.set()


def supervise(
    port: int,
    state_dir: Path,
    *,
    extra_args: list[str] | None = None,
    root: Path | None = None,
    probe_s: float = 10.0,
    max_failures: int = 3,
    boot_grace_s: float = 60.0,
) -> int:
    url = f"http://127.0.0.1:{port}/health"
    backoff = BACKOFF_START_S
    stopping = threading.Event()
    crashes = CrashTracker()
    state_dir.mkdir(parents=True, exist_ok=True)
    restart_file = state_dir / "restart.request"
    stop_file = state_dir / "stop.request"
    crash_file = state_dir / "crashloop.json"
    for stale in (crash_file, stop_file):
        with contextlib.suppress(OSError):
            stale.unlink()

    def _on_signal(signum: int, _frame: Any) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_signal)

    while not stopping.is_set():
        started = time.monotonic()
        argv = [child_python(root), "-m", "speechforge", "serve", *(extra_args or [])]
        log.info("starting: %s", " ".join(argv[1:]))
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if sys.platform == "win32" else 0
        proc = subprocess.Popen(argv, creationflags=flags)  # noqa: S603 - fixed argv, no shell
        failures = 0
        requested = False
        while not stopping.is_set() and proc.poll() is None:
            time.sleep(1.0)
            if stop_file.exists():
                with contextlib.suppress(OSError):
                    stop_file.unlink()
                log.info("stop requested")
                stopping.set()
                break
            if restart_file.exists():
                with contextlib.suppress(OSError):
                    restart_file.unlink()
                log.info("restart requested")
                requested = True
                _stop(proc)
                break
            elapsed = time.monotonic() - started
            if elapsed < boot_grace_s or int(elapsed) % int(probe_s):
                continue
            if _healthy(url):
                failures = 0
            else:
                failures += 1
                log.warning("health probe failed (%d/%d)", failures, max_failures)
                if failures >= max_failures:
                    log.error("server stopped answering; restarting it")
                    _stop(proc)
        if stopping.is_set():
            _stop(proc)
            break
        if requested:
            backoff = BACKOFF_START_S
            continue
        code = proc.returncode
        if code == EXIT_CONFIG:
            log.error("server refused its configuration (exit %d); not restarting", code)
            return code
        now = time.monotonic()
        if now - started > STABLE_AFTER_S:
            backoff = BACKOFF_START_S
        if crashes.crashed(now):
            log.error(
                "%d crashes in %.0f min: crash loop; not restarting",
                CRASH_LIMIT,
                CRASH_WINDOW_S / 60,
            )
            crash_file.write_text(
                json.dumps({"state": "crashloop", "at": time.time(), "last_exit": code})
            )
            serve_crashloop(port, stopping, stop_file)
            return 1
        log.warning("server exited with code %s; restarting in %.0f s", code, backoff)
        stopping.wait(backoff)
        backoff = min(BACKOFF_MAX_S, backoff * 2)
    log.info("supervisor stopped")
    return 0
