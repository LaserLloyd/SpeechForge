"""A system-tray controller for the server (DECISIONS S25).

``speechforge tray`` puts an "SF" icon in the notification area, like the other
Forges on the same host. It is a *controller*, not a second supervisor:

* **adopt or launch.** If ``GET /health`` on the configured loopback port
  answers, the tray shows that server and never fights it. Otherwise it starts
  ``speechforge supervise`` as a child (same interpreter, console hidden on
  Windows, ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, the same data dir) and owns it.
* **polls**: ``/health`` (no auth) every 2 s, ``/ready`` (bearer token, read
  from the data dir's env file through :class:`~speechforge.auth.TokenStore`)
  every 5 s. The token is held in memory only; nothing here logs or shows it.
* **status dot**: grey = no server, amber = loading / cooldown / idle, green =
  ready, blue = ready with sessions open, red = crash loop or health failing,
  purple = draining for a vacate or a lost lease.
* **menu**: status line, logs / data folders, copy the health URL, restart
  (writes ``state/restart.request``: the supervisor's graceful restart), stop
  (CTRL_BREAK to the supervisor child the tray launched; an adopted server is
  left alone and the menu says so), start-at-logon shim, quit.

Everything that needs no desktop session is a plain function at the top of
the module (status mapping, the Startup shim, the restart request, the
adopt-or-launch decision) so ``tests/test_tray.py`` can exercise it; the
pystray / Pillow imports happen inside :func:`make_icon_image` and
:func:`main`, so the rest of the package never needs the ``tray`` extra.
"""

from __future__ import annotations

import contextlib
import json
import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speechforge import __version__
from speechforge.config import Config

log = logging.getLogger("speechforge.tray")

APP_NAME = "SpeechForge"
TRAY_LAUNCHER = "SpeechForge Tray.bat"
STARTUP_SHIM = "SpeechForge.bat"

HEALTH_INTERVAL_S = 2.0
READY_INTERVAL_S = 5.0
HTTP_TIMEOUT_S = 2.5
#: After a launch, a silent ``/health`` is "starting", not "failing", this long.
BOOT_GRACE_S = 90.0
#: How long a graceful stop of the supervisor child may take (its own child
#: gets 20 s, then terminate, then kill).
STOP_GRACE_S = 45.0
#: A loopback socket held while a tray runs: a second tray exits at once.
MUTEX_PORT_ENV = "SPEECHFORGE_TRAY_MUTEX_PORT"
MUTEX_PORT_DEFAULT = 47828
SUPERVISOR_LOG = "supervisor.out.log"
SUPERVISOR_LOG_MAX_BYTES = 5_000_000

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

# Colours: the "SF" glyph is teal; the dot carries the state.
COLOURS: dict[str, tuple[int, int, int, int]] = {
    "grey": (130, 134, 142, 255),
    "amber": (245, 180, 50, 255),
    "green": (64, 200, 96, 255),
    "blue": (72, 140, 255, 255),
    "red": (230, 70, 70, 255),
    "purple": (170, 90, 230, 255),
}
ICON_BG = (20, 34, 38, 255)
ICON_TEAL = (72, 200, 190, 255)


# --------------------------------------------------------------------------
# Pure helpers (unit-tested, no display, no network)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Status:
    """What the icon and the tooltip show."""

    colour: str
    text: str
    detail: str = ""

    @property
    def tooltip(self) -> str:
        line = f"{APP_NAME}: {self.text}"
        return f"{line} | {self.detail}" if self.detail else line


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/health"


def ready_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/ready"


def _ready_detail(ready: dict[str, Any]) -> str:
    parts: list[str] = []
    device = ready.get("device")
    if device is not None:
        parts.append(f"CUDA {device}")
    sessions = ready.get("sessions_open", ready.get("sessions"))
    if sessions is not None:
        parts.append(f"sessions {sessions}")
    lease = ready.get("lease")
    if isinstance(lease, dict) and lease.get("state"):
        parts.append(f"lease {lease['state']}")
    elif ready.get("lease_mode") == "off":
        parts.append("lease off")
    elif ready.get("unleased_reason"):
        parts.append("unleased")
    selftest = ready.get("selftest")
    ttfa = selftest.get("ttfa_ms") if isinstance(selftest, dict) else None
    if isinstance(ttfa, int | float):
        parts.append(f"TTFA {ttfa:.0f} ms")
    return " | ".join(parts)


def classify(
    health: dict[str, Any] | None,
    ready: dict[str, Any] | None,
    *,
    child_alive: bool = False,
    seen_up: bool = False,
    booting: bool = False,
    ready_error: str = "",
) -> Status:
    """Map the two probes (and what the tray knows about its child) to a
    colour and a short text.

    ``health``/``ready`` are the decoded bodies, ``None`` when the probe got no
    JSON answer. ``/ready`` answers 503 *with* a body while not ready, so a
    503 still arrives here as a dict; ``ready_error`` is why there is none
    (a rejected token, a timeout) and is shown, never swallowed.
    """
    if health is None:
        if child_alive and booting and not seen_up:
            return Status("amber", "starting...")
        if seen_up or child_alive:
            return Status("red", "server not answering /health")
        return Status("grey", "no server")
    if health.get("ok") is False or health.get("crashloop") or health.get("state") == "crashloop":
        return Status("red", "crash loop: 5 crashes in 10 min; see the logs")
    if ready is None:
        text = "up; /ready " + (ready_error or "not probed yet")
        return Status("amber", text)
    state = str(ready.get("state") or "")
    detail = _ready_detail(ready)
    sessions = ready.get("sessions_open", ready.get("sessions")) or 0
    if state == "ready":
        if sessions:
            return Status("blue", f"ready, {sessions} session(s) open", detail)
        return Status("green", "ready", detail)
    if state == "draining":
        return Status("purple", "draining (vacate / lease lost)", detail)
    if state == "cooldown":
        cd = ready.get("cooldown") if isinstance(ready.get("cooldown"), dict) else {}
        code = str(cd.get("code") or "")
        message = str(cd.get("message") or "")
        left = cd.get("retry_after_s")
        when = f" ({left:.0f} s)" if isinstance(left, int | float) else ""
        if code == "gpu_vacated" or "lease was lost" in message:
            return Status("purple", f"vacated{when}: {message or code}", detail)
        return Status("amber", f"cooldown{when}: {code or 'unknown'}", detail)
    if state in ("acquiring", "loading"):
        return Status("amber", f"{state}...", detail)
    if state == "unloaded":
        return Status("amber", "idle (models not loaded)", detail)
    return Status("amber", f"state {state or 'unknown'}", detail)


def decide(health_answers: bool) -> str:
    """``"adopt"`` when a server already answers on the port, else ``"launch"``."""
    return "adopt" if health_answers else "launch"


def _request(state_dir: Path, name: str) -> Path:
    """Drop a request file for the supervisor, atomically: it unlinks the file
    the moment it sees it, and on Windows an unlink fails while the writer
    still has it open."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / name
    tmp = marker.with_name(marker.name + ".tmp")
    tmp.write_text("tray\n", encoding="utf-8")
    os.replace(tmp, marker)
    return marker


def request_restart(state_dir: Path) -> Path:
    """Ask the supervisor for a graceful restart (it polls this file every second)."""
    return _request(state_dir, "restart.request")


def request_stop(state_dir: Path) -> Path:
    """Ask the supervisor to stop its server and exit (same mechanism as the
    restart request; a console signal cannot reach a window-less child)."""
    return _request(state_dir, "stop.request")


def kill_tree(proc: Any) -> None:
    """Last resort for *our own* supervisor child: take its tree down so the
    server it spawned does not keep the port and the lease. Never by name."""
    pid = getattr(proc, "pid", None)
    if not pid or proc.poll() is not None:
        return
    if sys.platform == "win32":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed argv, our child's pid
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=20,
                creationflags=CREATE_NO_WINDOW,
            )
    else:
        with contextlib.suppress(OSError):
            proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        proc.wait(timeout=10)
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.kill()


def console_python() -> str:
    """``python.exe`` even when the tray itself runs under ``pythonw.exe``."""
    exe = Path(sys.executable)
    if exe.name.lower().startswith("pythonw"):
        candidate = exe.with_name(exe.name.lower().replace("pythonw", "python", 1))
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def supervise_argv(config_path: Path | None = None, python: str | None = None) -> list[str]:
    argv = [python or console_python(), "-m", "speechforge", "supervise"]
    if config_path is not None:
        argv += ["--config", str(config_path)]
    return argv


def child_env(data_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["SPEECHFORGE_DATA_DIR"] = str(data_dir)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def repo_root() -> Path:
    """The checkout (or release) this module runs from."""
    return Path(__file__).resolve().parents[2]


def tray_launcher_path() -> Path:
    """The ``.bat`` the Startup shim should run.

    In the releases layout that is ``<root>\\current\\launchers\\...`` so the
    shim follows a blue/green switch; in a checkout it is the checkout's own.
    """
    from speechforge.update import root_from_here

    root = (
        Path(os.environ["SPEECHFORGE_ROOT"])
        if os.environ.get("SPEECHFORGE_ROOT")
        else root_from_here()
    )
    base = root / "current" if root is not None else repo_root()
    return base / "launchers" / TRAY_LAUNCHER


def startup_dir() -> Path:
    appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def startup_shim_path(folder: Path | None = None) -> Path:
    return (folder or startup_dir()) / STARTUP_SHIM


def startup_shim_content(launcher: Path) -> str:
    """The same two lines ``install.ps1`` writes, so either writer can replace
    the other's file."""
    return f'@echo off\nstart "{APP_NAME}" /min "{launcher}"\n'


def write_startup_shim(launcher: Path, folder: Path | None = None) -> Path:
    path = startup_shim_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(startup_shim_content(Path(launcher)), encoding="utf-8")
    return path


def remove_startup_shim(folder: Path | None = None) -> bool:
    path = startup_shim_path(folder)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def startup_shim_installed(folder: Path | None = None) -> bool:
    return startup_shim_path(folder).exists()


def read_token(env_file: Path) -> str:
    """The current bearer token through the shared store (re-read on change).
    Returned to the caller only; never written anywhere."""
    from speechforge.auth import TokenStore

    store = TokenStore(env_file)
    if not store.configured:
        return ""
    # The store has no public getter on purpose (`bench` reads it the same way).
    return store._current


# --------------------------------------------------------------------------
# Loopback HTTP (the tray talks to 127.0.0.1 only)
# --------------------------------------------------------------------------


def get_json(
    url: str, *, token: str = "", timeout: float = HTTP_TIMEOUT_S
) -> tuple[dict[str, Any] | None, str]:
    """``(body, "")`` on a JSON answer (any status that carries one), else
    ``(None, why)``. The ``Authorization`` header never reaches a log."""
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - loopback
            raw = r.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return None, "token rejected (rotate it or check speechforge.env)"
        try:
            raw = exc.read()
        except Exception:  # noqa: BLE001
            return None, f"HTTP {exc.code}"
        if exc.code != 503:
            return None, f"HTTP {exc.code}"
    except (OSError, ValueError) as exc:
        return None, f"unreachable ({type(exc).__name__})"
    try:
        body = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        return None, "not JSON"
    if not isinstance(body, dict):
        return None, "unexpected body"
    return body, ""


def probe_health(port: int) -> dict[str, Any] | None:
    body, _ = get_json(health_url(port))
    return body


def probe_ready(port: int, token: str) -> tuple[dict[str, Any] | None, str]:
    return get_json(ready_url(port), token=token)


# --------------------------------------------------------------------------
# Desktop bits (lazy imports; no test ever calls these)
# --------------------------------------------------------------------------


def make_icon_image(colour: str, size: int = 64) -> Any:
    """Rounded square, "SF" glyph, status dot in the corner (Pillow)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(1, size // 16)
    d.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=max(3, size // 5),
        fill=ICON_BG,
        outline=ICON_TEAL,
        width=max(2, size // 24),
    )
    font = None
    for name in ("arialbd.ttf", "seguisb.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        with contextlib.suppress(OSError):
            font = ImageFont.truetype(name, max(8, int(size * 0.42)))
            break
    if font is None:
        try:
            font = ImageFont.load_default(size=max(8, int(size * 0.42)))
        except TypeError:
            font = ImageFont.load_default()
    try:
        d.text((size / 2, size * 0.46), "SF", font=font, fill=ICON_TEAL, anchor="mm")
    except (ValueError, TypeError):
        d.text((size * 0.22, size * 0.26), "SF", font=font, fill=ICON_TEAL)
    r = size * 0.185
    cx = cy = size - pad - r * 0.95
    d.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=COLOURS.get(colour, COLOURS["grey"]),
        outline=ICON_BG,
        width=max(1, size // 32),
    )
    return img


def copy_to_clipboard(text: str) -> bool:
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.update()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001
        pass
    if os.name == "nt":
        try:
            subprocess.run(  # noqa: S603 - fixed argv
                ["cmd", "/c", "clip"],
                input=text.encode("utf-8"),
                check=True,
                creationflags=CREATE_NO_WINDOW,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            pass
    return False


def show_message(text: str, title: str = APP_NAME) -> None:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showinfo(title, text)
        root.destroy()
    except Exception:  # noqa: BLE001
        log.warning("%s: %s", title, text)


def ask_yes_no(text: str, title: str = APP_NAME) -> bool:
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        answer = bool(messagebox.askyesno(title, text))
        root.destroy()
        return answer
    except Exception:  # noqa: BLE001
        return False


def open_folder(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(path)])  # noqa: S603 - fixed argv
        return True
    except OSError:
        return False


def acquire_mutex(port: int) -> socket.socket | None:
    """Bind the tray's loopback mutex; ``None`` when another tray holds it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
    except OSError:
        with contextlib.suppress(OSError):
            sock.close()
        return None
    return sock


# --------------------------------------------------------------------------
# The tray
# --------------------------------------------------------------------------


class TrayApp:
    """Icon + poller + the supervisor child it may own.

    Every collaborator that touches the desktop, the network or a process is
    injectable so the tests drive :meth:`tick` with fakes and
    ``create_icon=False``.
    """

    def __init__(
        self,
        config: Config,
        *,
        config_path: Path | None = None,
        create_icon: bool = True,
        health: Callable[[int], dict[str, Any] | None] | None = None,
        ready: Callable[[int, str], tuple[dict[str, Any] | None, str]] | None = None,
        token: Callable[[], str] | None = None,
        popen: Callable[..., Any] | None = None,
        confirm: Callable[[str], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.port = config.server.port
        self._health = health or probe_health
        self._ready = ready or probe_ready
        self._token = token or (lambda: read_token(config.env_file))
        self._popen = popen or subprocess.Popen
        self._confirm = confirm or ask_yes_no
        self._clock = clock
        self.proc: Any = None
        self.adopted = False
        self.seen_up = False
        self.launched_at = 0.0
        self.status = Status("grey", "no server")
        self.last_health: dict[str, Any] | None = None
        self.last_ready: dict[str, Any] | None = None
        self._ready_error = ""
        self._next_ready = 0.0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._log_handle: Any = None
        self.icon: Any = None
        if create_icon:
            import pystray

            self.icon = pystray.Icon(
                APP_NAME.lower(),
                icon=make_icon_image(self.status.colour),
                title=self.status.tooltip,
                menu=self._build_menu(),
            )

    # ---- adopt or launch ---------------------------------------------------

    @property
    def child_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> str:
        """Adopt a server that answers, else launch the supervisor. Returns the
        decision (``"adopt"``, ``"launch"``, or ``"owned"`` when our child is
        already running)."""
        with self._lock:
            if self.child_alive:
                return "owned"
            body = self._safe_health()
            choice = decide(body is not None)
            if choice == "adopt":
                self.adopted = True
                self.seen_up = True
                self.last_health = body
                log.info("adopted the server already answering on :%d", self.port)
            else:
                self._launch()
        self.tick()
        return choice

    def _open_supervisor_log(self) -> Any:
        logs = self.config.logs_dir
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / SUPERVISOR_LOG
        with contextlib.suppress(OSError):
            if path.stat().st_size > SUPERVISOR_LOG_MAX_BYTES:
                os.replace(path, path.with_name(path.name + ".1"))
        return open(path, "ab", buffering=0)  # noqa: SIM115 - handed to Popen

    def _launch(self) -> bool:
        self._close_log()
        try:
            handle = self._open_supervisor_log()
        except OSError as exc:
            log.error("cannot open the supervisor log: %s", exc)
            handle = subprocess.DEVNULL
        flags = (CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW) if sys.platform == "win32" else 0
        try:
            self.proc = self._popen(
                supervise_argv(self.config_path),
                cwd=str(repo_root()),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=child_env(self.config.data_dir),
                creationflags=flags,
            )
        except OSError as exc:
            log.error("could not start the supervisor: %s", exc)
            if handle is not subprocess.DEVNULL:
                with contextlib.suppress(OSError):
                    handle.close()
            self.status = Status("red", f"could not start the supervisor ({type(exc).__name__})")
            return False
        self._log_handle = handle if handle is not subprocess.DEVNULL else None
        self.adopted = False
        self.seen_up = False
        self.launched_at = self._clock()
        log.info("launched speechforge supervise (pid %s)", getattr(self.proc, "pid", "?"))
        return True

    def _close_log(self) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(OSError):
                self._log_handle.close()
            self._log_handle = None

    # ---- polling -----------------------------------------------------------

    def _safe_health(self) -> dict[str, Any] | None:
        try:
            return self._health(self.port)
        except Exception:  # noqa: BLE001
            return None

    def tick(self) -> Status:
        """One poll: ``/health`` every call, ``/ready`` when due. Never raises."""
        with self._lock:
            now = self._clock()
            health = self._safe_health()
            self.last_health = health
            if health is not None:
                self.seen_up = True
            if health is None or health.get("ok") is False:
                self.last_ready, self._ready_error = None, ""
                self._next_ready = 0.0
            elif now >= self._next_ready:
                self._next_ready = now + READY_INTERVAL_S
                try:
                    self.last_ready, self._ready_error = self._ready(self.port, self._token())
                except Exception as exc:  # noqa: BLE001
                    self.last_ready, self._ready_error = (
                        None,
                        f"probe failed ({type(exc).__name__})",
                    )
            child_alive = self.child_alive
            if self.proc is not None and not child_alive and health is None:
                rc = self.proc.returncode
                self.status = Status(
                    "red" if rc else "grey",
                    f"supervisor exited (code {rc})" if rc else "supervisor stopped",
                )
                return self.status
            self.status = classify(
                health,
                self.last_ready,
                child_alive=child_alive,
                seen_up=self.seen_up,
                booting=(now - self.launched_at) < BOOT_GRACE_S,
                ready_error=self._ready_error,
            )
            return self.status

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
                self._refresh()
            except Exception:  # noqa: BLE001
                log.exception("poll failed")
            self._stop.wait(HEALTH_INTERVAL_S)

    # ---- actions -----------------------------------------------------------

    def restart_server(self) -> Path:
        return request_restart(self.config.state_dir)

    def stop_server(self, grace_s: float = STOP_GRACE_S) -> bool:
        """Stop the supervisor child *we* launched (it stops the server, which
        releases its lease). An adopted server is not ours to stop: False.

        The request is a file (``state/stop.request``) because the child runs
        without a console on Windows and a CTRL_BREAK from the tray cannot
        reach it; the signal is still sent as a fallback, then the process
        tree of our own child is taken down.
        """
        with self._lock:
            proc = self.proc
            if proc is None or proc.poll() is not None:
                return False
            request_stop(self.config.state_dir)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=grace_s)
            if proc.poll() is None:
                log.warning("supervisor ignored stop.request for %.0f s; signalling it", grace_s)
                with contextlib.suppress(OSError, ValueError):
                    if sys.platform == "win32":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                    else:
                        proc.send_signal(signal.SIGINT)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=10)
            if proc.poll() is None:
                log.error("supervisor still running; stopping its process tree")
                kill_tree(proc)
            self._close_log()
            self.proc = None
            self.seen_up = False
            return True

    def toggle_startup(self) -> bool:
        """Flip the Startup-folder shim; True when it is installed afterwards."""
        if startup_shim_installed():
            remove_startup_shim()
            return False
        write_startup_shim(tray_launcher_path())
        return True

    # ---- icon / menu -------------------------------------------------------

    def _refresh(self) -> None:
        if self.icon is None:
            return
        try:
            self.icon.icon = make_icon_image(self.status.colour)
            self.icon.title = self.status.tooltip[:127]
            self.icon.update_menu()
        except Exception:  # noqa: BLE001
            log.exception("could not refresh the icon")

    def _notify(self, message: str) -> None:
        try:
            if self.icon is not None and getattr(self.icon, "HAS_NOTIFICATION", False):
                self.icon.notify(message, APP_NAME)
                return
        except Exception:  # noqa: BLE001
            pass
        log.info("%s", message)

    def _in_thread(self, fn: Callable[[], None]) -> None:
        def work() -> None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("%s failed", getattr(fn, "__name__", "action"))
            self._refresh()

        threading.Thread(target=work, daemon=True, name="speechforge-tray-action").start()

    def _build_menu(self) -> Any:
        import pystray

        item = pystray.MenuItem
        sep = pystray.Menu.SEPARATOR
        up = lambda _i: self.last_health is not None  # noqa: E731
        return pystray.Menu(
            item(lambda _i: f"Status: {self.status.text}", None, enabled=False),
            item(
                lambda _i: (
                    "Server: launched by this tray"
                    if self.child_alive
                    else ("Server: adopted (started elsewhere)" if self.adopted else "Server: none")
                ),
                None,
                enabled=False,
            ),
            sep,
            item("Open logs folder", self._on_open_logs),
            item("Open data folder", self._on_open_data),
            item("Copy health URL", self._on_copy_health),
            sep,
            item("Start server", self._on_start, enabled=lambda _i: not up(_i)),
            item("Restart server", self._on_restart, enabled=up),
            item("Stop server", self._on_stop, enabled=lambda _i: self.child_alive),
            item(
                "Start at logon",
                self._on_toggle_startup,
                checked=lambda _i: startup_shim_installed(),
            ),
            sep,
            item("Quit tray", self._on_quit),
        )

    def _on_open_logs(self, _icon: Any = None, _item: Any = None) -> None:
        if not open_folder(self.config.logs_dir):
            self._notify("Could not open the logs folder")

    def _on_open_data(self, _icon: Any = None, _item: Any = None) -> None:
        if not open_folder(self.config.data_dir):
            self._notify("Could not open the data folder")

    def _on_copy_health(self, _icon: Any = None, _item: Any = None) -> None:
        url = health_url(self.port)
        self._in_thread(lambda: self._notify(f"Copied {url}" if copy_to_clipboard(url) else url))

    def _on_start(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            choice = self.start()
            self._notify(
                {
                    "adopt": f"Attached to the server already on :{self.port}",
                    "launch": "Starting the supervisor",
                    "owned": "Already running",
                }[choice]
            )

        self._in_thread(work)

    def _on_restart(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            self.restart_server()
            self._notify("Restart requested; the supervisor restarts the server")

        self._in_thread(work)

    def _on_stop(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            if self.stop_server():
                self._notify("Server stopped")
            else:
                self._notify("Nothing to stop: this tray only stops what it launched")
            self.tick()

        self._in_thread(work)

    def _on_toggle_startup(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            try:
                on = self.toggle_startup()
            except OSError as exc:
                self._notify(f"Could not change the Startup entry ({type(exc).__name__})")
                return
            self._notify("Starts at logon" if on else "Removed from Startup")

        self._in_thread(work)

    def _on_quit(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            if self.child_alive:
                if not self._confirm(
                    "Quit the tray and stop the SpeechForge server it launched?\n\n"
                    "The server releases its GPU lease on the way out."
                ):
                    return
                self.stop_server()
            self._stop.set()
            if self.icon is not None:
                with contextlib.suppress(Exception):
                    self.icon.stop()

        self._in_thread(work)

    # ---- lifecycle ---------------------------------------------------------

    def _setup(self, icon: Any) -> None:
        icon.visible = True
        self._in_thread(lambda: self.start())
        threading.Thread(target=self._poll_loop, daemon=True, name="speechforge-tray-poll").start()

    def run(self) -> int:
        try:
            self.icon.run(setup=self._setup)
        finally:
            self._stop.set()
            self._close_log()
        return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _configure_logging(logs_dir: Path) -> None:
    handlers: list[logging.Handler] = []
    with contextlib.suppress(OSError):
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            logs_dir / "tray.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handlers.append(fh)
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers or None, force=True)


def main(config: Config, config_path: Path | None = None) -> int:
    """``speechforge tray``: one tray per machine, pystray + Pillow required."""
    _configure_logging(config.logs_dir)
    try:
        import PIL  # noqa: F401
        import pystray  # noqa: F401
    except ImportError:
        msg = (
            "The tray needs pystray and Pillow. Install the tray extra:\n\n"
            "    uv sync --extra gpu --extra tray\n\n"
            "or run `speechforge supervise` from a console instead."
        )
        log.error("%s", msg.replace("\n", " "))
        show_message(msg)
        return 1
    try:
        mutex_port = int(os.environ.get(MUTEX_PORT_ENV, "") or MUTEX_PORT_DEFAULT)
    except ValueError:
        mutex_port = MUTEX_PORT_DEFAULT
    guard = acquire_mutex(mutex_port)
    if guard is None:
        log.warning("another %s tray is already running", APP_NAME)
        show_message(f"{APP_NAME} is already in the system tray.")
        return 0
    log.info("SpeechForge %s tray; server port %d", __version__, config.server.port)
    try:
        return TrayApp(config, config_path=config_path).run()
    finally:
        with contextlib.suppress(OSError):
            guard.close()
