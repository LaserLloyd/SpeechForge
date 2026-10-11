"""The tray's desktop-free parts (S23): status mapping, the Startup shim, the
adopt-or-launch decision, the restart/stop requests, and the supervisor's
side of ``stop.request``. No pystray, no Pillow, no sockets."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from speechforge import supervise as sup
from speechforge import tray
from speechforge.config import Config
from speechforge.tray import Status, TrayApp, classify, decide

HEALTH_OK = {"ok": True, "version": "0.2.0", "api": 1}


def ready_body(state: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "state": state,
        "ready": state == "ready",
        "device": 3,
        "sessions": 0,
        "sessions_open": 0,
        "lease": {"id": "lease1", "state": "active", "devices": [3]},
        "lease_mode": "hold",
        "selftest": {"pass": True, "ttfa_ms": 187.4},
    }
    body.update(extra)
    return body


# ------------------------------------------------------------- status mapping
@pytest.mark.parametrize(
    ("health", "ready", "kw", "colour", "needle"),
    [
        (None, None, {}, "grey", "no server"),
        (None, None, {"child_alive": True, "booting": True}, "amber", "starting"),
        (None, None, {"seen_up": True}, "red", "not answering"),
        (None, None, {"child_alive": True, "booting": False}, "red", "not answering"),
        ({"ok": False, "state": "crashloop"}, None, {}, "red", "crash loop"),
        ({**HEALTH_OK, "crashloop": True}, None, {}, "red", "crash loop"),
        (HEALTH_OK, None, {}, "amber", "not probed"),
        (HEALTH_OK, None, {"ready_error": "token rejected"}, "amber", "token rejected"),
        (HEALTH_OK, ready_body("ready"), {}, "green", "ready"),
        (HEALTH_OK, ready_body("ready", sessions_open=2, sessions=2), {}, "blue", "2 session"),
        (HEALTH_OK, ready_body("loading"), {}, "amber", "loading"),
        (HEALTH_OK, ready_body("acquiring"), {}, "amber", "acquiring"),
        (HEALTH_OK, ready_body("unloaded", device=None, lease=None), {}, "amber", "idle"),
        (HEALTH_OK, ready_body("draining"), {}, "purple", "draining"),
        (
            HEALTH_OK,
            ready_body(
                "cooldown",
                cooldown={"code": "gpu_vacated", "message": "vacated", "retry_after_s": 120.0},
            ),
            {},
            "purple",
            "vacated (120 s)",
        ),
        (
            HEALTH_OK,
            ready_body(
                "cooldown",
                cooldown={
                    "code": "gpu_unavailable",
                    "message": "the GPU lease was lost",
                    "retry_after_s": 30.0,
                },
            ),
            {},
            "purple",
            "lease was lost",
        ),
        (
            HEALTH_OK,
            ready_body("cooldown", cooldown={"code": "load_failed", "retry_after_s": 59.6}),
            {},
            "amber",
            "cooldown (60 s): load_failed",
        ),
    ],
)
def test_classify(health, ready, kw, colour, needle):
    st = classify(health, ready, **kw)
    assert st.colour == colour
    assert needle in st.text
    assert st.colour in tray.COLOURS


def test_tooltip_carries_device_sessions_lease_and_ttfa():
    st = classify(HEALTH_OK, ready_body("ready", sessions_open=1, sessions=1))
    assert st.tooltip.startswith("SpeechForge: ready, 1 session(s) open")
    for piece in ("CUDA 3", "sessions 1", "lease active", "TTFA 187 ms"):
        assert piece in st.detail
    bare = classify(HEALTH_OK, ready_body("ready", lease=None, lease_mode="off", selftest={}))
    assert "lease off" in bare.detail and "TTFA" not in bare.detail
    assert Status("grey", "no server").tooltip == "SpeechForge: no server"


def test_health_and_ready_urls_are_loopback_only():
    assert tray.health_url(1236) == "http://127.0.0.1:1236/health"
    assert tray.ready_url(1236) == "http://127.0.0.1:1236/ready"


# ------------------------------------------------------------- startup shim
def test_startup_shim_roundtrip(tmp_path):
    launcher = tmp_path / "current" / "launchers" / "SpeechForge Tray.bat"
    folder = tmp_path / "Startup"
    assert not tray.startup_shim_installed(folder)
    assert not tray.remove_startup_shim(folder)
    path = tray.write_startup_shim(launcher, folder)
    assert path == folder / "SpeechForge.bat"
    text = path.read_text(encoding="utf-8")
    assert text.splitlines() == ["@echo off", f'start "SpeechForge" /min "{launcher}"']
    assert tray.startup_shim_installed(folder)
    assert tray.remove_startup_shim(folder)
    assert not path.exists()


def test_tray_launcher_follows_current_in_the_releases_layout(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_ROOT", str(tmp_path))
    assert tray.tray_launcher_path() == tmp_path / "current" / "launchers" / "SpeechForge Tray.bat"
    monkeypatch.delenv("SPEECHFORGE_ROOT")
    monkeypatch.setattr("speechforge.update.root_from_here", lambda: None)
    assert tray.tray_launcher_path() == tray.repo_root() / "launchers" / "SpeechForge Tray.bat"


# ------------------------------------------------------------- requests
def test_restart_and_stop_requests_create_the_state_dir(tmp_path):
    state = tmp_path / "data" / "state"
    assert tray.request_restart(state) == state / "restart.request"
    assert (state / "restart.request").read_text() == "tray\n"
    assert tray.request_stop(state) == state / "stop.request"
    assert (state / "stop.request").exists()


# ------------------------------------------------------------- adopt or launch
class FakeProc:
    """A supervisor child that exits when its ``stop.request`` appears."""

    def __init__(self, argv: list[str], state_dir: Path | None = None, **kw: Any) -> None:
        self.argv, self.kw, self.state_dir = argv, kw, state_dir
        self.pid = 4242
        self.returncode: int | None = None
        self.signals: list[Any] = []

    def poll(self) -> int | None:
        if (
            self.returncode is None
            and self.state_dir
            and (self.state_dir / "stop.request").exists()
        ):
            (self.state_dir / "stop.request").unlink()
            self.returncode = 0
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.poll() is None:
            raise subprocess.TimeoutExpired(self.argv, timeout or 0)
        return self.returncode or 0

    def send_signal(self, sig: Any) -> None:
        self.signals.append(sig)
        self.returncode = 0

    def terminate(self) -> None:
        self.returncode = -15


def make_app(config: Config, health, ready=None, **kw) -> tuple[TrayApp, list[FakeProc]]:
    spawned: list[FakeProc] = []

    def popen(argv, **pkw):
        proc = FakeProc(argv, config.state_dir, **pkw)
        spawned.append(proc)
        return proc

    app = TrayApp(
        config,
        create_icon=False,
        health=health,
        ready=ready or (lambda port, token: (None, "")),
        token=lambda: "not-a-real-token",
        popen=popen,
        confirm=lambda text: True,
        **kw,
    )
    return app, spawned


def test_decide():
    assert decide(True) == "adopt"
    assert decide(False) == "launch"


def test_adopts_a_server_that_answers(config):
    app, spawned = make_app(config, health=lambda port: dict(HEALTH_OK))
    assert app.start() == "adopt"
    assert app.adopted and not spawned
    assert app.stop_server() is False  # not ours to stop
    assert app.status.colour == "amber"  # up, /ready not answered by the fake


def test_launches_the_supervisor_when_nothing_answers(config):
    app, spawned = make_app(config, health=lambda port: None)
    assert app.start() == "launch"
    assert len(spawned) == 1
    proc = spawned[0]
    assert proc.argv[1:4] == ["-m", "speechforge", "supervise"]
    assert "python" in Path(proc.argv[0]).name.lower()
    env = proc.kw["env"]
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["SPEECHFORGE_DATA_DIR"] == str(config.data_dir)
    assert proc.kw["stdin"] is subprocess.DEVNULL
    if sys.platform == "win32":
        assert proc.kw["creationflags"] & tray.CREATE_NO_WINDOW
        assert proc.kw["creationflags"] & tray.CREATE_NEW_PROCESS_GROUP
    assert (config.logs_dir / tray.SUPERVISOR_LOG).exists()
    assert app.status == Status("amber", "starting...")
    assert app.start() == "owned"
    assert len(spawned) == 1
    app._close_log()


def test_stop_server_uses_the_file_request_then_signals(config):
    app, spawned = make_app(config, health=lambda port: None)
    app.start()
    assert app.stop_server(grace_s=0.01) is True
    assert spawned[0].returncode == 0
    assert spawned[0].signals == []  # the request was enough
    assert not (config.state_dir / "stop.request").exists()
    assert app.proc is None
    assert app.tick().colour == "grey"
    # A child that ignores the file gets the signal next.
    app2, spawned2 = make_app(config, health=lambda port: None)
    app2.start()
    spawned2[0].state_dir = None
    assert app2.stop_server(grace_s=0.01) is True
    assert spawned2[0].signals


def test_tick_uses_the_token_and_never_shows_it(config):
    seen: list[str] = []

    def ready(port, token):
        seen.append(token)
        return ready_body("ready", sessions_open=1, sessions=1), ""

    app, _ = make_app(config, health=lambda port: dict(HEALTH_OK), ready=ready)
    st = app.tick()
    assert st.colour == "blue"
    assert seen == ["not-a-real-token"]
    assert "not-a-real-token" not in st.tooltip
    # /ready is rate-limited to its own interval; /health is every tick.
    app.tick()
    assert seen == ["not-a-real-token"]


def test_tick_marks_a_vanished_server_red_and_a_failed_probe_never_raises(config):
    answers = {"body": dict(HEALTH_OK)}

    def health(port):
        if answers["body"] is None:
            raise OSError("boom")
        return answers["body"]

    app, _ = make_app(config, health=health, ready=lambda p, t: (ready_body("ready"), ""))
    assert app.tick().colour == "green"
    answers["body"] = None
    assert app.tick().colour == "red"


def test_tick_reports_a_supervisor_that_exited(config):
    app, spawned = make_app(config, health=lambda port: None)
    app.start()
    spawned[0].returncode = 78
    st = app.tick()
    assert st.colour == "red" and "78" in st.text
    app._close_log()


def test_toggle_startup_writes_and_removes_the_shim(config, tmp_path, monkeypatch):
    folder = tmp_path / "Startup"
    monkeypatch.setattr(tray, "startup_dir", lambda: folder)
    monkeypatch.setattr(tray, "tray_launcher_path", lambda: tmp_path / "SpeechForge Tray.bat")
    app, _ = make_app(config, health=lambda port: None)
    assert app.toggle_startup() is True
    assert (folder / "SpeechForge.bat").exists()
    assert app.toggle_startup() is False
    assert not (folder / "SpeechForge.bat").exists()


def test_supervise_argv_and_child_env(tmp_path):
    argv = tray.supervise_argv(tmp_path / "c.yaml", python="py")
    assert argv == ["py", "-m", "speechforge", "supervise", "--config", str(tmp_path / "c.yaml")]
    env = tray.child_env(tmp_path)
    assert env["SPEECHFORGE_DATA_DIR"] == str(tmp_path)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert os.environ.get("SPEECHFORGE_DATA_DIR") != "mutated"  # a copy, not os.environ


def test_tray_main_without_pystray_explains_and_exits(config, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("pystray", "PIL"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    shown: list[str] = []
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(tray, "_configure_logging", lambda logs_dir: None)
    monkeypatch.setattr(tray, "show_message", lambda text, title="": shown.append(text))
    assert tray.main(config) == 1
    assert shown and "--extra tray" in shown[0]


# ------------------------------------------------------------- the supervisor side
def test_supervisor_honours_stop_request(tmp_path, monkeypatch):
    class Child:
        def __init__(self, argv, **kw):
            self.argv, self.kw = argv, kw
            self.returncode = None
            self.signals = []

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            self.signals.append(sig)
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15

    children: list[Child] = []

    def popen(argv, **kw):
        children.append(Child(argv, **kw))
        return children[-1]

    monkeypatch.setattr(sup.subprocess, "Popen", popen)
    monkeypatch.setattr(sup.time, "sleep", lambda s: None)
    state = tmp_path / "state"
    state.mkdir()
    (state / "stop.request").write_text("stale")  # cleared at start, must not stop us at once
    (state / "crashloop.json").write_text("{}")

    def ask_stop():
        # The stale file is gone before the first child starts; write the real one.
        for _ in range(200):
            if children:
                break
            threading.Event().wait(0.01)
        tray.request_stop(state)  # atomic, like the tray itself

    # supervise() installs its own signal handlers; put the test runner's back after.
    import signal

    sigs = [getattr(signal, n) for n in ("SIGINT", "SIGTERM", "SIGBREAK") if hasattr(signal, n)]
    previous = {s: signal.getsignal(s) for s in sigs}
    try:
        threading.Thread(target=ask_stop, daemon=True).start()
        assert sup.supervise(1, state, probe_s=1000, boot_grace_s=1000) == 0
    finally:
        for s, handler in previous.items():
            if handler is not None:
                signal.signal(s, handler)
    assert len(children) == 1 and children[0].signals
    assert not (state / "stop.request").exists()
    assert not (state / "crashloop.json").exists()
