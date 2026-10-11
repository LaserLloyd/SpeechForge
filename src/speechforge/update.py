"""Blue/green update and rollback (SPEC-speechforge §10.4).

Layout (``install.ps1`` creates it)::

    <root>\\releases\\<tag>\\     a clone at that tag + its own .venv (uv sync --locked)
    <root>\\current            junction (Windows) / symlink -> releases\\<tag>
    <root>\\data\\              shared: config, env file, models (by lock hash), logs, state

``speechforge update --check [--source URL|PATH]`` lists the source's release
tags (``v*``, newest last) against the running one. The source is a git URL or
a local bundle/clone; it defaults to the ``origin`` of the running release and
then to the public repository. ``--apply`` then:

1. clones the tag into ``releases\\<tag>`` (``git clone --branch <tag> --depth 1``,
   after checking the tag exists at the source; never ``git pull`` in place);
2. ``uv sync --locked --extra gpu`` there; fetches any model file its lock adds
   (shared files are verified by hash, not re-downloaded);
3. starts it as a **canary** on loopback ``:1237`` with leases off (the live
   process holds the card's lease; the canary shares the card for the test)
   and runs the gate over sf.1: ``ready`` (which means the self-test passed:
   golden STT, TTFA, cancel, round trip), a TTS round trip and a cancel;
4. only then waits for the live sessions to drain (<= 15 s), switches
   ``current`` and asks the supervisor to restart;
5. watches the switched server for 2 minutes (a session must reach
   ``ready``); a failure switches ``current`` back and restarts again;
6. keeps the two most recent releases.

A failed canary leaves ``current`` untouched. Every subprocess is a fixed argv.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

CANARY_PORT = 1237
POST_SWITCH_S = 120.0
DEFAULT_SOURCE = "https://github.com/LaserLloyd/SpeechForge.git"
_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|[\w.-]+@[\w.-]+:)", re.IGNORECASE)
_TAG_RE = re.compile(r"^v(\d+(?:\.\d+)*)(.*)$")


def release_from_here() -> Path:
    """The checkout this code runs from (``<root>/releases/<tag>`` or a plain clone)."""
    return Path(__file__).resolve().parents[2]


def root_from_here() -> Path | None:
    """``<root>`` when this code runs from ``<root>/releases/<tag>``."""
    rel = release_from_here()
    if rel.parent.name == "releases":
        return rel.parent.parent
    return None


def run(argv: list[str], cwd: Path | None = None, timeout: float = 1800) -> str:
    out = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,  # noqa: S603
        timeout=timeout,
    )
    return out.stdout


def is_url(source: str) -> bool:
    """A git URL (``https://``, ``ssh://``, ``git://``, ``user@host:path``) as opposed
    to a local bundle file or clone directory."""
    return bool(_URL_RE.match(source.strip()))


def version_key(tag: str) -> tuple[int, ...]:
    """Sort key for ``v<major>.<minor>.<patch>[-pre]`` tags: numeric, a pre-release
    suffix sorts before the final release of the same number."""
    m = _TAG_RE.match(tag)
    if not m:
        return ()
    nums = [int(x) for x in m.group(1).split(".")]
    while len(nums) < 3:
        nums.append(0)
    return (*nums, 0 if m.group(2) else 1)


def release_tags(lines: str) -> list[str]:
    """The ``v*`` tags in ``git ls-remote --tags`` output, oldest first (version
    order, not text order: v0.10.0 is newer than v0.9.0)."""
    found: set[str] = set()
    for ln in lines.splitlines():
        if "refs/tags/" not in ln:
            continue
        tag = ln.split("refs/tags/", 1)[1].strip().removesuffix("^{}")
        if _TAG_RE.match(tag):
            found.add(tag)
    return sorted(found, key=version_key)


def source_tags(source: str) -> list[str]:
    """Release tags at ``source`` (a git URL, or a bundle/clone path): one
    ``git ls-remote --tags``, nothing fetched."""
    return release_tags(run(["git", "ls-remote", "--tags", source], timeout=120))


def default_source(release: Path | None = None) -> str:
    """Where updates come from when ``--source`` is not given: the ``origin`` of
    the running release when it has one that still exists (a URL, or a local
    bundle/clone that is still there), else the public repository."""
    rel = release or release_from_here()
    try:
        url = run(["git", "-C", str(rel), "remote", "get-url", "origin"], timeout=30).strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return DEFAULT_SOURCE
    if not url:
        return DEFAULT_SOURCE
    if is_url(url) or Path(url).exists():
        return url
    return DEFAULT_SOURCE


def clone_argv(source: str, tag: str, target: Path) -> list[str]:
    """One release = one shallow clone of one tag. Never ``git pull`` (S17)."""
    return ["git", "clone", "--quiet", "--branch", tag, "--depth", "1", source, str(target)]


def current_target(root: Path) -> Path | None:
    link = root / "current"
    try:
        return Path(os.path.realpath(link)) if link.exists() else None
    except OSError:
        return None


def switch(root: Path, target: Path) -> None:
    link = root / "current"
    if sys.platform == "win32":
        import _winapi  # type: ignore[import-not-found]

        if link.exists() or os.path.lexists(link):
            os.rmdir(link)  # removes the junction, never the target
        _winapi.CreateJunction(str(target), str(link))
    else:
        tmp = root / "current.new"
        if os.path.lexists(tmp):
            tmp.unlink()
        tmp.symlink_to(target, target_is_directory=True)
        os.replace(tmp, link)


def venv_python(release: Path) -> Path:
    win = release / ".venv" / "Scripts" / "python.exe"
    return win if win.exists() else release / ".venv" / "bin" / "python"


async def gate(url: str, token: str, voice: bytes) -> dict[str, Any]:
    """The canary/post-switch gate over sf.1. Raises on failure."""
    from speechforge.bench import cancel_once, speak, upload_voice
    from speechforge.client import RigClient

    c = RigClient(url, token)
    try:
        ready = await c.connect(ready_timeout=180)
        if ready.get("proto") != "sf.1":
            raise RuntimeError("canary did not speak sf.1")
        vid = await upload_voice(c, voice)
        r = await speak(c, 1, 1, vid, "Checking the new release.", 0)
        stop_ms = await cancel_once(c, 2, 2, vid)
        if stop_ms > 50:
            raise RuntimeError(f"cancel took {stop_ms} ms")
        if r["ttfa_ms"] > 250:
            raise RuntimeError(f"TTFA {r['ttfa_ms']:.0f} ms")
        return {"ttfa_ms": round(r["ttfa_ms"]), "cancel_ms": stop_ms}
    finally:
        await c.close()


def apply(
    root: Path,
    source: str,
    tag: str | None,
    *,
    token: str,
    voice: bytes,
    live_port: int,
    data_dir: Path,
) -> int:
    tags = source_tags(source)
    if not tags:
        print(f"no release tags (v*) at {source}")
        return 1
    if tag and tag not in tags:
        print(f"tag {tag} is not at {source} (have: {', '.join(tags)})")
        return 1
    tag = tag or tags[-1]
    target = root / "releases" / tag
    previous = current_target(root)
    if previous is not None and previous == target.resolve():
        print(f"{tag} is already current")
        return 0
    if target.exists():
        shutil.rmtree(target)
    print(f"[1/6] cloning {tag} from {source}")
    run(clone_argv(source, tag, target))
    print("[2/6] uv sync --locked --extra gpu")
    run(["uv", "sync", "--locked", "--extra", "gpu", "--extra", "dev"], cwd=target)
    py = str(venv_python(target))
    env = dict(os.environ, SPEECHFORGE_DATA_DIR=str(data_dir))
    subprocess.run([py, "-m", "speechforge", "fetch-models"], check=True, env=env)  # noqa: S603
    print(f"[3/6] canary on 127.0.0.1:{CANARY_PORT}")
    canary = subprocess.Popen(  # noqa: S603
        [
            py,
            "-m",
            "speechforge",
            "serve",
            "--port",
            str(CANARY_PORT),
            "--host",
            "127.0.0.1",
            "--no-lease",
        ],
        env=env,
    )
    try:
        time.sleep(3)
        result = asyncio.run(gate(f"ws://127.0.0.1:{CANARY_PORT}/v1/realtime", token, voice))
        print(f"      canary passed: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"      canary FAILED ({type(exc).__name__}: {exc}); current is untouched")
        return 1
    finally:
        canary.terminate()
        try:
            canary.wait(30)
        except subprocess.TimeoutExpired:
            canary.kill()
    print("[4/6] draining live sessions (<= 15 s)")
    deadline = time.time() + 15
    while time.time() < deadline and _sessions_open(live_port, token) > 0:
        time.sleep(1)
    print("[5/6] switching current and restarting")
    switch(root, target)
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    (data_dir / "state" / "restart.request").write_text(tag)
    print(f"[6/6] watching the switched server for {POST_SWITCH_S:.0f} s")
    time.sleep(15)
    try:
        result = asyncio.run(
            asyncio.wait_for(
                gate(f"ws://127.0.0.1:{live_port}/v1/realtime", token, voice), POST_SWITCH_S
            )
        )
        print(f"      live passed: {result}")
    except Exception as exc:  # noqa: BLE001
        print(f"      post-switch check FAILED ({type(exc).__name__}); rolling back")
        if previous is not None:
            switch(root, previous)
            (data_dir / "state" / "restart.request").write_text("rollback")
        return 1
    prune(root, keep={target.resolve(), previous.resolve() if previous else target.resolve()})
    return 0


def _sessions_open(port: int, token: str) -> int:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/ready", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as r:  # noqa: S310
            return int(json.loads(r.read()).get("sessions_open", 0))
    except Exception:  # noqa: BLE001 - 503 when unloaded: nobody is talking
        return 0


def prune(root: Path, keep: set[Path]) -> None:
    rel = root / "releases"
    for d in sorted(rel.iterdir(), key=lambda p: p.stat().st_mtime):
        if d.resolve() not in keep:
            shutil.rmtree(d, ignore_errors=True)


def rollback(root: Path, data_dir: Path) -> int:
    cur = current_target(root)
    others = sorted(
        (d for d in (root / "releases").iterdir() if d.resolve() != cur),
        key=lambda p: p.stat().st_mtime,
    )
    if not others:
        print("no previous release to roll back to")
        return 1
    switch(root, others[-1])
    (data_dir / "state").mkdir(parents=True, exist_ok=True)
    (data_dir / "state" / "restart.request").write_text("rollback")
    print(f"current -> {others[-1].name}; restart requested")
    return 0
