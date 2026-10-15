"""``speechforge`` -- serve, supervise, fetch/verify models, doctor, token, bench, update.

speechforge serve [--port N --host IP --no-lease]   run the server (foreground)
speechforge supervise                               the restart loop (the launcher runs this)
speechforge tray                                    system-tray controller: adopts a running
                                                    server or launches the supervisor (S23)
speechforge fetch-models [--group tts-fp16 ...]     download what models.lock lists, verified
speechforge verify-models [--full]                  re-check sizes/stamps (or every sha256)
speechforge doctor                                  driver, CUDA, ORT, cards, models, bind, token,
                                                    firewall, StudioForge (names only, no secrets)
speechforge token new|rotate|fingerprint
speechforge bench --url ws://HOST:1236/v1/realtime [--corpus DIR] [--soak S]
speechforge update --check|--apply|--rollback [--source URL|PATH] [--tag T]
                                                    source: a git URL or a bundle/clone path
                                                    (default: the release's origin, else GitHub)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from speechforge import __version__
from speechforge.config import (
    DATA_DIR_ENV,
    ENV_FILE_NAME,
    TOKEN_ENV,
    Config,
    load_config,
    validate_for_serving,
)
from speechforge.errors import ConfigError
from speechforge.supervise import EXIT_CONFIG

ORT_MIN, ORT_MAX = (1, 28), (1, 32)


def _load(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if getattr(args, "port", None):
        config.server.port = args.port
    if getattr(args, "host", None):
        config.server.hosts = [args.host]
    if getattr(args, "no_lease", False):
        config.studioforge.lease_mode = "off"
    return config


def cmd_serve(args: argparse.Namespace) -> int:
    from speechforge import logsetup

    try:
        config = _load(args)
        validate_for_serving(config)
    except ConfigError as exc:
        logsetup.setup(None, to_file=False)
        logging.getLogger("speechforge").error("%s", exc)
        return EXIT_CONFIG
    logsetup.setup(config.logs_dir, to_file=True)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    cache = config.data_dir / "cuda-cache"
    os.environ.setdefault("CUDA_CACHE_PATH", str(cache))  # JIT'd PTX survives restarts (§7.4)
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 2**30))
    import uvicorn

    from speechforge.app import create_app

    log = logging.getLogger("speechforge")
    log.info(
        "SpeechForge %s on %s:%d (devices %s, lease %s)",
        __version__,
        ",".join(config.server.hosts),
        config.server.port,
        config.gpu.devices,
        config.studioforge.lease_mode,
    )
    sockets = []
    for host in config.server.hosts:
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        sockets.append(socket.create_server((host, config.server.port), family=fam))
    uconf = uvicorn.Config(
        create_app(config),
        log_level="warning",
        access_log=False,
        workers=1,
        ws_max_size=64 * 1024,
        ws_ping_interval=None,
        ws_per_message_deflate=False,
        timeout_graceful_shutdown=30,
        ssl_certfile=str(config.server.tls_cert) if config.tls else None,
        ssl_keyfile=str(config.server.tls_key) if config.tls else None,
    )
    uvicorn.Server(uconf).run(sockets=sockets)
    return 0


def cmd_supervise(args: argparse.Namespace) -> int:
    from speechforge import logsetup
    from speechforge.supervise import supervise
    from speechforge.update import root_from_here

    try:
        config = _load(args)
        validate_for_serving(config)
    except ConfigError as exc:
        logsetup.setup(None, to_file=False)
        logging.getLogger("speechforge").error("%s", exc)
        return EXIT_CONFIG
    logsetup.setup(config.logs_dir, to_file=True)
    extra = ["--config", str(args.config)] if args.config else []
    root = (
        Path(os.environ["SPEECHFORGE_ROOT"])
        if os.environ.get("SPEECHFORGE_ROOT")
        else root_from_here()
    )
    return supervise(config.server.port, config.state_dir, extra_args=extra, root=root)


def cmd_tray(args: argparse.Namespace) -> int:
    """The system-tray controller (S23). Needs the ``tray`` extra; it says so."""
    from speechforge.tray import main as tray_main

    try:
        config = _load(args)
    except ConfigError as exc:
        print(f"[FAIL] config            {exc}", file=sys.stderr)
        return EXIT_CONFIG
    return tray_main(config, args.config)


def cmd_engine_setup(args: argparse.Namespace) -> int:
    """Build and install an add-on engine with the engine's own tooling."""
    config = load_config(args.config)
    if args.engine == "drama3":
        from speechforge.engines import drama3_tts

        try:
            out = drama3_tts.setup(
                config,
                root=Path(args.root) if args.root else None,
                build=not args.no_build,
                install_model=not args.no_model,
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            print(f"engine-setup drama3 failed: {exc}", file=sys.stderr)
            return 1
        ok = out["server_exe_exists"] and out["model_exists"]
        print("ready" if ok else "incomplete: " + json.dumps(out, indent=2))
        return 0 if ok else 1
    print(f"unknown engine {args.engine!r}", file=sys.stderr)
    return 2


def cmd_fetch(args: argparse.Namespace) -> int:
    from speechforge import models

    config = load_config(args.config)
    got = models.fetch(config, groups=args.group or None, force=args.force)
    print(f"fetched {len(got)} file(s) into {config.models_dir}")
    left = models.missing(config, args.group or None)
    if left:
        print("still missing:", *left, sep="\n  ")
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from speechforge import models

    config = load_config(args.config)
    if args.full:
        config.models.full_verify_days = 1e-9
    try:
        res = models.verify(config)
    except models.ModelError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"ok: {len(res)} file(s), {sum(v == 'rehashed' for v in res.values())} re-hashed")
    return 0


def _line(ok: bool | None, label: str, detail: str) -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "info"}[ok]
    print(f"[{mark}] {label:18} {detail}")
    return ok is not False


def cmd_doctor(args: argparse.Namespace) -> int:
    """Everything that would stop a start or a fast session. Names, never values."""
    from speechforge import models
    from speechforge.auth import TokenStore

    good = True
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[FAIL] config            {exc}")
        return EXIT_CONFIG
    _line(None, "version", f"{__version__} (python {platform.python_version()})")
    _line(None, "layout", _layout_detail(config))
    if sys.platform == "win32":
        build = sys.getwindowsversion().build  # type: ignore[attr-defined]
        good &= _line(
            build >= 22000,
            "windows",
            f"build {build} (Windows 11 = 22000+; Windows 10 gets no new NVIDIA drivers)",
        )
    try:
        validate_for_serving(config)
        good &= _line(True, "bind", f"{', '.join(config.server.hosts)}:{config.server.port}")
    except ConfigError as exc:
        good &= _line(False, "bind", str(exc))
    good &= _line(
        TokenStore(config.env_file).configured, "token", f"{TOKEN_ENV} in {ENV_FILE_NAME}"
    )
    left = models.missing(config)
    good &= _line(
        not left,
        "models",
        "all present (sizes match models.lock)" if not left else f"missing {', '.join(left[:3])}",
    )
    if not left and args.full:
        try:
            models.verify(config)
            _line(True, "models sha256", "verified")
        except models.ModelError as exc:
            good &= _line(False, "models sha256", str(exc))
    try:
        import onnxruntime as ort

        ver = tuple(int(x) for x in ort.__version__.split(".")[:2])
        provs = ort.get_available_providers()
        good &= _line(
            ORT_MIN <= ver < ORT_MAX,
            "onnxruntime",
            f"{ort.__version__} (window >={ORT_MIN[0]}.{ORT_MIN[1]},"
            f"<{ORT_MAX[0]}.{ORT_MAX[1]}); providers {', '.join(provs)}",
        )
        if config.gpu.provider == "cuda":
            good &= _line(
                "CUDAExecutionProvider" in provs,
                "CUDA provider",
                "present" if "CUDAExecutionProvider" in provs else "MISSING: install the gpu extra",
            )
    except ImportError:
        good &= _line(False, "onnxruntime", "not installed (uv sync --extra gpu)")
    if config.gpu.provider == "cuda":
        good &= _doctor_nvml(config)
    if config.studioforge.lease_mode == "hold":
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{config.studioforge.url}/api/leases",  # noqa: S310
                timeout=3,
            ) as r:
                rows = json.loads(r.read()).get("leases", [])
            mine = [r for r in rows if r.get("holder") == config.studioforge.holder]
            others = [
                r
                for r in rows
                if set(r.get("devices") or []) & set(config.gpu.devices)
                and r.get("holder") != config.studioforge.holder
            ]
            _line(
                True,
                "StudioForge",
                f"reachable; {len(rows)} lease(s), {len(mine)} ours, "
                f"{len(others)} other(s) on our cards",
            )
        except Exception as exc:  # noqa: BLE001
            _line(
                None,
                "StudioForge",
                f"unreachable ({type(exc).__name__}); "
                f"on_unreachable={config.studioforge.on_unreachable}",
            )
    vdir = str(config.voices_dir)
    good &= _line(
        "onedrive" not in vdir.lower(),
        "voices dir",
        f"{vdir} ({'inside OneDrive: move it' if 'onedrive' in vdir.lower() else 'ok'})",
    )
    if sys.platform == "win32":
        good &= _doctor_windows(config)
    print("\nOK" if good else "\nPROBLEMS FOUND")
    return 0 if good else 1


def _layout_detail(config: Config) -> str:
    """Which layout this is running from, and where the data lives.

    ``releases`` = install.ps1's ``<root>\\releases\\<tag>`` + ``current`` junction;
    ``checkout`` = a plain clone (a developer layout), where the data dir comes from
    ``SPEECHFORGE_DATA_DIR`` or defaults to ``<repo>/data`` and there is no junction
    to look for.
    """
    from speechforge.update import current_target, release_from_here, root_from_here

    data_from = "from " + DATA_DIR_ENV if os.environ.get(DATA_DIR_ENV, "").strip() else "default"
    root = (
        Path(os.environ["SPEECHFORGE_ROOT"])
        if os.environ.get("SPEECHFORGE_ROOT")
        else root_from_here()
    )
    if root is None:
        return f"checkout {release_from_here()}; data {config.data_dir} ({data_from})"
    cur = current_target(root)
    current = cur.name if cur else "MISSING (install.ps1 or update --apply creates it)"
    return f"releases under {root}; current -> {current}; data {config.data_dir} ({data_from})"


def _doctor_nvml(config: Config) -> bool:
    good = True
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            drv = drv.decode() if isinstance(drv, bytes) else drv
            major = int(str(drv).split(".")[0])
            good &= _line(
                major >= config.gpu.min_driver,
                "driver",
                f"{drv} (floor R{config.gpu.min_driver} for CUDA 13)",
            )
            cuda = pynvml.nvmlSystemGetCudaDriverVersion()
            _line(None, "CUDA (driver)", f"{cuda // 1000}.{cuda % 1000 // 10}")
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                mj, mn = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                mine = i in config.gpu.devices
                match = not config.gpu.expect_name or config.gpu.expect_name.lower() in name.lower()
                if mine:
                    good &= match
                _line(
                    None if not mine else match,
                    f"CUDA {i}{' *' if mine else ''}",
                    f"{name} sm_{mj}{mn}, {mem.free / 2**30:.1f}/{mem.total / 2**30:.1f} GiB "
                    f"free{'' if match or not mine else ' <- does not match gpu.expect_name'}",
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        good &= _line(False, "NVML", f"unavailable ({type(exc).__name__})")
    return good


def _doctor_windows(config: Config) -> bool:
    good = True

    def q(argv: list[str]) -> str:
        try:
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=10,  # noqa: S603
                check=False,
            ).stdout
        except Exception:  # noqa: BLE001
            return ""

    rule = q(["netsh", "advfirewall", "firewall", "show", "rule", "name=SpeechForge"])
    good &= _line(
        "SpeechForge" in rule,
        "firewall rule",
        "present" if "SpeechForge" in rule else "missing (install.ps1 adds it)",
    )
    dumps = q(
        [
            "reg",
            "query",
            r"HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting"
            r"\LocalDumps\python.exe",
            "/v",
            "DumpType",
        ]
    )
    good &= _line(
        "0x1" in dumps,
        "crash dumps",
        "mini dumps (no heap)"
        if "0x1" in dumps
        else "LocalDumps DumpType=1 not set (install.ps1 sets it)",
    )
    return good


def cmd_token(args: argparse.Namespace) -> int:
    from speechforge.auth import write_token

    config = load_config(args.config)
    if args.action in ("new", "rotate"):
        token = write_token(config.env_file, rotate=args.action == "rotate")
        if args.action == "new":
            # Shown once, to the terminal, for copying to the box; never logged.
            print(token)
            print(
                f"\nwrote {TOKEN_ENV} to {config.env_file} (fingerprint "
                f"{hashlib.sha256(token.encode()).hexdigest()[:8]}). Copy the line above to the "
                "box's speechforge.env now; it is not shown again.",
                file=sys.stderr,
            )
        return 0
    from speechforge.config import parse_env_file

    token = parse_env_file(config.env_file).get(TOKEN_ENV, "") or os.environ.get(TOKEN_ENV, "")
    if not token:
        print(f"{TOKEN_ENV} is not set", file=sys.stderr)
        return 1
    print(hashlib.sha256(token.encode()).hexdigest()[:8])
    return 0


def _bench_voice(args: argparse.Namespace) -> bytes:
    if args.voice:
        return Path(args.voice).read_bytes()
    if not args.cpu_voice:
        raise SystemExit(
            "bench needs --voice FILE.npz, or --cpu-voice MODELS_DIR to build the "
            "synthetic test voice on this machine (the reference clip never goes "
            "to the rig)"
        )
    from speechforge.engines.chatterbox import ChatterboxEngine
    from speechforge.selftest import golden_clip

    clip, rate = golden_clip()
    from speechforge.audio import reference_clip

    eng = ChatterboxEngine(
        Path(args.cpu_voice) / "chatterbox-turbo",
        dtype=args.cpu_dtype,
        provider="cpu",
        cpu_threads=2,
    )
    prepared, _ = reference_clip(clip, rate)
    return eng.build_voice(prepared).to_npz()


def cmd_bench(args: argparse.Namespace) -> int:
    from speechforge.auth import TokenStore
    from speechforge.bench import main_bench

    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        try:
            config = load_config(args.config)
            ts = TokenStore(config.env_file)
            ts.configured  # noqa: B018 - loads the file
            token = ts._current
        except Exception:  # noqa: BLE001
            token = ""
    return main_bench(
        args.url,
        token,
        _bench_voice(args),
        n=args.n,
        corpus=Path(args.corpus) if args.corpus else None,
        soak_s=args.soak,
        out=Path(args.out) if args.out else None,
    )


def cmd_update(args: argparse.Namespace) -> int:
    from speechforge import update

    config = load_config(args.config)
    root = Path(args.root) if args.root else update.root_from_here()
    source = args.source or update.default_source()
    if args.check:
        # Needs no layout: a plain checkout can ask what is published too.
        tags = update.source_tags(source)
        cur = update.current_target(root) if root else None
        newest = f"; newest {tags[-1]}" if tags else ""
        print(
            f"current: {cur.name if cur else '-'}; source {source}; "
            f"release tags: {', '.join(tags) or '-'}{newest}"
        )
        return 0
    if root is None:
        print("not running from a releases/<tag> layout; pass --root")
        return 1
    if args.rollback:
        return update.rollback(root, config.data_dir)
    from speechforge.auth import TokenStore

    ts = TokenStore(config.env_file)
    ts.configured  # noqa: B018
    if not args.voice and not args.cpu_voice:
        # The gate's voice: the shipped synthetic clip through this machine's own
        # speech encoder, CPU provider (not a person; D3 does not apply).
        args.cpu_voice, args.cpu_dtype = str(config.models_dir), config.models.tts_dtype
    voice = _bench_voice(args)
    return update.apply(
        root,
        source,
        args.tag,
        token=ts._current,
        voice=voice,
        live_port=config.server.port,
        data_dir=config.data_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="speechforge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn: Any, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", type=Path, default=None, help="config.yaml (default: data dir)")
        p.set_defaults(fn=fn)
        return p

    p = add("serve", cmd_serve, "run the server")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--host", default="")
    p.add_argument("--no-lease", action="store_true")
    add("supervise", cmd_supervise, "run the server under the restart loop")
    add("tray", cmd_tray, "system-tray controller (adopts a running server or launches supervise)")
    p = add("fetch-models", cmd_fetch, "download the models in models.lock")
    p.add_argument(
        "--group", action="append", help="lock group(s), e.g. tts-fp16 (default: what config needs)"
    )
    p.add_argument("--force", action="store_true")
    p = add("engine-setup", cmd_engine_setup,
            "build and install an add-on TTS engine with its own tooling (drama3)")
    p.add_argument("engine", choices=["drama3"])
    p.add_argument("--root", default="", help="where to clone/build (default <data>/engines/...)")
    p.add_argument("--no-build", action="store_true", help="skip the build step")
    p.add_argument("--no-model", action="store_true", help="skip the model install")
    p = add("verify-models", cmd_verify, "verify model files against models.lock")
    p.add_argument("--full", action="store_true", help="re-hash every file")
    p = add("doctor", cmd_doctor, "check this machine")
    p.add_argument("--full", action="store_true", help="also re-hash every model file")
    p = add("token", cmd_token, "manage the bearer token")
    p.add_argument("action", choices=["new", "rotate", "fingerprint"])
    for name, fn, help_ in (
        ("bench", cmd_bench, "time a running server against the gates"),
        ("update", cmd_update, "blue/green update or rollback"),
    ):
        p = add(name, fn, help_)
        p.add_argument("--voice", default="", help="voice conditionals .npz")
        p.add_argument(
            "--cpu-voice", default="", help="models dir: build the synthetic test voice on this CPU"
        )
        p.add_argument("--cpu-dtype", default="q8")
        if name == "bench":
            p.add_argument("--url", default="ws://127.0.0.1:1236/v1/realtime")
            p.add_argument("--corpus", default="")
            p.add_argument("--soak", type=float, default=0.0)
            p.add_argument("-n", type=int, default=10)
            p.add_argument("--out", default="")
        else:
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--check", action="store_true")
            g.add_argument("--apply", action="store_true")
            g.add_argument("--rollback", action="store_true")
            p.add_argument(
                "--source",
                "--bundle",
                dest="source",
                default="",
                help="git URL, or a bundle/clone path (default: the release's origin, "
                "else the public repository); --bundle is the old name",
            )
            p.add_argument("--tag", default=None, help="release tag (default: the newest v*)")
            p.add_argument("--root", default="")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
