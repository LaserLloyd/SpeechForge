"""Model files: ``models.lock`` (URL with the pinned revision, size, sha256 per
file), verified download, and verification at load (SPEC-speechforge §7.1).

* **Download** streams each file to ``<dest>.partial``, hashing as it goes, and
  renames only when size and sha256 match the lock. A mismatch deletes the
  partial and fails loudly. Standard library only (no huggingface_hub cache:
  ONNX Runtime refuses external-data files behind the cache's symlinks).
* **Load** (:func:`verify_for_load`) checks every file the configured models
  need: present, size equal to the lock, and a stamp (``<data>/state/model-stamps/``,
  JSON ``{size, mtime_ns, sha256, at}``; kept out of the models directory, which
  may be read-only or shared) whose size and mtime still match.
  A stamp older than ``models.full_verify_days`` (7) triggers a full re-hash.
  No stamp at all = full hash once (the first load after a manual copy).

Groups name what a configuration needs: ``tts-<dtype>`` (Chatterbox),
``tts-qwen3`` / ``tts-qwen3-0.6b`` (Qwen3-TTS, phase 2), ``tts-dots`` /
``tts-dots-stts`` (dots.tts, phase 2), ``stt-fp32`` or ``stt-int8`` (Parakeet),
``stt-nemotron`` or ``stt-nemotron-int8`` (Nemotron streaming, ``stt.engine:
nemotron``), ``eou-gpu``/``eou-cpu`` (Smart Turn), ``vad`` (Silero).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from importlib import resources
from pathlib import Path
from typing import Any

LOCK_NAME = "models.lock"
CHUNK = 1 << 20


class ModelError(RuntimeError):
    pass


def load_lock() -> dict[str, Any]:
    text = resources.files("speechforge").joinpath(LOCK_NAME).read_text(encoding="utf-8")
    return json.loads(text)


def tts_groups(config: Any) -> list[str]:
    """The model group(s) the configured TTS engine loads from disk."""
    tts = config.tts
    if tts.engine == "qwen3-tts":
        size = tts.qwen3.model
        return ["tts-qwen3" if size == "1.7b" else f"tts-qwen3-{size}"]
    if tts.engine == "dots-tts":
        dots = tts.dots
        if dots.backend == "http":
            return []  # the weights live with the server, not in models_dir
        return ["tts-dots-stts" if dots.double_streaming else "tts-dots"]
    if tts.engine == "drama3-tts":
        return []  # installed by audio.cpp's model manager (engine-setup), not models.lock
    return [f"tts-{config.models.tts_dtype}"]


def tts_group_for(config: Any) -> str:
    """The first TTS group (callers that expect exactly one)."""
    groups = tts_groups(config)
    return groups[0] if groups else ""


def stt_group(config: Any) -> str:
    if config.stt.engine == "nemotron":
        q = config.stt.nemotron.quantization
        return "stt-nemotron-int8" if q == "int8" else "stt-nemotron"
    return "stt-int8" if config.models.stt_quantization == "int8" else "stt-fp32"


def groups_for(config: Any) -> list[str]:
    g = [*tts_groups(config), stt_group(config), "vad"]
    if config.eou.enabled:
        g.append("eou-gpu" if config.gpu.provider == "cuda" else "eou-cpu")
    return g


def files_for(config: Any, groups: list[str] | None = None) -> list[dict[str, Any]]:
    want = set(groups or groups_for(config))
    return [f for f in load_lock()["files"] if want & set(f["groups"])]


def path_of(config: Any, what: str) -> Path:
    """Paths the loaders need that are not a model directory."""
    root = config.models_dir
    if what == "smart-turn":
        name = "gpu" if config.gpu.provider == "cuda" else "cpu"
        return root / "smart-turn" / f"smart-turn-v3.2-{name}.onnx"
    if what == "silero-vad":
        return root / "silero-vad" / "silero_vad.onnx"
    raise KeyError(what)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _stamp_path(stamps: Path, dest: str) -> Path:
    return stamps / (dest.replace("/", "__") + ".json")


def _write_stamp(stamps: Path, dest: str, path: Path, sha: str) -> None:
    st = path.stat()
    stamps.mkdir(parents=True, exist_ok=True)
    _stamp_path(stamps, dest).write_text(
        json.dumps(
            {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha, "at": time.time()}
        )
    )


def verify_file(path: Path, entry: dict[str, Any], *, full_after_s: float, stamps: Path) -> str:
    """'ok' | 'rehashed'. Raises :class:`ModelError` on any mismatch."""
    if not path.is_file():
        raise ModelError(f"missing {entry['dest']} (run `speechforge fetch-models`)")
    st = path.stat()
    if st.st_size != entry["size"]:
        raise ModelError(f"{entry['dest']}: size {st.st_size} != lock {entry['size']}")
    try:
        stamp = json.loads(_stamp_path(stamps, entry["dest"]).read_text())
    except (OSError, ValueError):
        stamp = None
    fresh = (
        stamp
        and stamp.get("size") == st.st_size
        and stamp.get("mtime_ns") == st.st_mtime_ns
        and stamp.get("sha256") == entry["sha256"]
        and time.time() - float(stamp.get("at", 0)) < full_after_s
    )
    if fresh:
        return "ok"
    sha = sha256_file(path)
    if sha != entry["sha256"]:
        raise ModelError(f"{entry['dest']}: sha256 does not match models.lock")
    _write_stamp(stamps, entry["dest"], path, sha)
    return "rehashed"


def verify(config: Any, groups: list[str] | None = None) -> dict[str, str]:
    root = config.models_dir
    full_after = float(config.models.full_verify_days) * 86400
    stamps = config.state_dir / "model-stamps"
    return {
        f["dest"]: verify_file(root / f["dest"], f, full_after_s=full_after, stamps=stamps)
        for f in files_for(config, groups)
    }


def verify_for_load(config: Any) -> None:
    verify(config)


def missing(config: Any, groups: list[str] | None = None) -> list[str]:
    root = config.models_dir
    return [
        f["dest"]
        for f in files_for(config, groups)
        if not (root / f["dest"]).is_file() or (root / f["dest"]).stat().st_size != f["size"]
    ]


def download(entry: dict[str, Any], dest: Path, *, stamps: Path, opener: Any = None) -> None:
    """Stream ``entry['url']`` to ``dest`` via ``dest.partial``; verify; rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".partial")
    h = hashlib.sha256()
    size = 0
    req = urllib.request.Request(entry["url"], headers={"User-Agent": "speechforge"})
    open_ = opener or urllib.request.urlopen
    try:
        with open_(req, timeout=60) as r, part.open("wb") as out:  # noqa: S310 - pinned URL
            while chunk := r.read(CHUNK):
                h.update(chunk)
                size += len(chunk)
                out.write(chunk)
        if size != entry["size"] or h.hexdigest() != entry["sha256"]:
            raise ModelError(
                f"{entry['dest']}: downloaded bytes do not match models.lock "
                f"(size {size} vs {entry['size']})"
            )
        os.replace(part, dest)
        _write_stamp(stamps, entry["dest"], dest, entry["sha256"])
    finally:
        if part.exists():
            part.unlink()


def fetch(
    config: Any, *, groups: list[str] | None = None, force: bool = False, opener: Any = None
) -> list[Path]:
    got: list[Path] = []
    root = config.models_dir
    stamps = config.state_dir / "model-stamps"
    for f in files_for(config, groups):
        dest = root / f["dest"]
        if not force and dest.is_file() and dest.stat().st_size == f["size"]:
            try:
                verify_file(dest, f, full_after_s=float("inf"), stamps=stamps)
                continue
            except ModelError:
                pass
        download(f, dest, stamps=stamps, opener=opener)
        got.append(dest)
    return got


def engine_shas(config: Any) -> dict[str, str]:
    """sha8 per loaded model (``ready.engines``): a hash of its files' hashes."""
    out: dict[str, str] = {}
    for f in files_for(config):
        key = f["dest"].split("/", 1)[0]
        out[key] = hashlib.sha256((out.get(key, "") + f["sha256"]).encode()).hexdigest()
    return {k: v[:8] for k, v in out.items()}
