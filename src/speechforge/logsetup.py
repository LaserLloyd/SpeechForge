"""JSON-lines logs, rotated 10 MB x 5, with secrets redacted at the formatter
(SPEC-speechforge §9). Log lines carry ``turn_id``, durations, codes and sizes
-- never audio, transcripts, reply text or word timings (§8.4)."""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import time
from pathlib import Path
from typing import Any

#: Values that look like a bearer/vacate token or an Authorization header.
_REDACT = [
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)((?:vacate_token|x-sf-vacate-token|authorization|token)[\"'=:\s]+)"
        r"[A-Za-z0-9._~+/=-]{8,}"
    ),
]
_EXTRA_KEYS = ("turn_id", "session", "ctx", "gen", "code", "ms", "bytes", "state", "device")


def redact(text: str) -> str:
    for rx in _REDACT:
        text = rx.sub(r"\1[redacted]", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for key in _EXTRA_KEYS:
            if hasattr(record, key):
                out[key] = getattr(record, key)
        if record.exc_info:
            out["exc"] = redact(self.formatException(record.exc_info))[-4000:]
        return json.dumps(out, ensure_ascii=False)


def setup(logs_dir: Path | None, *, to_file: bool) -> None:
    handlers: list[logging.Handler] = []
    console = logging.StreamHandler()
    console.setFormatter(JsonFormatter())
    handlers.append(console)
    if logs_dir is not None and to_file:
        logs_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            logs_dir / "speechforge.jsonl", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(JsonFormatter())
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    import faulthandler

    faulthandler.enable()  # stacks only: no heap, no audio (§8.4)
