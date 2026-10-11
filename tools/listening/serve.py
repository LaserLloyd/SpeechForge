"""Step 3: the rating page the family opens on the home network.

    uv run python -m tools.listening.serve --set <blind-set> [--host <lan-or-tailnet-address>] \\
        [--port 8181] [--owner-label "Dad"]

A standard-library ``http.server`` on loopback by default. One page, inline
CSS/JS, no CDN, no request to anything but itself. The rater types a nickname;
for every line the page plays A/B(/C/D) in a random order with a play button
each, asks "how natural" (1-5) per sample, "which sounds most like <owner>"
(forced choice) and an optional comment. Each answer is one JSON line appended
to ``<set>/ratings/<nickname>.jsonl``; a rater who comes back is taken to the
first unanswered line.

The server reads the set directory only: ``manifest.json`` and ``NN/<label>.wav``.
It never sees ``key.json`` (``make_set`` wrote it elsewhere), so nothing it can
serve names an engine.

Keyboard: ``1``-``4`` play a sample (and select it), ``space`` replays the selected one,
``q w e r t`` rate the selected sample 1-5, ``m`` marks it "most like", ``Enter`` submits.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from tools.listening import kit

AUDIO_RE = re.compile(r"^/audio/(\d{2})/([A-D])\.wav$")
#: The single page (inline CSS/JS, no external request); ``__OWNER__`` is filled in at start.
PAGE_FILE = Path(__file__).resolve().parent / "page.html"
MAX_BODY = 64 * 1024


class RatingStore:
    """Append-only JSONL per rater, with the progress needed to resume."""

    def __init__(self, set_dir: Path):
        self.dir = set_dir / "ratings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, nick: str) -> Path:
        return self.dir / f"{nick}.jsonl"

    def done(self, nick: str) -> list[str]:
        p = self.path(nick)
        if not p.exists():
            return []
        ids = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            try:
                ids.append(str(json.loads(raw)["line"]))
            except (ValueError, KeyError, TypeError):
                continue
        return sorted(set(ids))

    def append(self, nick: str, record: dict) -> int:
        with self._lock:
            with self.path(nick).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            return len(self.done(nick))


def validate_rating(body: dict, manifest: dict) -> tuple[dict | None, str]:
    """The answer as stored, or an error message."""
    nick = kit.safe_nick(str(body.get("rater", "")))
    if not nick:
        return None, "nickname: 1-32 letters, digits, - or _"
    line_id = str(body.get("line", ""))
    line = next((ln for ln in manifest["lines"] if ln["id"] == line_id), None)
    if line is None:
        return None, "unknown line"
    labels = sorted(line["samples"])
    mos = body.get("mos")
    if not isinstance(mos, dict) or sorted(mos) != labels:
        return None, "a 1-5 rating is needed for every sample"
    try:
        mos = {k: int(v) for k, v in mos.items()}
    except (TypeError, ValueError):
        return None, "ratings must be integers"
    if any(not 1 <= v <= 5 for v in mos.values()):
        return None, "ratings must be 1-5"
    most_like = body.get("most_like")
    if most_like not in labels:
        return None, "pick the sample that sounds most like the voice owner"
    order = body.get("order")
    if not isinstance(order, list) or sorted(map(str, order)) != labels:
        order = labels
    comment = str(body.get("comment", "") or "")[:500]
    try:
        spent = max(0, min(int(body.get("spent_ms", 0) or 0), 3_600_000))
    except (TypeError, ValueError):
        spent = 0
    return {
        "rater": nick,
        "line": line_id,
        "mos": mos,
        "most_like": most_like,
        "order": [str(x) for x in order],
        "comment": comment,
        "spent_ms": spent,
        "t": int(time.time()),
    }, ""


def make_handler(set_dir: Path, manifest: dict, store: RatingStore, owner_label: str):
    # The label lands inside a <script>: json.dumps leaves "<" alone, so a label
    # containing "</script>" would end the script early. Escape it (valid JSON).
    owner_js = json.dumps(owner_label).replace("<", "\\u003c").replace(">", "\\u003e")
    page = PAGE_FILE.read_text(encoding="utf-8").replace("__OWNER__", owner_js)

    class Handler(BaseHTTPRequestHandler):
        server_version = "SpeechForgeListening/1"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet: no paths or nicknames on the console
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "media-src 'self'; connect-src 'self'; img-src 'self' data:",
            )
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, status: int, data: dict) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def do_HEAD(self) -> None:  # noqa: N802 - http.server API
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            if parts.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parts.path == "/api/manifest":
                self._json(HTTPStatus.OK, manifest)
                return
            if parts.path == "/api/progress":
                nick = kit.safe_nick(parse_qs(parts.query).get("rater", [""])[0])
                if not nick:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "bad nickname"})
                    return
                self._json(HTTPStatus.OK, {"rater": nick, "done": store.done(nick)})
                return
            m = AUDIO_RE.match(parts.path)
            if m:
                rel = f"{m.group(1)}/{m.group(2)}.wav"
                known = any(
                    s["file"] == rel for ln in manifest["lines"] for s in ln["samples"].values()
                )
                p = set_dir / m.group(1) / f"{m.group(2)}.wav"
                if not known or not p.is_file():
                    self._json(HTTPStatus.NOT_FOUND, {"error": "no such sample"})
                    return
                self._send(HTTPStatus.OK, p.read_bytes(), "audio/wav")
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/api/rate":
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if not 0 < n <= MAX_BODY:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "bad body"})
                return
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self._json(HTTPStatus.BAD_REQUEST, {"error": "not a JSON object"})
                return
            record, err = validate_rating(body, manifest)
            if record is None:
                self._json(HTTPStatus.BAD_REQUEST, {"error": err})
                return
            done = store.append(record["rater"], record)
            self._json(HTTPStatus.OK, {"ok": True, "done": done, "total": len(manifest["lines"])})

    return Handler


def classify_host(host: str) -> str:
    """loopback | tailnet | other (for the warning printed at start)."""
    h = host.strip("[]")
    if h in ("localhost",):
        return "loopback"
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return "other"
    if ip.is_loopback:
        return "loopback"
    if ip in ipaddress.ip_network("100.64.0.0/10") or ip in ipaddress.ip_network(
        "fd7a:115c:a1e0::/48"
    ):
        return "tailnet"
    return "other"


def make_server(set_dir: Path, host: str, port: int, owner_label: str) -> ThreadingHTTPServer:
    manifest = kit.json_load(set_dir / "manifest.json")
    store = RatingStore(set_dir)
    handler = make_handler(set_dir, manifest, store, owner_label)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", required=True, type=Path, help="the blind set from make_set")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default loopback)")
    ap.add_argument("--port", type=int, default=8181)
    ap.add_argument("--owner-label", default="the owner", help="who the voice belongs to, as shown")
    args = ap.parse_args(argv)
    if not (args.set / "manifest.json").exists():
        raise SystemExit(f"{args.set} has no manifest.json (run make_set first)")
    if (args.set / "key.json").exists() or any(args.set.rglob("key.json")):
        raise SystemExit("key.json is inside the set directory; move it out before serving")
    kind = classify_host(args.host)
    if kind == "other":
        print(
            "WARNING: this address is neither loopback nor a tailnet address. The page and the\n"
            "         family's audio travel UNENCRYPTED on that network and anyone on it can\n"
            "         open the page and submit ratings. Use it on a trusted home LAN only.",
            file=sys.stderr,
        )
    elif kind == "tailnet":
        print("note: tailnet address; traffic is encrypted by the tailnet.", file=sys.stderr)
    server = make_server(args.set, args.host, args.port, args.owner_label)
    host, port = server.server_address[0], server.server_address[1]
    shown = f"[{host}]" if ":" in host else host
    print(f"listening test: open  http://{shown}:{port}/  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
