"""The family listening-test kit (tools/listening, docs/LISTENING-TEST.md): blinding,
loudness matching, the rating server, the scorer's maths, WER and similarity, all
offline on synthetic audio and an ephemeral loopback port."""

from __future__ import annotations

import http.client
import json
import struct
import threading
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
from tools.listening import kit, make_set, render_set, score, serve, similarity, wer

ENGINES = ["chatterbox", "qwen3-tts", "dots-tts"]


def tone(seconds: float, hz: float, amp: float, rate: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.fixture
def lines() -> list[str]:
    return kit.read_lines(None)


@pytest.fixture
def renders(tmp_path: Path, lines: list[str]) -> Path:
    """Three fake engines, every line rendered, at deliberately different levels/lengths."""
    root = tmp_path / "renders"
    for k, eng in enumerate(ENGINES):
        for n in range(1, len(lines) + 1):
            audio = tone(0.3 + 0.05 * k, 220 + 110 * k, 0.1 + 0.3 * k)
            kit.write_wav(root / eng / f"{kit.line_name(n)}.wav", audio, 24000)
    return root


@pytest.fixture
def blind(tmp_path: Path, renders: Path, lines: list[str]):
    set_dir, key_dir = tmp_path / "set", tmp_path / "key"
    manifest, key = make_set.build_set(
        [(e, renders / e) for e in ENGINES], set_dir, key_dir, seed=7, lines=lines
    )
    return set_dir, key_dir, manifest, key


# --------------------------------------------------------------------- lines
def test_default_lines_are_twenty_varied(lines):
    assert len(lines) == 20
    counts = [len(x.split()) for x in lines]
    assert min(counts) >= 4 and max(counts) <= 25
    assert any(x.endswith("?") for x in lines)
    assert any("[laugh]" in x for x in lines)
    assert any(any(ch.isdigit() for ch in x) for x in lines)
    assert any("Peter Piper" in x for x in lines)


def test_lines_file_override(tmp_path):
    p = tmp_path / "l.txt"
    p.write_text("# c\n\nOne.\nTwo two.\n", encoding="utf-8")
    assert kit.read_lines(p) == ["One.", "Two two."]
    with pytest.raises(ValueError):
        kit.read_lines(tmp_path.joinpath("e.txt").write_text("") or tmp_path / "e.txt")


# --------------------------------------------------------------------- loudness
def test_rms_normalisation_hits_target_or_peak_guard():
    quiet, s = kit.normalise_rms(tone(0.5, 440, 0.05), target_dbfs=-23.0)
    assert abs(kit.dbfs(kit.rms(quiet)) + 23.0) < 0.1
    assert s["peak_guard"] == 0.0
    # A sine at -23 dBFS RMS peaks at -20 dBFS; ask for -3 dBFS RMS and the -1 dBFS peak
    # guard has to step in.
    loud, s2 = kit.normalise_rms(tone(0.5, 440, 0.9), target_dbfs=-3.0, peak_dbfs=-1.0)
    assert s2["peak_guard"] == 1.0
    assert float(np.max(np.abs(loud))) <= 10 ** (-1.0 / 20) + 1e-6
    silence, s3 = kit.normalise_rms(np.zeros(100, np.float32))
    assert not silence.any() and s3["gain_db"] == 0.0


def test_streamed_wav_header_with_placeholder_sizes_is_read():
    pcm = (np.arange(100, dtype="<i2") * 10).tobytes()
    hdr = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )
    out, rate = kit.streamed_wav_to_pcm(hdr + pcm)
    assert rate == 24000 and out == pcm
    with pytest.raises(ValueError):
        kit.streamed_wav_to_pcm(b"nope")


# --------------------------------------------------------------------- make_set
def test_make_set_blinds_shuffles_and_is_deterministic(tmp_path, renders, lines, blind):
    set_dir, key_dir, manifest, key = blind
    assert sorted(manifest["labels"]) == ["A", "B", "C"]
    assert len(manifest["lines"]) == 20
    text = json.dumps(manifest)
    for eng in ENGINES:
        assert eng not in text
    for ln in manifest["lines"]:
        for label, s in ln["samples"].items():
            assert s["file"] == f"{ln['id']}/{label}.wav"
            assert (set_dir / s["file"]).exists()
    # the key maps every label to an engine, with a per-line shuffle
    assignments = [tuple(key["lines"][kit.line_name(n)][lb] for lb in "ABC") for n in range(1, 21)]
    assert all(sorted(a) == sorted(ENGINES) for a in assignments)
    assert len(set(assignments)) > 1
    # the audio under each label is the engine the key says (levels differ per engine)
    for n in (1, 2, 3):
        stem = kit.line_name(n)
        for label, eng in key["lines"][stem].items():
            a, _ = kit.read_wav(set_dir / stem / f"{label}.wav")
            src, _ = kit.read_wav(renders / eng / f"{stem}.wav")
            assert len(a) == len(src)
            # every sample sits at the common level
            assert abs(kit.dbfs(kit.rms(a)) - kit.TARGET_RMS_DBFS) < 0.2
    # the key stays out of the set; ratings dir is ready
    assert not list(set_dir.rglob("key.json"))
    assert (set_dir / "ratings").is_dir()
    # same seed -> same set
    again_set, again_key = tmp_path / "set2", tmp_path / "key2"
    m2, k2 = make_set.build_set(
        [(e, renders / e) for e in ENGINES], again_set, again_key, seed=7, lines=lines
    )
    assert m2 == manifest and k2["lines"] == key["lines"]
    m3, k3 = make_set.build_set(
        [(e, renders / e) for e in ENGINES],
        tmp_path / "set3",
        tmp_path / "key3",
        seed=8,
        lines=lines,
    )
    assert k3["lines"] != key["lines"]


def test_make_set_refuses_bad_layouts(tmp_path, renders, lines):
    with pytest.raises(ValueError, match="inside"):
        make_set.build_set(
            [(e, renders / e) for e in ENGINES[:2]],
            tmp_path / "s",
            tmp_path / "s" / "k",
            seed=1,
            lines=lines,
        )
    with pytest.raises(ValueError, match="2-4"):
        make_set.build_set([("x", renders / "chatterbox")], tmp_path / "s", tmp_path / "k", seed=1)
    with pytest.raises(ValueError, match="differ"):
        make_set.build_set(
            [("x", renders / "chatterbox"), ("x", renders / "dots-tts")],
            tmp_path / "s",
            tmp_path / "k",
            seed=1,
        )
    assert make_set.engine_name("qwen=" + str(tmp_path / "nope")) == ("qwen", tmp_path / "nope")
    assert make_set.engine_name(str(renders / "dots-tts")) == ("dots-tts", renders / "dots-tts")


def test_make_set_skips_lines_an_engine_missed(tmp_path, renders, lines):
    (renders / "dots-tts" / "05.wav").unlink()
    m, k = make_set.build_set(
        [(e, renders / e) for e in ENGINES], tmp_path / "s", tmp_path / "k", seed=1, lines=lines
    )
    assert len(m["lines"]) == 19 and "05" not in k["lines"]


# --------------------------------------------------------------------- serve
class Server:
    def __init__(self, set_dir: Path, owner: str = "the owner"):
        self.srv = serve.make_server(set_dir, "127.0.0.1", 0, owner)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def request(self, method: str, path: str, body: dict | bytes | None = None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        c.request(method, path, body=data, headers=headers or {})
        r = c.getresponse()
        payload = r.read()
        c.close()
        return r.status, dict(r.getheaders()), payload

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def server(blind):
    set_dir, *_ = blind
    s = Server(set_dir, owner="Dad")
    yield s
    s.close()


def test_serve_page_manifest_and_audio(server, blind):
    set_dir, _key_dir, manifest, _key = blind
    status, headers, body = server.request("GET", "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    page = body.decode()
    assert '"Dad"' in page and "http://" not in page and "https://" not in page
    assert "cdn" not in page.lower() and "<script src" not in page
    assert "Content-Security-Policy" in headers
    status, _, body = server.request("GET", "/api/manifest")
    assert status == 200
    text = body.decode()
    assert json.loads(text) == manifest
    for eng in ENGINES:
        assert eng not in text
    status, headers, body = server.request("GET", "/audio/01/A.wav")
    assert status == 200 and headers["Content-Type"] == "audio/wav"
    assert body == (set_dir / "01" / "A.wav").read_bytes()
    assert headers["Cache-Control"] == "no-store"
    for bad in (
        "/audio/01/E.wav",
        "/audio/99/A.wav",
        "/audio/../key/key.json",
        "/key.json",
        "/manifest.json",
        "/ratings/x.jsonl",
    ):
        status, _, _ = server.request("GET", bad)
        assert status == 404, bad


def test_serve_rate_resume_and_validation(server, blind):
    set_dir, *_ = blind
    good = {
        "rater": "sam",
        "line": "01",
        "mos": {"A": 4, "B": 2, "C": 5},
        "most_like": "C",
        "order": ["B", "C", "A"],
        "comment": "C had a click | pipe",
        "spent_ms": 12345,
    }
    status, _, body = server.request("POST", "/api/rate", good)
    assert status == 200 and json.loads(body) == {"ok": True, "done": 1, "total": 20}
    # resume: progress lists the answered line; a second answer to the same line counts once
    status, _, body = server.request("GET", "/api/progress?rater=sam")
    assert json.loads(body) == {"rater": "sam", "done": ["01"]}
    server.request("POST", "/api/rate", {**good, "most_like": "A"})
    status, _, body = server.request("GET", "/api/progress?rater=sam")
    assert json.loads(body)["done"] == ["01"]
    rows = [json.loads(x) for x in (set_dir / "ratings" / "sam.jsonl").read_text().splitlines()]
    assert (
        len(rows) == 2 and rows[-1]["most_like"] == "A" and rows[0]["comment"].startswith("C had")
    )
    assert rows[0]["order"] == ["B", "C", "A"] and rows[0]["spent_ms"] == 12345
    assert not any(eng in json.dumps(rows) for eng in ENGINES)
    # validation
    for bad, why in (
        ({**good, "rater": "../x"}, "nickname"),
        ({**good, "rater": "a b"}, "nickname"),
        ({**good, "line": "77"}, "unknown line"),
        ({**good, "mos": {"A": 4, "B": 2}}, "every sample"),
        ({**good, "mos": {"A": 9, "B": 2, "C": 1}}, "1-5"),
        ({**good, "most_like": "D"}, "most like"),
    ):
        status, _, body = server.request("POST", "/api/rate", bad)
        assert status == 400 and why in json.loads(body)["error"], (bad, body)
    status, _, _ = server.request("POST", "/api/rate", b"not json", headers={"Content-Length": "8"})
    assert status == 400
    status, _, _ = server.request("POST", "/api/rate", b"x" * (serve.MAX_BODY + 1))
    assert status == 400
    status, _, _ = server.request("POST", "/api/other", good)
    assert status == 404
    status, _, _ = server.request("GET", "/api/progress?rater=bad%20nick")
    assert status == 400
    assert sorted(p.name for p in (set_dir / "ratings").iterdir()) == ["sam.jsonl"]


def test_serve_owner_label_cannot_close_the_script(blind):
    set_dir, *_ = blind
    s = Server(set_dir, owner="</script><script>alert(1)//")
    try:
        _status, _headers, body = s.request("GET", "/")
    finally:
        s.close()
    page = body.decode()
    assert "</script><script>alert" not in page
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)//" in page
    assert page.count("</script>") == 1


def test_classify_host():
    assert serve.classify_host("127.0.0.1") == "loopback"
    assert serve.classify_host("localhost") == "loopback"
    assert serve.classify_host("::1") == "loopback"
    assert serve.classify_host("100.64.0.0") == "tailnet"
    assert serve.classify_host("fd7a:115c:a1e0::1") == "tailnet"
    assert serve.classify_host("192.168.1.20") == "other"
    assert serve.classify_host("0.0.0.0") == "other"


# --------------------------------------------------------------------- score
def synthetic_ratings(key: dict, raters=("sam", "kim", "lee")) -> list[dict]:
    """Everyone rates qwen3-tts 5, chatterbox 3, dots-tts 1; everyone picks qwen3-tts
    except 'lee', who picks chatterbox on even lines."""
    mos_by_engine = {"qwen3-tts": 5, "chatterbox": 3, "dots-tts": 1}
    out = []
    for nick in raters:
        for stem, labels in key["lines"].items():
            inv = {eng: lbl for lbl, eng in labels.items()}
            pick = "chatterbox" if nick == "lee" and int(stem) % 2 == 0 else "qwen3-tts"
            out.append(
                {
                    "rater": nick,
                    "line": stem,
                    "mos": {lbl: mos_by_engine[eng] for lbl, eng in labels.items()},
                    "most_like": inv[pick],
                    "comment": "nice" if stem == "01" and nick == "sam" else "",
                }
            )
    return out


def test_score_maths_on_a_synthetic_set(blind):
    set_dir, key_dir, manifest, key = blind
    ratings = synthetic_ratings(key)
    joined = score.join(ratings, key)
    assert len(joined) == 60
    agg = score.aggregate(joined, key, n_boot=500, seed=1)
    e = agg["engines"]
    assert e["qwen3-tts"]["mos_mean"] == 5.0 and e["dots-tts"]["mos_mean"] == 1.0
    assert e["chatterbox"]["mos_ci"] == [3.0, 3.0]
    assert e["qwen3-tts"]["pref_wins"] == 50 and e["chatterbox"]["pref_wins"] == 10
    assert e["dots-tts"]["pref_rate"] == 0.0
    lo, hi = e["qwen3-tts"]["pref_ci"]
    assert lo <= 50 / 60 <= hi and lo > 0.6
    assert agg["ranked"][0] == "qwen3-tts" and agg["ranked"][-1] == "dots-tts"
    # per line / per rater
    assert agg["lines"]["02"]["picks"] == {"chatterbox": 1, "qwen3-tts": 2, "dots-tts": 0}
    assert agg["lines"]["02"]["majority"] == "qwen3-tts"
    assert agg["raters"]["sam"]["agreement"] == 1.0
    assert agg["raters"]["lee"]["agreement"] == 0.5
    assert agg["pairwise"]["kim / sam"]["agreement"] == 1.0
    assert agg["pairwise"]["kim / lee"]["agreement"] == 0.5
    assert agg["comments"] == [{"rater": "sam", "line": "01", "comment": "nice"}]
    report = score.render_report(
        agg,
        key,
        manifest,
        {"qwen3-tts": {"summary": {"ttfa_p50_ms": 120.0, "ttfa_p90_ms": 180.0, "rtf_mean": 0.2}}},
    )
    assert "| qwen3-tts | 5.0 |" in report and "50/60 (83%)" in report
    assert "120 / 180 ms | 0.20 |" in report and "Most often picked" in report
    assert "## Per line" in report and "## Raters" in report and "- line 01, sam: nice" in report


def test_score_helpers(tmp_path, blind):
    set_dir, key_dir, manifest, key = blind
    assert score.majority(["a", "b", "a"]) == "a"
    assert score.majority(["a", "b"]) is None and score.majority([]) is None
    lo, hi = score.bootstrap_ci([1, 2, 3, 4, 5] * 10, n_boot=300, seed=0)
    assert lo < 3.0 < hi and hi - lo < 1.5
    assert score.bootstrap_ci([2.0]) == (2.0, 2.0)
    assert all(x != x for x in score.bootstrap_ci([]))
    # last answer per (rater, line) wins; junk lines are skipped
    rdir = set_dir / "ratings"
    (rdir / "sam.jsonl").write_text(
        json.dumps(
            {"rater": "sam", "line": "01", "mos": {"A": 1, "B": 1, "C": 1}, "most_like": "A"}
        )
        + "\nnot json\n"
        + json.dumps(
            {"rater": "sam", "line": "01", "mos": {"A": 5, "B": 1, "C": 1}, "most_like": "B"}
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = score.load_ratings(set_dir)
    assert len(loaded) == 1 and loaded[0]["most_like"] == "B"
    # incomplete answers never reach the aggregate
    assert score.join([{"rater": "x", "line": "01", "mos": {"A": 1}, "most_like": "A"}], key) == []
    assert score.join([{"rater": "x", "line": "zz", "mos": {}, "most_like": "A"}], key) == []
    # the CLI end to end
    rc = score.main(
        [
            "--set",
            str(set_dir),
            "--key",
            str(key_dir),
            "--bootstrap",
            "100",
            "--json",
            str(tmp_path / "agg.json"),
        ]
    )
    assert rc == 0 and (set_dir / "report.md").exists() and (tmp_path / "agg.json").exists()


# --------------------------------------------------------------------- wer
def test_wer_normalisation_and_distance():
    assert wer.normalise_for_wer("It's 7:15 on the 14th. [laugh]") == (
        ["it's", "seven", "fifteen", "on", "the", "fourteenth"]
    )
    assert wer.normalise_for_wer("Platform 3 at 9:05, 4:30 and 8:00") == (
        "platform three at nine oh five four thirty and eight o'clock".split()  # noqa: SIM905
    )
    assert wer.normalise_for_wer("twenty-three degrees") == ["twenty", "three", "degrees"]
    assert wer.normalise_for_wer("“Hello”… — world") == ["hello", "world"]
    assert wer.number_words(0) == "zero" and wer.number_words(115) == "one hundred fifteen"
    assert wer.ordinal_words(20) == "twentieth" and wer.ordinal_words(21) == "twenty first"
    assert wer.wer("Sure, I can help with that.", "sure i can help with that") == 0.0
    assert wer.wer("Sure, I can help with that.", "Sure I can help with this.") == pytest.approx(
        1 / 6
    )
    assert wer.wer("a b c", "") == 1.0 and wer.wer("", "") == 0.0 and wer.wer("", "x") == 1.0
    assert wer.wer("The train leaves at 7:15.", "the train leaves at seven fifteen") == 0.0


def test_wer_against_a_fake_server(tmp_path, renders, lines, monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_TOKEN", "t" * 32)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"].startswith("Bearer ")
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                503, headers={"Retry-After": "0.01"}, json={"error": {"code": "not_ready"}}
            )
        n = int(request.headers.get("content-length", "0"))
        assert n > 44
        return httpx.Response(
            200, json={"text": lines[(calls["n"] - 2) % len(lines)].replace("[laugh]", "")}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    headers = kit.auth_headers("http://example.test:1236")
    res = wer.score_dir(
        client, "http://example.test:1236", headers, renders / "chatterbox", lines, retries=3
    )
    assert res["engine"] == "chatterbox" and len(res["lines"]) == 20 and res["wer_mean"] == 0.0
    assert calls["n"] == 21


# --------------------------------------------------------------------- render_set
def test_render_set_streams_retries_and_records_timing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPEECHFORGE_TOKEN", "t" * 32)
    seen = {"posts": 0, "bodies": []}
    pcm = tone(0.2, 440, 0.3)
    pcm16 = (pcm * 32767).astype("<i2").tobytes()
    header = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer " + "t" * 32
        if request.url.path == "/ready":
            return httpx.Response(200, json={"version": "0.2.0", "engines": {"tts": "fake"}})
        assert request.url.path == "/v1/audio/speech"
        seen["posts"] += 1
        body = json.loads(request.content)
        seen["bodies"].append(body)
        if seen["posts"] == 1:
            return httpx.Response(
                503, headers={"Retry-After": "0.01"}, json={"error": {"code": "not_ready"}}
            )
        return httpx.Response(200, content=header + pcm16, headers={"Content-Type": "audio/wav"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    out = tmp_path / "renders" / "fake"
    render = render_set.render_all(
        "http://example.test:1236",
        "fake",
        "sha256:abc",
        ["One.", "Two two?"],
        out,
        seed=100,
        retries=5,
        timeout_s=5,
        client=client,
        sleep=lambda s: None,
    )
    assert seen["posts"] == 3
    assert seen["bodies"][1] == {
        "model": "fake",
        "input": "One.",
        "voice": "sha256:abc",
        "response_format": "wav",
        "stream": True,
        "seed": 101,
    }
    assert sorted(p.name for p in out.iterdir()) == ["01.wav", "02.wav", "render.json"]
    with wave.open(str(out / "01.wav")) as w:
        assert w.getframerate() == 24000 and w.getnframes() == len(pcm)
    saved = json.loads((out / "render.json").read_text())
    assert (
        saved == render
        and saved["engine"] == "fake"
        and saved["server"]["engines"] == {"tts": "fake"}
    )
    assert [r["n"] for r in saved["lines"]] == [1, 2]
    assert all(r["ttfa_ms"] >= 0 and r["wall_ms"] >= r["ttfa_ms"] for r in saved["lines"])
    assert saved["lines"][0]["audio_s"] == 0.2 and saved["summary"]["n"] == 2
    assert "token" not in json.dumps(saved).lower()


def test_render_set_needs_the_token_off_loopback(monkeypatch):
    monkeypatch.delenv("SPEECHFORGE_TOKEN", raising=False)
    assert kit.auth_headers("http://127.0.0.1:1236") == {}
    assert kit.auth_headers("http://localhost:1236") == {}
    with pytest.raises(SystemExit):
        kit.auth_headers("http://example.test:1236")
    monkeypatch.setenv("SPEECHFORGE_TOKEN", "secret-value-123456")
    assert kit.auth_headers("http://example.test:1236") == {
        "Authorization": "Bearer secret-value-123456"
    }


def test_retry_after_parsing():
    assert kit.retry_after_s({"Retry-After": "3"}) == 3.0
    assert kit.retry_after_s({"Retry-After": "nonsense"}) == 2.0
    assert kit.retry_after_s({}) == 2.0
    assert kit.retry_after_s({"Retry-After": "900"}) == 60.0
    waits = []
    out = kit.with_retries(
        iter([503, 503, 200]).__next__,
        retries=5,
        is_retryable=lambda r: r == 503,
        headers_of=lambda r: {"Retry-After": "0.5"},
        sleep=waits.append,
    )
    assert out == 200 and waits == [0.5, 0.5]
    out = kit.with_retries(
        lambda: 503,
        retries=2,
        is_retryable=lambda r: r == 503,
        headers_of=lambda r: {},
        sleep=lambda s: None,
    )
    assert out == 503


# --------------------------------------------------------------------- similarity
def test_fbank_shape_and_normalisation():
    audio = tone(1.0, 300, 0.2, rate=16000)
    f = similarity.fbank(audio)
    assert f.shape == (1 + (16000 - 400) // 160, 80) and f.dtype == np.float32
    assert np.allclose(f.mean(axis=0), 0.0, atol=1e-4)
    low = similarity.fbank(audio, cmn=False).mean(axis=0).argmax()
    high = similarity.fbank(tone(1.0, 3000, 0.2, rate=16000), cmn=False).mean(axis=0).argmax()
    # mel bins are ordered by frequency: a 300 Hz tone peaks well below a 3 kHz one
    assert low < 20 < high < 80
    assert similarity.fbank(np.zeros(100, np.float32)).shape[0] == 1
    fb = similarity.mel_filterbank()
    assert fb.shape == (80, 257) and (fb >= 0).all() and fb[:, -1].sum() == 0


class FakeEmbedder:
    """Embeds the mean level and the dominant frequency, so engines come out apart."""

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        spec = np.abs(np.fft.rfft(audio16k))
        peak = float(spec.argmax()) / len(spec)
        return np.array([1.0, peak * 10, kit.rms(audio16k) * 10], np.float32)


def test_similarity_scores_and_deltas(tmp_path, renders):
    emb = FakeEmbedder()
    assert similarity.cosine(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(1.0)
    assert similarity.cosine(np.array([1.0, 0]), np.array([0, 1.0])) == 0.0
    assert similarity.cosine(np.zeros(2), np.ones(2)) == 0.0
    ref = tmp_path / "ref.wav"
    kit.write_wav(ref, tone(0.5, 220, 0.1, rate=16000), 16000)  # sounds like 'chatterbox'
    r = similarity.reference_embedding(emb, [ref])
    results = similarity.compare([similarity.score_dir(emb, r, renders / e) for e in ENGINES], None)
    by = {x["engine"]: x for x in results}
    assert by["chatterbox"]["delta"] == 0.0 and by["chatterbox"]["baseline"] == "chatterbox"
    assert len(by["chatterbox"]["lines"]) == 20
    assert by["chatterbox"]["mean"] > by["qwen3-tts"]["mean"] > by["dots-tts"]["mean"]
    assert by["dots-tts"]["delta"] < 0
    table = similarity.format_table(results, 0.05)
    assert "chatterbox" in table and "delta" in table
    explicit = similarity.compare([dict(x) for x in results], "dots-tts")
    assert all(x["baseline"] == "dots-tts" for x in explicit)
