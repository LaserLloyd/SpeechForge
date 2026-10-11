"""Chunk pipeline arithmetic, voice parsing/store, configuration."""

from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest

from speechforge.config import Config, load_config, load_env_file, validate_for_serving
from speechforge.engines.chatterbox import SAMPLES_PER_TOKEN, ChatterboxEngine, slice_prompt
from speechforge.errors import BadRequest, ConfigError, NotFound
from speechforge.pipeline import TRAILING_SILENCE, ChunkPlanConfig, Plan, pipelined
from speechforge.voices import VoiceStore, parse_npz, parse_voice_id, sha256_hex
from tests.conftest import make_voice


# --------------------------------------------------------------------- plan
def test_plan_first_chunk_then_regular_chunks_with_context():
    p = Plan(first=15, chunk=25, left=8, lookahead=3)
    assert p.next_chunk(17, done=False) is None  # 15 + 3 lookahead not there yet
    assert p.next_chunk(18, done=False) == (0, 18, 0, 3)
    assert p.next_chunk(40, done=False) is None
    assert p.next_chunk(43, done=False) == (7, 43, 8, 3)
    assert p.next_chunk(50, done=True) == (32, 50, 8, 0)  # remainder, no lookahead
    assert p.next_chunk(50, done=True) is None


def test_plan_short_sentence_is_one_chunk():
    p = Plan(first=15, chunk=25, left=8, lookahead=3)
    assert p.next_chunk(9, done=True) == (0, 9, 0, 0)


class TokenEngine:
    """Real ``decode_chunk`` slicing over a fake decoder: every token decodes to
    960 samples of a value derived from the token, so seams are checkable."""

    def __init__(self, n_tokens: int, delay_s: float = 0.0) -> None:
        self.n_tokens = n_tokens
        self.delay_s = delay_s
        self.stopped_at: int | None = None
        eng = ChatterboxEngine.__new__(ChatterboxEngine)
        eng._dec_lock = threading.Lock()

        def run(component, feeds):
            # Like the real decoder: audio for the tokens after the prompt only.
            n = feeds["speech_tokens"].shape[1] - feeds["speaker_features"].shape[1] // 2
            return [np.full((1, n * SAMPLES_PER_TOKEN), 0.1, np.float32)]

        eng._run = run
        self._eng = eng

    def stream_tokens(self, text, voice, stop=None, seed=None, *, prefix=None):
        i = 0
        try:
            for i in range(self.n_tokens):
                if stop is not None and stop.is_set():
                    return
                if self.delay_s:
                    time.sleep(self.delay_s)
                yield 100 + i
        finally:
            if i < self.n_tokens - 1:
                self.stopped_at = i

    def decode_chunk(self, tokens, voice, **kw):
        return self._eng.decode_chunk(tokens, voice, **kw)


@pytest.mark.parametrize("n", [1, 9, 18, 19, 43, 100, 257])
def test_pipelined_emits_every_token_once_plus_the_closing_silence(n):
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    stats: dict = {}
    pieces = list(pipelined(TokenEngine(n), "x", voice, plan=ChunkPlanConfig(), stats=stats))
    total = sum(len(p) for p in pieces)
    # every speech token once, plus the last chunk's 3 trailing-silence tokens
    assert total == (n + 3) * SAMPLES_PER_TOKEN
    assert stats["tokens"] == n and stats["first_chunk_ms"] is not None
    assert all(p.dtype == np.dtype("<i2") for p in pieces)
    # a constant signal crossfaded equal-power stays within sqrt(2) of itself
    allpcm = np.concatenate(pieces).astype(np.float32) / 32767
    assert allpcm.max() <= 0.1 * 1.42 + 1e-3


def test_pipelined_halts_the_lm_when_the_consumer_leaves():
    eng = TokenEngine(5000, delay_s=0.001)
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    gen = pipelined(eng, "x", voice, plan=ChunkPlanConfig())
    next(gen)
    gen.close()
    assert eng.stopped_at is not None and eng.stopped_at < 5000


def test_pipelined_honours_stop():
    eng = TokenEngine(5000, delay_s=0.001)
    stop = threading.Event()
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    gen = pipelined(eng, "x", voice, plan=ChunkPlanConfig(), stop=stop)
    next(gen)
    stop.set()
    rest = list(gen)
    assert eng.stopped_at is not None and len(rest) <= 2


def test_slice_prompt_keeps_cond_and_cuts_decoder_prompt():
    v = make_voice(prompt=250)
    s = slice_prompt(v, 4.0)
    assert s.prompt_token.shape == (1, 100) and s.speaker_features.shape == (1, 200, 80)
    assert s.cond_emb is v.cond_emb
    assert slice_prompt(v, 20.0) is v


# --------------------------------------------------------------------- voices
def test_parse_npz_both_naming_schemes():
    v = make_voice()
    assert parse_npz(v.to_npz()).prompt_token.dtype == np.int64
    buf = io.BytesIO()
    np.savez(
        buf,
        audio_features=v.cond_emb,
        audio_tokens=v.prompt_token,
        speaker_embeddings=v.speaker_embeddings,
        speaker_features=v.speaker_features,
    )
    assert parse_npz(buf.getvalue()).cond_emb.shape == (1, 20, 1024)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("speaker_embeddings"),
        lambda d: d.__setitem__("cond_emb", np.zeros((1, 20, 512), np.float32)),
        lambda d: d.__setitem__("prompt_token", np.full((1, 10), 9999, np.int64)),
        lambda d: d.__setitem__("speaker_features", np.full((1, 10, 80), np.nan, np.float32)),
        lambda d: d.__setitem__("cond_emb", np.zeros((2, 20, 1024), np.float32)),
    ],
)
def test_parse_npz_rejects_bad_tensors(mutate):
    v = make_voice()
    d = {
        "cond_emb": v.cond_emb,
        "prompt_token": v.prompt_token,
        "speaker_embeddings": v.speaker_embeddings,
        "speaker_features": v.speaker_features,
    }
    mutate(d)
    buf = io.BytesIO()
    np.savez(buf, **d)
    with pytest.raises(BadRequest):
        parse_npz(buf.getvalue())


def test_parse_npz_refuses_pickles_and_junk():
    buf = io.BytesIO()
    np.savez(buf, cond_emb=np.array([object()], dtype=object))
    with pytest.raises(BadRequest):
        parse_npz(buf.getvalue())
    with pytest.raises(BadRequest):
        parse_npz(b"not a zip")


def test_voice_ids():
    h = "a" * 64
    assert parse_voice_id(f"sha256:{h}") == h == parse_voice_id(h)
    for bad in ("", "sha256:xyz", "A" * 64, "../" + h):
        with pytest.raises(BadRequest):
            parse_voice_id(bad)


def test_store_roundtrip_lru_and_prune(tmp_path):
    store = VoiceStore(tmp_path, max_files=2, max_memory=1)
    ids = []
    for i in range(3):
        data = make_voice(i).to_npz()
        vid = sha256_hex(data)
        store.put(vid, data)
        ids.append(vid)
        time.sleep(0.01)
    assert store.count() == 2 and not store.exists(ids[0])
    assert store.get(ids[2]).cond_emb.shape == (1, 20, 1024)
    with pytest.raises(NotFound):
        store.get(ids[0])
    assert store.delete(ids[1]) and not store.delete(ids[1])


# --------------------------------------------------------------------- config
def test_defaults_are_loopback_class_2_device_3(tmp_path):
    c = Config(data_dir=tmp_path)
    assert c.server.hosts == ["127.0.0.1"] and c.studioforge.priority == 2
    assert c.gpu.devices == [3] and c.lease_idle_ttl_s == 1200
    assert c.vacate_url == "http://127.0.0.1:1236/internal/vacate"
    assert c.studioforge.on_unreachable == "proceed_if_free"
    assert c.voices.allow_reference_clip is False


@pytest.mark.parametrize(
    ("hosts", "token", "tls", "ok"),
    [
        (["127.0.0.1"], False, False, True),
        (["127.0.0.1", "100.101.102.103"], True, False, True),
        (["100.101.102.103"], False, False, False),  # tailnet but no token
        (["0.0.0.0"], True, False, False),  # never every interface
        (["::"], True, True, False),
        (["192.168.50.4"], True, False, False),  # LAN needs TLS
        (["192.168.50.4"], True, True, True),
        (["gpu-host.example"], True, True, False),  # names are not bind addresses
    ],
)
def test_bind_rules(tmp_path, monkeypatch, hosts, token, tls, ok):
    if token:
        monkeypatch.setenv("SPEECHFORGE_TOKEN", "x" * 43)
    else:
        monkeypatch.delenv("SPEECHFORGE_TOKEN", raising=False)
    server = {"hosts": hosts}
    if tls:
        server.update(tls_cert=str(tmp_path / "c.pem"), tls_key=str(tmp_path / "k.pem"))
    c = Config(data_dir=tmp_path, server=server)
    if ok:
        validate_for_serving(c)
    else:
        with pytest.raises(ConfigError):
            validate_for_serving(c)


def test_env_file_names_only_and_no_override(tmp_path, monkeypatch):
    monkeypatch.delenv("SPEECHFORGE_TOKEN", raising=False)
    monkeypatch.setenv("SPEECHFORGE_SF_KEY", "from-env")
    (tmp_path / "speechforge.env").write_text(
        "# comment\nexport SPEECHFORGE_TOKEN='abc'\nSPEECHFORGE_SF_KEY=file\nPATH=/evil\n"
    )
    names = load_env_file(tmp_path / "speechforge.env")
    assert names == ["SPEECHFORGE_TOKEN"]
    import os

    assert os.environ["SPEECHFORGE_SF_KEY"] == "from-env" and os.environ["PATH"] != "/evil"
    monkeypatch.delenv("SPEECHFORGE_TOKEN")


def test_config_file_and_unknown_keys(tmp_path):
    (tmp_path / "config.yaml").write_text("gpu:\n  devices: [2]\n  expect_name: '3090'\n")
    assert load_config(data_dir=tmp_path).gpu.devices == [2]
    (tmp_path / "config.yaml").write_text("gpu:\n  device: [2]\n")
    with pytest.raises(ConfigError):
        load_config(data_dir=tmp_path)
    (tmp_path / "config.yaml").write_text("data_dir: /elsewhere\n")
    with pytest.raises(ConfigError):
        load_config(data_dir=tmp_path)


def test_config_example_is_valid_and_documents_every_section():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config.example.yaml").read_text())
    assert set(raw) == set(Config.model_fields) - {"data_dir"}
    Config(data_dir=root, **raw)


def test_install_and_launchers_reference_real_commands():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "install.ps1").read_text() + "".join(
        p.read_text() for p in (root / "launchers").glob("*.bat")
    )
    for cmd in (
        "fetch-models",
        "token new",
        "doctor",
        "supervise",
        "update --apply",
        "update --rollback",
        "--locked",
    ):
        assert cmd in text, cmd
    from speechforge.__main__ import main

    with __import__("pytest").raises(SystemExit):
        main(["--help"])


def test_launchers_are_crlf():
    from pathlib import Path

    for bat in (Path(__file__).resolve().parents[1] / "launchers").glob("*.bat"):
        data = bat.read_bytes()
        assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b""), bat.name


def test_decoder_buckets_pad_and_trim():
    from speechforge.engines.chatterbox import DECODE_BUCKETS, SILENCE, bucket_padding

    assert bucket_padding(21, DECODE_BUCKETS) == 19  # 18 tokens + 3 silence -> 40
    assert bucket_padding(40, DECODE_BUCKETS) == 0
    assert bucket_padding(41, DECODE_BUCKETS) == 23  # -> 64
    assert bucket_padding(999, DECODE_BUCKETS) == 0  # beyond the last bucket: unpadded
    eng = TokenEngine(1)._eng
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    seen: list[int] = []
    inner = eng._run

    def run(component, feeds):
        seen.append(int(feeds["speech_tokens"].shape[1] - voice.prompt_token.shape[1]))
        return inner(component, feeds)

    eng._run = run
    # 5 tokens + 3 trailing silence = 8 -> padded to 40; the audio is still 5 tokens
    wav = eng.decode_chunk([SILENCE] * 5, voice, left_context=0)
    assert seen == [40]
    assert len(wav) == 5 * SAMPLES_PER_TOKEN
    # the closing chunk keeps its 3 silence tokens but never the padding
    wav = eng.decode_chunk([SILENCE] * 5, voice, left_context=0, keep_silence=True)
    assert len(wav) == 8 * SAMPLES_PER_TOKEN
    # warm_decoder visits every bucket exactly once
    seen.clear()
    assert eng.warm_decoder(voice) == list(DECODE_BUCKETS)
    assert seen == list(DECODE_BUCKETS)


def test_warm_lm_workers_degrades_when_a_worker_is_busy():
    """A busy LM worker never reaches the barrier; the warm-up must then still
    run on every thread it can get and must not fail the load."""
    from speechforge import pipeline

    lm_pool = pipeline._pools()[0]
    release = threading.Event()
    busy = lm_pool.submit(release.wait, 10.0)
    ran: list[str] = []
    lock = threading.Lock()

    def fn() -> None:
        with lock:
            ran.append(threading.current_thread().name)

    # Free the worker after the barrier has given up, so the queued task runs.
    threading.Timer(0.6, release.set).start()
    t0 = time.perf_counter()
    assert pipeline.warm_lm_workers(fn, barrier_timeout_s=0.2) == pipeline.LM_WORKERS
    assert time.perf_counter() - t0 < 5.0
    assert busy.result(timeout=5.0) is True
    assert len(ran) == pipeline.LM_WORKERS
    assert all(name.startswith("sf-tts-lm") for name in ran)
    # Unloaded pool: every task lands on its own thread.
    ran.clear()
    assert pipeline.warm_lm_workers(fn, barrier_timeout_s=5.0) == pipeline.LM_WORKERS
    assert len(set(ran)) == pipeline.LM_WORKERS


@pytest.mark.parametrize(
    ("delay_s", "expect_small"),
    [(0.0005, False), (0.03, True)],  # fast LM keeps 25-token chunks; a slow one shrinks them
)
def test_pipelined_sizes_chunks_to_the_token_rate(delay_s, expect_small):
    eng = TokenEngine(120, delay_s=delay_s)
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    stats: dict = {}
    plan = ChunkPlanConfig(10, 25, 8, 3, 4.0)
    out = np.concatenate(list(pipelined(eng, "x", voice, plan=plan, stats=stats)))
    sizes = stats["chunk_tokens"]
    assert sizes[0] == 10
    assert sum(sizes) == 120 and len(out) == (120 + TRAILING_SILENCE) * SAMPLES_PER_TOKEN
    if expect_small:
        # 30 ms per token against 40 ms of audio per token: the second chunk must be
        # smaller than the fixed 25 or the first chunk's 400 ms would run out first
        assert sizes[1] < 25 and all(plan.min_chunk <= s <= 25 for s in sizes[1:-1])
    else:
        assert sizes[1] == 25


def test_pipelined_adaptive_off_keeps_fixed_chunks():
    eng = TokenEngine(60, delay_s=0.03)
    voice = slice_prompt(make_voice(prompt=200), 4.0)
    stats: dict = {}
    plan = ChunkPlanConfig(10, 25, 8, 3, 4.0, adaptive=False)
    list(pipelined(eng, "x", voice, plan=plan, stats=stats))
    assert stats["chunk_tokens"][:2] == [10, 25]
