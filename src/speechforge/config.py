"""Configuration: one YAML file in the data directory, secrets in an env file beside it.

Where things live (S8, SPEC-speechforge §8):

* ``SPEECHFORGE_DATA_DIR`` -- else ``<repo>/data`` -- holds ``config.yaml``,
  ``speechforge.env``, ``models/``, ``logs/`` and ``state/``. Voice conditionals
  live in ``voices.dir`` (default ``%LOCALAPPDATA%\\SpeechForge\\voices`` on
  Windows, ``<data>/voices`` elsewhere): biometric-grade data, ACL'd by
  ``install.ps1``, excluded from indexing and backups.
* ``speechforge.env`` holds the secrets by NAME: ``SPEECHFORGE_TOKEN``,
  ``SPEECHFORGE_TOKEN_PREVIOUS`` + ``SPEECHFORGE_TOKEN_PREVIOUS_UNTIL`` (the
  5-minute rotation overlap, :mod:`speechforge.auth`), and ``SPEECHFORGE_SF_KEY``
  only when StudioForge has ``server.api_key`` set.

Binding (§8.1): ``server.hosts`` is an explicit list. Loopback and tailnet
addresses (100.64.0.0/10, fd7a:115c:a1e0::/48) are allowed; ``0.0.0.0``/``::``
never; any other address only with TLS. A non-loopback bind needs a token.
"""

from __future__ import annotations

import ipaddress
import os
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from speechforge.errors import ConfigError

TOKEN_ENV = "SPEECHFORGE_TOKEN"
TOKEN_PREV_ENV = "SPEECHFORGE_TOKEN_PREVIOUS"
TOKEN_PREV_UNTIL_ENV = "SPEECHFORGE_TOKEN_PREVIOUS_UNTIL"
SF_KEY_ENV = "SPEECHFORGE_SF_KEY"
DATA_DIR_ENV = "SPEECHFORGE_DATA_DIR"
ENV_FILE_NAME = "speechforge.env"
CONFIG_FILE_NAME = "config.yaml"

#: Pinned so a re-download can never silently change the voice or the
#: transcripts (the per-file hashes are in models.lock).
CHATTERBOX_REVISION = "d21799bd0354adb85e348b8a0442a8405110a2cf"
PARAKEET_REVISION = "0bbb45a3365852604aef28b538a8f066f4ccaa85"
#: Qwen3-TTS-12Hz-*-Base per ``tts.qwen3.model`` (phase-2 bake-off, docs/ENGINES.md).
QWEN3_TTS_REVISIONS = {
    "1.7b": "fd4b254389122332181a7c3db7f27e918eec64e3",
    "0.6b": "5d83992436eae1d760afd27aff78a71d676296fc",
}
#: sherpa-onnx exports of nvidia/nemotron-speech-streaming-en-0.6b at a fixed
#: 160 ms chunk (the chunk size is baked in at export): fp32 and int8 repos.
NEMOTRON_REVISION = "9660703a1289d5b5c0c6470ced80eb52096fc740"
NEMOTRON_INT8_REVISION = "237e551abd7a411ef92d3595454d9f6ab5fe7d6c"

TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServerConfig(_Section):
    #: Every address to listen on. Rig: 127.0.0.1 plus its tailscale address.
    hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1"])
    port: int = Field(1236, ge=1, le=65535)
    max_body_mb: float = Field(4.0, gt=0, le=64)
    #: TLS (only needed for a non-tailnet, non-loopback address, e.g. the LAN).
    tls_cert: Path | None = None
    tls_key: Path | None = None
    #: WebSocket upgrades per source IP per minute (§8.3).
    upgrades_per_ip_per_min: int = Field(10, ge=1)


class GpuConfig(_Section):
    provider: Literal["cuda", "cpu"] = "cuda"
    #: CUDA ordinals in PCI-bus order (StudioForge's numbering), preference order.
    devices: list[int] = Field(default_factory=lambda: [3])
    #: Refuse to load unless NVML names the card with this substring ("3090").
    expect_name: str = ""
    mem_limit_mb: int = Field(6144, ge=512)
    #: NVIDIA driver floor (R580 = the CUDA 13 runtime the ORT wheels use).
    min_driver: int = Field(580, ge=0)
    #: One ONNX Runtime arena for every CUDA session on the card, created once at
    #: load with a large first chunk (S24, ``engines/arena.py``); off = one arena
    #: per session (S22). A registration the runtime refuses falls back to off.
    shared_arena: bool = False
    #: The shared arena's first chunk. The runtime's field is a 32-bit int, so
    #: 2047 MiB is the most it accepts (the bound, and the default).
    arena_initial_mb: int = Field(2047, ge=64, le=2047)


class ModelsConfig(_Section):
    dir: Path | None = None
    tts_dtype: Literal["fp32", "fp16", "q4", "q4f16", "q8"] = "fp32"
    stt_quantization: Literal["int8"] | None = None
    #: Full sha256 re-check of every model file at load after this many days.
    full_verify_days: float = Field(7.0, gt=0)


class Qwen3TtsConfig(_Section):
    """``tts.qwen3``: read only when ``tts.engine`` is ``qwen3-tts`` (docs/ENGINES.md).
    The engine needs ``voices.allow_reference_clip: true`` (decision D3)."""

    model: Literal["1.7b", "0.6b"] = "1.7b"
    dtype: Literal["bf16", "fp16"] = "bf16"
    attn: Literal["sdpa", "flash_attention_2"] = "sdpa"
    #: Passed to the model per utterance ("English", "Chinese", ..., or "Auto").
    language: str = "English"
    #: Codec steps (12 per second of audio) per streamed chunk; smaller = earlier first audio.
    chunk_size: int = Field(8, ge=1, le=48)
    #: Static KV cache length: reference codes + text + generated steps.
    max_seq_len: int = Field(2048, ge=256, le=8192)
    max_new_tokens: int = Field(1024, ge=12, le=4096)
    temperature: float = Field(0.9, ge=0.0, le=2.0)
    top_k: int = Field(50, ge=1)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.05, ge=1.0, le=3.0)
    #: Speaker embedding only (ignores every transcript): no phoneme bleed, less similarity.
    xvec_only: bool = False
    #: Half a second of silence after the reference so its last phoneme ends there.
    append_silence: bool = True


class DotsTtsConfig(_Section):
    """``tts.dots``: the dots.tts-mf adapter (``engines/dots_tts.py``, phase-2 bake-off).

    The engine clones from a reference clip + transcript that must be on the rig,
    so it loads only with ``voices.allow_reference_clip: true`` (decision D3).
    """

    #: ``native`` runs the ``dots_tts`` PyTorch runtime in this process on the
    #: leased card; ``http`` talks to an SGLang Omni ``/v1/audio/speech`` server
    #: (Linux / WSL2 on this machine) over loopback only.
    backend: Literal["native", "http"] = "native"
    url: str = "http://127.0.0.1:8000"
    http_model: str = ""
    http_timeout_s: float = Field(30.0, gt=0, le=300)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    #: MeanFlow checkpoint: 4 recommended; ``null`` = the artifact's own default
    #: (required for the fixed-step ``-2steps`` / ``-stts`` artifacts).
    num_steps: int | None = Field(4, ge=1, le=64)
    guidance_scale: float | None = Field(1.2, ge=0.0, le=10.0)
    speaker_scale: float = Field(1.5, ge=0.0, le=5.0)
    eos_threshold: float = Field(0.8, gt=0.0, le=2.0)
    #: torch.compile (Triton) with a warm-up at load; off = eager.
    optimize: bool = False
    warmup_on_optimize: bool = True
    #: Longest utterance in 160 ms latent patches (500 = 80 s).
    max_generate_length: int = Field(500, ge=16, le=2000)
    #: The model speaks at 48 kHz; 24000 decimates to the sf.1 frame rate.
    output_rate: Literal[24000, 48000] = 24000
    #: Incremental text in, audio out, on the ``dots.tts-mf-2steps-stts`` artifact
    #: (model group ``tts-dots-stts``); native backend only.
    double_streaming: bool = False
    #: Language hint for the runtime (``null`` = none, ``auto_detect``, or a code).
    language: str | None = None
    normalize_text: bool = False
    #: WeTextProcessing (pynini) has no Windows wheels; with this on, a missing
    #: ``tn`` package is shimmed so the runtime imports. Text normalisation is
    #: then unavailable (``normalize_text`` must stay false).
    allow_missing_text_normalizer: bool = True

    @field_validator("url")
    @classmethod
    def _loopback_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parts = urlsplit(value.strip())
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("tts.dots.url must be an http:// URL")
        if not is_loopback_host(parts.hostname):
            raise ValueError(
                "tts.dots.url must be loopback: the reference clip leaves this process "
                "only to a server on this machine (D3, SPEC 8.4)"
            )
        return value.strip().rstrip("/")

    @field_validator("language")
    @classmethod
    def _language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        if not text or text.lower() == "none":
            return None
        if len(text) > 32:
            raise ValueError("tts.dots.language is too long")
        return text


class TtsConfig(_Section):
    engine: str = "chatterbox"
    qwen3: Qwen3TtsConfig = Field(default_factory=Qwen3TtsConfig)
    first_chunk_tokens: int = Field(15, ge=4, le=100)
    chunk_tokens: int = Field(25, ge=4, le=200)
    left_context: int = Field(8, ge=0, le=50)
    lookahead_tokens: int = Field(3, ge=0, le=10)
    prompt_s: float = Field(4.0, ge=1.0, le=10.0)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(1000, ge=1)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.2, ge=1.0, le=3.0)
    #: HTTP utility route only: concurrent /v1/audio/speech calls.
    max_http_streams: int = Field(1, ge=1, le=8)
    #: Knobs of the ``dots-tts`` engine (docs/ENGINES.md); ignored by the others.
    dots: DotsTtsConfig = Field(default_factory=DotsTtsConfig)


class NemotronConfig(_Section):
    """``stt.engine: nemotron`` -- NVIDIA Nemotron Speech Streaming EN 0.6B through
    sherpa-onnx (docs/ENGINES.md). Weights: NVIDIA Open Model License."""

    #: ``auto`` = ``subprocess`` on Windows with the CUDA provider (sherpa-onnx's
    #: GPU wheel bundles its own ONNX Runtime + CUDA 12 DLLs, which clash by name
    #: with this process's CUDA 13 onnxruntime-gpu), ``sherpa-onnx`` in-process
    #: everywhere else.
    backend: Literal["auto", "sherpa-onnx", "subprocess"] = "auto"
    #: The export's chunk (its ``encoder.onnx`` carries it): the model decodes a
    #: new chunk of text every 160 ms of audio. Other exports exist (80/560/1120 ms,
    #: same publisher); adding one is a models.lock group plus this Literal.
    chunk_ms: Literal[160] = 160
    #: ``int8`` = the dynamically quantised export (CPU; S4 explains why not CUDA).
    quantization: Literal["int8"] | None = None
    #: sherpa-onnx intra-op threads (CPU provider; the CUDA provider ignores it).
    num_threads: int = Field(2, ge=1, le=32)
    decoding: Literal["greedy_search", "modified_beam_search"] = "greedy_search"
    #: Silence fed before ``finalize`` so the encoder's right context and the
    #: last chunk flush (sherpa-onnx's own example pads 660 ms).
    tail_pad_ms: int = Field(480, ge=0, le=2000)
    #: ``subprocess`` backend: the interpreter that has sherpa-onnx (its CUDA
    #: wheel lives in its own venv); empty = this process's interpreter.
    python: str = ""
    #: ``subprocess`` backend: how long a load / a stream call may take.
    worker_timeout_s: float = Field(120.0, ge=5.0, le=900.0)


class SttConfig(_Section):
    engine: str = "parakeet"
    #: HTTP utility route only.
    max_audio_s: float = Field(30.0, gt=0, le=120)
    nemotron: NemotronConfig = Field(default_factory=NemotronConfig)


class EouConfig(_Section):
    enabled: bool = True
    #: Server-owned floor on the silence before Smart Turn is consulted; the client's
    #: ``eou.min_pause_ms`` can raise the trigger but never lower it. The pause that
    #: follows a running partial ending in ``. ? !`` is scored once it reaches this.
    #: Provisional, to be set from the measurement.
    min_silence_ms: int = Field(300, ge=60, le=3000)
    #: The longer floor for a pause whose running partial does NOT end a sentence
    #: (a mid-sentence breath on a long utterance); values below ``min_silence_ms``
    #: act as ``min_silence_ms``. Provisional, to be set from the measurement.
    unpunctuated_silence_ms: int = Field(800, ge=60, le=5000)


class RealtimeConfig(_Section):
    """sf.1 session limits (SPEC-speechforge §4.3, §4.4, §8.3)."""

    max_sessions: int = Field(3, ge=1, le=16)
    partial_ms: int = Field(160, ge=80, le=2000)
    min_pause_ms: int = Field(150, ge=60, le=2000)
    #: Rolling partials re-decode at most this much open tail; a longer tail is
    #: committed at its last >= 300 ms pause (or cut hard).
    max_tail_s: float = Field(8.0, ge=2.0, le=20.0)
    max_utterance_s: float = Field(60.0, ge=5.0, le=300.0)
    mic_ring_s: float = Field(10.0, ge=2.0, le=60.0)
    max_ingest_ratio: float = Field(1.25, ge=1.0, le=4.0)
    max_text_per_ctx: int = Field(2000, ge=100)
    max_text_per_min: int = Field(20000, ge=1000)
    max_contexts: int = Field(2, ge=1, le=4)
    default_ahead_ms: int = Field(3000, ge=200, le=30000)
    max_ahead_ms: int = Field(10000, ge=500, le=60000)
    idle_close_s: float = Field(30.0, ge=5.0)
    resume_window_s: float = Field(10.0, ge=0.0, le=120.0)
    error_flood: int = Field(50, ge=5)
    error_window_s: float = Field(10.0, ge=1.0)
    voice_puts_per_hour: int = Field(30, ge=1)
    #: How long a new session waits for the models before ``ready`` (the box
    #: runs on its CPU meanwhile); past it the session gets error{not_ready}.
    ready_wait_s: float = Field(20.0, ge=0.0, le=120.0)


class VoicesConfig(_Section):
    dir: Path | None = None
    max_bytes: int = Field(2 * 1024 * 1024, ge=64 * 1024)
    max_files: int = Field(256, ge=1)
    #: D3: the reference clip stays on the box. True enables
    #: POST /v1/voices/from-reference (the owner's decision only).
    allow_reference_clip: bool = False


class SelftestConfig(_Section):
    enabled: bool = True
    #: Gates (GPU). On provider "cpu" the timing gates are reported, not enforced.
    max_ttfa_ms: float = 250.0
    max_cancel_ms: float = 50.0
    max_stt_wer: float = 0.15
    max_roundtrip_wer: float = 0.25


class ResidencyConfig(_Section):
    idle_unload_s: float = Field(900.0, ge=0)
    warm_on_boot: bool = False
    drain_timeout_s: float = Field(15.0, ge=0, le=300)
    failure_cooldown_s: float = Field(60.0, ge=1)


class StudioForgeConfig(_Section):
    lease_mode: Literal["hold", "off"] = "hold"
    url: str = "http://127.0.0.1:1234"
    holder: str = "speechforge"
    priority: int = Field(2, ge=1, le=3)
    lease_idle_ttl_s: float | None = None
    vacate_url: str = ""
    timeout_s: float = Field(5.0, gt=0, le=60)
    #: StudioForge unreachable: ``proceed_if_free`` loads only when NVML shows at
    #: least ``unreachable_min_free_gib`` free on the card and no llama-server
    #: holds memory there (§7.3); ``refuse`` never loads unleased.
    on_unreachable: Literal["proceed_if_free", "refuse"] = "proceed_if_free"
    unreachable_min_free_gib: float = Field(8.0, ge=0)
    refusal_cooldown_s: float = Field(30.0, ge=1)
    touch_interval_s: float = Field(30.0, ge=1)
    vacate_cooldown_s: float = Field(180.0, ge=0)

    @field_validator("holder")
    @classmethod
    def _holder(cls, value: str) -> str:
        text = value.strip()
        if not text or len(text) > 64:
            raise ValueError("holder must be 1-64 characters")
        return text


class Config(_Section):
    data_dir: Path = Path("data")
    server: ServerConfig = Field(default_factory=ServerConfig)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    eou: EouConfig = Field(default_factory=EouConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    voices: VoicesConfig = Field(default_factory=VoicesConfig)
    selftest: SelftestConfig = Field(default_factory=SelftestConfig)
    residency: ResidencyConfig = Field(default_factory=ResidencyConfig)
    studioforge: StudioForgeConfig = Field(default_factory=StudioForgeConfig)

    @property
    def models_dir(self) -> Path:
        return Path(self.models.dir) if self.models.dir else self.data_dir / "models"

    @property
    def voices_dir(self) -> Path:
        if self.voices.dir:
            return Path(self.voices.dir)
        local = os.environ.get("LOCALAPPDATA", "")
        if sys.platform == "win32" and local:
            return Path(local) / "SpeechForge" / "voices"
        return self.data_dir / "voices"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def env_file(self) -> Path:
        return self.data_dir / ENV_FILE_NAME

    @property
    def lease_idle_ttl_s(self) -> float | None:
        if self.studioforge.lease_idle_ttl_s is not None:
            return self.studioforge.lease_idle_ttl_s
        if self.residency.idle_unload_s <= 0:
            return 3600.0
        return self.residency.idle_unload_s + 300.0

    @property
    def vacate_url(self) -> str:
        """StudioForge runs on the same box: the callback is loopback, and the
        vacate route accepts loopback peers only (§8.2)."""
        if self.studioforge.vacate_url:
            return self.studioforge.vacate_url
        return f"http://127.0.0.1:{self.server.port}/internal/vacate"

    @property
    def token(self) -> str:
        return os.environ.get(TOKEN_ENV, "").strip()

    @property
    def studioforge_key(self) -> str:
        return os.environ.get(SF_KEY_ENV, "").strip()

    @property
    def tls(self) -> bool:
        return bool(self.server.tls_cert and self.server.tls_key)


def default_data_dir() -> Path:
    env = os.environ.get(DATA_DIR_ENV, "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def parse_env_file(path: Path) -> dict[str, str]:
    """``SPEECHFORGE_*`` names from a ``KEY=VALUE`` file (comments, ``export``,
    quotes allowed). Missing file = empty."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.startswith("SPEECHFORGE_"):
            out[key] = value
    return out


def load_env_file(path: Path) -> list[str]:
    """Read the env file into ``os.environ`` without overriding. Returns NAMES."""
    names: list[str] = []
    for key, value in parse_env_file(path).items():
        if key not in os.environ:
            os.environ[key] = value
            names.append(key)
    return names


def load_config(path: Path | None = None, *, data_dir: Path | None = None) -> Config:
    base = Path(data_dir) if data_dir is not None else default_data_dir()
    cfg_path = Path(path) if path is not None else base / CONFIG_FILE_NAME
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path.name} must be a mapping")
        raw = loaded
    if "data_dir" in raw:
        raise ConfigError("data_dir is not a config key; set SPEECHFORGE_DATA_DIR instead")
    try:
        config = Config(data_dir=base, **raw)
    except Exception as exc:
        raise ConfigError(f"{cfg_path.name}: {exc}") from exc
    load_env_file(config.env_file)
    return config


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address((host or "").strip().strip("[]"))
    except ValueError:
        return None


def is_loopback_host(host: str) -> bool:
    if (host or "").strip().lower() == "localhost":
        return True
    ip = _ip(host)
    return bool(ip and ip.is_loopback)


def is_tailnet_host(host: str) -> bool:
    ip = _ip(host)
    if ip is None:
        return False
    return ip in (TAILNET_V4 if ip.version == 4 else TAILNET_V6)


def validate_for_serving(config: Config) -> None:
    """Refuse a configuration that would be open wider than §8.1 allows."""
    if not config.server.hosts:
        raise ConfigError("server.hosts is empty")
    for host in config.server.hosts:
        ip = _ip(host)
        if host.strip().lower() != "localhost" and ip is None:
            raise ConfigError(f"server.hosts: {host!r} is not an IP address")
        if ip is not None and ip.is_unspecified:
            raise ConfigError(
                f"server.hosts: {host} binds every interface; list the "
                f"tailscale address and 127.0.0.1 instead"
            )
        if is_loopback_host(host):
            continue
        if not config.token:
            raise ConfigError(
                f"server.hosts includes {host} (not loopback) and {TOKEN_ENV} is not set: "
                f"refusing to serve speech to the network without a token. Run "
                f"`speechforge token new` or bind 127.0.0.1 only."
            )
        if not is_tailnet_host(host) and not config.tls:
            raise ConfigError(
                f"server.hosts: {host} is neither loopback nor tailnet; it "
                f"needs server.tls_cert and server.tls_key"
            )
    if config.gpu.provider == "cuda" and not config.gpu.devices:
        raise ConfigError("gpu.devices is empty: name at least one CUDA ordinal")
