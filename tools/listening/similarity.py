"""Step 5b (optional): objective speaker similarity (SPEC §11.3, "ECAPA cosine").

    uv run --extra cpu python -m tools.listening.similarity \\
        --onnx <models>/wespeaker-ecapa/voxceleb_ECAPA512.onnx --reference <owner-clip.wav> \\
        --engines <renders>/chatterbox <renders>/qwen3-tts ...

Embeds the reference clip and every rendered line with a speaker-verification
model on the CPU (``onnxruntime``) and prints, per engine, the mean cosine
similarity to the reference and the difference to the Chatterbox baseline. SPEC
§11.3 asks for a delta of at most 0.05 between the GPU and CPU Chatterbox; the
same number says how close each candidate engine gets to the owner's voice.

The model is WeSpeaker's ECAPA-TDNN (512 channels, 192-dim embedding) trained on
VoxCeleb2, Apache-2.0, pinned in ``src/speechforge/models.lock`` under group
``eval``: ``speechforge fetch-models --group eval`` downloads and verifies it.
Features are WeSpeaker's: 80-bin Kaldi log-mel filterbank on 16 kHz audio
scaled to the 16-bit range, 25 ms Hamming frames every 10 ms, no dither, then
mean normalisation over time -- implemented here in numpy so the kit needs no
torch.

The reference clip is the owner's voice. It stays where it is (D3/D10); this
script only reads it and writes numbers. Run it where the clip lives.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

from speechforge.audio import resample_linear
from tools.listening import kit

SAMPLE_RATE = 16000
FRAME_LEN, FRAME_SHIFT, N_FFT, N_MELS = 400, 160, 512, 80
PREEMPH = 0.97


# ------------------------------------------------------------------ features
def _mel(hz: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + hz / 700.0)


def mel_filterbank(
    n_mels: int = N_MELS, n_fft: int = N_FFT, rate: int = SAMPLE_RATE, low_hz: float = 20.0
) -> np.ndarray:
    """Kaldi's triangular mel bins over the ``n_fft // 2 + 1`` FFT bins."""
    nyquist = rate / 2.0
    n_bins = n_fft // 2 + 1
    fft_hz = np.arange(n_bins) * (rate / n_fft)
    mel_lo, mel_hi = _mel(np.array(low_hz)), _mel(np.array(nyquist))
    delta = (mel_hi - mel_lo) / (n_mels + 1)
    centres = mel_lo + delta * np.arange(n_mels + 2)
    bins_mel = _mel(fft_hz)
    fb = np.zeros((n_mels, n_bins), np.float32)
    for m in range(n_mels):
        left, centre, right = centres[m], centres[m + 1], centres[m + 2]
        up = (bins_mel - left) / (centre - left)
        down = (right - bins_mel) / (right - centre)
        fb[m] = np.clip(np.minimum(up, down), 0.0, None)
    # Kaldi leaves the Nyquist bin out of the filterbank.
    fb[:, -1] = 0.0
    return fb


_FB = None


def fbank(audio16k: np.ndarray, cmn: bool = True) -> np.ndarray:
    """Kaldi-style log-mel filterbank as WeSpeaker computes it: (frames, 80) float32."""
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    x = np.asarray(audio16k, np.float32) * 32768.0  # WeSpeaker feeds the 16-bit range
    if len(x) < FRAME_LEN:
        x = np.pad(x, (0, FRAME_LEN - len(x)))
    n_frames = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n_frames)[:, None]
    frames = x[idx].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)  # remove_dc_offset
    frames[:, 1:] -= PREEMPH * frames[:, :-1]  # pre-emphasis (first sample against itself)
    frames[:, 0] *= 1.0 - PREEMPH
    window = np.hamming(FRAME_LEN)
    spec = np.abs(np.fft.rfft(frames * window, n=N_FFT)) ** 2
    mel = spec @ _FB.T
    feats = np.log(np.maximum(mel, np.finfo(np.float32).eps)).astype(np.float32)
    if cmn:
        feats -= feats.mean(axis=0, keepdims=True)
    return feats


# ------------------------------------------------------------------ embedders
class Embedder(Protocol):
    def embed(self, audio16k: np.ndarray) -> np.ndarray: ...


class OnnxEmbedder:
    """WeSpeaker ONNX export: ``feats`` (B, T, 80) -> ``embs`` (B, D)."""

    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment
            raise SystemExit(
                "onnxruntime is not installed: run with `uv run --extra cpu ...`"
            ) from exc
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.output = self.sess.get_outputs()[0].name

    def embed(self, audio16k: np.ndarray) -> np.ndarray:
        feats = fbank(audio16k)[None, :, :]
        out = self.sess.run([self.output], {self.input: feats})[0]
        return np.asarray(out, np.float32).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def load16k(path: Path) -> np.ndarray:
    audio, rate = kit.read_wav(path)
    return resample_linear(audio, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else audio


def reference_embedding(embedder: Embedder, refs: list[Path]) -> np.ndarray:
    embs = []
    for p in refs:
        e = embedder.embed(load16k(p))
        embs.append(e / (np.linalg.norm(e) or 1.0))
    return np.mean(embs, axis=0)


def score_dir(embedder: Embedder, ref: np.ndarray, d: Path) -> dict:
    rows = []
    for p in sorted(d.glob("[0-9][0-9].wav")):
        rows.append({"file": p.name, "cosine": round(cosine(embedder.embed(load16k(p)), ref), 4)})
    cs = [r["cosine"] for r in rows]
    return {
        "engine": d.name,
        "lines": rows,
        "mean": round(statistics.fmean(cs), 4) if cs else None,
        "std": round(statistics.pstdev(cs), 4) if len(cs) > 1 else 0.0,
        "min": min(cs) if cs else None,
    }


def compare(results: list[dict], baseline: str | None) -> list[dict]:
    """Add ``delta`` against the baseline engine (default: the one named chatterbox*)."""
    base = None
    if baseline:
        base = next((r for r in results if r["engine"] == baseline), None)
    if base is None:
        base = next((r for r in results if r["engine"].lower().startswith("chatterbox")), None)
    for r in results:
        r["delta"] = (
            round(r["mean"] - base["mean"], 4)
            if base and base["mean"] is not None and r["mean"] is not None
            else None
        )
        r["baseline"] = base["engine"] if base else None
    return results


def format_table(results: list[dict], threshold: float) -> str:
    lines = [f"{'engine':16} {'cosine':>7} {'std':>6} {'min':>6} {'delta':>7}  n"]
    for r in results:
        d = "-" if r["delta"] is None else f"{r['delta']:+.3f}"
        flag = ""
        if r["delta"] is not None and r["engine"] != r["baseline"] and abs(r["delta"]) > threshold:
            flag = f"  (|delta| > {threshold})"
        m = "-" if r["mean"] is None else f"{r['mean']:.3f}"
        mn = "-" if r["min"] is None else f"{r['min']:.3f}"
        lines.append(
            f"{r['engine']:16} {m:>7} {r['std']:6.3f} {mn:>6} {d:>7}  {len(r['lines'])}{flag}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--onnx", required=True, type=Path, help="speaker-embedding ONNX model")
    ap.add_argument(
        "--reference",
        required=True,
        type=Path,
        nargs="+",
        help="reference clip(s) of the voice owner",
    )
    ap.add_argument("--engines", nargs="+", required=True, type=Path, metavar="DIR")
    ap.add_argument(
        "--baseline", default=None, help="engine to take deltas against (default chatterbox*)"
    )
    ap.add_argument("--threshold", type=float, default=0.05, help="SPEC 11.3 delta")
    args = ap.parse_args(argv)
    if not args.onnx.is_file():
        raise SystemExit(
            f"{args.onnx} not found. `speechforge fetch-models --group eval` downloads the pinned "
            "WeSpeaker ECAPA-TDNN model (Apache-2.0) into the models directory."
        )
    embedder = OnnxEmbedder(args.onnx)
    ref = reference_embedding(embedder, args.reference)
    results = compare([score_dir(embedder, ref, d) for d in args.engines], args.baseline)
    for r in results:
        kit.json_dump(
            Path(next(d for d in args.engines if d.name == r["engine"])) / "similarity.json", r
        )
    print(format_table(results, args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
