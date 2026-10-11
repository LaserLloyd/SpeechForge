"""The benchmark harness with the phase-1 acceptance thresholds (SPEC-speechforge §11.3).

``speechforge bench --url ws://<rig>:1236/v1/realtime`` drives a RUNNING server
over sf.1 and prints every measured number next to its threshold, PASS/FAIL,
and exits non-zero if any gate it could measure failed. It measures what can
be measured from one client against the rig:

=========================================  =====================  ==========
Gate                                       How                    Threshold
=========================================  =====================  ==========
TTS TTFA p50 / p90, 1 session              tts.open -> 1st frame  150 / 250 ms
TTS TTFA p90, 2 sessions                   two clients at once    320 ms
RTF                                        tts.done.rtf           <= 0.25
cancel -> GPU idle for that gen, p99       cancelled.stop_ms      <= 50 ms
partial lag p90                            stt.partial.lag_ms     <= 250 ms
EOU decision after speech end p50 / p90    speech end -> eou      <= 350 / 550 ms
                                           (+ the box's policy)
synthesised-speech WER (TTS -> STT)        round trip             <= 3 %
final WER, clean / noisy                   ``--corpus DIR``       <= 6 % / 12 %
names correct                              ``--corpus DIR``       >= 90 %
false end-of-turn on unfinished clips      ``--corpus DIR``       <= 4 %
soak: underruns / VRAM drift / errors      ``--soak SECONDS``     0 / <= 100 MB / 0
=========================================  =====================  ==========

The corpus is the family's recorded set (SPEC §11.2): ``<name>.wav`` (16-bit,
any rate) + ``<name>.txt`` (reference text) in ``clean/``, ``noisy/``,
``unfinished/``; ``names.txt`` lists the names to score. It never leaves the box
and is never committed. Gates that need the box's own pipeline (end-to-end first
audio, box CPU per session, GPU-vs-CPU speaker similarity, rig-killed failover)
are run by the add-on's harness and listed as "box" here.

The voice used is the shipped synthetic one (``assets/selftest-voice.wav``),
built on the box with ``--cpu-voice`` (needs the CPU extra and the models) or,
when the server allows reference clips, on the rig; ``--voice FILE.npz`` uses
existing conditionals.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

from speechforge.audio import resample_linear
from speechforge.selftest import TTS_LINE, golden_clip, wer

THRESHOLDS: dict[str, tuple[str, float]] = {
    "tts_ttfa_p50_ms": ("<=", 150),
    "tts_ttfa_p90_ms": ("<=", 250),
    "tts_ttfa_p90_2sessions_ms": ("<=", 320),
    "tts_rtf_p50": ("<=", 0.25),
    "cancel_stop_p99_ms": ("<=", 50),
    "partial_lag_p90_ms": ("<=", 250),
    "eou_after_speech_end_p50_ms": ("<=", 350),
    "eou_after_speech_end_p90_ms": ("<=", 550),
    "tts_roundtrip_wer": ("<=", 0.03),
    "final_wer_clean": ("<=", 0.06),
    "final_wer_noisy": ("<=", 0.12),
    "names_correct": (">=", 0.90),
    "false_eou_rate": ("<=", 0.04),
    "soak_underruns": ("<=", 0),
    "soak_vram_drift_mib": ("<=", 100),
    "soak_errors": ("<=", 0),
}
BOX_GATES = (
    "end-to-end first audio p50<=1.6 s / p90<=2.2 s",
    "box CPU <=15 % of a core",
    "GPU vs CPU speaker similarity delta <=0.05",
    "rig killed -> CPU <=1.0 s",
    "co-tenant TTFA p90 within +20 % of idle",
)
SENTENCES = [
    "Sure, I can help with that.",
    "The weather looks clear for Saturday morning.",
    "Give me one second to check.",
    "That sounds like a great plan to me.",
    "Your next meeting starts at three this afternoon.",
    "Okay, I added milk to the list.",
    "Here is what I found about the museum hours.",
    "It takes about twenty minutes by car.",
    "Let me read the first one back to you.",
    "Good question, let me think about it.",
]


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p * len(s)))], 2)


def read_wav16(path: Path) -> np.ndarray:
    with wave.open(io.BytesIO(path.read_bytes())) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return resample_linear(a, rate, 16000)


def to16(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, -1, 1) * 32767).astype("<i2")


async def upload_voice(client: Any, data: bytes) -> str:
    import base64

    vid = hashlib.sha256(data).hexdigest()
    await client.send({"type": "voice.has", "id": vid})
    if (await client.next("voice.has"))["present"]:
        return vid
    b64 = base64.b64encode(data).decode()
    step = 48 * 1024
    parts = [b64[i : i + step] for i in range(0, len(b64), step)]
    for i, part in enumerate(parts):
        await client.send(
            {"type": "voice.put", "id": vid, "part": i, "parts": len(parts), "data_b64": part}
        )
    await client.next("voice.stored", timeout=30)
    return vid


async def speak(client: Any, ctx: int, gen: int, vid: str, text: str, seed: int) -> dict:
    t0 = time.perf_counter()
    await client.send(
        {
            "type": "tts.open",
            "ctx": ctx,
            "gen": gen,
            "voice": f"sha256:{vid}",
            "seed": seed,
            "turn_id": f"bench-{ctx}",
        }
    )
    await client.send({"type": "tts.text", "ctx": ctx, "text": text, "final": True})
    await client.send({"type": "tts.close", "ctx": ctx})
    started = await client.next("tts.started", timeout=30)
    done = await client.next("tts.done", timeout=60)
    first = client.audio[ctx][0][0] if client.audio.get(ctx) else started["_t"]
    return {
        "ttfa_ms": (first - t0) * 1000,
        "rig_ttfa_ms": started["ttfa_ms"],
        "rtf": done.get("rtf"),
        "audio_ms": done.get("audio_ms"),
    }


async def cancel_once(client: Any, ctx: int, gen: int, vid: str) -> float:
    await client.send({"type": "tts.open", "ctx": ctx, "gen": gen, "voice": f"sha256:{vid}"})
    await client.send(
        {
            "type": "tts.text",
            "ctx": ctx,
            "final": True,
            "text": "This is a long sentence that keeps on going so there is "
            "plenty of speech to interrupt in the middle of it.",
        }
    )
    await client.next("tts.started", timeout=30)
    await client.send({"type": "cancel", "gen": gen})
    return float((await client.next("cancelled", timeout=10))["stop_ms"])


async def stream_clip(client: Any, audio16k: np.ndarray, tail_silence_s: float = 1.2) -> dict:
    """Send a clip in real time, then silence; collect partial lag, eou and final."""
    for k in ("stt.partial", "eou", "stt.final", "stt.vad"):
        client.drain(k)
    start_pts = client.mic_pts
    speech_end = start_pts + len(audio16k)
    pcm = to16(np.concatenate([audio16k, np.zeros(int(tail_silence_s * 16000), np.float32)]))
    t0 = time.perf_counter()
    # The moment the last speech frame actually left, not the nominal clock:
    # a paced sender that falls behind would otherwise inflate every eou figure.
    t_speech_end = await client.send_mic(pcm, realtime=True, mark_pts=speech_end)
    if t_speech_end is None:
        t_speech_end = t0 + len(audio16k) / 16000
    await client.send({"type": "stt.flush", "id": "b"})
    final = await client.next("stt.final", timeout=10)
    partials = client.drain("stt.partial")
    eous = [e for e in client.drain("eou") if e["at"] >= speech_end - 8000]
    eou_ms = None
    if eous:
        e = eous[0]
        eou_ms = (e["_t"] - t_speech_end) * 1000
    return {
        "final": final.get("text", ""),
        "lags": [p["lag_ms"] for p in partials],
        "eou_ms": eou_ms,
        "eou_p": eous[0]["p"] if eous else None,
        "final_ms": final.get("ms"),
    }


async def run(
    url: str,
    token: str,
    *,
    voice: bytes,
    n: int = 10,
    corpus: Path | None = None,
    soak_s: float = 0.0,
) -> dict[str, Any]:
    from speechforge.client import RigClient

    res: dict[str, Any] = {"url": url}
    c = RigClient(url, token)
    t_conn = time.perf_counter()
    ready = await c.connect()
    res["ready_s"] = round(time.perf_counter() - t_conn, 2)
    res["engines"] = ready.get("engines")
    res["gpu"] = ready.get("gpu")
    vid = await upload_voice(c, voice)
    # TTS, one session
    ttfa, rtf, audio = [], [], {}
    for i in range(n):
        r = await speak(c, 100 + i, 100 + i, vid, SENTENCES[i % len(SENTENCES)], i)
        ttfa.append(r["ttfa_ms"])
        if r["rtf"] is not None:
            rtf.append(r["rtf"])
        audio[100 + i] = SENTENCES[i % len(SENTENCES)]
    res["tts_ttfa_p50_ms"], res["tts_ttfa_p90_ms"] = q(ttfa, 0.5), q(ttfa, 0.9)
    res["tts_rtf_p50"] = q(rtf, 0.5)
    # round trip WER via the rig's own STT
    errs = []
    for ctx, text in list(audio.items())[:5]:
        pcm = c.ctx_pcm(ctx).astype(np.float32) / 32768
        hyp = await stream_clip(c, resample_linear(pcm, 24000, 16000), 0.6)
        errs.append(wer(text, hyp["final"]))
    res["tts_roundtrip_wer"] = round(statistics.mean(errs), 3) if errs else None
    # cancel
    stops = [await cancel_once(c, 200 + i, 200 + i, vid) for i in range(max(5, n))]
    res["cancel_stop_p99_ms"] = q(stops, 0.99)
    # STT on the golden clip, partial lag + eou
    clip, rate = golden_clip()
    g = await stream_clip(c, resample_linear(clip, rate, 16000))
    res["partial_lag_p90_ms"] = q(g["lags"], 0.9)
    res["golden_wer"] = round(
        wer(
            "hey welcome back to the show today we're digging into something i've been curious "
            "about for weeks honestly it's one of those topics that sound simple but the more you "
            "look the stranger it gets",
            g["final"],
        ),
        3,
    )
    eou_ms = [g["eou_ms"]] if g["eou_ms"] is not None else []
    # two sessions at once
    c2 = RigClient(url, token)
    await c2.connect()
    vid2 = await upload_voice(c2, voice)
    both: list[float] = []
    for i in range(max(3, n // 2)):
        a, b = await asyncio.gather(
            speak(c, 300 + i, 300 + i, vid, SENTENCES[i % 10], i),
            speak(c2, 300 + i, 300 + i, vid2, SENTENCES[(i + 3) % 10], i),
        )
        both += [a["ttfa_ms"], b["ttfa_ms"]]
    res["tts_ttfa_p90_2sessions_ms"] = q(both, 0.9)
    await c2.close()
    # corpus
    if corpus is not None:
        res.update(await run_corpus(c, corpus, eou_ms))
    res["eou_after_speech_end_p50_ms"] = q(eou_ms, 0.5)
    res["eou_after_speech_end_p90_ms"] = q(eou_ms, 0.9)
    if soak_s > 0:
        res.update(await soak(url, token, voice, soak_s))
    res["ping_rtt_p50_ms"] = q(c.rtts, 0.5)
    await c.close()
    return res


async def run_corpus(c: Any, corpus: Path, eou_ms: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = []
    if (corpus / "names.txt").exists():
        names = [
            x.strip().lower() for x in (corpus / "names.txt").read_text().splitlines() if x.strip()
        ]
    name_hits = name_total = 0
    for split in ("clean", "noisy"):
        errs = []
        for wav in sorted((corpus / split).glob("*.wav")):
            ref = wav.with_suffix(".txt").read_text().strip()
            r = await stream_clip(c, read_wav16(wav))
            errs.append(wer(ref, r["final"]))
            if r["eou_ms"] is not None:
                eou_ms.append(r["eou_ms"])
            for nm in names:
                if nm in ref.lower():
                    name_total += 1
                    name_hits += nm in r["final"].lower()
        out[f"final_wer_{split}"] = round(statistics.mean(errs), 3) if errs else None
    out["names_correct"] = round(name_hits / name_total, 3) if name_total else None
    false = total = 0
    for wav in sorted((corpus / "unfinished").glob("*.wav")):
        r = await stream_clip(c, read_wav16(wav), 0.5)
        total += 1
        # The box's table: p >= 0.9 ends; 0.5-0.9 ends only with terminal punctuation.
        p = r["eou_p"] or 0.0
        if p >= 0.9 or (p >= 0.5 and r["final"].rstrip().endswith((".", "?", "!"))):
            false += 1
    out["false_eou_rate"] = round(false / total, 3) if total else None
    return out


async def soak(url: str, token: str, voice: bytes, seconds: float) -> dict[str, Any]:
    """Two sessions alternating sentences for ``seconds``; an underrun is a gap
    between consecutive frames of one ctx larger than real time allows once
    playback has started (40 ms frames, 200 ms jitter allowance)."""
    from speechforge.client import RigClient

    clients = [RigClient(url, token), RigClient(url, token)]
    for cl in clients:
        await cl.connect()
    vids = [await upload_voice(cl, voice) for cl in clients]
    underruns = errors = 0
    vram: list[int] = []
    t_end = time.perf_counter() + seconds
    i = 0
    while time.perf_counter() < t_end:
        rs = await asyncio.gather(
            *(
                speak(cl, 1000 + i, 1000 + i, v, SENTENCES[i % 10], i)
                for cl, v in zip(clients, vids, strict=True)
            )
        )
        for cl in clients:
            frames = cl.audio.pop(1000 + i, [])
            if frames:
                t0 = frames[0][0]
                for k, (t, _f) in enumerate(frames):
                    if t - t0 > k * 0.040 + 0.2:
                        underruns += 1
                        break
            errors += len(cl.drain("error"))
            pongs = cl.drain("pong")
            vram += [p["vram_used_mib"] for p in pongs if "vram_used_mib" in p]
        del rs
        i += 1
    for cl in clients:
        await cl.close()
    return {
        "soak_underruns": underruns,
        "soak_errors": errors,
        "soak_vram_drift_mib": (max(vram) - min(vram)) if vram else None,
        "soak_turns": i,
    }


def verdict(res: dict[str, Any]) -> tuple[list[str], bool]:
    lines = []
    ok = True
    for key, (op, limit) in THRESHOLDS.items():
        v = res.get(key)
        if v is None:
            lines.append(f"  {key:34} {'-':>10}   (not measured)")
            continue
        good = v <= limit if op == "<=" else v >= limit
        ok &= good
        lines.append(f"  {key:34} {v:>10}   {op} {limit:<6} {'PASS' if good else 'FAIL'}")
    for gate in BOX_GATES:
        lines.append(f"  {gate:60} (box harness)")
    return lines, ok


def main_bench(
    url: str,
    token: str,
    voice: bytes,
    *,
    n: int,
    corpus: Path | None,
    soak_s: float,
    out: Path | None,
) -> int:
    res = asyncio.run(run(url, token, voice=voice, n=n, corpus=corpus, soak_s=soak_s))
    lines, ok = verdict(res)
    print(json.dumps(res, indent=1, default=str))
    print("\nphase-1 gates (SPEC-speechforge §11.3):")
    print("\n".join(lines))
    print(f"\n{'ALL MEASURED GATES PASS' if ok else 'SOME GATES FAIL'}")
    if out is not None:
        out.write_text(json.dumps(res, indent=1, default=str))
    return 0 if ok else 1


def tts_line() -> str:
    return TTS_LINE
