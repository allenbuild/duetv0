"""Cross-stream time alignment by audio cross-correlation.

Every camera hears the same room. We extract mono 8 kHz audio from each stream, compute an
onset-strength envelope (positive change of log energy in 10 ms hops), and cross-correlate
each stream's envelope against the reference's over a +-max_lag window. The lag with the
highest normalised correlation gives ``offset_s`` (t_ref = t_stream + offset). Confidence is
the ratio of the best peak to the second-best peak at least 0.5 s away; below ~1.3 the
alignment should be checked by eye (a clap at the start of every block makes this trivial).

Streams without audio keep offset 0 and are flagged.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .episode import Episode

SR, HOP = 8000, 80  # 10 ms hops


def _audio_envelope(path: Path) -> np.ndarray | None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.f32"
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", str(wav)],
                           capture_output=True)
        if r.returncode != 0 or not wav.exists() or wav.stat().st_size < SR * 4:
            return None
        x = np.fromfile(wav, np.float32)
    n = len(x) // HOP
    e = np.log1p((x[: n * HOP].reshape(n, HOP) ** 2).sum(1) * 1e3)
    onset = np.diff(e, prepend=e[:1]); onset[onset < 0] = 0
    onset -= onset.mean(); onset /= (onset.std() + 1e-9)
    return onset


def _xcorr_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Return (lag in hops, confidence). lag > 0 means sig starts LATER than ref (sig events occur at ref_index - lag)."""
    n = min(len(ref), len(sig))
    ref, sig = ref[:n], sig[:n]
    L = int(2 ** np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(ref, L) * np.conj(np.fft.rfft(sig, L))
    cc = np.fft.irfft(F, L)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])  # lags -max_lag..max_lag
    lags = np.arange(-max_lag, max_lag + 1)
    best = int(np.argmax(cc)); peak = cc[best]
    mask = np.abs(lags - lags[best]) > 50  # 0.5 s away
    second = cc[mask].max() if mask.any() else 1e-9
    return int(lags[best]), float(peak / max(second, 1e-9))


def align(ep: Episode, max_lag_s: float = 120.0) -> None:
    ep.set_status("align", "running")
    env = {s.name: (_audio_envelope(ep.dir / s.path) if s.has_audio else None) for s in ep.streams}
    ref = env[ep.reference]
    notes = []
    for s in ep.streams:
        if s.name == ep.reference:
            s.offset_s, s.offset_confidence = 0.0, None; continue
        if ref is None or env[s.name] is None:
            s.offset_s, s.offset_confidence = 0.0, 0.0; notes.append(f"{s.name}: no audio, offset left at 0"); continue
        lag, conf = _xcorr_lag(ref, env[s.name], int(max_lag_s * 1000 / 10))
        # sig[i] aligns with ref[i + lag]  ->  t_ref = t_stream + lag*0.01
        s.offset_s, s.offset_confidence = lag * HOP / SR, conf
        if conf < 1.3:
            notes.append(f"{s.name}: low alignment confidence {conf:.2f}; check by eye")
    # common window in reference time
    starts = [s.offset_s for s in ep.streams]; ends = [s.offset_s + s.duration_s for s in ep.streams]
    ep.common_start_s, ep.common_end_s = max(starts), min(ends)
    ep.notes = [n for n in ep.notes if not n.startswith(("align:",))] + [f"align: {n}" for n in notes]
    ep.set_status("align", "done", "; ".join(f"{s.name} {s.offset_s:+.2f}s" + (f" (x{s.offset_confidence:.1f})" if s.offset_confidence else "") for s in ep.streams))
