"""Cross-stream time alignment by audio cross-correlation (stage "align").

Every camera hears the same room. Time convention (C1, seconds): t_stream is a stream's own timeline with 0 at its
first VIDEO frame (container pts - ``video_start_s``); t_ref is the reference stream's. Per stream, align estimates

    t_ref = (1 + drift_ppm * 1e-6) * t_stream + offset_s        (convert with ep.ref_time / ep.stream_time only)

* Audio: the first audio track, mono 8 kHz, decoded on the frames' own timestamps (``-copyts``) and padded with
  silence / trimmed so that sample 0 is t_stream 0 (a late audio start, AAC priming and edit lists are honoured;
  gaps > 0.1 s are filled). Streamed from ffmpeg in chunks: memory is the envelope only (~3 MB per hour).
* Envelope: onset strength per 10 ms hop = positive change of log(1 + 1e3 * hop energy); digital silence (padding,
  filled gaps) contributes no onsets. A track peaking below ``SILENT_DBFS`` is "silent".
* Correlation: envelopes clipped at the ``CLIP_Q`` quantile of their above-median values (a few loud transients
  must not dominate a short overlap), then LINEAR cross-correlation (FFT length >= len(ref) + len(sig); full
  signals, nothing truncated) over the lags |L| <= max_lag_s whose overlap is >= min_overlap_s, normalised per lag
  by the overlap: s(L) = pearson_r(overlap) * sqrt(overlap), ~N(0, 1)-like for unrelated audio at any overlap.
  Lag L > 0: the stream started L hops after the reference (its sample i hears what the reference hears at i + L).
* Acceptance (all required): robust z-score of the peak ((s - median) / (1.4826 MAD) over the searched lags)
  >= ``Z_MIN``; peak / best peak >= ``SEP_S`` away >= ``RATIO_MIN`` (periodic sounds are ambiguous); peak not within
  ``EDGE_S`` of +-max_lag (the true offset may lie outside the window: raise ``ep.max_lag_s``).
  ``offset_confidence`` = that ratio. Calibration (synthetic speech-like scenes: random bursts and claps, reverb,
  independent mic noise; 600 unrelated pairs per case): the old rule (unclipped, ratio >= 1.3 only) accepted 6-12 %
  of unrelated 20-60 s pairs; this rule accepts 0-0.33 % (worst: 30 s clips) and 0 % of 600 s pairs, and accepted no
  wrong offset in 800 related pairs, while aligning 100 % of 10 s overlaps at 10 dB SNR, 68 % of 20 s at 0 dB and
  79 % of 120 s at -5 dB. Missed alignments are surfaced ("unaligned" + note); wrong ones would be silent.
* Drift: when the overlap is >= ``DRIFT_MIN_S``, offsets are re-measured in ``DRIFT_WIN_S`` windows (search
  +-(1 s + ``DRIFT_MAX_PPM`` over the overlap) around the global lag) and a robust line (Theil-Sen, then least
  squares on inliers) gives ``offset_s`` (intercept at t_stream 0) and ``drift_ppm``. The windows also rescue long
  recordings whose global peak is smeared by drift. Shorter overlaps: drift 0 (noted).
* Per-stream result (``offset_status``): "reference"; "manual" (``offset_override_s`` set: used as is, confidence
  None, drift 0); "audio"; or "unaligned" (no / silent / undecodable audio, or no accepted peak): offset 0, excluded
  from the common window, skipped by consumers (``ep.usable``). A reference without usable audio: if it was only
  the default (``ep.reference_explicit`` False) and no stream has an override, align switches the reference to the
  stream with the longest usable audio (ties by name) and records it; otherwise the stage FAILS unless every other
  stream has an override.
* Common window: [max start, min end] in reference time of ``ep.ref_time(s, [0, duration_s])`` over the usable
  streams; the stage fails if it is shorter than ``min_overlap_s``.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .episode import Episode, Stream
from .runtime import media_input

SR, HOP = 8000, 80  # mono 8 kHz, 10 ms hops
DT = HOP / SR  # envelope sample period [s]
MIN_OVERLAP_S = 5.0  # estimate_offset default (align uses ep.min_overlap_s): minimum audio overlap per lag
CLIP_Q = 0.98  # envelopes are clipped at this quantile of their above-median values before correlating
Z_MIN = 8.0  # robust z-score a correlation peak needs
RATIO_MIN = 1.5  # peak / best peak >= SEP_S away (unrelated audio: 90 % of pairs below ~1.3)
SEP_S = 0.5
EDGE_S = 0.5  # a peak this close to +-max_lag may be the flank of an offset outside the window
SILENT_DBFS = -70.0  # peak level below which a track counts as silent
DRIFT_MIN_S, DRIFT_WIN_S, DRIFT_MAX_WINDOWS, DRIFT_MIN_WINDOWS = 300.0, 60.0, 40, 4
DRIFT_MAX_PPM = 500.0  # drift windows search +-(1 s + this over the overlap) around the global lag
CONF_CAP = 100.0  # offset_confidence is capped (JSON has no inf)


# ------------------------------------------------------------------------------------------ audio -> envelope

def onset_envelope(energy: np.ndarray) -> np.ndarray:
    """Onset strength per hop from per-hop energy (sum of squared samples, full scale = 1): positive first difference
    of log(1 + 1e3 * energy). Digital-silence hops (padding before a late audio start, filled gaps) and the hop right
    after them get 0, so padding boundaries do not create onsets."""
    e = np.asarray(energy, np.float64)
    le = np.log1p(e * 1e3)
    on = np.diff(le, prepend=le[:1]); on[on < 0] = 0.0
    absent = e < 1e-9
    on[absent] = 0.0; on[1:][absent[:-1]] = 0.0
    return on


def decode_envelope(path: str | Path, video_start_s: float = 0.0, timeout_s: float | None = None) -> tuple[np.ndarray | None, float, str]:
    """Onset envelope of the first audio track of ``path``; index i <-> t_stream = i * DT s, i.e. sample 0 is container
    time ``video_start_s`` (the stream's first video frame). Returns (envelope or None, peak level dBFS, error ('' if
    ok)). The decoded audio is streamed and reduced to hop energies on the fly, so memory stays bounded."""
    first = round(video_start_s * SR)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-copyts", *media_input(path), "-map", "0:a:0",
           "-vn", "-sn", "-dn", "-af", f"aresample={SR},aresample=async=1:first_pts={first}", "-ac", "1", "-f", "f32le", "pipe:1"]
    energy, peak, rest, killed = [], 0.0, b"", threading.Event()
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        timer = threading.Timer(timeout_s, lambda: (killed.set(), p.kill())) if timeout_s else None
        if timer:
            timer.daemon = True; timer.start()
        try:
            while chunk := p.stdout.read(HOP * 4 * 4096):  # ~41 s of audio per read
                buf = rest + chunk; n = len(buf) // (HOP * 4); rest = buf[n * HOP * 4:]
                if n:
                    x = np.frombuffer(buf, np.float32, n * HOP).reshape(n, HOP).astype(np.float64)
                    energy.append(np.einsum("ij,ij->i", x, x)); peak = max(peak, float(np.abs(x).max()))
        finally:
            p.stdout.close(); rc = p.wait()
            if timer:
                timer.cancel()
        if rc != 0 or killed.is_set():
            err.seek(0); tail = err.read()[-400:].decode(errors="replace").strip().replace("\n", " | ")
            return None, -math.inf, ("timed out" if killed.is_set() else f"ffmpeg exit {rc}") + (f": {tail}" if tail else "")
    e = np.concatenate(energy) if energy else np.zeros(0)
    if len(e) < round(1.0 / DT):
        return None, -math.inf, f"only {len(e) * DT:.2f} s of audio decoded"
    return onset_envelope(e), (20 * math.log10(peak) if peak > 0 else -math.inf), ""


# ------------------------------------------------------------------------------------------ correlation

def robust_envelope(env: np.ndarray) -> np.ndarray:
    """``env`` clipped at the CLIP_Q quantile of its above-median values, so a few loud transients (claps, door
    slams) cannot dominate a correlation over a short overlap."""
    e = np.asarray(env, np.float64); m = e > np.median(e)
    return np.minimum(e, np.quantile(e[m], CLIP_Q)) if m.any() else e


def xcorr_score(ref: np.ndarray, sig: np.ndarray, lo: int, hi: int, min_overlap: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Overlap-normalised LINEAR cross-correlation of two envelopes for integer lags lo..hi (hops).

    Lag L: sig[i] lines up with ref[i + L] (L > 0: sig started L hops after ref, t_ref = t_sig + L * DT). The range
    is clamped to the lags whose overlap is >= min_overlap. Returns (lags, s), s(L) = r(L) * sqrt(n(L)) with r the
    Pearson correlation over the n(L) overlapping samples (NaN where either side is constant); empty arrays when no
    lag qualifies."""
    ref = np.asarray(ref, np.float64); sig = np.asarray(sig, np.float64)
    assert ref.ndim == 1 and sig.ndim == 1, "envelopes must be 1-D"
    nr, ns = len(ref), len(sig); min_overlap = max(int(min_overlap), 2)
    # overlap(L) = min(ns + min(0, L), nr - max(0, L)) >= min_overlap  <=>  both envelopes are at least min_overlap long
    # and min_overlap - ns <= L <= nr - min_overlap
    lo, hi = max(int(lo), min_overlap - ns), min(int(hi), nr - min_overlap)
    if lo > hi or min(nr, ns) < min_overlap:
        return np.zeros(0, np.int64), np.zeros(0)
    ref = (ref - ref.mean()) / (ref.std() or 1.0); sig = (sig - sig.mean()) / (sig.std() or 1.0)  # conditioning only
    m = 1 << (nr + ns - 2).bit_length()  # >= nr + ns - 1: linear correlation, no circular wrap-around
    lags = np.arange(lo, hi + 1)
    cc = np.fft.irfft(np.fft.rfft(ref, m) * np.conj(np.fft.rfft(sig, m)), m)[lags % m]  # sum_i ref[i + L] * sig[i]
    assert len(cc) == len(lags)
    i0, i1 = np.maximum(0, -lags), np.minimum(ns, nr - lags)  # overlapping sig samples [i0, i1); ref [i0 + L, i1 + L)
    n = (i1 - i0).astype(np.float64)
    assert n.min() >= min_overlap
    cr, cr2, cs, cs2 = (np.concatenate(([0.0], np.cumsum(v))) for v in (ref, ref * ref, sig, sig * sig))
    sx, sxx = cr[i1 + lags] - cr[i0 + lags], cr2[i1 + lags] - cr2[i0 + lags]
    sy, syy = cs[i1] - cs[i0], cs2[i1] - cs2[i0]
    cov, vx, vy = cc - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
    ok = (vx > 1e-9 * n) & (vy > 1e-9 * n)
    r = np.full(len(lags), np.nan); r[ok] = cov[ok] / np.sqrt(vx[ok] * vy[ok])
    return lags, np.clip(r, -1.0, 1.0) * np.sqrt(n)


@dataclass
class Peak:
    lag: float  # hops (parabolic sub-hop refinement)
    z: float  # robust z-score: (s - median) / (1.4826 * MAD) over the searched lags
    ratio: float  # s(peak) / max s over lags >= SEP_S away (inf when none is positive)


def find_peak(lags: np.ndarray, s: np.ndarray) -> Peak | None:
    """Best lag of an ``xcorr_score`` curve with its robust z-score and peak ratio (NaN lags ignored); None if flat
    or < 3 valid lags."""
    v = np.isfinite(s)
    if v.sum() < 3:
        return None
    b = int(np.nanargmax(s)); med = float(np.median(s[v])); mad = 1.4826 * float(np.median(np.abs(s[v] - med)))
    if not mad > 0:
        return None
    far = v & (np.abs(lags - lags[b]) > SEP_S / DT)
    sec = float(s[far].max()) if far.any() else 0.0
    frac = 0.0
    if 0 < b < len(s) - 1 and v[b - 1] and v[b + 1]:
        a, c = s[b - 1], s[b + 1]; den = a - 2 * s[b] + c
        frac = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0
    return Peak(float(lags[b]) + frac, (float(s[b]) - med) / mad, float(s[b]) / sec if sec > 0 else math.inf)


# ------------------------------------------------------------------------------------------ offset + drift

@dataclass
class AudioOffset:
    """One stream's audio alignment against the reference: t_ref = (1 + drift_ppm * 1e-6) * t_stream + offset_s."""
    ok: bool
    offset_s: float = 0.0
    drift_ppm: float = 0.0
    confidence: float = 0.0  # peak ratio (global correlation, else median over the drift windows)
    z: float = 0.0  # robust z-score of the peak (global, else median over the drift windows)
    notes: list[str] = field(default_factory=list)  # human-readable diagnosis


@dataclass
class _Win:
    t: float  # window centre, stream time [s]
    lag: float  # hops
    z: float
    ratio: float


def _window_offsets(ref: np.ndarray, sig: np.ndarray, center: int | None, max_lag: int, search: int) -> tuple[list[_Win], int]:
    """Local offsets of up to DRIFT_MAX_WINDOWS DRIFT_WIN_S windows of ``sig`` spread over its overlap with ``ref``,
    searching center +- search hops (center None: the full +-max_lag range). Returns (accepted windows, windows
    tried)."""
    W = round(DRIFT_WIN_S / DT); nr, ns = len(ref), len(sig)
    if center is None:
        (lo_c, hi_c), (a, b) = (-max_lag, max_lag), (0, ns)
    else:
        (lo_c, hi_c), (a, b) = (max(-max_lag, center - search), min(max_lag, center + search)), (max(0, -center), min(ns, nr - center))
    k = int(min(DRIFT_MAX_WINDOWS, max(0, b - a) // W))
    out = []
    for i0 in (np.linspace(a, b - W, k).astype(int) if k else []):
        j0, j1 = max(0, i0 + lo_c), min(nr, i0 + W + hi_c)
        lags, s = xcorr_score(ref[j0:j1], sig[i0:i0 + W], lo_c - (j0 - i0), hi_c - (j0 - i0), int(0.8 * W))
        p = find_peak(lags, s)
        if p is not None and p.z >= Z_MIN and p.ratio >= RATIO_MIN and lo_c + 2 <= p.lag + (j0 - i0) <= hi_c - 2:
            out.append(_Win((i0 + W / 2) * DT, p.lag + (j0 - i0), p.z, p.ratio))
    return out, k


def _robust_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """y ~ a * t + b: Theil-Sen, then least squares on the inliers (|resid| <= max(3 hops, 4 robust sigma))."""
    i, j = np.triu_indices(len(t), 1); dt = t[j] - t[i]; k = dt > 0
    a = float(np.median((y[j] - y[i])[k] / dt[k])) if k.any() else 0.0
    b = float(np.median(y - a * t))
    res = y - (a * t + b); inl = np.abs(res) <= max(3 * DT, 4 * 1.4826 * float(np.median(np.abs(res))))
    if inl.sum() >= 2 and np.ptp(t[inl]) > 0:
        a, b = (float(v) for v in np.polyfit(t[inl], y[inl], 1))
    return a, b, inl


def estimate_offset(ref: np.ndarray, sig: np.ndarray, max_lag_s: float = 120.0, min_overlap_s: float = MIN_OVERLAP_S) -> AudioOffset:
    """Offset (and, for overlaps >= DRIFT_MIN_S, drift) of envelope ``sig`` against envelope ``ref``, both DT-spaced
    with index 0 at t_stream 0 of their stream. See the module docstring for the acceptance rules."""
    assert max_lag_s > 0 and min_overlap_s > 0
    max_lag, min_ov = round(max_lag_s / DT), max(2, round(min_overlap_s / DT))
    ref, sig = robust_envelope(ref), robust_envelope(sig); nr, ns = len(ref), len(sig)
    lags, s = xcorr_score(ref, sig, -max_lag, max_lag, min_ov)
    g = find_peak(lags, s)
    if g is None:
        why = (f"audio overlap < {min_overlap_s:g} s at every lag within +-{max_lag_s:g} s" if len(lags) < 3
               else "flat correlation (no audio structure)")
        return AudioOffset(False, notes=[why])
    edge = (lags[0] == -max_lag and g.lag - lags[0] < EDGE_S / DT) or (lags[-1] == max_lag and lags[-1] - g.lag < EDGE_S / DT)
    stats = f"peak {g.lag * DT:+.2f} s, z {g.z:.1f}, x{min(g.ratio, CONF_CAP):.2f}"
    if edge:
        return AudioOffset(False, confidence=min(g.ratio, CONF_CAP), z=g.z, notes=[
            f"correlation peak at the +-{max_lag_s:g} s search edge ({stats}); the offset may be larger: raise max_lag_s or set offset_override_s"])
    g_ok = g.z >= Z_MIN and g.ratio >= RATIO_MIN
    weak = f"no clear audio match within +-{max_lag_s:g} s ({stats}; need z >= {Z_MIN:g} and x{RATIO_MIN:g})"
    ov_s = (min(ns, nr - round(g.lag)) - max(0, -round(g.lag))) * DT
    if (ov_s if g_ok else min(nr, ns) * DT) < DRIFT_MIN_S:
        if not g_ok:
            return AudioOffset(False, confidence=min(g.ratio, CONF_CAP), z=g.z, notes=[weak])
        return AudioOffset(True, g.lag * DT, 0.0, min(g.ratio, CONF_CAP), g.z, [f"drift not estimated (overlap {ov_s:.0f} s < {DRIFT_MIN_S:g} s)"])
    # long recording: windowed offsets -> drift; also recovers a global peak smeared by drift
    search = round((1.0 + DRIFT_MAX_PPM * 1e-6 * min(nr, ns) * DT) / DT)
    center, notes = (round(g.lag) if g_ok else None), []
    if center is None:
        wins, _ = _window_offsets(ref, sig, None, max_lag, search)
        lag_w = np.array([w.lag for w in wins])
        if len(wins) >= DRIFT_MIN_WINDOWS:
            med = float(np.median(lag_w)); agree = int((np.abs(lag_w - med) <= search).sum())
            if agree >= max(DRIFT_MIN_WINDOWS, 0.5 * len(wins)):
                center = round(med); notes.append(f"global correlation weak ({stats}); {agree}/{len(wins)} windows agree")
        if center is None:
            return AudioOffset(False, confidence=min(g.ratio, CONF_CAP), z=g.z, notes=[weak + f"; {len(wins)} windows matched, no consensus"])
    wins, tried = _window_offsets(ref, sig, center, max_lag, search)
    ov_c = (min(ns, nr - center) - max(0, -center)) * DT  # overlap at the window search centre
    fit = None
    if len(wins) >= DRIFT_MIN_WINDOWS:
        t, y = np.array([w.t for w in wins]), np.array([w.lag for w in wins]) * DT
        a, b, inl = _robust_line(t, y)
        span = np.ptp(t[inl]) if inl.any() else 0.0
        if inl.sum() >= DRIFT_MIN_WINDOWS and span >= 0.5 * ov_c - DRIFT_WIN_S:
            rms = float(np.sqrt(np.mean((y[inl] - (a * t[inl] + b)) ** 2)))
            fit = (a, b, int(inl.sum()), rms)
    if fit is None:
        if not g_ok:
            return AudioOffset(False, confidence=min(g.ratio, CONF_CAP), z=g.z, notes=[weak + f"; only {len(wins)}/{tried} windows usable"])
        return AudioOffset(True, g.lag * DT, 0.0, min(g.ratio, CONF_CAP), g.z, [f"drift not estimated (only {len(wins)}/{tried} windows usable)"])
    a, b, n_in, rms = fit
    conf, z = (g.ratio, g.z) if g_ok else (float(np.median([w.ratio for w in wins])), float(np.median([w.z for w in wins])))
    notes.append(f"drift {a * 1e6:+.1f} ppm from {n_in}/{tried} windows (fit rms {rms * 1e3:.1f} ms)")
    return AudioOffset(True, b, round(a * 1e6, 3), min(conf, CONF_CAP), z, notes)


# ------------------------------------------------------------------------------------------ stage

def _audio_problem(s: Stream, dec: tuple[np.ndarray | None, float, str] | None, min_overlap_s: float) -> str:
    """'' if the stream's decoded audio is usable for alignment, else a short reason."""
    if not s.has_audio:
        return "no audio track"
    env, peak_db, err = dec if dec is not None else (None, -math.inf, "not decoded")
    if env is None:
        return f"audio decode failed ({err})"
    if peak_db < SILENT_DBFS:
        return f"silent audio (peak {peak_db:.0f} dBFS)"
    if len(env) * DT < min_overlap_s:
        return f"audio {len(env) * DT:.1f} s < min overlap {min_overlap_s:g} s"
    return ""


def align(ep: Episode, max_lag_s: float | None = None, min_overlap_s: float | None = None) -> None:
    """Stage "align": sets every stream's offset_s / drift_ppm / offset_status / offset_confidence and the common
    window [common_start_s, common_end_s] (reference time, s). ``max_lag_s`` / ``min_overlap_s`` default to
    ``ep.max_lag_s`` / ``ep.min_overlap_s``.

    A DEFAULT reference (``ep.reference_explicit`` False) without usable audio is replaced by the stream with the
    longest usable audio (ties by name; the switch is recorded in notes and the status detail), unless a stream has
    ``offset_override_s``: overrides are relative to the current reference, so it is then kept.

    Raises RuntimeError (stage failed; per-stream results and notes are saved, the window is set to [0, 0]) when the
    reference has no usable audio and some other stream has no ``offset_override_s`` (those streams are "unaligned"),
    or when the usable streams' common window is shorter than ``min_overlap_s``."""
    max_lag_s = float(ep.max_lag_s if max_lag_s is None else max_lag_s)
    min_overlap_s = float(ep.min_overlap_s if min_overlap_s is None else min_overlap_s)
    if not (math.isfinite(max_lag_s) and max_lag_s > 0 and math.isfinite(min_overlap_s) and min_overlap_s > 0):
        raise ValueError(f"need max_lag_s > 0 and min_overlap_s > 0, got {max_lag_s}, {min_overlap_s}")
    ep.set_status("align", "running")
    ref = ep.stream(ep.reference)
    others = [s for s in ep.streams if s is not ref]
    need = [s for s in others if s.offset_override_s is None]  # streams to align by audio
    for s in ep.streams:
        if not (math.isfinite(s.duration_s) and s.duration_s > 0 and math.isfinite(s.video_start_s)):
            raise RuntimeError(f"stream {s.name}: duration {s.duration_s} / video start {s.video_start_s} invalid (run probe first)")
    todo = [s for s in ([ref] + need if need else []) if s.has_audio]
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(todo)))) as pool:  # ffmpeg decodes in parallel
        futs = {s.name: pool.submit(decode_envelope, ep.dir / s.path, s.video_start_s, 120.0 + 0.5 * s.duration_s) for s in todo}
        dec = {k: f.result() for k, f in futs.items()}
    notes: list[str] = []
    ref_why, switched = (_audio_problem(ref, dec.get(ref.name), min_overlap_s) if need else ""), ""
    if ref_why and not ep.reference_explicit:  # default reference without usable audio: pick another
        overrides = [s.name for s in ep.streams if s.offset_override_s is not None]
        cands = [s for s in others if not _audio_problem(s, dec.get(s.name), min_overlap_s)]
        if cands and not overrides:
            new = min(cands, key=lambda s: (-len(dec[s.name][0]), s.name))  # longest usable audio, ties by name
            notes.append(f"reference {ref.name} was the default and has no usable audio ({ref_why}): switched to {new.name}, the "
                         f"longest usable audio ({len(dec[new.name][0]) * DT:.0f} s); choose the reference explicitly to override")
            switched = f"reference {ref.name} -> {new.name} (auto: {ref.name} {ref_why}); "
            ep.reference, ref, ref_why = new.name, new, ""
            others = [s for s in ep.streams if s is not ref]; need = [s for s in others if s.offset_override_s is None]
        else:
            ref_why += (f"; not switched automatically: manual offsets ({', '.join(overrides)}) are relative to it" if overrides
                        else "; no other stream has usable audio either")
    ref.offset_s, ref.drift_ppm, ref.offset_status, ref.offset_confidence = 0.0, 0.0, "reference", None
    if ref.offset_override_s not in (None, 0.0):
        notes.append(f"{ref.name}: offset_override_s ignored on the reference")
    summary = {ref.name: "reference"}
    for s in others:
        s.offset_confidence, s.drift_ppm = None, 0.0
        if s.offset_override_s is not None:
            s.offset_s, s.offset_status = float(s.offset_override_s), "manual"; summary[s.name] = f"manual {s.offset_s:+.3f}s"; continue
        why = f"reference {ref.name}: {ref_why}" if ref_why else _audio_problem(s, dec.get(s.name), min_overlap_s)
        r = AudioOffset(False, notes=[why]) if why else estimate_offset(dec[ref.name][0], dec[s.name][0], max_lag_s, min_overlap_s)
        if r.ok:
            s.offset_s, s.drift_ppm, s.offset_status, s.offset_confidence = float(r.offset_s), float(r.drift_ppm), "audio", float(r.confidence)
            dur_drift = abs(r.drift_ppm) * 1e-6 * s.duration_s
            if ep.proc_fps > 0 and dur_drift > 0.5 / ep.proc_fps:
                r.notes.append(f"drift moves this stream {dur_drift * 1e3:.0f} ms over {s.duration_s:.0f} s (> half a processed frame): "
                               "applied via drift_ppm; convert times only with ep.ref_time/ep.stream_time")
            summary[s.name] = f"{s.offset_s:+.3f}s x{r.confidence:.1f} z{r.z:.0f}" + (f" {s.drift_ppm:+.1f}ppm" if s.drift_ppm else "")
        else:
            s.offset_s, s.offset_status = 0.0, "unaligned"; summary[s.name] = "UNALIGNED"
        notes += [f"{s.name}: {n}" for n in r.notes]
    ep.notes = [n for n in ep.notes if not n.startswith("align:")] + [f"align: {n}" for n in notes]

    def fail(msg: str) -> None:
        ep.common_start_s = ep.common_end_s = 0.0; ep.save()
        raise RuntimeError(msg)

    if ref_why:
        fail(f"reference stream {ref.name} has no usable audio ({ref_why}); streams {', '.join(s.name for s in need)} need audio "
             "alignment: set offset_override_s for them or choose a reference with audio")
    usable = [s for s in ep.streams if ep.usable(s)]
    span = {s.name: (float(ep.ref_time(s, 0.0)), float(ep.ref_time(s, s.duration_s))) for s in usable}
    assert all(math.isfinite(a) and math.isfinite(b) and b > a for a, b in span.values()), span
    start, end = max(a for a, _ in span.values()), min(b for _, b in span.values())
    if not end - start >= min_overlap_s:
        fail(f"common window of the usable streams is {end - start:.2f} s (< {min_overlap_s:g} s); spans in reference time: "
             + ", ".join(f"{k} [{a:.2f}, {b:.2f}]" for k, (a, b) in span.items()) + ". Fix offsets (offset_override_s) or the reference.")
    ep.common_start_s, ep.common_end_s = start, end
    n_un = sum(s.offset_status == "unaligned" for s in ep.streams)
    ep.set_status("align", "done", switched + "; ".join(f"{k} {v}" for k, v in summary.items())
                  + f"; window [{start:.2f}, {end:.2f}] s" + (f"; {n_un} stream(s) unaligned, excluded (see notes)" if n_un else ""))
