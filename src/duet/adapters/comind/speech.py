"""Per-frame speech features from CoMind word-level transcripts (WhisperX-style JSON).

Each ``<role>_trimmed_sync_transcript.json`` is transcribed from that wearer's Aria
microphone; word ``start``/``end`` are seconds on the synced video timeline, so
frame ``i`` is at ``i / 30`` s and needs no device-time anchoring. Speaker labels
are per-file diarization clusters and are NOT used (they are not identity-stable).

Per role, per frame (SPEECH_DIM = 36):
  0  speaking now (a word is active)
  1  seconds since the last word ended (clipped to 10 s, /10)
  2  words in the last 3 s (/10)
  3  request-like keyword in the last 3 s
  4..35  hashed bag-of-words over the last 3 s, 32 bins, L1-normalised
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path

import numpy as np

FPS = 30
WINDOW_S = 3.0
N_BINS = 32
SPEECH_DIM = 4 + N_BINS
REQUEST_WORDS = {"pass", "hand", "give", "take", "grab", "need", "can", "could", "please", "here", "want", "get", "bring"}
_TOKEN = re.compile(r"[a-z']+")


def load_words(path: Path) -> list[tuple[float, float, str]]:
    d = json.load(open(path))
    out = []
    for seg in d.get("segments", []):
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                continue
            tok = _TOKEN.findall(w["word"].lower())
            if tok:
                out.append((float(w["start"]), float(w["end"]), tok[0]))
    out.sort()
    return out


def _bin(tok: str) -> int:
    return zlib.crc32(tok.encode()) % N_BINS


def speech_features(words, n_frames: int) -> np.ndarray:
    f = np.zeros((n_frames, SPEECH_DIM), np.float32)
    if not words:
        f[:, 1] = 1.0
        return f
    starts = np.array([w[0] for w in words]); ends = np.array([w[1] for w in words])
    t = np.arange(n_frames) / FPS
    # speaking now
    for s, e, _ in words:
        a, b = int(np.floor(s * FPS)), int(np.ceil(e * FPS))
        if b >= a:
            f[max(0, a): min(n_frames, b + 1), 0] = 1.0
    # time since last word end (causal)
    idx = np.searchsorted(ends, t, side="right") - 1
    last_end = np.where(idx >= 0, ends[np.clip(idx, 0, None)], -np.inf)
    f[:, 1] = np.clip(t - last_end, 0, 10.0) / 10.0
    # window counts / keywords / hashed BoW: words whose END falls in (t-3, t]
    win = int(WINDOW_S * FPS)
    cnt = np.zeros(n_frames + 1, np.float32); kw = np.zeros(n_frames + 1, np.float32); bow = np.zeros((n_frames + 1, N_BINS), np.float32)
    for s, e, tok in words:
        k = int(np.ceil(e * FPS))
        if 0 <= k < n_frames:
            cnt[k] += 1; bow[k, _bin(tok)] += 1
            if tok in REQUEST_WORDS:
                kw[k] += 1
    cs = np.cumsum(cnt); ck = np.cumsum(kw); cb = np.cumsum(bow, axis=0)
    lo = np.clip(np.arange(n_frames) - win, 0, None)
    f[:, 2] = (cs[:n_frames] - cs[lo]) / 10.0
    f[:, 3] = ((ck[:n_frames] - ck[lo]) > 0).astype(np.float32)
    b = cb[:n_frames] - cb[lo]
    f[:, 4:] = b / np.maximum(b.sum(1, keepdims=True), 1.0)
    return f


def build_recording_speech(raw_root: Path, recording_id: str, n_frames: int) -> np.ndarray:
    """[N, 2*SPEECH_DIM]: leader block then helper block (same order as kinematics)."""
    blocks = []
    for role in ("leader", "helper"):
        p = raw_root / "recordings" / recording_id / "transcripts" / f"{role}_trimmed_sync_transcript.json"
        blocks.append(speech_features(load_words(p) if p.exists() else [], n_frames))
    return np.concatenate(blocks, axis=1)
