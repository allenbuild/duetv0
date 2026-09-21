"""CoMind kinematic feature extraction on the shared sync-frame index.

Frame-time anchor
-----------------
Each ``<role>_trimmed_sync.mp4`` carries a single ``desc`` metadata value in its
``moov`` atom: the DEVICE_TIME (ns) of MP4 frame 0. This was cross-checked against
the exhaustive VRS image-matching result for recording 43276420 (helper offset of
exactly 7 native frames, leader 0). Frame ``i`` maps to
``T0 + i * NATIVE_FRAME_NS`` where ``NATIVE_FRAME_NS`` is the measured Aria RGB
capture interval (33,327,337 ns). Residual error from dropped/repeated frames is
bounded by a few frames (~100 ms) over a 12-minute video, which is acceptable for
second-scale event anticipation, but NOT for sub-frame geometric work.

All positions are in the wearer's Aria device frame, meters. Hands with
``tracking_confidence == -1`` are missing and zero-filled with ``valid=0``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_FRAME_NS = 33_327_337
ROLES = ("leader", "helper")
# Project Aria hand landmark ordering: 0-4 fingertips (thumb..pinky), 5 wrist, 20 palm center.
KEY_LANDMARKS = (0, 1, 2, 3, 4, 5, 20)
HAND_MAX_GAP_US = 20_000
GAZE_MAX_GAP_US = 60_000

# Per-hand: 7 keypoints*3 + palm normal 3 + confidence 1 + valid 1 = 26; two hands = 52.
# Gaze: point 3 + direction 3 + valid 1 = 7.  Per-person total = 59.
HAND_DIM = 26
GAZE_DIM = 7
PERSON_DIM = 2 * HAND_DIM + GAZE_DIM


@dataclass(frozen=True)
class Mp4Anchor:
    frame0_device_time_ns: int
    n_frames: int
    duration_s: float


def _iter_atoms(buf: bytes, start: int, end: int):
    off = start
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off : off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[off + 8 : off + 16])[0]
            hdr = 16
        if size < hdr:
            return
        yield typ, off + hdr, off + size
        off += size


def parse_mp4_tail(tail: bytes) -> Mp4Anchor:
    """Extract frame-0 DEVICE_TIME and frame count from the trailing bytes of an MP4.

    The tail must contain the complete ``moov`` atom (2 MB suffices for CoMind).
    """
    m = tail.find(b"moov")
    if m < 4:
        raise ValueError("moov atom not found in MP4 tail")
    moov_start = m - 4
    moov_size = struct.unpack(">I", tail[moov_start : moov_start + 4])[0]
    moov_end = moov_start + moov_size
    desc = tail.find(b"desc\x00\x00\x00", moov_start, moov_end)
    if desc < 0:
        raise ValueError("desc metadata not found in moov")
    # desc atom -> data atom (8 hdr + 4 type + 4 locale) -> payload
    data_at = tail.find(b"data", desc, desc + 64)
    data_size = struct.unpack(">I", tail[data_at - 4 : data_at])[0]
    payload = tail[data_at + 12 : data_at - 4 + data_size].decode("ascii").strip()
    t0 = int(payload)

    n_frames = None
    duration_s = None
    for typ, a, b in _iter_atoms(tail, moov_start + 8, moov_end):
        if typ == b"mvhd":
            ver = tail[a]
            if ver == 0:
                timescale, duration = struct.unpack(">II", tail[a + 12 : a + 20])
            else:
                timescale = struct.unpack(">I", tail[a + 20 : a + 24])[0]
                duration = struct.unpack(">Q", tail[a + 24 : a + 32])[0]
            duration_s = duration / timescale
        if typ == b"trak":
            for t2, a2, b2 in _iter_atoms(tail, a, b):
                if t2 != b"mdia":
                    continue
                is_video = False
                for t3, a3, b3 in _iter_atoms(tail, a2, b2):
                    if t3 == b"hdlr":
                        is_video = tail[a3 + 8 : a3 + 12] == b"vide"
                    if t3 == b"minf" and is_video:
                        for t4, a4, b4 in _iter_atoms(tail, a3, b3):
                            if t4 != b"stbl":
                                continue
                            for t5, a5, b5 in _iter_atoms(tail, a4, b4):
                                if t5 == b"stsz":
                                    sample_size, count = struct.unpack(">II", tail[a5 + 4 : a5 + 12])
                                    n_frames = int(count)
    if n_frames is None or duration_s is None:
        raise ValueError("could not parse frame count / duration from moov")
    return Mp4Anchor(frame0_device_time_ns=t0, n_frames=n_frames, duration_s=duration_s)


def frame_device_times_ns(anchor: Mp4Anchor) -> np.ndarray:
    return anchor.frame0_device_time_ns + np.arange(anchor.n_frames, dtype=np.int64) * NATIVE_FRAME_NS


def _nearest(sample_ts_us: np.ndarray, query_ts_us: np.ndarray, max_gap_us: int):
    """Nearest-sample index for each query, with validity mask (sorted sample_ts)."""
    idx = np.searchsorted(sample_ts_us, query_ts_us)
    idx_lo = np.clip(idx - 1, 0, len(sample_ts_us) - 1)
    idx_hi = np.clip(idx, 0, len(sample_ts_us) - 1)
    d_lo = np.abs(sample_ts_us[idx_lo] - query_ts_us)
    d_hi = np.abs(sample_ts_us[idx_hi] - query_ts_us)
    best = np.where(d_lo <= d_hi, idx_lo, idx_hi)
    gap = np.minimum(d_lo, d_hi)
    return best, gap <= max_gap_us


def load_hand_features(csv_path: Path, query_ts_ns: np.ndarray) -> np.ndarray:
    """Return [N, 52] hand features on the query timeline (device frame, meters)."""
    cols = ["tracking_timestamp_us", "left_tracking_confidence", "right_tracking_confidence"]
    for side in ("left", "right"):
        for k in KEY_LANDMARKS:
            cols += [f"{ax}_{side}_landmark_{k}_device" for ax in ("tx", "ty", "tz")]
        cols += [f"n{ax}_{side}_palm_device" for ax in "xyz"]
    df = pd.read_csv(csv_path, usecols=cols)
    ts = df["tracking_timestamp_us"].to_numpy(np.int64)
    order = np.argsort(ts, kind="stable")
    df = df.iloc[order]
    ts = ts[order]
    idx, ok = _nearest(ts, query_ts_ns // 1000, HAND_MAX_GAP_US)
    out = np.zeros((len(query_ts_ns), 2 * HAND_DIM), np.float32)
    for h, side in enumerate(("left", "right")):
        conf = df[f"{side}_tracking_confidence"].to_numpy(np.float32)[idx]
        valid = ok & (conf != -1.0)
        feats = []
        for k in KEY_LANDMARKS:
            feats.append(df[[f"tx_{side}_landmark_{k}_device", f"ty_{side}_landmark_{k}_device", f"tz_{side}_landmark_{k}_device"]].to_numpy(np.float32)[idx])
        feats.append(df[[f"nx_{side}_palm_device", f"ny_{side}_palm_device", f"nz_{side}_palm_device"]].to_numpy(np.float32)[idx])
        block = np.concatenate(feats, axis=1)  # 24
        valid &= np.isfinite(block).all(axis=1)  # a few CSV rows contain NaN; treat as missing
        block[~valid] = 0.0
        sl = slice(h * HAND_DIM, (h + 1) * HAND_DIM)
        out[:, sl.start : sl.start + 24] = block
        out[:, sl.start + 24] = np.where(valid, np.clip(conf, 0, 1), 0.0)
        out[:, sl.start + 25] = valid.astype(np.float32)
    return out


def load_gaze_features(csv_path: Path, query_ts_ns: np.ndarray) -> np.ndarray:
    """Return [N, 7] gaze features: 3D gaze point (CPF frame, m), unit direction, valid."""
    df = pd.read_csv(csv_path, usecols=["tracking_timestamp_us", "left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf", "depth_m"])
    ts = df["tracking_timestamp_us"].to_numpy(np.int64)
    order = np.argsort(ts, kind="stable")
    df = df.iloc[order]
    ts = ts[order]
    idx, ok = _nearest(ts, query_ts_ns // 1000, GAZE_MAX_GAP_US)
    yaw = 0.5 * (df["left_yaw_rads_cpf"].to_numpy(np.float32) + df["right_yaw_rads_cpf"].to_numpy(np.float32))[idx]
    pitch = df["pitch_rads_cpf"].to_numpy(np.float32)[idx]
    depth = df["depth_m"].to_numpy(np.float32)[idx]
    # CPF convention: +Z forward, yaw about Y, pitch about X (direction only; sign convention
    # is consistent within the dataset which is all a learned model needs).
    d = np.stack([np.tan(yaw), np.tan(pitch), np.ones_like(yaw)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    depth = np.where(np.isfinite(depth) & (depth > 0) & (depth < 10), depth, 1.0)
    point = d * depth[:, None]
    ok &= np.isfinite(point).all(axis=1)
    out = np.concatenate([point, d, ok[:, None].astype(np.float32)], axis=1).astype(np.float32)
    out[~ok, :6] = 0.0
    return out


def build_recording_kinematics(raw_root: Path, proc_root: Path, recording_id: str) -> dict:
    """Build per-frame features for both roles on the shared sync-frame index.

    Returns dict with 'features' [N, 2*PERSON_DIM] (leader block first), 'n_frames',
    'anchors', and coverage stats. Raises FileNotFoundError if a role lacks hands.
    """
    rec = raw_root / "recordings" / recording_id
    anchors = {}
    for role in ROLES:
        tail = (proc_root / recording_id / f"mp4_tail_{role}.bin").read_bytes()
        anchors[role] = parse_mp4_tail(tail)
    n = min(a.n_frames for a in anchors.values())
    blocks = []
    stats = {}
    for role in ROLES:
        ts = frame_device_times_ns(anchors[role])[:n]
        hands = load_hand_features(rec / f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv", ts)
        gaze = load_gaze_features(rec / f"mps_{role}_trimmed_vrs/eye_gaze/general_eye_gaze.csv", ts)
        blocks.append(np.concatenate([hands, gaze], axis=1))
        stats[role] = {
            "n_frames_mp4": anchors[role].n_frames,
            "frame0_device_time_ns": anchors[role].frame0_device_time_ns,
            "left_hand_valid": float(hands[:, 25].mean()),
            "right_hand_valid": float(hands[:, HAND_DIM + 25].mean()),
            "gaze_valid": float(gaze[:, 6].mean()),
        }
    feats = np.concatenate(blocks, axis=1)
    assert feats.shape == (n, 2 * PERSON_DIM), feats.shape
    return {"features": feats, "n_frames": n, "stats": stats}


def person_slice(role: str) -> slice:
    i = ROLES.index(role)
    return slice(i * PERSON_DIM, (i + 1) * PERSON_DIM)
