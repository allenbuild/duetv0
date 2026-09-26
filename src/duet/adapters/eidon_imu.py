"""Eidon Tracker POV IMU (eidon-ai/tracker-pov-imu): parquet schema, body-slot map and per-slot sample parsing.

One row per (time_ms, slot). time_ms = ms since the start of the IMU recording, on the IMU's own clock (not wall
clock). Snapshots arrive at ~24 Hz, shared by all slots. 20-40 % of snapshots repeat the previous quaternion
(sample-and-hold of the BLE stream). Slots per the dataset card: 0 left_hand, 1 left_forearm, 2 left_shoulder (upper
arm), 3 right_hand, 4 right_forearm, 5 right_shoulder (upper arm), 6 chest. quat_[xyzw] are [x, y, z, w],
sensor -> world, world Z-up. accel/gyro/mag are null on ~79 % of recordings; where present their axes are rotated
180 deg about Z relative to the quaternion body frame (body = diag(-1, -1, 1) @ raw). The card says camera and IMU
capture were started "back to back". Recording 10004 nevertheless has the IMU ~0.4 s ahead of the video, so clock
alignment is estimated downstream (duet.playground.imu_arm), not assumed. Checked on recording 10004.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS = ("left_hand", "left_forearm", "left_shoulder", "right_hand", "right_forearm", "right_shoulder", "chest")
SLOT = {name: i for i, name in enumerate(SLOTS)}
QUAT_COLUMNS = ("quat_x", "quat_y", "quat_z", "quat_w")  # [x, y, z, w], sensor -> world, world Z-up
REQUIRED_COLUMNS = ("time_ms", "slot", *QUAT_COLUMNS)


def load_imu(path) -> pd.DataFrame:
    """Read an Eidon-schema IMU parquet (REQUIRED_COLUMNS must be present)."""
    imu = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(imu.columns)
    if missing:
        raise ValueError(f"IMU parquet lacks columns {sorted(missing)} (Eidon 7-slot schema expected)")
    return imu


def slot_samples(imu: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per slot: (t_imu_s [N], strictly increasing, IMU clock seconds = time_ms / 1000; q [N,4] unit quats [x,y,z,w]).

    Rows with non-finite or degenerate quaternions (|q| outside 0.5-1.5) are dropped, and so are repeated time
    stamps (the first is kept)."""
    out = {}
    for slot, d in imu.groupby("slot"):
        if not 0 <= int(slot) < len(SLOTS):
            raise ValueError(f"unknown IMU slot {slot} (Eidon schema has 0-6)")
        d = d.sort_values("time_ms", kind="stable")
        t = d["time_ms"].to_numpy(np.float64) / 1000.0
        q = d[list(QUAT_COLUMNS)].to_numpy(np.float64)
        nrm = np.linalg.norm(q, axis=1)
        ok = np.isfinite(t) & np.isfinite(nrm) & (nrm > 0.5) & (nrm < 1.5)
        t, q = t[ok], q[ok] / nrm[ok, None]
        keep = np.r_[True, np.diff(t) > 0]
        if len(t) and keep.any():
            out[int(slot)] = (t[keep], q[keep])
    return out
