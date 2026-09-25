"""Upper-body arm chain from a 7-slot IMU harness (Eidon Tracker schema).

Ported from Eidon Sim (MIT, github.com/Eidon-AI/eidon-sim: docs/kinematics.md, core/mathUtils.ts,
core/chestUtils.ts). Their web app does this live over WebHID; we do it offline on the
recorded parquet so the same arm chain can be drawn next to video and compared with vision.

Slots: 0 left_hand, 1 left_forearm, 2 left_shoulder(upper arm), 3 right_hand, 4 right_forearm,
5 right_shoulder(upper arm), 6 chest. Quaternions are [x, y, z, w], sensor->world.

Eidon's frame mapping (quaternionToVectors): each sensor's device +Y is the bone "forward"
(pointing distally) and device +Z is the bone "up"; scene axes are (-x_dev, z_dev, y_dev) i.e.
scene +Y up, +Z forward. Chest yaw (method 3 in chestUtils) removes heading so the chain is
torso-relative and invariant to which way the wearer faces.

Chain: shoulder origins at (-/+0.15, 0.05, -0.25)... Eidon uses (-0.25, 0.05, +-0.15) in a
different axis convention; we place shoulders at x = +-0.18, y = 1.40 (standing) and grow
humerus 0.30 m, radius 0.26 m, hand 0.10 m along each bone's forward vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .episode import Episode

SLOTS = ["left_hand", "left_forearm", "left_shoulder", "right_hand", "right_forearm", "right_shoulder", "chest"]
L_HUMERUS, L_RADIUS, L_HAND = 0.30, 0.26, 0.10
SHOULDER = {"left": np.array([-0.18, 1.40, 0.0]), "right": np.array([0.18, 1.40, 0.0])}


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v [N,3] by quaternions q [N,4] (x,y,z,w)."""
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    u = q[:, :3]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def sensor_vectors(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eidon quaternionToVectors: (fwd, up) in scene space for [N,4] quats."""
    n = len(q)
    fwd_d = quat_rotate(q, np.tile([0.0, 1.0, 0.0], (n, 1)))
    up_d = quat_rotate(q, np.tile([0.0, 0.0, 1.0], (n, 1)))
    to_scene = lambda v: np.stack([-v[:, 0], v[:, 2], v[:, 1]], 1)
    return to_scene(fwd_d), to_scene(up_d)


def yaw_rotate(v: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate scene vectors about +Y by -yaw (remove heading)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], 1)


def arm_chain(imu: pd.DataFrame, rate_hz: float = 24.0) -> dict:
    """Resample each slot to a common grid and build the 3D arm chain per sample.

    Returns dict(t_s [T], points [T, 2 sides, 4 joints, 3] (shoulder, elbow, wrist, fingertip), valid [T, 2],
                 chest_yaw [T], elbow_flex_deg [T, 2]).
    """
    t0, t1 = imu.time_ms.min(), imu.time_ms.max()
    grid = np.arange(t0, t1, 1000.0 / rate_hz)
    Q = {}
    for slot in range(7):
        d = imu[imu.slot == slot].sort_values("time_ms")
        if len(d) < 2:
            continue
        q = np.stack([np.interp(grid, d.time_ms.to_numpy(), d[c].to_numpy()) for c in ("quat_x", "quat_y", "quat_z", "quat_w")], 1)
        Q[slot] = q / np.linalg.norm(q, axis=1, keepdims=True)
    T = len(grid)
    # chest yaw from the chest sensor's up-vector projection (Eidon chestUtils method 2), else 0
    yaw = np.zeros(T)
    if 6 in Q:
        _, up = sensor_vectors(Q[6]); yaw = np.arctan2(up[:, 0], up[:, 2])
    points = np.full((T, 2, 4, 3), np.nan); valid = np.zeros((T, 2), bool); flex = np.full((T, 2), np.nan)
    for si, (side, up_slot, fa_slot, hand_slot) in enumerate((("left", 2, 1, 0), ("right", 5, 4, 3))):
        if up_slot not in Q or fa_slot not in Q:
            continue
        f_up = yaw_rotate(sensor_vectors(Q[up_slot])[0], yaw)
        f_fa = yaw_rotate(sensor_vectors(Q[fa_slot])[0], yaw)
        f_h = yaw_rotate(sensor_vectors(Q[hand_slot])[0], yaw) if hand_slot in Q else f_fa
        p0 = np.tile(SHOULDER[side], (T, 1)); p1 = p0 + f_up * L_HUMERUS; p2 = p1 + f_fa * L_RADIUS; p3 = p2 + f_h * L_HAND
        points[:, si] = np.stack([p0, p1, p2, p3], 1); valid[:, si] = True
        flex[:, si] = np.degrees(np.arccos(np.clip((f_up * f_fa).sum(1), -1, 1)))
    return {"t_s": grid / 1000.0, "points": points.astype(np.float32), "valid": valid, "chest_yaw": yaw.astype(np.float32), "elbow_flex_deg": flex.astype(np.float32)}


def imu_arm(ep: Episode) -> None:
    ep.set_status("imu_arm", "running")
    out_dir = ep.derived / "imu_arm"; out_dir.mkdir(exist_ok=True); done = []
    for s in ep.streams:
        if not s.imu:
            continue
        imu = pd.read_parquet(ep.dir / s.imu)
        res = arm_chain(imu)
        # IMU time_ms is relative to the recording start of that stream -> shift to reference time
        res["t_s"] = res["t_s"] + s.offset_s
        np.savez_compressed(out_dir / f"{s.name}.npz", **res); done.append(s.name)
    ep.set_status("imu_arm", "done" if done else "skipped", f"arm chains for {done}" if done else "no IMU streams")
