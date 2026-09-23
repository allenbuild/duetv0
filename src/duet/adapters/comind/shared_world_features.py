"""Cross-person geometric features from Multi-SLAM shared-world trajectories.

Multi-SLAM (`multislam_output/<folder>/slam/closed_loop_trajectory.csv`) gives each
participant's T_world_device in ONE shared world frame (same graph_uid). Hands are
in the wearer's device frame; transforming both people's hands into the shared
world is what makes inter-person geometry observable.

Frame -> pose: each role's MP4 frame device time (see kinematics.py) is matched to
the nearest 1 kHz trajectory sample (max gap 20 ms).

Features per frame (WORLD_DIM = 34), all metres or unit-less, 0 where invalid:
  0..3   |leader wrist_i - helper wrist_j| for (L,R)x(L,R)
  4      min of the four wrist distances
  5      d/dt of (4), m/s (causal difference)
  6      head-to-head distance
  7..9   helper head position in leader device frame
  10..12 leader head position in helper device frame
  13..18 helper L/R wrist in leader device frame
  19..24 leader L/R wrist in helper device frame
  25     |leader gaze point (world) - nearest helper wrist|
  26     |helper gaze point (world) - nearest leader wrist|
  27     |leader gaze point - helper head|
  28     |helper gaze point - leader head|
  29     cos(angle between leader RGB optical axis and direction to helper head)
  30     cos(angle between helper RGB optical axis and direction to leader head)
  31     both poses valid
  32     leader hands valid (any)
  33     helper hands valid (any)

UNVERIFIED FRAME ASSUMPTION: the gaze point is produced in CPF (central pupil frame) by
kinematics.load_gaze_features and is treated here as a device-frame point. T_Device_CPF is
NOT present in MPS output (online_calibration.jsonl carries only T_Device_Camera and
T_Device_Imu); it lives in the factory calibration inside the VRS, which this pipeline does
not download. So the claim that CPF-to-device is a small offset is currently UNTESTED here.
Features 25-28 (gaze-to-partner distances) depend on it and should be treated as provisional
until the transform is obtained and applied. The forward axis (29-30) no longer guesses:
it uses the measured RGB optical axis; see rgb_forward_axis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from duet.adapters.comind.kinematics import (
    HAND_DIM, PERSON_DIM, ROLES, Mp4Anchor, frame_device_times_ns, parse_mp4_tail, person_slice,
)

WORLD_DIM = 34
POSE_MAX_GAP_US = 20_000
FPS = 30


# Nominal camera-rgb optical axis in the Aria device frame, measured from the first record of
# mps_leader_trimmed_vrs/slam/online_calibration.jsonl on recording 43276420. Used only when a
# recording's own rgb_calib_<role>.json is absent; it is 38.7 deg off device +Z, so it is a far
# better default than +Z, but per-recording calibration is preferred and fetched when available.
NOMINAL_RGB_AXIS_DEVICE = np.array([0.0878, -0.6196, 0.7800])


def quat_wxyz_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Project Aria JSON quaternion convention [w, [x, y, z]] -> 3x3 active rotation."""
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rgb_forward_axis(proc_dir: Path, role: str) -> np.ndarray:
    """Unit optical axis of camera-rgb in the wearer's device frame.

    The device frame is anchored to a SLAM camera, not to the viewing direction, so device
    +Z is NOT where the wearer is facing. Reads rgb_calib_<role>.json (written by
    scripts/fetch_comind_kinematic_subset.py from the head of online_calibration.jsonl);
    falls back to the measured nominal axis, which is near-identical across Aria units.
    """
    p = proc_dir / f"rgb_calib_{role}.json"
    if not p.exists():
        return NOMINAL_RGB_AXIS_DEVICE / np.linalg.norm(NOMINAL_RGB_AXIS_DEVICE)
    T = json.load(open(p))["T_Device_Camera"]
    w, (x, y, z) = T["UnitQuaternion"][0], T["UnitQuaternion"][1]
    axis = quat_wxyz_to_R(w, x, y, z)[:, 2]
    return axis / np.linalg.norm(axis)


def multislam_dir(rec_dir: Path, role: str) -> Path | None:
    ms = rec_dir / "multislam_output"
    cand = ms / f"{role}_trimmed" / "slam" / "closed_loop_trajectory.csv"
    if cand.exists():
        return cand
    mp = ms / "vrs_to_multi_slam.json"
    if mp.exists():
        for k, v in json.load(open(mp)).items():
            if k.endswith(f"{role}_trimmed.vrs"):
                cand = ms / str(v) / "slam" / "closed_loop_trajectory.csv"
                if cand.exists():
                    return cand
    return None


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """[N,4] Hamilton xyzw -> [N,3,3] (active rotation), same convention as duet.geometry.rotations."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((len(q), 3, 3), np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_poses(csv_path: Path, query_ts_ns: np.ndarray):
    """Nearest T_world_device per query frame: returns R [N,3,3], t [N,3], valid [N], graph_uid."""
    cols = ["graph_uid", "tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device",
            "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]
    df = pd.read_csv(csv_path, usecols=cols)
    uid = df["graph_uid"].mode().iloc[0]
    df = df[df["graph_uid"] == uid]
    ts = df["tracking_timestamp_us"].to_numpy(np.int64)
    order = np.argsort(ts, kind="stable"); df = df.iloc[order]; ts = ts[order]
    q = query_ts_ns // 1000
    idx = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
    lo = idx - 1
    pick = np.where(np.abs(ts[lo] - q) <= np.abs(ts[idx] - q), lo, idx)
    valid = np.abs(ts[pick] - q) <= POSE_MAX_GAP_US
    t = df[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy(np.float64)[pick]
    R = quat_xyzw_to_R(df[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy(np.float64)[pick])
    return R, t, valid, str(uid)


def _hand_points(feat: np.ndarray, role: str):
    """Per role: wrists [N,2,3] (L,R), valid [N,2], gaze point [N,3], gaze valid [N] in device frame."""
    s = person_slice(role)
    wr = np.stack([feat[:, s.start + h * HAND_DIM + 15: s.start + h * HAND_DIM + 18] for h in range(2)], axis=1).astype(np.float64)
    hv = np.stack([feat[:, s.start + h * HAND_DIM + 25] > 0 for h in range(2)], axis=1)
    g = feat[:, s.start + 2 * HAND_DIM: s.start + 2 * HAND_DIM + 3].astype(np.float64)
    gv = feat[:, s.start + 2 * HAND_DIM + 6] > 0
    return wr, hv, g, gv


def to_world(R, t, p):  # p [N,K,3] device -> world
    return np.einsum("nij,nkj->nki", R, p) + t[:, None, :]


def to_device(R, t, p_world):  # p_world [N,K,3] -> device of (R,t)
    return np.einsum("nji,nkj->nki", R, p_world - t[:, None, :])


def build_recording_world(raw_root: Path, proc_root: Path, recording_id: str, kin_features: np.ndarray) -> tuple[np.ndarray, dict]:
    rec = raw_root / "recordings" / recording_id
    n = len(kin_features)
    poses, uids = {}, {}
    for role in ROLES:
        csv = multislam_dir(rec, role)
        if csv is None:
            raise FileNotFoundError(f"no multi-SLAM trajectory for {role} in {recording_id}")
        anchor = parse_mp4_tail((proc_root / recording_id / f"mp4_tail_{role}.bin").read_bytes())
        ts = frame_device_times_ns(anchor)[:n]
        poses[role] = load_poses(csv, ts)
        uids[role] = poses[role][3]
    shared_graph = uids["leader"] == uids["helper"]
    RL, tL, vL, _ = poses["leader"]; RH, tH, vH, _ = poses["helper"]
    both = vL & vH & shared_graph
    wL, hvL, gL, gvL = _hand_points(kin_features, "leader"); wH, hvH, gH, gvH = _hand_points(kin_features, "helper")
    WL, WH = to_world(RL, tL, wL), to_world(RH, tH, wH)                      # [N,2,3]
    GL, GH = to_world(RL, tL, gL[:, None])[:, 0], to_world(RH, tH, gH[:, None])[:, 0]
    f = np.zeros((n, WORLD_DIM), np.float32)
    big = 3.0  # metres; used to mask invalid pairs out of min()
    d = np.linalg.norm(WL[:, :, None, :] - WH[:, None, :, :], axis=-1)     # [N,2,2]
    pv = hvL[:, :, None] & hvH[:, None, :] & both[:, None, None]
    d = np.where(pv, d, big)
    f[:, 0:4] = d.reshape(n, 4)
    dmin = d.reshape(n, 4).min(1); anyp = pv.reshape(n, 4).any(1)
    f[:, 4] = np.where(anyp, dmin, 0)
    dd = np.zeros(n); dd[1:] = (dmin[1:] - dmin[:-1]) * FPS
    f[:, 5] = np.where(anyp & np.r_[False, anyp[:-1]], np.clip(dd, -5, 5), 0)
    f[:, 6] = np.where(both, np.linalg.norm(tL - tH, axis=1), 0)
    f[:, 7:10] = np.where(both[:, None], to_device(RL, tL, tH[:, None])[:, 0], 0)
    f[:, 10:13] = np.where(both[:, None], to_device(RH, tH, tL[:, None])[:, 0], 0)
    hl = to_device(RL, tL, WH); hh = to_device(RH, tH, WL)                   # partner wrists in my frame
    f[:, 13:19] = np.where((both[:, None] & hvH).repeat(3, axis=1) if False else np.repeat(both[:, None] & hvH, 3, axis=1), hl.reshape(n, 6), 0)
    f[:, 19:25] = np.where(np.repeat(both[:, None] & hvL, 3, axis=1), hh.reshape(n, 6), 0)
    gwH = np.where((gvL & both)[:, None, None] & hvH[:, :, None], np.linalg.norm(GL[:, None] - WH, axis=-1)[..., None], big)[..., 0].min(1)
    gwL = np.where((gvH & both)[:, None, None] & hvL[:, :, None], np.linalg.norm(GH[:, None] - WL, axis=-1)[..., None], big)[..., 0].min(1)
    f[:, 25] = np.where(gvL & both & hvH.any(1), gwH, 0)
    f[:, 26] = np.where(gvH & both & hvL.any(1), gwL, 0)
    f[:, 27] = np.where(gvL & both, np.linalg.norm(GL - tH, axis=1), 0)
    f[:, 28] = np.where(gvH & both, np.linalg.norm(GH - tL, axis=1), 0)
    # Forward axis = the RGB camera's optical axis, NOT device +Z. Measured on recording
    # 43276420, camera-rgb sits 38.7 deg off device +Z (the glasses angle the camera down),
    # so using +Z mis-states where a wearer is facing by that much.
    axL = rgb_forward_axis(proc_root / recording_id, "leader")
    axH = rgb_forward_axis(proc_root / recording_id, "helper")
    fwdL, fwdH = RL @ axL, RH @ axH
    toH = tH - tL; toL = -toH
    nrm = np.linalg.norm(toH, axis=1) + 1e-6
    f[:, 29] = np.where(both, (fwdL * toH).sum(1) / nrm, 0)
    f[:, 30] = np.where(both, (fwdH * toL).sum(1) / nrm, 0)
    f[:, 31] = both; f[:, 32] = hvL.any(1); f[:, 33] = hvH.any(1)
    stats = {"shared_graph": bool(shared_graph), "graph_uids": uids, "pose_valid_leader": float(vL.mean()), "pose_valid_helper": float(vH.mean()),
             "both_valid": float(both.mean()), "median_head_dist_m": float(np.median(f[both, 6])) if both.any() else None,
             "median_min_wrist_dist_m": float(np.median(f[anyp, 4])) if anyp.any() else None}
    return f, stats
