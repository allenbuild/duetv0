"""World-frame stages: calibration, tags, head poses, 3D triangulation, cross-person features.

Rig config (episode dir / rig.json, optional; sensible defaults otherwise):
{
  "board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037},
  "tag_size_m": 0.08,
  "head_tags": {"leader": 1, "helper": 2},              # AprilTag id glued on each head strap
  "object_tags": {"bowl": 10, "knife": 11},
  "wall_tags": [20, 21],
  "T_tag_headcam": {"leader": [[...4x4...]]},           # once-measured offset tag -> head camera (identity if absent)
  "hfov_deg": {"gopro_front": 120}                       # nominal lens FOV for cameras that never saw the board
}

Stages
  calib     fixed cameras: intrinsics from board views (or nominal from FOV), T_world_cam from the first frames that show
            the board; world == board frame. Ego cameras: intrinsics from ZED export calibration, board, or nominal.
  tags      AprilTag detections in every fixed view per frame -> tag poses in world.
  headpose  per ego stream, T_world_cam per frame from (a) ZED export pose.csv registered to the board, else
            (b) head tag seen by a fixed camera, else none. Records the backend used.
  world3d   triangulate body keypoints and hand landmarks from the calibrated fixed cameras (+ ego camera when its pose is
            known); ZED depth at hand landmarks when present; object positions from tags; cross-person features.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import geometry as G
from .episode import Episode, Stream
from .perception import _frame_list

DEFAULT_RIG = {"board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037}, "tag_size_m": 0.08, "head_tags": {}, "object_tags": {}, "wall_tags": [],
               "T_tag_headcam": {}, "hfov_deg": {}, "intrinsics": {}}
HAND_LINKS_2D = 21


def rig(ep: Episode) -> dict:
    p = ep.dir / "rig.json"; r = dict(DEFAULT_RIG)
    if p.exists():
        r.update(json.load(open(p)))
    return r


def _spec(r) -> G.BoardSpec:
    return G.BoardSpec(**r["board"])


def _gray(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _zed_dir(ep: Episode, s: Stream) -> Path | None:
    d = ep.dir / "zed" / s.name
    # a ZED folder needs the factory calibration; pose.csv (SDK tracking) is optional (Mac UVC bundles have none)
    return d if (d / "calibration.json").exists() else None


# ----------------------------------------------------------------------------- calib

def calib(ep: Episode) -> None:
    ep.set_status("calib", "running"); r = rig(ep); spec = _spec(r)
    out = ep.derived / "calib"; out.mkdir(exist_ok=True); report = {"world": "board", "cameras": {}}
    for s in ep.streams:
        frames = _frame_list(ep, s); n = len(frames)
        if n == 0:
            continue
        sample = [_gray(f) for f in frames[: min(n, 20 * int(ep.proc_fps))][:: max(1, int(ep.proc_fps) // 2)]]  # first 20 s at ~2 Hz
        intr, source = None, None
        zd = _zed_dir(ep, s)
        if zd is not None:
            c = json.load(open(zd / "calibration.json")); sc = sample[0].shape[1] / c["width"]
            K = np.array(c["K"], float); K[:2] *= sc; intr = G.Intrinsics(K, np.array(c["dist"][:5], float), sample[0].shape[1], sample[0].shape[0]); source = "zed_factory"
        if intr is None and s.name in r.get("intrinsics", {}):  # calibrated once at install (moving board), stored in rig.json
            intr = G.Intrinsics.from_json(r["intrinsics"][s.name]); source = "rig_json"
        if intr is None:
            # board-based intrinsics need the board seen from different places; a fixed camera watching a static board is degenerate
            corners = [G.detect_board(g, spec)[0] for g in sample]; corners = [c.reshape(-1, 2) for c in corners if c is not None]
            spread = float(np.mean([np.std([c.mean(0) for c in corners], axis=0).sum()])) if len(corners) >= 6 else 0.0
            if spread > 30.0:
                intr = G.calibrate_intrinsics(sample, spec); source = "board" if intr is not None else None
        if intr is None:
            intr = G.Intrinsics.nominal(sample[0].shape[1], sample[0].shape[0], r["hfov_deg"].get(s.name, 100.0 if s.role == "exo" else 110.0)); source = "nominal_fov"
        T_world_cam, err, frame_used = None, None, None
        for k, g in enumerate(sample):  # first frame that shows the board fixes this camera in the world
            bp = G.board_pose(g, intr, spec)
            if bp is not None and bp[1] < 3.0:
                T_world_cam, err, frame_used = G.inv(bp[0]), bp[1], k; break
        report["cameras"][s.name] = {"role": s.role, "intrinsics": intr.to_json(), "intrinsics_source": source,
                                     "T_world_cam": None if T_world_cam is None else T_world_cam.tolist(), "board_reproj_px": err, "board_sample_frame": frame_used}
    json.dump(report, open(out / "calib.json", "w"), indent=1)
    reg = [k for k, v in report["cameras"].items() if v["T_world_cam"] is not None]
    ep.set_status("calib", "done" if reg else "skipped", f"registered to board: {reg}" if reg else "board not seen by any camera; no world frame")


def load_calib(ep: Episode) -> dict[str, tuple[G.Intrinsics, np.ndarray | None]]:
    p = ep.derived / "calib" / "calib.json"
    if not p.exists():
        return {}
    d = json.load(open(p))["cameras"]
    return {k: (G.Intrinsics.from_json(v["intrinsics"]), None if v["T_world_cam"] is None else np.array(v["T_world_cam"])) for k, v in d.items()}


# ----------------------------------------------------------------------------- tags

def tags(ep: Episode) -> None:
    ep.set_status("tags", "running"); r = rig(ep); cal = load_calib(ep)
    out = ep.derived / "tags"; out.mkdir(exist_ok=True); total = 0
    for s in ep.exos():
        if s.name not in cal:
            continue
        intr, T_wc = cal[s.name]; frames = _frame_list(ep, s); rows = []
        for k, f in enumerate(frames):
            for d in G.detect_tags(_gray(f), intr, r["tag_size_m"]):
                if d.T_cam_tag is None:
                    continue
                T_wt = (T_wc @ d.T_cam_tag) if T_wc is not None else None
                rows.append({"frame": k, "tag_id": d.tag_id, "err_px": d.err_px, "cx": float(d.corners_px[:, 0].mean()), "cy": float(d.corners_px[:, 1].mean()),
                             **{f"Tc{i}{j}": float(d.T_cam_tag[i, j]) for i in range(3) for j in range(4)},
                             **({f"Tw{i}{j}": float(T_wt[i, j]) for i in range(3) for j in range(4)} if T_wt is not None else {})})
        pd.DataFrame(rows).to_parquet(out / f"{s.name}.parquet", index=False); total += len(rows)
    ep.set_status("tags", "done" if total else "skipped", f"{total} tag detections" if total else "no tags seen (no fixed cameras registered, or no tags in view)")


def _T_from_row(row, prefix) -> np.ndarray:
    T = np.eye(4)
    for i in range(3):
        for j in range(4):
            T[i, j] = row[f"{prefix}{i}{j}"]
    return T


def tag_world_poses(ep: Episode, tag_id: int) -> dict[int, np.ndarray]:
    """frame -> T_world_tag, averaging (by taking the lowest-error) over fixed cameras that saw it."""
    out: dict[int, tuple[float, np.ndarray]] = {}
    for p in (ep.derived / "tags").glob("*.parquet"):
        df = pd.read_parquet(p)
        if df.empty or "Tw00" not in df:
            continue
        for _, row in df[df.tag_id == tag_id].iterrows():
            k = int(row.frame)
            if k not in out or row.err_px < out[k][0]:
                out[k] = (float(row.err_px), _T_from_row(row, "Tw"))
    return {k: v[1] for k, v in out.items()}


# ----------------------------------------------------------------------------- head pose

def _quat_xyzw_to_R(q):
    x, y, z, w = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def headpose(ep: Episode) -> None:
    ep.set_status("headpose", "running"); r = rig(ep); cal = load_calib(ep)
    out = ep.derived / "headpose"; out.mkdir(exist_ok=True); used = {}
    for s in ep.egos():
        frames = _frame_list(ep, s); n = len(frames); T = np.full((n, 4, 4), np.nan); backend = None
        zd = _zed_dir(ep, s)
        if zd is not None and not (zd / "pose.csv").exists():
            zd = None  # calibration/depth only (Mac recorder): no SDK tracking, fall through to the head tag
        if zd is not None:  # (a) ZED positional tracking, registered to the board via the frame where this camera saw the board
            pose = pd.read_csv(zd / "pose.csv"); t0 = pose.timestamp_ns.min()
            # map processed frames to ZED frames by time: processed frame k is at common_start + k/proc_fps in reference time
            ts_ref = ep.common_start_s + np.arange(n) / ep.proc_fps - s.offset_s  # seconds in this stream
            zed_t = (pose.timestamp_ns.to_numpy() - t0) / 1e9
            idx = np.clip(np.searchsorted(zed_t, ts_ref), 0, len(zed_t) - 1)
            Tz = np.array([np.block([[_quat_xyzw_to_R(row[["qx", "qy", "qz", "qw"]].to_numpy(float)), row[["tx", "ty", "tz"]].to_numpy(float)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for _, row in pose.iloc[idx].iterrows()])
            # registration: ZED world -> board world using the frame where the board was seen
            T_world_cam0 = cal.get(s.name, (None, None))[1]; T_reg = np.eye(4)
            if T_world_cam0 is not None:
                cj = json.load(open(ep.derived / "calib" / "calib.json"))["cameras"][s.name]; k0 = cj["board_sample_frame"] * max(1, int(ep.proc_fps) // 2)
                T_reg = T_world_cam0 @ G.inv(Tz[min(k0, n - 1)])
                backend = "zed_tracking+board"
            else:
                backend = "zed_tracking_unregistered"
            T = np.einsum("ij,njk->nik", T_reg, Tz)
        else:  # (b) head tag seen by a fixed camera
            tag_id = r["head_tags"].get(s.person or s.name) or r["head_tags"].get(s.name)
            if tag_id is not None and (ep.derived / "tags").exists():
                poses = tag_world_poses(ep, int(tag_id)); T_th = np.array(r["T_tag_headcam"].get(s.person or s.name, np.eye(4).tolist()))
                for k, T_wt in poses.items():
                    if k < n:
                        T[k] = T_wt @ T_th
                backend = "head_tag" if poses else None
        np.savez_compressed(out / f"{s.name}.npz", T_world_cam=T.astype(np.float32), backend=str(backend))
        used[s.name] = (backend, float(np.isfinite(T[:, 0, 0]).mean()) if n else 0.0)
    ep.set_status("headpose", "done" if any(b for b, _ in used.values()) else "skipped", "; ".join(f"{k}: {b} ({c:.0%} frames)" for k, (b, c) in used.items()))


# ----------------------------------------------------------------------------- world 3D

def _cams(ep: Episode, cal, k: int, headposes) -> list[G.Camera]:
    cams = []
    for s in ep.streams:
        if s.name not in cal:
            continue
        intr, T_wc = cal[s.name]
        if s.role == "ego" and s.name in headposes and np.isfinite(headposes[s.name][k, 0, 0]):
            T_wc = headposes[s.name][k]
        if T_wc is not None:
            cams.append(G.Camera(s.name, intr, T_wc))
    return cams


def world3d(ep: Episode) -> None:
    ep.set_status("world3d", "running"); r = rig(ep); cal = load_calib(ep)
    fixed = [s for s in ep.exos() if s.name in cal and cal[s.name][1] is not None]
    posed_egos = [s for s in ep.egos() if (ep.derived / "headpose" / f"{s.name}.npz").exists() and np.isfinite(np.load(ep.derived / "headpose" / f"{s.name}.npz")["T_world_cam"][:, 0, 0]).any()]
    if len(fixed) + len(posed_egos) < 2:
        ep.set_status("world3d", "skipped", f"need >= 2 cameras with world poses (have {len(fixed)} fixed, {len(posed_egos)} ego)"); return
    n = min(len(_frame_list(ep, s)) for s in ep.streams)
    headposes = {s.name: np.load(ep.derived / "headpose" / f"{s.name}.npz")["T_world_cam"] for s in ep.egos() if (ep.derived / "headpose" / f"{s.name}.npz").exists()}
    body2d = {s.name: np.load(ep.derived / "body2d" / f"{s.name}.npz") for s in ep.streams if (ep.derived / "body2d" / f"{s.name}.npz").exists()}
    hands2d = {s.name: np.load(ep.derived / "hands" / f"{s.name}.npz") for s in ep.egos() if (ep.derived / "hands" / f"{s.name}.npz").exists()}
    # scale 2D detections (made on proc-size frames) -> same frames used for calibration, so no rescale needed
    bodies = np.full((n, 2, 17, 3), np.nan, np.float32); body_err = np.full((n, 2, 17), np.nan, np.float32)
    hands3d = {s.name: np.full((n, 2, 21, 3), np.nan, np.float32) for s in ep.egos()}
    heads = {s.name: np.full((n, 3), np.nan, np.float32) for s in ep.egos()}
    objects = {name: np.full((n, 3), np.nan, np.float32) for name in r["object_tags"]}
    obj_poses = {name: tag_world_poses(ep, int(tid)) for name, tid in r["object_tags"].items()} if (ep.derived / "tags").exists() else {}
    for k in range(n):
        cams = _cams(ep, cal, k, headposes)
        fixed_cams = [c for c in cams if c.name in {s.name for s in fixed}]
        # bodies: person slot p in each fixed camera is matched greedily by left-right image order (2 people)
        if len(fixed_cams) >= 2 and all(c.name in body2d for c in fixed_cams):
            per_cam = []
            for c in fixed_cams:
                kp = body2d[c.name]["kpts"][k]; ok = np.isfinite(kp[:, 0, 0]); order = np.argsort(np.where(ok, kp[:, 11, 0], np.inf))[:2]  # sort by hip x
                per_cam.append([kp[i] if ok[i] else None for i in order] + [None] * (2 - len(order)))
            for p in range(2):
                views = [pc[p] for pc in per_cam]
                if sum(v is not None for v in views) >= 2:
                    pts = np.stack([v[:, :2] if v is not None else np.full((17, 2), np.nan) for v in views]); conf = np.stack([v[:, 2] if v is not None else np.zeros(17) for v in views])
                    X, E = G.triangulate_many(fixed_cams, pts, conf); bodies[k, p] = X; body_err[k, p] = E
        # hands: ego camera (needs its pose) + fixed cameras see the same wrists; without fixed-camera hand detectors we use
        #        ZED depth when available, else leave to bodies' wrist joints
        for s in ep.egos():
            if s.name in headposes and np.isfinite(headposes[s.name][k, 0, 0]):
                heads[s.name][k] = headposes[s.name][k][:3, 3]
            zd = _zed_dir(ep, s)
            if zd is not None and s.name in hands2d and s.name in cal:
                intr, _ = cal[s.name]; T_wc = headposes.get(s.name, np.full((n, 4, 4), np.nan))[k]
                if not np.isfinite(T_wc[0, 0]):
                    continue
                kz = int(round((ep.common_start_s + k / ep.proc_fps - s.offset_s) * json.load(open(zd / "export.json"))["fps"])) if k == 0 or True else 0
                dp = zd / "depth" / f"{kz:06d}.png"
                if not dp.exists():
                    continue
                depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                sc_x = depth.shape[1] / intr.width; sc_y = depth.shape[0] / intr.height
                for h in range(2):
                    lm = hands2d[s.name]["lm2d"][k, h]
                    if not np.isfinite(lm[0, 0]):
                        continue
                    u = np.clip((lm[:, 0] * sc_x).astype(int), 0, depth.shape[1] - 1); v = np.clip((lm[:, 1] * sc_y).astype(int), 0, depth.shape[0] - 1)
                    z = depth[v, u]; good = z > 0.05
                    Kinv = np.linalg.inv(intr.K); pix = np.stack([lm[:, 0], lm[:, 1], np.ones(21)], 1)
                    rays = pix @ Kinv.T; P_cam = rays * z[:, None]; P_w = G.apply(T_wc, P_cam); P_w[~good] = np.nan
                    hands3d[s.name][k, h] = P_w
        for name, poses in obj_poses.items():
            if k in poses:
                objects[name][k] = poses[k][:3, 3]
    # cross-person features (metres): head distance, wrist-wrist min distance, facing, each head's forward vs the other
    egos = [s.name for s in ep.egos()]; feats = {}
    if len(egos) >= 2:
        a, b = egos[0], egos[1]; feats["head_dist_m"] = np.linalg.norm(heads[a] - heads[b], axis=1)
        wa = bodies[:, 0, [9, 10], :]; wb = bodies[:, 1, [9, 10], :]
        d = np.linalg.norm(wa[:, :, None, :] - wb[:, None, :, :], axis=-1).reshape(n, -1); feats["min_wrist_dist_m"] = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1); feats["min_wrist_dist_m"][~np.isfinite(feats["min_wrist_dist_m"])] = np.nan
        for name in (a, b):
            other = b if name == a else a
            if name in headposes:
                fwd = headposes[name][:n, :3, 2]; to = heads[other] - heads[name]; nrm = np.linalg.norm(to, axis=1) + 1e-9
                feats[f"{name}_facing_partner_cos"] = (fwd * to).sum(1) / nrm
    out = ep.derived / "world3d"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "world3d.npz", bodies=bodies, body_err=body_err, **{f"hands3d_{k}": v for k, v in hands3d.items()}, **{f"head_{k}": v for k, v in heads.items()},
                        **{f"object_{k}": v for k, v in objects.items()}, **{f"feat_{k}": v.astype(np.float32) for k, v in feats.items()})
    cov = float(np.isfinite(bodies[:, :, 9, 0]).any(1).mean()); ep.set_status("world3d", "done", f"triangulated body coverage {cov:.0%}; features {list(feats)}")
