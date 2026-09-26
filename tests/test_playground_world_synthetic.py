"""End-to-end check of the world stages on a synthetic 3-camera episode: two fixed cameras and one 'head' camera
see a ChArUco board, a head tag rigidly attached to the head camera, and an object tag. Frames are written straight
into derived/frames (no video), then calib -> tags -> headpose -> world3d run through the real stage code."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from duet.playground import geometry as G
from duet.playground.episode import Episode, Stream
from duet.playground import world as W
from test_playground_geometry import W as IMW, H as IMH, render_plane, look_at

# wide-angle fixed cameras (f=600 px at 1280 wide ~ 94 deg) mounted high, and a bigger board, as a real station would have
K_TRUE = np.array([[600.0, 0, 640], [0, 600.0, 360], [0, 0, 1]])
SPEC = G.BoardSpec(7, 5, 0.08, 0.06); TAG_M = 0.08
BC = (SPEC.squares_x * SPEC.square_m / 2, SPEC.squares_y * SPEC.square_m / 2)  # board centre


def _tag_canvas(tag_id):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11); tag = cv2.aruco.generateImageMarker(d, tag_id, 400); border = 60
    c = np.full((400 + 2 * border, 400 + 2 * border), 255, np.uint8); c[border:-border, border:-border] = tag
    return c, 400 / TAG_M, border


def _compose(layers):
    out = np.full((IMH, IMW), 255, np.uint8)
    for img in layers:
        out = np.minimum(out, img)
    return out


def test_world_stages_on_synthetic_episode(tmp_path):
    board_img = SPEC.board().generateImage((int(SPEC.squares_x * SPEC.square_m * 2500), int(SPEC.squares_y * SPEC.square_m * 2500))); ppm_b = 2500
    n = 12; fps = 10.0
    T_wc = {"exoA": look_at([BC[0] - 0.5, BC[1] - 0.3, -1.1], [*BC, 0]), "exoB": look_at([BC[0] + 0.6, BC[1] - 0.25, -1.0], [*BC, 0])}
    # head camera moves along a small arc above the board; head tag is 5 cm above the camera, facing the fixed cameras (tag z toward -Z world)
    T_tag_headcam = np.eye(4); T_tag_headcam[:3, 3] = [0, 0.05, 0]
    head_poses = [look_at([BC[0] + 0.12 * np.cos(2 * np.pi * k / n), BC[1] + 0.08 * np.sin(2 * np.pi * k / n), -0.5], [*BC, 0]) for k in range(n)]
    # object tag lying flat on the table (in the board plane, offset), so its centre is at (0.45, 0.10, 0)
    obj_canvas, ppm_t, border = _tag_canvas(10)
    ep_dir = tmp_path / "synthetic"; (ep_dir / "streams").mkdir(parents=True); (ep_dir / "imu").mkdir()
    ep = Episode(name="synthetic", root=str(ep_dir), streams=[Stream("head", "ego", "streams/head.mp4", person="leader"), Stream("exoA", "exo", "streams/exoA.mp4"), Stream("exoB", "exo", "streams/exoB.mp4")],
                 reference="head", proc_fps=fps, common_start_s=0.0, common_end_s=n / fps)
    for s in ep.streams:
        (ep.derived / "frames" / s.name).mkdir(parents=True)
    head_tag_canvas, _, _ = _tag_canvas(1)
    intr = G.Intrinsics(K_TRUE, np.zeros(5), IMW, IMH)
    for k in range(n):
        T_head = head_poses[k]
        # head tag plane: a small plane attached to the head camera. Plane frame = tag frame with origin at tag centre; render via T_cam_plane.
        T_world_tag = T_head @ G.inv(T_tag_headcam)
        for name in ("exoA", "exoB"):
            T_cam_world = G.inv(T_wc[name])
            layers = [render_plane(board_img, ppm_b, T_cam_world, K_TRUE),
                      render_plane(obj_canvas, ppm_t, T_cam_world, K_TRUE, plane_origin=(0.65 - TAG_M / 2 - border / ppm_t, 0.15 - TAG_M / 2 - border / ppm_t)),
                      render_plane(head_tag_canvas, ppm_t, T_cam_world @ T_world_tag, K_TRUE, plane_origin=(-(TAG_M / 2 + border / ppm_t), -(TAG_M / 2 + border / ppm_t)))]
            cv2.imwrite(str(ep.derived / "frames" / name / f"{k + 1:06d}.jpg"), _compose(layers))
        cv2.imwrite(str(ep.derived / "frames" / "head" / f"{k + 1:06d}.jpg"), render_plane(board_img, ppm_b, G.inv(T_head), K_TRUE))
    # fixed cameras: intrinsics calibrated once at install (a static camera watching a static board cannot self-calibrate)
    json.dump({"board": {"squares_x": SPEC.squares_x, "squares_y": SPEC.squares_y, "square_m": SPEC.square_m, "marker_m": SPEC.marker_m}, "tag_size_m": TAG_M, "head_tags": {"leader": 1}, "object_tags": {"bowl": 10}, "T_tag_headcam": {"leader": T_tag_headcam.tolist()}, "hfov_deg": {},
               "intrinsics": {name: intr.to_json() for name in ("exoA", "exoB", "head")}}, open(ep_dir / "rig.json", "w"))
    for s in ep.streams:
        s.width, s.height, s.fps, s.duration_s = IMW, IMH, fps, n / fps
    ep.save()
    W.calib(ep); assert ep.status["calib"]["state"] == "done", ep.status["calib"]
    cal = W.load_calib(ep)
    for name in ("exoA", "exoB"):
        assert cal[name][1] is not None and np.linalg.norm(cal[name][1][:3, 3] - T_wc[name][:3, 3]) < 0.03
    W.tags(ep); assert ep.status["tags"]["state"] == "done"
    obj = W.tag_world_poses(ep, 10); assert len(obj) >= n - 2
    assert np.linalg.norm(np.mean([p[:3, 3] for p in obj.values()], axis=0) - [0.65, 0.15, 0]) < 0.03
    W.headpose(ep); assert "head_tag" in ep.status["headpose"]["detail"]
    hp = np.load(ep.derived / "headpose" / "head.npz")["T_world_cam"]
    got = [k for k in range(n) if np.isfinite(hp[k, 0, 0])]; assert len(got) >= n - 3
    errs = [np.linalg.norm(hp[k][:3, 3] - head_poses[k][:3, 3]) for k in got]
    assert np.median(errs) < 0.04, errs
    W.world3d(ep); assert ep.status["world3d"]["state"] == "done"
    w = np.load(ep.derived / "world3d" / "world3d.npz"); assert np.isfinite(w["object_bowl"][:, 0]).sum() >= n - 2
