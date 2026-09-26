"""Synthetic checks of the playground's metric geometry: ChArUco boards and OFFICIAL AprilTag bitmaps are rendered into
virtual cameras (pinhole, rational, fisheye) with known poses, then intrinsics, board/tag poses (full rotation and
translation), head poses and triangulated points are recovered and compared with the truth, including adversarial
inputs (noise, distortion, oblique views, degenerate sets, points behind cameras, missing and outlier views).
Tolerances are a small multiple of the accuracy measured when these tests were written.

Helpers reused by tests/test_playground_world_synthetic.py: W, H, look_at, render_plane, official_tag_bitmap,
official_tag_canvas."""
import json
import threading
import warnings
from functools import lru_cache

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from duet.playground import geometry as G

W, H = 1280, 720
K_TRUE = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
SPEC = G.BoardSpec(7, 5, 0.05, 0.037)
BOARD_CENTRE = (0.175, 0.125, 0.0)

# Official AprilTag 36h11 ids 0 and 7 (AprilRobotics/apriltag-imgs tag36_11_0000{0,7}.png, 1 = white), transcribed from the
# PNGs and checked against libapriltag's own code table. Independent ground truth for official_tag_bitmap.
OFFICIAL_TAG36H11 = {
    0: ["1111111111", "1000000001", "1011010101", "1001110101", "1001100001", "1010100001", "1001011001", "1000010001", "1000000001", "1111111111"],
    7: ["1111111111", "1000000001", "1000010001", "1000011001", "1001010001", "1010111001", "1000011101", "1001010001", "1000000001", "1111111111"]}

# A GoPro-like lens fitted with the 5-coefficient pinhole model (review experiment E12); folds near the image corners.
K_GOPRO5 = np.array([[610.5562951944, 0.0, 638.3785781942], [0.0, 610.7287436065, 361.1111043301], [0.0, 0.0, 1.0]])
D_GOPRO5 = np.array([-2.9762507870e-01, 1.0309725169e-01, 1.1196336232e-05, 2.8145164183e-04, -1.7440619349e-02])
F_EQ = 640 / np.radians(60)  # equidistant fisheye whose 120 deg HFOV spans 1280 px


# ----------------------------------------------------------------------------- rendering helpers

def look_at(cam_pos, target, up=(0, 1, 0)):
    """T_world_cam for a camera at cam_pos looking at target (OpenCV: +Z forward, +Y down)."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float); z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float)); x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam_pos; return T


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))[0]


def T_from(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2, -1, 1))))


def _shrink(img_plane, px_per_m, texels_needed):
    """Pre-filter the texture (INTER_AREA) to ~texels_needed across its width: warp/remap sample without an area filter."""
    h, w = img_plane.shape[:2]; f = min(1.0, texels_needed / w)
    if f >= 1.0:
        return img_plane, px_per_m, px_per_m
    nw, nh = max(8, round(w * f)), max(8, round(h * f))
    return cv2.resize(img_plane, (nw, nh), interpolation=cv2.INTER_AREA), px_per_m * nw / w, px_per_m * nh / h


@lru_cache(maxsize=8)
def _ray_map(K_bytes, dist_bytes, model, width, height, ss):
    intr = G.Intrinsics(np.frombuffer(K_bytes).reshape(3, 3), np.frombuffer(dist_bytes), width, height, model)
    u, v = np.meshgrid((np.arange(width * ss) + 0.5) / ss - 0.5, (np.arange(height * ss) + 0.5) / ss - 0.5)
    return G.undistort_points(intr, np.stack([u, v], -1))  # [h*ss, w*ss, 2], NaN outside the model's range


def render_plane(img_plane, px_per_m, T_cam_plane, K, size=(W, H), plane_origin=(0.0, 0.0), intr=None, ss=1):
    """Render a planar texture into a camera. Plane frame: texture columns = +X, rows = +Y, Z = X x Y (into the printed
    face), metres; the texture's top-left CORNER sits at plane_origin. T_cam_plane: plane -> camera. Pinhole K by default;
    pass intr (any lens model; its K and size are used) for a RAW distorted image. OpenCV pixel-centre conventions,
    white background, uint8; ss = supersampling factor."""
    h, w = img_plane.shape[:2]
    corners = np.array([[0, 0, 0], [w / px_per_m, 0, 0], [w / px_per_m, h / px_per_m, 0], [0, h / px_per_m, 0]], float) + [*plane_origin, 0.0]
    Kp = K if intr is None else intr.K; size = size if intr is None else (intr.width, intr.height)
    Kss = np.asarray(Kp, float).copy(); Kss[:2] *= ss; Kss[0, 2] = (Kp[0, 2] + 0.5) * ss - 0.5; Kss[1, 2] = (Kp[1, 2] + 0.5) * ss - 0.5
    Xc = corners @ T_cam_plane[:3, :3].T + T_cam_plane[:3, 3]
    dst = (Xc[:, :2] / Xc[:, 2:3]) @ Kss[:2, :2].T + Kss[:2, 2]
    tex, ppx, ppy = _shrink(img_plane, px_per_m, 1.5 * max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3])))
    th, tw = tex.shape[:2]; out_size = (size[0] * ss, size[1] * ss)
    if intr is None:  # exact for a pinhole: plane -> image is a homography
        src = np.array([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]], np.float32)
        out = cv2.warpPerspective(tex, cv2.getPerspectiveTransform(src, dst.astype(np.float32)), out_size, flags=cv2.INTER_LINEAR, borderValue=255)
    else:  # any lens: cast each output pixel's ray onto the plane
        n = _ray_map(intr.K.tobytes(), intr.dist.tobytes(), intr.model, intr.width, intr.height, ss)
        d = np.concatenate([n, np.ones(n.shape[:2] + (1,))], -1)
        R, t = T_cam_plane[:3, :3], T_cam_plane[:3, 3]
        with np.errstate(all="ignore"):  # rays outside the lens model are NaN
            s = (R[:, 2] @ t) / (d @ R[:, 2]); q = (d * s[..., None] - t) @ R
        bad = ~(np.isfinite(s) & (s > 0))
        mx = ((q[..., 0] - plane_origin[0]) * ppx - 0.5).astype(np.float32); my = ((q[..., 1] - plane_origin[1]) * ppy - 0.5).astype(np.float32)
        mx[bad] = -1e6; my[bad] = -1e6
        out = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return cv2.resize(out, size, interpolation=cv2.INTER_AREA) if ss > 1 else out


def _board_image(spec=SPEC, px_per_m=4000):
    return spec.board().generateImage((round(spec.squares_x * spec.square_m * px_per_m), round(spec.squares_y * spec.square_m * px_per_m))), px_per_m


def official_tag_bitmap(tag_id: int) -> np.ndarray:
    """10x10 official AprilTag 36h11 bitmap incl. the white quiet-zone ring: cv2.aruco's DICT_APRILTAG_36h11 marker is the
    official bitmap ROTATED 180 deg (test_official_tag_bitmaps pins this), so rotate it back."""
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), int(tag_id), 8)
    return np.pad(np.rot90(m, 2), 1, constant_values=255)


def official_tag_canvas(tag_id: int, tag_m: float, cell_px: int = 50, border_cells: int = 1):
    """(canvas, px_per_m, plane_origin) for render_plane such that the plane frame IS the tag frame (origin at the tag
    centre, X right, Y down, Z into the face). tag_m = black-border edge = 8 cells; border_cells of white around it."""
    c = cv2.resize(official_tag_bitmap(tag_id), (10 * cell_px, 10 * cell_px), interpolation=cv2.INTER_NEAREST)
    if border_cells > 1:
        c = np.pad(c, (border_cells - 1) * cell_px, constant_values=255)
    ppm = 8 * cell_px / tag_m; half = (4 + border_cells) * cell_px / ppm
    return c, ppm, (-half, -half)


def _render_tag(tag_id, tag_m, T_cam_tag, K=K_TRUE, intr=None, ss=2):
    canvas, ppm, origin = official_tag_canvas(tag_id, tag_m)
    return render_plane(canvas, ppm, T_cam_tag, K, plane_origin=origin, intr=intr, ss=ss)


def _handheld_views(n, rng, K=K_TRUE, noise=0.0):
    img, ppm = _board_image(); frames, poses = [], []
    for _ in range(n):
        pos = np.array(BOARD_CENTRE) + [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -rng.uniform(0.45, 0.7)]
        T_wc = look_at(pos, [BOARD_CENTRE[0] + rng.uniform(-0.05, 0.05), BOARD_CENTRE[1] + rng.uniform(-0.05, 0.05), 0])
        f = render_plane(img, ppm, G.inv(T_wc), K, ss=2)
        if noise:
            f = np.clip(f + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        frames.append(f); poses.append(T_wc)
    return frames, poses


def _pinhole(K=K_TRUE, size=(W, H)):
    return G.Intrinsics(np.asarray(K, float), np.zeros(5), *size)


# ----------------------------------------------------------------------------- conventions, intrinsics, validation

def test_board_frame_convention():
    """Origin at the top-left outer corner, X along squares_x, Y down the print, Z into the board: a camera in front of
    the board sits at Z < 0 and a camera behind it (mirrored view) does not detect it."""
    cc = SPEC.corners_m()
    assert np.allclose(cc[0], [0.05, 0.05, 0]) and np.allclose(cc[1], [0.10, 0.05, 0]) and np.allclose(cc[-1], [0.30, 0.20, 0])
    img, ppm = _board_image(); intr = _pinhole()
    # fronto-parallel views couple tilt with lateral shift (achieved: 1.5 mm / 0.13 deg on OpenCV 4.8, 0.2 mm on 5.0)
    for pos, tol_m, tol_deg in (([0.175, 0.125, -0.6], 0.002, 0.2), ([0.0, -0.2, -0.55], 5e-4, 0.1), ([0.45, 0.35, -0.5], 5e-4, 0.1)):
        T_wc = look_at(pos, BOARD_CENTRE); r = G.board_pose(render_plane(img, ppm, G.inv(T_wc), K_TRUE, ss=2), intr, SPEC)
        assert r is not None and r[1] < 0.2
        T_est = G.inv(r[0])
        assert np.linalg.norm(T_est[:3, 3] - T_wc[:3, 3]) < tol_m and rot_err_deg(T_est[:3, :3], T_wc[:3, :3]) < tol_deg
    behind = look_at([0.175, 0.125, 0.6], BOARD_CENTRE)
    assert G.board_pose(render_plane(img, ppm, G.inv(behind), K_TRUE, ss=2), intr, SPEC) is None


def test_board_calibration_and_pose_recovery():
    rng = np.random.default_rng(0); frames, poses = _handheld_views(10, rng, noise=1.0)
    intr, rep = G.calibrate_intrinsics_report(frames, SPEC)
    assert intr is not None, rep["reason"]
    assert rep["ok"] and intr.source == "board" and intr.model == "pinhole" and intr.rms_px < 0.3
    assert abs(intr.K[0, 0] / 900 - 1) < 0.005 and abs(intr.K[1, 1] / 900 - 1) < 0.005 and abs(intr.K[0, 2] - 640) < 4 and abs(intr.K[1, 2] - 360) < 4
    assert rep["tilt_ok"] and rep["std_px"]["fx"] < 3 and rep["coverage"] > 0.3
    json.dumps(rep, allow_nan=False)  # the report is written into calib.json as is
    r = G.board_pose(frames[0], intr, SPEC); assert r is not None and r[1] < 0.5
    T_est = G.inv(r[0])
    assert np.linalg.norm(T_est[:3, 3] - poses[0][:3, 3]) < 0.003 and rot_err_deg(T_est[:3, :3], poses[0][:3, :3]) < 0.2


def test_intrinsics_validation_and_json():
    i = G.Intrinsics(K_TRUE, np.zeros(5), W, H, "pinhole", "rig_json", 0.3)
    j = G.Intrinsics.from_json(i.to_json())
    assert np.allclose(j.K, i.K) and (j.model, j.source, j.rms_px, j.width, j.height) == ("pinhole", "rig_json", 0.3, W, H)
    old = {"K": K_TRUE.tolist(), "dist": [0.1, -0.02, 0, 0, 0.001], "width": W, "height": H, "rms_px": None}  # written before model/source
    assert (G.Intrinsics.from_json(old).model, G.Intrinsics.from_json(old).source) == ("pinhole", "")
    assert G.Intrinsics.from_json({**old, "dist": [0.1] * 8}).model == "rational"
    f = G.Intrinsics.from_json({**old, "dist": [0.01, 0, 0, 0], "model": "fisheye"}); assert f.model == "fisheye" and f.dist.shape == (4,)
    with pytest.raises(G.GeometryError):
        G.Intrinsics.from_json({"K": K_TRUE.tolist(), "width": W, "height": H})  # missing dist is not "no distortion"
    for bad in ({"model": "gopro"}, {"model": 0.3}, {"dist": np.zeros(6)}, {"K": np.eye(2)}, {"K": np.diag([-1.0, 1, 1])}, {"width": 0}, {"height": 720.5}):
        kw = {"K": K_TRUE, "dist": np.zeros(5), "width": W, "height": H, **bad}
        with pytest.raises(G.GeometryError):
            G.Intrinsics(**kw)
    with pytest.raises(G.GeometryError):
        G.Intrinsics(K_TRUE, np.zeros(5), W, H, "fisheye")  # fisheye takes exactly 4 coefficients
    assert issubclass(G.GeometryError, ValueError) and issubclass(G.GeometryError, AssertionError)


def test_scaled_to_matches_resized_frames():
    native = G.Intrinsics(np.array([[1350.0, 0, 959.5], [0, 1350.0, 539.5], [0, 0, 1]]), np.array([0.05, -0.1, 0, 0, 0.02]), 1920, 1080, source="rig_json")
    s = native.scaled_to(640, 360)
    assert np.allclose(s.K, [[450, 0, 319.5], [0, 450, 179.5], [0, 0, 1]]) and np.allclose(s.dist, native.dist) and (s.width, s.height, s.source) == (640, 360, "rig_json")
    X = np.array([[0.1, -0.05, 1.0], [-0.3, 0.2, 2.0]])  # pixel-centre convention: u' = (u + 0.5) / 3 - 0.5
    assert np.allclose(G.project_cam(s, X), (G.project_cam(native, X) + 0.5) / 3 - 0.5, atol=1e-9)
    for bad in ((640, 480), (1280, 1080)):
        with pytest.raises(G.GeometryError):
            native.scaled_to(*bad)
    assert np.allclose(native.scaled_to(640, 361).K[1, 1], 1350 * 361 / 1080)  # even-rounded heights (< 1 % aspect change) are fine
    # the frame-size contract: a 1280x720 calibration on a 640x360 extracted frame must be rescaled, never used as is
    img, ppm = _board_image(); T_wc = look_at([0.0, -0.1, -0.5], BOARD_CENTRE)
    small = cv2.resize(render_plane(img, ppm, G.inv(T_wc), K_TRUE, ss=2), (640, 360), interpolation=cv2.INTER_AREA)
    with pytest.raises(G.GeometryError):
        G.board_pose(small, _pinhole(), SPEC)
    r = G.board_pose(small, _pinhole().scaled_to(640, 360), SPEC); assert r is not None
    assert np.linalg.norm(G.inv(r[0])[:3, 3] - T_wc[:3, 3]) < 0.004 and rot_err_deg(G.inv(r[0])[:3, :3], T_wc[:3, :3]) < 0.3


def test_nominal_intrinsics():
    p = G.Intrinsics.nominal(1280, 720, 120.0)
    assert p.source == "nominal_fov" and p.model == "pinhole" and abs(p.K[0, 0] - 369.5) < 0.1 and np.allclose(p.K[:2, 2], [639.5, 359.5])
    f = G.Intrinsics.nominal(1280, 720, 120.0, model="fisheye")
    assert f.model == "fisheye" and abs(f.K[0, 0] - F_EQ) < 1e-6 and f.dist.shape == (4,)
    assert np.allclose(G.project_cam(f, [[np.tan(np.radians(60)), 0, 1]]), [[1279.5, 359.5]])  # the quoted HFOV spans the width
    d = G.Intrinsics.nominal(1280, 720, 100.0, axis="diagonal"); assert abs(d.K[0, 0] - np.hypot(1280, 720) / 2 / np.tan(np.radians(50))) < 1e-6
    for kw in ({"fov_deg": 180.0}, {"fov_deg": 0.0}, {"fov_deg": 90.0, "axis": "width"}, {"fov_deg": 90.0, "model": "rational"}):
        with pytest.raises(G.GeometryError):
            G.Intrinsics.nominal(1280, 720, **kw)


def test_board_spec_units_and_legacy_pattern():
    for kw in ({"square_m": 50.0, "marker_m": 37.0}, {"marker_m": 0.06}, {"squares_x": 2}, {"dictionary": "DICT_NOPE"}, {"square_m": float("nan")}):
        with pytest.raises(G.GeometryError):
            G.BoardSpec(**kw)
    # an 8x6 board printed with the pre-4.6 ("legacy") layout is only found with legacy=True
    spec_new, spec_legacy = G.BoardSpec(8, 6, 0.04, 0.03), G.BoardSpec(8, 6, 0.04, 0.03, legacy=True)
    img, ppm = _board_image(spec_legacy); ctr = [0.16, 0.12, 0]; T_wc = look_at([0.26, -0.08, -0.6], ctr)
    frame = render_plane(img, ppm, G.inv(T_wc), K_TRUE, ss=2)
    assert G.board_pose(frame, _pinhole(), spec_new) is None
    r = G.board_pose(frame, _pinhole(), spec_legacy); assert r is not None
    assert np.linalg.norm(G.inv(r[0])[:3, 3] - T_wc[:3, 3]) < 0.002


def test_opencv_version_gate(monkeypatch):
    monkeypatch.setattr(cv2, "__version__", "4.7.0")
    with pytest.raises(RuntimeError, match="4.8"):
        G.BoardSpec(6, 4, 0.03, 0.02).board()
    with pytest.raises(RuntimeError, match="4.8"):
        G.detect_board(np.full((100, 100), 255, np.uint8), G.BoardSpec(6, 4, 0.03, 0.02))


def test_validate_rigid_and_inv():
    T = T_from(rot([1, 2, 3], 40), [0.1, -2.0, 3.0])
    assert np.allclose(G.inv(T) @ T, np.eye(4)) and np.allclose(G.validate_rigid(T.astype(np.float32)), T, atol=1e-6)  # float32-stored poses pass
    bad = {"scaled": np.diag([1.01, 1, 1, 1]), "reflection": np.diag([1.0, 1, -1, 1]), "nan": np.full((4, 4), np.nan), "row": T_from(np.eye(3), [0, 0, 0]) + np.eye(4)[[3]] * 0.1,
           "shape": np.eye(3), "shear": T_from(np.array([[1, 0.01, 0], [0, 1, 0], [0, 0, 1]]), [0, 0, 0])}
    for M in bad.values():
        with pytest.raises(G.GeometryError, match="T_test"):
            G.validate_rigid(M, "T_test")
        with pytest.raises(G.GeometryError):
            G.inv(M)
    with pytest.raises(G.GeometryError):
        G.Camera("c", _pinhole(), np.diag([1.0, 1, -1, 1]))
    with pytest.raises(G.GeometryError):
        G.head_pose_from_tag(np.eye(4), np.eye(4), np.diag([2.0, 2, 2, 1]))


# ----------------------------------------------------------------------------- AprilTags

def test_official_tag_bitmaps():
    for tag_id, rows in OFFICIAL_TAG36H11.items():
        truth = np.array([[255 if ch == "1" else 0 for ch in r] for r in rows], np.uint8)
        assert np.array_equal(official_tag_bitmap(tag_id), truth)
        raw_cv2 = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, 8)
        assert not np.array_equal(raw_cv2, truth[1:-1, 1:-1])  # cv2.aruco's image is NOT the printable official tag


TAG_M = 0.08
TAG_POSES = {"tilted_x30": T_from(rot([1, 0, 0], 30), [0.05, -0.02, 0.7]), "rot_z90_tilt": T_from(rot([0, 0, 1], 90) @ rot([0, 1, 0], 25), [-0.1, 0.05, 0.8]),
             "upside_down": T_from(rot([0, 0, 1], 180) @ rot([1, 0, 0], -20), [0.0, 0.1, 0.65]), "oblique60": T_from(rot([0, 1, 0], 60) @ rot([0, 0, 1], 15), [0.1, 0.0, 0.7]),
             "far_1.6m": T_from(rot([1, 1, 0], 35), [0.3, -0.2, 1.6])}


@pytest.mark.parametrize("name", list(TAG_POSES))
def test_tag_pose_full_rotation(name):
    """Tag frame = board convention (X right, Y down, Z into the face): the FULL pose matches, corners are raw pixels."""
    pytest.importorskip("pupil_apriltags")
    T_true = TAG_POSES[name]; intr = _pinhole()
    dets = G.detect_tags(_render_tag(7, TAG_M, T_true), intr, TAG_M)
    assert [d.tag_id for d in dets] == [7]
    d = dets[0]
    assert rot_err_deg(d.T_cam_tag[:3, :3], T_true[:3, :3]) < 1.0 and np.linalg.norm(d.T_cam_tag[:3, 3] - T_true[:3, 3]) < 0.003
    true_corners = G.project(intr, T_true, G.tag_object_points(TAG_M))
    assert np.abs(d.corners_px - true_corners).max() < 0.35
    # err_px is a real reprojection error in raw pixels (pupil's pose_err was an object-space residual ~1e-9)
    assert np.isclose(d.err_px, np.linalg.norm(G.project(intr, d.T_cam_tag, G.tag_object_points(TAG_M)) - d.corners_px, axis=1).mean(), rtol=1e-6)
    assert d.err_px < 0.3 and d.T_cam_tag_alt is not None and d.alt_err_px >= d.err_px and d.hamming == 0 and d.family == "tag36h11"
    G.validate_rigid(d.T_cam_tag)


def test_cv2_aruco_tag_bitmap_reads_rotated_180():
    """Why the old tests could not see the diag(1,-1,-1) bug: cv2.aruco's tag image decodes as the official tag turned
    180 deg about its normal, so translation-only asserts passed while the rotation was 180 deg off."""
    pytest.importorskip("pupil_apriltags")
    T_true = TAG_POSES["tilted_x30"]
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 7, 400)
    canvas = np.pad(m, 50, constant_values=255); ppm = 400 / TAG_M; half = 250 / ppm
    d = G.detect_tags(render_plane(canvas, ppm, T_true, K_TRUE, plane_origin=(-half, -half), ss=2), _pinhole(), TAG_M)[0]
    assert rot_err_deg(d.T_cam_tag[:3, :3], T_true[:3, :3] @ rot([0, 0, 1], 180)) < 1.0 and np.linalg.norm(d.T_cam_tag[:3, 3] - T_true[:3, 3]) < 0.003


def test_head_pose_from_tag_full_pose_with_rotated_lever_arm():
    """Head camera pose from a head tag seen by a fixed exo camera; the tag mount is rotated (forehead tag facing
    forward, pitched 10 deg) with a lever arm on all three axes."""
    pytest.importorskip("pupil_apriltags")
    T_tag_headcam = T_from(np.diag([-1.0, 1, -1]) @ rot([1, 0, 0], 10), [0.03, 0.06, -0.02])
    T_world_fixed = look_at([0.2, -0.2, -1.1], [0.2, 0.9, -0.3])  # exo camera facing the wearer across the table
    intr = _pinhole(); pos_err, rot_errs = [], []
    for k in range(6):
        head = look_at([0.1 + 0.05 * k, 0.9, -0.45 - 0.02 * k], [0.15 - 0.03 * k, -0.1, 0.0])  # wearer looks back toward the table
        T_world_tag = head @ G.inv(T_tag_headcam)
        T_fixed_tag_true = G.inv(T_world_fixed) @ T_world_tag
        dets = G.detect_tags(_render_tag(1, 0.06, T_fixed_tag_true), intr, 0.06)
        assert [d.tag_id for d in dets] == [1]
        est = G.head_pose_from_tag(T_world_fixed, dets[0].T_cam_tag, T_tag_headcam)
        pos_err.append(np.linalg.norm(est[:3, 3] - head[:3, 3])); rot_errs.append(rot_err_deg(est[:3, :3], head[:3, :3]))
        old_flip = T_world_fixed @ dets[0].T_cam_tag @ np.diag([1.0, -1, -1, 1]) @ T_tag_headcam  # the removed convention
        assert rot_err_deg(old_flip[:3, :3], head[:3, :3]) > 170
    assert max(pos_err) < 0.006 and max(rot_errs) < 1.0, (pos_err, rot_errs)


def test_tags_on_raw_wide_lens_image_near_the_edge():
    """Tags are detected on the RAW image (full field), corners come back in raw pixels, poses use the lens model."""
    pytest.importorskip("pupil_apriltags")
    fish = G.Intrinsics(np.array([[F_EQ, 0, 639.5], [0, F_EQ, 359.5], [0, 0, 1]]), np.array([0.02, -0.01, 0.0, 0.0]), W, H, "fisheye")
    for T_true in (T_from(rot([0, 1, 0], 40), [0.62, 0.0, 0.55]), T_from(rot([1, 0, 0], 20), [-0.05, 0.05, 0.5])):  # 48 deg off-axis; centre
        dets = G.detect_tags(_render_tag(3, 0.1, T_true, intr=fish, ss=2), fish, 0.1)
        assert [d.tag_id for d in dets] == [3]
        d = dets[0]
        assert np.abs(d.corners_px - G.project(fish, T_true, G.tag_object_points(0.1))).max() < 1.0
        assert rot_err_deg(d.T_cam_tag[:3, :3], T_true[:3, :3]) < 1.0 and np.linalg.norm(d.T_cam_tag[:3, 3] - T_true[:3, 3]) < 0.005


def test_tag_frame_size_and_unit_contracts():
    pytest.importorskip("pupil_apriltags")
    frame = _render_tag(7, TAG_M, TAG_POSES["tilted_x30"])
    with pytest.raises(G.GeometryError):
        G.detect_tags(cv2.resize(frame, (640, 360)), _pinhole(), TAG_M)
    with pytest.raises(AssertionError):  # the contract's "assert" wording also catches it
        G.detect_tags(frame, _pinhole().scaled_to(640, 360), TAG_M)
    with pytest.raises(G.GeometryError):
        G.detect_tags(frame, _pinhole(), 80.0)  # millimetres by mistake
    no_pose = G.detect_tags(frame, None, None)
    assert [d.tag_id for d in no_pose] == [7] and no_pose[0].T_cam_tag is None and no_pose[0].err_px is None


def test_tag_detector_cache_is_per_family_and_thread_safe():
    pytest.importorskip("pupil_apriltags")
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9), 4, 350)
    frame25 = render_plane(np.pad(m, 50, constant_values=255), 350 / 0.07, TAG_POSES["tilted_x30"], K_TRUE, plane_origin=(-0.045, -0.045), ss=2)
    assert [d.tag_id for d in G.detect_tags(frame25, None, None)] == []  # tag36h11 detector
    assert [d.tag_id for d in G.detect_tags(frame25, None, None, families="tag25h9")] == [4]  # used to reuse the first detector
    frame = _render_tag(7, TAG_M, TAG_POSES["oblique60"]); ref = G.detect_tags(frame, _pinhole(), TAG_M)[0].T_cam_tag
    results, errors = [], []

    def worker():
        try:
            for _ in range(5):
                results.append(G.detect_tags(frame, _pinhole(), TAG_M)[0].T_cam_tag)
        except Exception as e:  # noqa: BLE001 - any failure in a worker thread is collected and asserted below
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(6)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors and len(results) == 30 and all(np.allclose(r, ref) for r in results)


# ----------------------------------------------------------------------------- board degeneracy, registration

def test_board_pose_rejects_degenerate_corner_sets():
    rng = np.random.default_rng(3); intr = _pinhole(); cc = SPEC.corners_m()
    T_wc = look_at([0.3, -0.25, -0.6], BOARD_CENTRE); uv = G.project(intr, G.inv(T_wc), cc) + rng.normal(0, 0.3, (len(cc), 2))
    row = np.arange(6)  # one row of inner corners: 6 collinear points (rotation about the row is unconstrained)
    assert G.pose_from_board_corners(uv[row], row, intr, SPEC) is None
    assert G.pose_from_board_corners(uv[:5], np.arange(5), intr, SPEC) is None
    two_rows = np.arange(12); r = G.pose_from_board_corners(uv[two_rows], two_rows, intr, SPEC)
    assert r is not None and np.linalg.norm(G.inv(r[0])[:3, 3] - T_wc[:3, 3]) < 0.01
    nan = uv.copy(); nan[3] = np.nan
    assert G.pose_from_board_corners(nan, np.arange(len(cc)), intr, SPEC) is None
    assert G.pose_from_board_corners(uv[rng.permutation(len(cc))], np.arange(len(cc)), intr, SPEC) is None  # scrambled ids: huge error
    with pytest.raises(G.GeometryError):
        G.pose_from_board_corners(uv, np.arange(len(cc)) + 30, intr, SPEC)


def test_register_cameras_gate_and_triangulation():
    img, ppm = _board_image(); intr = _pinhole()
    T_a = look_at([-0.1, -0.2, -0.65], BOARD_CENTRE); T_b = look_at([0.5, -0.15, -0.6], BOARD_CENTRE)
    views = {n: render_plane(img, ppm, G.inv(T), K_TRUE, ss=2) for n, T in (("a", T_a), ("b", T_b))}
    # camera c has a wide rational lens and fills its view with the board: with its real model it registers to 0.2 mm, with a
    # pinhole guess (no distortion) the gate rejects it
    wide = G.Intrinsics(np.array([[330.0, 0, 322], [0, 330.0, 178], [0, 0, 1]]), np.array([0.3, 0.05, 0.001, -0.001, 0.0, 0.6, 0.1, 0.0]), 640, 360, "rational")
    guess = G.Intrinsics(wide.K, np.zeros(5), 640, 360)
    T_c = look_at([0.175, 0.2, -0.18], BOARD_CENTRE); view_c = render_plane(img, ppm, G.inv(T_c), None, intr=wide, ss=2)
    reg = G.register_cameras_from_board({"a": (views["a"], intr), "b": (views["b"], intr), "c": (view_c, guess)}, SPEC)
    assert set(reg) == {"a", "b"} and G.board_pose(view_c, guess, SPEC, max_err_px=1e3)[1] > 3.0
    for name, T_true in (("a", T_a), ("b", T_b)):
        assert np.linalg.norm(reg[name][0][:3, 3] - T_true[:3, 3]) < 0.001 and rot_err_deg(reg[name][0][:3, :3], T_true[:3, :3]) < 0.1
    reg_c = G.register_cameras_from_board({"c": (view_c, wide)}, SPEC)["c"]
    assert reg_c[1] < 0.3 and np.linalg.norm(reg_c[0][:3, 3] - T_c[:3, 3]) < 0.001 and rot_err_deg(reg_c[0][:3, :3], T_c[:3, :3]) < 0.2
    # the gate is necessary, not sufficient: a board small in the view hides the missing distortion (0.4 px, 1 cm off) and a
    # planar board hides a 20 % focal error; only calibrated intrinsics should register cameras
    T_far = look_at([0.2, 0.45, -0.4], BOARD_CENTRE); r = G.board_pose(render_plane(img, ppm, G.inv(T_far), None, intr=wide, ss=2), guess, SPEC)
    assert r is not None and r[1] < 1.0 and np.linalg.norm(G.inv(r[0])[:3, 3] - T_far[:3, 3]) > 0.005
    assert G.board_pose(views["a"], G.Intrinsics(K_TRUE * [[0.8], [0.8], [1]], np.zeros(5), W, H), SPEC) is not None
    cams = [G.Camera("a", intr, reg["a"][0]), G.Camera("b", intr, reg["b"][0])]
    X_true = np.array([[0.1, 0.2, -0.15], [0.3, 0.05, -0.30], [0.2, 0.15, -0.05]])  # above the board = negative Z
    pts = np.stack([G.project(intr, G.inv(T_a), X_true), G.project(intr, G.inv(T_b), X_true)])
    X, E = G.triangulate_many(cams, pts, np.ones((2, 3)))
    assert np.all(np.linalg.norm(X - X_true, axis=1) < 0.002) and np.all(E < 0.5)


# ----------------------------------------------------------------------------- calibration adversarial

def _table_views(rng, cam_pos, n=12, tilt=0.0):
    """A FIXED camera watching the board slide/spin on the table plane (optionally tilted by up to +-tilt deg)."""
    img, ppm = _board_image(px_per_m=3000); T_cw = G.inv(look_at(cam_pos, [0, 0.001, 0])); frames = []
    for _ in range(n):
        T_tb = T_from(rot([0, 0, 1], rng.uniform(-40, 40)) @ rot([1, 0, 0], rng.uniform(-tilt, tilt) + 1e-9) @ rot([0, 1, 0], rng.uniform(-tilt, tilt) + 1e-9),
                      [rng.uniform(-0.25, 0.25), rng.uniform(-0.2, 0.2), 0])
        f = render_plane(img, ppm, T_cw @ T_tb, K_TRUE, plane_origin=(-0.175, -0.125), ss=2)
        frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
    return frames


@pytest.mark.parametrize("cam_pos", [[-0.6, -0.5, -0.8], [0.0, 0.0, -0.9]], ids=["oblique", "top_down"])
def test_calibration_rejects_one_plane_fixed_camera(cam_pos):
    """The review's degenerate case: fx 8,275-95,392 at 0.05 px RMS was returned as a calibration."""
    intr, rep = G.calibrate_intrinsics_report(_table_views(np.random.default_rng(1), cam_pos), SPEC)
    assert intr is None and not rep["ok"] and rep["n_views"] >= 8 and "tilt" in rep["reason"]
    assert G.calibrate_intrinsics(_table_views(np.random.default_rng(2), cam_pos), SPEC) is None


def _corner_views(truth, rng, n=25, noise=0.2):
    cc = SPEC.corners_m(); views = []
    while len(views) < n:
        d = np.array([rng.uniform(-0.9, 0.9), rng.uniform(-0.6, 0.6), 1.0]); ctr = d / np.linalg.norm(d) * rng.uniform(0.35, 0.6)
        R = rot([1, 0, 0], rng.uniform(-45, 45)) @ rot([0, 1, 0], rng.uniform(-45, 45)) @ rot([0, 0, 1], rng.uniform(-30, 30))
        uv = G.project_cam(truth, cc @ R.T + (ctr - R @ [0.175, 0.125, 0]))
        if np.isfinite(uv).all() and np.all((uv > 5) & (uv < [truth.width - 5, truth.height - 5])):
            views.append(uv + rng.normal(0, noise, uv.shape))
    return views, [np.arange(len(cc))] * n


def _ray_err_deg(a: G.Intrinsics, b: G.Intrinsics, px):
    na, nb = G.undistort_points(a, px), G.undistort_points(b, px)
    va, vb = np.c_[na, np.ones(len(na))], np.c_[nb, np.ones(len(nb))]
    c = (va * vb).sum(1) / np.linalg.norm(va, axis=1) / np.linalg.norm(vb, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def test_calibration_wide_lens_models():
    """120 deg equidistant fisheye: the pinhole model is rejected as too simple (it fits the corners to 0.3 px while its
    focal length and principal point are biased); the fisheye model recovers the lens across the whole image."""
    truth = G.Intrinsics(np.array([[F_EQ, 0, 639.5], [0, F_EQ, 359.5], [0, 0, 1]]), np.zeros(4), W, H, "fisheye")
    corners, ids = _corner_views(truth, np.random.default_rng(7))
    grid = np.array([[u, v] for u in np.linspace(0, W - 1, 33) for v in np.linspace(0, H - 1, 19)])
    pin, rep = G.calibrate_from_corners_report(corners, ids, (W, H), SPEC, "pinhole")
    assert pin is None and "too simple" in rep["reason"] and rep["rms_px"] < 0.5
    fish, rep = G.calibrate_from_corners_report(corners, ids, (W, H), SPEC, "fisheye")
    assert fish is not None and fish.model == "fisheye" and rep["rms_px"] < 0.3 and abs(fish.K[0, 0] / F_EQ - 1) < 0.005
    assert _ray_err_deg(fish, truth, grid).max() < 0.5
    rat, rep = G.calibrate_from_corners_report(corners, ids, (W, H), SPEC, "rational")
    assert rat is not None and rat.dist.size == 8 and rat.model == "rational" and np.nanmax(_ray_err_deg(rat, truth, grid[np.hypot(*(grid - [640, 360]).T) < 500])) < 0.5


def test_calibration_drops_outlier_view_and_needs_enough_views():
    truth = _pinhole(); corners, ids = _corner_views(truth, np.random.default_rng(5), n=12, noise=0.2)
    corners[4] = corners[4] + np.random.default_rng(6).normal(0, 3.0, corners[4].shape)  # one badly detected view
    intr, rep = G.calibrate_from_corners_report(corners, ids, (W, H), SPEC)
    assert intr is not None and rep["dropped_views"] == [4] and abs(intr.K[0, 0] / 900 - 1) < 0.005 and rep["rms_px"] < 0.35
    intr, rep = G.calibrate_from_corners_report(corners[:5], ids[:5], (W, H), SPEC)
    assert intr is None and "need >= 8" in rep["reason"]
    with pytest.raises(G.GeometryError):
        G.calibrate_intrinsics_report([np.zeros((720, 1280), np.uint8), np.zeros((360, 640), np.uint8)], SPEC)


# ----------------------------------------------------------------------------- lens models

def test_undistort_points_converges_on_strong_distortion():
    gopro = G.Intrinsics(K_GOPRO5, D_GOPRO5, W, H)
    grid = np.array([[u, v] for u in np.linspace(0, W - 1, 65) for v in np.linspace(0, H - 1, 37)])
    n = G.undistort_points(gopro, grid); ok = np.isfinite(n).all(1)
    rd = np.hypot(*((grid - K_GOPRO5[:2, 2]) / np.diag(K_GOPRO5)[:2]).T)  # the model never produces rd > 0.994: those pixels have no ray
    assert ok[rd < 0.98].all() and not ok[rd > 1.01].any() and ok.mean() > 0.85
    assert np.abs(G.project_cam(gopro, np.c_[n[ok], np.ones(ok.sum())]) - grid[ok]).max() < 1e-3
    assert G.undistort_points(gopro, grid.reshape(65, 37, 2)).shape == (65, 37, 2)
    fish = G.Intrinsics(np.array([[F_EQ, 0, 639.5], [0, F_EQ, 359.5], [0, 0, 1]]), np.array([0.05, -0.02, 0.01, -0.002]), W, H, "fisheye")
    n = G.undistort_points(fish, grid); assert np.isfinite(n).all()
    assert np.abs(G.project_cam(fish, np.c_[n, np.ones(len(n))]) - grid).max() < 1e-3
    assert np.isnan(G.undistort_points(fish, [[np.nan, 3.0]])).all()


def test_project_nan_behind_camera_and_beyond_fold():
    gopro = G.Intrinsics(K_GOPRO5, D_GOPRO5, W, H)  # radial mapping peaks at r = 1.62 (58 deg off-axis)
    out = G.project_cam(gopro, [[0, 0, -1.0], [1.5, 0.0, 1.0], [2.0, 0.0, 1.0]])  # behind; 56 deg; 63 deg (beyond the fold)
    assert np.isnan(out[0]).all() and 1200 < out[1, 0] < W and np.isnan(out[2]).all()
    folded, _ = cv2.projectPoints(np.array([[[2.0, 0.0, 1.0]]]), np.zeros(3), np.zeros(3), K_GOPRO5, D_GOPRO5)
    assert 1000 < folded.ravel()[0] < 1100  # OpenCV alone: the 63 deg ray lands inside the image, where a ~45 deg ray belongs


# ----------------------------------------------------------------------------- triangulation adversarial

K300 = np.array([[300.0, 0, 320], [0, 300.0, 180], [0, 0, 1]])


def _cam(name, pos, tgt, K=K300, size=(640, 360), intr=None):
    return G.Camera(name, intr or G.Intrinsics(K, np.zeros(5), *size), look_at(pos, tgt))


STATION = [_cam("ego", [0.2, -0.1, -0.45], [0.2, 0.2, 0]), _cam("exoA", [-1.5, -0.6, -1.6], [0.2, 0.2, 0]), _cam("exoB", [1.9, -0.4, -1.5], [0.2, 0.2, 0])]


def _obs(cams, X, rng=None, sigma=0.0):
    p = np.stack([G.project(c.intr, G.inv(c.T_world_cam), np.atleast_2d(X)) for c in cams])  # [C, N, 2]
    return p if rng is None else p + rng.normal(0, sigma, p.shape)


def _rand_points(rng, n):
    return np.c_[rng.uniform(0.0, 0.4, n), rng.uniform(0.1, 0.3, n), rng.uniform(-0.25, -0.05, n)]


def test_triangulation_is_maximum_likelihood():
    """Gauss-Newton refinement reaches the reprojection-error optimum (the old unweighted DLT was ~45 % worse here)."""
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    rng = np.random.default_rng(11); X_true = _rand_points(rng, 150); pts = _obs(STATION, X_true, rng, 1.0)
    X, E = G.triangulate_many(STATION, pts, np.ones(pts.shape[:2]))
    assert np.isfinite(X).all() and np.nanmedian(E) < 1.5
    for n in range(0, 150, 15):
        ml = least_squares(lambda Y, n=n: (_obs(STATION, Y)[:, 0] - pts[:, n]).ravel(), X[n]).x
        assert np.linalg.norm(ml - X[n]) < 1e-5
    assert np.median(np.linalg.norm(X - X_true, axis=1)) < 0.007


@pytest.mark.parametrize("case", ["behind_both", "parallel", "far_small_baseline", "single_view", "all_low_conf"])
def test_triangulation_rejects_degenerate(case):
    rig = [_cam("a", [0, 0, -1], [0, 0, 0], K_TRUE, (W, H)), _cam("b", [0.3, 0, -1], [0.3, 0, 0], K_TRUE, (W, H))]
    conf = np.ones((2, 1))
    if case == "behind_both":  # rays whose lines cross 0.6 m BEHIND both cameras (old code: accepted at 0.000 px)
        Xb = np.array([0.25, 0.1, -1.6]); pts = np.stack([[(K_TRUE @ (G.apply(G.inv(c.T_world_cam), Xb) / G.apply(G.inv(c.T_world_cam), Xb)[2]))[:2]] for c in rig])
    elif case == "parallel":  # identical pixels in parallel cameras: meet at infinity (old code: 1e15 m at 1e-13 px)
        pts = np.array([[[700.0, 400.0]], [[700.0, 400.0]]])
    elif case == "far_small_baseline":  # 20 m away, 0.3 m baseline: 0.9 deg between rays
        pts = _obs(rig, np.array([0.1, 0.0, 20.0]), np.random.default_rng(0), 0.5)
    elif case == "single_view":
        pts = _obs(rig, np.array([0.1, 0.1, 0.5])); pts[1] = np.nan
    else:
        pts = _obs(rig, np.array([0.1, 0.1, 0.5])); conf[:] = 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        X, E = G.triangulate_many(rig, pts, conf)
        assert np.isnan(X).all() and np.isnan(E).all()
        assert G.triangulate(rig, [None if np.isnan(p).any() or conf[c, 0] < 0.4 else p for c, p in enumerate(pts[:, 0])]) is None


def test_triangulation_accepts_moderate_range():
    rig = [_cam("a", [0, 0, -1], [0, 0, 0], K_TRUE, (W, H)), _cam("b", [0.3, 0, -1], [0.3, 0, 0], K_TRUE, (W, H))]
    r = G.triangulate(rig, list(_obs(rig, np.array([0.1, 0.0, 3.0]))[:, 0]))
    assert r is not None and np.linalg.norm(r[0] - [0.1, 0, 3.0]) < 1e-6


def test_triangulation_drops_one_outlier_view_of_three():
    """One view 30 px off (e.g. the wrong person in one camera): the old code dropped 194/200 such points."""
    rng = np.random.default_rng(12); X_true = _rand_points(rng, 60); pts = _obs(STATION, X_true, rng, 1.0); pts[2, :, 0] += 30.0
    X, _, used = G.triangulate_many(STATION, pts, np.ones(pts.shape[:2]), return_used=True)
    assert np.isfinite(X).all() and (used[:2].all()) and not used[2].any()
    X2, _ = G.triangulate_many(STATION[:2], pts[:2], np.ones((2, 60)))  # the ML estimate from the two good views
    assert np.abs(X - X2).max() < 1e-6
    err = np.linalg.norm(X - X_true, axis=1)
    X1, _, used1 = G.triangulate_many(STATION, pts, np.ones(pts.shape[:2]), outlier_px=float("inf"), max_err_px=None, return_used=True)
    assert used1.all() and np.median(np.linalg.norm(X1 - X_true, axis=1)) > 3 * np.median(err)  # without rejection the bad view biases every point


def test_triangulation_confidence_weighting_and_missing_views():
    rng = np.random.default_rng(13); X_true = _rand_points(rng, 200); pts = _obs(STATION, X_true)
    pts += rng.normal(0, 1, pts.shape) * np.array([0.5, 0.5, 6.0])[:, None, None]  # third view much noisier
    w = np.ones(pts.shape[:2]); w[2] = (0.5 / 6.0) ** 2
    Xw, _ = G.triangulate_many(STATION, pts, w, min_conf=0.0, outlier_px=float("inf"))
    Xu, _ = G.triangulate_many(STATION, pts, np.ones(pts.shape[:2]), outlier_px=float("inf"))
    assert np.median(np.linalg.norm(Xw - X_true, axis=1)) < 0.5 * np.median(np.linalg.norm(Xu - X_true, axis=1))
    miss = _obs(STATION, X_true[:3]); miss[0, 0] = np.nan; conf = np.ones((3, 3)); conf[1, 1] = 0.1; conf[:2, 2] = 0.0
    X, _, used = G.triangulate_many(STATION, miss, conf, return_used=True)
    assert np.allclose(X[:2], X_true[:2], atol=1e-6) and np.isnan(X[2]).all()
    assert used[:, 0].tolist() == [False, True, True] and used[:, 1].tolist() == [True, False, True]


def test_triangulation_through_distorted_lenses():
    fish = G.Intrinsics(np.array([[F_EQ / 2, 0, 319.5], [0, F_EQ / 2, 179.5], [0, 0, 1]]), np.array([0.03, -0.01, 0.0, 0.0]), 640, 360, "fisheye")
    rat = G.Intrinsics(np.array([[330.0, 0, 322], [0, 330.0, 178], [0, 0, 1]]), np.array([0.3, 0.05, 0.001, -0.001, 0.0, 0.6, 0.1, 0.0]), 640, 360, "rational")
    rig = [_cam("ego", [0.2, -0.1, -0.45], [0.2, 0.2, 0], intr=fish), _cam("exo", [-1.5, -0.6, -1.6], [0.2, 0.2, 0], intr=rat)]
    X_true = _rand_points(np.random.default_rng(14), 40)
    X, E = G.triangulate_many(rig, _obs(rig, X_true), np.ones((2, 40)))
    assert np.abs(X - X_true).max() < 1e-6 and np.nanmax(E) < 1e-4


def test_triangulation_shape_contracts():
    with pytest.raises(G.GeometryError):
        G.triangulate(STATION, [np.zeros(2)] * 2)  # 3 cameras, 2 observations (zip used to truncate silently)
    with pytest.raises(G.GeometryError):
        G.triangulate_many(STATION, np.zeros((2, 5, 2)), np.ones((2, 5)))
    with pytest.raises(G.GeometryError):
        G.triangulate_many(STATION, np.zeros((3, 5, 2)), np.ones((3, 4)))
    X, E = G.triangulate_many(STATION, np.zeros((3, 0, 2)), np.ones((3, 0)))
    assert X.shape == (0, 3) and E.shape == (0,)


# ----------------------------------------------------------------------------- similarity helpers

def test_similarity_helpers():
    rng = np.random.default_rng(15); R = rot([0.3, -1, 0.5], 70); t = np.array([0.3, -0.2, 1.1]); s = 2.7
    src = rng.normal(0, 1, (10, 3)); dst = s * src @ R.T + t
    T, s_est = G.fit_similarity(src, dst)
    assert np.allclose(T[:3, :3], R, atol=1e-9) and np.allclose(T[:3, 3], t) and abs(s_est - s) < 1e-9
    two_centres = np.repeat(src[:2], 4, axis=0) + rng.normal(0, 1e-4, (8, 3))
    assert G.fit_similarity(two_centres, s * two_centres @ R.T + t) is None  # rotation about the baseline is free
    # the same two cameras' ORIENTATIONS determine the rotation (scene_scan's two fixed cameras)
    T_b_a = T_from(R, t); cams_a = [T_from(rot(rng.normal(size=3), 50), rng.normal(0, 0.5, 3)) for _ in range(2)]
    cams_b = [T_b_a @ np.block([[Ta[:3, :3], s * Ta[:3, 3:]], [np.zeros((1, 3)), np.ones((1, 1))]]) for Ta in cams_a]
    T, s_est = G.similarity_from_camera_poses(np.stack(cams_a), np.stack(cams_b))
    assert np.allclose(T[:3, :3], R, atol=1e-9) and np.allclose(T[:3, 3], t) and abs(s_est - s) < 1e-9
    assert G.similarity_from_camera_poses(np.stack(cams_a[:1]), np.stack(cams_b[:1])) is None  # one centre: no scale
    noisy = np.stack([R @ rot(rng.normal(size=3), 2) for _ in range(50)])
    assert rot_err_deg(G.average_rotations(noisy), R) < 0.5
