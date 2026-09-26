"""Synthetic end-to-end check of the metric geometry: render a ChArUco board and AprilTags into two virtual
cameras with known poses, then recover intrinsics, camera poses, tag poses and triangulated points."""
import cv2
import numpy as np
import pytest

from duet.playground import geometry as G

W, H = 1280, 720
K_TRUE = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
SPEC = G.BoardSpec(7, 5, 0.05, 0.037)


def _board_image_and_plane(spec):
    px_per_m = 4000
    img = spec.board().generateImage((int(spec.squares_x * spec.square_m * px_per_m), int(spec.squares_y * spec.square_m * px_per_m)))
    return img, px_per_m


def render_plane(img_plane, px_per_m, T_cam_plane, K, size=(W, H), plane_origin=(0.0, 0.0)):
    """Warp a planar image (metres, z=0 in the plane frame) into a camera view. Returns uint8 grayscale, white background."""
    h, w = img_plane.shape[:2]
    # OpenCV board convention: origin at the top-left corner, X right, Y DOWN the image, Z into the board.
    # A camera above the board therefore sits at negative Z. Using this frame directly avoids a mirrored view.
    corners_m = np.array([[0, 0, 0], [w / px_per_m, 0, 0], [w / px_per_m, h / px_per_m, 0], [0, h / px_per_m, 0]], float) + np.array([*plane_origin, 0.0])
    rvec, _ = cv2.Rodrigues(T_cam_plane[:3, :3])
    proj, _ = cv2.projectPoints(corners_m.reshape(-1, 1, 3), rvec, T_cam_plane[:3, 3], K, None)
    dst = proj.reshape(-1, 2).astype(np.float32)
    # anti-alias: shrink the plane image to ~1.5x its projected size before warping (warpPerspective has no area filter)
    proj_w = max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3])); factor = min(1.0, 1.5 * proj_w / w)
    if factor < 1.0:
        img_plane = cv2.resize(img_plane, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA); h, w = img_plane.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img_plane, Hm, size, flags=cv2.INTER_LINEAR, borderValue=255)
    return out


def look_at(cam_pos, target, up=(0, 1, 0)):
    """T_world_cam for a camera at cam_pos looking at target (OpenCV: +Z forward, +Y down)."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float); z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float)); x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam_pos; return T


def test_board_calibration_and_pose_recovery():
    img, ppm = _board_image_and_plane(SPEC)
    intr_true = G.Intrinsics(K_TRUE, np.zeros(5), W, H)
    frames, poses = [], []
    rng = np.random.default_rng(0)
    for k in range(10):  # ten views of the board from different places
        pos = np.array([0.175, 0.125, 0]) + np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -rng.uniform(0.45, 0.7)])
        T_wc = look_at(pos, [0.175 + rng.uniform(-0.05, 0.05), 0.125 + rng.uniform(-0.05, 0.05), 0])
        frames.append(render_plane(img, ppm, G.inv(T_wc), K_TRUE)); poses.append(T_wc)
    intr = G.calibrate_intrinsics(frames, SPEC)
    assert intr is not None and intr.rms_px < 1.0
    assert abs(intr.K[0, 0] - 900) / 900 < 0.03 and abs(intr.K[0, 2] - 640) < 15
    # pose of the first view, recovered with the recovered intrinsics
    r = G.board_pose(frames[0], intr, SPEC); assert r is not None
    T_cam_board, err = r; T_wc_est = G.inv(T_cam_board)
    assert err < 1.0
    assert np.linalg.norm(T_wc_est[:3, 3] - poses[0][:3, 3]) < 0.02  # 2 cm
    assert np.degrees(np.arccos(np.clip((np.trace(T_wc_est[:3, :3].T @ poses[0][:3, :3]) - 1) / 2, -1, 1))) < 1.0


def test_two_camera_registration_and_triangulation():
    img, ppm = _board_image_and_plane(SPEC)
    intr = G.Intrinsics(K_TRUE, np.zeros(5), W, H)
    T_a = look_at([-0.1, -0.2, -0.65], [0.175, 0.125, 0]); T_b = look_at([0.5, -0.15, -0.6], [0.175, 0.125, 0])
    fa, fb = render_plane(img, ppm, G.inv(T_a), K_TRUE), render_plane(img, ppm, G.inv(T_b), K_TRUE)
    reg = G.register_cameras_from_board({"a": (fa, intr), "b": (fb, intr)}, SPEC)
    assert set(reg) == {"a", "b"}
    for name, T_true in (("a", T_a), ("b", T_b)):
        assert np.linalg.norm(reg[name][0][:3, 3] - T_true[:3, 3]) < 0.02
    cams = [G.Camera("a", intr, reg["a"][0]), G.Camera("b", intr, reg["b"][0])]
    # triangulate points hovering above the table (like wrists), from their true projections
    X_true = np.array([[0.1, 0.2, -0.15], [0.3, 0.05, -0.30], [0.2, 0.15, -0.05]])  # above the board = negative Z
    pa = G.project(intr, G.inv(T_a), X_true); pb = G.project(intr, G.inv(T_b), X_true)
    X, E = G.triangulate_many(cams, np.stack([pa, pb]), np.ones((2, 3)))
    assert np.all(np.linalg.norm(X - X_true, axis=1) < 0.02) and np.all(E < 2.0)


def test_apriltag_pose_recovery():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(d, 7, 400); border = 60
    canvas = np.full((400 + 2 * border, 400 + 2 * border), 255, np.uint8); canvas[border:-border, border:-border] = tag
    tag_m = 0.08; ppm = 400 / tag_m
    intr = G.Intrinsics(K_TRUE, np.zeros(5), W, H)
    # tag frame: pupil-apriltags puts the origin at the tag centre; our plane image origin is the canvas corner
    T_wc = look_at([0.15, -0.3, -0.5], [0, 0, 0])
    frame = render_plane(canvas, ppm, G.inv(T_wc), K_TRUE, plane_origin=(-(tag_m / 2 + border / ppm), -(tag_m / 2 + border / ppm)))
    dets = G.detect_tags(frame, intr, tag_m)
    assert [t.tag_id for t in dets] == [7]
    T_cam_tag = dets[0].T_cam_tag; assert T_cam_tag is not None
    # tag centre should be at the world origin: T_world_tag = T_wc @ T_cam_tag -> translation ~ 0
    t_world = (T_wc @ T_cam_tag)[:3, 3]
    assert np.linalg.norm(t_world) < 0.03, t_world
