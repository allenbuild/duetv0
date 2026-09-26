"""Metric geometry for the playground: board calibration, tag poses, camera registration, triangulation.

World frame = the ChArUco board frame at station setup (origin at the board's corner, X along the
long side, Y along the short side, Z up out of the board). Every camera, head tag and object tag is
expressed in this frame, so two people recorded by different cameras land in one metric space.

Conventions (OpenCV): a camera pose is (R, t) with X_cam = R @ X_world + t. We store camera-to-world
as a 4x4 T_world_cam. Intrinsics K are 3x3 with fx, fy, cx, cy; distortion is OpenCV's 5- or 8-vector.

Tags: AprilTag 36h11 (pupil-apriltags). Tag poses need K and the physical tag size; the returned
T_cam_tag places the tag frame (Z out of the tag face) in the camera.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_5X5_100


@dataclass
class BoardSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_m: float = 0.05
    marker_m: float = 0.037

    def board(self) -> "cv2.aruco.CharucoBoard":
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m, cv2.aruco.getPredefinedDictionary(ARUCO_DICT))


@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms_px: float | None = None

    def to_json(self) -> dict:
        return {"K": self.K.tolist(), "dist": self.dist.ravel().tolist(), "width": self.width, "height": self.height, "rms_px": self.rms_px}

    @classmethod
    def from_json(cls, d: dict) -> "Intrinsics":
        return cls(np.array(d["K"], float), np.array(d["dist"], float), int(d["width"]), int(d["height"]), d.get("rms_px"))

    @classmethod
    def nominal(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Pinhole guess from a horizontal field of view; use only until the board has been seen."""
        f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
        return cls(np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], float), np.zeros(5), width, height, None)


# ----------------------------------------------------------------------------- board

def detect_board(gray: np.ndarray, spec: BoardSpec):
    """Return (charuco_corners [N,1,2], charuco_ids [N,1]) or (None, None)."""
    board = spec.board()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)
    if corners is None or ids is None or len(ids) < 6:
        return None, None
    return corners, ids


def calibrate_intrinsics(frames: list[np.ndarray], spec: BoardSpec) -> Intrinsics | None:
    """Intrinsics from >= 8 board views (grayscale frames). Returns None if too few detections."""
    board = spec.board(); all_c, all_i = [], []
    for g in frames:
        c, i = detect_board(g, spec)
        if c is not None:
            all_c.append(c); all_i.append(i)
    if len(all_c) < 6:
        return None
    h, w = frames[0].shape[:2]
    obj_pts, img_pts = [], []
    for c, i in zip(all_c, all_i):
        o, im = board.matchImagePoints(c, i)
        if o is not None and len(o) >= 6:
            obj_pts.append(o); img_pts.append(im)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None)
    return Intrinsics(K, dist, w, h, float(rms))


def board_pose(gray: np.ndarray, intr: Intrinsics, spec: BoardSpec) -> tuple[np.ndarray, float] | None:
    """T_cam_board (4x4) from one frame, plus mean reprojection error in px; None if the board is not seen."""
    c, i = detect_board(gray, spec)
    if c is None:
        return None
    o, im = spec.board().matchImagePoints(c, i)
    if o is None or len(o) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(o, im, intr.K, intr.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(o, rvec, tvec, intr.K, intr.dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - im.reshape(-1, 2), axis=1).mean())
    return rt_to_T(rvec, tvec), err


# ----------------------------------------------------------------------------- tags

@dataclass
class TagDetection:
    tag_id: int
    corners_px: np.ndarray  # [4,2]
    T_cam_tag: np.ndarray | None  # 4x4
    err_px: float | None


_DETECTOR = None


def detect_tags(gray: np.ndarray, intr: Intrinsics | None, tag_size_m: float, families: str = "tag36h11") -> list[TagDetection]:
    global _DETECTOR
    import pupil_apriltags as apr
    if _DETECTOR is None:
        _DETECTOR = apr.Detector(families=families, nthreads=4, quad_decimate=1.0, refine_edges=True)
    out = []
    if intr is not None:
        und = cv2.undistort(gray, intr.K, intr.dist) if np.any(intr.dist) else gray
        dets = _DETECTOR.detect(und, estimate_tag_pose=True, camera_params=(intr.K[0, 0], intr.K[1, 1], intr.K[0, 2], intr.K[1, 2]), tag_size=tag_size_m)
    else:
        dets = _DETECTOR.detect(gray)
    # pupil-apriltags' tag frame is X right, Y up, Z out of the tag toward the camera. Our world/board frame is
    # OpenCV's: X right, Y down, Z into the surface. Right-multiply by diag(1,-1,-1) so tag frames match board frames
    # (translation unchanged; rotation flipped about X).
    FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
    for d in dets:
        T = None
        if intr is not None and d.pose_R is not None:
            T = np.eye(4); T[:3, :3] = d.pose_R; T[:3, 3] = d.pose_t.ravel(); T = T @ FLIP
        out.append(TagDetection(int(d.tag_id), np.asarray(d.corners, float), T, float(d.pose_err) if intr is not None else None))
    return out


# ----------------------------------------------------------------------------- transforms

def rt_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1)); T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(tvec, float).ravel(); return T


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]; Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t; return Ti


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """T [4,4] applied to points [..., 3]."""
    return p @ T[:3, :3].T + T[:3, 3]


def project(intr: Intrinsics, T_cam_world: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
    px, _ = cv2.projectPoints(pts_world.reshape(-1, 1, 3).astype(float), rvec, T_cam_world[:3, 3], intr.K, intr.dist)
    return px.reshape(-1, 2)


# ----------------------------------------------------------------------------- triangulation

@dataclass
class Camera:
    name: str
    intr: Intrinsics
    T_world_cam: np.ndarray  # 4x4

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix for undistorted normalised... no: for pixel coords after undistortPoints with K."""
        T_cw = inv(self.T_world_cam); return self.intr.K @ T_cw[:3, :4]


def triangulate(cams: list[Camera], pts_px: list[np.ndarray | None]) -> tuple[np.ndarray, float] | None:
    """Multi-view DLT for one point. pts_px: per camera [2] pixel (distorted) or None. Returns (X_world [3], mean reproj px)."""
    rows, used = [], []
    for cam, p in zip(cams, pts_px):
        if p is None or not np.all(np.isfinite(p)):
            continue
        u = cv2.undistortPoints(np.asarray(p, float).reshape(1, 1, 2), cam.intr.K, cam.intr.dist, P=cam.intr.K).reshape(2)
        P = cam.P; rows.append(u[0] * P[2] - P[0]); rows.append(u[1] * P[2] - P[1]); used.append((cam, p))
    if len(used) < 2:
        return None
    A = np.stack(rows); _, _, Vt = np.linalg.svd(A); X = Vt[-1]; X = X[:3] / X[3]
    errs = [np.linalg.norm(project(c.intr, inv(c.T_world_cam), X[None])[0] - np.asarray(p, float)) for c, p in used]
    return X, float(np.mean(errs))


def triangulate_many(cams: list[Camera], pts: np.ndarray, conf: np.ndarray, min_conf: float = 0.4, max_err_px: float = 25.0):
    """pts [C, N, 2] pixels per camera per keypoint, conf [C, N]. Returns X [N, 3] (NaN where < 2 views or high error), err [N]."""
    C, N = conf.shape; X = np.full((N, 3), np.nan); E = np.full(N, np.nan)
    for n in range(N):
        views = [pts[c, n] if conf[c, n] >= min_conf else None for c in range(C)]
        r = triangulate(cams, views)
        if r is not None and r[1] <= max_err_px:
            X[n], E[n] = r
    return X, E


# ----------------------------------------------------------------------------- rig registration

def register_cameras_from_board(views: dict[str, tuple[np.ndarray, Intrinsics]], spec: BoardSpec) -> dict[str, tuple[np.ndarray, float]]:
    """Given one grayscale frame per camera that shows the board, return name -> (T_world_cam, reproj px). World = board."""
    out = {}
    for name, (gray, intr) in views.items():
        r = board_pose(gray, intr, spec)
        if r is not None:
            T_cam_board, err = r; out[name] = (inv(T_cam_board), err)  # T_world_cam with world == board
    return out


def head_pose_from_tag(T_world_fixedcam: np.ndarray, T_fixedcam_tag: np.ndarray, T_tag_headcam: np.ndarray) -> np.ndarray:
    """Head camera pose in world from a fixed camera seeing the head tag, given the once-measured tag-to-head-camera offset."""
    return T_world_fixedcam @ T_fixedcam_tag @ T_tag_headcam
