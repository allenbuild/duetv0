"""World-frame stages: camera calibration + board registration, AprilTags, head poses, 3D bodies/hands, cross-person
features. Everything is metric (metres) and expressed in ONE board frame.

Frames and units
  world   the ChArUco board frame at the calibration anchor (calib.json "anchor_frame"): origin at the board's top-left
          outer corner, X along squares_x, Y along squares_y, Z INTO the board (geometry.py). Above a face-up board is Z < 0.
  camera  OpenCV axes (x right, y down, z forward). T_world_cam maps camera coordinates to world (translation = camera
          centre). ZED poses are converted to these axes from the convention recorded in zed/<s>/export.json.
  pixels  2D detections (body2d, hands) live in the EXTRACTED frames (derived/frames, C4); calib.json intrinsics are
          scaled to that size. Board and tags are detected on NATIVE frames decoded from the source video at the frames
          index's t_src_s (long side capped at detect_max_px, verified against the extracted frame), with the intrinsics
          scaled to the decoded size; the extracted frames are the fallback.
  time    processed frame k = frames index row k; t_src_s is the stream time of the source frame actually used.

rig.json (episode dir, optional; DEFAULT_RIG fills the rest)
  board            {"squares_x", "squares_y", "square_m", "marker_m"}  (metres)
  tag_size_m       edge of the tag's BLACK border square (metres); required when head_tags/object_tags are set (no default)
  head_tags        {person or stream name: tag id}      AprilTag on each head strap, seen by the fixed cameras
  object_tags      {object name: tag id}
  T_tag_headcam    {same key as head_tags: 4x4}  REQUIRED for each head tag: the head camera's pose IN THE TAG FRAME
                   (head-camera coords -> tag coords, metres; rigid, |t| < 0.5 m). There is no identity default.
  intrinsics       {stream: Intrinsics JSON} at any resolution with the display aspect (e.g. the native 3840x2160 K;
                   rescaled to every image size used), for the display orientation
  hfov_deg         {stream: nominal horizontal FOV} fallback; nominal cameras are NOT registered unless "allow_nominal"
  lens_model       {stream: "pinhole" | "rational" | "fisheye"} for board self-calibration / nominal FOV (wide lenses)
  allow_nominal    false
  calib_window_s   20.0, calib_hz 2.0: calib samples the first calib_window_s of the common window at calib_hz
  detect_max_px    1920: long-side cap for native board/tag detection
  tags_native      true: tags on native frames (decodes the whole source video; see tags())
  wall_tags        not supported (ignored with a note)

Stages
  calib     per usable stream: intrinsics (ZED factory > rig.json > board self-calibration > nominal FOV) and board poses on
            the sampled native frames. Fixed cameras are registered at the ANCHOR frame where most of them see the board
            at once, averaged over the samples where the board stays still; a camera that sees the board only at other
            times is chained through a registered camera that sees it at the same frame. Inter-camera residuals are
            reported and inconsistent cameras unregistered. Ego (moving) cameras get board-frame poses at their board
            sightings (simultaneous with a registered fixed camera, or inside the anchor's still-board span, or - with no
            fixed camera - assuming a static board, flagged). Nothing is written when no camera registers.
  tags      configured AprilTag ids in every registered fixed view -> tag poses in world (skipped without tags in rig.json).
  headpose  per usable ego: T_world_cam per processed frame from (a) ZED tracking converted to OpenCV camera axes and
            registered to the board through the ego's board sightings, else (b) the head tag seen by fixed cameras
            composed with T_tag_headcam. Unregistered tracking gives NaN (never board-frame outputs).
  world3d   bodies triangulated from registered fixed cameras (+ egos with a head pose) after cross-view association and 3D
            identity tracking; slot p is the wearer of ep.egos()[p] when a track's head stays at that ego camera, else
            tracks fill the free slots. ZED-depth hands, per-frame head poses, tag objects, cross-person features.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from duet.geometry.alignment import PosePairs, fit_pose_alignment, pose_alignment_residuals
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.schemas.common import DistanceUnit, FrameId, Provenance

from . import geometry as G
from . import perception as P
from . import runtime as RT
from .episode import Episode, Stream

DEFAULT_RIG = {"board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037}, "tag_size_m": None, "head_tags": {},
               "object_tags": {}, "T_tag_headcam": {}, "hfov_deg": {}, "intrinsics": {}, "allow_nominal": False, "calib_window_s": 20.0,
               "calib_hz": 2.0, "detect_max_px": 1920, "tags_native": True, "lens_model": {}}
BOARD_MAX_ERR_PX = 1.0         # board PnP gate, mean reprojection error in EXTRACTED-frame px (x3 for nominal intrinsics)
STATIC_TOL = (0.01, 1.0)       # (m, deg): a static camera's two board poses closer than this = the board did not move
REG_TOL = (0.03, 2.0)          # (m, deg): median inter-camera registration residual above this unregisters a camera
ZED_REG_TOL = (0.05, 3.0)      # (m, deg): ZED->board registrations from different sightings farther than this are outliers
MAX_T_TAG_HEADCAM_M = 0.5
TAG_AGREE = (0.05, 10.0)       # (m, deg): tag poses from different cameras that agree
TAG_AMBIGUOUS = 1.5            # alt_err_px / err_px below this: the tag's orientation is ambiguous (planar flip)
# ZED camera axes -> OpenCV camera axes, per export.json coordinate_system: T_zedcam_cvcam (OpenCV coords -> ZED camera coords)
ZED_AXES = {"IMAGE": np.eye(4), "RIGHT_HANDED_Y_UP": np.diag([1.0, -1.0, -1.0, 1.0])}  # Y_UP camera: x right, y up, z backward
ZED_LEGACY_CS = "RIGHT_HANDED_Y_UP, metres, T_world_camera"  # what scripts/zed_export.py wrote before export format 2
# body association / tracking (COCO-17)
KP_CONF = 0.4                  # keypoint confidence used for association and triangulation
MIN_KP = 6                     # confident keypoints a detection (or a detection pair) needs
GATE_M = 0.25                  # detection -> track gate: median lateral distance (m) at the track's depth ...
V_MAX = 2.0                    # ... plus V_MAX (m/s) x time since the track was last triangulated
MAX_GAP_S = 1.0                # a track not triangulated for this long ends
SEED_GATE_M = 0.08             # new person from a view pair: median lateral reprojection residual (m)
SCALE_RANGE = (0.6, 1.6)       # plausible body size vs BONE_M (median bone-length ratio)
MIN_TRACK_FRAMES = 3           # shorter tracks are dropped from the output
HEAD_LINK_M = 0.35             # a track whose head stays within this of an ego camera is that camera's wearer
HEAD_KP = [0, 1, 2, 3, 4]      # nose, eyes, ears
FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])  # COCO-17 left <-> right
BONES = np.array([(5, 6), (11, 12), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)])
BONE_M = np.array([0.36, 0.26, 0.29, 0.29, 0.26, 0.26, 0.50, 0.50, 0.42, 0.42, 0.40, 0.40])  # typical adult COCO bone lengths


def _req(cond, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def rig(ep: Episode) -> dict:
    """rig.json over DEFAULT_RIG, validated (sizes in metres, tag ids unique non-negative ints). Raises ValueError."""
    r = json.loads(json.dumps(DEFAULT_RIG)); p = ep.dir / "rig.json"
    if p.exists():
        user = json.loads(p.read_text())
        _req(isinstance(user, dict), "rig.json must be a JSON object")
        r.update(user)
    b = r["board"]
    _req(isinstance(b, dict) and {"squares_x", "squares_y", "square_m", "marker_m"} <= set(b), f"rig.json board needs squares_x, squares_y, square_m, marker_m: {b}")
    _req(0.005 <= float(b["square_m"]) <= 1.0 and 0.0 < float(b["marker_m"]) < float(b["square_m"]), f"rig.json board sizes must be metres with marker_m < square_m: {b}")
    if r["head_tags"] or r["object_tags"]:  # a default tag size would silently scale every tag distance (5 cm tag read as 8 cm: x1.6)
        _req(r["tag_size_m"] is not None, "rig.json tag_size_m is required when head_tags/object_tags are set: the edge of the tag's BLACK "
                                          "border square in metres (e.g. 0.08); there is no default")
    if r["tag_size_m"] is not None:
        _req(0.005 <= float(r["tag_size_m"]) <= 1.0, f"rig.json tag_size_m must be metres (black border edge): {r['tag_size_m']}")
    for key in ("head_tags", "object_tags"):
        _req(isinstance(r[key], dict) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in r[key].values()),
             f"rig.json {key} must map names to non-negative integer tag ids: {r[key]}")
    ids = [*r["head_tags"].values(), *r["object_tags"].values()]
    _req(len(ids) == len(set(ids)), f"rig.json tag ids must be unique across head_tags/object_tags: {ids}")
    _req(float(r["calib_window_s"]) > 0 and 0 < float(r["calib_hz"]) <= 60 and int(r["detect_max_px"]) >= 320,
         "rig.json calib_window_s > 0, 0 < calib_hz <= 60 and detect_max_px >= 320 required")
    _req(isinstance(r["lens_model"], dict) and set(r["lens_model"].values()) <= set(G.LENS_MODELS), f"rig.json lens_model values must be in {G.LENS_MODELS}")
    return r


def _spec(r: dict) -> G.BoardSpec:
    b = r["board"]
    return G.BoardSpec(int(b["squares_x"]), int(b["squares_y"]), float(b["square_m"]), float(b["marker_m"]), **{k: b[k] for k in ("legacy", "dictionary") if k in b})


def _file_identity(p: Path) -> dict:
    """size and sha256 of the first + last MiB (as zed_export.py records for left.mp4)."""
    st = p.stat(); h = hashlib.sha256(str(st.st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END); h.update(f.read(1 << 20))
    return {"size": st.st_size, "sha256_head_tail": h.hexdigest()}


def _zed_dir(ep: Episode, s: Stream) -> Path | None:
    """zed/<s>/ when it holds a ZED export (pose.csv), after checking it belongs to this stream's video: the video
    identity zed_export.py records (frames, fps, left.mp4 size and head/tail hash; format 1: frames = grabs, fps) must
    match the probed stream (and its file when present). A mismatch (an export left from another take) raises."""
    d = ep.dir / "zed" / s.name
    if not (d / "pose.csv").exists():
        return None
    e = json.loads((d / "export.json").read_text()) if (d / "export.json").exists() else {}
    v = e.get("video") or {"frames": e.get("frames"), "fps": e.get("fps")}
    bad = []
    if v.get("frames") is not None and s.nb_frames is not None and int(v["frames"]) != int(s.nb_frames):
        bad.append(f"{v['frames']} video frames vs {s.nb_frames}")
    if v.get("fps") and s.fps and abs(float(v["fps"]) / s.fps - 1) > 0.005:
        bad.append(f"{v['fps']} fps vs {s.fps:.3f}")
    if v.get("frames") and v.get("fps") and s.duration_s and abs(int(v["frames"]) / float(v["fps"]) - s.duration_s) > 1.5 / float(v["fps"]):
        bad.append(f"{int(v['frames']) / float(v['fps']):.2f} s vs {s.duration_s:.2f} s")
    src = ep.dir / s.path
    if v.get("size") is not None and src.exists():
        ident = _file_identity(src)
        if ident["size"] != v["size"] or (v.get("sha256_head_tail") and ident["sha256_head_tail"] != v["sha256_head_tail"]):
            bad.append(f"{s.path} is not the export's left.mp4 (size {ident['size']} vs {v['size']} or content differs)")
    _req(not bad, f"zed/{s.name} does not belong to stream {s.name} ({'; '.join(bad)}): an export from another take? Re-export this take with "
                  f"scripts/zed_export.py (--overwrite) or remove zed/{s.name}")
    return d


def _gray(path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"cannot read frame {Path(path).name}")
    return g


def _frames(ep: Episode, s: Stream) -> tuple[list[Path], pd.DataFrame | None]:
    """Extracted frames of a usable stream and its C4 index, checked to hold exactly ep.n_frames() rows k = 0..n-1."""
    paths = P.frame_paths(ep, s)
    if not paths:
        return [], None
    idx = P.frame_index(ep, s); n = ep.n_frames()
    if len(paths) != n or len(idx) != n or not np.array_equal(idx["k"].to_numpy(), np.arange(n)):
        raise RuntimeError(f"{s.name}: {len(paths)} frames / {len(idx)} index rows, expected n_frames()={n} (re-run frames)")
    return paths, idx


# ----------------------------------------------------------------------------- small pose helpers

def _pose_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(|t_A - t_B| m, rotation angle between R_A and R_B deg)."""
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _mean_pose(Ts) -> np.ndarray:
    """Chordal mean rotation and mean translation of rigid transforms."""
    Ts = np.asarray(Ts, float).reshape(-1, 4, 4); T = np.eye(4)
    T[:3, :3] = G.average_rotations(Ts[:, :3, :3]); T[:3, 3] = Ts[:, :3, 3].mean(0)
    return T


def _robust_mean_pose(Ts, tol: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Mean of the poses within tol of the medoid -> (T, inlier mask, (max m, max deg) inlier deviation from T)."""
    Ts = np.asarray(Ts, float).reshape(-1, 4, 4)
    D = np.array([[sum(np.divide(_pose_diff(a, b), tol)) for b in Ts] for a in Ts])
    med = Ts[int(np.argmin(D.sum(1)))]
    inl = np.array([_pose_diff(med, T)[0] <= tol[0] and _pose_diff(med, T)[1] <= tol[1] for T in Ts])
    T = _mean_pose(Ts[inl]); dev = np.array([_pose_diff(T, X) for X in Ts[inl]])
    return T, inl, (float(dev[:, 0].max()), float(dev[:, 1].max()))


# ----------------------------------------------------------------------------- native frames

_LUMA8 = ("yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p", "nv12", "nv21", "gray")


def _luma(f) -> np.ndarray:
    """8-bit luma of a decoded PyAV frame: the Y plane itself for 8-bit YUV/NV12 (no colour conversion; limited-range
    values are fine for board/tag detection), else swscale's gray conversion (e.g. 10-bit video)."""
    if f.format.name in _LUMA8:
        p = f.planes[0]
        return np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, :p.width].copy()  # own the pixels
    return f.to_ndarray(format="gray")


class _Native:
    """Grayscale source-video frames at C1 stream times (s, 0 = first video frame), in DISPLAY orientation
    (perception.apply_display with Stream.rotation/flip, pixel-identical to ffmpeg's autorotate), long side downscaled to max_px. Opened with runtime.pyav_open
    (file protocol + demuxer whitelist, ep.hwaccel). Decodes forward and seeks (to the keyframe before) across gaps > 2 s;
    returns None when no source frame lies within half a frame period."""

    def __init__(self, ep: Episode, s: Stream, max_px: int):
        self.inp = RT.pyav_open(ep.dir / s.path, video_index=s.video_index, hwaccel=getattr(ep, "hwaccel", "auto"))  # safe input + probe's stream
        self.c, self.v = self.inp.container, self.inp.stream; self.v.thread_type = "AUTO"
        fps = s.fps or float(self.v.average_rate or 0) or 30.0
        self.t0, self.rot, self.half, self.max_px = float(s.video_start_s or 0.0), int(s.rotation or 0) % 360, 0.5 / fps, int(max_px)
        self.flip = bool(getattr(s, "flip", False))  # display-matrix reflection (episode.Stream.flip), see perception.apply_display
        self._it, self._look, self._last, self._cache = None, None, -math.inf, None

    def close(self) -> None:
        self.inp.close()

    def _next(self):
        if self._look is None:
            for f in self._it:
                if f.time is not None:
                    self._look = f; break
        return self._look

    def get(self, t: float) -> np.ndarray | None:
        T = float(t) + self.t0
        if not math.isfinite(T):
            return None
        if self._cache is not None and abs(self._cache[0] - T) < 1e-6:  # the index repeats t_src_s when proc_fps >= source fps
            return self._cache[1].copy()
        if self._it is None or T < self._last - self.half or T - self._last > 2.0:
            self.c.seek(max(0, int(math.floor((T - 1.0) / self.v.time_base))), stream=self.v, backward=True, any_frame=False)
            self._it, self._look = self.c.decode(self.v), None
        while (f := self._next()) is not None and f.time < T - self.half:
            self._look = None
        if f is None or f.time > T + self.half:
            return None
        self._look, self._last = None, f.time
        g = P.apply_display(_luma(f), self.rot, self.flip)  # the canonical display transform, shared with frames/probe
        h, w = g.shape; sc = self.max_px / max(h, w)
        if sc < 1.0:
            g = cv2.resize(g, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
        g = np.ascontiguousarray(g); self._cache = (T, g)
        return g.copy()


def _same_view(native: np.ndarray, extracted: np.ndarray) -> bool:
    """The native decode shows the same picture as the extracted frame (aspect within 1 %, NCC > 0.9 at 160 px)."""
    (h, w), (he, we) = native.shape, extracted.shape
    if abs((w / h) / (we / he) - 1) > 0.01:
        return False
    sz = (160, max(1, int(round(160 * he / we))))
    a = cv2.resize(native, sz, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    b = cv2.resize(extracted, sz, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    if a.std() < 2 and b.std() < 2:
        return abs(float(a.mean() - b.mean())) < 10
    return a.std() > 1e-3 and b.std() > 1e-3 and float(np.corrcoef(a, b)[0, 1]) > 0.9


def _iter_views(ep: Episode, s: Stream, ks, paths: list[Path], idx: pd.DataFrame, native: bool, max_px: int, note: list[str]):
    """Yield (k, gray, is_native) for processed frames ks (ascending): the source frame at t_src_s decoded at native
    resolution (long side <= max_px) when it matches the extracted frame, else the extracted frame. Native decoding stops
    (with a note) when it is unavailable or keeps disagreeing with the extracted frames."""
    nat = None; t_src = idx["t_src_s"].to_numpy(float); bad = 0
    if native and not np.isfinite(t_src[list(ks)] if len(ks) else t_src).any():
        note.append("frames index has no source times (frames extracted by the old code): extracted frames used"); native = False
    if native:
        try:
            nat = _Native(ep, s, max_px)
        except Exception as e:  # noqa: BLE001 - reported, extracted frames are the fallback
            note.append(f"native decode unavailable ({type(e).__name__}: {e}); extracted frames used")
    try:
        for i, k in enumerate(ks):
            ext = _gray(paths[k]); g = None
            if nat is not None:
                g = nat.get(t_src[k])
                if g is not None and not _same_view(g, ext):
                    g = None
                if g is None:
                    bad += 1
                    if bad >= 3 and bad > 0.05 * (i + 1):
                        note.append(f"native frames do not match the extracted ones ({bad}/{i + 1}); extracted frames from frame {k} on")
                        nat.close(); nat = None
            yield k, (ext if g is None else g), g is not None
    finally:
        if nat is not None:
            nat.close()


# ----------------------------------------------------------------------------- calib

def _sample_ks(ep: Episode, r: dict, n: int) -> np.ndarray:
    step = max(1, int(round(ep.proc_fps / float(r["calib_hz"]))))
    return np.arange(0, min(n, int(round(float(r["calib_window_s"]) * ep.proc_fps)) + 1), step)


def _intrinsics(ep: Episode, s: Stream, r: dict, spec: G.BoardSpec, imgs: dict[int, np.ndarray]) -> G.Intrinsics:
    """Native-resolution intrinsics for stream s: ZED factory > rig.json > board self-calibration > nominal FOV (source set)."""
    zd = _zed_dir(ep, s)
    if zd is not None:
        c = json.loads((zd / "calibration.json").read_text()); d = np.array(c["dist"], float)
        return G.Intrinsics(np.array(c["K"], float), d, int(c["width"]), int(c["height"]), "pinhole" if d.size <= 5 else "rational", "zed_factory")
    if s.name in r["intrinsics"]:
        i = G.Intrinsics.from_json(r["intrinsics"][s.name]); i.source = "rig_json"; return i
    model = r["lens_model"].get(s.name, "pinhole")
    intr = G.calibrate_intrinsics(list(imgs.values()), spec, model) if len(imgs) >= 8 else None
    if intr is not None:
        return intr
    h, w = next(iter(imgs.values())).shape
    return G.Intrinsics.nominal(w, h, float(r["hfov_deg"].get(s.name, 100.0 if s.role == "exo" else 110.0)), model="fisheye" if model == "fisheye" else "pinhole")


def _close(A: np.ndarray, B: np.ndarray, tol: tuple[float, float]) -> bool:
    d = _pose_diff(A, B)
    return d[0] <= tol[0] and d[1] <= tol[1]


def _solve_fixed(obs: dict[str, dict[int, np.ndarray]], cams: list[str]) -> tuple[dict, dict, dict, list[int], int | None]:
    """Static cameras from board views T_cam_board(k). World = board at the anchor frame (most cameras at once,
    earliest), averaged over the anchor segment (neighbouring samples in which every anchor camera sees the board
    unmoved). Cameras that never see the board in the segment are chained through frames where a registered camera sees
    it at the same time. Returns X {cam: T_world_cam}, B {k: T_world_board(k)}, how {cam: "anchor" | "chain via ..."},
    segment, anchor."""
    frames = sorted({k for c in cams for k in obs[c]})
    if not frames:
        return {}, {}, {}, [], None
    cnt = {k: sum(k in obs[c] for c in cams) for k in frames}; top = max(cnt.values())
    anchor = min(k for k in frames if cnt[k] == top); ref = {c: obs[c][anchor] for c in cams if anchor in obs[c]}

    def still(k):
        d = [_close(obs[c][k], ref[c], STATIC_TOL) for c in ref if k in obs[c]]
        return None if not d else all(d)
    i0 = frames.index(anchor); seg = [anchor]
    for step in (1, -1):
        j = i0 + step
        while 0 <= j < len(frames):
            st = still(frames[j])
            if st is False:
                break
            if st:
                seg.append(frames[j])
            j += step
    B = {k: np.eye(4) for k in seg}; X, how, changed = {}, {}, True
    while changed:
        changed = False
        for c in cams:
            ks = [k for k in obs[c] if k in B]
            if c not in X and ks:
                X[c] = _mean_pose([B[k] @ G.inv(obs[c][k]) for k in ks]); changed = True
                how[c] = "anchor" if set(ks) & set(seg) else "chain via " + ",".join(sorted({o for k in ks for o in X if o != c and k in obs[o]}))
        for k in frames:
            cs = [c for c in X if k in obs[c]]
            if k not in B and cs:
                B[k] = _mean_pose([X[c] @ obs[c][k] for c in cs]); changed = True
    for _ in range(2):  # refine each camera over the frames with a known board pose that agree with it, then those board poses
        for c in X:
            est = [B[k] @ G.inv(obs[c][k]) for k in obs[c] if k in B]; inl = [T for T in est if _close(T, X[c], REG_TOL)]
            X[c] = _mean_pose(inl or est)
        B.update({k: _mean_pose([X[c] @ obs[c][k] for c in X if k in obs[c]]) for k in B if k not in seg})
    return X, B, how, sorted(seg), anchor


def _consistency(obs: dict[str, dict[int, np.ndarray]], X: dict[str, np.ndarray], seg: list[int]) -> tuple[dict, dict]:
    """Per registered camera: residual = median (m, deg, n) between its pose and the pose implied by the OTHER cameras'
    simultaneous board views; moved = frames at which the cameras disagree and this camera's own board view changed
    (vs. its first registration view) while another camera's did not, i.e. this camera moved (the board did not)."""
    ref = {c: obs[c][next((k for k in sorted(obs[c]) if k in seg), min(obs[c]))] for c in X}
    per: dict[str, list] = {c: [] for c in X}; moved = dict.fromkeys(X, 0)
    for k in sorted({k for c in X for k in obs[c]}):
        cs = [c for c in X if k in obs[c]]
        if len(cs) < 2:
            continue
        r = {c: _pose_diff(Xc, _mean_pose([X[o] @ obs[o][k] for o in cs if o != c]) @ G.inv(obs[c][k])) for c, Xc in X.items() if c in cs}
        for c in cs:
            per[c].append(r[c])
        if any(a > REG_TOL[0] or b > REG_TOL[1] for a, b in r.values()):
            ch = [c for c in cs if not _close(obs[c][k], ref[c], STATIC_TOL)]
            if 0 < len(ch) < len(cs):
                for c in ch:
                    moved[c] += 1
    res = {c: (float(np.median([a for a, _ in d])), float(np.median([b for _, b in d])), len(d)) if d else (None, None, 0) for c, d in per.items()}
    return res, moved


def _register(obs: dict[str, dict[int, np.ndarray]], roles: dict[str, str]) -> dict:
    """Board-frame registration from per-camera board views {cam: {k: T_cam_board}} (see calib). Fixed cameras that moved
    during the window, or whose median residual exceeds REG_TOL, are dropped and the rest re-solved (a residual tie
    between two cameras that cannot be told apart drops both)."""
    fixed = sorted(c for c in obs if roles[c] == "exo"); egos = sorted(c for c in obs if roles[c] == "ego"); dropped: dict[str, str] = {}
    while True:
        X, B, how, seg, anchor = _solve_fixed(obs, [c for c in fixed if c not in dropped]); res, moved = _consistency(obs, X, seg)
        mv = {c: f"moved during calibration ({moved[c]} of {res[c][2]} simultaneous board views)" for c in X if moved[c] and moved[c] >= 0.2 * res[c][2]}
        if not mv:
            bad = {c: res[c][0] / REG_TOL[0] + res[c][1] / REG_TOL[1] for c in X if res[c][2] and (res[c][0] > REG_TOL[0] or res[c][1] > REG_TOL[1])}
            mv = {c: (f"inconsistent with the other cameras: median residual {100 * res[c][0]:.1f} cm / {res[c][1]:.1f} deg over {res[c][2]} "
                      "simultaneous board views") for c in bad if bad[c] >= max(bad.values()) - 1e-9}
        if not mv:
            break
        dropped.update(mv)
    sightings: dict[str, list[dict]] = {e: [] for e in egos}; assumed = not X
    if assumed and egos:  # no registered fixed camera: world = the board where most egos see it at once (assumed static)
        cnt: dict[int, int] = {}
        for e in egos:
            for k in obs[e]:
                cnt[k] = cnt.get(k, 0) + 1
        anchor = min(k for k in cnt if cnt[k] == max(cnt.values()))
    for e in egos:
        for k, T in sorted(obs[e].items()):
            if k in B:
                Bk, how_b = B[k], "observed"
            elif X and seg and seg[0] <= k <= seg[-1]:
                Bk, how_b = np.eye(4), "static_inferred"
            elif assumed:
                Bk, how_b = np.eye(4), "assumed_static"
            else:
                continue
            sightings[e].append({"k": int(k), "T_world_cam": Bk @ G.inv(T), "board": how_b})
    return {"X": X, "B": B, "how": how, "segment": seg, "anchor": anchor, "residual": res, "dropped": dropped, "sightings": sightings,
            "board_assumed_static": bool(assumed and egos)}


def calib(ep: Episode) -> None:
    """Intrinsics + board registration -> derived/calib/calib.json (see module docstring). Skipped (nothing written)
    when no camera registers to the board."""
    ep.set_status("calib", "running")
    if not ep.stage_ok("frames"):
        ep.set_status("calib", "skipped", "needs frames"); return
    r = rig(ep); spec = _spec(r); n = ep.n_frames(); ks_all = _sample_ks(ep, r, n)
    cams, obs, errs, roles, notes = {}, {}, {}, {}, []
    if "wall_tags" in r:
        notes.append("rig.json wall_tags is not supported and was ignored")
    for s in ep.streams:
        if not ep.usable(s):
            notes.append(f"{s.name}: skipped, offset unaligned (set a manual offset, e.g. --offset {s.name}=<seconds>)"); continue
        paths, idx = _frames(ep, s)
        if not paths:
            notes.append(f"{s.name}: no frames"); continue
        fh, fw = _gray(paths[0]).shape; note: list[str] = []
        views = list(_iter_views(ep, s, [int(k) for k in ks_all], paths, idx, max(s.width, s.height) > max(fw, fh), int(r["detect_max_px"]), note))
        nat = {k: g for k, g, is_nat in views if is_nat}
        if len(nat) >= 0.95 * len(views):  # one image size for every sample: native (a few unmatched samples dropped) ...
            imgs, det = nat, f"native ({len(views) - len(nat)} unmatched samples dropped)" if len(nat) < len(views) else "native"
        else:  # ... or the extracted frames throughout
            imgs, det = {k: _gray(paths[k]) for k, _, _ in views}, "extracted"
        dh, dw = next(iter(imgs.values())).shape
        try:  # a wrong rig.json / ZED calibration is a configuration error: fail loudly, naming the stream
            intr0 = _intrinsics(ep, s, r, spec, imgs); intr_f, intr_d = intr0.scaled_to(fw, fh), intr0.scaled_to(dw, dh)
        except ValueError as e:
            raise ValueError(f"{s.name} intrinsics: {e}") from e
        nominal = intr0.source == "nominal_fov"; gate = BOARD_MAX_ERR_PX * (3.0 if nominal else 1.0)
        poses = {}
        for k, g in imgs.items():
            bp = G.board_pose(g, intr_d, spec)
            if bp is not None and bp[1] * fw / dw <= gate:
                poses[k] = (bp[0], bp[1] * fw / dw)
        cams[s.name] = cam = {"role": s.role, "intrinsics": intr_f.to_json(), "intrinsics_source": intr0.source, "intrinsics_input": intr0.to_json(),
                              "frame_size": [fw, fh], "intrinsics_scale": fw / intr0.width, "detect_size": [dw, dh], "detect_frames": det,
                              "board_views": len(poses), "registered": False, "T_world_cam": None, "moving": s.role == "ego", "registration": {}, "notes": note}
        if nominal and not r["allow_nominal"]:
            cam["registration"]["reason"] = "nominal-FOV intrinsics (a guess): add rig.json intrinsics, or set allow_nominal: true"
        elif not poses:
            cam["registration"]["reason"] = f"board not seen (reprojection <= {gate} px) in the first {r['calib_window_s']} s"
        else:
            obs[s.name] = {k: T for k, (T, _) in poses.items()}; errs[s.name] = {k: e for k, (_, e) in poses.items()}; roles[s.name] = s.role
    reg = _register(obs, roles)
    for c, T in reg["X"].items():
        ks = sorted(obs[c]); seg_ks = [k for k in ks if k in reg["segment"]]; rm, rd, rn = reg["residual"][c]
        spread = [_pose_diff(T, reg["B"][k] @ G.inv(obs[c][k])) for k in ks if k in reg["B"]]
        cams[c].update(registered=True, T_world_cam=T, board_reproj_px=float(np.median([errs[c][k] for k in ks])), board_sample_frame=seg_ks[0] if seg_ks else ks[0])
        cams[c]["registration"] = {"method": reg["how"][c], "frames": ks, "spread_m": max(a for a, _ in spread), "spread_deg": max(b for _, b in spread),
                                   "residual_m": rm, "residual_deg": rd, "residual_frames": rn}
    for c, why in reg["dropped"].items():
        cams[c]["registration"] = {"reason": why}
    for e, sg in reg["sightings"].items():
        cams[e].update(registered=bool(sg), sightings=[{**x, "err_px": errs[e][x["k"]]} for x in sg])
        cams[e]["registration"] = {"method": "board sightings", "n": len(sg), "board": sorted({x["board"] for x in sg})} if sg else \
            {"reason": "board pose unknown at this camera's board views (no registered fixed camera saw it then)"}
    registered = [c for c, v in cams.items() if v["registered"]]
    parts = [f"{c} [{v['intrinsics_source']}, {v['registration'].get('method')}"
             + (f", residual {100 * v['registration']['residual_m']:.1f} cm/{v['registration']['residual_deg']:.1f} deg" if v["registration"].get("residual_m") is not None else "")
             + (f", {len(v.get('sightings', []))} sightings" if v.get("moving") else "") + "]" for c, v in cams.items() if v["registered"]]
    parts += [f"NOT registered {c}: {v['registration'].get('reason')}" for c, v in cams.items() if not v["registered"]]
    parts += [f"{c}: {'; '.join(v['notes'])}" for c, v in cams.items() if v["notes"]] + notes
    if reg["board_assumed_static"]:
        parts.append("no registered fixed camera: ego sightings assume the board did not move")
    if not registered:
        ep.set_status("calib", "skipped", "no camera registered to the board; no world frame. " + "; ".join(parts)); return
    rep = {"world": "board", "units": "m", "board": r["board"], "anchor_frame": reg["anchor"], "anchor_segment": reg["segment"],
           "board_assumed_static": reg["board_assumed_static"], "cameras": cams}
    RT.atomic_write_json(ep.derived / "calib" / "calib.json", rep)
    ep.set_status("calib", "done", f"world = board at frame {reg['anchor']}; " + "; ".join(parts))


def calib_report(ep: Episode) -> dict | None:
    """Parsed calib.json when the calib stage is done, else None."""
    p = ep.derived / "calib" / "calib.json"
    return json.loads(p.read_text()) if ep.stage_ok("calib") and p.exists() else None


def load_calib(ep: Episode) -> dict[str, tuple[G.Intrinsics, np.ndarray | None]]:
    """stream -> (intrinsics scaled to the extracted frames, static T_world_cam or None (unregistered or moving ego)); {}
    unless calib is done. Also reads calib.json files written by the old code (unscaled intrinsics there: check frame_size)."""
    p = ep.derived / "calib" / "calib.json"
    if not (ep.stage_ok("calib") and p.exists()):
        return {}
    d = json.loads(p.read_text())["cameras"]
    return {k: (G.Intrinsics.from_json(v["intrinsics"]), None if v.get("T_world_cam") is None else G.validate_rigid(np.array(v["T_world_cam"]), f"{k} T_world_cam"))
            for k, v in d.items()}


# ----------------------------------------------------------------------------- tags

def _tag_ids(r: dict) -> set[int]:
    return {int(v) for v in (*r["head_tags"].values(), *r["object_tags"].values())}


def tags(ep: Episode) -> None:
    """Configured AprilTags in every registered fixed view -> derived/tags/<s>.parquet (T_cam_tag, T_world_tag, both
    planar solutions, err_px in detection px, err_rad = err_px / f). Skipped without head_tags/object_tags in rig.json.
    tags_native: frames decoded from the source at native resolution (<= detect_max_px), i.e. one full decode of each
    fixed camera's video plus detection on the larger image (several x the cost of the extracted 640 px frames)."""
    ep.set_status("tags", "running"); r = rig(ep); ids = _tag_ids(r)
    if not ids:
        ep.set_status("tags", "skipped", "no head_tags/object_tags in rig.json: tag detection not needed"); return
    if not ep.stage_ok("frames"):
        ep.set_status("tags", "skipped", "needs frames"); return
    rep = calib_report(ep)
    fixed = [s for s in ep.exos() if ep.usable(s) and rep and (rep["cameras"].get(s.name) or {}).get("T_world_cam") is not None]
    if not fixed:
        ep.set_status("tags", "skipped", "no registered fixed camera"); return
    tables, counts, notes = {}, {}, []
    for s in fixed:
        cam = rep["cameras"][s.name]; intr_f = G.Intrinsics.from_json(cam["intrinsics"])
        T_wc = G.validate_rigid(np.array(cam["T_world_cam"]), f"{s.name} T_world_cam"); paths, idx = _frames(ep, s); rows = []; note: list[str] = []
        fh, fw = intr_f.height, intr_f.width; intr_d = None; native = bool(r["tags_native"]) and max(s.width, s.height) > max(fw, fh)
        for k, g, _ in _iter_views(ep, s, range(len(paths)), paths, idx, native, int(r["detect_max_px"]), note):
            if intr_d is None or (intr_d.height, intr_d.width) != g.shape:
                intr_d = intr_f.scaled_to(g.shape[1], g.shape[0])
            sc = fw / g.shape[1]
            for d in G.detect_tags(g, intr_d, float(r["tag_size_m"])):
                if d.tag_id not in ids or d.T_cam_tag is None:
                    continue
                Tw = T_wc @ d.T_cam_tag; row = {"frame": k, "tag_id": d.tag_id, "err_px": d.err_px, "err_rad": d.err_px / intr_d.K[0, 0], "det_w": g.shape[1],
                                                "cx": float(d.corners_px[:, 0].mean() * sc), "cy": float(d.corners_px[:, 1].mean() * sc), "alt_err_px": d.alt_err_px,
                                                "ambiguous": bool(d.alt_err_px is not None and d.alt_err_px < TAG_AMBIGUOUS * max(d.err_px, 1e-3))}
                row.update({f"Tc{i}{j}": float(d.T_cam_tag[i, j]) for i in range(3) for j in range(4)})
                row.update({f"Tw{i}{j}": float(Tw[i, j]) for i in range(3) for j in range(4)})
                if d.T_cam_tag_alt is not None:
                    Ta = T_wc @ d.T_cam_tag_alt; row.update({f"Ta{i}{j}": float(Ta[i, j]) for i in range(3) for j in range(4)})
                rows.append(row)
        tables[s.name] = pd.DataFrame(rows); counts[s.name] = len(rows)
        if note:
            notes.append(f"{s.name}: {'; '.join(note)}")
    total = sum(counts.values())
    if not total:
        ep.set_status("tags", "skipped", f"none of tags {sorted(ids)} seen by {[s.name for s in fixed]}. " + "; ".join(notes)); return
    for name, df in tables.items():
        RT.atomic_write_parquet(df, ep.derived / "tags" / f"{name}.parquet", metadata={"units": "m", "frame": "T_world_tag: board world <- tag; T_cam_tag: camera <- tag"})
    ep.set_status("tags", "done", f"{total} detections of tags {sorted(ids)}: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ("; " + "; ".join(notes) if notes else ""))


def _T_from_row(row, prefix) -> np.ndarray:
    T = np.eye(4)
    for i in range(3):
        for j in range(4):
            T[i, j] = row[f"{prefix}{i}{j}"]
    return T


def tag_world_poses(ep: Episode, tag_id: int, median_window: int = 1) -> dict[int, np.ndarray]:
    """frame -> T_world_tag fused over the fixed cameras that saw the tag: the largest set of cameras whose solutions agree
    (TAG_AGREE; each camera contributes its primary or, for an ambiguous detection, its alternative planar solution),
    ties broken by angular reprojection error; a lone ambiguous detection takes the solution closest in rotation to the
    previous fused pose. median_window > 1: running median of the positions over that many frames."""
    cands: dict[int, list] = {}
    if ep.stage_ok("tags"):
        for p in sorted((ep.derived / "tags").glob("*.parquet")):
            df = pd.read_parquet(p)
            if df.empty or "Tw00" not in df:
                continue
            for row in df[df.tag_id == tag_id].itertuples(index=False):
                row = row._asdict(); sols = [_T_from_row(row, "Tw")]
                if row.get("ambiguous") and "Ta00" in row and np.isfinite(row["Ta00"]):
                    sols.append(_T_from_row(row, "Ta"))
                cands.setdefault(int(row["frame"]), []).append((p.stem, float(row.get("err_rad", row["err_px"])), sols))
    out: dict[int, np.ndarray] = {}; prev = None
    for k in sorted(cands):
        c = cands[k]; best = None
        for i, (_, e_i, sols_i) in enumerate(c):
            for T_i in sols_i:
                members = [T_i] + [min(sols_j, key=lambda T: sum(np.divide(_pose_diff(T_i, T), TAG_AGREE))) for j, (_, _, sols_j) in enumerate(c) if j != i]
                members = [T for T in members if _pose_diff(T_i, T)[0] <= TAG_AGREE[0] and _pose_diff(T_i, T)[1] <= TAG_AGREE[1]]
                cont = 0.0 if prev is None else _pose_diff(prev, T_i)[1]
                key = (len(members), -(cont if len(c) == 1 else 0.0), -e_i, T_i is sols_i[0])
                if best is None or key > best[0]:
                    best = (key, members)
        prev = out[k] = _mean_pose(best[1])
    if median_window > 1 and out:
        ks = np.array(sorted(out)); P_ = np.array([out[k][:3, 3] for k in ks]); h = median_window // 2
        for i, k in enumerate(ks):
            win = (ks >= k - h) & (ks <= k + h); out[k] = out[k].copy(); out[k][:3, 3] = np.median(P_[win], axis=0)
    return out


# ----------------------------------------------------------------------------- head pose

def zed_export_info(zd: Path) -> dict:
    """Parse zed/<s>/export.json: camera-axis change to OpenCV (ZED_AXES), units, fps, video timeline. Raises ValueError on
    a missing or unknown convention (never guessed). Legacy exports (format 1, RIGHT_HANDED_Y_UP) are recognised by the
    exact string the old exporter wrote."""
    e = json.loads((zd / "export.json").read_text()); cs, units = e.get("coordinate_system"), e.get("coordinate_units")
    legacy = cs == ZED_LEGACY_CS and units is None
    if legacy:
        cs, units = "RIGHT_HANDED_Y_UP", "METER"
    _req(cs in ZED_AXES, f"zed/{zd.name}/export.json: coordinate_system {cs!r} is not one of {sorted(ZED_AXES)}; re-export with scripts/zed_export.py")
    _req(units == "METER", f"zed/{zd.name}/export.json: coordinate_units {units!r} must be METER")
    _req(e.get("depth_units", "mm") == "mm", f"zed/{zd.name}/export.json: depth_units {e.get('depth_units')!r} must be mm")
    fps = float(e.get("fps") or 0); _req(0 < fps <= 200, f"zed/{zd.name}/export.json: fps {e.get('fps')!r}")
    return {"coordinate_system": cs, "axes": ZED_AXES[cs], "fps": fps, "legacy": legacy, "timeline": e.get("video_timeline", "grab_count"),
            "depth_every": int(e.get("depth_every_n_frames") or 1)}


def zed_track(zd: Path, info: dict) -> dict:
    """pose.csv -> per grab g: T [G,4,4] = T_zedworld_cam in OpenCV camera axes (NaN when tracking_state != OK or the
    quaternion is not unit within 1e-3; renormalised otherwise), t [G] capture time (s) relative to video frame 0,
    video_frame [G] (mp4 frame showing the grab, -1 none), frame_index [G]. Format 1 wrote each grab as the next mp4 frame."""
    df = pd.read_csv(zd / "pose.csv"); need = {"frame_index", "timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "tracking_state"}
    _req(need <= set(df.columns), f"zed/{zd.name}/pose.csv lacks {sorted(need - set(df.columns))}")
    ts = df["timestamp_ns"].to_numpy(np.int64); _req(len(ts) > 0 and bool(np.all(np.diff(ts) > 0)), f"zed/{zd.name}/pose.csv timestamps must strictly increase")
    vf = df["video_frame"].to_numpy(np.int64) if "video_frame" in df else df["frame_index"].to_numpy(np.int64)
    _req(bool((vf == 0).any()), f"zed/{zd.name}/pose.csv has no grab for video frame 0")
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(float); t = df[["tx", "ty", "tz"]].to_numpy(float); qn = np.linalg.norm(q, axis=1)
    ok = np.isfinite(qn) & (np.abs(qn - 1) < 1e-3) & np.isfinite(t).all(1) & (df["tracking_state"].astype(str).str.strip().str.upper() == "OK").to_numpy()
    T = np.full((len(df), 4, 4), np.nan); T[ok] = np.eye(4)
    T[ok, :3, :3] = quaternion_xyzw_batch_to_rotations(q[ok] / qn[ok, None]); T[ok, :3, 3] = t[ok]
    T = T @ info["axes"]  # T_zedworld_cvcam = T_zedworld_zedcam @ T_zedcam_cvcam
    return {"T": T, "ok": ok, "t": (ts - ts[np.argmax(vf == 0)]) / 1e9, "video_frame": vf, "frame_index": df["frame_index"].to_numpy(np.int64)}


def _zed_grabs(idx: pd.DataFrame, trk: dict, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Processed frame k -> grab g whose image is the source frame of k (video_frame == src_frame), and whether its
    capture time is within half a ZED frame of t_src_s (so the image is on the episode timeline). -> (g [n] or -1, on_time [n])."""
    lut = {int(v): g for g, v in enumerate(trk["video_frame"]) if v >= 0}
    g = np.array([lut.get(int(j), -1) for j in idx["src_frame"].to_numpy()], np.int64)
    dt = np.where(g >= 0, np.abs(idx["t_src_s"].to_numpy(float) - trk["t"][np.maximum(g, 0)]), np.inf)
    return g, dt <= 0.5 / fps + 1e-3


def _zed_headpose(ep: Episode, s: Stream, zd: Path, cam: dict | None, n: int) -> tuple[np.ndarray, dict, str, str]:
    """ZED tracking -> board frame: T_world_zedworld is fitted to the ego's board sightings (_zed_registration: board-frame
    pose at the sighting vs the tracked pose of the same grab); T_world_cam(k) = T_world_zedworld @ T_zedworld_cam(grab
    of k) for frames whose grab is on the capture timeline with tracking OK. -> (T [n,4,4], extra npz keys, backend, detail)."""
    info = zed_export_info(zd); trk = zed_track(zd, info); _, idx = _frames(ep, s)
    g, on_time = _zed_grabs(idx, trk, info["fps"]); ok = (g >= 0) & on_time & trk["ok"][np.maximum(g, 0)]
    use = [x for x in (cam or {}).get("sightings", []) if x["k"] < n and ok[x["k"]]]
    src = [trk["T"][g[x["k"]]] for x in use]; dst = [np.array(x["T_world_cam"], float) for x in use]
    T = np.full((n, 4, 4), np.nan); extra = {"zed_grab": g.astype(np.int32), "zed_t": np.where(g >= 0, trk["t"][np.maximum(g, 0)], np.nan)}
    why = f"{int((g < 0).sum())} no grab, {int(((g >= 0) & ~on_time).sum())} off the capture timeline, {int(((g >= 0) & on_time & ~trk['ok'][np.maximum(g, 0)]).sum())} tracking not OK"
    conv = f"{info['coordinate_system']}{' legacy CFR' if info['legacy'] else ''} -> OpenCV axes"
    if not use:
        return T, extra, "zed_tracking_unregistered", (f"zed_tracking_unregistered: WARNING never registered to the board (no board sighting at a "
                                                       f"tracked frame): board-frame pose left NaN [{conv}]")
    T_reg, inl, (sm, sd) = _zed_registration(src, dst); G.validate_rigid(T_reg, f"{s.name} T_world_zedworld")
    T[ok] = np.einsum("ij,njk->nik", T_reg, trk["T"][g[ok]])
    return T, extra, "zed_tracking+board", (f"zed_tracking+board ({ok.mean():.0%} frames; registered from {int(inl.sum())}/{len(use)} board sightings, "
                                            f"spread {100 * sm:.1f} cm/{sd:.1f} deg; rejected: {why}) [{conv}]")


def _zed_registration(src: list[np.ndarray], dst: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """T_world_zedworld from matched camera poses (src: T_zedworld_cam, dst: T_world_cam at the board sightings) ->
    (T, inlier mask, (max m, max deg) inlier residual at the camera). >= 3 pairs: duet.geometry.alignment.
    fit_pose_alignment (IRLS on camera centres + orientations, 1 m orientation length scale, Huber 2 cm); inliers are
    the pairs within ZED_REG_TOL. 1-2 pairs: robust mean of the per-sighting estimates T_world_cam @ inv(T_zedworld_cam)."""
    if len(src) < 3:
        return _robust_mean_pose([d @ G.inv(s_) for s_, d in zip(src, dst)], ZED_REG_TOL)
    S, D = np.asarray(src, float), np.asarray(dst, float)
    pairs = PosePairs(S[:, :3, 3], D[:, :3, 3], S[:, :3, :3], D[:, :3, :3], FrameId("zed_world"), FrameId("board_world"), FrameId("zed_camera_opencv"),
                      DistanceUnit.METERS, Provenance("world.headpose", "ZED tracking poses vs board-frame poses at the board sightings"))
    fit = fit_pose_alignment(pairs, orientation_length_scale_m=1.0, huber_delta_m=0.02)
    dist, ang = pose_alignment_residuals(pairs, fit.transform); ang = np.degrees(ang); inl = (dist <= ZED_REG_TOL[0]) & (ang <= ZED_REG_TOL[1])
    return np.array(fit.transform.matrix, float), inl, (float(dist[inl].max()) if inl.any() else float(dist.min()), float(ang[inl].max()) if inl.any() else float(ang.min()))


def _head_tag_key(r: dict, s: Stream) -> str | None:
    return next((key for key in (s.person, s.name) if key is not None and key in r["head_tags"]), None)


def head_tag_offset(r: dict, key: str) -> np.ndarray:
    """rig T_tag_headcam[key] validated: present, rigid, |t| < MAX_T_TAG_HEADCAM_M (head camera pose in the tag frame)."""
    T = r["T_tag_headcam"].get(key)
    _req(T is not None, f"rig.json head_tags[{key!r}] is set but T_tag_headcam[{key!r}] is missing: give the head camera's pose in the tag frame "
                        "(4x4, head-camera coords -> tag coords, metres); identity is not a neutral default")
    T = G.validate_rigid(np.array(T, float), f"rig.json T_tag_headcam[{key!r}]")
    _req(np.linalg.norm(T[:3, 3]) < MAX_T_TAG_HEADCAM_M, f"rig.json T_tag_headcam[{key!r}] translation {np.linalg.norm(T[:3, 3]):.2f} m >= {MAX_T_TAG_HEADCAM_M} m: not metres?")
    return T


def headpose(ep: Episode) -> None:
    """Per usable ego: derived/headpose/<s>.npz with T_world_cam [n,4,4] (board frame, OpenCV camera axes, NaN unknown),
    valid [n], backend, and for ZED zed_grab [n] (grab index used, -1 none) / zed_t [n] (its capture time, s)."""
    ep.set_status("headpose", "running")
    if not ep.stage_ok("frames"):
        ep.set_status("headpose", "skipped", "needs frames"); return
    r = rig(ep); rep = calib_report(ep); n = ep.n_frames(); results, parts = {}, []
    keys = {s.name: _head_tag_key(r, s) for s in ep.egos()}
    offsets = {k: head_tag_offset(r, k) for k in {k for k in keys.values() if k is not None}}  # config errors fail the stage loudly
    for s in ep.egos():
        if not ep.usable(s):
            parts.append(f"{s.name}: skipped, offset unaligned (no audio sync; set a manual offset, e.g. --offset {s.name}=<seconds>)"); continue
        zd = _zed_dir(ep, s); cam = (rep or {}).get("cameras", {}).get(s.name)
        if zd is not None:
            T, extra, backend, detail = _zed_headpose(ep, s, zd, cam, n)
        elif keys[s.name] is not None:
            poses = tag_world_poses(ep, r["head_tags"][keys[s.name]]); T = np.full((n, 4, 4), np.nan); extra = {}
            for k, T_wt in poses.items():
                if k < n:
                    T[k] = T_wt @ offsets[keys[s.name]]
            backend = "head_tag"; detail = f"head_tag {r['head_tags'][keys[s.name]]} ({np.isfinite(T[:, 0, 0]).mean():.0%} frames)" + ("" if ep.stage_ok("tags") else ": tags not done")
        else:
            parts.append(f"{s.name}: no ZED export and no head tag"); continue
        results[s.name] = (T, extra, backend); parts.append(f"{s.name}: {detail}")
    if not any(np.isfinite(T[:, 0, 0]).any() for T, _, _ in results.values()):
        ep.set_status("headpose", "skipped", "no board-frame head pose. " + "; ".join(parts)); return
    for name, (T, extra, backend) in results.items():
        RT.atomic_savez(ep.derived / "headpose" / f"{name}.npz", T_world_cam=T.astype(np.float32), valid=np.isfinite(T[:, 0, 0]), backend=np.array(backend), **extra)
    ep.set_status("headpose", "done", "; ".join(parts))


# ----------------------------------------------------------------------------- world 3D: bodies

@dataclass
class _View:
    """One camera for body association: world -> camera rotation/translation per frame (NaN where the pose is unknown;
    one row for a static camera) and the undistorted normalised coordinates of its body2d keypoints."""
    name: str
    ego: bool
    intr: G.Intrinsics
    R: np.ndarray        # [n,3,3] or [1,3,3]: camera <- world rotation
    t: np.ndarray        # [n,3] or [1,3]
    C: np.ndarray        # [n,3] or [1,3]: camera centre in world
    xn: np.ndarray       # [n,4,17,2] undistorted normalised coordinates (NaN absent)
    conf: np.ndarray     # [n,4,17] confidence where usable (>= KP_CONF and undistortable), else 0
    px: np.ndarray       # [n,4,17,2] raw pixels

    def pose(self, k: int):
        i = 0 if len(self.R) == 1 else k
        return (self.R[i], self.t[i], self.C[i]) if np.isfinite(self.R[i, 0, 0]) else None


@dataclass
class _Track:
    id: int
    X: np.ndarray                                # [17,3] last known position per keypoint (NaN unknown)
    last: int                                    # last frame it was triangulated
    slots: dict = field(default_factory=dict)    # view index -> body2d slot matched most recently
    out: dict = field(default_factory=dict)      # k -> {view index: (slot, flipped)} for frames triangulated from >= 2 views
    chir: list = field(default_factory=list)     # per triangulated frame: +1 labels anatomical, -1 mirrored, 0 unknown
    head: dict = field(default_factory=dict)     # k -> head point [3] (nan when unknown)
    centre: dict = field(default_factory=dict)   # k -> mean of the triangulated keypoints [3]


def _masked_median(e: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Median over the last axis of the entries where use is True (inf where none)."""
    v = np.sort(np.where(use, e, np.inf), axis=-1); c = use.sum(-1)
    lo = np.take_along_axis(v, np.maximum((c - 1) // 2, 0)[..., None], -1)[..., 0]; hi = np.take_along_axis(v, np.maximum(c // 2, 0)[..., None], -1)[..., 0]
    return np.where(c > 0, (lo + hi) / 2, np.inf)


def _bone_scale(X: np.ndarray) -> np.ndarray:
    """Median ratio of bone lengths to BONE_M over [..., 17, 3] skeletons (NaN with < 3 measurable bones)."""
    L = np.linalg.norm(X[..., BONES[:, 0], :] - X[..., BONES[:, 1], :], axis=-1) / BONE_M; ok = np.isfinite(L)
    return np.where(ok.sum(-1) >= 3, _masked_median(np.where(ok, L, 0.0), ok), np.nan)


def _head_point(X: np.ndarray) -> np.ndarray:
    h = X[HEAD_KP]; h = h[np.isfinite(h).all(1)]
    return h.mean(0) if len(h) else np.full(3, np.nan)


def _chirality(X: np.ndarray) -> int:
    """+1 when a 3D COCO skeleton's left/right labels are anatomical (the face points along left x up, left = L - R
    shoulder, up = shoulders - hips, face = nose - ears/eyes), -1 when mirrored, 0 when undecided."""
    face = X[[3, 4]] if np.isfinite(X[[3, 4]]).all() else X[[1, 2]]; hips = X[[11, 12]]
    if not (np.isfinite(X[[0, 5, 6]]).all() and np.isfinite(face).all() and np.isfinite(hips).any()):
        return 0
    f_lab = np.cross(X[5] - X[6], (X[5] + X[6]) / 2 - np.nanmean(hips, axis=0)); f_face = X[0] - face.mean(0)
    c = float(f_lab @ f_face / (np.linalg.norm(f_lab) * np.linalg.norm(f_face) + 1e-12))
    return 1 if c > 0.3 else (-1 if c < -0.3 else 0)


def _kp(v: _View, k: int, slot: int, flip: bool, what: str = "xn") -> np.ndarray:
    """Keypoint array (xn | conf | px) of one detection, left/right swapped when flip."""
    a = getattr(v, what)[k, slot]
    return a[FLIP] if flip else a


def _dlt(R: np.ndarray, t: np.ndarray, xn: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted linear triangulation in normalised coordinates (inhomogeneous least squares via 3x3 normal equations;
    for tracking and association only, the output uses geometry.triangulate_many). R [V,3,3], t [V,3], xn [V,N,2],
    w [V,N] (0 = unused) -> X [N,3], NaN with < 2 views or (near-)parallel rays."""
    P = np.concatenate([R, t[:, :, None]], axis=2); w = np.where(np.isfinite(xn).all(-1), w, 0.0); x = np.nan_to_num(xn)
    A = np.concatenate([(x[..., 0, None] * P[:, None, 2] - P[:, None, 0]) * w[..., None], (x[..., 1, None] * P[:, None, 2] - P[:, None, 1]) * w[..., None]], 0)
    M = np.einsum("vni,vnj->nij", A[..., :3], A[..., :3]); b = -np.einsum("vni,vn->ni", A[..., :3], A[..., 3])
    ev = np.linalg.eigvalsh(M); ok = ((w > 0).sum(0) >= 2) & (ev[:, 0] > 1e-9 * np.maximum(ev[:, 2], 1e-300))
    X = np.full((xn.shape[1], 3), np.nan)
    if ok.any():
        X[ok] = np.linalg.solve(M[ok], b[ok][..., None])[..., 0]
    return X


def _lateral_residual(R: np.ndarray, t: np.ndarray, X: np.ndarray, xn: np.ndarray) -> np.ndarray:
    """|projection - observation| in normalised units x depth = lateral distance (m) at the point's depth; NaN when the
    point is closer than 0.1 m / behind the camera or not observed. X [..., 3], xn [..., 2] (broadcast)."""
    Xc = X @ R.T + t; z = Xc[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.linalg.norm(Xc[..., :2] / z[..., None] - xn, axis=-1) * z
    return np.where(z > 0.1, e, np.nan)


def _prepare_views(ep: Episode, rep: dict, headposes: dict[str, np.ndarray], n: int, notes: list[str], allow_nominal: bool = False) -> list[_View]:
    """Registered fixed cameras with body2d; ego cameras with head poses join only when fewer than 2 fixed cameras do
    (their poses are less accurate, and each ego sees only the partner), and only with real (non nominal-FOV)
    intrinsics unless rig allow_nominal."""
    views = []
    if not ep.stage_ok("body2d"):
        notes.append("bodies: body2d not done"); return views
    for s in [*ep.exos(), *ep.egos()]:
        cam = rep["cameras"].get(s.name); p = ep.derived / "body2d" / f"{s.name}.npz"
        if cam is None or not ep.usable(s) or not p.exists():
            continue
        if s.role == "exo" and cam.get("T_world_cam") is not None:
            T_wc = G.validate_rigid(np.array(cam["T_world_cam"]), f"{s.name} T_world_cam")[None]
        elif s.role == "ego" and s.name in headposes and sum(not v.ego for v in views) < 2:
            if cam.get("intrinsics_source") == "nominal_fov" and not allow_nominal:
                notes.append(f"{s.name}: not a body view (nominal-FOV intrinsics; add rig.json intrinsics or allow_nominal)"); continue
            T_wc = headposes[s.name]
        else:
            continue
        z = RT.load_npz(p); kp = z["kpts"].astype(float); intr = G.Intrinsics.from_json(cam["intrinsics"])
        if kp.shape[0] != n or kp.shape[2:] != (17, 3):
            notes.append(f"{s.name}: body2d kpts {kp.shape} do not match {n} frames x 17 keypoints (re-run body2d)"); continue
        if "img_w" in z and (int(z["img_w"]), int(z["img_h"])) != (intr.width, intr.height):
            notes.append(f"{s.name}: body2d is in {int(z['img_w'])}x{int(z['img_h'])} px but calib intrinsics are {intr.width}x{intr.height} (re-run)"); continue
        flat = kp[..., :2].reshape(-1, 2); xn = np.full(flat.shape, np.nan)
        for c0 in range(0, len(flat), 50_000):  # bounded memory (the Newton polish builds Jacobians per point)
            xn[c0:c0 + 50_000] = G.undistort_points(intr, flat[c0:c0 + 50_000])
        xn = xn.reshape(kp.shape[:-1] + (2,))
        conf = np.where((kp[..., 2] >= KP_CONF) & np.isfinite(xn).all(-1), kp[..., 2], 0.0)
        good = np.isfinite(T_wc[:, 0, 0]); R = np.full((len(T_wc), 3, 3), np.nan); t = np.full((len(T_wc), 3), np.nan)
        R[good] = np.transpose(T_wc[good, :3, :3], (0, 2, 1)); t[good] = -np.einsum("nij,nj->ni", R[good], T_wc[good, :3, 3])
        views.append(_View(s.name, s.role == "ego", intr, R, t, T_wc[:, :3, 3].copy(), xn, conf, kp[..., :2]))
    return views


def _match_tracks(views: list[_View], k: int, tracks: list[_Track], free: dict[int, list[int]], fps: float) -> dict[int, dict[int, tuple[int, bool]]]:
    """Per view, Hungarian assignment of free detections to tracks by the median lateral residual (m) of the track's
    last keypoints, gated by GATE_M + V_MAX * time since it was last triangulated; each detection is also tried with
    left/right swapped (detectors mislabel sides of people seen from behind); the slot a track had in that view halves
    its cost (slot continuity). Returns {track index: {view index: (slot, flipped)}}; matched slots leave free."""
    from scipy.optimize import linear_sum_assignment
    got: dict[int, dict[int, tuple[int, bool]]] = {}
    if not tracks:
        return got
    Xs = np.stack([tr.X for tr in tracks]); gate = np.array([GATE_M + V_MAX * (k - tr.last) / fps for tr in tracks])
    for vi, v in enumerate(views):
        pose = v.pose(k); ds = free.get(vi, [])
        if pose is None or not ds:
            continue
        R, t, Cc = pose; costs = []
        for fl in (FLIP, slice(None)):  # swapped, as labelled
            e = _lateral_residual(R, t, Xs[None], v.xn[k, ds][:, fl][:, None]); use = (v.conf[k, ds][:, fl][:, None] > 0) & np.isfinite(e)  # [D,T,17]
            costs.append(np.where(use.sum(-1) >= MIN_KP, _masked_median(e, use), np.inf))
        flipped = costs[0] < 0.8 * costs[1]; cost = np.where(flipped, costs[0], costs[1])  # swap only when clearly better
        if v.ego:  # the wearer is never a detection of their own head camera
            cost[:, np.array([np.linalg.norm(_head_point(tr.X) - Cc) < HEAD_LINK_M for tr in tracks])] = np.inf
        ok = np.isfinite(cost) & (cost <= gate[None])
        if not ok.any():
            continue
        rank = np.where(ok, cost * np.array([[0.5 if tr.slots.get(vi) == d else 1.0 for tr in tracks] for d in ds]), 1e9)
        for a, b in zip(*linear_sum_assignment(rank)):
            if ok[a, b]:
                got.setdefault(int(b), {})[vi] = (ds[a], bool(flipped[a, b]))
        taken = {m[vi][0] for m in got.values() if vi in m}; free[vi] = [d for d in ds if d not in taken]
    return got


def _seed(views: list[_View], k: int, free: dict[int, list[int]]) -> list[tuple[np.ndarray, dict[int, tuple[int, bool]]]]:
    """New people from pairs of free detections in two views: per view pair, Hungarian on median lateral reprojection
    residual + 0.3 |log body scale| over the pairs that pass SEED_GATE_M, SCALE_RANGE, >= MIN_KP keypoints in front
    of both cameras, and (ego views) a head away from that camera; the second view's detection is also tried with
    left/right swapped. -> [(X [17,3], {view: (slot, flipped)})]."""
    from scipy.optimize import linear_sum_assignment
    new = []
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            pa, pb = views[a].pose(k), views[b].pose(k); da, db = free.get(a, []), free.get(b, [])
            if pa is None or pb is None or not da or not db:
                continue
            I, J = np.meshgrid(np.arange(len(da)), np.arange(len(db)), indexing="ij"); I, J = np.tile(I.ravel(), 2), np.tile(J.ravel(), 2); npair = len(I)
            fl = np.repeat([False, True], npair // 2)  # every pair as labelled, then with view b's left/right swapped
            xa = views[a].xn[k, da][I]; xb = np.where(fl[:, None, None], views[b].xn[k, db][J][:, FLIP], views[b].xn[k, db][J])  # [P,17,2]
            cb = np.where(fl[:, None], views[b].conf[k, db][J][:, FLIP], views[b].conf[k, db][J])
            w = np.stack([views[a].conf[k, da][I], cb]); w[:, (w == 0).any(0)] = 0.0  # keypoints seen in both views
            R, t = np.stack([pa[0], pb[0]]), np.stack([pa[1], pb[1]])
            X = _dlt(R, t, np.stack([xa, xb]).reshape(2, -1, 2), w.reshape(2, -1)).reshape(npair, 17, 3)
            e = np.fmax(_lateral_residual(R[0], t[0], X, xa), _lateral_residual(R[1], t[1], X, xb))
            use = (w[0] > 0) & np.isfinite(e); X[~use] = np.nan; sc = _bone_scale(X); med = _masked_median(e, use)
            good = (use.sum(-1) >= MIN_KP) & (med <= SEED_GATE_M) & (sc >= SCALE_RANGE[0]) & (sc <= SCALE_RANGE[1])
            for vi, pose in ((a, pa), (b, pb)):
                if views[vi].ego:
                    good &= ~np.array([np.linalg.norm(_head_point(x) - pose[2]) < HEAD_LINK_M for x in X])
            if not good.any():
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                c2 = np.where(good, med + 0.3 * np.abs(np.log(sc)), np.inf).reshape(2, len(da), len(db))
            swap = c2[1] < 0.8 * c2[0]; cost = np.where(swap, c2[1], c2[0])  # swap only when clearly better
            for i, j in zip(*linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9))):
                if np.isfinite(cost[i, j]):
                    new.append((X[(npair // 2) * int(swap[i, j]) + i * len(db) + j], {a: (da[i], False), b: (db[j], bool(swap[i, j]))}))
            used_a = {m[a][0] for _, m in new if a in m}; used_b = {m[b][0] for _, m in new if b in m}
            free[a] = [d for d in da if d not in used_a]; free[b] = [d for d in db if d not in used_b]
    return new


def _triangulate_frame(views: list[_View], k: int, obs: list[dict[int, int]]) -> np.ndarray:
    """Linear triangulation of several people at frame k from their matched detections {view: slot}, one batched solve,
    with cheirality (tracking state only; the output is triangulated by geometry.triangulate_many). -> [len(obs),17,3]."""
    vs = sorted({v for m in obs for v in m}); R = np.stack([views[v].pose(k)[0] for v in vs]); t = np.stack([views[v].pose(k)[1] for v in vs])
    xn = np.full((len(vs), len(obs), 17, 2), np.nan); w = np.zeros((len(vs), len(obs), 17))
    for i, m in enumerate(obs):
        for j, v in enumerate(vs):
            if v in m:
                xn[j, i], w[j, i] = _kp(views[v], k, *m[v]), _kp(views[v], k, *m[v], what="conf")
    X = _dlt(R, t, xn.reshape(len(vs), -1, 2), w.reshape(len(vs), -1))
    z = np.einsum("vij,nj->vni", R, X)[..., 2] + t[:, None, 2]
    X[((z <= 0.1) & (w.reshape(len(vs), -1) > 0)).any(0)] = np.nan
    return X.reshape(len(obs), 17, 3)


def track_bodies(views: list[_View], n: int, fps: float) -> list[_Track]:
    """Cross-view association + 3D identity tracking. Per frame: detections (>= MIN_KP confident keypoints) are matched
    to live tracks view by view (_match_tracks); the rest seed new tracks from view pairs (_seed); every track seen in
    >= 2 views is re-triangulated and kept if its body scale stays plausible. Tracks end after MAX_GAP_S without a
    triangulation; tracks with < MIN_TRACK_FRAMES triangulated frames are dropped."""
    tracks: list[_Track] = []; max_gap = int(round(MAX_GAP_S * fps)); usable = [(v.conf > 0).sum(-1) >= MIN_KP for v in views]  # [n,4] per view
    for k in range(n):
        free = {vi: np.flatnonzero(usable[vi][k]).tolist() for vi, v in enumerate(views) if usable[vi][k].any() and v.pose(k) is not None}
        live = [tr for tr in tracks if k - tr.last <= max_gap]
        got = _match_tracks(views, k, live, free, fps)
        for X0, m in _seed(views, k, free):
            tr = _Track(len(tracks), X0, k); tracks.append(tr); live.append(tr)
            sub = {v: ds for v, ds in free.items() if v not in m}  # a third view of the new person, never the seed's own views
            got[len(live) - 1] = {**m, **_match_tracks(views, k, [tr], sub, fps).get(0, {})}; free.update(sub)
        for tr, m in zip(live, (got.get(i, {}) for i in range(len(live)))):
            tr.slots.update({v: sl for v, (sl, _) in m.items()})
        multi = [i for i, m in got.items() if len(m) >= 2]
        if not multi:
            continue
        Xs = _triangulate_frame(views, k, [got[i] for i in multi])
        for i, X in zip(multi, Xs):
            tr = live[i]
            if np.isfinite(X).all(1).sum() < MIN_KP or not (SCALE_RANGE[0] <= _bone_scale(np.where(np.isfinite(X), X, tr.X)) <= SCALE_RANGE[1]):
                continue
            tr.X = np.where(np.isfinite(X), X, tr.X); tr.last = k; tr.out[k] = dict(got[i])
            tr.head[k] = _head_point(X); tr.centre[k] = np.nanmean(X, axis=0); tr.chir.append(_chirality(X))
    return [tr for tr in tracks if len(tr.out) >= MIN_TRACK_FRAMES]


def _assign_slots(tracks: list[_Track], heads: dict[str, np.ndarray], egos: list[str]) -> tuple[dict[int, dict[int, int]], list[str], list[str]]:
    """Track -> output slot per frame. A track whose head is within HEAD_LINK_M of ego camera egos[p] in >= 3 frames and
    in >= half the frames where both are known is that camera's wearer and goes to slot p (stronger links win
    overlapping frames); only egos with a linked track reserve their slot. Unlinked tracks fill the other slots, longest
    first, never overlapping in time, preferring the slot whose previous track ended nearest to where the new one starts.
    -> ({slot: {k: track index}}, slot labels (the linked ego's name, "" when not linked), notes)."""
    slots: dict[int, dict[int, int]] = {0: {}, 1: {}}; notes = []
    cand = [(p, e) for p, e in enumerate(egos[:2]) if e in heads and np.isfinite(heads[e][:, 0]).any()]
    link = {}
    for i, tr in enumerate(tracks):
        ks = np.array(sorted(tr.head)); H = np.stack([tr.head[k] for k in ks])
        for p, e in cand:
            d = np.linalg.norm(H - heads[e][ks], axis=1); d = d[np.isfinite(d)]; votes = int((d < HEAD_LINK_M).sum())
            if votes >= 3 and votes >= 0.5 * len(d) and votes > link.get(i, (None, 0))[1]:
                link[i] = (p, votes)
    bound = {p for p, _ in link.values()}; names = [egos[p] if p in bound else "" for p in (0, 1)]
    for i in sorted(link, key=lambda i: -link[i][1]):
        for k in tracks[i].out:
            slots[link[i][0]].setdefault(k, i)
    open_ = [p for p in (0, 1) if p not in bound]
    for i in sorted((i for i in range(len(tracks)) if i not in link), key=lambda i: -len(tracks[i].out)):
        ks = set(tracks[i].out); first = min(ks); free = [p for p in open_ if not ks & set(slots[p])]
        if not free:
            notes.append(f"track {tracks[i].id} ({len(ks)} frames) left out: no free slot"); continue
        p = min(free, key=lambda p, first=first, i=i: _slot_gap(tracks, slots[p], first, tracks[i].centre[first]))
        slots[p].update({k: i for k in ks})
    if link:
        notes.append("identity from head position: " + ", ".join(f"track {tracks[i].id} -> {egos[p]}" for i, (p, _) in link.items()))
    notes += [f"{e}'s wearer not identified (head never near that camera): slot {p} unlinked" for p, e in cand if p not in bound]
    return slots, names, notes


def _slot_gap(tracks: list[_Track], slot: dict[int, int], first: int, start: np.ndarray) -> float:
    """Slot preference for a track starting at frame first at position start (lower is better): the distance (m) from
    where the slot's previous track ended when <= 1 m (the same person re-acquired), else an unused slot (1.5), else
    2 + that distance."""
    prev = max((k for k in slot if k < first), default=None)
    if prev is None:
        return 1.5
    d = float(np.linalg.norm(tracks[slot[prev]].centre[prev] - start))
    return d if d <= 1.0 else 2.0 + d


def _T_world(v: _View, k: int) -> np.ndarray:
    i = 0 if len(v.R) == 1 else k; T = np.eye(4); T[:3, :3] = v.R[i].T; T[:3, 3] = v.C[i]
    return T


def _triangulate_bodies(views: list[_View], tracks: list[_Track], slots: dict[int, dict[int, int]], n: int, px_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Output triangulation with geometry.triangulate_many (cheirality, ray angle, range, per-view outliers, reprojection
    gate 10 px at 640 px frames), batched: frames seen only by static cameras in chunks, ego-view frames one by one."""
    bodies = np.full((n, 2, 17, 3), np.nan, np.float32); err = np.full((n, 2, 17), np.nan, np.float32); tid = np.full((n, 2), -1, np.int32)
    static = [i for i, v in enumerate(views) if len(v.R) == 1]; jobs_static, jobs_ego = [], []
    for p, frames in slots.items():
        for k, i in frames.items():
            obs = tracks[i].out[k]; tid[k, p] = tracks[i].id
            (jobs_static if all(v in static for v in obs) else jobs_ego).append((k, p, obs))
    kw = {"min_conf": KP_CONF, "max_err_px": 10.0 * px_scale, "outlier_px": 6.0 * px_scale}
    mirror = {tracks[i].id: sum(tracks[i].chir) < 0 for i in {i for fr in slots.values() for i in fr.values()}}  # track labels mirrored
    if jobs_static:
        cams = [G.Camera(views[v].name, views[v].intr, _T_world(views[v], 0)) for v in static]
        for c0 in range(0, len(jobs_static), 4000):
            chunk = jobs_static[c0:c0 + 4000]; M = len(chunk) * 17
            pts = np.full((len(static), M, 2), np.nan); conf = np.zeros((len(static), M))
            for j, (k, p, obs) in enumerate(chunk):
                for ci, v in enumerate(static):
                    if v in obs:
                        pts[ci, j * 17:(j + 1) * 17] = _kp(views[v], k, *obs[v], what="px"); conf[ci, j * 17:(j + 1) * 17] = _kp(views[v], k, *obs[v], what="conf")
            X, E = G.triangulate_many(cams, pts, conf, **kw)
            for j, (k, p, _) in enumerate(chunk):
                bodies[k, p] = X[j * 17:(j + 1) * 17]; err[k, p] = E[j * 17:(j + 1) * 17]
    for k, p, obs in jobs_ego:
        vs = list(obs); cams = [G.Camera(views[v].name, views[v].intr, _T_world(views[v], k)) for v in vs]
        X, E = G.triangulate_many(cams, np.stack([_kp(views[v], k, *obs[v], what="px") for v in vs]), np.stack([_kp(views[v], k, *obs[v], what="conf") for v in vs]), **kw)
        bodies[k, p] = X; err[k, p] = E
    for p in (0, 1):  # anatomical left/right: tracks whose labels came out mirrored (majority of _chirality votes) are swapped back
        ks = np.array([k for k in range(n) if tid[k, p] >= 0 and mirror[int(tid[k, p])]], int)
        if len(ks):
            bodies[ks, p] = bodies[ks, p][:, FLIP]; err[ks, p] = err[ks, p][:, FLIP]
    return bodies, err, tid


# ----------------------------------------------------------------------------- world 3D: hands from ZED depth

def _load_depth(p: Path) -> np.ndarray:
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None or d.dtype != np.uint16 or d.ndim != 2:
        raise RuntimeError(f"{p.name}: not a 16-bit single-channel depth PNG")
    return d


def _zed_hands(ep: Episode, s: Stream, zd: Path, hp: dict, hands: dict, intr: G.Intrinsics, n: int) -> tuple[np.ndarray, str]:
    """Wearer hand landmarks in world from ZED depth: for each frame with a board-frame pose, the depth map of the SAME
    grab as the image (no depth from another instant: there is no motion compensation; frames whose grab has no exported
    depth stay NaN, so export depth for every grab) gives the 3x3 median depth (mm, Z along the optical axis) at each
    landmark; camera point = depth * undistorted ray, then T_world_cam of the frame."""
    info = zed_export_info(zd); trk = zed_track(zd, info); out = np.full((n, 2, 21, 3), np.nan, np.float32); lm = hands["lm2d"].astype(float)
    if lm.shape != (n, 2, 21, 2):
        return out, f"skipped: hands lm2d {lm.shape} != ({n}, 2, 21, 2) (re-run hands)"
    present = hands["present"].astype(bool) if "present" in hands else (np.nan_to_num(hands["score"]) > 0)  # v2 | v1
    files = {int(p.stem): p for p in (zd / "depth").glob("*.png") if p.stem.isdigit()} if (zd / "depth").is_dir() else {}
    if not files:
        return out, "no depth frames exported"
    T = hp["T_world_cam"].astype(float); zg = hp["zed_grab"].astype(int) if "zed_grab" in hp else np.full(n, -1)
    ks = np.nonzero(present.any(1) & np.isfinite(T[:, 0, 0]) & (zg >= 0))[0]
    jobs = [(int(k), files[int(trk["frame_index"][zg[k]])]) for k in ks if int(trk["frame_index"][zg[k]]) in files]
    fh, fw = intr.height, intr.width; off = np.array([(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    with ThreadPoolExecutor(4) as pool:
        for c0 in range(0, len(jobs), 64):
            chunk = jobs[c0:c0 + 64]
            for (k, _), depth in zip(chunk, pool.map(lambda job: _load_depth(job[1]), chunk)):
                dh, dw = depth.shape
                _req(abs((dw / dh) / (fw / fh) - 1) < 0.01, f"{s.name}: depth {dw}x{dh} and frames {fw}x{fh} differ in aspect")
                for h in np.nonzero(present[k])[0]:
                    uv = lm[k, h]; ok = np.isfinite(uv).all(1)
                    if not ok.any():
                        continue
                    u = np.clip(np.round(np.nan_to_num(uv[:, 0]) * dw / fw).astype(int)[:, None] + off[:, 1], 0, dw - 1)
                    v = np.clip(np.round(np.nan_to_num(uv[:, 1]) * dh / fh).astype(int)[:, None] + off[:, 0], 0, dh - 1)
                    zz = depth[v, u].astype(float); zz[zz == 0] = np.nan
                    with np.errstate(all="ignore"):
                        z = np.nanmedian(np.where(np.isfinite(zz).any(1, keepdims=True), zz, 0.0), axis=1) / 1000.0
                    xn = G.undistort_points(intr, uv); P_w = G.apply(T[k], np.c_[xn, np.ones(21)] * z[:, None])
                    P_w[~(ok & (z > 0.05) & np.isfinite(P_w).all(1))] = np.nan; out[k, h] = P_w
    return out, f"{len(jobs)}/{len(ks)} posed hand frames have their own grab's depth"


# ----------------------------------------------------------------------------- world 3D stage

def world3d(ep: Episode) -> None:
    """derived/world3d/world3d.npz:
      bodies [n,2,17,3] (m, world), body_err [n,2,17] (px), body_track [n,2] (track id, -1), body_person [2] (ego name the
      slot is linked to, "" if not linked); hands3d_<ego> [n,2,21,3] (ZED depth, slot 0 left 1 right); head_<ego> [n,3]
      (camera centre); T_world_cam_<ego> [n,4,4] (per-frame head camera pose, OpenCV axes); object_<name> [n,3];
      feat_head_dist_m, feat_min_wrist_dist_m, feat_<ego>_facing_partner_cos [n].
    Bodies need >= 2 cameras that triangulate (registered fixed cameras with body2d, or egos with a head pose)."""
    ep.set_status("world3d", "running"); r = rig(ep); rep = calib_report(ep); n = ep.n_frames(); notes = []
    if rep is None or not ep.stage_ok("frames"):
        ep.set_status("world3d", "skipped", "needs frames and calib"); return
    egos = [s for s in ep.egos() if ep.usable(s)]; headposes = {}
    if ep.stage_ok("headpose"):
        for s in egos:
            p = ep.derived / "headpose" / f"{s.name}.npz"
            if p.exists():
                hp = RT.load_npz(p); T = hp["T_world_cam"].astype(float)
                _req(T.shape == (n, 4, 4), f"headpose {s.name}: {T.shape} != ({n}, 4, 4) (re-run headpose)"); headposes[s.name] = hp
    Tw = {k: v["T_world_cam"].astype(float) for k, v in headposes.items()}
    heads = {s.name: Tw[s.name][:, :3, 3] if s.name in Tw else np.full((n, 3), np.nan) for s in egos}
    # bodies
    views = _prepare_views(ep, rep, Tw, n, notes, bool(r["allow_nominal"])); bodies = np.full((n, 2, 17, 3), np.nan, np.float32); body_err = np.full((n, 2, 17), np.nan, np.float32)
    body_track = np.full((n, 2), -1, np.int32); names = ["", ""]
    if len(views) >= 2:
        tracks = track_bodies(views, n, ep.proc_fps)
        slots, names, sn = _assign_slots(tracks, heads, [s.name for s in egos]); notes += sn
        bodies, body_err, body_track = _triangulate_bodies(views, tracks, slots, n, ep.proc_size / 640.0)
        cov = [float(np.isfinite(bodies[:, p, :, 0]).any(1).mean()) for p in (0, 1)]; wr = [float(np.isfinite(bodies[:, p, [9, 10], 0]).all(1).mean()) for p in (0, 1)]
        body_note = (f"bodies from {[v.name for v in views]}: {len(tracks)} tracks; slot coverage {cov[0]:.0%}/{cov[1]:.0%} (both wrists {wr[0]:.0%}/{wr[1]:.0%}), "
                     f"median reproj {np.nanmedian(body_err) if np.isfinite(body_err).any() else float('nan'):.1f} px")
    else:
        body_note = f"bodies: need >= 2 triangulating cameras (registered fixed cameras with body2d, or egos with head poses); have {[v.name for v in views]}"
    # hands (ZED depth), objects
    hands3d, hand_notes = {}, []
    for s in egos:
        zd = _zed_dir(ep, s); p = ep.derived / "hands" / f"{s.name}.npz"; cam = rep["cameras"].get(s.name)
        if zd is not None and s.name in headposes and ep.stage_ok("hands") and p.exists() and cam is not None:
            hands3d[s.name], msg = _zed_hands(ep, s, zd, headposes[s.name], RT.load_npz(p), G.Intrinsics.from_json(cam["intrinsics"]), n)
            hand_notes.append(f"{s.name} {msg}, {np.isfinite(hands3d[s.name][..., 0, 0]).any(1).mean():.0%} with a hand")
    objects = {}
    if ep.stage_ok("tags"):
        for name, tid in r["object_tags"].items():
            objects[name] = np.full((n, 3), np.nan, np.float32)
            for k, T in tag_world_poses(ep, int(tid)).items():
                if k < n:
                    objects[name][k] = T[:3, 3]
    # cross-person features (metres), always written (NaN when unknown) so the export schema is stable: min wrist distance
    # between the two body slots; with >= 2 usable egos also head distance and each of the first two egos' facing cosine
    wa, wb = bodies[:, 0, [9, 10]].astype(float), bodies[:, 1, [9, 10]].astype(float)
    d = np.linalg.norm(wa[:, :, None] - wb[:, None], axis=-1).reshape(n, -1)
    feats = {"min_wrist_dist_m": np.where(np.isfinite(d).any(1), np.min(np.where(np.isfinite(d), d, np.inf), axis=1), np.nan)}
    en = [s.name for s in egos]
    if len(en) >= 2:
        a, b = en[0], en[1]; feats["head_dist_m"] = np.linalg.norm(heads[a] - heads[b], axis=1)
        for me, other in ((a, b), (b, a)):
            to = heads[other] - heads[me]; fwd = Tw[me][:, :3, 2] if me in Tw else np.full((n, 3), np.nan)
            with np.errstate(all="ignore"):
                feats[f"{me}_facing_partner_cos"] = (fwd * to).sum(1) / np.linalg.norm(to, axis=1)
    if not (np.isfinite(bodies).any() or any(np.isfinite(h).any() for h in heads.values()) or any(np.isfinite(h).any() for h in hands3d.values())
            or any(np.isfinite(o).any() for o in objects.values())):
        ep.set_status("world3d", "skipped", "nothing to put in the world frame: " + body_note + ("; " + "; ".join(notes) if notes else "")); return
    RT.atomic_savez(ep.derived / "world3d" / "world3d.npz", bodies=bodies, body_err=body_err, body_track=body_track, body_person=np.array(names),
                    **{f"hands3d_{k}": v for k, v in hands3d.items()}, **{f"head_{k}": v.astype(np.float32) for k, v in heads.items()},
                    **{f"T_world_cam_{k}": v.astype(np.float32) for k, v in Tw.items()}, **{f"object_{k}": v for k, v in objects.items()},
                    **{f"feat_{k}": v.astype(np.float32) for k, v in feats.items()})
    heads_note = ", ".join(f"{k} {np.isfinite(v[:, 0]).mean():.0%}" for k, v in heads.items())
    ep.set_status("world3d", "done", "; ".join([body_note, f"heads: {heads_note or 'none'}", f"ZED hands: {', '.join(hand_notes) or 'none'}",
                                                f"objects: {', '.join(f'{k} {np.isfinite(v[:, 0]).mean():.0%}' for k, v in objects.items()) or 'none'}",
                                                f"features with values {sorted(k for k, v in feats.items() if np.isfinite(v).any())}", *notes]))
