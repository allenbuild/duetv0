"""Stereo depth from a ZED left/right pair without the ZED SDK (OpenCV SGBM), writes zed/<stream>/depth/*.png.

Input, per ego stream:  <episode>/zed/<stream>/left.mp4, right.mp4, calibration.json (with a "stereo" block, see
scripts/zed_mac_record.py) and optionally export.json (its "fps" is the frame clock world3d uses).
Output:                 <episode>/zed/<stream>/depth/<kz:06d>.png  16-bit unsigned, millimetres, 0 = unknown,
                        at the resolution of left.mp4, in the ORIGINAL (unrectified) left-image pixel grid.
                        derived/stereo_depth/records_extra.parquet with per-frame coverage columns for the export.

Naming/indexing convention (must match world.py::world3d and scripts/zed_export.py):
    world3d looks up depth for processed frame k at
        kz = round((common_start_s + k / proc_fps - stream.offset_s) * export.json["fps"])
        zed/<stream>/depth/f"{kz:06d}.png"
    i.e. kz is the 0-based frame index of left.mp4 at its native fps (the SDK export names depth PNGs by the grabbed
    frame counter), and the PNG is read with cv2.IMREAD_UNCHANGED then divided by 1000 to get metres. Hand landmarks
    (detected on the proc-size frame) are scaled by depth.shape / intrinsics size, so any resolution works; we keep the
    native one. We therefore compute depth only at the kz values the processed frames map to, not for every frame.

Method: cv2.stereoRectify from the factory left/right intrinsics + (R, T)  ->  remap both eyes  ->  StereoSGBM
(3-way) on grayscale, optionally at a reduced working width  ->  WLS post-filter (opencv-contrib ximgproc) when
available  ->  Z = f_rect * B / disparity in the rectified left frame  ->  rotate points back by R1^T so Z is the
original left camera's depth  ->  resample onto the original pixel grid (orig->rect map from cv2.undistortPoints).

Skipped (status "skipped" + reason) when a stream has no left/right pair, no stereo block, or depth PNGs already exist
(the SDK export ships its own; we never overwrite). Nothing in this stage needs pose.csv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .episode import Episode, Stream
from .perception import _frame_list

DEFAULTS = {
    "min_depth_m": 0.25,   # sets numDisparities (closest matchable depth)
    "max_depth_m": 8.0,    # farther than this is written as 0 (unknown); disparity < 1 px is noise anyway
    "block_size": 5,
    "work_width": 1280,    # match at most this wide (disparity is scaled back); 720p runs native
    "wls": True,           # use ximgproc WLS filter if available
    "wls_lambda": 8000.0,
    "wls_sigma": 1.5,
    "alpha": -1,           # cv2.stereoRectify alpha; -1 = OpenCV default (no crop, no zoom)
}


# ----------------------------------------------------------------------------- calibration / rectification

@dataclass
class Rectification:
    width: int
    height: int
    map_left: tuple[np.ndarray, np.ndarray]     # rectified -> original (for cv2.remap of the left image)
    map_right: tuple[np.ndarray, np.ndarray]
    inv_left: tuple[np.ndarray, np.ndarray]     # original left pixel -> rectified left pixel (to bring depth back)
    R1: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    baseline_m: float

    @property
    def f(self) -> float:
        return float(self.P1[0, 0])


def has_stereo_block(c: dict) -> bool:
    st = c.get("stereo")
    return bool(st) and "right" in st and "R" in st and "T" in st and "K" in st["right"]


def rectification_from_calibration(c: dict, width: int, height: int, alpha: float = DEFAULTS["alpha"]) -> Rectification:
    """Build rectification maps for a video of (width, height) from a calibration.json with a stereo block.
    Intrinsics are scaled if the calibration was written for another width (same aspect assumed)."""
    st = c["stereo"]
    sc = width / float(c["width"])
    K1 = np.array(c["K"], float); K1[:2] *= sc; d1 = np.array(c["dist"][:5], float)
    K2 = np.array(st["right"]["K"], float); K2[:2] *= sc; d2 = np.array(st["right"]["dist"][:5], float)
    R = np.array(st["R"], float); T = np.array(st["T"], float).reshape(3, 1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2, (width, height), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    ml = cv2.initUndistortRectifyMap(K1, d1, R1, P1, (width, height), cv2.CV_32FC1)
    mr = cv2.initUndistortRectifyMap(K2, d2, R2, P2, (width, height), cv2.CV_32FC1)
    # original -> rectified: where does each original left pixel land after undistortion + R1 + P1
    u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
    rect = cv2.undistortPoints(pts, K1, d1, R=R1, P=P1).reshape(height, width, 2)
    inv = (np.ascontiguousarray(rect[..., 0], dtype=np.float32), np.ascontiguousarray(rect[..., 1], dtype=np.float32))
    return Rectification(width, height, ml, mr, inv, R1, P1, P2, float(np.linalg.norm(T)))


# ----------------------------------------------------------------------------- matching

def _matchers(rect: Rectification, work_w: int, p: dict):
    f_work = rect.f * work_w / rect.width
    nd = int(np.ceil(f_work * rect.baseline_m / p["min_depth_m"] / 16.0)) * 16
    nd = int(np.clip(nd, 32, 512))
    bs = int(p["block_size"])
    left = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1,
                                 preFilterCap=63, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    right = wls = None
    if p["wls"] and hasattr(cv2, "ximgproc"):
        right = cv2.ximgproc.createRightMatcher(left)
        wls = cv2.ximgproc.createDisparityWLSFilter(left); wls.setLambda(p["wls_lambda"]); wls.setSigmaColor(p["wls_sigma"])
    return left, right, wls, nd


def depth_from_pair(rect: Rectification, left_bgr: np.ndarray, right_bgr: np.ndarray, params: dict | None = None, _cache: dict | None = None) -> np.ndarray:
    """Depth in metres (float32, HxW, 0 = unknown) of the ORIGINAL left image pixels, from one left/right frame."""
    p = dict(DEFAULTS); p.update(params or {})
    gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY) if left_bgr.ndim == 3 else left_bgr
    gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY) if right_bgr.ndim == 3 else right_bgr
    rl = cv2.remap(gl, *rect.map_left, cv2.INTER_LINEAR); rr = cv2.remap(gr, *rect.map_right, cv2.INTER_LINEAR)
    W, H = rect.width, rect.height
    work_w = min(W, int(p["work_width"])); s = work_w / W; work_h = int(round(H * s))
    if s < 1.0:
        rl = cv2.resize(rl, (work_w, work_h), interpolation=cv2.INTER_AREA); rr = cv2.resize(rr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    cache = _cache if _cache is not None else {}
    key = (work_w, p["block_size"], p["min_depth_m"], p["wls"])
    if key not in cache:
        cache[key] = _matchers(rect, work_w, p)
    left_m, right_m, wls, _nd = cache[key]
    raw = left_m.compute(rl, rr)  # int16, disparity * 16
    if wls is not None:
        raw_r = right_m.compute(rr, rl)
        disp = wls.filter(raw, rl, disparity_map_right=raw_r).astype(np.float32) / 16.0
    else:
        disp = raw.astype(np.float32) / 16.0
    valid = (raw > 0) & (disp > 0)
    if s < 1.0:  # back to full rectified resolution; disparity scales with width
        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_NEAREST) / s
        valid = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(valid, rect.f * rect.baseline_m / disp, 0.0).astype(np.float32)
    # rectified-left-frame points -> original-left-frame depth (R1 rotates original -> rectified; undo it)
    cx, cy = float(rect.P1[0, 2]), float(rect.P1[1, 2])
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    X = (u - cx) * Z / rect.f; Y = (v - cy) * Z / rect.f
    r = rect.R1.T[2]  # third row of R1^T
    Zo = (r[0] * X + r[1] * Y + r[2] * Z).astype(np.float32)
    Zo[~valid] = 0.0; Zo[(Zo < p["min_depth_m"] * 0.5) | (Zo > p["max_depth_m"])] = 0.0
    # onto the original pixel grid (nearest: never blend across a hole or a depth edge)
    return cv2.remap(Zo, *rect.inv_left, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def depth_to_png(depth_m: np.ndarray) -> np.ndarray:
    """metres float -> uint16 millimetres as zed_export.py writes them (0 = unknown, clipped at 65.535 m)."""
    d = np.where(np.isfinite(depth_m), depth_m, 0.0)
    return np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16)


# ----------------------------------------------------------------------------- frame indexing (mirror of world3d)

def zed_fps(zed_dir: Path, s: Stream) -> float:
    """The frame clock world3d uses: export.json["fps"], else the probed fps of the stream."""
    ej = zed_dir / "export.json"
    if ej.exists():
        try:
            fps = float(json.load(open(ej)).get("fps", 0) or 0)
            if fps > 0:
                return fps
        except (ValueError, OSError):
            pass
    return float(s.fps) if s.fps else 30.0


def zed_frame_indices(ep: Episode, s: Stream, n: int, fps: float) -> np.ndarray:
    """kz for processed frames 0..n-1, exactly as world3d computes it (-1 where it falls before the video start)."""
    t_stream = ep.common_start_s + np.arange(n) / ep.proc_fps - s.offset_s
    kz = np.array([int(round(float(t) * fps)) for t in t_stream])
    kz[kz < 0] = -1
    return kz


# ----------------------------------------------------------------------------- the stage

def _open(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    return cap


def compute_stream(zed_dir: Path, calib: dict, kz: np.ndarray, out_dir: Path, params: dict | None = None, log=None) -> pd.DataFrame:
    """Compute and write depth PNGs for the requested left.mp4 frame indices. Returns one row per requested index
    (in input order) with zed_frame, valid_frac, median_depth_m (NaN rows where the frame was not in the video)."""
    p = dict(DEFAULTS); p.update(params or {})
    capL, capR = _open(zed_dir / "left.mp4"), _open(zed_dir / "right.mp4")
    W, H = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect = rectification_from_calibration(calib, W, H, p["alpha"])
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = sorted({int(k) for k in kz if k >= 0}); results: dict[int, tuple[float, float]] = {}
    cache: dict = {}; k = 0; wi = 0
    while wi < len(wanted):
        okL, fl = capL.read(); okR, fr = capR.read()
        if not (okL and okR):
            break
        if k == wanted[wi]:
            d = depth_from_pair(rect, fl, fr, p, cache)
            cv2.imwrite(str(out_dir / f"{k:06d}.png"), depth_to_png(d))
            good = d > 0; results[k] = (float(good.mean()), float(np.median(d[good])) if good.any() else float("nan"))
            wi += 1
            if log and len(results) % 50 == 0:
                log(f"    stereo_depth: {len(results)}/{len(wanted)} frames")
        k += 1
    capL.release(); capR.release()
    rows = [{"zed_frame": int(z), "valid_frac": results.get(int(z), (np.nan, np.nan))[0], "median_depth_m": results.get(int(z), (np.nan, np.nan))[1]} for z in kz]
    return pd.DataFrame(rows)


def stereo_depth(ep: Episode, params: dict | None = None) -> None:
    ep.set_status("stereo_depth", "running")
    done, skipped, extra = [], [], {}
    for s in ep.egos():
        zd = ep.dir / "zed" / s.name
        if not ((zd / "left.mp4").exists() and (zd / "right.mp4").exists() and (zd / "calibration.json").exists()):
            skipped.append(f"{s.name}: no left/right stereo pair under zed/{s.name}"); continue
        calib = json.load(open(zd / "calibration.json"))
        if not has_stereo_block(calib):
            skipped.append(f"{s.name}: calibration.json has no stereo block (right K/dist, R, T)"); continue
        depth_dir = zd / "depth"
        if depth_dir.exists() and any(depth_dir.glob("*.png")):
            skipped.append(f"{s.name}: depth PNGs already present (SDK export or earlier run); not recomputed"); continue
        frames = _frame_list(ep, s); n = len(frames)
        if n == 0:
            skipped.append(f"{s.name}: no extracted frames (run the frames stage first)"); continue
        fps = zed_fps(zd, s); kz = zed_frame_indices(ep, s, n, fps)
        df = compute_stream(zd, calib, kz, depth_dir, params)
        written = int(df.valid_frac.notna().sum())
        for col in df.columns:
            extra[f"{s.name}_{col}"] = df[col].to_numpy()
        done.append(f"{s.name}: {written}/{n} frames, valid {np.nanmean(df.valid_frac) if written else 0:.0%}, "
                    f"{'WLS' if DEFAULTS['wls'] and hasattr(cv2, 'ximgproc') else 'SGBM'} at fps clock {fps:g} -> zed/{s.name}/depth/")
    if extra:
        out = ep.derived / "stereo_depth"; out.mkdir(exist_ok=True)
        pd.DataFrame(extra).to_parquet(out / "records_extra.parquet", index=False)
    detail = "; ".join(done + skipped)
    ep.set_status("stereo_depth", "done" if done else "skipped", detail or "no ego streams")
