"""Once-per-station static scene reconstruction with COLMAP (opt-in stage: runs only when named in --stages).

The registered fixed cameras' intrinsics and board poses are already known (calib), so by default ("triangulate")
COLMAP only has to find and triangulate features: one COLMAP camera per stream with that camera's own K/dist (OPENCV
family model, no intrinsic refinement), a TXT model with every image at its camera's board pose (ids read back from
the database), ONLY cross-camera image pairs matched (images of one fixed camera share a centre: zero baseline), then
`point_triangulator`, which keeps the poses fixed, so the points land in board metres without an alignment step
(checked after the run; realigned, or left unaligned, if COLMAP moved a pose). View set per camera, same budget and
even stride for every camera: a temporal-median clean plate of n_median frames with people (body2d boxes) masked,
plus extra_frames raw frames with their own people masks.

"mapper" (rig.json {"scene_scan": {"method": "mapper"}}) runs incremental SfM on the same views and can also use fixed
cameras that never saw the board; it keeps the sub-model with the most registered images and aligns it with
align_to_board(). Points are stored under board keys only when that alignment passes its gates, otherwise they
stay in COLMAP's frame (xyz_colmap).

Conventions (verified against COLMAP 3.11.1 and 4.2.0 on synthetic known-pose scenes): images.txt stores
WORLD-TO-CAMERA poses (X_cam = R(q) X_world + t, OpenCV camera axes); COLMAP puts the centre of the top-left pixel at
(0.5, 0.5), OpenCV at (0, 0), so cx, cy shift by +0.5 (0.06 px median reprojection with the shift, 0.12 px without);
Mapper.tri_ignore_two_view_tracks defaults to 1 and drops almost every track of a 2-3 camera rig (40 vs 1,407 points),
so it is set to 0; the GPU flags are SiftExtraction/SiftMatching.use_gpu in 3.x, FeatureExtraction/FeatureMatching in 4.x.

Output derived/scene_scan/: points.npz (schema 2): xyz_board [M,3] float32 metres in the board frame OR xyz_colmap
[M,3] (arbitrary COLMAP frame and scale), rgb [M,3] uint8, err_px [M] (COLMAP reprojection error), track_len [M],
aligned, T_world_colmap [4,4] + scale (X_board = T_world_colmap @ [scale * X_colmap, 1]; identity/1 when COLMAP kept
the board poses); scan.json: views and source frames per camera, pairs, registration, alignment residuals, COLMAP
version and flags, cache. Full COLMAP output of the latest run: derived/logs/scene_scan/<step>.log (outside the stage's
outputs, so it survives the cleanup after a failure). Every COLMAP call has a timeout; the workspace is a temporary
directory removed on success and on failure. Results are cached per station (hash of rig.json + camera
intrinsics + parameters + this code and geometry.py + COLMAP version/GPU build) under $PLAYGROUND_CACHE (default
<episodes_root>/_cache)/scene_scan/<key> and reused while the cached board poses agree within pose_tol_m / pose_tol_deg;
force (run.py --force) bypasses and refreshes the entry. Skipped when `colmap` is not on PATH or fewer than 2
fixed cameras are registered to the board.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import time
import warnings
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from duet.geometry.alignment import rotation_angles
from duet.geometry.rotations import quaternion_xyzw_to_rotation

from . import geometry as G
from . import perception as P
from . import runtime
from .episode import Episode

SCHEMA = 2
DEFAULTS = dict(method="triangulate", colmap="colmap", n_median=15, extra_frames=2, max_images=60, use_gpu="auto", timeout_s=1800.0,
                tri_min_angle_deg=1.5, max_reproj_px=4.0, max_rot_deg=1.0, max_centre_rms_m=0.03, pose_tol_m=0.01, pose_tol_deg=0.5, cache=True)
_CACHE_PARAMS = ("method", "n_median", "extra_frames", "max_images", "tri_min_angle_deg", "max_reproj_px")


def scene_scan(ep: Episode, force: bool = False, **overrides) -> None:
    """Opt-in stage (module docstring). force (run.py --force) bypasses the station cache and refreshes it; overrides
    (tests, CLI) take precedence over rig.json "scene_scan"."""
    ep.set_status("scene_scan", "running"); cfg = _config(ep, overrides)
    exe = shutil.which(cfg["colmap"])
    if exe is None:
        ep.set_status("scene_scan", "skipped", f"{cfg['colmap']} not installed (not on PATH)"); return
    if not (ep.stage_ok("frames") and ep.stage_ok("calib")):
        ep.set_status("scene_scan", "skipped", "needs frames and calib"); return
    cams = _cameras(ep, cfg["method"] == "triangulate", cfg["allow_nominal"])
    reg = {n: T for n, (_, T) in cams.items() if T is not None}
    if len(reg) < 2:
        ep.set_status("scene_scan", "skipped", f"needs >= 2 fixed cameras registered to the board (have {sorted(reg)})"); return
    out, logs = ep.derived / "scene_scan", ep.derived / "logs" / "scene_scan"
    shutil.rmtree(logs, ignore_errors=True)  # latest run only; outside STAGE_OUTPUTS, so it survives a failure
    colmap = _Colmap(exe, logs, float(cfg["timeout_s"]))  # runs `colmap help`: version and GPU build are part of the cache key
    gpu = int(bool(colmap.cuda and shutil.which("nvidia-smi"))) if cfg["use_gpu"] == "auto" else int(bool(cfg["use_gpu"]))
    key = _station_key(ep, cams, cfg, {"colmap": colmap.version, "cuda": colmap.cuda, "use_gpu": gpu}); entry = _cache_root(ep) / key
    if cfg["cache"] and not force and _cache_hit(entry, reg, cfg):
        out.mkdir(parents=True, exist_ok=True); shutil.copy2(entry / "points.npz", out / "points.npz")
        scan = {**json.loads((entry / "scan.json").read_text()), "cache": {"hit": True, "key": key}}
        runtime.atomic_write_json(out / "scan.json", scan)
        ep.set_status("scene_scan", "done", f"station cache hit ({key}, from episode {scan.get('episode')}): {_summary(scan)}"); return
    with tempfile.TemporaryDirectory(prefix="duet_colmap_") as tmp:  # removed on success and on any exception
        arrays, scan = _reconstruct(ep, cams, cfg, colmap, gpu, Path(tmp))
    scan.update(episode=ep.name, station_key=key, colmap={"version": colmap.version, "cuda": colmap.cuda, "steps": colmap.steps,
                                                          "dropped_options": colmap.dropped, "logs": str(logs.relative_to(ep.dir))})
    out.mkdir(parents=True, exist_ok=True); runtime.atomic_savez(out / "points.npz", **arrays); scan["cache"] = {"hit": False, "key": key, "stored": False}
    runtime.atomic_write_json(out / "scan.json", scan)
    if cfg["cache"] and scan["aligned"]:  # only board-frame results are worth reusing
        scan["cache"].update(_cache_store(entry, out, reg)); runtime.atomic_write_json(out / "scan.json", scan)
    ep.set_status("scene_scan", "done", _summary(scan))


def _summary(scan: dict) -> str:
    a = scan["alignment"]
    return (f"{scan['n_points']} points in the {scan['frame']} frame from {scan['n_registered']}/{scan['n_images']} images, method "
            f"{scan['method']}, aligned={scan['aligned']}" + ("" if scan["aligned"] else f" ({a.get('reason')})"))


def _config(ep: Episode, overrides: dict) -> dict:
    p = ep.dir / "rig.json"; rig = json.loads(p.read_text()) if p.exists() else {}
    cfg = {**DEFAULTS, **rig.get("scene_scan", {}), **{k: v for k, v in overrides.items() if v is not None}}
    bad = sorted(set(cfg) - set(DEFAULTS))
    if bad or cfg["method"] not in ("triangulate", "mapper"):
        raise ValueError(f"scene_scan: unknown option(s) {bad} or method {cfg['method']!r}")
    assert cfg["n_median"] >= 1 and cfg["extra_frames"] >= 0 and cfg["max_images"] >= 1 and cfg["timeout_s"] > 0
    return {**cfg, "allow_nominal": bool(rig.get("allow_nominal", False))}


def _cameras(ep: Episode, registered_only: bool, allow_nominal: bool = False) -> dict[str, tuple[G.Intrinsics, np.ndarray | None]]:
    """Usable, static exo streams in calib.json: name -> (frame-size intrinsics, T_world_cam board <- camera, or None).
    Unregistered cameras (mapper only) need real intrinsics: nominal FOV guesses stay out unless allow_nominal. Cameras
    calib marks "moving" have no single pose (and a median clean plate of a moving view is meaningless): left out."""
    cal = json.loads((ep.derived / "calib" / "calib.json").read_text())["cameras"]; out = {}
    for s in ep.exos():
        c = cal.get(s.name)
        if c is None or not ep.usable(s) or c.get("moving"):
            continue
        reg = bool(c.get("registered", True)) and c.get("T_world_cam") is not None
        if reg or (not registered_only and (allow_nominal or c.get("intrinsics_source") != "nominal_fov")):
            out[s.name] = (G.Intrinsics.from_json(c["intrinsics"]), G.validate_rigid(np.array(c["T_world_cam"], float), f"T_world_cam[{s.name}]") if reg else None)
    return out


# ----------------------------------------------------------------------------- reconstruction

def _reconstruct(ep: Episode, cams: dict, cfg: dict, colmap: "_Colmap", gpu: int, work: Path) -> tuple[dict, dict]:
    img_dir, mask_dir, db = work / "images", work / "masks", work / "database.db"
    views = _prepare_views(ep, cams, cfg, img_dir, mask_dir)
    for name, (intr, _) in cams.items():  # one extraction per stream: its own COLMAP camera with its own K/dist
        model, params = colmap_camera(intr); lst = work / f"list_{name}.txt"; lst.write_text("".join(v["file"] + "\n" for v in views[name]))
        colmap.run("feature_extractor", f"extract_{name}", [("database_path", db), ("image_path", img_dir), ("image_list_path", lst),
                   ("ImageReader.mask_path", mask_dir), ("ImageReader.camera_model", model), ("ImageReader.single_camera", 1),
                   ("ImageReader.camera_params", ",".join(f"{p:.12g}" for p in params)), (("SiftExtraction.use_gpu", "FeatureExtraction.use_gpu"), gpu)])
    ids = _db_images(db)
    names = {v["file"]: n for n, vs in views.items() for v in vs}
    assert set(ids) == set(names), f"COLMAP database images {sorted(ids)} != written views {sorted(names)}"
    files = sorted(names); pairs = [(a, b) for i, a in enumerate(files) for b in files[i + 1:] if names[a] != names[b]]  # cross-camera only
    (work / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs))
    colmap.run("matches_importer", "match", [("database_path", db), ("match_list_path", work / "pairs.txt"), ("match_type", "pairs"),
                                            (("SiftMatching.use_gpu", "FeatureMatching.use_gpu"), gpu)])
    matches = {}  # camera pair -> geometrically verified matches summed over its image pairs
    for (a, b), m in _db_matches(db, ids).items():
        k = "-".join(sorted((names[a], names[b]))); matches[k] = matches.get(k, 0) + m
    if not matches:  # COLMAP's point_triangulator/mapper abort (SIGABRT) on a database without matches
        raise RuntimeError(f"no verified feature matches in {len(pairs)} cross-camera image pairs: the fixed cameras' views do not overlap or lack texture")
    mapper_opts = [("Mapper.ba_refine_focal_length", 0), ("Mapper.ba_refine_principal_point", 0), ("Mapper.ba_refine_extra_params", 0),
                   ("Mapper.tri_ignore_two_view_tracks", 0), ("Mapper.tri_min_angle", cfg["tri_min_angle_deg"]), ("Mapper.filter_max_reproj_error", cfg["max_reproj_px"])]
    reg = {n: T for n, (_, T) in cams.items() if T is not None}; models = None
    if cfg["method"] == "triangulate":
        known = work / "known"; tri = work / "triangulated"; known.mkdir(); tri.mkdir()
        _write_known_model(known, cams, ids, names)
        colmap.run("point_triangulator", "triangulate", [("database_path", db), ("image_path", img_dir), ("input_path", known), ("output_path", tri),
                                                         ("clear_points", 1), ("refine_intrinsics", 0), *mapper_opts])
        model_dir = tri
    else:
        sparse = work / "sparse"; sparse.mkdir()
        colmap.run("mapper", "mapper", [("database_path", db), ("image_path", img_dir), ("output_path", sparse), *mapper_opts])
        model_dir, models = _pick_model(sparse)
        if model_dir is None:
            raise RuntimeError(f"colmap mapper produced no model (log: {colmap.log_dir / 'mapper.log'})")
    txt = work / "txt"; txt.mkdir()
    colmap.run("model_converter", "convert", [("input_path", model_dir), ("output_path", txt), ("output_type", "TXT")])
    images, pts = read_text_model(txt)
    poses = {}  # camera -> [(R_cam_colmap, C_colmap)] of its registered images
    for f, (R, t) in images.items():
        if f in names:
            poses.setdefault(names[f], []).append((R, -R.T @ t))
    if cfg["method"] == "triangulate":
        moved = _pose_change(poses, reg)
        al = {"aligned": True, "reason": "board poses kept by point_triangulator", "T_world_colmap": np.eye(4), "scale": 1.0, **moved}
        if moved["max_centre_change_m"] > 1e-4 or moved["max_rot_change_deg"] > 0.01:  # COLMAP refined poses after all: realign
            al = {**align_to_board(poses, reg, max_rot_deg=cfg["max_rot_deg"], max_centre_rms_m=cfg["max_centre_rms_m"]), **moved}
    else:
        al = align_to_board(poses, reg, max_rot_deg=cfg["max_rot_deg"], max_centre_rms_m=cfg["max_centre_rms_m"])
    xyz = pts["xyz"]; frame = "board" if al["aligned"] else "colmap"
    arrays = {"schema": np.int64(SCHEMA), "rgb": pts["rgb"], "err_px": pts["err_px"], "track_len": pts["track_len"], "aligned": np.bool_(al["aligned"]),
              "T_world_colmap": np.asarray(al["T_world_colmap"] if al["aligned"] else np.full((4, 4), np.nan)),
              "scale": np.float64(al["scale"] if al["aligned"] else np.nan), "frame": np.array(frame)}
    if al["aligned"]:
        T = al["T_world_colmap"]; arrays["xyz_board"] = (al["scale"] * xyz @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    else:
        arrays["xyz_colmap"] = xyz.astype(np.float32)  # never under board-frame keys
    scan = {"schema": SCHEMA, "method": cfg["method"], "frame": frame, "aligned": bool(al["aligned"]), "n_points": int(len(xyz)),
            "n_images": len(names), "n_registered": sum(len(v) for v in poses.values()), "n_pairs": len(pairs), "matches": matches, "use_gpu": gpu,
            "views": views, "registered_per_camera": {n: len(v) for n, v in poses.items()}, "mapper_models": models,
            "alignment": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in al.items()},
            "params": {k: cfg[k] for k in DEFAULTS if k != "colmap"}}
    return arrays, scan


def _pose_change(poses: dict, reg: dict) -> dict:
    """Largest centre (m) and rotation (deg) difference between COLMAP's output poses and the board poses we fed in."""
    dc = [np.linalg.norm(C - reg[n][:3, 3]) for n, v in poses.items() if n in reg for _, C in v]
    dr = [np.degrees(rotation_angles(R @ reg[n][:3, :3])) for n, v in poses.items() if n in reg for R, _ in v]  # R_cam_board @ R_board_cam = I
    return {"max_centre_change_m": float(max(dc, default=np.inf)), "max_rot_change_deg": float(max(dr, default=np.inf))}


# ----------------------------------------------------------------------------- views

def _even(n: int, m: int) -> np.ndarray:
    """m indices spread evenly over range(n) (bin centres), unique."""
    return np.unique(((np.arange(min(m, n)) + 0.5) * n / min(m, n)).astype(int)) if n > 0 and m > 0 else np.zeros(0, int)


def _people_mask(shape: tuple[int, int], boxes: np.ndarray | None, pad: int) -> np.ndarray:
    """uint8 mask (255 = static scene, 0 = person box grown by pad px); boxes [4,5] x0,y0,x1,y1,conf, NaN = empty slot."""
    m = np.full(shape, 255, np.uint8)
    for b in boxes if boxes is not None else ():
        if np.all(np.isfinite(b[:4])):
            x0, y0, x1, y1 = (int(round(v)) for v in b[:4]); m[max(0, y0 - pad): max(0, y1 + pad), max(0, x0 - pad): max(0, x1 + pad)] = 0
    return m


def clean_plate(imgs: list[np.ndarray], masks: list[np.ndarray], min_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Temporal median of frames from ONE fixed camera ignoring masked (person) pixels. Returns (uint8 image, mask:
    255 where >= max(1, min_frac * n) frames saw the static scene, else 0)."""
    stack = np.stack(imgs).astype(np.float32); keep = np.stack(masks) > 0
    assert stack.shape[:3] == keep.shape, "frames and masks must share one size"
    stack[~keep] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning); med = np.nanmedian(stack, axis=0)  # all-NaN pixels -> NaN
    ok = keep.sum(0) >= max(1, int(np.ceil(min_frac * len(imgs))))
    return np.nan_to_num(med, nan=0.0).round().astype(np.uint8), np.where(ok, 255, 0).astype(np.uint8)


def _prepare_views(ep: Episode, cams: dict, cfg: dict, img_dir: Path, mask_dir: Path) -> dict[str, list[dict]]:
    """Per camera: clean plate + raw frames (even stride, equal budget per camera, >= 1 image each) into
    images/<stream>/ with COLMAP masks at masks/<stream>/<file>.png."""
    per_cam = max(1, min(1 + cfg["extra_frames"], cfg["max_images"] // len(cams))); t_ref = ep.frame_times_ref(); views = {}
    for name, (intr, _) in cams.items():
        paths = P.frame_paths(ep, ep.stream(name)); n = len(paths)
        if n == 0 or n != len(t_ref):
            raise RuntimeError(f"frames/{name} has {n} frames, the common grid {len(t_ref)}")
        b2 = ep.derived / "body2d" / f"{name}.npz"
        boxes = runtime.load_npz(b2)["boxes"] if ep.stage_ok("body2d") and b2.exists() else None
        assert boxes is None or len(boxes) == n, f"body2d/{name} has {len(boxes)} frames, frames/{name} has {n}"
        (img_dir / name).mkdir(parents=True); (mask_dir / name).mkdir(parents=True); hw = (intr.height, intr.width); pad = max(4, int(0.03 * intr.width)); v = []
        ks = _even(n, cfg["n_median"])
        plate, ok = clean_plate([_imread(paths[k], hw) for k in ks], [_people_mask(hw, None if boxes is None else boxes[k], pad) for k in ks])
        cv2.imwrite(str(img_dir / name / "plate.png"), plate); cv2.imwrite(str(mask_dir / name / "plate.png.png"), ok)
        v.append({"file": f"{name}/plate.png", "kind": "clean_plate", "frames": ks.tolist(), "t_ref_s": t_ref[ks].round(4).tolist()})
        for k in _even(n, per_cam - 1):
            f = f"{name}/k{k:06d}.jpg"; _imread(paths[k], hw); shutil.copyfile(paths[k], img_dir / f)  # size-checked, copied as is
            cv2.imwrite(str(mask_dir / (f + ".png")), _people_mask(hw, None if boxes is None else boxes[k], pad))
            v.append({"file": f, "kind": "frame", "frames": [int(k)], "t_ref_s": [round(float(t_ref[k]), 4)]})
        views[name] = v
    return views


def _imread(path: Path, hw: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None or img.shape[:2] != hw:
        raise RuntimeError(f"{path}: unreadable or not {hw[1]}x{hw[0]} (the calib.json intrinsics size)")
    return img


# ----------------------------------------------------------------------------- COLMAP I/O

class _Colmap:
    """COLMAP CLI: a timeout on every call, full output in logs/<step>.log. Option names differ across 3.x releases
    (e.g. SiftExtraction.use_gpu vs FeatureExtraction.use_gpu), so an option may list candidate names; the first one
    the binary's `<command> -h` knows is used. Unknown dotted (tuning) options are dropped and reported; unknown plain
    options mean an incompatible CLI and raise."""

    def __init__(self, exe: str, log_dir: Path, timeout_s: float):
        self.exe, self.log_dir, self.timeout_s, self.known, self.dropped, self.steps = exe, log_dir, timeout_s, {}, [], []
        log_dir.mkdir(parents=True, exist_ok=True); head = self._text(["help"])
        m = re.search(r"COLMAP (\d[\w.\-]*)", head); self.version = m.group(1) if m else "unknown"
        self.cuda = bool(re.search(r"with (CUDA|GPU)", head))  # 3.x prints "with/without CUDA", 4.x "with/without GPU support"

    def _text(self, args: list[str]) -> str:
        try:
            r = subprocess.run([self.exe, *args], capture_output=True, text=True, timeout=min(60.0, self.timeout_s))
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"colmap {' '.join(args)} timed out") from None
        return r.stdout + r.stderr

    def run(self, cmd: str, step: str, opts: list[tuple[str | tuple[str, ...], object]]) -> None:
        if cmd not in self.known:
            self.known[cmd] = set(re.findall(r"--([A-Za-z][\w.]*)", self._text([cmd, "-h"])))
        argv = [self.exe, cmd]
        for names, value in opts:
            names = (names,) if isinstance(names, str) else names; name = next((n for n in names if n in self.known[cmd]), None)
            if name is None:
                if "." not in names[0]:
                    raise RuntimeError(f"colmap {self.version} {cmd} has no --{names[0]} option")
                self.dropped.append(f"{cmd} --{names[0]}"); continue
            argv += [f"--{name}", str(value)]
        log = self.log_dir / f"{step}.log"; t0 = time.time()
        with open(log, "w") as f:
            f.write(" ".join(argv) + "\n"); f.flush()
            try:
                rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=self.timeout_s).returncode  # killed on timeout
            except subprocess.TimeoutExpired:
                self.steps.append({"step": step, "seconds": round(time.time() - t0, 2), "exit": "timeout"})
                raise RuntimeError(f"colmap {cmd} timed out after {self.timeout_s:.0f} s (log: {log})") from None
        self.steps.append({"step": step, "seconds": round(time.time() - t0, 2), "exit": rc})
        if rc != 0:
            raise RuntimeError(f"colmap {cmd} exited {rc} (log: {log}): {log.read_text(errors='replace')[-800:]}")


def colmap_camera(intr: G.Intrinsics) -> tuple[str, list[float]]:
    """COLMAP camera model + params for OpenCV-convention intrinsics. COLMAP's pixel origin is the top-left CORNER, so
    cx, cy gain +0.5 px. OpenCV k1,k2,p1,p2[,k3[,k4,k5,k6]] -> OPENCV or FULL_OPENCV (same rational model);
    cv2.fisheye k1..k4 (Intrinsics.model "fisheye") -> OPENCV_FISHEYE. Thin-prism/tilt terms are not representable."""
    K, d = intr.K, np.ravel(intr.dist).astype(float); f = [K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5]
    if intr.model == "fisheye":
        assert len(d) <= 4, f"fisheye dist must be k1..k4, got {len(d)}"
        return "OPENCV_FISHEYE", f + np.pad(d, (0, 4 - len(d))).tolist()
    if len(d) > 8 and np.any(d[8:]):
        raise ValueError(f"OpenCV thin-prism/tilt distortion ({len(d)} coefficients) has no COLMAP camera model")
    d = np.pad(d[:8], (0, 8 - min(8, len(d))))
    return ("FULL_OPENCV", f + d.tolist()) if np.any(d[4:]) else ("OPENCV", f + d[:4].tolist())


def _db_images(db: Path) -> dict[str, tuple[int, int]]:
    """COLMAP database: image name -> (image_id, camera_id)."""
    with closing(sqlite3.connect(db)) as c:
        return {name: (int(i), int(cam)) for i, name, cam in c.execute("SELECT image_id, name, camera_id FROM images")}


def _db_matches(db: Path, ids: dict[str, tuple[int, int]]) -> dict[tuple[str, str], int]:
    """Geometrically verified matches per image pair (two_view_geometries.rows > 0); COLMAP's pair_id = id1 * 2147483647 + id2
    with id1 < id2."""
    by_id = {i: n for n, (i, _) in ids.items()}; out = {}
    with closing(sqlite3.connect(db)) as c:
        for pid, n in c.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
            i1, i2 = divmod(int(pid), 2147483647)
            if i1 in by_id and i2 in by_id:
                out[(by_id[i1], by_id[i2])] = int(n)
    return out


def _write_known_model(path: Path, cams: dict, ids: dict[str, tuple[int, int]], names: dict[str, str]) -> None:
    """TXT model with every database image at its camera's board pose: ids/camera ids from the database; COLMAP
    stores world-to-camera (R_cam_board, t = -R_cam_board @ C) with the same camera axes as OpenCV."""
    cam_lines, img_lines = {}, []
    for f, (iid, cid) in sorted(ids.items(), key=lambda kv: kv[1][0]):
        intr, T = cams[names[f]]; model, params = colmap_camera(intr); R = T[:3, :3].T; t = -R @ T[:3, 3]
        cam_lines[cid] = f"{cid} {model} {intr.width} {intr.height} " + " ".join(f"{p:.17g}" for p in params)
        img_lines.append(f"{iid} " + " ".join(f"{v:.17g}" for v in (*_R2q(R), *t)) + f" {cid} {f}\n\n")  # empty POINTS2D line
    (path / "cameras.txt").write_text("".join(v + "\n" for _, v in sorted(cam_lines.items())))
    (path / "images.txt").write_text("".join(img_lines)); (path / "points3D.txt").write_text("")


def read_text_model(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """COLMAP TXT model -> ({image name: (R_cam_world, t)} world-to-camera, points: xyz [M,3], rgb [M,3] uint8,
    err_px [M], track_len [M]). images.txt has two lines per image; the second (2D points) may be empty."""
    lines = [ln for ln in (path / "images.txt").read_text().splitlines() if not ln.startswith("#")]; images = {}
    for i in range(0, len(lines), 2):
        v = lines[i].split()
        if len(v) >= 10:
            images[v[9]] = (_q2R(*map(float, v[1:5])), np.array(v[5:8], float))
    rows = [ln.split() for ln in (path / "points3D.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    a = np.array([r[1:8] for r in rows], float).reshape(-1, 7)
    return images, {"xyz": a[:, :3], "rgb": a[:, 3:6].astype(np.uint8), "err_px": a[:, 6].astype(np.float32),
                    "track_len": np.array([(len(r) - 8) // 2 for r in rows], np.int32)}


def _pick_model(sparse: Path) -> tuple[Path | None, dict[str, int]]:
    """Mapper sub-models sparse/0, 1, ... -> the one with the most registered images (numeric tie-break), plus all counts
    (images.bin starts with the uint64 number of registered images)."""
    counts = {}
    for d in sparse.iterdir():
        if (d / "images.bin").exists():
            with open(d / "images.bin", "rb") as f:
                counts[d.name] = struct.unpack("<Q", f.read(8))[0]
    if not counts:
        return None, counts
    best = max(counts, key=lambda k: (counts[k], -int(k) if k.isdigit() else 0))
    return sparse / best, counts


# ----------------------------------------------------------------------------- alignment

def align_to_board(colmap_poses: dict[str, list[tuple[np.ndarray, np.ndarray]]], board: dict[str, np.ndarray], *, max_rot_deg: float = 1.0,
                   max_centre_rms_m: float = 0.03, min_baseline_m: float = 0.05, min_minor_ratio: float = 0.05) -> dict:
    """Similarity X_board = R @ (scale * X_colmap) + t from fixed cameras with known board poses.

    colmap_poses[cam] = [(R_cam_colmap, C_colmap), ...] for that camera's images: COLMAP's world-to-camera rotation and
    the camera centre in COLMAP coordinates. board[cam] = T_world_cam (board <- camera; OpenCV axes = COLMAP axes).
    G.similarity_from_camera_poses: rotation = chordal mean of R_world_cam @ R_cam_colmap over the images (one camera
    already fixes it, so the 2-camera rig works, where Umeyama on 2 centres leaves the roll about the baseline free:
    1.3-2.4 m errors), scale and translation from the centres (>= 2 cameras >= min_baseline_m apart). With >= 3
    non-collinear cameras (G.fit_similarity's min_minor_ratio) Umeyama on per-camera centres is an independent check.
    aligned = max per-image rotation residual <= max_rot_deg, centre RMS <= max_centre_rms_m and the check (when run)
    within 2 * max_rot_deg. Returns aligned, reason, T_world_colmap (rotation + translation; scale separate), scale,
    cams, rot_resid_deg, centre_rms_m, umeyama_diff_deg (None when not run)."""
    names = sorted(n for n in set(colmap_poses) & set(board) if colmap_poses[n])
    res = {"aligned": False, "cams": names, "T_world_colmap": np.full((4, 4), np.nan), "scale": np.nan, "rot_resid_deg": None, "centre_rms_m": None, "umeyama_diff_deg": None}
    if len(names) < 2:
        return {**res, "reason": f"{len(names)} registered fixed camera(s) in the model, need >= 2"}
    Cw = np.stack([board[n][:3, 3] for n in names]); base = max(np.linalg.norm(a - b) for i, a in enumerate(Cw) for b in Cw[i + 1:])
    if base < min_baseline_m:
        return {**res, "reason": f"camera centres {base:.3f} m apart (< {min_baseline_m} m): scale undetermined"}
    A = np.stack([_pose(R.T, C) for n in names for R, C in colmap_poses[n]]); B = np.stack([board[n] for n in names for _ in colmap_poses[n]])
    fit = G.similarity_from_camera_poses(A, B)
    if fit is None:
        return {**res, "reason": "COLMAP camera centres coincide or scale <= 0"}
    T, s = fit; R, t = T[:3, :3], T[:3, 3]
    rot = np.degrees(rotation_angles(np.swapaxes(B[:, :3, :3], 1, 2) @ R @ A[:, :3, :3]))  # R_world_cam^T R R_colmap_cam = I
    rms = float(np.sqrt(np.mean(np.sum((B[:, :3, 3] - (s * A[:, :3, 3] @ R.T + t)) ** 2, axis=1))))
    res.update(T_world_colmap=T, scale=s, rot_resid_deg=float(rot.max()), centre_rms_m=rms)
    Cc = np.stack([np.mean([C for _, C in colmap_poses[n]], axis=0) for n in names])
    chk = G.fit_similarity(Cc, Cw, min_minor_ratio=min_minor_ratio) if len(names) >= 3 else None
    if chk is not None:
        res["umeyama_diff_deg"] = float(np.degrees(rotation_angles(chk[0][:3, :3].T @ R)))
    why = [w for w, bad in ((f"rotation residual {rot.max():.2f} deg > {max_rot_deg}", rot.max() > max_rot_deg),
                            (f"centre RMS {rms:.3f} m > {max_centre_rms_m}", rms > max_centre_rms_m),
                            (f"Umeyama check differs by {res['umeyama_diff_deg']} deg", (res["umeyama_diff_deg"] or 0.0) > 2 * max_rot_deg)) if bad]
    return {**res, "aligned": not why, "reason": "; ".join(why) or "ok"}


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, t; return T


def _q2R(w, x, y, z):
    """COLMAP quaternion (w, x, y, z; Hamilton) -> rotation matrix via duet.geometry's xyzw conversion (no silent
    normalisation: COLMAP writes unit quaternions with 17 significant digits)."""
    return quaternion_xyzw_to_rotation([x, y, z, w], norm_tolerance=1e-6)


def _R2q(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), Hamilton as in COLMAP (Shepperd: largest component first)."""
    tr = np.trace(R); i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        w = np.sqrt(1 + tr) / 2; q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    elif i == 1:
        x = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2; q = [(R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)]
    elif i == 2:
        y = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2; q = [(R[0, 2] - R[2, 0]) / (4 * y), (R[0, 1] + R[1, 0]) / (4 * y), y, (R[1, 2] + R[2, 1]) / (4 * y)]
    else:
        z = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2; q = [(R[1, 0] - R[0, 1]) / (4 * z), (R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), z]
    q = np.array(q); return q / np.linalg.norm(q)


# ----------------------------------------------------------------------------- station cache

def _cache_root(ep: Episode) -> Path:
    return Path(os.environ.get("PLAYGROUND_CACHE") or ep.dir.parent / "_cache") / "scene_scan"


def _code_version() -> str:
    """sha256 of this module's and geometry.py's source: a code change invalidates every cached scan."""
    h = hashlib.sha256()
    for f in (Path(__file__), Path(G.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _station_key(ep: Episode, cams: dict, cfg: dict, tool: dict) -> str:
    """Station + producer identity: rig.json (minus the depth_mono section), camera intrinsics (rounded), scan
    parameters, the code (_code_version) and the COLMAP build (tool: version, CUDA build, effective use_gpu)."""
    p = ep.dir / "rig.json"; rig = {k: v for k, v in (json.loads(p.read_text()) if p.exists() else {}).items() if k != "depth_mono"}
    cam = {n: [np.round(i.K, 1).tolist(), np.round(np.ravel(i.dist), 4).tolist(), i.width, i.height, i.model] for n, (i, _) in sorted(cams.items())}
    blob = json.dumps({"schema": SCHEMA, "rig": rig, "cams": cam, "params": {k: cfg[k] for k in _CACHE_PARAMS}, "code": _code_version(), "tool": tool},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_hit(entry: Path, reg: dict, cfg: dict) -> bool:
    """A cached scan is reused only if the station's cameras are still where they were (board poses within tolerance)."""
    try:
        cached = json.loads((entry / "poses.json").read_text())
    except (OSError, ValueError):
        return False
    if sorted(cached) != sorted(reg) or not (entry / "points.npz").exists() or not (entry / "scan.json").exists():
        return False
    for n, T in reg.items():
        Tc = np.array(cached[n], float)
        if np.linalg.norm(Tc[:3, 3] - T[:3, 3]) > cfg["pose_tol_m"] or np.degrees(rotation_angles(Tc[:3, :3].T @ T[:3, :3])) > cfg["pose_tol_deg"]:
            return False
    return True


def _cache_store(entry: Path, out: Path, reg: dict) -> dict:
    """Best effort: copy points.npz + scan.json (+ the board poses) into the station cache, atomically per entry."""
    tmp = entry.parent / f".tmp-{entry.name}-{os.getpid()}"
    try:
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        shutil.copy2(out / "points.npz", tmp); shutil.copy2(out / "scan.json", tmp)
        (tmp / "poses.json").write_text(json.dumps({n: T.tolist() for n, T in reg.items()}))
        if entry.exists():
            shutil.rmtree(entry)
        os.replace(tmp, entry)
        return {"stored": True}
    except OSError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"stored": False, "error": f"{type(e).__name__}: {e}"}
