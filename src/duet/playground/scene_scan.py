"""Once-per-station static scene reconstruction with COLMAP (optional stage).

COLMAP is the right tool for a still scene photographed from many angles and the wrong tool for hours of
video with moving people, so this stage feeds it a *sparse, static* view set: frames from the FIXED cameras
at 1 fps, with people masked out using the body2d boxes, plus the board frames. Output is a sparse point
cloud and camera poses in COLMAP's own frame, then aligned to the board world frame using the fixed
cameras' known board poses (similarity transform: rotation, translation, scale).

Runs only if the `colmap` binary is on PATH; otherwise the stage is skipped with a message. CPU-only COLMAP
(Homebrew build) is fine for a few hundred images.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .episode import Episode
from .perception import _frame_list
from .world import load_calib


def _mask_people(img, boxes):
    m = np.full(img.shape[:2], 255, np.uint8)
    for b in boxes:
        if np.isfinite(b[4]):
            x0, y0, x1, y1 = [int(v) for v in b[:4]]; pad = 20
            m[max(0, y0 - pad): y1 + pad, max(0, x0 - pad): x1 + pad] = 0
    return m


def scene_scan(ep: Episode, max_images: int = 300) -> None:
    ep.set_status("scene_scan", "running")
    if shutil.which("colmap") is None:
        ep.set_status("scene_scan", "skipped", "colmap not installed"); return
    cal = load_calib(ep); fixed = [s for s in ep.exos() if s.name in cal and cal[s.name][1] is not None]
    if not fixed:
        ep.set_status("scene_scan", "skipped", "no fixed camera registered to the board"); return
    work = Path(tempfile.mkdtemp(prefix="duet_colmap_")); imgs = work / "images"; masks = work / "masks"; imgs.mkdir(); masks.mkdir()
    step = max(1, int(ep.proc_fps)); count = 0; index = []
    for s in fixed:
        frames = _frame_list(ep, s); b2 = ep.derived / "body2d" / f"{s.name}.npz"; boxes = np.load(b2)["boxes"] if b2.exists() else None
        for k in range(0, len(frames), step):
            if count >= max_images:
                break
            img = cv2.imread(str(frames[k])); name = f"{s.name}_{k:06d}.jpg"; cv2.imwrite(str(imgs / name), img)
            cv2.imwrite(str(masks / (name + ".png")), _mask_people(img, boxes[k]) if boxes is not None else np.full(img.shape[:2], 255, np.uint8))
            index.append((name, s.name, k)); count += 1
    db = work / "db.db"; sparse = work / "sparse"; sparse.mkdir()
    def run(*args):
        r = subprocess.run(["colmap", *args], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-600:])
    # intrinsics from our calibration, shared per camera (SIMPLE_RADIAL: f, cx, cy, k)
    run("feature_extractor", "--database_path", str(db), "--image_path", str(imgs), "--ImageReader.mask_path", str(masks), "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera_per_folder", "1")
    run("exhaustive_matcher", "--database_path", str(db))
    run("mapper", "--database_path", str(db), "--image_path", str(imgs), "--output_path", str(sparse))
    models = sorted(sparse.glob("*"))
    if not models:
        ep.set_status("scene_scan", "failed", "mapper produced no model"); return
    txt = work / "txt"; txt.mkdir(); run("model_converter", "--input_path", str(models[0]), "--output_path", str(txt), "--output_type", "TXT")
    pts = []
    for line in open(txt / "points3D.txt"):
        if line.startswith("#"): continue
        v = line.split(); pts.append([float(v[1]), float(v[2]), float(v[3]), int(v[4]), int(v[5]), int(v[6])])
    pts = np.array(pts, np.float32) if pts else np.zeros((0, 6), np.float32)
    # align COLMAP frame -> board world using the fixed cameras' board poses (Umeyama on camera centres)
    cams_colmap = {}
    lines = [l for l in open(txt / "images.txt") if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        v = lines[i].split(); qw, qx, qy, qz = map(float, v[1:5]); t = np.array(list(map(float, v[5:8]))); name = v[9]
        R = _q2R(qx, qy, qz, qw); C = -R.T @ t; cams_colmap[name] = C
    src, dst = [], []
    for name, sname, k in index:
        if name in cams_colmap and cal[sname][1] is not None:
            src.append(cams_colmap[name]); dst.append(cal[sname][1][:3, 3])
    T_world_colmap, scale = None, None
    if len(src) >= 3 and len({tuple(np.round(d, 3)) for d in dst}) >= 2:
        T_world_colmap, scale = _umeyama(np.array(src), np.array(dst))
        pts[:, :3] = (scale * (pts[:, :3] @ T_world_colmap[:3, :3].T)) + T_world_colmap[:3, 3]
    out = ep.derived / "scene_scan"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "points.npz", xyz_rgb=pts, aligned=T_world_colmap is not None, scale=scale if scale else np.nan)
    json.dump({"n_images": count, "n_registered": len(cams_colmap), "n_points": int(len(pts)), "aligned_to_board": T_world_colmap is not None, "scale": scale}, open(out / "scan.json", "w"), indent=1)
    shutil.rmtree(work, ignore_errors=True)
    ep.set_status("scene_scan", "done", f"{len(pts)} points from {len(cams_colmap)}/{count} images; aligned={T_world_colmap is not None}")


def _q2R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0); S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src)); d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt; var_s = (S ** 2).sum() / len(src); scale = (sig * np.diag(Dm)).sum() / var_s
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = mu_d - scale * R @ mu_s
    return T, float(scale)
