"""stereo_depth stage on a synthetic stereo pair: two pinhole cameras 63 mm apart look at two textured planes at
known depths (0.8 m and 1.5 m). Checks depth accuracy per plane, the PNG naming/indexing convention world3d
relies on, the skip conditions, and the export side table."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import stereo_depth as SD, zed_bundle  # noqa: E402
from duet.playground.align import align  # noqa: E402
from duet.playground.episode import Episode, probe  # noqa: E402
from duet.playground.perception import extract_frames  # noqa: E402

W, H, F = 1280, 720, 700.0
K = [[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]]
B = 0.063
NEAR_Z, FAR_Z = 0.8, 1.5


def texture(rng, shape):
    """Grey texture with fine and coarse structure so SGBM has something to match everywhere."""
    fine = cv2.GaussianBlur(rng.random(shape, dtype=np.float32), (0, 0), 1.2)
    coarse = cv2.GaussianBlur(rng.random(shape, dtype=np.float32), (0, 0), 12)
    t = 0.6 * fine + 0.4 * coarse; t = (t - t.min()) / (t.max() - t.min() + 1e-9)
    return (30 + 195 * t).astype(np.uint8)


def render_plane(tex, ppm, origin, cam_x, K, size=(W, H)):
    """Fronto-parallel textured plane with its top-left corner at `origin` (x, y, z in LEFT-camera metres), seen by a
    camera displaced by cam_x along +x (same orientation). Returns (image, mask)."""
    h, w = tex.shape; x0, y0, z = origin
    corners = np.array([[x0, y0, z], [x0 + w / ppm, y0, z], [x0 + w / ppm, y0 + h / ppm, z], [x0, y0 + h / ppm, z]]) - np.array([cam_x, 0, 0])
    Km = np.asarray(K, float); proj = (Km @ corners.T).T; dst = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(tex, Hm, size, flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hm, size, flags=cv2.INTER_NEAREST, borderValue=0) > 127
    return img, mask


def render_stereo(seed=0, near_z=NEAR_Z, far_z=FAR_Z, baseline=B):
    """(left, right, gt_depth_left, near_mask_left). Far plane fills the whole view; near plane is a 0.6 x 0.45 m panel."""
    rng = np.random.default_rng(seed)
    far = texture(rng, (1440, 2400)); near = texture(rng, (450, 600))
    out = []
    for cam_x in (0.0, baseline):
        img, _ = render_plane(far, 800, (-1.5, -0.9, far_z), cam_x, K)
        nimg, nm = render_plane(near, 1000, (-0.45, -0.25, near_z), cam_x, K)
        img = img.copy(); img[nm] = nimg[nm]; out.append(img)
        if cam_x == 0.0:
            gt = np.full((H, W), far_z, np.float32); gt[nm] = near_z; near_mask = nm
    return out[0], out[1], gt, near_mask


def synthetic_calib(width=W, height=H, fps=30.0):
    return {"K": K, "dist": [0.0] * 5, "width": width, "height": height, "fps": fps, "serial": 0, "model": "synthetic", "baseline_m": B, "source": "test",
            "stereo": {"right": {"K": K, "dist": [0.0] * 5}, "R": np.eye(3).tolist(), "T": [-B, 0.0, 0.0], "units": "m"}}


def plane_errors(depth, gt, near_mask):
    """{plane: (median relative error, coverage)} over pixels with a depth value."""
    out = {}
    for name, m, z in (("near", near_mask, NEAR_Z), ("far", ~near_mask, FAR_Z)):
        good = m & (depth > 0)
        out[name] = (float(np.median(np.abs(depth[good] - z) / z)) if good.any() else np.inf, float(good.sum() / m.sum()))
    return out


def write_video(frames, path: Path, fps=30):
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "fast", "-crf", "15", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close(); assert ff.wait() == 0


def write_bundle(d: Path, n_frames=30, fps=30, seed=0):
    """A uvc_mac-style bundle (left.mp4, right.mp4, calibration.json with stereo block, export.json) of a static scene."""
    d.mkdir(parents=True, exist_ok=True)
    left, right, gt, near = render_stereo(seed)
    write_video([left] * n_frames, d / "left.mp4", fps); write_video([right] * n_frames, d / "right.mp4", fps)
    json.dump(synthetic_calib(fps=fps), open(d / "calibration.json", "w"))
    json.dump({"source": "uvc_mac", "fps": fps, "frames": n_frames, "depth_mode": "none"}, open(d / "export.json", "w"))
    return gt, near


# ----------------------------------------------------------------------------- tests

@pytest.mark.parametrize("wls", [True, False])
def test_depth_from_pair_two_planes(wls):
    left, right, gt, near = render_stereo()
    rect = SD.rectification_from_calibration(synthetic_calib(), W, H)
    depth = SD.depth_from_pair(rect, left, right, {"wls": wls})
    assert depth.shape == (H, W) and depth.dtype == np.float32
    errs = plane_errors(depth, gt, near)
    for name, (med, cov) in errs.items():
        assert med < 0.03, (name, errs)
        assert cov > 0.7, (name, errs)


def distort(img, K, dist):
    """Apply lens distortion to an ideal pinhole image (each distorted pixel samples its ideal source pixel)."""
    h, w = img.shape[:2]; u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    ideal = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), np.asarray(K, float), np.asarray(dist, float), P=np.asarray(K, float)).reshape(h, w, 2)
    return cv2.remap(img, ideal[..., 0].astype(np.float32), ideal[..., 1].astype(np.float32), cv2.INTER_NEAREST, borderValue=0)


def render_rotated_distorted_rig(seed=1, rod=(0.00933, 0.01167, -0.00142)):
    """Like render_stereo, but the right eye is rotated by Rodrigues(rod) (~0.7 deg, a few times a factory value) and both
    eyes carry ZED-like distortion, so the rectification and the back-mapping to the original grid are exercised.
    Returns (left, right, gt_depth_left, near_mask_left, calibration dict)."""
    rng = np.random.default_rng(seed); far = texture(rng, (1440, 2400)); near = texture(rng, (450, 600))
    R, _ = cv2.Rodrigues(np.array(rod)); T_r = -R @ np.array([B, 0, 0])  # X_r = R X_l + T_r, right centre at (+B, 0, 0)
    dist_l = [-0.0419, 0.0127, 0.0001, -0.0002, -0.003]; dist_r = [-0.0402, 0.0110, 0.0002, -0.0001, -0.0028]
    Km = np.asarray(K, float)

    def view(Rc, tc):
        img = np.zeros((H, W), np.uint8); gt = np.zeros((H, W), np.float32); nm = None
        for tex, ppm, origin, z in ((far, 800, (-1.5, -0.9, FAR_Z), FAR_Z), (near, 1000, (-0.45, -0.25, NEAR_Z), NEAR_Z)):
            h, w = tex.shape; x0, y0, z0 = origin
            corners = np.array([[x0, y0, z0], [x0 + w / ppm, y0, z0], [x0 + w / ppm, y0 + h / ppm, z0], [x0, y0 + h / ppm, z0]])
            cc = (Rc @ corners.T).T + tc; proj = (Km @ cc.T).T; dst = (proj[:, :2] / proj[:, 2:3]).astype(np.float32)
            Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32), dst)
            wimg = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hm, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 127
            img[m] = wimg[m]; gt[m] = z; nm = m
        return img, gt, nm
    left, gt, near_mask = view(np.eye(3), np.zeros(3)); right, _, _ = view(R, T_r)
    calib = synthetic_calib(); calib["dist"] = dist_l; calib["stereo"] = {"right": {"K": K, "dist": dist_r}, "R": R.tolist(), "T": T_r.tolist(), "units": "m"}
    return distort(left, Km, dist_l), distort(right, Km, dist_r), distort(gt, Km, dist_l), distort(near_mask.astype(np.uint8), Km, dist_l) > 0, calib


def test_depth_rotated_distorted_rig():
    left, right, gt, near, calib = render_rotated_distorted_rig()
    rect = SD.rectification_from_calibration(calib, W, H)
    assert np.abs(rect.R1 - np.eye(3)).max() > 1e-3  # the rectification actually rotates
    depth = SD.depth_from_pair(rect, left, right)
    errs = {}
    for name, m, z in (("near", near, NEAR_Z), ("far", ~near, FAR_Z)):
        good = m & (depth > 0) & (gt > 0); errs[name] = (float(np.median(np.abs(depth[good] - gt[good]) / gt[good])), float(good.sum() / m.sum()))
    for name, (med, cov) in errs.items():
        assert med < 0.03 and cov > 0.7, (name, errs)


def test_depth_png_units_and_indexing_helpers(tmp_path):
    d = np.array([[0.0, 0.8, 70.0, np.nan]], np.float32)
    png = SD.depth_to_png(d); assert png.dtype == np.uint16 and png.tolist() == [[0, 800, 65535, 0]]
    ep = Episode(name="x", root=str(tmp_path), proc_fps=10.0, common_start_s=1.0)
    from duet.playground.episode import Stream
    s = Stream("ego", "ego", "streams/ego.mp4", offset_s=-0.5, fps=30.0)
    kz = SD.zed_frame_indices(ep, s, 3, 30.0)
    # world3d: round((common_start + k/proc_fps - offset) * fps) = round((1 + 0.1k + 0.5) * 30)
    assert kz.tolist() == [45, 48, 51]
    assert SD.zed_fps(tmp_path, s) == 30.0
    json.dump({"fps": 15}, open(tmp_path / "export.json", "w")); assert SD.zed_fps(tmp_path, s) == 15.0


def test_stage_end_to_end(tmp_path):
    bundle = tmp_path / "bundle"; gt, near = write_bundle(bundle, n_frames=30)
    assert zed_bundle.is_bundle(bundle)
    root = tmp_path / "episodes"; left = zed_bundle.ingest(bundle, root / "ep", "leader")
    ep = Episode.create(root, "ep", [("leader", "ego", left, "p1"), ("other", "ego", bundle / "left.mp4", "p2")]); ep.proc_fps = 5.0; ep.save()
    probe(ep); align(ep); extract_frames(ep)
    n = len(sorted((ep.derived / "frames" / "leader").glob("*.jpg"))); assert n >= 4
    SD.stereo_depth(ep)
    st = ep.status["stereo_depth"]; assert st["state"] == "done", st
    assert "other: no left/right stereo pair" in st["detail"]
    depth_dir = ep.dir / "zed" / "leader" / "depth"; s = ep.stream("leader")
    fps = json.load(open(ep.dir / "zed" / "leader" / "export.json"))["fps"]
    for k in range(n):  # exactly what world3d will look for
        kz = int(round((ep.common_start_s + k / ep.proc_fps - s.offset_s) * fps))
        assert (depth_dir / f"{kz:06d}.png").exists(), (k, kz, sorted(p.name for p in depth_dir.iterdir()))
    names = sorted(p.name for p in depth_dir.glob("*.png"))
    assert names[:3] == ["000000.png", "000006.png", "000012.png"], names  # 5 fps processed on a 30 fps video
    im = cv2.imread(str(depth_dir / "000006.png"), cv2.IMREAD_UNCHANGED)
    assert im.dtype == np.uint16 and im.shape == (H, W)
    depth = im.astype(np.float32) / 1000.0  # world3d's read
    errs = plane_errors(depth, gt, near)
    for name, (med, cov) in errs.items():
        assert med < 0.03 and cov > 0.7, (name, errs)
    ex = ep.derived / "stereo_depth" / "records_extra.parquet"; assert ex.exists()
    import pandas as pd
    df = pd.read_parquet(ex); assert len(df) == n and df["leader_zed_frame"].tolist()[:2] == [0, 6] and (df["leader_valid_frac"] > 0.7).all()
    # second run: depth exists -> skipped, nothing recomputed
    mtimes = {p.name: p.stat().st_mtime_ns for p in depth_dir.glob("*.png")}
    SD.stereo_depth(ep)
    assert ep.status["stereo_depth"]["state"] == "skipped" and "already present" in ep.status["stereo_depth"]["detail"]
    assert {p.name: p.stat().st_mtime_ns for p in depth_dir.glob("*.png")} == mtimes


def test_skips_without_stereo_block(tmp_path):
    bundle = tmp_path / "bundle"; write_bundle(bundle, n_frames=6)
    c = json.load(open(bundle / "calibration.json")); c.pop("stereo"); json.dump(c, open(bundle / "calibration.json", "w"))
    root = tmp_path / "episodes"; left = zed_bundle.ingest(bundle, root / "ep", "leader")
    ep = Episode.create(root, "ep", [("leader", "ego", left, None)]); ep.proc_fps = 5.0; ep.save()
    probe(ep); align(ep); extract_frames(ep); SD.stereo_depth(ep)
    assert ep.status["stereo_depth"]["state"] == "skipped" and "no stereo block" in ep.status["stereo_depth"]["detail"]
    assert not (ep.dir / "zed" / "leader" / "depth").exists()
