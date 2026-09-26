"""depth_mono and scene_scan: disparity-fit math, tag anchors, scene alignment, and both stages end to end on synthetic
episodes with a fake Depth Anything (exactly d = s/Z + t) and a fake `colmap` executable that enforces what COLMAP 3.11
needs (known options only, database ids, world-to-camera poses, cross-camera pairs, masks). test_scene_scan_real_colmap
runs the stage against a real COLMAP when one is on PATH."""
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from duet.playground import depth_mono as DM  # noqa: E402
from duet.playground import geometry as G  # noqa: E402
from duet.playground import scene_scan as SC  # noqa: E402
from duet.playground.episode import Episode, Stream  # noqa: E402

W, H = 640, 360
K = np.array([[450.0, 0, 319.5], [0, 450.0, 179.5], [0, 0, 1]])


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """No test may see this machine's pre-fetched weights (`playground.py fetch-models`), Hugging Face cache or station cache."""
    monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "models")); monkeypatch.setenv("PLAYGROUND_CACHE", str(tmp_path / "cache"))
    for v in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(v, str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _rot(rng) -> np.ndarray:
    q = rng.normal(size=4); q /= np.linalg.norm(q); return SC._q2R(*q)


def look_at(pos, target, up=(0, 1, 0)) -> np.ndarray:
    """T_world_cam for a camera at pos looking at target (OpenCV axes: +Z forward, +Y down)."""
    z = np.asarray(target, float) - pos; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x)
    T = np.eye(4); T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, pos; return T


def _episode(tmp_path, name, streams, n, fps=10.0, rig=None, device="cpu") -> Episode:
    ep_dir = tmp_path / name; (ep_dir / "streams").mkdir(parents=True)
    ep = Episode(name=name, root=str(ep_dir), streams=streams, reference=streams[0].name, proc_fps=fps, common_start_s=0.0, common_end_s=n / fps, device=device)
    for s in streams:
        s.offset_status = "reference" if s.name == ep.reference else "manual"
    if rig is not None:
        (ep_dir / "rig.json").write_text(json.dumps(rig))
    ep.save(); return ep


def _write_frames(ep: Episode, name: str, imgs: list[np.ndarray]) -> None:
    """Frames as the frames stage leaves them (C4): NNNNNN.jpg (1-based), index.parquet, stage status done."""
    d = ep.derived / "frames" / name; d.mkdir(parents=True, exist_ok=True); t = ep.frame_times_ref()
    assert len(imgs) == len(t)
    for k, img in enumerate(imgs):
        cv2.imwrite(str(d / f"{k + 1:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    pd.DataFrame({"k": np.arange(len(imgs)), "file": [f"{k + 1:06d}.jpg" for k in range(len(imgs))], "t_ref_s": t,
                  "t_src_s": t + 0.004, "src_frame": np.arange(len(imgs)) * 3}).to_parquet(d / "index.parquet", index=False)
    ep.status["frames"] = {"state": "done"}; ep.save()


def _write_calib(ep: Episode, cams: dict) -> None:
    """calib.json per C7: name -> (Intrinsics at frame size, source, T_world_cam or None)."""
    d = ep.derived / "calib"; d.mkdir(parents=True, exist_ok=True)
    js = {n: {"role": ep.stream(n).role, "intrinsics": i.to_json(), "intrinsics_source": src, "registered": T is not None,
              "T_world_cam": None if T is None else np.asarray(T).tolist()} for n, (i, src, T) in cams.items()}
    (d / "calib.json").write_text(json.dumps({"world": "board", "cameras": js})); ep.status["calib"] = {"state": "done"}; ep.save()


def _render_tag(img, tag_id, centre_cam, size_m, Kc, border_frac=0.25):
    """Draw a fronto-parallel AprilTag 36h11 (black-border edge size_m) centred at centre_cam (camera frame, metres)."""
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tag_id, 400)
    b = int(400 * border_frac); canvas = np.full((400 + 2 * b, 400 + 2 * b), 255, np.uint8); canvas[b:-b, b:-b] = tag
    half = size_m / 2 * (400 + 2 * b) / 400; X, Y, Z = centre_cam
    quad = np.array([[X - half, Y - half, Z], [X + half, Y - half, Z], [X + half, Y + half, Z], [X - half, Y + half, Z]])
    dst = (quad @ Kc.T)[:, :2] / Z; src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) * canvas.shape[0]
    warp = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst.astype(np.float32)), (img.shape[1], img.shape[0]), borderValue=255)
    cover = cv2.warpPerspective(np.full_like(canvas, 255), cv2.getPerspectiveTransform(src, dst.astype(np.float32)), (img.shape[1], img.shape[0]), borderValue=0)
    return np.where(cover > 128, warp, img), dst


# ============================================================================ depth_mono: fit math

def test_fit_recovers_affine_disparity_with_two_and_three_anchors():
    s, t = 5.0, 1.0; z = np.array([2.0, 0.8, 1.3]); d = s / z + t  # exactly the relative checkpoint's contract
    for sel in ([0, 1], [0, 1, 2]):
        a, b, rms, inl, r = DM.fit_inverse_depth(d[sel], z[sel])
        assert np.isclose(a, 1 / s) and np.isclose(b, -t / s) and rms < 1e-9 and inl.all()
    zz = np.linspace(0.3, 6.0, 50); a, b, *_ = DM.fit_inverse_depth(d, z)
    assert np.allclose(1 / (a * (s / zz + t) + b), zz)  # the whole range, not only the anchors


def test_fit_keeps_the_nearest_anchor_that_the_old_code_dropped():
    # the old min-max normalisation mapped the nearest pixel to rel ~ 0 and `(r > 0.02).all()` then dropped every anchor
    s, t = 5.0, 1.0; z = np.array([0.4, 2.0]); a, b, rms, inl, _ = DM.fit_inverse_depth(s / z + t, z)
    assert inl.all() and np.isclose(1 / (a * (s / 0.4 + t) + b), 0.4)


def test_fit_drops_a_bad_anchor_individually():
    s, t = 4.0, 0.5; z = np.array([0.6, 1.0, 1.8, 3.0]); d = s / z + t; zbad = z.copy(); zbad[2] *= 1.5  # one tag with a wrong pose
    a, b, rms, inl, r = DM.fit_inverse_depth(d, zbad)
    assert inl.tolist() == [True, True, False, True] and np.isclose(a, 1 / s) and np.isclose(b, -t / s) and abs(r[2]) > 0.2
    d2 = d.copy(); d2[1] = np.nan; assert DM.fit_inverse_depth(d2, z)[3].tolist() == [True, False, True, True]  # invalid sample


def test_fit_needs_two_distinct_depths_unless_scale_only():
    s, t = 5.0, 1.0
    assert DM.fit_inverse_depth([s / 1.0 + t], [1.0]) is None  # one anchor: scale AND shift unknown
    assert DM.fit_inverse_depth(s / np.array([1.0, 1.1]) + t, [1.0, 1.1]) is None  # too similar (ratio < 1.5)
    assert DM.fit_inverse_depth(s / np.array([1.0, 1.1]) + t, [1.0, 1.1], min_depth_ratio=1.05) is not None
    zpred = np.array([1.1, 2.2]); a, b, rms, inl, _ = DM.fit_inverse_depth(1 / zpred, [1.0, 2.0], scale_only=True)  # metric model 10 % long
    assert b == 0 and np.isclose(a, 1.1) and np.allclose(zpred / a, [1.0, 2.0]) and inl.all()  # depth = Z_pred / a
    assert DM.fit_inverse_depth(1 / zpred[:1], [1.0], scale_only=True) is not None  # one anchor is enough for scale


def test_borrow_nearest_fit_within_window():
    t = np.arange(8) * 0.5; own = np.array([0, 1, 0, 0, 0, 0, 0, 1], bool)
    assert DM._borrow(t, own, 1.0).tolist() == [1, 1, 1, 1, -1, 7, 7, 7]
    assert DM._borrow(t, own, 0.0).tolist() == [-1, 1, -1, -1, -1, -1, -1, 7]


# ============================================================================ depth_mono: anchors

def test_anchor_is_the_raw_distorted_pixel_of_the_tag_centre():
    pytest.importorskip("pupil_apriltags")
    Kd = np.array([[300.0, 0, 319.5], [0, 300.0, 179.5], [0, 0, 1]]); dist = np.array([-0.28, 0.08, 0, 0, 0])
    centre = np.array([0.75, 0.40, 1.1]); pin, _ = _render_tag(np.full((H, W), 255, np.uint8), 7, centre, 0.12, Kd)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))  # distorted view of the pinhole render
    und = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), Kd, dist, P=Kd).reshape(H, W, 2)
    gray = cv2.remap(pin, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=255)
    intr = G.Intrinsics(Kd, dist, W, H)
    anc = DM.tag_anchors(gray, intr, 0.12, max_err_px=3.0); assert len(anc) == 1 and anc[0]["tag_id"] == 7
    raw = cv2.projectPoints(centre[None], np.zeros(3), np.zeros(3), Kd, dist)[0].ravel(); ideal = (Kd @ centre)[:2] / centre[2]
    assert np.linalg.norm(ideal - raw) > 20  # the undistorted position would be this far off on the distorted depth map
    assert np.linalg.norm([anc[0]["u"] - raw[0], anc[0]["v"] - raw[1]]) < 1.5 and abs(anc[0]["z_m"] - 1.1) < 0.03


# ============================================================================ depth_mono: the stage

def _true_depth(u, v, tags):
    """Scene depth (m) at frame pixel (u, v): floor-like ramp (1/Z linear in v, 3.0 m top -> 0.4 m bottom), fronto-parallel
    tag patches at their depth, a 'hand' at 0.5 m."""
    Z = 1 / (1 / 3.0 + (1 / 0.4 - 1 / 3.0) * (v / (H - 1))) + 0 * u
    for c, half_px in tags:
        Z = np.where((abs(u - c[0]) < half_px) & (abs(v - c[1]) < half_px), c[2], Z)
    return np.where((u >= 30) & (u < 110) & (v >= 270) & (v < 350), 0.5, Z)


def _grid(w, h):  # pixel-centre-aligned sample positions of a w x h map in frame pixels (cv2.resize convention)
    return np.meshgrid((np.arange(w) + 0.5) * W / w - 0.5, (np.arange(h) + 0.5) * H / h - 0.5)


def _fake_transformers(tags, s_aff=5.0, t_aff=1.0, calls=None, model="relative", metric_err=1.1, local=None):
    """transformers stand-in: DA-V2's processor makes 640x360 -> 518x294 inputs; the relative model returns d = s/Z + t,
    the metric one metric_err * Z (metres, 10 % long by default)."""
    import torch

    class Proc:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros(len(images), 3, 294, 518)}

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, pixel_values):
            if calls is not None:
                calls.append(len(pixel_values))
            Z = _true_depth(*_grid(518, 294), tags); d = metric_err * Z if model == "metric_indoor" else s_aff / Z + t_aff
            return types.SimpleNamespace(predicted_depth=torch.from_numpy(np.repeat(d[None], len(pixel_values), 0).astype(np.float32)))

    def load(kind):
        def f(model_id, revision=None, **kw):
            assert (model_id, revision) == DM.MODELS[model][:2] if local is None else (model_id, kw) == (str(local), {"local_files_only": True})  # pinned
            return Proc() if kind == "proc" else Model()
        return f
    return types.SimpleNamespace(AutoImageProcessor=types.SimpleNamespace(from_pretrained=load("proc")),
                                 AutoModelForDepthEstimation=types.SimpleNamespace(from_pretrained=load("model")))


def _depth_episode(tmp_path, n=10, tags=True, rig_extra=None, tagless_from=None):
    """Ego episode whose frames show two fronto-parallel tags (2.0 m and 0.8 m) up to frame tagless_from (none after)."""
    pytest.importorskip("torch")
    tag_list = [((0.30, -0.40, 2.0), 20), ((-0.25, 0.02, 0.8), 21)] if tags else []
    plain = np.full((H, W), 200, np.uint8); img = plain.copy(); patches = []
    for c, tid in tag_list:
        img, quad = _render_tag(img, tid, np.array(c), 0.16, K)
        uc, vc = (K @ np.array(c))[:2] / c[2]; patches.append(((uc, vc, c[2]), 0.5 * (quad[1, 0] - quad[0, 0]) - 2))
    ep = _episode(tmp_path, "dm", [Stream("ego1", "ego", "streams/ego1.mp4", person="a")], n, rig={"tag_size_m": 0.16, **(rig_extra or {})})
    _write_frames(ep, "ego1", [cv2.cvtColor(img if tagless_from is None or k < tagless_from else plain, cv2.COLOR_GRAY2BGR) for k in range(n)])
    _write_calib(ep, {"ego1": (G.Intrinsics(K, np.zeros(5), W, H), "rig_json", None)})
    return ep, patches


def test_depth_mono_recovers_metric_depth_from_two_tags(tmp_path, monkeypatch):
    pytest.importorskip("pupil_apriltags")
    ep, patches = _depth_episode(tmp_path)
    calls = []; monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches, calls=calls))
    n_list = []; real = DM.P.frame_paths
    monkeypatch.setattr(DM.P, "frame_paths", lambda e, s: n_list.append(1) or real(e, s))
    DM.depth_mono(ep, batch=2)
    st = ep.status["depth_mono"]; assert st["state"] == "done", st
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False)
    assert z["frame_idx"].tolist() == [0, 5] and np.allclose(z["t_ref_s"], [0.0, 0.5]) and str(z["units"]) == "m"
    assert z["metric"].all() and (z["fit_source"] == "anchors").all() and z["anchor_used"].all() and sorted(set(z["anchor_tag"].tolist())) == [20, 21]
    assert str(z["model_revision"]) == DM.MODELS["relative"][1] and str(z["intrinsics_source"]) == "rig_json"
    Zt = _true_depth(*_grid(256, 144), patches); err = np.abs(z["depth_m"][0].astype(float) - Zt) / Zt
    assert np.nanmedian(err) < 0.005, np.nanmedian(err)  # the old code: 419 % median error, inverted ordering
    assert np.corrcoef(Zt.ravel(), z["depth_m"][0].astype(float).ravel())[0, 1] > 0.99
    assert n_list == [1] and calls == [2]  # frame list read once per stream; 2 sampled frames batched together


def test_depth_mono_relative_only_without_anchors(tmp_path, monkeypatch):
    ep, patches = _depth_episode(tmp_path, tags=False)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches))
    DM.depth_mono(ep)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False)
    assert not z["metric"].any() and np.isnan(z["depth_m"].astype(float)).all() and (z["fit_source"] == "none").all()
    assert str(z["pred_units"]) == "disparity_affine" and np.isfinite(z["pred"].astype(float)).all()
    assert "0/2 frames metric" in ep.status["depth_mono"]["detail"]


def test_depth_mono_borrows_the_nearest_fit_and_flags_it(tmp_path, monkeypatch):
    pytest.importorskip("pupil_apriltags")
    ep, patches = _depth_episode(tmp_path, n=15, tagless_from=5)  # sampled k = 0 (tags), 5 and 10 (no tags); 0.5 s apart
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches))
    DM.depth_mono(ep, borrow_s=0.6)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False)
    assert z["fit_source"].tolist() == ["anchors", "borrowed", "none"] and z["fit_from"].tolist() == [0, 0, -1] and z["metric"].tolist() == [True, True, False]
    assert np.array_equal(z["depth_m"][1], z["depth_m"][0]) and np.isnan(z["depth_m"][2].astype(float)).all() and set(z["anchor_row"].tolist()) == {0}


def test_depth_mono_metric_checkpoint_scale_correction(tmp_path, monkeypatch):
    pytest.importorskip("pupil_apriltags")
    ep, patches = _depth_episode(tmp_path, n=10, tagless_from=5, rig_extra={"depth_mono": {"model": "metric_indoor", "borrow_s": 0}})
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches, model="metric_indoor"))  # predicts 1.1 x the true depth
    DM.depth_mono(ep)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False); Zt = _true_depth(*_grid(256, 144), patches)
    assert str(z["pred_units"]) == "m" and str(z["model_kind"]) == "metric" and z["fit_source"].tolist() == ["anchors", "model"] and z["metric"].all()
    assert np.isclose(z["fit_a"][0], 1.1, rtol=1e-3) and z["fit_b"].tolist() == [0.0, 0.0]
    assert np.nanmedian(np.abs(z["depth_m"][0].astype(float) / Zt - 1)) < 0.005  # corrected by the tags
    assert np.nanmedian(np.abs(z["depth_m"][1].astype(float) / Zt - 1.1)) < 0.005  # no anchor: the model's own metres, flagged


def test_depth_mono_rescales_native_intrinsics_and_refuses_a_different_aspect(tmp_path, monkeypatch):
    pytest.importorskip("pupil_apriltags")
    ep, patches = _depth_episode(tmp_path)
    K2 = K.copy(); K2[:2] *= 2; K2[:2, 2] += 0.5  # the same lens calibrated at the native 1280x720 (pixel-centre convention)
    _write_calib(ep, {"ego1": (G.Intrinsics(K2, np.zeros(5), 2 * W, 2 * H), "rig_json", None)})
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches))
    DM.depth_mono(ep)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False); Zt = _true_depth(*_grid(256, 144), patches)
    assert "rescaled 1280x720 -> 640x360" in str(z["intrinsics_source"]) and z["metric"].all()
    assert np.nanmedian(np.abs(z["depth_m"][0].astype(float) / Zt - 1)) < 0.005
    _write_calib(ep, {"ego1": (G.Intrinsics(K2, np.zeros(5), 2 * W, 960), "rig_json", None)})  # 4:3 sensor mode: K does not apply
    DM.depth_mono(ep)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False)
    assert not z["metric"].any() and "do not fit" in str(z["intrinsics_source"]) and ep.status["depth_mono"]["state"] == "done"


def test_depth_mono_prefers_prefetched_weights(tmp_path, monkeypatch):
    ep, patches = _depth_episode(tmp_path, tags=False); monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "models"))
    local = DM.weights_dir(*DM.MODELS["relative"][:2]); local.mkdir(parents=True)
    assert local.name == "depth-anything--Depth-Anything-V2-Small-hf@" + DM.MODELS["relative"][1]
    for f in DM.WEIGHT_FILES[:-1]:  # incomplete fetch: the Hub (pinned revision) is used
        (local / f).write_text("{}")
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches))
    DM.depth_mono(ep); assert str(np.load(ep.derived / "depth_mono" / "ego1.npz")["model_weights"]) == "hub"
    (local / DM.WEIGHT_FILES[-1]).write_bytes(b"")
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches, local=local))
    DM.depth_mono(ep); assert str(np.load(ep.derived / "depth_mono" / "ego1.npz")["model_weights"]) == f"local:{local.name}"


def test_depth_mono_nominal_intrinsics_give_no_anchors(tmp_path, monkeypatch):
    ep, patches = _depth_episode(tmp_path)
    _write_calib(ep, {"ego1": (G.Intrinsics(K, np.zeros(5), W, H), "nominal_fov", None)})
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(patches))
    DM.depth_mono(ep)
    z = np.load(ep.derived / "depth_mono" / "ego1.npz", allow_pickle=False)
    assert not z["metric"].any() and "nominal" in str(z["intrinsics_source"])


def test_depth_mono_skips_zed_streams_without_loading_the_model(tmp_path, monkeypatch):
    ep, _ = _depth_episode(tmp_path)
    (ep.dir / "zed" / "ego1" / "depth").mkdir(parents=True); cv2.imwrite(str(ep.dir / "zed" / "ego1" / "depth" / "000000.png"), np.zeros((4, 4), np.uint16))
    boom = types.SimpleNamespace(from_pretrained=lambda *a, **k: pytest.fail("model loaded although no stream needs it"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoImageProcessor=boom, AutoModelForDepthEstimation=boom))
    DM.depth_mono(ep)
    assert ep.status["depth_mono"]["state"] == "skipped" and "ZED" in ep.status["depth_mono"]["detail"]
    assert not (ep.derived / "depth_mono").exists()


def test_depth_mono_empty_zed_dir_is_not_depth_and_offline_host_skips(tmp_path, monkeypatch):
    ep, _ = _depth_episode(tmp_path)
    (ep.dir / "zed" / "ego1" / "depth").mkdir(parents=True)  # zed_export.py --no-depth leaves an empty depth/

    def offline(*a, **k):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load this file")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoImageProcessor=types.SimpleNamespace(from_pretrained=offline),
                                                                          AutoModelForDepthEstimation=types.SimpleNamespace(from_pretrained=offline)))
    DM.depth_mono(ep)
    st = ep.status["depth_mono"]; assert st["state"] == "skipped" and "unavailable" in st["detail"] and "OSError" in st["detail"]
    assert not (ep.derived / "depth_mono").exists()


# ============================================================================ scene_scan: alignment math

def _colmap_views(Tw: dict, rng, sim, jitter=1e-3, per_cam=3) -> dict:
    """COLMAP-frame poses of per_cam images per fixed camera. sim = (R0, s0, t0): X_colmap = s0 * R0 @ X_board + t0."""
    R0, s0, t0 = sim; out = {}
    for n, T in Tw.items():
        v = []
        for _ in range(per_cam):
            dR = cv2.Rodrigues(rng.normal(0, jitter, 3))[0]
            v.append((dR @ T[:3, :3].T @ R0.T, s0 * R0 @ T[:3, 3] + t0 + rng.normal(0, jitter, 3)))  # (R_cam_colmap, C_colmap)
        out[n] = v
    return out


def _sim(seed):
    rng = np.random.default_rng(seed); return _rot(rng), 3.7, rng.normal(0, 2, 3)


TWO = {"exoA": look_at(np.array([-0.3, -0.4, -1.2]), [0.3, 0.2, 0]), "exoB": look_at(np.array([0.9, -0.3, -1.1]), [0.3, 0.2, 0])}
THREE = {**TWO, "exoC": look_at(np.array([0.3, 1.1, -1.3]), [0.3, 0.2, 0])}
RAIL = {n: look_at(np.array([x, -0.4, -1.2]), [0.3, 0.2, 0]) for n, x in (("exoA", -0.3), ("exoB", 0.3), ("exoC", 0.9))}


@pytest.mark.parametrize("cams", [TWO, THREE, RAIL], ids=["two", "three", "rail"])
def test_alignment_recovers_board_frame(cams):
    rng = np.random.default_rng(1); R0, s0, t0 = _sim(1); views = _colmap_views(cams, rng, (R0, s0, t0))
    res = SC.align_to_board(views, cams)
    assert res["aligned"], res
    X = np.random.default_rng(2).uniform(-0.5, 1.5, (200, 3)); Xc = s0 * X @ R0.T + t0
    T = res["T_world_colmap"]; err = np.linalg.norm(res["scale"] * Xc @ T[:3, :3].T + T[:3, 3] - X, axis=1)
    assert np.median(err) < 0.005 and res["rot_resid_deg"] < 0.5 and abs(res["scale"] - 1 / s0) < 1e-3
    assert (res["umeyama_diff_deg"] is not None) == (cams is THREE)  # the centre-only check needs non-collinear centres
    if cams is THREE:
        assert res["umeyama_diff_deg"] < 0.5
    if cams is TWO:  # the old alignment, Umeyama on the centres of 2 cameras: the roll about the baseline is free
        src = np.concatenate([[C for _, C in v] for v in views.values()]); dst = np.repeat([cams[n][:3, 3] for n in cams], 3, axis=0)
        assert G.fit_similarity(src, dst) is None  # refused by the conditioning check
        Tu, su = G.fit_similarity(src, dst, min_minor_ratio=0.0)  # forced, as the old code did: 1 mm jitter picks the roll
        assert np.median(np.linalg.norm(su * Xc @ Tu[:3, :3].T + Tu[:3, 3] - X, axis=1)) > 0.05


def test_alignment_refuses_one_camera_and_inconsistent_orientations():
    rng = np.random.default_rng(4); views = _colmap_views(TWO, rng, _sim(4))
    one = SC.align_to_board({"exoA": views["exoA"]}, TWO); assert not one["aligned"] and "need >= 2" in one["reason"]
    swapped = {"exoA": views["exoB"], "exoB": views["exoA"]}; assert not SC.align_to_board(swapped, TWO)["aligned"]  # scale < 0
    tilt = cv2.Rodrigues(np.array([0.0, np.radians(5), 0.0]))[0]  # camera B's COLMAP orientation off by 5 deg
    bad = SC.align_to_board({**views, "exoB": [(R @ tilt, C) for R, C in views["exoB"]]}, TWO)
    assert not bad["aligned"] and "rotation residual" in bad["reason"] and bad["rot_resid_deg"] > 2


def test_quaternion_round_trip_and_colmap_camera_models():
    rng = np.random.default_rng(5)
    for R in [_rot(rng) for _ in range(50)] + [np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]:
        assert np.allclose(SC._q2R(*SC._R2q(R)), R, atol=1e-12)
    m, p = SC.colmap_camera(G.Intrinsics(K, np.array([-0.1, 0.02, 0.001, 0.002, 0.0]), W, H))
    assert m == "OPENCV" and p == [450.0, 450.0, 320.0, 180.0, -0.1, 0.02, 0.001, 0.002]  # +0.5 px: COLMAP's pixel origin
    m, p = SC.colmap_camera(G.Intrinsics(K, np.array([-0.1, 0.02, 0.0, 0.0, 0.01]), W, H))
    assert m == "FULL_OPENCV" and p[4:] == [-0.1, 0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        SC.colmap_camera(G.Intrinsics(K, np.r_[np.zeros(8), 0.01, 0, 0, 0], W, H))


def test_read_text_model_last_image_without_points_line(tmp_path):
    (tmp_path / "images.txt").write_text("# header\n1 1 0 0 0 0.1 0.2 0.3 1 a/plate.png\n10 20 -1\n2 1 0 0 0 0 0 0 2 b/plate.png")
    (tmp_path / "points3D.txt").write_text("# h\n1 0.5 0.6 0.7 10 20 30 0.25 1 0 2 0\n")
    imgs, pts = SC.read_text_model(tmp_path)
    assert sorted(imgs) == ["a/plate.png", "b/plate.png"] and np.allclose(imgs["a/plate.png"][1], [0.1, 0.2, 0.3])
    assert pts["xyz"].tolist() == [[0.5, 0.6, 0.7]] and pts["track_len"].tolist() == [2] and pts["rgb"].tolist() == [[10, 20, 30]]


# ============================================================================ scene_scan: the stage with a fake colmap

FAKE_COLMAP = r'''
import json, os, sqlite3, struct, sys, time
from pathlib import Path
OPTS = {
 "feature_extractor": ["database_path", "image_path", "image_list_path", "camera_mode", "ImageReader.mask_path", "ImageReader.camera_model",
                       "ImageReader.single_camera", "ImageReader.single_camera_per_folder", "ImageReader.camera_params", "SiftExtraction.use_gpu"],
 "matches_importer": ["database_path", "match_list_path", "match_type", "SiftMatching.use_gpu"],
 "point_triangulator": ["database_path", "image_path", "input_path", "output_path", "clear_points", "refine_intrinsics", "Mapper.ba_refine_focal_length",
                        "Mapper.ba_refine_principal_point", "Mapper.ba_refine_extra_params", "Mapper.tri_ignore_two_view_tracks", "Mapper.tri_min_angle",
                        "Mapper.filter_max_reproj_error"],
 "mapper": ["database_path", "image_path", "output_path", "Mapper.ba_refine_focal_length", "Mapper.ba_refine_principal_point",
            "Mapper.ba_refine_extra_params", "Mapper.tri_ignore_two_view_tracks", "Mapper.tri_min_angle", "Mapper.filter_max_reproj_error"],
 "model_converter": ["input_path", "output_path", "output_type"]}
args = sys.argv[1:]
def log(obj):
    with open(os.environ["FAKE_COLMAP_LOG"], "a") as f:
        f.write(json.dumps(obj) + "\n")
def die(msg):
    sys.stderr.write("E: " + msg + "\n"); sys.exit(1)
if not args or args[0] == "help":
    print(f"COLMAP {os.environ.get('FAKE_COLMAP_VERSION', '3.11.1')} -- Structure-from-Motion and Multi-View Stereo\n(Commit fake on fake without CUDA)"); sys.exit(0)
cmd = args[0]
if args[1:2] == ["-h"]:
    print("\n".join(f"  --{o} arg" for o in OPTS[cmd])); sys.exit(0)
log(args)
kv = {args[i][2:]: args[i + 1] for i in range(1, len(args) - 1, 2)}
if [k for k in kv if k not in OPTS[cmd]]:
    die(f"unrecognised option(s) {[k for k in kv if k not in OPTS[cmd]]}")
if os.environ.get("FAKE_COLMAP_FAIL") == cmd:
    sys.stderr.write("E: fake failure in " + cmd + "\n" + "x" * 3000 + "\nEND OF FAKE LOG\n"); sys.exit(1)
if os.environ.get("FAKE_COLMAP_SLEEP", ":").split(":")[0] == cmd:
    time.sleep(float(os.environ["FAKE_COLMAP_SLEEP"].split(":")[1]))
import numpy as np
sc = json.load(open(os.environ["FAKE_COLMAP_SCENARIO"]))
MODELS = {"OPENCV": (4, 8), "FULL_OPENCV": (6, 12), "OPENCV_FISHEYE": (5, 8)}
def db(path):
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY, model INTEGER, width INTEGER, height INTEGER, params BLOB, prior_focal_length INTEGER)")
    c.execute("CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY, name TEXT UNIQUE, camera_id INTEGER)")
    return c
def q2R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
def R2q(R):  # Bar-Itzhack: eigenvector of the symmetric 4x4 K (stable at 180 deg, where w -> 0)
    Km = np.array([[R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[2, 1] - R[1, 2]],
                   [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[0, 2] - R[2, 0]],
                   [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[1, 0] - R[0, 1]],
                   [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3
    x, y, z, w = np.linalg.eigh(Km)[1][:, -1]
    return [w, x, y, z]
def write_model(out, poses, points):
    """poses: name -> (image_id, camera_id, R_cam_world, C_world). A 'binary' model dir holding TXT (the fake converter copies it)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "images.bin").write_bytes(struct.pack("<Q", len(poses)))
    g = lambda vals: " ".join(f"{float(v):.17g}" for v in vals)  # noqa: E731
    (out / "images.txt").write_text("# fake\n" + "".join(f"{i} {g(R2q(R))} {g(-R @ C)} {c} {n}\n\n" for n, (i, c, R, C) in poses.items()))
    (out / "points3D.txt").write_text("# fake\n" + "".join(f"{j + 1} {g(p)} 10 20 30 0.5 1 0 2 0\n" for j, p in enumerate(points)))
if cmd == "feature_extractor":
    root, masks = Path(kv["image_path"]), Path(kv["ImageReader.mask_path"])
    names = Path(kv["image_list_path"]).read_text().split()
    model = kv["ImageReader.camera_model"]; params = [float(v) for v in kv["ImageReader.camera_params"].split(",")]
    if kv.get("ImageReader.single_camera") != "1" or len(params) != MODELS[model][1] or kv.get("SiftExtraction.use_gpu") not in ("0", "1"):
        die("camera setup")
    for n in names:
        if not (root / n).exists() or not (masks / (n + ".png")).exists():
            die(f"missing image or mask for {n}")
    c = db(kv["database_path"]); cid = c.execute("INSERT INTO cameras (model, width, height, params, prior_focal_length) VALUES (?, 640, 360, ?, 1)",
                                                (MODELS[model][0], np.array(params, np.float64).tobytes())).lastrowid
    base = c.execute("SELECT COALESCE(MAX(image_id), 0) FROM images").fetchone()[0]
    for j, n in enumerate(sorted(names, reverse=True)):  # ids deliberately NOT in name order
        c.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)", (base + 7 + j, n, cid))
    c.commit(); log({"extracted": sorted(names), "camera": cid})
elif cmd == "matches_importer":
    pairs = [ln.split() for ln in Path(kv["match_list_path"]).read_text().splitlines() if ln.strip()]
    log({"pairs": len(pairs), "same_camera": sum(a.split("/")[0] == b.split("/")[0] for a, b in pairs)})
    c = db(kv["database_path"]); ids = {n: i for i, n in c.execute("SELECT image_id, name FROM images")}
    c.execute("CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY, rows INTEGER)")
    for a, b in pairs:
        i1, i2 = sorted((ids[a], ids[b])); c.execute("INSERT INTO two_view_geometries VALUES (?, ?)", (i1 * 2147483647 + i2, 0 if os.environ.get("FAKE_COLMAP_NOMATCH") else 40))
    c.commit()
elif cmd == "point_triangulator":
    for k in ("Mapper.ba_refine_focal_length", "Mapper.ba_refine_principal_point", "Mapper.ba_refine_extra_params", "Mapper.tri_ignore_two_view_tracks", "refine_intrinsics"):
        if kv.get(k) != "0":
            die(f"{k} must be 0")
    c = db(kv["database_path"]); ids = {n: (i, cam) for i, n, cam in c.execute("SELECT image_id, name, camera_id FROM images")}
    dbcams = {i: np.frombuffer(p, np.float64) for i, p in c.execute("SELECT camera_id, params FROM cameras")}
    inp = Path(kv["input_path"]); poses = {}
    for ln in (inp / "cameras.txt").read_text().splitlines():
        v = ln.split()
        if not np.allclose([float(x) for x in v[4:]], dbcams[int(v[0])]):
            die("cameras.txt params differ from the database")
    lines = [ln for ln in (inp / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        v = lines[i].split(); name = v[9]
        if ids.get(name) != (int(v[0]), int(v[8])):
            die(f"image {name} id/camera {v[0]}/{v[8]} != database {ids.get(name)}")
        R = q2R(*map(float, v[1:5])); C = -R.T @ np.array(v[5:8], float)
        T = np.array(sc["cams"][name.split("/")[0]]); Rt, Ct = T[:3, :3].T, T[:3, 3]  # world-to-camera expected
        if not (np.allclose(R, Rt, atol=1e-9) and np.allclose(C, Ct, atol=1e-9)):
            die(f"pose of {name} is not the world-to-camera form of T_world_cam")
        poses[name] = (int(v[0]), int(v[8]), R, C)
    pts = np.array(sc["points"])
    if sc.get("move"):  # a COLMAP that re-gauges the model: every pose and point through one similarity
        R0, s0, t0 = np.array(sc["R0"]), sc["s0"], np.array(sc["t0"])
        poses = {n: (i, cam, R @ R0.T, s0 * R0 @ C + t0) for n, (i, cam, R, C) in poses.items()}; pts = s0 * pts @ R0.T + t0
    write_model(kv["output_path"], poses, pts)
elif cmd == "mapper":
    c = db(kv["database_path"]); ids = {n: (i, cam) for i, n, cam in c.execute("SELECT image_id, name, camera_id FROM images")}
    R0, s0, t0 = np.array(sc["R0"]), sc["s0"], np.array(sc["t0"])
    for mname, keep in sc["mapper_models"].items():
        poses = {}
        for n, (i, cam) in ids.items():
            if n.split("/")[0] in keep:
                T = np.array(sc["cams"][n.split("/")[0]]); poses[n] = (i, cam, T[:3, :3].T @ R0.T, s0 * R0 @ T[:3, 3] + t0)
        write_model(Path(kv["output_path"]) / mname, poses, s0 * np.array(sc["points"]) @ R0.T + t0)
elif cmd == "model_converter":
    if kv["output_type"] != "TXT":
        die("TXT only")
    for f in ("images.txt", "points3D.txt"):
        (Path(kv["output_path"]) / f).write_text((Path(kv["input_path"]) / f).read_text())
'''


@pytest.fixture
def fake_colmap(tmp_path, monkeypatch):
    b = tmp_path / "bin"; b.mkdir(); exe = b / "colmap"
    exe.write_text(f"#!{sys.executable}\n" + FAKE_COLMAP); exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_COLMAP_LOG", str(tmp_path / "colmap_log.jsonl")); monkeypatch.setenv("FAKE_COLMAP_SCENARIO", str(tmp_path / "scenario.json"))
    monkeypatch.setenv("PLAYGROUND_CACHE", str(tmp_path / "cache"))
    for v in ("FAKE_COLMAP_FAIL", "FAKE_COLMAP_SLEEP", "FAKE_COLMAP_NOMATCH", "FAKE_COLMAP_VERSION"):
        monkeypatch.delenv(v, raising=False)
    tmp = tmp_path / "tmp"; tmp.mkdir(); monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    (tmp_path / "colmap_log.jsonl").write_text("")
    return tmp_path


def _colmap_log(root):
    return [json.loads(ln) for ln in (root / "colmap_log.jsonl").read_text().splitlines()]


def _scan_episode(root, name, cams=TWO, n=20, move=False, models=None, unregistered=(), boxes=True):
    streams = [Stream(c, "exo", f"streams/{c}.mp4") for c in [*cams, *unregistered]]
    ep = _episode(root, name, streams, n, rig={"board": {"squares_x": 7}})
    rng = np.random.default_rng(0); frame = rng.integers(0, 255, (H, W, 3), np.uint8)
    for s in streams:
        _write_frames(ep, s.name, [frame] * n)
        if boxes:  # one 'person' box per frame (body2d layout C5: [n, 4 slots, x0 y0 x1 y1 conf], NaN = empty slot)
            b = np.full((n, 4, 5), np.nan, np.float32); b[:, 0] = [100, 50, 220, 300, 0.9]
            (ep.derived / "body2d").mkdir(exist_ok=True); np.savez(ep.derived / "body2d" / f"{s.name}.npz", boxes=b)
    if boxes:
        ep.status["body2d"] = {"state": "done"}; ep.save()
    intr = G.Intrinsics(K, np.array([-0.05, 0.01, 0, 0, 0]), W, H)
    _write_calib(ep, {**{c: (intr, "rig_json", T) for c, T in cams.items()}, **{c: (intr, "rig_json", None) for c in unregistered}})
    R0, s0, t0 = _sim(7); X = np.random.default_rng(8).uniform([-0.3, -0.2, -0.5], [0.9, 0.6, 0.0], (50, 3))
    (root / "scenario.json").write_text(json.dumps({"cams": {c: T.tolist() for c, T in {**cams, **{u: np.eye(4) for u in unregistered}}.items()},
                                                    "points": X.tolist(), "move": move, "R0": R0.tolist(), "s0": s0, "t0": t0.tolist(),
                                                    "mapper_models": models or {}}))
    return ep, X


def _leftover_workspaces(root):
    return glob.glob(str(root / "tmp" / "duet_colmap_*"))


def test_scene_scan_known_poses_land_in_board_frame(fake_colmap):
    ep, X = _scan_episode(fake_colmap, "sc1")
    SC.scene_scan(ep)
    st = ep.status["scene_scan"]; assert st["state"] == "done" and "aligned=True" in st["detail"], st
    z = np.load(ep.derived / "scene_scan" / "points.npz", allow_pickle=False)
    assert "xyz_colmap" not in z.files and bool(z["aligned"]) and np.abs(z["xyz_board"] - X).max() < 1e-5 and str(z["frame"]) == "board"
    scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text())
    assert scan["method"] == "triangulate" and scan["alignment"]["max_centre_change_m"] < 1e-6 and scan["colmap"]["version"] == "3.11.1"
    assert scan["matches"] == {"exoA-exoB": 9 * 40}
    assert {n: len(v) for n, v in scan["views"].items()} == {"exoA": 3, "exoB": 3} and scan["views"]["exoA"][0]["kind"] == "clean_plate"
    log = _colmap_log(fake_colmap); pairs = next(e for e in log if isinstance(e, dict) and "pairs" in e)
    assert pairs == {"pairs": 9, "same_camera": 0}  # 3 x 3 cross-camera pairs only
    ext = next(e for e in log if isinstance(e, list) and e[0] == "feature_extractor"); assert ext[ext.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert (ep.derived / "logs" / "scene_scan" / "triangulate.log").exists() and not _leftover_workspaces(fake_colmap)


def test_scene_scan_clean_plate_masks_people():
    rng = np.random.default_rng(0); bg = rng.integers(0, 255, (60, 80, 3), np.uint8); imgs, masks = [], []
    for k in range(9):  # a 'person' block moving across the frame, masked in every frame where it is
        im = bg.copy(); x = 5 + 7 * k; im[10:40, x:x + 12] = 255; m = np.full((60, 80), 255, np.uint8); m[10:40, x:x + 12] = 0; imgs.append(im); masks.append(m)
    plate, ok = SC.clean_plate(imgs, masks)
    assert np.array_equal(plate, bg) and ok.min() == 255
    box = np.full((4, 5), np.nan); box[0] = [10, 20, 30, 40, 0.8]; m = SC._people_mask((60, 80), box, 2)
    assert m[20:40, 10:30].max() == 0 and m[18, 8] == 0 and m[50, 70] == 255 and m[17, 7] == 255


def test_scene_scan_budget_is_shared_evenly(fake_colmap):
    ep, _ = _scan_episode(fake_colmap, "sc2", cams=THREE)
    SC.scene_scan(ep, max_images=6, extra_frames=5)  # the old loop gave all of max_images to the first camera
    scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text())
    assert {n: len(v) for n, v in scan["views"].items()} == {"exoA": 2, "exoB": 2, "exoC": 2}
    ks = [v[1]["frames"][0] for v in scan["views"].values()]; assert ks == [10, 10, 10]  # even stride (bin centre), not frames 0..5 of camera A


def test_scene_scan_realigns_when_colmap_moves_poses(fake_colmap):
    ep, X = _scan_episode(fake_colmap, "sc3", move=True)
    SC.scene_scan(ep)
    z = np.load(ep.derived / "scene_scan" / "points.npz", allow_pickle=False); scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text())
    assert bool(z["aligned"]) and scan["alignment"]["max_centre_change_m"] > 0.1 and np.abs(z["xyz_board"] - X).max() < 1e-4 and abs(float(z["scale"]) - 1 / 3.7) < 1e-6


def test_scene_scan_mapper_picks_largest_model_and_aligns(fake_colmap):
    models = {"0": ["exoA"], "1": ["exoB"], "10": ["exoA", "exoB", "exoC"]}  # "10" < "2" lexicographically: the old code took sparse/0
    ep, X = _scan_episode(fake_colmap, "sc4", models=models, unregistered=("exoC",))
    SC.scene_scan(ep, method="mapper")
    scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text()); z = np.load(ep.derived / "scene_scan" / "points.npz", allow_pickle=False)
    assert scan["mapper_models"] == {"0": 3, "1": 3, "10": 9} and scan["n_registered"] == 9 and set(scan["views"]) == {"exoA", "exoB", "exoC"}
    assert bool(z["aligned"]) and np.abs(z["xyz_board"] - X).max() < 1e-6


def test_scene_scan_unaligned_points_never_under_board_keys(fake_colmap):
    ep, _ = _scan_episode(fake_colmap, "sc5", models={"0": ["exoA", "exoC"]}, unregistered=("exoC",))  # one registered camera in the model
    SC.scene_scan(ep, method="mapper")
    z = np.load(ep.derived / "scene_scan" / "points.npz", allow_pickle=False); scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text())
    assert "xyz_board" not in z.files and "xyz_colmap" in z.files and not bool(z["aligned"]) and str(z["frame"]) == "colmap"
    assert "need >= 2" in scan["alignment"]["reason"] and scan["cache"]["stored"] is False and "aligned=False" in ep.status["scene_scan"]["detail"]


@pytest.mark.parametrize("step", ["feature_extractor", "matches_importer", "point_triangulator"])
def test_scene_scan_failure_cleans_workspace_and_keeps_full_log(fake_colmap, monkeypatch, step):
    ep, _ = _scan_episode(fake_colmap, "sc6")
    monkeypatch.setenv("FAKE_COLMAP_FAIL", step)
    with pytest.raises(RuntimeError, match=f"colmap {step} exited 1"):
        SC.scene_scan(ep)
    assert not _leftover_workspaces(fake_colmap) and not (ep.derived / "scene_scan" / "points.npz").exists()
    log = ep.derived / "logs" / "scene_scan" / {"feature_extractor": "extract_exoA", "matches_importer": "match", "point_triangulator": "triangulate"}[step]
    text = log.with_suffix(".log").read_text(); assert "x" * 3000 in text and "END OF FAKE LOG" in text  # full stderr, not a 600-char tail


def test_scene_scan_without_matches_fails_before_colmap_aborts(fake_colmap, monkeypatch):
    ep, _ = _scan_episode(fake_colmap, "sc9"); monkeypatch.setenv("FAKE_COLMAP_NOMATCH", "1")
    with pytest.raises(RuntimeError, match="no verified feature matches in 9 cross-camera image pairs"):
        SC.scene_scan(ep)
    assert not any(isinstance(e, list) and e[0] == "point_triangulator" for e in _colmap_log(fake_colmap)) and not _leftover_workspaces(fake_colmap)


def test_scene_scan_timeout_kills_colmap(fake_colmap, monkeypatch):
    ep, _ = _scan_episode(fake_colmap, "sc7")
    monkeypatch.setenv("FAKE_COLMAP_SLEEP", "point_triangulator:30")
    t0 = time.time()
    with pytest.raises(RuntimeError, match="timed out"):
        SC.scene_scan(ep, timeout_s=2)
    assert time.time() - t0 < 20 and not _leftover_workspaces(fake_colmap)


def test_scene_scan_station_cache(fake_colmap, monkeypatch):
    def ran(ep):  # did COLMAP process anything (not just `colmap help`/-h) in this run?
        hit = "cache hit" in ep.status["scene_scan"]["detail"]; log = _colmap_log(fake_colmap); (fake_colmap / "colmap_log.jsonl").write_text("")
        assert hit == (log == []), (hit, log); return not hit
    ep1, X = _scan_episode(fake_colmap, "ep1"); SC.scene_scan(ep1); assert ran(ep1)
    scan1 = json.loads((ep1.derived / "scene_scan" / "scan.json").read_text()); assert scan1["cache"]["stored"] is True
    ep2, _ = _scan_episode(fake_colmap, "ep2"); SC.scene_scan(ep2); assert not ran(ep2)  # same station: COLMAP not run at all
    z = np.load(ep2.derived / "scene_scan" / "points.npz", allow_pickle=False); assert np.abs(z["xyz_board"] - X).max() < 1e-5
    SC.scene_scan(ep2, force=True); assert ran(ep2)  # --force bypasses the cache and refreshes the entry
    entry = SC._cache_root(ep2) / json.loads((ep2.derived / "scene_scan" / "scan.json").read_text())["cache"]["key"]
    assert json.loads((entry / "scan.json").read_text())["episode"] == "ep2"
    SC.scene_scan(ep1); assert not ran(ep1)
    monkeypatch.setenv("FAKE_COLMAP_VERSION", "4.2.0"); SC.scene_scan(ep1); assert ran(ep1)  # another COLMAP build: new key
    monkeypatch.setattr(SC, "_code_version", lambda: "changed"); SC.scene_scan(ep1); assert ran(ep1)  # another scene_scan/geometry code
    moved = {**TWO, "exoB": TWO["exoB"] @ np.array([[1, 0, 0, 0.0], [0, 1, 0, 0], [0, 0, 1, 0.05], [0, 0, 0, 1]])}  # camera B moved 5 cm
    ep3, _ = _scan_episode(fake_colmap, "ep3", cams=moved); SC.scene_scan(ep3); assert ran(ep3)


def test_scene_scan_leaves_out_moving_and_nominal_cameras(fake_colmap):
    ep, _ = _scan_episode(fake_colmap, "sc10", cams=THREE, unregistered=("exoD",))
    p = ep.derived / "calib" / "calib.json"; cal = json.loads(p.read_text())
    cal["cameras"]["exoC"]["moving"] = True; cal["cameras"]["exoD"]["intrinsics_source"] = "nominal_fov"; p.write_text(json.dumps(cal))
    assert sorted(SC._cameras(ep, False)) == ["exoA", "exoB"] and sorted(SC._cameras(ep, False, allow_nominal=True)) == ["exoA", "exoB", "exoD"]
    assert sorted(SC._cameras(ep, True)) == ["exoA", "exoB"]


def test_scene_scan_force_through_run_stages_refreshes_the_cache(fake_colmap):
    from duet.playground import run as R
    # the real stage function as the only spec: this synthetic episode has no probe/align history for the full graph
    specs = {"scene_scan": R.StageSpec(SC.scene_scan, outputs=("scene_scan",), opt_in=True)}; go = lambda ep, **kw: R.run_stages(ep, ["scene_scan"], log=lambda *a: None, log_file=None, specs=specs, **kw)  # noqa: E731
    ep1, _ = _scan_episode(fake_colmap, "f1"); SC.scene_scan(ep1)  # fills the station cache
    ep2, _ = _scan_episode(fake_colmap, "f2"); (fake_colmap / "colmap_log.jsonl").write_text("")
    assert go(ep2) == {"scene_scan": "done"} and "cache hit" in ep2.status["scene_scan"]["detail"] and _colmap_log(fake_colmap) == []
    assert go(ep2, force=True) == {"scene_scan": "done"}  # run.py passes force=True: the cache is bypassed and refreshed
    assert "cache hit" not in ep2.status["scene_scan"]["detail"] and any(isinstance(e, list) for e in _colmap_log(fake_colmap))


def test_scene_scan_skips_with_one_camera_or_without_colmap(fake_colmap, monkeypatch):
    ep, _ = _scan_episode(fake_colmap, "sc8", cams={"exoA": TWO["exoA"]}, unregistered=("exoB",)); SC.scene_scan(ep)
    st = ep.status["scene_scan"]; assert st["state"] == "skipped" and "needs >= 2" in st["detail"] and not (ep.derived / "scene_scan").exists()
    monkeypatch.setenv("PATH", str(fake_colmap / "nothing")); SC.scene_scan(ep)
    assert ep.status["scene_scan"]["state"] == "skipped" and "not installed" in ep.status["scene_scan"]["detail"]


# ============================================================================ scene_scan: real COLMAP (only when installed)

BOXES = ((0.2, 0.1, 0.30, 0.20, 0.25), (0.75, 0.45, 0.22, 0.30, 0.35))  # x0, y0, size x, size y, height (m); board z = 0 is the table


def _scene_quads(rng):
    """Textured rectangles (board frame, metres; up = -z): the table and the faces of two boxes standing on it."""
    def tex(h, w):
        t = sum(cv2.resize(rng.random((max(2, h // s), max(2, w // s))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) for s in (6, 16, 48))
        return ((t - t.min()) / np.ptp(t) * 255).astype(np.uint8)
    out = [(np.array([[-0.35, -0.2, 0], [1.35, -0.2, 0], [1.35, 0.933, 0], [-0.35, 0.933, 0]]), tex(1200, 1800))]
    for x0, y0, sx, sy, hz in BOXES:
        x1, y1, zt = x0 + sx, y0 + sy, -hz
        for f in ([(x0, y0, zt), (x1, y0, zt), (x1, y1, zt), (x0, y1, zt)], [(x0, y0, 0), (x1, y0, 0), (x1, y0, zt), (x0, y0, zt)],
                  [(x0, y1, 0), (x1, y1, 0), (x1, y1, zt), (x0, y1, zt)], [(x0, y0, 0), (x0, y1, 0), (x0, y1, zt), (x0, y0, zt)],
                  [(x1, y0, 0), (x1, y1, 0), (x1, y1, zt), (x1, y0, zt)]):
            out.append((np.array(f, float), tex(240, 240)))
    return out


def _surface_dist(p):
    """Distance (m) from a board-frame point to the nearest scene surface (table top or box faces)."""
    x, y, z = p; c = [abs(z)] if -0.35 <= x <= 1.35 and -0.2 <= y <= 0.933 else []
    for x0, y0, sx, sy, hz in BOXES:
        x1, y1, zt = x0 + sx, y0 + sy, -hz
        if x0 <= x <= x1 and y0 <= y <= y1:
            c.append(abs(z - zt))
        if zt <= z <= 0:
            c += ([abs(x - x0), abs(x - x1)] if y0 <= y <= y1 else []) + ([abs(y - y0), abs(y - y1)] if x0 <= x <= x1 else [])
    return min(c, default=np.inf)


@pytest.mark.skipif(shutil.which("colmap") is None, reason="colmap not on PATH")
@pytest.mark.parametrize("method", ["triangulate", "mapper"])
def test_scene_scan_real_colmap(tmp_path, monkeypatch, method):
    """Real COLMAP on a textured table with two boxes (board frame) seen by 4 fixed cameras with known K, mild distortion
    and board poses (painter's-algorithm render, then distorted). Verified with COLMAP 3.11.1 and 4.2.0: points lie on the
    true surfaces to ~0.4 mm (triangulate) / ~0.6 mm (mapper + align_to_board) median."""
    monkeypatch.setenv("PLAYGROUND_CACHE", str(tmp_path / "cache"))
    quads = _scene_quads(np.random.default_rng(0)); dist = np.array([-0.08, 0.02, 0, 0, 0]); Kc = np.array([[420.0, 0, 319.5], [0, 420.0, 179.5], [0, 0, 1]])
    cams = {"exoA": look_at(np.array([-0.3, -0.4, -1.2]), [0.5, 0.4, 0]), "exoB": look_at(np.array([1.1, -0.3, -1.1]), [0.5, 0.4, 0]),
            "exoC": look_at(np.array([0.4, 1.4, -1.0]), [0.5, 0.4, 0]), "exoD": look_at(np.array([0.3, -0.5, -0.9]), [0.5, 0.4, 0])}
    ep = _episode(tmp_path, "real", [Stream(c, "exo", f"streams/{c}.mp4") for c in cams], 6)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    und = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), Kc, dist, P=Kc).reshape(H, W, 2)
    for c, T in cams.items():
        Tcw = np.linalg.inv(T); img = np.full((H, W), 255, np.uint8)
        for corners, tex in sorted(quads, key=lambda q: -np.linalg.norm(q[0].mean(0) - T[:3, 3])):  # far to near
            pc = corners @ Tcw[:3, :3].T + Tcw[:3, 3]; h, w = tex.shape
            Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32), ((pc @ Kc.T)[:, :2] / pc[:, 2:3]).astype(np.float32))
            img = np.where(cv2.warpPerspective(np.full_like(tex, 255), Hm, (W, H)) > 128, cv2.warpPerspective(tex, Hm, (W, H)), img)
        _write_frames(ep, c, [cv2.cvtColor(cv2.remap(img, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=255), cv2.COLOR_GRAY2BGR)] * 6)
    _write_calib(ep, {c: (G.Intrinsics(Kc, dist, W, H), "rig_json", T) for c, T in cams.items()})
    SC.scene_scan(ep, extra_frames=0, timeout_s=600, method=method)
    z = np.load(ep.derived / "scene_scan" / "points.npz", allow_pickle=False); scan = json.loads((ep.derived / "scene_scan" / "scan.json").read_text())
    assert bool(z["aligned"]) and scan["n_registered"] == 4 and len(z["xyz_board"]) > 1000, scan["alignment"]
    d = np.array([_surface_dist(p) for p in z["xyz_board"].astype(float)])
    assert np.median(d) < 0.003 and np.percentile(d, 90) < 0.01
