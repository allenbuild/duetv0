"""World-stage regression tests (review 2026-09-26): ZED axis convention and capture timeline, nearest depth frames,
cross-view body association with 3D identity tracking, simultaneous board registration, nominal-intrinsics gating,
T_tag_headcam validation, idle tag detection, world3d's O(n) runtime, and scripts/zed_export.py's control flow on a fake
pyzed.sl. Everything is synthetic (tmp_path); upstream stage states are set the way run.py would."""
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation  # noqa: E402

from duet.playground import geometry as G  # noqa: E402
from duet.playground import world as W  # noqa: E402
from duet.playground.episode import Stream  # noqa: E402
from test_playground_world_synthetic import (BC, K_TRUE, NH, NW, T_from, board_image, look_at, new_episode, render_plane, rig_json,  # noqa: E402
                                             rot, rot_err_deg, set_done, write_frames)

REPO = Path(__file__).resolve().parents[1]
C_GL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes <-> ZED RIGHT_HANDED_Y_UP camera axes


def write_calib(ep, cams: dict) -> None:
    """calib.json in the C7 layout, stage marked done (as calib would leave it)."""
    (ep.derived / "calib").mkdir(parents=True, exist_ok=True)
    (ep.derived / "calib" / "calib.json").write_text(json.dumps({"world": "board", "units": "m", "cameras": cams}))
    set_done(ep, "calib")


# ============================================================================= ZED: axes, capture timeline, depth

ZFPS = 30.0
K_ZED = G.Intrinsics(np.array([[700.0, 0, 639.5], [0, 700.0, 359.5], [0, 0, 1]]), np.zeros(5), 1280, 720)
K_ZFRAME = K_ZED.scaled_to(640, 360)
T_ZW_W = T_from(Rotation.from_rotvec([1.2, -0.4, 0.3]).as_matrix(), [0.5, -1.0, 2.0])  # ZED world <- board world (arbitrary)
T0_NS = 1_700_000_000_000_000_000
HAND_CAM = np.array([[0.05 + 0.02 * np.cos(a), 0.10 + 0.02 * np.sin(a), 0.42 + 0.01 * np.sin(2 * a)] for a in np.linspace(0, 2 * np.pi, 21, endpoint=False)])


def head_T(t: float) -> np.ndarray:
    """Head camera in the board world (OpenCV camera): at the table edge looking down at the table, turning its head."""
    p = [0.2 + 0.25 * np.sin(0.5 * t), -0.35 + 0.08 * np.cos(0.3 * t), -0.55 + 0.05 * np.sin(0.9 * t)]
    return look_at(p, [0.2 + 0.45 * np.sin(0.4 * t + 1.0), 0.25 + 0.15 * np.sin(0.7 * t), 0.0], up=(0, 0, -1))


def grab_times(n_grabs: int, drop: tuple = (), seed: int = 0, collide_at: int | None = None) -> np.ndarray:
    """Capture times (s) of the successful grabs: nominal 30 fps with +-2 ms jitter, the SVO frames in `drop` lost (their
    time passes, no grab), optionally one extra grab 8 ms after grab collide_at (it lands in an already written slot)."""
    rng = np.random.default_rng(seed); m = n_grabs + len(drop); t = np.arange(m) / ZFPS + rng.uniform(-0.002, 0.002, m)
    t = np.delete(t, list(drop)); t -= t[0]
    if collide_at is not None:
        t = np.sort(np.r_[t, t[collide_at] + 0.008])
    return t


def zed_export(zd: Path, t_grab: np.ndarray, fmt: int, depth_every: int = 1, states=None) -> np.ndarray:
    """zed/<s>/ as scripts/zed_export.py would write it: fmt 1 = the old exporter (RIGHT_HANDED_Y_UP poses, one video frame
    per grab), fmt 2 = the current one (IMAGE poses, capture-timeline video with filled gaps). -> capture time of the
    image shown in each video frame."""
    zd.mkdir(parents=True)
    vf = np.arange(len(t_grab)); shown = list(t_grab)
    if fmt == 2:
        vf = np.full(len(t_grab), -1); shown, last = [], -1
        for g, t in enumerate(t_grab):
            slot = int(round(t * ZFPS))
            if slot > last:
                shown += [shown[-1]] * (slot - last - 1) if shown else []
                shown.append(t); vf[g] = last = slot
    states = states or ["OK"] * len(t_grab)
    with open(zd / "pose.csv", "w") as f:
        f.write("frame_index,timestamp_ns," + ("video_frame," if fmt == 2 else "") + "tx,ty,tz,qx,qy,qz,qw,tracking_state\n")
        for g, t in enumerate(t_grab):
            T = T_ZW_W @ head_T(t) @ (C_GL if fmt == 1 else np.eye(4)); q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f"{g},{T0_NS + int(round(t * 1e9))}," + (f"{vf[g]}," if fmt == 2 else "") + f"{T[0, 3]},{T[1, 3]},{T[2, 3]},{q[0]},{q[1]},{q[2]},{q[3]},{states[g]}\n")
    exp = ({"svo": "x.svo2", "frames": len(t_grab), "fps": ZFPS, "depth_mode": "neural", "depth_every_n_frames": depth_every, "coordinate_system": W.ZED_LEGACY_CS}
           if fmt == 1 else {"format": 2, "fps": ZFPS, "coordinate_system": "IMAGE", "coordinate_units": "METER", "video_timeline": "capture",
                             "depth_every_n_frames": depth_every, "depth_units": "mm"})
    (zd / "export.json").write_text(json.dumps(exp))
    (zd / "calibration.json").write_text(json.dumps({"K": K_ZED.K.tolist(), "dist": [0.0] * 12, "width": 1280, "height": 720, "fps": ZFPS}))
    (zd / "depth").mkdir()
    uvz = HAND_CAM / HAND_CAM[:, 2:] @ K_ZED.K.T
    for g in range(0, len(t_grab), depth_every):
        d = np.zeros((720, 1280), np.uint16)
        for (u, v, _), z in zip(uvz, HAND_CAM[:, 2]):
            d[int(round(v)) - 3:int(round(v)) + 4, int(round(u)) - 3:int(round(u)) + 4] = int(round(z * 1000))
        cv2.imwrite(str(zd / "depth" / f"{g:06d}.png"), d)
    return np.array(shown)


def zed_episode(tmp_path, t_grab, fmt, depth_every=1, sight=(0, 5, 10), states=None, offset_status=None):
    """One ZED ego stream (plus a fixed camera as the reference when offset_status is given) with C4 frames at 10 fps
    that pick video frame round(k/10 * 30) of the CFR left.mp4, board sightings at frames `sight`."""
    n = int((len(t_grab) - 2) / ZFPS * 10)
    streams = [Stream("zed", "ego", "streams/zed.mp4", person="leader", fps=ZFPS, width=1280, height=720)]
    if offset_status:
        streams = [Stream("exo", "exo", "streams/exo.mp4", fps=ZFPS, width=1280, height=720), *streams]
    ep = new_episode(tmp_path, streams, n)
    if offset_status:
        ep.stream("zed").offset_status = offset_status; ep.save()
    video_t = zed_export(ep.dir / "zed" / "zed", t_grab, fmt, depth_every, states)
    src = np.minimum(np.round(np.arange(n) / 10 * ZFPS).astype(int), len(video_t) - 1)
    write_frames(ep, "zed", None, n=n, src_fps=ZFPS, src_frames=src)
    truth = [head_T(video_t[j]) for j in src]  # pose of the camera when the image of processed frame k was captured
    write_calib(ep, {"zed": {"role": "ego", "intrinsics": K_ZFRAME.to_json(), "intrinsics_source": "zed_factory", "registered": bool(sight), "T_world_cam": None,
                             "moving": True, "sightings": [{"k": k, "T_world_cam": truth[k].tolist(), "board": "observed", "err_px": 0.1} for k in sight]}})
    set_done(ep, "probe", "align", "frames")
    return ep, n, src, video_t, truth


def _errs(T, truth, ks):
    return (np.array([np.linalg.norm(T[k][:3, 3] - truth[k][:3, 3]) for k in ks]), np.array([rot_err_deg(T[k][:3, :3], truth[k][:3, :3]) for k in ks]))


def test_zed_capture_timeline_with_dropped_grabs(tmp_path):
    """Format 2 (IMAGE axes, capture timeline): every processed frame whose video frame shows a grab gets that grab's
    pose; filled slots (dropped grabs) are NaN, never a pose from another time."""
    t = grab_times(120, drop=(30, 31, 32, 57, 88), collide_at=40)  # video slots 30-32 (processed frame 10), 57, 88 are filled
    ep, _, src, _, truth = zed_episode(tmp_path, t, fmt=2)
    W.headpose(ep); assert ep.stage_ok("headpose"), ep.status["headpose"]
    hp = np.load(ep.derived / "headpose" / "zed.npz"); T = hp["T_world_cam"].astype(float)
    trk = W.zed_track(ep.dir / "zed" / "zed", W.zed_export_info(ep.dir / "zed" / "zed"))
    has_grab = np.isin(src, trk["video_frame"]); assert (~has_grab).sum() >= 1  # some frames land on filled slots
    np.testing.assert_array_equal(hp["valid"], has_grab)
    pe, re = _errs(T, truth, np.nonzero(has_grab)[0]); assert pe.max() < 1e-5 and re.max() < 1e-3, (pe.max(), re.max())
    assert str(hp["backend"]) == "zed_tracking+board" and "IMAGE -> OpenCV axes" in ep.status["headpose"]["detail"]


def test_zed_legacy_y_up_export_converted_and_drift_rejected(tmp_path):
    """Format 1 (old exporter: RIGHT_HANDED_Y_UP, video frame = grab index). Poses must be converted to OpenCV camera
    axes (the old code mirrored yaw/roll: 18 cm / 86 deg median); frames whose video time drifted from capture time by
    dropped grabs are rejected instead of taking a pose (and an image) from another time."""
    t = grab_times(120, drop=(47, 90), seed=1)  # 2 of 122 SVO frames lost: from grab 47 on the video runs 33 ms behind capture
    ep, n, src, _, truth = zed_episode(tmp_path, t, fmt=1)
    W.headpose(ep); assert ep.stage_ok("headpose"), ep.status["headpose"]
    hp = np.load(ep.derived / "headpose" / "zed.npz"); T = hp["T_world_cam"].astype(float)
    on_time = np.abs(src / ZFPS - t[src]) <= 0.5 / ZFPS + 1e-3
    np.testing.assert_array_equal(hp["valid"], on_time); assert 0 < on_time.sum() < n
    pe, re = _errs(T, truth, np.nonzero(on_time)[0]); assert pe.max() < 1e-5 and re.max() < 1e-3, (pe.max(), re.max())
    assert "legacy" in ep.status["headpose"]["detail"] and "off the capture timeline" in ep.status["headpose"]["detail"]


def test_zed_tracking_state_and_quaternion_checks(tmp_path):
    t = grab_times(90); states = ["OK"] * len(t); states[30:36] = ["SEARCHING"] * 6
    ep, _, src, _, _ = zed_episode(tmp_path, t, fmt=2, states=states)
    W.headpose(ep); hp = np.load(ep.derived / "headpose" / "zed.npz")
    bad = np.isin(src, np.arange(30, 36)); assert bad.any() and not hp["valid"][bad].any() and hp["valid"][~bad].all()


@pytest.mark.parametrize("cs", ["LEFT_HANDED_Y_UP", None])
def test_zed_unknown_convention_fails(tmp_path, cs):
    ep, *_ = zed_episode(tmp_path, grab_times(60), fmt=2)
    p = ep.dir / "zed" / "zed" / "export.json"; e = json.loads(p.read_text()); e["coordinate_system"] = cs; p.write_text(json.dumps(e))
    with pytest.raises(ValueError, match="coordinate_system"):
        W.headpose(ep)


def test_zed_unregistered_tracking_never_enters_the_board_frame(tmp_path):
    ep, *_ = zed_episode(tmp_path, grab_times(60), fmt=2, sight=())
    W.headpose(ep)
    assert ep.status["headpose"]["state"] == "skipped" and "zed_tracking_unregistered" in ep.status["headpose"]["detail"]
    assert not (ep.derived / "headpose").exists() or not list((ep.derived / "headpose").iterdir())


def test_unaligned_ego_is_skipped(tmp_path):
    ep, *_ = zed_episode(tmp_path, grab_times(60), fmt=2, offset_status="unaligned")
    assert not ep.usable("zed")
    W.headpose(ep)
    assert ep.status["headpose"]["state"] == "skipped" and "offset unaligned" in ep.status["headpose"]["detail"] and "--offset zed=" in ep.status["headpose"]["detail"]


def test_zed_hands_use_only_their_own_grabs_depth(tmp_path):
    """zed_export --fps 15 writes depth for every 2nd grab: a frame gets ZED hands only from the depth of its OWN grab
    (the depth of a neighbouring grab would be sampled at pixels from another instant); the others stay NaN."""
    t = grab_times(120, drop=(50,))
    ep, n, src, _, truth = zed_episode(tmp_path, t, fmt=2, depth_every=2)
    W.headpose(ep)
    uv = G.project(K_ZFRAME, np.eye(4), HAND_CAM); lm = np.full((n, 2, 21, 2), np.nan, np.float32); lm[:, 1] = uv; present = np.zeros((n, 2), bool); present[:, 1] = True
    (ep.derived / "hands").mkdir(); np.savez_compressed(ep.derived / "hands" / "zed.npz", lm2d=lm, lm3d=np.zeros((n, 2, 21, 3), np.float32), score=present.astype(np.float32),
                                                        present=present, schema=2)
    set_done(ep, "hands", "body2d"); W.world3d(ep); assert ep.stage_ok("world3d"), ep.status["world3d"]
    h = np.load(ep.derived / "world3d" / "world3d.npz")["hands3d_zed"].astype(float); have = np.isfinite(h[:, 1, 0, 0])
    trk = W.zed_track(ep.dir / "zed" / "zed", W.zed_export_info(ep.dir / "zed" / "zed"))
    grab = {int(v): g for g, v in enumerate(trk["video_frame"]) if v >= 0}; own = np.array([grab.get(int(j), -1) % 2 == 0 and int(j) in grab for j in src])
    np.testing.assert_array_equal(have, own); assert 0 < have.sum() < n and not np.isfinite(h[:, 0]).any()
    err = np.array([np.linalg.norm(h[k, 1] - G.apply(truth[k], HAND_CAM), axis=1).max() for k in np.nonzero(have)[0]]); assert err.max() < 0.005, err.max()


def test_zed_export_from_another_take_is_refused(tmp_path):
    """zed/<s> left over from a previous take (different video) must not be used for this stream's video."""
    ep, *_ = zed_episode(tmp_path, grab_times(60), fmt=2)
    s = ep.stream("zed"); e = json.loads((ep.dir / "zed" / "zed" / "export.json").read_text())
    e["video"] = {"frames": 60, "fps": ZFPS}; (ep.dir / "zed" / "zed" / "export.json").write_text(json.dumps(e))
    s.nb_frames = 60; ep.save(); W.headpose(ep); assert ep.stage_ok("headpose")  # consistent: accepted
    s.nb_frames = 75; ep.save()
    with pytest.raises(ValueError, match="does not belong to stream zed"):
        W.headpose(ep)
    s.nb_frames = 60; e["video"].update(size=123, sha256_head_tail="0" * 64); (ep.dir / "zed" / "zed" / "export.json").write_text(json.dumps(e))
    (ep.dir / "streams" / "zed.mp4").write_bytes(b"x" * 1000); ep.save()
    with pytest.raises(ValueError, match="is not the export's left.mp4"):
        W.headpose(ep)


def test_zed_registration_uses_the_pose_fit(tmp_path):
    """>= 3 sightings: T_world_zedworld comes from duet.geometry.alignment.fit_pose_alignment (camera centres and
    orientations); with 5 mm / 0.3 deg sighting noise, cameras 2-4 m from the ZED origin, the camera positions come out
    within a few mm (the per-sighting mean it replaces: about 9 mm median, 13 mm max)."""
    rng = np.random.default_rng(0); worst = []
    for trial in range(10):
        T_true = T_from(Rotation.random(random_state=trial).as_matrix(), rng.normal(0, 2, 3)); src, dst = [], []
        for i in range(12):
            d = rng.normal(0, 1, 3); d *= rng.uniform(2, 4) / np.linalg.norm(d)
            S = T_from(Rotation.random(random_state=100 * trial + i).as_matrix(), d); D = T_true @ S
            src.append(S); dst.append(T_from(Rotation.from_rotvec(rng.normal(0, np.radians(0.3), 3)).as_matrix() @ D[:3, :3], D[:3, 3] + rng.normal(0, 0.005, 3)))
        T, inl, _ = W._zed_registration(src, dst); assert inl.all()
        worst.append(max(np.linalg.norm((T @ S)[:3, 3] - (T_true @ S)[:3, 3]) for S in src))
    assert np.median(worst) < 0.008 and max(worst) < 0.012, worst


def test_zed_registered_through_calib(tmp_path):
    """End to end: calib finds the ZED ego's board sightings (rendered views, simultaneous with a registered fixed
    camera), headpose registers the tracking with them; the result matches the true head trajectory."""
    t = grab_times(60)
    streams = [Stream("exoA", "exo", "streams/exoA.mp4", fps=10.0, width=NW, height=NH), Stream("zed", "ego", "streams/zed.mp4", person="leader", fps=ZFPS, width=1280, height=720)]
    n = 15; ep = new_episode(tmp_path, streams, n)
    video_t = zed_export(ep.dir / "zed" / "zed", t, 2); src = np.round(np.arange(n) / 10 * ZFPS).astype(int); truth = [head_T(video_t[j]) for j in src]
    board, ppm = board_image()
    write_frames(ep, "exoA", [render_plane(board, ppm, G.inv(T_WC["exoA"]), K_TRUE) for _ in range(n)])
    write_frames(ep, "zed", [render_plane(board, ppm, G.inv(T), K_ZED.K) for T in truth], src_fps=ZFPS, src_frames=src)
    rig_json(ep, calib_hz=10.0, intrinsics={"exoA": G.Intrinsics(K_TRUE, np.zeros(5), NW, NH).to_json()})
    set_done(ep, "probe", "align", "frames"); W.calib(ep); assert ep.stage_ok("calib"), ep.status["calib"]
    cam = W.calib_report(ep)["cameras"]["zed"]
    assert cam["intrinsics_source"] == "zed_factory" and len(cam["sightings"]) >= 5 and {x["board"] for x in cam["sightings"]} == {"observed"}
    W.headpose(ep); hp = np.load(ep.derived / "headpose" / "zed.npz"); T = hp["T_world_cam"].astype(float)
    assert hp["valid"].all(), ep.status["headpose"]
    pe, re = _errs(T, truth, range(n)); assert np.median(pe) < 0.01 and np.median(re) < 0.5, (pe, re)


def test_ego_only_rig_assumes_a_static_board(tmp_path):
    """Two head cameras and no fixed camera (the paired-egocentric case): both register to the board where most of them
    see it at once, flagged as assuming the board did not move."""
    board, ppm = board_image(); n = 10
    poses = {e: [look_at([BC[0] + dx + 0.03 * k, BC[1] - 0.35, -0.6], [BC[0], BC[1], 0]) for k in range(n)] for e, dx in (("leader", -0.25), ("helper", 0.25))}
    streams = [Stream(e, "ego", f"streams/{e}.mp4", person=e, fps=10.0, width=NW, height=NH) for e in poses]
    ep = new_episode(tmp_path, streams, n)
    for e, Ts in poses.items():
        write_frames(ep, e, [render_plane(board, ppm, G.inv(T), K_TRUE) if (e == "leader" or k >= 4) else np.full((NH, NW), 255, np.uint8) for k, T in enumerate(Ts)])
    rig_json(ep, calib_hz=10.0, intrinsics={e: G.Intrinsics(K_TRUE, np.zeros(5), NW, NH).to_json() for e in poses})
    set_done(ep, "probe", "align", "frames"); W.calib(ep); assert ep.stage_ok("calib"), ep.status["calib"]
    rep = W.calib_report(ep); assert rep["board_assumed_static"] and "assume the board did not move" in ep.status["calib"]["detail"] and rep["anchor_frame"] == 4
    for e, Ts in poses.items():
        sg = rep["cameras"][e]["sightings"]; assert sg and {x["board"] for x in sg} == {"assumed_static"}
        for x in sg:
            X = np.array(x["T_world_cam"]); assert np.linalg.norm(X[:3, 3] - Ts[x["k"]][:3, 3]) < 0.02 and rot_err_deg(X[:3, :3], Ts[x["k"]][:3, :3]) < 1.0


# ============================================================================= cross-view body association

INTR = G.Intrinsics(np.array([[420.0, 0, 319.5], [0, 420.0, 179.5], [0, 0, 1]]), np.zeros(5), 640, 360)
BASE = {0: (0, 0.05, 1.62), 1: (0.03, 0.06, 1.66), 2: (-0.03, 0.06, 1.66), 3: (0.07, 0.0, 1.64), 4: (-0.07, 0.0, 1.64), 5: (0.19, 0, 1.42), 6: (-0.19, 0, 1.42),
        7: (0.24, 0.10, 1.15), 8: (-0.24, 0.10, 1.15), 9: (0.18, 0.35, 1.0), 10: (-0.18, 0.35, 1.0), 11: (0.11, 0, 0.95), 12: (-0.11, 0, 0.95),
        13: (0.11, 0.02, 0.5), 14: (-0.11, 0.02, 0.5), 15: (0.11, 0, 0.05), 16: (-0.11, 0, 0.05)}  # COCO-17: lateral, forward, up (m)


def skeleton(x: float, y: float, facing: int = 1) -> np.ndarray:
    """17x3 world keypoints of a person standing on the floor 0.9 m below the counter-top board (world up = -Z) at (x, y),
    facing +Y (facing=1) or -Y. Anatomical: facing +Y with up -Z the person's left is +X."""
    P = np.array([BASE[j] for j in range(17)], float)
    return np.stack([x + P[:, 0] * facing, y + P[:, 1] * facing, 0.9 - P[:, 2]], 1)


def assoc_run(tmp_path, cams: dict, people, n: int, occlude=None, egos: dict | None = None, slot_of=None, mislabel=()):
    """world3d on exact projections of people(k) -> [(x, y), ...] into registered fixed cameras. occlude: {(cam, person):
    frames}. egos: {ego name: person index, or None = a head camera nowhere near either person}. mislabel: cameras whose detector swaps
    left and right. Returns (bodies, truth, world3d arrays + "dt" = world3d seconds)."""
    names = list(cams); egos = egos or {}; w = {}
    streams = [Stream(nm, "exo", f"streams/{nm}.mp4", fps=10.0, width=640, height=360) for nm in names]
    streams += [Stream(e, "ego", f"streams/{e}.mp4", person=e, fps=10.0, width=640, height=360) for e in egos]
    ep = new_episode(tmp_path, streams, n)
    write_calib(ep, {nm: {"role": "exo", "intrinsics": INTR.to_json(), "intrinsics_source": "rig_json", "registered": True, "T_world_cam": T.tolist()} for nm, T in cams.items()}
                | {e: {"role": "ego", "intrinsics": INTR.to_json(), "registered": True, "T_world_cam": None, "moving": True, "sightings": []} for e in egos})
    truth = np.array([[skeleton(*xy) for xy in people(k)] for k in range(n)])
    (ep.derived / "body2d").mkdir(parents=True)
    for nm, T in cams.items():
        kp = np.full((n, 4, 17, 3), np.nan, np.float32)
        for k in range(n):
            for p in range(truth.shape[1]):
                if occlude and k in occlude.get((nm, p), ()):
                    continue
                uv = G.project(INTR, G.inv(T), truth[k, p]); vis = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 360)
                s = p if slot_of is None else slot_of(nm, k, p); kp[k, s, :, :2] = uv; kp[k, s, :, 2] = np.where(vis, 0.9, 0.05)
                if nm in mislabel:
                    kp[k, s] = kp[k, s][W.FLIP]
        np.savez_compressed(ep.derived / "body2d" / f"{nm}.npz", kpts=kp, boxes=np.full((n, 4, 5), np.nan, np.float32), img_w=640, img_h=360, schema=2)
    set_done(ep, "probe", "align", "frames", "body2d")
    if egos:
        (ep.derived / "headpose").mkdir()
        for e, p in egos.items():
            head = [truth[k, p, 0] + [0, 0, -0.08] if p is not None else np.array([3.0, 3.0, -1.5]) for k in range(n)]
            T = np.stack([look_at(head[k], head[k] + [0, 1.0, 0.5], up=(0, 0, -1)) for k in range(n)])
            np.savez_compressed(ep.derived / "headpose" / f"{e}.npz", T_world_cam=T.astype(np.float32), valid=np.ones(n, bool), backend="head_tag")
        set_done(ep, "headpose")
    t0 = time.perf_counter(); W.world3d(ep); w["dt"] = time.perf_counter() - t0
    assert ep.stage_ok("world3d"), ep.status["world3d"]
    w.update(np.load(ep.derived / "world3d" / "world3d.npz"))
    return w["bodies"].astype(float), truth, w


def score(bodies, truth):
    """-> (identity [n,2] of the true person each output slot holds (-1 empty), error [n,2] m (median over keypoints))."""
    n = len(bodies); ident = np.full((n, 2), -1); err = np.full((n, 2), np.nan)
    for k in range(n):
        for s in range(2):
            if np.isfinite(bodies[k, s]).any():
                d = [np.nanmedian(np.linalg.norm(bodies[k, s] - truth[k, p], axis=1)) for p in range(truth.shape[1])]
                ident[k, s], err[k, s] = int(np.argmin(d)), min(d)
    return ident, err


SIDE_BY_SIDE = lambda k: [(-0.35, -0.55), (0.35, -0.55)]  # both at the near side of the counter, facing it
FRONT_L, FRONT_R = look_at([-1.0, 1.4, -0.9], [0, -0.5, 0], up=(0, 0, -1)), look_at([1.0, 1.4, -0.9], [0, -0.5, 0], up=(0, 0, -1))


def test_front_back_cameras_see_people_in_mirrored_order(tmp_path):
    """CoMind layout: one camera across the counter, one behind the people. Left-right hip order is mirrored between the
    views; the old code triangulated 0 % of bodies."""
    cams = {"front": look_at([0.0, 1.6, -0.9], [0.0, -0.5, 0.0], up=(0, 0, -1)), "back": look_at([0.0, -2.8, -0.9], [0.0, -0.4, 0.0], up=(0, 0, -1))}
    bodies, truth, _ = assoc_run(tmp_path, cams, SIDE_BY_SIDE, 30)
    ident, err = score(bodies, truth)
    assert (ident >= 0).all() and np.nanmax(err) < 0.02 and (ident[:, 0] != ident[:, 1]).all(), (ident, np.nanmax(err))
    assert len(set(ident[:, 0])) == 1  # one identity per slot


def test_occluded_person_gives_no_ghost(tmp_path):
    """Person 0 missing in one view for 2 s: the old code paired the other person's detection with it (a ghost 1.17 m
    away at 0.0 px). Now person 0 is simply not triangulated then, and keeps its slot afterwards."""
    occ = set(range(10, 30))
    bodies, truth, _ = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, SIDE_BY_SIDE, 40, occlude={("frontR", 0): occ})
    ident, err = score(bodies, truth)
    assert np.nanmax(err) < 0.02, np.nanmax(err)  # every triangulated body is a real person
    s0 = int(np.nonzero(ident[0] == 0)[0][0])
    assert (ident[sorted(occ), s0] == -1).all() and (ident[[k for k in range(40) if k not in occ], s0] == 0).all()
    assert (ident[:, 1 - s0] == 1).all()


def test_people_crossing_keep_their_identity(tmp_path):
    """Two people walk past each other (one nearer the counter): left-right order swaps mid-sequence, identity must not."""
    n = 40
    people = lambda k: [(-0.6 + 1.2 * k / (n - 1), -0.45), (0.6 - 1.2 * k / (n - 1), -0.95)]
    bodies, truth, w = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, people, n)
    ident, err = score(bodies, truth)
    assert np.nanmax(err) < 0.02 and (ident >= 0).mean() > 0.95
    for s in (0, 1):
        assert len(set(ident[ident[:, s] >= 0, s])) == 1, ident[:, s]
    assert len(set(w["body_track"][:, 0][w["body_track"][:, 0] >= 0])) == 1


def test_side_camera_people_at_different_depths(tmp_path):
    """Front + side (90 deg) camera: the old code accepted 41 % of keypoints at 0.44 m error under its 25 px gate."""
    side = look_at([2.6, -0.5, -0.9], [0, -0.5, 0], up=(0, 0, -1))
    bodies, truth, _ = assoc_run(tmp_path, {"frontL": FRONT_L, "side": side}, lambda k: [(-0.35, -0.75), (0.35, -0.35)], 30)
    ident, err = score(bodies, truth)
    assert (ident >= 0).all() and np.nanmax(err) < 0.02 and (ident[:, 0] != ident[:, 1]).all()


def test_view_with_left_right_mislabelled(tmp_path):
    """Detectors often swap left/right for people seen from behind: the back camera's labels are mirrored here. People are
    still associated and triangulated, and the output keeps anatomical labels whichever camera is listed first."""
    front, back = look_at([0.0, 1.6, -0.9], [0.0, -0.5, 0.0], up=(0, 0, -1)), look_at([0.0, -2.8, -0.9], [0.0, -0.4, 0.0], up=(0, 0, -1))
    for i, cams in enumerate(({"front": front, "back": back}, {"back": back, "front": front})):
        bodies, truth, _ = assoc_run(tmp_path / str(i), cams, SIDE_BY_SIDE, 20, mislabel=("back",))
        ident, err = score(bodies, truth)
        assert (ident >= 0).all() and np.nanmax(err) < 0.02, (list(cams), np.nanmax(err))
        lw = np.array([np.linalg.norm(bodies[k, s, 9] - truth[k, ident[k, s], 9]) for k in range(20) for s in (0, 1)])
        assert np.nanmax(lw) < 0.02  # the left wrist is the anatomical left wrist


def test_seed_does_not_take_another_persons_detection(tmp_path):
    """Reviewer's case: P2 is seen only by camera frontL, close to P1's ray. When P1 is seeded from (frontL, frontR), the
    follow-up search for more views must not replace P1's frontL detection with P2's (P1 came out 9-23 cm off for the
    whole track)."""
    Ca = FRONT_L[:3, 3]; P1 = np.array([-0.35, -0.55]); d = P1 - Ca[:2]; P2 = Ca[:2] + 1.3 * d + 0.2 * np.array([-d[1], d[0]]) / np.linalg.norm(d)
    bodies, truth, _ = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, lambda k: [tuple(P1), tuple(P2)], 20, occlude={("frontR", 1): set(range(20))})
    ident, err = score(bodies, truth)
    assert (ident[ident >= 0] == 0).all() and (ident >= 0).sum() == 20 and np.nanmax(err) < 0.005, (ident, np.nanmax(err))


def test_slots_follow_the_ego_wearers(tmp_path):
    """With head poses, slot p is the wearer of ep.egos()[p] (here person 1 wears 'leader'), not detection order."""
    bodies, truth, w = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, SIDE_BY_SIDE, 30, egos={"leader": 1, "helper": 0})
    ident, err = score(bodies, truth)
    assert list(w["body_person"]) == ["leader", "helper"] and (ident[:, 0] == 1).all() and (ident[:, 1] == 0).all() and np.nanmax(err) < 0.02
    assert "feat_head_dist_m" in w and np.isfinite(w["feat_min_wrist_dist_m"]).all() and "T_world_cam_leader" in w
    assert {"feat_leader_facing_partner_cos", "feat_helper_facing_partner_cos"} <= set(w)  # stable feature keys (export schema)


def test_unidentified_wearer_does_not_reserve_a_slot(tmp_path):
    """An ego whose head never comes near a tracked person (e.g. the wearer's head is never triangulated) must not keep
    a slot empty: both people are output, that slot is labelled "" (not linked)."""
    bodies, truth, w = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, SIDE_BY_SIDE, 20, egos={"leader": None, "helper": 0})
    ident, err = score(bodies, truth)
    assert list(w["body_person"]) == ["", "helper"] and (ident[:, 1] == 0).all() and (ident[:, 0] == 1).all() and np.nanmax(err) < 0.02


@pytest.mark.parametrize("allow", [False, True])
def test_nominal_ego_is_not_a_body_view(tmp_path, allow):
    """One fixed camera + a head camera with a head pose: the ego joins the body views only with real intrinsics (its
    nominal-FOV guess gave 4-11 cm median, 47 cm max errors), unless rig.json allows nominal intrinsics."""
    n = 10; side = look_at([2.6, -0.5, -0.9], [0, -0.5, 0], up=(0, 0, -1)); ego = look_at([-0.2, 1.2, -1.5], [0, -0.55, -0.9], up=(0, 0, -1))
    ep = new_episode(tmp_path, [Stream("a", "exo", "streams/a.mp4"), Stream("head", "ego", "streams/head.mp4", person="leader")], n); rig_json(ep, allow_nominal=allow)
    write_calib(ep, {"a": {"role": "exo", "intrinsics": INTR.to_json(), "intrinsics_source": "rig_json", "registered": True, "T_world_cam": side.tolist()},
                     "head": {"role": "ego", "intrinsics": INTR.to_json(), "intrinsics_source": "nominal_fov", "registered": True, "T_world_cam": None, "moving": True, "sightings": []}})
    truth = np.array([[skeleton(*xy) for xy in SIDE_BY_SIDE(k)] for k in range(n)]); (ep.derived / "body2d").mkdir(); (ep.derived / "headpose").mkdir()
    for nm, T in (("a", side), ("head", ego)):
        kp = np.full((n, 4, 17, 3), np.nan, np.float32)
        for k in range(n):
            for p in range(2):
                kp[k, p, :, :2] = G.project(INTR, G.inv(T), truth[k, p]); kp[k, p, :, 2] = 0.9
        np.savez_compressed(ep.derived / "body2d" / f"{nm}.npz", kpts=kp, img_w=640, img_h=360)
    np.savez_compressed(ep.derived / "headpose" / "head.npz", T_world_cam=np.repeat(ego[None], n, 0).astype(np.float32), valid=np.ones(n, bool), backend="head_tag")
    set_done(ep, "probe", "align", "frames", "body2d", "headpose"); W.world3d(ep)
    b = np.load(ep.derived / "world3d" / "world3d.npz")["bodies"]
    assert np.isfinite(b).any() == allow and ("nominal-FOV" in ep.status["world3d"]["detail"]) != allow


def test_world3d_runtime_is_linear(tmp_path):
    """5k frames x 2 cameras x 2 people: arrays are loaded once (the old per-frame NpzFile access was O(n^2): 21 s at 2.25k
    frames, 296 s at 9k)."""
    n = 5000
    people = lambda k: [(-0.35 + 0.1 * np.sin(k / 50), -0.55), (0.35, -0.55 + 0.1 * np.cos(k / 70))]
    bodies, truth, w = assoc_run(tmp_path, {"frontL": FRONT_L, "frontR": FRONT_R}, people, n)
    ident, err = score(bodies[::50], truth[::50]); assert np.nanmax(err) < 0.02 and (ident >= 0).all()
    assert w["dt"] < 30, f"world3d on {n} frames took {w['dt']:.1f} s (about 3.5 s on an M-series Mac)"


# ============================================================================= board registration (rendered views)

T_WC = {"exoA": look_at([BC[0] - 0.5, BC[1] - 0.3, -1.1], [*BC, 0]), "exoB": look_at([BC[0] + 0.6, BC[1] - 0.25, -1.0], [*BC, 0])}
BOARD_MOVE = T_from(rot("z", 10), [0.10, 0.03, 0.0])


def calib_run(tmp_path, n, board_T, cam_T, hide=lambda name, k: False, intrinsics=True, **rig_extra):
    """calib on rendered 1280x720 board views (extracted to 640x360, no source video): board_T(k) = board pose in the
    initial board frame, cam_T(name, k) = camera pose in that frame."""
    board, ppm = board_image()
    streams = [Stream(nm, "exo", f"streams/{nm}.mp4", fps=10.0, width=NW, height=NH) for nm in T_WC]
    ep = new_episode(tmp_path, streams, n)
    for nm in T_WC:
        write_frames(ep, nm, [np.full((NH, NW), 255, np.uint8) if hide(nm, k) else render_plane(board, ppm, G.inv(cam_T(nm, k)) @ board_T(k), K_TRUE) for k in range(n)])
    rig_json(ep, calib_hz=10.0, **({"intrinsics": {nm: G.Intrinsics(K_TRUE, np.zeros(5), NW, NH).to_json() for nm in T_WC}} if intrinsics else {}), **rig_extra)
    set_done(ep, "probe", "align", "frames"); W.calib(ep)
    return ep


def test_board_shown_in_turn_registers_in_one_frame(tmp_path):
    """exoB first sees the board after it moved 10 cm / 10 deg. The old code registered each camera to its own first
    sighting (20 cm / 10 deg error for exoB). Now world = the board where both see it at once."""
    ep = calib_run(tmp_path, 12, lambda k: BOARD_MOVE if k >= 5 else np.eye(4), lambda nm, k: T_WC[nm], hide=lambda nm, k: nm == "exoB" and k < 5)
    assert ep.stage_ok("calib"), ep.status["calib"]
    rep = W.calib_report(ep); assert rep["anchor_frame"] == 5
    for nm in T_WC:
        X = np.array(rep["cameras"][nm]["T_world_cam"]); truth = G.inv(BOARD_MOVE) @ T_WC[nm]  # world = board at the anchor
        assert np.linalg.norm(X[:3, 3] - truth[:3, 3]) < 0.02 and rot_err_deg(X[:3, :3], truth[:3, :3]) < 1.0, (nm, X, truth)
    A, B = (np.array(rep["cameras"][nm]["T_world_cam"]) for nm in T_WC)
    rel, rel_true = G.inv(A) @ B, G.inv(T_WC["exoA"]) @ T_WC["exoB"]; assert np.linalg.norm(rel[:3, 3] - rel_true[:3, 3]) < 0.02


def test_camera_bumped_during_calibration_is_unregistered(tmp_path):
    bump = T_from(rot("y", 3), [0.05, 0.0, 0.0])
    ep = calib_run(tmp_path, 20, lambda k: np.eye(4), lambda nm, k: T_WC[nm] @ bump if nm == "exoB" and k >= 8 else T_WC[nm])
    rep = W.calib_report(ep); cams = rep["cameras"]
    assert cams["exoA"]["registered"] and not cams["exoB"]["registered"] and "moved during calibration" in cams["exoB"]["registration"]["reason"]
    X = np.array(cams["exoA"]["T_world_cam"]); assert np.linalg.norm(X[:3, 3] - T_WC["exoA"][:3, 3]) < 0.02


def test_nominal_intrinsics_are_not_registered_unless_allowed(tmp_path):
    ep = calib_run(tmp_path / "a", 6, lambda k: np.eye(4), lambda nm, k: T_WC[nm], intrinsics=False)
    assert ep.status["calib"]["state"] == "skipped" and "nominal-FOV" in ep.status["calib"]["detail"] and not (ep.derived / "calib").exists()
    ep = calib_run(tmp_path / "b", 6, lambda k: np.eye(4), lambda nm, k: T_WC[nm], intrinsics=False, allow_nominal=True, hfov_deg={"exoA": 93.7, "exoB": 93.7})
    rep = W.calib_report(ep); assert ep.stage_ok("calib") and all(c["registered"] and c["intrinsics_source"] == "nominal_fov" for c in rep["cameras"].values())


@pytest.mark.parametrize("rotation,flips", [(90, []), (-90, []), (180, []), (0, ["-display_hflip:v:0"]), (90, ["-display_hflip:v:0"]),
                                             (0, ["-display_vflip:v:0"])])
def test_native_frames_follow_display_rotation(tmp_path, rotation, flips):
    """Native decodes (PyAV does not autorotate) must show the same picture as ffmpeg's autorotated extracted frames, with
    Stream.rotation / Stream.flip as episode.ffprobe records them (mirrored display matrices included)."""
    av = pytest.importorskip("av")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    from fractions import Fraction

    from duet.playground.episode import ffprobe
    img = np.full((360, 640), 255, np.uint8); cv2.rectangle(img, (20, 20), (200, 120), 0, -1); cv2.circle(img, (560, 300), 40, 90, -1)
    with av.open(str(tmp_path / "a.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=10); st.width, st.height, st.pix_fmt = 640, 360, "yuv420p"
        for i in range(3):
            f = av.VideoFrame.from_ndarray(np.repeat(img[..., None], 3, -1), format="rgb24"); f.pts, f.time_base = i, Fraction(1, 10)
            for q in st.encode(f):
                c.mux(q)
        for q in st.encode():
            c.mux(q)
    out = tmp_path / "r.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation:v:0", str(rotation), *flips, "-i", str(tmp_path / "a.mp4"), "-c", "copy", str(out)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-frames:v", "1", str(tmp_path / "x.png")], check=True)  # autorotated
    info = ffprobe(out); ext = cv2.imread(str(tmp_path / "x.png"), cv2.IMREAD_GRAYSCALE)
    s = Stream("r", "exo", "r.mp4", fps=info["fps"], width=info["width"], height=info["height"], video_start_s=info["video_start_s"],
               rotation=info["rotation"], video_index=info["video_index"], flip=bool(info.get("flip", False)))
    nat = W._Native(type("E", (), {"dir": tmp_path})(), s, 4000); g = nat.get(0.0); nat.close()
    assert g.shape == ext.shape == (info["height"], info["width"]) and W._same_view(g, ext)
    assert not W._same_view(np.ascontiguousarray(np.rot90(g, 2)), ext)  # the check is sensitive to orientation


@pytest.mark.parametrize("display", [["-display_hflip:v:0"], ["-display_rotation:v:0", "90", "-display_hflip:v:0"]])
def test_native_frames_match_extracted_for_mirrored_sources(tmp_path, display):
    """Mirrored sources (hflip, rotate 90 + hflip): episode.probe + perception.extract_frames, then every native decode
    (Stream.flip then Stream.rotation) must agree with its extracted frame (NCC check), so detection stays native."""
    av = pytest.importorskip("av")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not installed")
    from fractions import Fraction

    from duet.playground import perception as P
    from duet.playground.episode import Episode, probe
    with av.open(str(tmp_path / "a.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=10); st.width, st.height, st.pix_fmt = 640, 360, "yuv420p"
        for i in range(6):
            img = np.full((360, 640), 255, np.uint8); cv2.rectangle(img, (20 + 10 * i, 20), (200, 120), 0, -1); cv2.circle(img, (560, 300), 40, 90, -1)
            f = av.VideoFrame.from_ndarray(np.repeat(img[..., None], 3, -1), format="rgb24"); f.pts, f.time_base = i, Fraction(1, 10)
            for q in st.encode(f):
                c.mux(q)
        for q in st.encode():
            c.mux(q)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *display, "-i", str(tmp_path / "a.mp4"), "-c", "copy", str(tmp_path / "m.mp4")], check=True)
    ep = Episode.create(tmp_path / "eps", "mir", [("cam", "exo", tmp_path / "m.mp4", None)], mode="copy", proc_size=320)
    probe(ep); ep.reload(); s = ep.stream("cam"); assert s.flip
    ep.common_start_s, ep.common_end_s = 0.0, 0.5; ep.save(); P.extract_frames(ep)
    paths, idx = P.frame_paths(ep, s), P.frame_index(ep, s); note = []
    out = list(W._iter_views(ep, s, range(len(paths)), paths, idx, True, 1920, note))
    assert not note and all(nat for _, _, nat in out)
    assert all(W._same_view(g, cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE)) for k, g, _ in out)


def test_native_repeated_source_time_reuses_the_frame(tmp_path):
    """With proc_fps >= the source fps the frames index repeats t_src_s; the native reader must return the same frame
    again (it used to return None, count a mismatch and switch native decoding off)."""
    pytest.importorskip("av")
    from test_playground_world_synthetic import write_video
    frames = [np.full((360, 640), 40 + 40 * i, np.uint8) for i in range(4)]; [cv2.putText(f, str(i), (300, 200), 0, 3, 255, 5) for i, f in enumerate(frames)]
    ep = new_episode(tmp_path, [Stream("a", "exo", "streams/a.mp4", fps=5.0, width=640, height=360)], 8, fps=10.0)
    write_video(ep.dir / "streams" / "a.mp4", frames, 5.0)
    src = np.repeat(np.arange(4), 2); write_frames(ep, "a", [frames[j] for j in src], size=(320, 180), src_fps=5.0, src_frames=src)
    from duet.playground import perception as P
    note = []; out = list(W._iter_views(ep, ep.stream("a"), range(8), P.frame_paths(ep, "a"), P.frame_index(ep, "a"), True, 1920, note))
    assert not note and all(nat for _, _, nat in out) and np.array_equal(out[0][1], out[1][1]) and out[0][1].shape == (360, 640)


def test_native_reader_refuses_disguised_playlists(tmp_path):
    """A concat playlist named .mp4 would make the decoder read other files; runtime.pyav_open's demuxer whitelist
    refuses it, so calib/tags fall back to the extracted frames with a note."""
    pytest.importorskip("av")
    ep = new_episode(tmp_path, [Stream("a", "exo", "streams/a.mp4", fps=10.0, width=1280, height=720)], 3)
    (ep.dir / "streams" / "a.mp4").write_text("ffconcat version 1.0\nfile other.mp4\n")
    with pytest.raises(Exception):  # noqa: B017 - PyAV raises an av error type that depends on the version
        W._Native(ep, ep.stream("a"), 1920)
    write_frames(ep, "a", None, n=3)
    from duet.playground import perception as P
    note = []; out = list(W._iter_views(ep, ep.stream("a"), range(3), P.frame_paths(ep, "a"), P.frame_index(ep, "a"), True, 1920, note))
    assert not any(nat for _, _, nat in out) and "native decode unavailable" in note[0]


# ============================================================================= rig validation, T_tag_headcam, idle tags

def test_t_tag_headcam_is_required_and_validated(tmp_path):
    streams = [Stream("head", "ego", "streams/head.mp4", person="leader", fps=10.0, width=640, height=360)]
    ep = new_episode(tmp_path, streams, 5); rig_json(ep, head_tags={"leader": 0}); set_done(ep, "probe", "align", "frames", "calib")
    r = W.rig(ep); assert W._head_tag_key(r, ep.stream("head")) == "leader"  # tag id 0 is a valid id
    with pytest.raises(ValueError, match="T_tag_headcam"):
        W.headpose(ep)
    good = T_from(rot("x", 20), [0.0, 0.05, 0.02])
    np.testing.assert_allclose(W.head_tag_offset({"T_tag_headcam": {"leader": good.tolist()}}, "leader"), good)
    for bad in (np.diag([1.0, 1.0, 1.0, 1.0]) * 2, T_from(np.diag([1.0, 1.0, -1.0]), [0, 0, 0]), T_from(np.eye(3), [0.0, 0.6, 0.0])):
        with pytest.raises(ValueError):
            W.head_tag_offset({"T_tag_headcam": {"leader": bad.tolist()}}, "leader")


@pytest.mark.parametrize("extra", [{"head_tags": {"a": 1}, "object_tags": {"b": 1}}, {"head_tags": {"a": True}}, {"object_tags": {"b": -1}},
                                   {"tag_size_m": 80}, {"board": {"squares_x": 7, "squares_y": 5, "square_m": 50, "marker_m": 37}}])
def test_rig_rejects_bad_values(tmp_path, extra):
    ep = new_episode(tmp_path, [Stream("a", "exo", "streams/a.mp4")], 3); rig_json(ep, **extra)
    with pytest.raises(ValueError):
        W.rig(ep)


def test_tag_size_is_required_with_tags(tmp_path):
    """A default tag size would silently scale every tag distance (a 5 cm tag read as 8 cm: 1.5 m -> 2.4 m)."""
    ep = new_episode(tmp_path, [Stream("a", "exo", "streams/a.mp4")], 3)
    (ep.dir / "rig.json").write_text(json.dumps({"object_tags": {"bowl": 10}}))
    with pytest.raises(ValueError, match="tag_size_m is required"):
        W.rig(ep)
    (ep.dir / "rig.json").write_text(json.dumps({"hfov_deg": {"a": 90}}))
    assert W.rig(ep)["tag_size_m"] is None  # not needed without tags


def test_tags_skipped_without_configured_tags(tmp_path):
    ep = new_episode(tmp_path, [Stream("exoA", "exo", "streams/exoA.mp4", fps=10.0, width=640, height=360)], 5)
    write_frames(ep, "exoA", None, n=5); rig_json(ep)
    write_calib(ep, {"exoA": {"role": "exo", "intrinsics": INTR.to_json(), "registered": True, "T_world_cam": np.eye(4).tolist()}})
    W.tags(ep)
    assert ep.status["tags"]["state"] == "skipped" and "no head_tags/object_tags" in ep.status["tags"]["detail"] and not (ep.derived / "tags").exists()


# ============================================================================= scripts/zed_export.py on a fake pyzed.sl

FAKE_SL = '''
import numpy as np
class _E:
    def __init__(self, name): self.name = name
    def __eq__(self, o): return isinstance(o, _E) and o.name == self.name
    def __hash__(self): return hash(self.name)
    def __repr__(self): return self.name
class _Enum:
    def __init__(self, *names):
        for n in names: setattr(self, n, _E(n))
ERROR_CODE = _Enum("SUCCESS", "END_OF_SVOFILE_REACHED", "CORRUPTED_FRAME", "FAILURE")
UNIT = _Enum("METER", "MILLIMETER"); COORDINATE_SYSTEM = _Enum("IMAGE", "RIGHT_HANDED_Y_UP")
DEPTH_MODE = _Enum("NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS", "ULTRA", "QUALITY", "PERFORMANCE", "NONE")
TIME_REFERENCE = _Enum("IMAGE", "CURRENT"); VIEW = _Enum("LEFT"); MEASURE = _Enum("DEPTH"); REFERENCE_FRAME = _Enum("WORLD")
POSITIONAL_TRACKING_STATE = _Enum("OK", "SEARCHING", "OFF")
SCRIPT = {}   # set by the test: events = [("grab", t_ns, T_world_cam_opencv 4x4, state) | ("error", name)], w, h, fps
class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)
class _Val:
    def __init__(self, v): self.v = np.asarray(v, float)
    def get(self): return self.v
class _Ts:
    def __init__(self, ns): self.ns = ns
    def get_nanoseconds(self): return self.ns
class InitParameters:
    def set_from_svo_file(self, p): self.svo = p
class PositionalTrackingParameters: pass
class RuntimeParameters: pass
class Translation: pass
class Orientation: pass
class Mat:
    def __init__(self): self.data = None
    def get_data(self): return self.data
class Pose:
    def get_translation(self, _=None): return _Val(self.T[:3, 3])
    def get_orientation(self, _=None):
        from scipy.spatial.transform import Rotation
        return _Val(Rotation.from_matrix(self.T[:3, :3]).as_quat())
class SensorsData:
    def get_imu_data(self): return _NS(timestamp=_Ts(self.t - 1000), get_linear_acceleration=lambda: np.array([0.0, 9.81, 0.0]), get_angular_velocity=lambda: np.array([0.1, 0.2, 0.3]))
class Camera:
    @staticmethod
    def get_sdk_version(): return "fake-4.1"
    def open(self, init):
        self.init, self.i, self.cur = init, 0, None; SCRIPT["init"] = init
        return ERROR_CODE.SUCCESS
    def get_camera_information(self):
        cal = _NS(left_cam=_NS(fx=700.0, fy=700.0, cx=SCRIPT["w"] / 2 - 0.5, cy=SCRIPT["h"] / 2 - 0.5, disto=np.zeros(12)), get_camera_baseline=lambda: 0.12)
        return _NS(serial_number=123, camera_model=_E("ZED_2i"), camera_configuration=_NS(calibration_parameters=cal, resolution=_NS(width=SCRIPT["w"], height=SCRIPT["h"]), fps=SCRIPT["fps"]))
    def enable_positional_tracking(self, tp): return ERROR_CODE.SUCCESS
    def get_svo_number_of_frames(self): return len(SCRIPT["events"])
    def get_svo_position(self): return self.i
    def grab(self, rt):
        if self.i >= len(SCRIPT["events"]): return ERROR_CODE.END_OF_SVOFILE_REACHED
        ev = SCRIPT["events"][self.i]; self.i += 1; self.cur = ev
        return ERROR_CODE.SUCCESS if ev[0] == "grab" else getattr(ERROR_CODE, ev[1])
    def get_timestamp(self, ref): return _Ts(self.cur[1])
    def retrieve_image(self, mat, view):
        mat.data = np.full((SCRIPT["h"], SCRIPT["w"], 4), (20 + 5 * self.i) % 250, np.uint8); return ERROR_CODE.SUCCESS
    def get_position(self, pose, ref):
        T = self.cur[2]
        if self.init.coordinate_system == COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP: T = T @ np.diag([1.0, -1.0, -1.0, 1.0])
        pose.T = T; return getattr(POSITIONAL_TRACKING_STATE, self.cur[3])
    def get_sensors_data(self, s, ref): s.t = self.cur[1]; return ERROR_CODE.SUCCESS
    def retrieve_measure(self, mat, m): mat.data = np.full((SCRIPT["h"], SCRIPT["w"]), 1.5, np.float32); return ERROR_CODE.SUCCESS
    def close(self): pass
'''


@pytest.fixture
def fake_zed(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    pkg = tmp_path / "fakezed" / "pyzed"; pkg.mkdir(parents=True); (pkg / "__init__.py").write_text(""); (pkg / "sl.py").write_text(FAKE_SL)
    monkeypatch.syspath_prepend(str(tmp_path / "fakezed"))
    for m in ("pyzed", "pyzed.sl"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    spec = importlib.util.spec_from_file_location("zed_export_under_test", REPO / "scripts" / "zed_export.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    import pyzed.sl as sl
    return mod, sl


def test_zed_export_writes_capture_timeline_and_image_axes(tmp_path, fake_zed):
    av = pytest.importorskip("av")
    mod, sl = fake_zed
    t = grab_times(45, drop=(12, 13, 30), seed=3, collide_at=20)
    events = [("grab", T0_NS + int(round(x * 1e9)), T_ZW_W @ head_T(x), "OK") for x in t]
    events.insert(10, ("error", "CORRUPTED_FRAME")); events[30] = events[30][:3] + ("SEARCHING",)
    sl.SCRIPT.update(events=events, w=64, h=36, fps=ZFPS)
    out = tmp_path / "ep" / "zed" / "zed"
    assert mod.main(["x.svo2", str(out), "--fps", "15"]) == 0
    assert sl.SCRIPT["init"].coordinate_system == sl.COORDINATE_SYSTEM.IMAGE and sl.SCRIPT["init"].coordinate_units == sl.UNIT.METER
    e = json.loads((out / "export.json").read_text())
    assert e["format"] == 2 and e["coordinate_system"] == "IMAGE" and e["coordinate_units"] == "METER" and e["video_timeline"] == "capture"
    assert e["grabs"] == len(t) and e["grab_errors"] == {"CORRUPTED_FRAME": 1} and e["grabs_not_in_video"] == 1 and e["filled_slots"] >= 1 and e["depth_every_n_frames"] == 2
    pose = pd.read_csv(out / "pose.csv"); assert list(pose.frame_index) == list(range(len(t))) and (pose.tracking_state == "SEARCHING").sum() == 1
    assert sorted(int(p.stem) for p in (out / "depth").glob("*.png")) == list(range(0, len(t), 2))
    imu = pd.read_csv(out / "imu.csv"); assert len(imu) == len(t) and (imu.timestamp_ns == pose.timestamp_ns - 1000).all()
    with av.open(str(out / "left.mp4")) as c:
        n_video = sum(1 for _ in c.decode(video=0))
    assert n_video == e["video_frames"] == int(pose.video_frame.max()) + 1
    assert e["video"]["frames"] == e["video_frames"] and {k: e["video"][k] for k in ("size", "sha256_head_tail")} == W._file_identity(out / "left.mp4")
    assert mod.main(["x.svo2", str(out), "--fps", "15"]) == 2  # refuses to write into an earlier export ...
    (out / "notes.txt").write_text("mine")
    assert mod.main(["x.svo2", str(out), "--fps", "15", "--overwrite"]) == 2 and (out / "notes.txt").exists()  # ... never deletes foreign files
    (out / "notes.txt").unlink(); assert mod.main(["x.svo2", str(out), "--fps", "15", "--overwrite"]) == 0
    info = W.zed_export_info(out); trk = W.zed_track(out, info)
    vf = trk["video_frame"]; ok = trk["ok"]
    assert (np.abs(vf[vf >= 0] / ZFPS - trk["t"][vf >= 0]) <= 0.5 / ZFPS + 1e-9).all()  # video time = capture time
    for g in np.nonzero(ok)[0]:
        np.testing.assert_allclose(T_ZW_W @ head_T(t[g]), trk["T"][g], atol=1e-9)  # OpenCV camera axes, no conversion needed
    assert not ok[pose.tracking_state != "OK"].any()


def test_zed_export_refuses_tracking_without_depth(tmp_path, fake_zed):
    mod, sl = fake_zed; sl.SCRIPT.update(events=[], w=64, h=36, fps=ZFPS)
    with pytest.raises(SystemExit):
        mod.main(["x.svo2", str(tmp_path / "o"), "--depth", "none"])
    sl.SCRIPT.update(events=[("grab", T0_NS, np.eye(4), "OK")] + [("error", "FAILURE")] * 5)
    assert mod.main(["x.svo2", str(tmp_path / "o2"), "--no-depth", "--max-grab-failures", "3"]) == 1  # bounded, flagged incomplete
    assert json.loads((tmp_path / "o2" / "export.json").read_text())["complete"] is False and not (tmp_path / "o2" / "depth").exists()
