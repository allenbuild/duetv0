"""imu_arm (IMU clock offset, slerp, validity), qc (None-for-missing, aggregates, NaN guards, proc_fps-invariant
motion) and export (records.parquet v2 contract C8) on synthetic episodes built in tmp_path."""
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.playground import export as X
from duet.playground import imu_arm as IA
from duet.playground.episode import Episode, Stream
from duet.playground.runtime import atomic_savez, atomic_write_json, atomic_write_parquet, load_npz

FPS = 10.0


def _qc():
    """qc needs cv2: skip without it."""
    pytest.importorskip("cv2")
    from duet.playground import qc
    return qc


def rot():
    """scipy Rotation (imu_arm slerps with scipy; tests that need it skip without it)."""
    return pytest.importorskip("scipy.spatial.transform").Rotation


# ----------------------------------------------------------------------------- synthetic builders

def make_ep(tmp_path, streams=(("ego", "ego"),), n=40, fps=FPS, imu=(), offsets=None, name="syn", rig=None) -> Episode:
    """Episode on disk: streams [(name, role)], first = reference, others "audio"-aligned; nothing run yet."""
    d = tmp_path / name; d.mkdir(parents=True)
    offsets = offsets or {}
    ss = [Stream(name=nm, role=role, path=f"streams/{nm}.mp4", fps=30.0, width=1280, height=720, imu=f"imu/{nm}.parquet" if nm in imu else None,
                 offset_s=offsets.get(nm, 0.0), offset_status="reference" if i == 0 else "audio") for i, (nm, role) in enumerate(streams)]
    ep = Episode(name=name, root=str(d.resolve()), streams=ss, reference=ss[0].name, proc_fps=fps, common_start_s=0.0, common_end_s=(n + 0.5) / fps)
    ep.save(); assert ep.n_frames() == n
    if rig is not None:
        (d / "rig.json").write_text(json.dumps(rig))
    return ep


def done(ep: Episode, *stages: str) -> None:
    for st in stages:
        ep.set_status(st, "done")


def write_frames(ep: Episode, s: str, t_src=None, jpgs: bool = False, n: int | None = None, size=(640, 360)) -> None:
    """The frames stage's C4 layout, as perception.extract_frames writes it: derived/frames/<s>/NNNNNN.jpg (1-based names,
    empty or tiny files here), <s>/index.parquet (k, file, t_ref_s, t_src_s, src_frame) and the stream's entry in
    derived/frames/manifest.json (width, height, scale = extracted / display width)."""
    n = ep.n_frames() if n is None else n; d = ep.derived / "frames" / s; d.mkdir(parents=True, exist_ok=True)
    t_ref = ep.frame_times_ref()[:n]; t_src = ep.stream_time(s, t_ref) + 0.004 if t_src is None else t_src
    idx = pd.DataFrame({"k": np.arange(n, dtype=np.int64), "file": [f"{k + 1:06d}.jpg" for k in range(n)], "t_ref_s": t_ref,
                        "t_src_s": np.asarray(t_src, np.float64), "src_frame": np.round(np.asarray(t_src) * 30).astype(np.int64)})
    atomic_write_parquet(idx, d / "index.parquet", metadata={"stream": s})
    for k in range(n):
        (d / f"{k + 1:06d}.jpg").write_bytes(b"" if not jpgs else b"\xff")
    mp = ep.derived / "frames" / "manifest.json"; m = json.loads(mp.read_text()) if mp.exists() else {
        "version": 2, "n_frames": ep.n_frames(), "proc_fps": ep.proc_fps, "proc_size": ep.proc_size, "reference": ep.reference, "streams": {}, "skipped": {}}
    st = ep.stream(s); m["streams"][s] = {"n_frames": n, "width": size[0], "height": size[1], "scale": size[0] / st.width,
                                          "display_width": st.width, "display_height": st.height}
    atomic_write_json(mp, m)


def texture(h=360, w=640, seed=0) -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(seed).uniform(0, 255, (h, w)).astype(np.float32)
    return cv2.GaussianBlur(img, (0, 0), 2.0 * w / 640) * 2.5 - 190  # contrast-stretched smooth noise


def render_frames(ep: Episode, s: str, speed_w_per_s: float, bright: float = 0.0, seed=0, t_src=None, size=(640, 360)) -> None:
    """Real JPEGs of a texture translating horizontally at speed (frame widths / s), rendered at the frames' source
    times t_src (default: the nominal times; repeated times give identical frames, as a repeated source frame does)."""
    cv2 = pytest.importorskip("cv2")
    w, h = size; tex = texture(h, w, seed); n = ep.n_frames(); t = ep.frame_times_stream(s) if t_src is None else np.asarray(t_src)
    write_frames(ep, s, t_src=t, size=size)
    for k in range(n):
        M = np.float32([[1, 0, speed_w_per_s * w * t[k]], [0, 1, 0]])
        img = np.clip(cv2.warpAffine(tex, M, (w, h), borderMode=cv2.BORDER_REFLECT) + bright, 0, 255).astype(np.uint8)
        cv2.imwrite(str(ep.derived / "frames" / s / f"{k + 1:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def nearest_source_times(ep: Episode, s: str, src_fps: float) -> np.ndarray:
    """What the frames stage does: frame k = the source frame nearest to its nominal time (repeats when src_fps <= proc_fps)."""
    return np.round(ep.frame_times_stream(s) * src_fps) / src_fps


def yaw_rate(t: np.ndarray, seed=0) -> np.ndarray:
    """Smooth random yaw rate (rad/s): Gaussian bursts, one per ~2 s."""
    rng = np.random.default_rng(seed); w = np.zeros_like(t)
    for c in rng.uniform(t.min(), t.max(), max(1, int((t.max() - t.min()) / 2))):
        amp, sig = rng.normal(0, 1.5), rng.uniform(0.1, 0.4); lo, hi = np.searchsorted(t, [c - 5 * sig, c + 5 * sig])
        w[lo:hi] += amp * np.exp(-0.5 * ((t[lo:hi] - c) / sig) ** 2)
    return w


def synth_chest(duration=60.0, rate=24.0, seed=0, repeat_frac=0.0):
    """(t_imu [m], q [m,4], fine grid tf, |w| on tf): chest yaw from yaw_rate, sampled at ~rate Hz on the IMU clock,
    optionally with sample-and-hold repeats like the Eidon stream."""
    R = rot(); tf = np.arange(-5.0, duration + 5.0, 0.001); w = yaw_rate(tf, seed); yaw = np.cumsum(w) * 0.001
    t = np.arange(0.042, duration, 1 / rate); q = R.from_euler("z", np.interp(t, tf, yaw)[:, None]).as_quat()
    rep = np.random.default_rng(seed + 1).random(len(t)) < repeat_frac; rep[0] = False
    for i in np.flatnonzero(rep):
        q[i] = q[i - 1]
    return t, q, tf, np.abs(w)


def video_motion(t_frames, off, tf, wabs, noise=0.3, seed=2, drift_ppm=0.0):
    """Camera motion over (t[k-1], t[k]] = mean |w| over the matching IMU-clock interval, noisy.
    Clock: t_stream = (1 + drift_ppm * 1e-6) * t_imu + off (off may be an array per frame)."""
    c = np.r_[0, np.cumsum(wabs) * 0.001]; tt = np.r_[tf[0] - 0.001, tf]
    ti = (t_frames - off) / (1 + drift_ppm * 1e-6); a, b = ti[:-1], ti[1:]
    m = (np.interp(b, tt, c) - np.interp(a, tt, c)) / (b - a)
    m = m * np.exp(np.random.default_rng(seed).normal(0, noise, len(m))) + 0.02
    return np.r_[np.nan, m]


def imu_frame(t, quats: dict[int, np.ndarray]) -> pd.DataFrame:
    """Eidon-schema IMU rows for slots -> quats [m,4] at times t (s)."""
    return pd.DataFrame([{"time_ms": round(tt * 1000), "slot": slot, "quat_x": q[0], "quat_y": q[1], "quat_z": q[2], "quat_w": q[3]}
                         for slot, qs in quats.items() for tt, q in zip(t, qs, strict=True)])


# ----------------------------------------------------------------------------- imu_arm: clock offset

@pytest.mark.parametrize("off", [-0.42, 0.3, 0.0])
def test_imu_offset_recovers_planted_shift(off):
    t, q, tf, wabs = synth_chest(seed=3, repeat_frac=0.3)
    tv = np.arange(0, 58, 1 / FPS) + 0.013
    r = IA.estimate_imu_offset(tv, video_motion(tv, off, tf, wabs), t, q, 1.5 / 24)
    assert r["status"] == "estimated", r["reason"]
    assert abs(r["offset_s"] - off) < 0.02, r["offset_s"]
    assert r["p"] <= 0.01 and r["peak_ratio"] > 1.3 and r["r"] > 0.5


def test_imu_offset_sign_convention_single_event():
    """One rotation burst at IMU time 20.0 s seen by the camera at stream time 19.58 s => t_stream = t_imu - 0.42."""
    R = rot(); tf = np.arange(-5.0, 45.0, 0.001); w = 3.0 * np.exp(-0.5 * ((tf - 20.0) / 0.15) ** 2) + 0.4 * np.exp(-0.5 * ((tf - 31.0) / 0.3) ** 2)
    yaw = np.cumsum(w) * 0.001; t = np.arange(0.0, 40.0, 1 / 24); q = R.from_euler("z", np.interp(t, tf, yaw)[:, None]).as_quat()
    tv = np.arange(0.0, 40.0, 1 / FPS); m = video_motion(tv, -0.42, tf, np.abs(w), noise=0.05)
    assert abs(tv[np.nanargmax(m)] - 19.6) < 0.11  # the camera sees the burst ~0.42 s before the IMU stamps it
    r = IA.estimate_imu_offset(tv, m, t, q, 1.5 / 24, {"min_shift_s": 3.0})
    assert abs(r["offset_s"] + 0.42) < 0.02, r


def test_imu_offset_unrelated_motion_is_unsynced():
    t, q, _, _ = synth_chest(seed=4)
    tv = np.arange(0, 58, 1 / FPS); m = np.r_[np.nan, np.random.default_rng(9).gamma(2.0, 0.05, len(tv) - 1)]
    r = IA.estimate_imu_offset(tv, m, t, q, 1.5 / 24)
    assert r["status"] == "unsynced" and r["reason"] and r["p"] > 0.01


def test_imu_offset_needs_overlap():
    t, q, tf, wabs = synth_chest(duration=8.0, seed=5)
    tv = np.arange(0, 7.5, 1 / FPS)
    r = IA.estimate_imu_offset(tv, video_motion(tv, 0.1, tf, wabs), t, q, 1.5 / 24)
    assert r["status"] == "unsynced" and "frames overlap" in r["reason"] and math.isnan(r["offset_s"])


def test_imu_clock_fits_planted_drift():
    """100 ppm drift over 20 min: window offsets spread ~80 ms > half a frame, so a line is fitted and applied."""
    off0, ppm = -0.42, 100.0
    t, q, tf, wabs = synth_chest(duration=1200.0, seed=7, repeat_frac=0.25)
    tv = np.arange(0, 1195, 1 / FPS) + 0.013
    r = IA.estimate_imu_clock(tv, video_motion(tv, off0, tf, wabs, drift_ppm=ppm), t, q, 1.5 / 24)
    assert r["status"] == "estimated" and r["drift_check"].startswith("drift fitted"), r["drift_check"]
    assert abs(r["drift_ppm"] - ppm) < 20 and abs(r["offset_s"] - off0) < 0.02, (r["drift_ppm"], r["offset_s"])
    assert r["window_spread_s"] > 0.05 and len(r["window_t_s"]) == 3
    ti = np.array([100.0, 600.0, 1100.0])  # the fitted clock is right everywhere, not just on average
    err = (1 + r["drift_ppm"] * 1e-6) * ti + r["offset_s"] - ((1 + ppm * 1e-6) * ti + off0)
    assert np.abs(err).max() < 0.02
    # control: the same recording without drift keeps the constant offset
    c = IA.estimate_imu_clock(tv, video_motion(tv, off0, tf, wabs), t, q, 1.5 / 24)
    assert c["status"] == "estimated" and c["drift_check"].startswith("consistent") and c["drift_ppm"] == 0.0
    assert abs(c["offset_s"] - off0) < 0.02 and c["offset_s"] == c["offset_global_s"]


def test_imu_clock_jump_is_drift_suspected():
    """Two windows whose offsets differ by 150 ms can't be checked against a line: nothing may be applied."""
    t, q, tf, wabs = synth_chest(duration=150.0, seed=8)
    tv = np.arange(0, 148, 1 / FPS); off = np.where(tv < 74, -0.40, -0.25)
    r = IA.estimate_imu_clock(tv, video_motion(tv, off, tf, wabs, noise=0.2), t, q, 1.5 / 24)
    assert r["status"] == "drift_suspected" and "drift suspected" in r["drift_check"], r["drift_check"]
    np.testing.assert_allclose(np.sort(r["window_offset_s"]), [-0.40, -0.25], atol=0.02)
    short = IA.estimate_imu_clock(tv[:900], video_motion(tv[:900], -0.4, tf, wabs), t, q, 1.5 / 24)  # 90 s: no windows
    assert short["status"] == "estimated" and short["drift_check"].startswith("not run")


# ----------------------------------------------------------------------------- imu_arm: slerp and validity

def test_slerp_handles_quaternion_sign_flip():
    """Yaw sweeping through 180 deg stored with w >= 0 flips q -> -q; slerp must follow the shortest arc."""
    R = rot(); t = np.arange(0, 3.0, 0.041); yaw = np.radians(120 + 40 * t)
    q = R.from_euler("z", yaw[:, None]).as_quat(); q[q[:, 3] < 0] *= -1
    assert (np.einsum("ij,ij->i", q[1:], q[:-1]) < 0).any()  # the sign flip is really there
    flip = int(np.flatnonzero(np.einsum("ij,ij->i", q[1:], q[:-1]) < 0)[0])
    te = np.r_[np.arange(0.01, 2.9, 1 / 24), (t[flip] + t[flip + 1]) / 2]  # includes the midpoint of the flipped pair
    qe, ok = IA.eval_slot(t, q, te, 1.5 * 0.041)
    assert ok.all()
    truth = R.from_euler("z", np.radians(120 + 40 * te)[:, None])
    assert np.degrees((R.from_quat(qe) * truth.inv()).magnitude()).max() < 0.01
    # the component-wise interpolation this replaces is ~180 deg off between the flipped samples
    qi = np.stack([np.interp(te, t, q[:, i]) for i in range(4)], 1); qi /= np.linalg.norm(qi, axis=1, keepdims=True)
    assert np.degrees((R.from_quat(qi) * truth.inv()).magnitude())[-1] > 90


def _rig_quats(t, seed=0):
    R = rot(); rng = np.random.default_rng(seed)
    return {s: R.from_euler("zyx", np.c_[np.sin(t + s), 0.3 * np.cos(0.7 * t + s), 0.2 * np.sin(1.3 * t)] + rng.normal(0, 0.01, (len(t), 3))).as_quat() for s in range(7)}


def test_validity_missing_chest_gaps_edges_and_hand():
    t = np.arange(0, 10, 1 / 24); Q = _rig_quats(t)
    full = IA.arm_chain(imu_frame(t, Q))
    assert full["valid"].all() and full["hand_valid"].all() and np.isfinite(full["points"]).all()
    np.testing.assert_allclose(full["t_imu_s"], np.round(t * 1000) / 1000)  # original sample times, no grid
    # missing chest: no torso heading -> invalid, not silently yaw = 0
    no_chest = IA.arm_chain(imu_frame(t, {s: q for s, q in Q.items() if s != IA.CHEST}))
    assert not no_chest["valid"].any() and np.isnan(no_chest["points"]).all() and np.isnan(no_chest["chest_yaw"]).all()
    # a 0.5 s gap in every slot: nothing inside it is valid (no bridging), the samples around it are
    keep = (t < 4.0) | (t > 4.5)
    gap = IA.arm_chain(imu_frame(t[keep], {s: q[keep] for s, q in Q.items()}), t_eval=np.array([3.9, 4.1, 4.25, 4.4, 4.6]))
    assert gap["valid"][:, 0].tolist() == [True, False, False, False, True]
    # edges: never held beyond the first / last sample
    t_last = full["t_imu_s"][-1]  # time_ms resolution
    edge = IA.arm_chain(imu_frame(t, Q), t_eval=np.array([-0.2, 0.0, 5.0, t_last, t_last + 0.02]))
    assert edge["valid"][:, 1].tolist() == [False, True, True, True, False]
    # missing hand sensor: fingertip unknown, the rest of the arm still valid
    no_hand = IA.arm_chain(imu_frame(t, {s: q for s, q in Q.items() if s != 0}))
    assert no_hand["valid"][:, 0].all() and not no_hand["hand_valid"][:, 0].any() and no_hand["hand_valid"][:, 1].all()
    assert np.isnan(no_hand["points"][:, 0, 3]).all() and np.isfinite(no_hand["points"][:, 0, :3]).all()
    np.testing.assert_allclose(no_hand["points"][:, 0, :3], full["points"][:, 0, :3], atol=1e-6)


def test_arm_chain_kinematics_unchanged():
    """Identity sensors: every bone points along scene +Z of the Eidon mapping (device +Y -> scene +Z)."""
    rot(); t = np.arange(0, 1, 1 / 24); Q = {s: np.tile([0.0, 0, 0, 1], (len(t), 1)) for s in range(7)}
    a = IA.arm_chain(imu_frame(t, Q))
    np.testing.assert_allclose(a["points"][0, 0], [[-0.18, 1.40, 0], [-0.18, 1.40, 0.30], [-0.18, 1.40, 0.56], [-0.18, 1.40, 0.66]], atol=1e-6)
    np.testing.assert_allclose(a["elbow_flex_deg"], 0, atol=1e-3)


def test_imu_arm_stage_estimates_and_applies_offset(tmp_path):
    off, n = -0.42, 580
    ep = make_ep(tmp_path, n=n, imu=("ego",)); t, q, tf, wabs = synth_chest(seed=6, repeat_frac=0.25)
    Q = _rig_quats(t); Q[IA.CHEST] = q; (ep.dir / "imu").mkdir(); imu_frame(t, Q).to_parquet(ep.dir / "imu" / "ego.parquet")
    tv = ep.frame_times_stream("ego") + 0.01
    atomic_savez(ep.derived / "qc" / "ego.npz", schema=2, motion=video_motion(tv, off, tf, wabs).astype(np.float32), t_src_s=tv,
                 sharp=np.ones(n), bright=np.ones(n))
    done(ep, "frames", "qc"); IA.imu_arm(ep)
    a = load_npz(ep.derived / "imu_arm" / "ego.npz"); st = ep.status["imu_arm"]
    assert st["state"] == "done" and str(a["imu_sync"]) == "estimated" and st["imu_sync"]["ego"]["sync"] == "estimated"
    assert abs(float(a["imu_offset_s"]) - off) < 0.02 and str(a["motion_source"]) == "qc"
    np.testing.assert_array_equal(a["imu_time_ms"], np.round(t * 1000).astype(int))  # original stamps preserved
    np.testing.assert_allclose(a["t_s"], ep.ref_time("ego", a["t_imu_s"] + a["imu_offset_s"]))
    # no motion signal at all (neither qc nor frames done): chain still written, offset 0, explicitly unsynced
    ep2 = make_ep(tmp_path, n=n, imu=("ego",), name="nosig"); (ep2.dir / "imu").mkdir(); imu_frame(t, Q).to_parquet(ep2.dir / "imu" / "ego.parquet")
    IA.imu_arm(ep2); b = load_npz(ep2.derived / "imu_arm" / "ego.npz")
    assert str(b["imu_sync"]) == "unsynced" and float(b["imu_offset_s"]) == 0.0 and "camera-motion" in str(b["imu_sync_reason"])
    assert "NOT applied" in ep2.status["imu_arm"]["detail"]


def test_imu_arm_manual_override_and_suspected_drift(tmp_path):
    n = 1480; t, q, tf, wabs = synth_chest(duration=150.0, seed=8); Q = _rig_quats(t); Q[IA.CHEST] = q

    def episode(name, off):
        ep = make_ep(tmp_path, n=n, imu=("ego",), name=name); (ep.dir / "imu").mkdir(); imu_frame(t, Q).to_parquet(ep.dir / "imu" / "ego.parquet")
        tv = ep.frame_times_stream("ego")
        atomic_savez(ep.derived / "qc" / "ego.npz", schema=2, motion=video_motion(tv, off(tv), tf, wabs, noise=0.2).astype(np.float32), t_src_s=tv,
                     sharp=np.ones(n), bright=np.ones(n))
        done(ep, "frames", "qc")
        return ep
    ep = episode("manual", lambda tv: np.full(len(tv), -0.42))
    ep.stream("ego").imu_offset_override_s = -0.3; ep.save()  # t_stream = t_imu - 0.3; the estimate is still reported
    ep = Episode.load(ep.dir); assert ep.stream("ego").imu_offset_override_s == -0.3
    IA.imu_arm(ep); a = load_npz(ep.derived / "imu_arm" / "ego.npz")
    assert str(a["imu_sync"]) == "manual" and float(a["imu_offset_s"]) == -0.3 and float(a["imu_drift_ppm"]) == 0.0
    assert str(a["imu_estimate_status"]) == "estimated" and abs(float(a["imu_offset_estimate_s"]) + 0.42) < 0.02
    np.testing.assert_allclose(a["t_s"], ep.ref_time("ego", a["t_imu_s"] - 0.3))
    assert ep.status["imu_arm"]["imu_sync"]["ego"]["sync"] == "manual" and "MANUAL" in ep.status["imu_arm"]["detail"]
    jump = episode("jump", lambda tv: np.where(tv < 74, -0.40, -0.25)); IA.imu_arm(jump)
    b = load_npz(jump.derived / "imu_arm" / "ego.npz")
    assert str(b["imu_sync"]) == "drift_suspected" and float(b["imu_offset_s"]) == 0.0 and np.isfinite(b["imu_offset_estimate_s"])
    assert "NOT applied (drift_suspected" in jump.status["imu_arm"]["detail"]


def test_imu_arm_manual_override_with_repeated_frames(tmp_path):
    """25 fps source at proc_fps 30 (repeated source times). A manual offset never depends on the camera-motion
    estimate: if the estimate can't run it is reported as failed; when it can, it is reported next to the offset."""
    n = 1500; ep = make_ep(tmp_path, n=n, fps=30.0, imu=("ego",), name="rep"); t, q, tf, wabs = synth_chest(seed=9)
    Q = _rig_quats(t); Q[IA.CHEST] = q; (ep.dir / "imu").mkdir(); imu_frame(t, Q).to_parquet(ep.dir / "imu" / "ego.parquet")
    tv = nearest_source_times(ep, "ego", 25.0); rep = np.diff(tv) == 0; assert rep.any()
    write_frames(ep, "ego", t_src=tv); done(ep, "frames")  # empty JPEGs: the frames fallback cannot read them
    ep.stream("ego").imu_offset_override_s = -0.3; ep.save(); IA.imu_arm(ep)
    a = load_npz(ep.derived / "imu_arm" / "ego.npz")
    assert ep.status["imu_arm"]["state"] == "done" and str(a["imu_sync"]) == "manual" and float(a["imu_offset_s"]) == -0.3
    assert str(a["imu_estimate_status"]) == "unsynced" and "estimate failed" in str(a["imu_sync_reason"])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = video_motion(tv, -0.42, tf, wabs)
    m[1:][rep] = np.nan  # what qc writes for repeated pairs
    atomic_savez(ep.derived / "qc" / "ego.npz", schema=2, motion=m.astype(np.float32), t_src_s=tv, sharp=np.ones(n), bright=np.ones(n))
    done(ep, "qc"); IA.imu_arm(ep); b = load_npz(ep.derived / "imu_arm" / "ego.npz")
    assert str(b["imu_sync"]) == "manual" and float(b["imu_offset_s"]) == -0.3
    assert str(b["imu_estimate_status"]) == "estimated" and abs(float(b["imu_offset_estimate_s"]) + 0.42) < 0.02
    ep.stream("ego").imu_offset_override_s = None; ep.save(); IA.imu_arm(ep); c = load_npz(ep.derived / "imu_arm" / "ego.npz")
    assert str(c["imu_sync"]) == "estimated" and abs(float(c["imu_offset_s"]) + 0.42) < 0.02


def test_arm_chain_axes_and_heading_removal_analytic():
    """Known sensor orientations -> known bone directions. Device +Y is the bone. World (Z up) -> scene is
    (-x, z, y), so world -X is scene +X and world +Y is scene +Z (forward). The chest's device +Z sets the heading.
    Turning the whole wearer about the vertical leaves the torso-relative chain unchanged."""
    R = rot(); t = np.arange(0, 1, 1 / 24)

    def rig(heading_deg):
        H = R.from_euler("z", heading_deg, degrees=True)
        body = {6: R.from_euler("x", -90, degrees=True),  # chest: device +Z -> world +Y -> scene +Z, yaw 0
                2: R.from_euler("x", -90, degrees=True),  # left upper arm: device +Y -> world -Z, hanging
                1: R.identity(),                          # left forearm: world +Y -> scene +Z, forward
                0: R.from_euler("z", 90, degrees=True),   # left hand: world -X -> scene +X
                5: R.from_euler("x", 45, degrees=True),   # right upper arm: world (0, .71, .71): forward and up 45 deg
                4: R.from_euler("z", -90, degrees=True),  # right forearm: world +X -> scene -X
                3: R.from_euler("x", 90, degrees=True)}   # right hand: world +Z -> scene +Y, up
        return {s: np.tile((H * r).as_quat(), (len(t), 1)) for s, r in body.items()}
    a = IA.arm_chain(imu_frame(t, rig(0.0)))
    Lsh, Rsh, c = np.array([-0.18, 1.40, 0]), np.array([0.18, 1.40, 0]), np.sqrt(0.5)
    le, re = Lsh + [0, -0.30, 0], Rsh + [0, 0.30 * c, 0.30 * c]
    want = np.array([[Lsh, le, le + [0, 0, 0.26], le + [0.10, 0, 0.26]], [Rsh, re, re + [-0.26, 0, 0], re + [-0.26, 0.10, 0]]])
    np.testing.assert_allclose(a["points"], np.broadcast_to(want, a["points"].shape), atol=1e-5)
    np.testing.assert_allclose(a["elbow_flex_deg"], 90, atol=1e-3); np.testing.assert_allclose(a["chest_yaw"], 0, atol=1e-6)
    b = IA.arm_chain(imu_frame(t, rig(70.0)))  # same pose, wearer facing 70 deg away
    np.testing.assert_allclose(b["chest_yaw"], np.radians(70), atol=1e-5)
    np.testing.assert_allclose(b["points"], a["points"], atol=1e-5)


def test_eidon_imu_adapter_parsing(tmp_path):
    from duet.adapters import eidon_imu as E
    assert IA.load_imu is E.load_imu and IA.slot_samples is E.slot_samples and IA.SLOTS == E.SLOTS
    assert E.SLOT["chest"] == IA.CHEST == 6 and IA.SIDES == (("left", 2, 1, 0), ("right", 5, 4, 3))
    df = pd.DataFrame({"time_ms": [0, 41, 41, 82, 123], "slot": [6] * 5, "quat_x": [0.0, 0, 0, np.nan, 0], "quat_y": 0.0,
                       "quat_z": [0.0, 0, 0, 0, 0.6], "quat_w": [2.0, 1, 1, 1, 1.0]})
    t, q = E.slot_samples(df)[6]  # |q| = 2 and NaN rows dropped, a repeated stamp keeps the first, quats renormalised
    np.testing.assert_allclose(t, [0.041, 0.123]); np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1)
    df.drop(columns="quat_w").to_parquet(tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="quat_w"):
        E.load_imu(tmp_path / "x.parquet")


def test_imu_arm_skips_without_imu(tmp_path):
    ep = make_ep(tmp_path); IA.imu_arm(ep)
    assert ep.status["imu_arm"]["state"] == "skipped" and not (ep.derived / "imu_arm").exists()


# ----------------------------------------------------------------------------- qc

def _strict(path):
    def bad(c):
        raise ValueError(f"non-JSON constant {c}")
    return json.loads(path.read_text(), parse_constant=bad)


def test_qc_missing_inputs_are_none_with_reason(tmp_path):
    Q = _qc()
    ep = make_ep(tmp_path, streams=(("leader", "ego"), ("helper", "ego"), ("front", "exo")), n=12)
    for s in ("leader", "helper", "front"):
        render_frames(ep, s, 0.05)
    atomic_savez(ep.derived / "hands" / "leader.npz", lm2d=np.where(np.arange(12)[:, None, None, None] % 2 == 0, 5.0, np.nan) * np.ones((12, 2, 21, 2)),
                 lm3d=np.zeros((12, 2, 21, 3)), score=np.ones((12, 2)))  # v1 layout; helper.npz missing on purpose
    done(ep, "frames", "hands"); ep.set_status("body2d", "failed", "boom")
    Q.qc(ep); rep = _strict(ep.derived / "qc" / "report.json"); S, E = rep["streams"], rep["episode"]
    assert S["leader"]["hand_presence_ratio"] == pytest.approx(0.5) and S["leader"]["hands_schema"] == 1
    assert S["helper"]["hand_presence_ratio"] is None and "no hands output" in S["helper"]["missing"]["hand_presence_ratio"]
    assert S["front"]["person_presence_ratio"] is None and "body2d stage failed" in S["front"]["missing"]["person_presence_ratio"]
    assert E["hands_all_egos_ratio"] is None and "helper" in E["missing"]["hands_all_egos_ratio"]  # not silently leader-only
    assert E["both_visible_ratio"] is None
    assert S["helper"]["flags"]["hand_presence_ratio"] == "missing" and S["helper"]["verdict"] in ("flag", "reject")
    assert E["verdict"] != "pass" and ep.status["qc"]["state"] == "done"
    assert rep["thresholds"]["farneback"]["winsize"] == 9 and rep["thresholds"]["flow_long_px"] == 160  # constants in the report
    q = load_npz(ep.derived / "qc" / "helper.npz")
    assert np.isnan(q["present"]).all() and np.isnan(q["motion"][0]) and np.isfinite(q["motion"][1:]).all()


def test_qc_aggregates_verdict_and_nan_guards(tmp_path):
    Q = _qc()
    ep = make_ep(tmp_path, streams=(("a", "ego"), ("b", "ego"), ("c", "exo")), n=10)
    ep.stream("c").offset_status = "unaligned"; ep.save()  # unusable stream: no frames, metrics None, verdict reject
    for s, pres in (("a", [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]), ("b", [1, 0, 0, 1, 1, 1, 1, 1, 1, 1])):
        render_frames(ep, s, 0.05)
        p = np.zeros((10, 2), bool); p[:, 0] = np.array(pres, bool)
        atomic_savez(ep.derived / "hands" / f"{s}.npz", schema=2, present=p, partner_present=np.zeros((10, 2), bool),
                     lm2d=np.zeros((10, 2, 21, 2)), lm3d=np.zeros((10, 2, 21, 3)), score=np.ones((10, 2)))
    done(ep, "frames", "hands", "body2d"); Q.qc(ep)
    rep = _strict(ep.derived / "qc" / "report.json"); S, E = rep["streams"], rep["episode"]
    assert E["hands_all_egos_ratio"] == pytest.approx(0.8)  # frames 2 and 3 lack a hand in some ego view
    assert S["c"]["usable"] is False and S["c"]["verdict"] == "reject" and S["c"]["good_frame_percent"] is None
    assert E["both_visible_ratio"] is None and E["min_good_frame_percent"] is None and E["verdict"] == "reject"
    assert S["a"]["verdict"] == "pass" and S["a"]["flags"]["good_frame_percent"] == "pass"
    # a frames stage that disagrees with n_frames() fails loudly instead of writing NaN or truncating
    for f in (ep.derived / "frames" / "a").glob("*.jpg"):
        f.unlink()
    with pytest.raises(ValueError, match="on disk"):
        Q.qc(ep)


def test_qc_one_frame_has_no_motion_metrics(tmp_path):
    Q = _qc()
    ep = make_ep(tmp_path, n=1); render_frames(ep, "ego", 0.1); done(ep, "frames"); Q.qc(ep)
    r = _strict(ep.derived / "qc" / "report.json")["streams"]["ego"]
    assert r["stability_score"] is None and "motion estimate" in r["missing"]["stability_score"] and r["good_frame_percent"] is not None


def test_qc_motion_is_proc_fps_invariant(tmp_path):
    """Same scene (texture panning at 0.1 frame widths/s) processed at 10 and 30 fps: same motion in widths/s."""
    Q = _qc()
    med = {}
    for fps in (10.0, 30.0):
        ep = make_ep(tmp_path, n=int(3 * fps), fps=fps, name=f"f{int(fps)}"); render_frames(ep, "ego", 0.1)
        m = Q.per_frame({"ego": (Q.frame_paths(ep, ep.stream("ego")), Q.frame_times(ep, ep.stream("ego"))[0])})["ego"][2]
        med[fps] = np.nanmedian(m)
    assert med[10.0] == pytest.approx(0.1, rel=0.1) and med[30.0] == pytest.approx(0.1, rel=0.1)
    assert abs(med[10.0] - med[30.0]) / med[30.0] < 0.05


@pytest.mark.parametrize(("src_fps", "n"), [(25.0, 60), (29.97, 510)])
def test_qc_repeated_source_frames(tmp_path, src_fps, n):
    """proc_fps 30 on a 25 or 29.97 fps source: the frames stage repeats source frames, so source times only
    non-decrease. qc must run; repeated pairs get no motion sample; the rest still measure the true pan speed."""
    Q = _qc(); ep = make_ep(tmp_path, n=n, fps=30.0, name=f"src{src_fps:g}")
    t_src = nearest_source_times(ep, "ego", src_fps); rep = np.diff(t_src) == 0
    assert rep.any() and (np.diff(t_src) >= 0).all()
    render_frames(ep, "ego", 0.1, t_src=t_src, size=(320, 180)); done(ep, "frames"); Q.qc(ep)
    r = _strict(ep.derived / "qc" / "report.json")["streams"]["ego"]; m = load_npz(ep.derived / "qc" / "ego.npz")["motion"]
    assert ep.status["qc"]["state"] == "done" and r["frame_times"] == "source pts" and r["stability_score"] is not None
    assert np.isnan(m[1:][rep]).all() and np.isfinite(m[1:][~rep]).all()
    assert np.nanmedian(m) == pytest.approx(0.1, rel=0.1)


def test_per_frame_cancels_queued_chunks_on_exit(monkeypatch):
    """SIGTERM arrives as SystemExit: per_frame must re-raise at once, not run every queued chunk first."""
    Q = _qc(); ran = []

    def chunk(paths, t, lo, hi, full):
        ran.append(lo)
        if lo == 0:
            raise SystemExit(143)
        time.sleep(0.02)
        return np.zeros(hi - lo), np.zeros(hi - lo), np.zeros(hi - lo)
    monkeypatch.setattr(Q, "_chunk", chunk)
    n = 10 * Q.CHUNK; t0 = time.time()
    with pytest.raises(SystemExit):
        Q.per_frame({"a": ([Path(f"{k}.jpg") for k in range(n)], np.arange(n) / 10.0)}, workers=1)
    time.sleep(0.1)
    assert len(ran) <= 2 and time.time() - t0 < 1.0  # at most the chunk already in flight; 8+ were cancelled


def test_camera_motion_falls_back_to_frames(tmp_path):
    Q = _qc()
    ep = make_ep(tmp_path, n=15); render_frames(ep, "ego", 0.1)
    atomic_savez(ep.derived / "qc" / "ego.npz", schema=2, motion=np.full(15, 9.0), t_src_s=np.arange(15.0))  # left by a qc that isn't done
    assert Q.camera_motion(ep, ep.stream("ego")) is None  # frames not done: nothing to use, stale qc file ignored
    done(ep, "frames"); t, m, src = Q.camera_motion(ep, ep.stream("ego"))
    assert src == "frames" and np.isnan(m[0]) and np.nanmedian(m) == pytest.approx(0.1, rel=0.15)
    np.testing.assert_allclose(t, ep.frame_times_stream("ego"))


# ----------------------------------------------------------------------------- export

STREAMS = (("leader", "ego"), ("helper", "ego"), ("front", "exo"))


def _full_outputs(ep: Episode, n: int, hands_v2: bool = True) -> None:
    """Every stage's artifact with n rows (synthetic values) + all statuses done."""
    rng = np.random.default_rng(0)
    for s in ep.streams:
        write_frames(ep, s.name)
        k = rng.uniform(0, 600, (n, 4, 17, 3)).astype(np.float32); b = rng.uniform(0, 600, (n, 4, 5)).astype(np.float32)
        k[:, 2:], b[:, 2:] = np.nan, np.nan
        atomic_savez(ep.derived / "body2d" / f"{s.name}.npz", schema=2, kpts=k, boxes=b, img_w=640, img_h=360)
        atomic_savez(ep.derived / "qc" / f"{s.name}.npz", schema=2, sharp=np.full(n, 50.0), bright=np.full(n, 100.0),
                     motion=np.r_[np.nan, np.full(n - 1, 0.05)], t_src_s=ep.frame_times_stream(s))
    for s in ep.egos():
        lm2d = rng.uniform(0, 600, (n, 2, 21, 2)); lm3d = rng.normal(0, 0.05, (n, 2, 21, 3)); pres = np.ones((n, 2), bool); pres[::3, 1] = False
        lm2d[~pres], lm3d[~pres] = np.nan, np.nan
        if hands_v2:
            lm3d = lm3d - lm3d[:, :, :1]
            atomic_savez(ep.derived / "hands" / f"{s.name}.npz", schema=2, lm2d=lm2d, lm3d=lm3d, lm3d_origin="wrist", score=np.where(pres, 0.9, np.nan),
                         present=pres, partner_lm2d=np.full_like(lm2d, np.nan), partner_lm3d=np.full_like(lm3d, np.nan),
                         partner_score=np.full((n, 2), np.nan), partner_present=np.zeros((n, 2), bool), n_detected=pres.sum(1),
                         p_wearer=np.where(pres, 0.95, np.nan), partner_p_wearer=np.full((n, 2), np.nan), wearer=s.person or "", partner="", owner_prior=np.float32(8.0), partner_seen_s=np.float32(np.nan))
        else:
            atomic_savez(ep.derived / "hands" / f"{s.name}.npz", lm2d=lm2d, lm3d=lm3d, score=np.where(pres, 0.9, 0.0))
        boxes = np.full((n, 12, 5), np.nan, np.float32); boxes[:, 0] = [10, 20, 30, 40, 0.8]; names = np.full((n, 12), "", "<U24"); names[:, 0] = "cup"
        atomic_savez(ep.derived / "objects" / f"{s.name}.npz", boxes=boxes, names=names)
        T = np.tile(np.eye(4), (n, 1, 1)); T[5] = np.nan
        atomic_savez(ep.derived / "headpose" / f"{s.name}.npz", T_world_cam=T, valid=np.isfinite(T[:, 0, 0]), backend="head_tag")
    w = {"bodies": rng.normal(0, 1, (n, 2, 17, 3)), "body_err": np.zeros((n, 2, 17))}
    for s in ep.egos():
        w[f"hands3d_{s.name}"] = rng.normal(0, 1, (n, 2, 21, 3)); w[f"head_{s.name}"] = rng.normal(0, 1, (n, 3))
    w["object_bowl"] = rng.normal(0, 1, (n, 3))
    for f in X.world_features(ep):
        w[f"feat_{f}"] = rng.normal(0, 1, n)
    atomic_savez(ep.derived / "world3d" / "world3d.npz", **w)
    atomic_savez(ep.derived / "body3d" / "body3d.npz", world=rng.normal(0, 1, (n, 2, 33, 3)), stream="front")
    done(ep, "frames", "body2d", "hands", "objects", "headpose", "qc", "world3d", "body3d")


def _read(ep):
    pq = pytest.importorskip("pyarrow.parquet")
    t = pq.read_table(ep.derived / "records.parquet")
    return t, json.loads(t.schema.metadata[b"duet.records"])


def test_export_schema_is_stable_across_stage_sets(tmp_path):
    n = 20; rig = {"object_tags": {"bowl": 10}}
    full = make_ep(tmp_path, STREAMS, n=n, name="full", rig=rig); _full_outputs(full, n); X.export(full)
    bare = make_ep(tmp_path, STREAMS, n=n, name="bare", rig=rig)
    for s in bare.streams:
        write_frames(bare, s.name)
    atomic_savez(bare.derived / "hands" / "leader.npz", lm2d=np.zeros((n + 5, 2, 21, 2)))  # stale file of a stage that isn't done
    done(bare, "frames"); X.export(bare)
    tf, mf = _read(full); tb, mb = _read(bare)
    assert tf.schema.remove_metadata() == tb.schema.remove_metadata()  # same names AND types, whatever ran
    assert tf.schema.metadata[b"duet.schema_version"] == b"2" and mf["schema_version"] == 2
    for c in ("frame_idx", "t_ref_s", "leader_t_src_s", "leader_frame_file", "leader_hand_L_lm3d", "leader_partner_hand_R_present",
              "front_body2d_p0", "world_object_bowl", "world_feat_head_dist_m", "body3d_p1", "helper_T_world_cam", "front_qc_motion"):
        assert c in tf.column_names, c
    assert tb.column("leader_hand_L_lm3d").null_count == n and tb.column("world_body_p0").null_count == n  # null, never stale data
    assert tb.column("leader_t_src_s").null_count == 0 and tb.column("leader_frame_file")[0].as_py() == "derived/frames/leader/000001.jpg"
    assert tf.column("leader_T_world_cam").null_count == 1 and tf.column("front_qc_motion")[0].as_py() is None
    assert tf.column("leader_hand_R_present").to_pylist()[:4] == [False, True, True, False]
    assert tf.column("leader_hand_R_p_wearer").to_pylist()[:2] == [None, pytest.approx(0.95)] and tf.column("leader_partner_hand_L_p_wearer").null_count == n
    assert tf.column("leader_hand_R_lm2d")[0].as_py() is None and len(tf.column("leader_hand_R_lm2d")[1].as_py()) == 42
    assert json.loads(tf.column("leader_objects")[0].as_py()) == [["cup", 0.8, 10.0, 20.0, 30.0, 40.0]]
    assert mf["stages"]["hands"]["state"] == "done" and mb["stages"].get("hands") is None and mf["body3d_stream"] == "front"
    assert mf["streams"]["leader"]["hands_schema"] == 2 and mf["streams"]["leader"]["headpose_backend"] == "head_tag"
    assert mf["streams"]["leader"]["hands_wearer"] == "" and mf["streams"]["leader"]["hands_partner"] == ""
    assert mf["streams"]["leader"]["hands_owner_prior"] == 8.0 and mf["streams"]["leader"]["hands_partner_seen_s"] is None  # NaN -> null
    assert mf["streams"]["front"]["frame_size"] == [640, 360] and mf["streams"]["front"]["frame_scale"] == pytest.approx(0.5)  # from manifest.json
    # the fixture is the real C4 layout: perception's own readers accept it
    from duet.playground import perception as P
    assert len(P.frame_paths(full, "leader")) == n and P.frame_scale(full, full.stream("leader")) == pytest.approx(0.5)
    assert {"name": "leader_hand_L_lm3d", "type": "list<float32>", "length": 63, "unit": "m", "frame": "hand_wrist"} in mf["columns"]
    assert "t_ref = (1 + drift_ppm*1e-6) * t_stream + offset_s" in mf["time_model"] and mf["frames"]["board"]
    assert "t_stream = (1 + imu_drift_ppm*1e-6) * t_imu + imu_offset_s" in mf["time_model"]


def test_export_hands_v1_is_converted(tmp_path):
    n = 9; ep = make_ep(tmp_path, STREAMS, n=n); _full_outputs(ep, n, hands_v2=False); X.export(ep)
    t, m = _read(ep); raw = load_npz(ep.derived / "hands" / "leader.npz")
    lm3d = np.array(t.column("leader_hand_L_lm3d")[1].as_py()).reshape(21, 3)
    np.testing.assert_allclose(lm3d, raw["lm3d"][1, 0] - raw["lm3d"][1, 0, 0], atol=1e-6)  # wrist-relative now
    assert np.allclose(lm3d[0], 0) and m["streams"]["leader"]["hands_schema"] == 1
    assert t.column("leader_partner_hand_L_present").null_count == n and t.column("leader_hands_n_detected").null_count == n
    assert t.column("leader_hand_R_handedness_prob")[0].as_py() is None and t.column("leader_hand_R_handedness_prob")[1].as_py() == pytest.approx(0.9)


def test_export_fails_on_row_count_mismatch(tmp_path):
    n = 12; ep = make_ep(tmp_path, STREAMS, n=n); _full_outputs(ep, n)
    k = load_npz(ep.derived / "body2d" / "front.npz")
    atomic_savez(ep.derived / "body2d" / "front.npz", **{**k, "kpts": np.concatenate([k["kpts"], k["kpts"][:3]]), "boxes": np.concatenate([k["boxes"], k["boxes"][:3]])})
    with pytest.raises(ValueError, match="body2d/front has 15 rows, expected n_frames"):
        X.export(ep)
    atomic_savez(ep.derived / "body2d" / "front.npz", **{**k, "kpts": k["kpts"][:7], "boxes": k["boxes"][:7]})
    with pytest.raises(ValueError, match="7 rows"):
        X.export(ep)
    assert not (ep.derived / "records.parquet").exists()


def test_export_frames_on_disk_must_match_without_index(tmp_path):
    ep = make_ep(tmp_path, n=6); d = ep.derived / "frames" / "ego"; d.mkdir(parents=True)
    for k in range(4):
        (d / f"{k + 1:06d}.jpg").write_bytes(b"")
    done(ep, "frames")
    with pytest.raises(ValueError, match="ego frames has 4 rows"):
        X.export(ep)
    for k in range(4, 6):  # old-code episode with every frame: source times unknown -> null, never guessed
        (d / f"{k + 1:06d}.jpg").write_bytes(b"")
    X.export(ep); t, m = _read(ep)
    assert t.column("ego_t_src_s").null_count == 6 and t.column("ego_src_frame").null_count == 6 and not m["streams"]["ego"]["frames_index"]
    assert t.column("ego_frame_file").to_pylist()[-1] == "derived/frames/ego/000006.jpg"


@pytest.mark.parametrize("ppm", [0.0, 20000.0])  # export alone: no scipy needed; 2 % drift makes the clock model visible in 3 s
def test_export_nearest_imu_sample_and_gaps(tmp_path, ppm):
    n = 30; ep = make_ep(tmp_path, n=n, imu=("ego",), offsets={}); write_frames(ep, "ego", t_src=np.arange(n) / FPS + 0.01)
    t_imu = np.arange(0.0, 3.2, 1 / 24); t_imu = t_imu[(t_imu < 1.0) | (t_imu > 1.5)]  # 0.5 s gap
    pts = np.repeat(t_imu[:, None, None, None], 2, 1) * np.ones((1, 2, 4, 3)); off = -0.4
    atomic_savez(ep.derived / "imu_arm" / "ego.npz", schema=2, t_imu_s=t_imu, points=pts.astype(np.float32), valid=np.ones((len(t_imu), 2), bool),
                 period_s=1 / 24, imu_offset_s=off, imu_drift_ppm=ppm, imu_sync="estimated", t_s=ep.ref_time("ego", (1 + ppm * 1e-6) * t_imu + off))
    done(ep, "frames", "imu_arm"); X.export(ep); t, m = _read(ep)
    tq = (np.arange(n) / FPS + 0.01 - off) / (1 + ppm * 1e-6)  # frame times on the IMU clock
    got = np.array([np.nan if v is None else v for v in t.column("ego_imu_t_src_s").to_pylist()])
    j = np.abs(t_imu[None, :] - tq[:, None]).argmin(1); want = np.where(np.abs(t_imu[j] - tq) <= 0.75 / 24, t_imu[j], np.nan)
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert np.isnan(got[(tq > 1.05) & (tq < 1.45)]).all() and np.isfinite(got[tq < 0.95]).all()  # gap -> null, not held
    arm = t.column("ego_imu_arm_L").to_pylist()
    assert all(a is None for a, g in zip(arm, got, strict=True) if np.isnan(g)) and arm[0][0] == pytest.approx(want[0], abs=1e-6)
    assert m["streams"]["ego"]["imu"]["imu_sync"] == "estimated" and m["streams"]["ego"]["imu"]["imu_drift_ppm"] == ppm


def test_export_survives_unreadable_rig(tmp_path):
    ep = make_ep(tmp_path, STREAMS, n=8); _full_outputs(ep, 8); (ep.dir / "rig.json").write_text("{not json")
    X.export(ep); t, m = _read(ep)
    assert ep.status["export"]["state"] == "done" and "rig.json unreadable" in ep.status["export"]["detail"] and m["rig_error"]
    assert not [c for c in t.column_names if c.startswith("world_object_")] and t.column("leader_hand_L_lm2d").null_count == 0


def test_export_skips_without_frames(tmp_path):
    ep = make_ep(tmp_path); X.export(ep)
    assert ep.status["export"]["state"] == "skipped" and not (ep.derived / "records.parquet").exists()
