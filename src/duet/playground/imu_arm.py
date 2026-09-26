"""Upper-body arm chain from a 7-slot IMU harness (Eidon Tracker schema), placed on the episode's reference timeline.

Ported from Eidon Sim (MIT, github.com/Eidon-AI/eidon-sim: docs/kinematics.md, core/mathUtils.ts,
core/chestUtils.ts). Their web app does this live over WebHID; we do it offline on the recorded parquet so the
same arm chain can be drawn next to video and compared with vision.

IMU file: parsed by duet.adapters.eidon_imu, which documents the schema, the slot map and the conventions. Summary:
time_ms is on the IMU clock; quaternions are [x, y, z, w], sensor -> world, world Z-up; ~24 Hz snapshots, 20-40 %
of them sample-and-hold repeats. Only the quaternions are used here.

Eidon's frame mapping (quaternionToVectors): each sensor's device +Y is the bone "forward" (pointing distally) and
device +Z is the bone "up"; scene axes are (-x_dev, z_dev, y_dev) i.e. scene +Y up, +Z forward. Chest yaw (chestUtils
method 2: chest up-vector projected on the horizontal plane) removes heading, so the chain is torso-relative and
invariant to which way the wearer faces.

Chain, output frame FRAME: metres, shoulders at x = -/+0.18, y = 1.40 (standing), z = 0; humerus 0.30 m, radius
0.26 m, hand 0.10 m along each bone's forward vector. (Eidon uses shoulder origins (-0.25, 0.05, +-0.15) in a
different axis convention.) Not registered to the board/world frame.

Validity, per sample: a slot is valid at t when t is one of its raw samples or lies between two of its raw samples
at most MAX_GAP_PERIODS median periods apart. Nothing is extrapolated or bridged across gaps, and in-between
orientations are slerped (hemisphere-safe). A side needs its upper-arm, forearm AND chest slots (without the chest
the torso heading is unknown, so the side is invalid). The fingertip also needs the hand slot, else it is NaN.
Invalid points are NaN.

Time: t_stream = (1 + imu_drift_ppm * 1e-6) * t_imu + imu_offset_s maps the IMU clock onto the stream's video
timeline (C1 t_stream), then t_ref = ep.ref_time(s, t_stream). The card says camera and IMU capture were started
"back to back" (a few ms apart). On eidon_10004 the IMU nevertheless runs ~0.4 s ahead of the video, so the clock
is estimated per recording (estimate_imu_clock): a constant offset, checked for drift on windows of long recordings.
Stream.imu_offset_override_s, when set, is used instead ("manual", drift 0), with the same sign: t_stream = t_imu +
override, so a negative value means the IMU time stamps run ahead of the video. An estimate is applied only when it passes every
SYNC test ("estimated"). Otherwise offset and drift are 0 (the card's claim) and imu_sync says "unsynced" or
"drift_suspected".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..adapters.eidon_imu import SLOT, SLOTS, load_imu, slot_samples
from .episode import Episode
from .runtime import atomic_savez

CHEST = SLOT["chest"]
SIDES = tuple((side, SLOT[f"{side}_shoulder"], SLOT[f"{side}_forearm"], SLOT[f"{side}_hand"]) for side in ("left", "right"))  # upper arm, forearm, hand
L_HUMERUS, L_RADIUS, L_HAND = 0.30, 0.26, 0.10
SHOULDER = {"left": np.array([-0.18, 1.40, 0.0]), "right": np.array([0.18, 1.40, 0.0])}
JOINTS = ("shoulder", "elbow", "wrist", "fingertip")
FRAME = ("eidon_torso: metres, Eidon Sim scene axes (+Y up), heading removed with the chest sensor; shoulders fixed at "
         "(-0.18, 1.40, 0) left and (+0.18, 1.40, 0) right; not registered to the board/world frame")
MAX_GAP_PERIODS = 1.5
# IMU clock estimation. Model: t_stream = (1 + drift_ppm * 1e-6) * t_imu + offset. The offset is applied only if r >= r_min,
# p <= p_max, peak_ratio >= peak_ratio_min, the peak is inside the lag window and >= min_frames frames were compared.
# Drift check (recordings >= 2 x drift_min_window_s): offsets per window of ~drift_window_s (2 windows below 3 x the
# minimum, else >= 3) must agree within half a processed-frame period, or be explained by a well-determined linear
# drift fit (>= drift_min_fit_windows significant windows, residuals within half a frame, slope standard error x
# recording length within half a frame, |slope| <= drift_max_ppm); otherwise the recording is "drift_suspected" and
# no offset is applied.
SYNC = {"max_lag_s": 2.0, "step_s": 0.01, "min_coverage": 0.9, "n_null": 200, "min_shift_s": 5.0,
        "r_min": 0.2, "p_max": 0.01, "peak_ratio_min": 1.3, "min_frames": 100,
        "drift_window_s": 300.0, "drift_min_window_s": 60.0, "drift_min_fit_windows": 3, "drift_max_ppm": 1000.0}


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v [N,3] by quaternions q [N,4] (x,y,z,w)."""
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    u = q[:, :3]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def sensor_vectors(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eidon quaternionToVectors: (fwd, up) in scene space for [N,4] quats."""
    n = len(q)
    fwd_d = quat_rotate(q, np.tile([0.0, 1.0, 0.0], (n, 1)))
    up_d = quat_rotate(q, np.tile([0.0, 0.0, 1.0], (n, 1)))
    to_scene = lambda v: np.stack([-v[:, 0], v[:, 2], v[:, 1]], 1)
    return to_scene(fwd_d), to_scene(up_d)


def yaw_rotate(v: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Rotate scene vectors about +Y by -yaw (remove heading)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], 1)


# ----------------------------------------------------------------------------- samples and validity

def median_period(samples: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    """Median raw sample interval (s) over all slots; NaN with fewer than 2 samples per slot."""
    d = [np.diff(t) for t, _ in samples.values() if len(t) > 1]
    return float(np.median(np.concatenate(d))) if d else float("nan")


def eval_slot(t: np.ndarray, q: np.ndarray, t_eval: np.ndarray, max_gap: float) -> tuple[np.ndarray, np.ndarray]:
    """One slot's orientation at t_eval: ([T,4] unit quats, NaN where invalid; valid [T]).

    Valid where t_eval hits a raw sample, or lies between two raw samples <= max_gap apart. Slerp between samples
    (scipy, shortest arc, so stored q / -q sign flips don't matter). No extrapolation, no bridging of gaps."""
    t_eval = np.asarray(t_eval, np.float64); out = np.full((len(t_eval), 4), np.nan)
    if len(t) == 0 or len(t_eval) == 0:
        return out, np.zeros(len(t_eval), bool)
    j = np.searchsorted(t, t_eval)  # t[j-1] < t_eval <= t[j]
    jc = np.minimum(j, len(t) - 1)
    exact = t[jc] == t_eval
    span = np.where((j > 0) & (j < len(t)), t[jc] - t[np.maximum(j - 1, 0)], np.inf)
    ok = exact | (span <= max_gap)
    if ok.any():
        if len(t) == 1:
            out[ok] = q[0]
        else:
            from scipy.spatial.transform import Rotation, Slerp
            out[ok] = Slerp(t, Rotation.from_quat(q))(t_eval[ok]).as_quat()
    return out, ok


def arm_chain(imu: pd.DataFrame, t_eval: np.ndarray | None = None, max_gap_periods: float = MAX_GAP_PERIODS) -> dict:
    """Build the 3D arm chain (frame FRAME, metres) at the raw IMU sample times (default) or at t_eval (IMU clock, s).

    Returns dict(t_imu_s [T], points [T, 2 sides, 4 joints (shoulder, elbow, wrist, fingertip), 3] (NaN invalid),
    valid [T,2], hand_valid [T,2], slot_valid [T,7], chest_yaw [T] (rad), elbow_flex_deg [T,2], period_s, max_gap_s).
    """
    samples = slot_samples(imu); period = median_period(samples); max_gap = max_gap_periods * period
    if t_eval is None:
        t = np.unique(np.concatenate([ts for ts, _ in samples.values()])) if samples else np.zeros(0)
    else:
        t = np.asarray(t_eval, np.float64)
    T = len(t); Q = {}; V = np.zeros((T, len(SLOTS)), bool)
    for slot, (ts, qs) in samples.items():
        Q[slot], V[:, slot] = eval_slot(ts, qs, t, max_gap)
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    quat = lambda slot: np.where(V[:, slot, None], Q[slot], ident) if slot in Q else np.tile(ident, (T, 1))
    # chest yaw from the chest sensor's up-vector projection (Eidon chestUtils method 2); unknown without the chest
    yaw = np.where(V[:, CHEST], np.arctan2(*sensor_vectors(quat(CHEST))[1][:, [0, 2]].T), np.nan)
    points = np.full((T, 2, 4, 3), np.nan); valid = np.zeros((T, 2), bool); hand_valid = np.zeros((T, 2), bool)
    flex = np.full((T, 2), np.nan)
    for si, (side, up_slot, fa_slot, hand_slot) in enumerate(SIDES):
        ok = V[:, up_slot] & V[:, fa_slot] & V[:, CHEST]; hv = ok & V[:, hand_slot]; y = np.where(ok, yaw, 0.0)
        f_up = yaw_rotate(sensor_vectors(quat(up_slot))[0], y)
        f_fa = yaw_rotate(sensor_vectors(quat(fa_slot))[0], y)
        f_h = yaw_rotate(sensor_vectors(quat(hand_slot))[0], y)
        p0 = np.tile(SHOULDER[side], (T, 1)); p1 = p0 + f_up * L_HUMERUS; p2 = p1 + f_fa * L_RADIUS; p3 = p2 + f_h * L_HAND
        p3[~hv] = np.nan; pts = np.stack([p0, p1, p2, p3], 1); pts[~ok] = np.nan
        points[:, si] = pts; valid[:, si] = ok; hand_valid[:, si] = hv
        flex[:, si] = np.where(ok, np.degrees(np.arccos(np.clip((f_up * f_fa).sum(1), -1, 1))), np.nan)
    return {"t_imu_s": t, "points": points.astype(np.float32), "valid": valid, "hand_valid": hand_valid, "slot_valid": V,
            "chest_yaw": yaw.astype(np.float32), "elbow_flex_deg": flex.astype(np.float32), "period_s": period, "max_gap_s": max_gap}


# ----------------------------------------------------------------------------- IMU -> video clock offset

def rotation_knots(t: np.ndarray, q: np.ndarray, max_gap: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Knots of Theta(t), the cumulative rotation angle (rad), and of V(t), the cumulative valid time (s).

    Returns (t_theta, theta, t_raw, v). Repeated (sample-and-hold) quaternions are not new measurements, so the
    rotation between two distinct samples is spread linearly over the time between them. Raw intervals > max_gap are
    gaps: they add no valid time, and a distinct-sample segment that contains one adds no angle."""
    dt = np.diff(t); ok = dt <= max_gap
    v = np.r_[0.0, np.cumsum(np.where(ok, dt, 0.0))]
    new = np.r_[True, np.abs(np.diff(q, axis=0)).max(1) > 0] if len(q) > 1 else np.ones(len(q), bool)
    new[-1] = True; idx = np.flatnonzero(new)
    ang = 2 * np.arccos(np.clip(np.abs((q[idx[1:]] * q[idx[:-1]]).sum(1)), 0, 1))
    gaps = np.r_[0, np.cumsum(~ok)]; has_gap = gaps[idx[1:]] > gaps[idx[:-1]]
    return t[idx], np.r_[0.0, np.cumsum(np.where(has_gap, 0.0, ang))], t, v


def interval_speed(knots: tuple, a: np.ndarray, b: np.ndarray, min_coverage: float = 0.9) -> np.ndarray:
    """Mean angular speed (rad/s) over IMU-clock intervals (a, b]; NaN where < min_coverage of it has valid data."""
    t_th, th, t_raw, v = knots
    cov = (np.interp(b, t_raw, v) - np.interp(a, t_raw, v)) / (b - a)
    w = (np.interp(b, t_th, th) - np.interp(a, t_th, th)) / (b - a)
    return np.where((a >= t_raw[0]) & (b <= t_raw[-1]) & (cov >= min_coverage), w, np.nan)


def unsynced(reason: str, cfg: dict | None = None) -> dict:
    """An estimate_imu_offset / estimate_imu_clock result with no usable offset."""
    c = {**SYNC, **(cfg or {})}; lags = np.round(np.arange(-c["max_lag_s"], c["max_lag_s"] + c["step_s"] / 2, c["step_s"]), 9)
    return {"status": "unsynced", "reason": reason, "offset_s": math.nan, "r": math.nan, "r_at_zero": math.nan, "p": math.nan, "z": math.nan,
            "null_p99": math.nan, "peak_ratio": math.nan, "n_frames": 0, "lags_s": lags, "curve": np.full(len(lags), np.nan),
            "offset_global_s": math.nan, "drift_ppm": 0.0, "drift_fit_ppm": math.nan, "drift_check": "not run", "window_spread_s": math.nan,
            "window_t_s": np.zeros(0), "window_offset_s": np.zeros(0)}


def estimate_imu_offset(t_frames: np.ndarray, motion: np.ndarray, t_imu: np.ndarray, q: np.ndarray, max_gap: float,
                        cfg: dict | None = None) -> dict:
    """Clock offset between an IMU and a video, as imu_offset_s with t_stream = t_imu + imu_offset_s.

    t_frames [n]: frame times on the stream timeline (s, strictly increasing where finite). motion [n]: camera motion
    rate over (t_frames[k-1], t_frames[k]] in any unit (qc's frame-widths/s), NaN where unknown (k = 0 always).
    t_imu [m], q [m,4]: raw samples of one IMU slot (the chest), IMU clock.
    For every lag L on a grid (+-max_lag_s, step_s) the IMU angular speed from quaternion differences is averaged
    over (t_frames[k-1] - L, t_frames[k] - L] and Pearson-correlated (after sqrt) with motion, over the frames that
    are covered at EVERY lag. The peak is refined with a parabola. Significance: p compares the peak with the
    max-over-lags r of circular shifts of the motion signal (>= min_shift_s, computed exactly by FFT), and
    peak_ratio compares it with the best secondary peak outside the main lobe (where r > peak/2).
    Returns dict(status "estimated" | "unsynced", reason, offset_s (NaN without a peak), r, r_at_zero, p, z (peak vs
    null, in null std), null_p99, peak_ratio, n_frames, lags_s, curve)."""
    c = {**SYNC, **(cfg or {})}; res = unsynced("", c); lags = res["lags_s"]
    t_frames = np.asarray(t_frames, np.float64); motion = np.asarray(motion, np.float64)
    if t_frames.shape != motion.shape or t_frames.ndim != 1:
        raise ValueError(f"t_frames {t_frames.shape} and motion {motion.shape} must be equal-length 1-D arrays")
    if len(t_imu) < 3 or len(t_frames) < 3:
        return {**res, "reason": "too few IMU samples or frames"}
    a, b, m = t_frames[:-1], t_frames[1:], motion[1:]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(m) & (m >= 0)
    ok[ok] &= b[ok] > a[ok]
    idx = np.flatnonzero(ok); knots = rotation_knots(t_imu, q, max_gap)
    for L in lags:  # keep only frames whose interval is covered at every lag (same sample for every lag)
        idx = idx[np.isfinite(interval_speed(knots, a[idx] - L, b[idx] - L, c["min_coverage"]))]
    n = len(idx); res["n_frames"] = n
    min_shift = math.ceil(max(c["min_shift_s"], 2 * c["max_lag_s"]) / float(np.median(b[idx] - a[idx]))) if n else 0
    if n < max(c["min_frames"], 4 * min_shift):
        return {**res, "reason": f"only {n} frames overlap the IMU at every lag (need {max(c['min_frames'], 4 * min_shift)})"}
    x = np.sqrt(m[idx])
    if x.std() == 0:
        return {**res, "reason": "no camera motion"}
    xs = (x - x.mean()) / x.std(); fx = np.conj(np.fft.rfft(xs))
    shifts = np.unique(np.linspace(min_shift, n - min_shift, c["n_null"]).round().astype(int))
    curve = np.full(len(lags), np.nan); null = np.full(len(shifts), -np.inf)
    for i, L in enumerate(lags):
        y = np.sqrt(interval_speed(knots, a[idx] - L, b[idx] - L, 0.0))
        if y.std() == 0:
            continue
        cc = np.fft.irfft(fx * np.fft.rfft((y - y.mean()) / y.std()), n) / n  # cc[s] = r with motion rolled by s
        curve[i] = cc[0]; null = np.maximum(null, cc[shifts])
    res["curve"] = curve
    if not np.isfinite(curve).any():
        return {**res, "reason": "no IMU motion"}
    k = int(np.nanargmax(curve)); r = float(curve[k]); off = float(lags[k])
    if 0 < k < len(lags) - 1 and np.isfinite(curve[k - 1]) and np.isfinite(curve[k + 1]):
        den = curve[k - 1] - 2 * r + curve[k + 1]
        if den < 0:
            off += float(np.clip(0.5 * (curve[k - 1] - curve[k + 1]) / den, -0.5, 0.5)) * c["step_s"]
    lobe = np.zeros(len(lags), bool); lo = hi = k
    while lo > 0 and curve[lo - 1] > r / 2:
        lo -= 1
    while hi < len(lags) - 1 and curve[hi + 1] > r / 2:
        hi += 1
    lobe[lo: hi + 1] = True
    cv = np.where(np.isfinite(curve), curve, -np.inf)
    peaks = np.r_[False, (cv[1:-1] >= cv[:-2]) & (cv[1:-1] >= cv[2:]), False] & ~lobe
    r2 = cv[peaks].max() if peaks.any() else (cv[~lobe].max() if (~lobe).any() else -np.inf)
    ratio = r / r2 if r2 > 0 else math.inf
    p = (1 + int((null >= r).sum())) / (1 + len(null)); z = (r - null.mean()) / null.std() if null.std() > 0 else math.nan
    res.update(offset_s=off, r=r, r_at_zero=float(curve[int(np.argmin(np.abs(lags)))]), p=p, z=float(z),
               null_p99=float(np.percentile(null, 99)), peak_ratio=float(ratio))
    fails = [f"r {r:.2f} < {c['r_min']}"] if r < c["r_min"] else []
    fails += [f"p {p:.3f} > {c['p_max']}"] if p > c["p_max"] else []
    fails += [f"peak ratio {ratio:.2f} < {c['peak_ratio_min']}"] if ratio < c["peak_ratio_min"] else []
    fails += [f"peak at the +-{c['max_lag_s']:g} s window edge"] if abs(lags[k]) >= c["max_lag_s"] - 1.5 * c["step_s"] else []
    return {**res, "status": "unsynced" if fails else "estimated", "reason": "; ".join(fails)}


def estimate_imu_clock(t_frames: np.ndarray, motion: np.ndarray, t_imu: np.ndarray, q: np.ndarray, max_gap: float,
                       cfg: dict | None = None) -> dict:
    """IMU clock model t_stream = (1 + drift_ppm * 1e-6) * t_imu + offset_s, from estimate_imu_offset on the whole
    recording plus the SYNC drift check on windows (arguments as estimate_imu_offset).

    Returns the whole-recording result, where offset_global_s is the constant-offset estimate, plus: offset_s and drift_ppm
    (the model to apply: the global offset with drift 0, or the fitted line when the windows show drift), drift_check
    (what the windows said), drift_fit_ppm (least-squares slope over significant windows, NaN if < 2), window_spread_s,
    window_t_s (window centres, stream time) and window_offset_s (NaN where a window's own estimate failed). status
    "drift_suspected": windows disagree by more than half a processed-frame period and no well-determined line explains it."""
    c = {**SYNC, **(cfg or {})}; g = estimate_imu_offset(t_frames, motion, t_imu, q, max_gap, c)
    res = {**unsynced("", c), **g, "offset_global_s": g["offset_s"]}
    t_frames = np.asarray(t_frames, np.float64); tf = t_frames[np.isfinite(t_frames)]
    if g["status"] != "estimated":
        return {**res, "drift_check": "not run (no constant-offset estimate)"}
    dur, w_min = float(tf[-1] - tf[0]), c["drift_min_window_s"]
    if dur < 2 * w_min:
        return {**res, "drift_check": f"not run (recording {dur:.0f} s < 2 x {w_min:g} s)"}
    nw = 2 if dur < 3 * w_min else max(3, int(dur // c["drift_window_s"]))
    edges = np.linspace(tf[0], tf[-1], nw + 1); centres = (edges[:-1] + edges[1:]) / 2; offs = np.full(nw, np.nan)
    for i in range(nw):
        sel = (t_frames >= edges[i]) & (t_frames <= edges[i + 1])
        w = estimate_imu_offset(t_frames[sel], np.asarray(motion, np.float64)[sel], t_imu, q, max_gap, c)
        offs[i] = w["offset_s"] if w["status"] == "estimated" else np.nan
    good = np.isfinite(offs); tol = 0.5 * dur / (len(tf) - 1)  # half the processed-frame period (repeats make some dt 0)
    res.update(window_t_s=centres, window_offset_s=offs)
    if good.sum() < 2:
        return {**res, "drift_check": f"unverified: {good.sum()} of {nw} windows gave a significant offset"}
    o = offs[good]; ti = centres[good] - o  # window centres on the IMU clock
    d, b = np.polyfit(ti, o, 1); resid = o - (d * ti + b); spread = float(o.max() - o.min())
    res.update(drift_fit_ppm=float(d * 1e6), window_spread_s=spread)
    if spread <= tol:
        return {**res, "drift_check": f"consistent: {good.sum()} of {nw} windows within {spread * 1000:.0f} ms (<= {tol * 1000:.0f} ms)"}
    se = float(np.sqrt((resid ** 2).sum() / (len(o) - 2)) / np.sqrt(((ti - ti.mean()) ** 2).sum())) if len(o) > 2 else math.inf
    fit_ok = (len(o) >= c["drift_min_fit_windows"] and np.abs(resid).max() <= tol and se * dur <= tol  # line trustworthy end to end
              and abs(d) * 1e6 <= c["drift_max_ppm"])
    what = f"{good.sum()} of {nw} windows spread {spread * 1000:.0f} ms (> {tol * 1000:.0f} ms)"
    if fit_ok:
        return {**res, "offset_s": float(b), "drift_ppm": float(d * 1e6),
                "drift_check": f"drift fitted: {what}; {d * 1e6:+.1f} ppm, max residual {np.abs(resid).max() * 1000:.0f} ms"}
    return {**res, "status": "drift_suspected", "drift_check": f"drift suspected: {what}, no well-determined linear fit",
            "reason": f"window offsets disagree ({what}) and no well-determined linear drift fit (needs >= {c['drift_min_fit_windows']} "
                      f"windows, residuals and slope uncertainty x length <= {tol * 1000:.0f} ms, |drift| <= {c['drift_max_ppm']:g} ppm)"}


# ----------------------------------------------------------------------------- stage

def imu_arm(ep: Episode) -> None:
    """Arm chain per IMU stream -> derived/imu_arm/<s>.npz (schema 2) on the reference timeline.

    Clock model applied: t_stream = (1 + imu_drift_ppm * 1e-6) * t_imu + imu_offset_s. imu_sync says where it came from:
    "manual" (Stream.imu_offset_override_s, drift 0; the estimate is still computed and reported), "estimated"
    (estimate_imu_clock passed), else "unsynced" / "drift_suspected" with offset 0 and drift 0 (the card's claim).
    npz: t_imu_s / imu_time_ms (original IMU sample times), t_stream_s, t_s (reference time), points [T,2,4,3] (FRAME, m,
    NaN invalid), valid, hand_valid, slot_valid, chest_yaw (rad), elbow_flex_deg, period_s, imu_sync, imu_offset_s and
    imu_drift_ppm (APPLIED), imu_estimate_status, imu_offset_estimate_s (constant-offset estimate), imu_drift_estimate_ppm,
    imu_drift_check, imu_offset_r / _p / _z / _peak_ratio / _n_frames / _r_at_zero, imu_sync_reason, motion_source,
    sync_lags_s / sync_r (correlation curve), sync_window_t_s / sync_window_offset_s (drift-check windows)."""
    from .qc import camera_motion
    ep.set_status("imu_arm", "running")
    streams = [s for s in ep.streams if s.imu]
    if not streams:
        ep.set_status("imu_arm", "skipped", "no IMU streams"); return
    out_dir = ep.derived / "imu_arm"; done, parts, info = [], [], {}
    for s in streams:
        if not ep.usable(s):
            parts.append(f"{s.name}: stream unaligned, IMU not placed"); continue
        imu = load_imu(ep.dir / s.imu); res = arm_chain(imu); samples = slot_samples(imu)
        if not len(res["t_imu_s"]):
            parts.append(f"{s.name}: no valid IMU samples"); continue
        manual = s.imu_offset_override_s
        try:
            cam = camera_motion(ep, s)
            if CHEST not in samples:
                sync = unsynced("no chest sensor")
            elif cam is None:
                sync = unsynced("no camera-motion signal (qc and frames not done)")
            else:
                sync = estimate_imu_clock(cam[0], cam[1], *samples[CHEST], res["max_gap_s"])
        except Exception as e:  # with a manual offset the estimate is only reported for comparison: never fatal
            if manual is None:
                raise
            cam, sync = None, unsynced(f"estimate failed: {type(e).__name__}: {e}")
        if manual is not None:
            if not math.isfinite(float(manual)):
                raise ValueError(f"{s.name}: imu_offset_override_s must be finite, got {manual!r}")
            status, b, d = "manual", float(manual), 0.0
        elif sync["status"] == "estimated":
            status, b, d = "estimated", sync["offset_s"], sync["drift_ppm"]
        else:
            status, b, d = sync["status"], 0.0, 0.0
        t_stream = (1 + d * 1e-6) * res["t_imu_s"] + b; t_ref = ep.ref_time(s, t_stream)
        assert np.all(np.diff(t_ref) > 0), "IMU times must be strictly increasing"
        atomic_savez(out_dir / f"{s.name}.npz", schema=2, **res, imu_time_ms=np.round(res["t_imu_s"] * 1000).astype(np.int64),
                     t_stream_s=t_stream, t_s=t_ref, imu_sync=status, imu_offset_s=b, imu_drift_ppm=d, imu_estimate_status=sync["status"],
                     imu_sync_reason=sync["reason"], imu_offset_estimate_s=sync["offset_global_s"], imu_drift_estimate_ppm=sync["drift_fit_ppm"],
                     imu_drift_check=sync["drift_check"], imu_offset_r=sync["r"], imu_offset_p=sync["p"], imu_offset_z=sync["z"],
                     imu_offset_peak_ratio=sync["peak_ratio"], imu_offset_n_frames=sync["n_frames"], imu_offset_r_at_zero=sync["r_at_zero"],
                     motion_source=cam[2] if cam is not None else "none", sync_lags_s=sync["lags_s"], sync_r=sync["curve"],
                     sync_window_t_s=sync["window_t_s"], sync_window_offset_s=sync["window_offset_s"],
                     slots=np.array(SLOTS), joints=np.array(JOINTS), frame=FRAME, heading_reference="chest")
        done.append(s.name); v = res["valid"].mean(0)
        info[s.name] = {"sync": status, "applied_offset_s": b, "applied_drift_ppm": d, "estimate_status": sync["status"],
                        **{k: sync[k] for k in ("reason", "offset_global_s", "drift_fit_ppm", "drift_check", "r", "r_at_zero", "p", "z", "peak_ratio", "n_frames")}}
        est = (f"estimate {sync['offset_global_s']:+.3f} s (r {sync['r']:.2f}, p {sync['p']:.3f}, ratio {sync['peak_ratio']:.1f}; {sync['drift_check']})"
               if np.isfinite(sync["offset_global_s"]) else f"no estimate ({sync['reason']})")
        applied = {"manual": f"MANUAL offset {b:+.3f} s applied", "estimated": f"applied {b:+.3f} s" + (f", drift {d:+.1f} ppm" if d else "")}.get(
            status, f"NOT applied ({status}: {sync['reason']})")
        parts.append(f"{s.name}: IMU->video {applied}; {est}; {len(res['t_imu_s'])} samples, valid L {v[0]:.0%} R {v[1]:.0%}")
    ep.set_status("imu_arm", "done" if done else "skipped", "; ".join(parts), imu_sync=info)
