"""Quality scores, per stream and per episode, in the style Eidon used to gate their 1,274 h release
(metadata.parquet: good_frame_percent, hand_presence_ratio, stability_score, lighting_score,
average_brightness, average_optical_flow). Ours are computed on the extracted frames so they are
cheap and reproducible. Every constant is in THR, which is written into the report so it can be tuned. The verdict
levels for hand presence (reject < 20 %, flag < 30 %) and brightness (reject < 20, flag < 40) follow the
tracker-pov card. They measure mean HSV V; we measure mean grey, which is <= V.

Per frame:  sharp    variance of the Laplacian of the grey extracted frame (grey levels^2; depends on frame size)
            bright   mean grey level, 0-255
            motion   median Farneback optical-flow magnitude between frames k-1 and k, computed with the long side
                     resized to flow_long_px and expressed in frame widths (LONG side) per SECOND over the frames'
                     actual source-time difference, so it doesn't depend on proc_fps or orientation; NaN at k = 0
            present  ego: a wearer hand detected (hands v2; v1 counts any hand); exo: a person box (body2d); NaN = unknown
Per stream: good_frame_percent (sharp & bright_min < bright < bright_max), hand/person_presence_ratio,
            stability_score = clip(1 - median(motion) / motion_ref_w_per_s, 0, 1), lighting_score (fraction of frames
            with bright_min <= bright <= bright_max), alignment info, per-metric flags and a verdict.
Episode:    both_visible_ratio (frames where some exo view shows >= 2 people), hands_all_egos_ratio (frames where
            every ego view shows a wearer hand), minimum good-frame percent and alignment confidence, and a verdict.
A metric whose input is missing (stage not done, stream unusable, no frames) is None, with the reason in "missing".
Aggregates are None unless every stream they need has the input. Verdict per metric: "reject" / "flag" below the
THR["verdict"] levels, "missing" when None. Overall it is "reject" if anything rejects, else "flag" if anything is
flagged or missing, else "pass".
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .episode import Episode, Stream
from .perception import frame_index, frame_paths
from .runtime import atomic_savez, atomic_write_json, load_npz

THR = {
    "sharp_min": 40.0, "bright_min": 35.0, "bright_max": 225.0,
    "flow_long_px": 160,  # frames are resized (INTER_AREA) to this long side before optical flow
    "farneback": {"pyr_scale": 0.5, "levels": 2, "winsize": 9, "iterations": 2, "poly_n": 5, "poly_sigma": 1.1, "flags": 0},
    "motion_ref_w_per_s": 0.1875,  # stability 0 at this median motion (= the old 12 px/frame at 640 px and 10 fps)
    # metric -> [reject below, flag below] (None = level not used)
    "verdict": {"good_frame_percent": [0.5, 0.8], "hand_presence_ratio": [0.2, 0.3], "person_presence_ratio": [None, 0.5],
                "lighting_score": [0.5, 0.8], "average_brightness": [20.0, 40.0], "stability_score": [None, 0.25]},
    "episode_verdict": {"both_visible_ratio": [None, 0.5], "hands_all_egos_ratio": [None, 0.3]},
}
UNITS = {"sharp": "variance of Laplacian of the grey extracted frame (grey levels^2)", "bright": "mean grey level 0-255",
         "motion": "median optical-flow magnitude, frame widths (long side) per second", "t_src_s": "stream time (s) of the source frame"}
CHUNK = 256  # frames per worker task


def _workers() -> int:
    return max(1, int(os.environ.get("PLAYGROUND_QC_WORKERS") or min(8, os.cpu_count() or 1)))


def frame_times(ep: Episode, s: Stream) -> tuple[np.ndarray, bool]:
    """Stream time (s) of each processed frame of s -> (t [n], actual).

    actual=True: the source frames' pts from the frames index (C4 t_src_s), non-decreasing. The same source frame
    serves consecutive k when proc_fps >= the source rate or across VFR gaps; such pairs get no motion sample.
    actual=False: an episode extracted by the old code (no index, t_src_s unknown), so the nominal
    ep.frame_times_stream(s) is returned; those frames lag nominal by ~1 source frame."""
    n = ep.n_frames(); df = frame_index(ep, s)
    if len(df) != n:
        raise ValueError(f"{s.name}: frames index has {len(df)} rows, expected n_frames() = {n}")
    t = df["t_src_s"].to_numpy(np.float64)
    if np.isfinite(t).all():
        assert np.all(np.diff(t) >= 0), f"{s.name}: frame source times must not decrease"
        return t, True
    return np.asarray(ep.frame_times_stream(s), np.float64), False


def _gray(path: Path, reduce: int = 1) -> np.ndarray:
    import cv2
    g = cv2.imread(str(path), {1: cv2.IMREAD_GRAYSCALE, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4}[reduce])
    if g is None:
        raise ValueError(f"cannot read frame {path}")
    return g


def _flow_img(g: np.ndarray, long_px: int) -> np.ndarray:
    import cv2
    h, w = g.shape; sc = long_px / max(h, w)
    return g if sc == 1 else cv2.resize(g, (max(1, round(w * sc)), max(1, round(h * sc))), interpolation=cv2.INTER_AREA)


def _chunk(paths: list[Path], t: np.ndarray, lo: int, hi: int, full: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sharp, bright, motion for frames lo..hi-1 (motion[k] needs frame k-1, so a chunk re-reads its predecessor).
    motion[k] is NaN when frames k-1 and k share a source time (a repeated source frame: no elapsed time)."""
    import cv2
    fb = THR["farneback"]; L = THR["flow_long_px"]; red = 1 if full else 4; n = hi - lo
    sharp, bright, motion = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    prev = _flow_img(_gray(paths[lo - 1], red), L) if lo > 0 else None
    for i in range(n):
        g = _gray(paths[lo + i], red)
        if full:
            sharp[i] = cv2.Laplacian(g, cv2.CV_64F).var(); bright[i] = g.mean()
        small = _flow_img(g, L); dt = t[lo + i] - t[lo + i - 1] if lo + i > 0 else np.nan
        if prev is not None and prev.shape == small.shape and dt > 0:
            flow = cv2.calcOpticalFlowFarneback(prev, small, None, fb["pyr_scale"], fb["levels"], fb["winsize"], fb["iterations"],
                                                fb["poly_n"], fb["poly_sigma"], fb["flags"])
            motion[i] = float(np.median(np.hypot(flow[..., 0], flow[..., 1]))) / max(small.shape) / dt
        prev = small
    return sharp, bright, motion


def per_frame(jobs: dict[str, tuple[list[Path], np.ndarray]], full: bool = True, workers: int | None = None) -> dict[str, tuple]:
    """{stream: (frame paths, frame times)} -> {stream: (sharp, bright, motion)}. Chunks of every stream run on a
    thread pool (cv2 releases the GIL). full=False computes motion only, from JPEGs decoded at 1/4 size (cheap)."""
    for name, (p, t) in jobs.items():
        assert len(p) == len(t), f"{name}: {len(p)} frames but {len(t)} frame times"
    tasks = [(name, lo, min(len(p), lo + CHUNK)) for name, (p, _) in jobs.items() for lo in range(0, len(p), CHUNK)]
    ex = ThreadPoolExecutor(max_workers=workers or _workers())
    try:
        futs = [ex.submit(_chunk, jobs[nm][0], jobs[nm][1], lo, hi, full) for nm, lo, hi in tasks]
        res = [f.result() for f in futs]
    except BaseException:  # a failed chunk, or SIGTERM/KeyboardInterrupt while waiting: drop queued chunks, re-raise now
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown()
    out = {}
    for name in jobs:
        parts = [r for (nm, _, _), r in zip(tasks, res) if nm == name]
        out[name] = tuple(np.concatenate([p[i] for p in parts]) if parts else np.zeros(0) for i in range(3))
    return out


def camera_motion(ep: Episode, s: Stream) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Per-frame camera motion of stream s for IMU clock estimation -> (t_stream_s [n], motion [n], source) or None.

    motion[k] is the rate over (t[k-1], t[k]], NaN at k = 0. source "qc": qc's signal (frame widths/s); "qc_v1": an
    older qc npz (px/frame, only useful for correlation), times nominal; "frames": computed now from the JPEGs
    (motion only, 1/4-size decode); None when neither qc nor frames is done."""
    try:
        n = ep.n_frames()
    except ValueError:
        return None
    z = ep.derived / "qc" / f"{s.name}.npz"
    if ep.stage_ok("qc") and z.exists():
        q = load_npz(z)
        if len(q["motion"]) == n:
            if int(q.get("schema", 1)) >= 2:
                return q["t_src_s"].astype(np.float64), q["motion"].astype(np.float64), "qc"
            m = q["motion"].astype(np.float64); m[0] = np.nan
            return np.asarray(ep.frame_times_stream(s), np.float64), m, "qc_v1"
    if ep.stage_ok("frames") and ep.usable(s):
        paths = frame_paths(ep, s)
        if len(paths) == n:
            t, _ = frame_times(ep, s)
            return t, per_frame({s.name: (paths, t)}, full=False)[s.name][2], "frames"
    return None


def _grade(values: dict, levels: dict, missing: dict) -> tuple[dict, str, list[str]]:
    """Flags per metric ("pass" | "flag" | "reject" | "missing") and the overall verdict."""
    flags, reasons = {}, []
    for key, v in values.items():
        rej, flg = levels[key]
        if v is None:
            flags[key] = "missing"; reasons.append(f"{key} missing: {missing.get(key, 'not measured')}")
        elif rej is not None and v < rej:
            flags[key] = "reject"; reasons.append(f"{key} {v:.3g} < reject level {rej:g}")
        elif flg is not None and v < flg:
            flags[key] = "flag"; reasons.append(f"{key} {v:.3g} < flag level {flg:g}")
        else:
            flags[key] = "pass"
    fl = set(flags.values())
    return flags, "reject" if "reject" in fl else "flag" if fl & {"flag", "missing"} else "pass", reasons


def _rows_ok(a: np.ndarray, n: int, name: str) -> np.ndarray:
    if a.shape[0] != n:  # a stale or inconsistent upstream output: fail loudly, never truncate
        raise ValueError(f"{name}: upstream output has {a.shape[0]} rows, expected n_frames() = {n}")
    return a


def _presence(ep: Episode, s: Stream, n: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None, dict]:
    """(present [n] bool, n_people [n] | None, reason if unknown, extras). Ego: wearer hands; exo: body2d boxes."""
    stage = "hands" if s.role == "ego" else "body2d"; z = ep.derived / stage / f"{s.name}.npz"
    if not ep.stage_ok(stage):
        return None, None, f"{stage} stage {(ep.status.get(stage) or {}).get('state', 'not run')}", {}
    if not z.exists():
        return None, None, f"no {stage} output for {s.name}", {}
    d = load_npz(z)
    if s.role == "ego":
        v2 = int(d.get("schema", 1)) >= 2
        pres = _rows_ok(d["present"], n, s.name).astype(bool) if v2 else np.isfinite(_rows_ok(d["lm2d"], n, s.name)[:, :, 0, 0])
        ext = {"hands_schema": 2 if v2 else 1}
        if v2 and "partner_present" in d:
            ext["partner_hand_presence_ratio"] = float(_rows_ok(d["partner_present"], n, s.name).astype(bool).any(1).mean())
        return pres.any(1), None, None, ext
    npeople = np.isfinite(_rows_ok(d["boxes"], n, s.name)[:, :, 4]).sum(1)
    return npeople > 0, npeople, None, {}


def qc(ep: Episode) -> None:
    ep.set_status("qc", "running")
    if not ep.stage_ok("frames"):
        ep.set_status("qc", "skipped", "needs frames"); return
    n = ep.n_frames(); out_dir = ep.derived / "qc"
    report = {"schema": 2, "thresholds": THR, "units": UNITS, "streams": {}, "episode": {}}
    usable = [s for s in ep.streams if ep.usable(s)]; jobs, times = {}, {}
    for s in usable:
        paths = frame_paths(ep, s)
        if len(paths) != n:  # never truncate or pad: the frames stage and the episode disagree
            raise ValueError(f"{s.name}: {len(paths)} frames on disk, expected n_frames() = {n}")
        gone = [p for p in paths if not p.exists()]
        if gone:
            raise ValueError(f"{s.name}: {len(gone)} of {n} indexed frames missing on disk (e.g. {gone[0].name}); re-run frames")
        times[s.name] = frame_times(ep, s); jobs[s.name] = (paths, times[s.name][0])
    pf = per_frame(jobs)
    ego_present, exo_people, miss_ep = {}, {}, {}
    for s in ep.streams:
        base = {"role": s.role, "usable": ep.usable(s), "offset_s": s.offset_s, "offset_status": s.offset_status,
                "alignment_confidence": s.offset_confidence}
        keys = ("good_frame_percent", "hand_presence_ratio", "person_presence_ratio", "stability_score", "lighting_score",
                "average_brightness", "average_sharpness", "median_motion_w_per_s", "average_motion_w_per_s")
        r = {**base, "n_frames": 0, **{k: None for k in keys}, "missing": {}}
        applicable = ["good_frame_percent", "hand_presence_ratio" if s.role == "ego" else "person_presence_ratio",
                      "lighting_score", "average_brightness", "stability_score"]
        if s.name not in pf:
            r["missing"] = {k: "stream unaligned (no usable offset), no frames" for k in applicable}
            r["flags"], _, _ = _grade({k: None for k in applicable}, THR["verdict"], r["missing"])
            r["verdict"], r["reasons"] = "reject", ["stream unaligned (no usable offset)"]
            report["streams"][s.name] = r
            (ego_present if s.role == "ego" else exo_people)[s.name] = None
            continue
        sharp, bright, motion = pf[s.name]; t, actual = times[s.name]
        present, npeople, why, ext = _presence(ep, s, n); r.update(ext); r["n_frames"] = n; r["frame_times"] = "source pts" if actual else "nominal"
        good = (sharp > THR["sharp_min"]) & (bright > THR["bright_min"]) & (bright < THR["bright_max"])  # n >= 1 (n_frames())
        lit = (bright >= THR["bright_min"]) & (bright <= THR["bright_max"]); mv = motion[np.isfinite(motion)]
        r.update(good_frame_percent=float(good.mean()), lighting_score=float(lit.mean()), average_brightness=float(bright.mean()),
                 average_sharpness=float(sharp.mean()))
        if len(mv):
            med = float(np.median(mv))
            r.update(median_motion_w_per_s=med, average_motion_w_per_s=float(mv.mean()), stability_score=float(np.clip(1 - med / THR["motion_ref_w_per_s"], 0, 1)))
        else:
            r["missing"]["stability_score"] = "no motion estimate (needs >= 2 frames with increasing source times)"
        pkey = "hand_presence_ratio" if s.role == "ego" else "person_presence_ratio"
        if present is None:
            r["missing"][pkey] = why
        else:
            r[pkey] = float(present.mean())
        (ego_present if s.role == "ego" else exo_people)[s.name] = present if s.role == "ego" else npeople
        r["flags"], r["verdict"], r["reasons"] = _grade({k: r[k] for k in applicable}, THR["verdict"], r["missing"])
        report["streams"][s.name] = r
        atomic_savez(out_dir / f"{s.name}.npz", schema=2, sharp=sharp.astype(np.float32), bright=bright.astype(np.float32),
                     motion=motion.astype(np.float32), t_src_s=t, t_src_actual=actual,
                     present=(present.astype(np.float32) if present is not None else np.full(n, np.nan, np.float32)),
                     n_people=(npeople.astype(np.float32) if npeople is not None else np.full(n, np.nan, np.float32)))
    e = {"n_frames": n, "duration_s": ep.common_end_s - ep.common_start_s}
    for key, per, agg in (("both_visible_ratio", exo_people, lambda a: (a.max(0) >= 2).mean()),  # needs every exo view
                          ("hands_all_egos_ratio", ego_present, lambda a: a.all(0).mean())):      # needs every ego view
        lack = [k for k, v in per.items() if v is None]
        e[key] = float(agg(np.stack(list(per.values())))) if per and not lack else None
        if per and lack:
            miss_ep[key] = f"no {'person counts' if per is exo_people else 'hand presence'} for {lack}"
    gfp = [v["good_frame_percent"] for v in report["streams"].values()]
    e["min_good_frame_percent"] = min(gfp) if None not in gfp else None
    if None in gfp:
        miss_ep["min_good_frame_percent"] = f"no good-frame percent for {[k for k, v in report['streams'].items() if v['good_frame_percent'] is None]}"
    conf = [v["alignment_confidence"] for k, v in report["streams"].items() if k != ep.reference and v["alignment_confidence"] is not None]
    e["min_alignment_confidence"] = min(conf) if conf else None
    graded = {k: e[k] for k, per in (("both_visible_ratio", exo_people), ("hands_all_egos_ratio", ego_present)) if per}  # applicable only
    e["missing"] = miss_ep; e["flags"], v_ep, reasons = _grade(graded, THR["episode_verdict"], miss_ep)
    order = {"pass": 0, "flag": 1, "reject": 2}; worst = max([v_ep] + [v["verdict"] for v in report["streams"].values()], key=order.get)
    e["verdict"] = worst; e["reasons"] = reasons + [f"{k}: {x}" for k, v in report["streams"].items() for x in v["reasons"]]
    report["episode"] = e
    atomic_write_json(out_dir / "report.json", report)
    mg = e["min_good_frame_percent"]
    ep.set_status("qc", "done", f"verdict {worst}; min good-frame {mg:.2f}" if mg is not None else f"verdict {worst}; good-frame percent missing",
                  verdict=worst)
