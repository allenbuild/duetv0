"""Export an episode to the Duet per-frame record, derived/records.parquet (schema v2, contract C8).

One row per processed frame k on the common reference timeline at proc_fps. The column set depends only on the
episode configuration (streams, roles, which streams carry an IMU, rig.json object tags), never on which stages ran
or which files happen to exist. A stage's columns are read only when ep.stage_ok(stage) and are null otherwise.
Every per-frame artifact must have exactly ep.n_frames() rows, else export fails. Nothing is truncated or padded.
Missing values: null = not available (stage not done, stream unusable, entity absent, no IMU sample within
tolerance). NaN inside a list = one component missing within a present entity (e.g. an undetected keypoint).

Columns (<s> = stream; [d] = list<float32> of length d, row-major; units and frames also in the metadata):
  frame_idx (int32), t_ref_s (float64, nominal reference time common_start_s + k / proc_fps)
  every stream   <s>_t_src_s (float64, stream time of the source frame used; null without a frames index),
                 <s>_src_frame (int64), <s>_frame_file (str, relative to the episode dir),
                 <s>_body2d_p{0..3} [51] 17 COCO x (x px, y px, conf) and <s>_body2d_box_p{0..3} [5] (x0, y0, x1, y1,
                 conf) in extracted-frame pixels (slots = tracker ids, stable within this stream only),
                 <s>_qc_sharp, <s>_qc_bright, <s>_qc_motion (float32, qc.UNITS)
  ego streams    <s>_hand_{L,R}_present (bool), _handedness_prob (float32, MediaPipe handedness probability, not a
                 detection confidence), _p_wearer (float32, probability the hand is the wearer's; hands v2 only),
                 _lm2d [42] (21 x (x, y) px), _lm3d [63] (21 x xyz m, WRIST-relative, MediaPipe world axes);
                 <s>_partner_hand_{L,R}_* the same for the other person's hands (hands v2 only; slot = handedness);
                 <s>_hands_n_detected (int16, v2 only); <s>_objects (JSON [[name, conf, x0, y0, x1, y1], ...] px);
                 <s>_T_world_cam [16] (row-major 4x4, camera -> board/world, OpenCV camera axes, m)
  IMU streams    <s>_imu_arm_{L,R} [12] (shoulder, elbow, wrist, fingertip xyz, m, imu_arm.FRAME; fingertip NaN
                 without a hand sensor), <s>_imu_t_src_s (float64, ORIGINAL IMU-clock time of the nearest raw sample,
                 used only if within IMU_TOL_PERIODS sample periods of the frame's time on the IMU clock)
  world (board)  world_body_p{0,1} [51] (17 x xyz m), world_hands3d_<ego>_{L,R} [63], world_head_<ego> [3],
                 world_object_<name> [3], world_feat_<name> (float32)
  body3d         body3d_p{0,1} [99] (33 x xyz m, hip-centred MediaPipe world frame of the body3d stream; monocular)
Parquet key-value metadata: "duet.schema_version" = "2"; "duet.records" = JSON (episode config and time model, offsets,
drift, frame sizes, stage states and fingerprints, IMU sync, hands schema, units/frames, column list).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .episode import Episode, Stream
from .runtime import atomic_write_parquet, load_npz

SCHEMA_VERSION = 2
IMU_TOL_PERIODS = 0.75  # a frame takes the nearest raw IMU sample only within this many median periods
N_BODY2D, N_WORLD_BODIES, N_BODY3D = 4, 2, 2
FRAMES = {"px": "extracted-frame pixels (derived/frames, see manifest scale)", "board": "world = calibration board frame, metres",
          "hand_wrist": "MediaPipe hand world axes, metres, origin at the wrist landmark",
          "body3d_hip": "MediaPipe pose world frame, metres, origin between the hips (monocular, per view)",
          "imu_torso": "see imu_arm.FRAME", "camera": "OpenCV camera axes (x right, y down, z forward)"}


def _pa():
    import pyarrow as pa
    return pa


def _rows(a: np.ndarray, n: int, what: str) -> np.ndarray:
    if a.shape[0] != n:  # never truncate or pad: the artifact and the episode disagree
        raise ValueError(f"{what} has {a.shape[0]} rows, expected n_frames() = {n} (stale or inconsistent output; re-run the stage)")
    return a


def _lists(v: np.ndarray, ok: np.ndarray):
    """[n, ...] -> list<float32> of each flattened row, null where ok is False."""
    pa = _pa(); n = len(v); flat = np.ascontiguousarray(v, np.float32).reshape(n, -1); ok = np.asarray(ok, bool)
    offs = np.r_[0, np.cumsum(np.where(ok, flat.shape[1], 0))].astype(np.int32)
    return pa.ListArray.from_arrays(pa.array(offs), pa.array(flat[ok].reshape(-1), pa.float32()), mask=pa.array(~ok))


def _floats(v, ok=None, typ: str = "float32"):
    """Numbers -> float column, null where not finite or not ok."""
    pa = _pa(); v = np.asarray(v, np.float64); ok = np.isfinite(v) & (True if ok is None else np.asarray(ok, bool))
    return pa.array(np.where(ok, v, 0).astype(typ), getattr(pa, typ)(), mask=~ok)


def _nearest(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Index of the nearest t (sorted) for each q."""
    if len(t) == 1:
        return np.zeros(len(q), int)
    j = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    return np.where(np.abs(t[j - 1] - q) <= np.abs(t[j] - q), j - 1, j)


class _Table:
    """Ordered columns with declared arrow types: every declared column exists, null unless filled."""

    def __init__(self, n: int):
        self.n, self.types, self.data, self.doc = n, {}, {}, []

    def add(self, name: str, typ: str, length: int | None = None, unit: str = "", frame: str = "") -> None:
        assert name not in self.types, f"duplicate column {name}"
        self.types[name] = typ; self.doc.append({"name": name, "type": typ, "length": length, "unit": unit, "frame": frame})

    def set(self, name: str, arr) -> None:
        assert name in self.types and len(arr) == self.n, (name, len(arr), self.n)
        self.data[name] = arr

    def table(self):
        pa = _pa()
        ty = {"int32": pa.int32(), "int16": pa.int16(), "int64": pa.int64(), "float32": pa.float32(), "float64": pa.float64(),
              "bool": pa.bool_(), "string": pa.string(), "list<float32>": pa.list_(pa.float32())}
        fields = [pa.field(k, ty[t]) for k, t in self.types.items()]
        cols = [self.data[k].cast(f.type) if k in self.data else pa.nulls(self.n, f.type) for k, f in zip(self.types, fields)]
        return pa.Table.from_arrays(cols, schema=pa.schema(fields))


def _declare(ep: Episode, T: _Table, objects: list[str], feats: list[str]) -> None:
    """The v2 column set: a function of the episode configuration only."""
    T.add("frame_idx", "int32"); T.add("t_ref_s", "float64", unit="s", frame="reference timeline")
    for s in ep.streams:
        n = s.name
        T.add(f"{n}_t_src_s", "float64", unit="s", frame=f"stream {n} timeline (0 = first video frame)")
        T.add(f"{n}_src_frame", "int64"); T.add(f"{n}_frame_file", "string")
        for p in range(N_BODY2D):
            T.add(f"{n}_body2d_p{p}", "list<float32>", 51, "px, px, conf", "px")
            T.add(f"{n}_body2d_box_p{p}", "list<float32>", 5, "px x4, conf", "px")
        if s.role == "ego":
            for who in ("", "partner_"):
                for side in "LR":
                    T.add(f"{n}_{who}hand_{side}_present", "bool")
                    T.add(f"{n}_{who}hand_{side}_handedness_prob", "float32", unit="probability")
                    T.add(f"{n}_{who}hand_{side}_p_wearer", "float32", unit="probability")
                    T.add(f"{n}_{who}hand_{side}_lm2d", "list<float32>", 42, "px", "px")
                    T.add(f"{n}_{who}hand_{side}_lm3d", "list<float32>", 63, "m", "hand_wrist")
            T.add(f"{n}_hands_n_detected", "int16"); T.add(f"{n}_objects", "string", unit="JSON [name, conf, x0, y0, x1, y1] px", frame="px")
            T.add(f"{n}_T_world_cam", "list<float32>", 16, "m", "camera -> board")
        if s.imu:
            for side in "LR":
                T.add(f"{n}_imu_arm_{side}", "list<float32>", 12, "m", "imu_torso")
            T.add(f"{n}_imu_t_src_s", "float64", unit="s", frame=f"IMU clock of {n} (time_ms / 1000)")
        T.add(f"{n}_qc_sharp", "float32", unit="grey levels^2"); T.add(f"{n}_qc_bright", "float32", unit="grey level 0-255")
        T.add(f"{n}_qc_motion", "float32", unit="frame widths (long side) per s")
    for p in range(N_WORLD_BODIES):
        T.add(f"world_body_p{p}", "list<float32>", 51, "m", "board")
    for s in ep.egos():
        for side in "LR":
            T.add(f"world_hands3d_{s.name}_{side}", "list<float32>", 63, "m", "board")
        T.add(f"world_head_{s.name}", "list<float32>", 3, "m", "board")
    for o in objects:
        T.add(f"world_object_{o}", "list<float32>", 3, "m", "board")
    for f in feats:
        T.add(f"world_feat_{f}", "float32", unit="m" if f.endswith("_m") else "")
    for p in range(N_BODY3D):
        T.add(f"body3d_p{p}", "list<float32>", 99, "m", "body3d_hip")


def world_features(ep: Episode) -> list[str]:
    """Every cross-person feature world3d may write (feat_<name>), from the episode config: min_wrist_dist_m (two
    triangulated bodies), plus head_dist_m and <ego>_facing_partner_cos for each ego when there are >= 2 egos."""
    e = [s.name for s in ep.egos()]
    return ["min_wrist_dist_m"] + (["head_dist_m"] + [f"{x}_facing_partner_cos" for x in e] if len(e) >= 2 else [])


def object_names(ep: Episode) -> list[str]:
    """Object tag names from rig.json (world3d writes object_<name> for each, when tags ran). Read directly rather than
    through world.rig (which also validates sizes and ids) so export needs no cv2. ValueError if rig.json is unreadable."""
    p = ep.dir / "rig.json"
    rig = json.loads(p.read_text()) if p.exists() else {}  # json.JSONDecodeError is a ValueError
    tags = rig.get("object_tags") if isinstance(rig, dict) else None
    if tags is not None and not isinstance(tags, dict):
        raise ValueError(f"rig.json object_tags must map names to tag ids, got {type(tags).__name__}")
    return sorted(str(k) for k in (tags or {}))


def _frames(ep: Episode, s: Stream, T: _Table, n: int, info: dict) -> np.ndarray:
    """Frame table columns (perception.frame_index, C4); returns the frames' stream times: the source pts, or nominal
    for episodes extracted by the old code (no index: t_src_s NaN and src_frame -1, exported as null)."""
    from .perception import frame_index
    df = frame_index(ep, s); k = _rows(df["k"].to_numpy(), n, f"{s.name} frames")
    if not np.array_equal(k, np.arange(n)):
        raise ValueError(f"{s.name}: frames index k is not 0..{n - 1}")
    t = df["t_src_s"].to_numpy(np.float64); src = df["src_frame"].to_numpy(np.int64); info["frames_index"] = bool(np.isfinite(t).any())
    T.set(f"{s.name}_t_src_s", _floats(t, typ="float64")); T.set(f"{s.name}_src_frame", _pa().array(src, mask=src < 0))
    T.set(f"{s.name}_frame_file", _pa().array([f"derived/frames/{s.name}/{Path(f).name}" for f in df["file"]], _pa().string()))
    return np.where(np.isfinite(t), t, ep.frame_times_stream(s))


def _body2d(z: dict, s: Stream, T: _Table, n: int, info: dict) -> None:
    k, b = _rows(z["kpts"], n, f"body2d/{s.name}"), _rows(z["boxes"], n, f"body2d/{s.name} boxes")
    for p in range(min(N_BODY2D, k.shape[1])):
        ok = np.isfinite(b[:, p, 4]) | np.isfinite(k[:, p, :, 0]).any(1)
        T.set(f"{s.name}_body2d_p{p}", _lists(k[:, p], ok)); T.set(f"{s.name}_body2d_box_p{p}", _lists(b[:, p], ok))
    if "img_w" in z:
        info.setdefault("frame_size", [int(z["img_w"]), int(z["img_h"])])


def _hands(z: dict, s: Stream, T: _Table, n: int, info: dict) -> None:
    """Hands v2 (wearer L/R + partner) or v1 (slot = handedness label of ANY hand; hand-centred lm3d; score 0 = absent)."""
    pa = _pa(); v2 = int(z.get("schema", 1)) >= 2; info["hands_schema"] = 2 if v2 else 1
    info.update({f"hands_{k}": str(z[k]) for k in ("wearer", "partner") if k in z})  # person labels ("" unknown)
    info.update({f"hands_{k}": float(z[k]) for k in ("owner_prior", "partner_seen_s") if k in z})  # attribution diagnostics
    groups = [("", "")] + ([("partner_", "partner_")] if v2 else [])
    for who, key in groups:
        lm2d, lm3d = _rows(z[f"{key}lm2d"], n, f"hands/{s.name}"), _rows(z[f"{key}lm3d"], n, f"hands/{s.name}").astype(np.float64)
        present = z[f"{key}present"].astype(bool) if v2 else np.isfinite(lm2d[:, :, 0, 0])
        if str(z.get("lm3d_origin", "wrist" if v2 else "hand_centre")) != "wrist":
            lm3d = lm3d - lm3d[:, :, :1]  # hand-centred -> wrist-relative (landmark 0): a pure translation
        for i, side in enumerate("LR"):
            c = f"{s.name}_{who}hand_{side}"
            T.set(f"{c}_present", pa.array(present[:, i])); T.set(f"{c}_handedness_prob", _floats(z[f"{key}score"][:, i], present[:, i]))
            if f"{key}p_wearer" in z:
                T.set(f"{c}_p_wearer", _floats(_rows(z[f"{key}p_wearer"], n, f"hands/{s.name}")[:, i], present[:, i]))
            T.set(f"{c}_lm2d", _lists(lm2d[:, i], present[:, i])); T.set(f"{c}_lm3d", _lists(lm3d[:, i], present[:, i]))
    if v2 and "n_detected" in z:
        T.set(f"{s.name}_hands_n_detected", pa.array(_rows(z["n_detected"], n, f"hands/{s.name}").astype(np.int16)))


def _objects(z: dict, s: Stream, T: _Table, n: int) -> None:
    b, nm = _rows(z["boxes"], n, f"objects/{s.name}"), _rows(z["names"], n, f"objects/{s.name} names")
    rows = [json.dumps([[str(nm[k, j]), round(float(b[k, j, 4]), 3), *[round(float(v), 1) for v in b[k, j, :4]]]
                        for j in np.flatnonzero(np.isfinite(b[k, :, 4]))]) for k in range(n)]
    T.set(f"{s.name}_objects", _pa().array(rows, _pa().string()))


def _imu(z: dict, ep: Episode, s: Stream, T: _Table, n: int, t_frames: np.ndarray, info: dict) -> None:
    """Nearest raw IMU sample per frame (never the next grid sample, never held across gaps)."""
    if "t_imu_s" in z:  # schema 2: original IMU sample times + applied clock model t_stream = (1 + drift) * t_imu + offset
        drift = float(z["imu_drift_ppm"]) * 1e-6 if "imu_drift_ppm" in z else 0.0
        t = z["t_imu_s"]; tq = (t_frames - float(z["imu_offset_s"])) / (1 + drift); tol = IMU_TOL_PERIODS * float(z["period_s"])
        info["imu"] = {k: z[k].item() for k in ("imu_sync", "imu_sync_reason", "imu_offset_s", "imu_drift_ppm", "imu_estimate_status",
                                                "imu_offset_estimate_s", "imu_drift_estimate_ppm", "imu_drift_check", "imu_offset_r",
                                                "imu_offset_p", "imu_offset_peak_ratio", "period_s") if k in z and z[k].ndim == 0}
    else:  # legacy 24 Hz grid in reference time; the IMU clock was assumed == the video clock
        t = z["t_s"]; tq = ep.frame_times_ref(); tol = IMU_TOL_PERIODS * float(np.median(np.diff(t)))
        info["imu"] = {"imu_sync": "legacy (IMU clock assumed == video clock, 24 Hz grid)"}
    if not len(t):
        return
    j = _nearest(t, tq); ok = np.abs(t[j] - tq) <= tol; valid = z["valid"][j]
    for si, side in enumerate("LR"):
        T.set(f"{s.name}_imu_arm_{side}", _lists(z["points"][j, si], ok & valid[:, si]))
    if "t_imu_s" in z:
        T.set(f"{s.name}_imu_t_src_s", _floats(t[j], ok, "float64"))


def _world(z: dict, ep: Episode, T: _Table, n: int, objects: list[str], feats: list[str]) -> list[str] | None:
    """World-frame columns; returns body_person (ego linked to each body slot, "" if none) when world3d records it."""
    unknown = sorted(k[5:] for k in z if k.startswith("feat_") and k[5:] not in feats)
    if unknown:
        raise ValueError(f"world3d writes features {unknown} that export.world_features() doesn't declare; update the v2 schema")
    b = _rows(z["bodies"], n, "world3d bodies")
    for p in range(min(N_WORLD_BODIES, b.shape[1])):
        T.set(f"world_body_p{p}", _lists(b[:, p], np.isfinite(b[:, p]).any((1, 2))))
    for s in ep.egos():
        if f"hands3d_{s.name}" in z:
            h = _rows(z[f"hands3d_{s.name}"], n, f"world3d hands3d_{s.name}")
            for i, side in enumerate("LR"):
                T.set(f"world_hands3d_{s.name}_{side}", _lists(h[:, i], np.isfinite(h[:, i]).any((1, 2))))
        if f"head_{s.name}" in z:
            h = _rows(z[f"head_{s.name}"], n, f"world3d head_{s.name}"); T.set(f"world_head_{s.name}", _lists(h, np.isfinite(h).all(1)))
    for o in objects:
        if f"object_{o}" in z:
            v = _rows(z[f"object_{o}"], n, f"world3d object_{o}"); T.set(f"world_object_{o}", _lists(v, np.isfinite(v).all(1)))
    for f in feats:
        if f"feat_{f}" in z:
            T.set(f"world_feat_{f}", _floats(_rows(z[f"feat_{f}"], n, f"world3d feat_{f}")))
    return [str(x) for x in z["body_person"]] if "body_person" in z else None


def _frame_sizes(ep: Episode) -> dict:
    p = ep.derived / "frames" / "manifest.json"
    m = json.loads(p.read_text()) if p.exists() else {}
    return {k: {kk: v.get(kk) for kk in ("width", "height", "scale")} for k, v in (m.get("streams") or {}).items() if isinstance(v, dict)}


def export(ep: Episode) -> None:
    ep.set_status("export", "running")
    if not ep.stage_ok("frames"):
        ep.set_status("export", "skipped", "needs frames"); return
    n = ep.n_frames(); feats = world_features(ep)
    try:
        objects, rig_error = object_names(ep), None
    except ValueError as e:  # the world stages fail on such a rig too; everything else is still exported
        objects, rig_error = [], f"rig.json unreadable, world_object_* columns omitted: {e}"
    T = _Table(n); _declare(ep, T, objects, feats)
    ok = lambda st: ep.stage_ok(st); pa = _pa(); missing = []; sizes = _frame_sizes(ep); streams = {}
    T.set("frame_idx", pa.array(np.arange(n, dtype=np.int32))); T.set("t_ref_s", pa.array(ep.frame_times_ref()))

    def npz(stage: str, name: str) -> dict | None:
        z = ep.derived / stage / name
        if not ok(stage):
            return None
        if not z.exists():
            missing.append(f"{stage}/{name}"); return None
        return load_npz(z)

    for s in ep.streams:
        info = {"role": s.role, "person": s.person, "usable": ep.usable(s), "offset_s": s.offset_s, "offset_status": s.offset_status,
                "offset_confidence": s.offset_confidence, "drift_ppm": s.drift_ppm, "video_start_s": s.video_start_s, "fps": s.fps,
                "width": s.width, "height": s.height, "rotation": s.rotation, **({"frame_size": [sizes[s.name]["width"], sizes[s.name]["height"]],
                "frame_scale": sizes[s.name]["scale"]} if s.name in sizes else {})}
        streams[s.name] = info
        if not ep.usable(s):
            continue
        t_frames = _frames(ep, s, T, n, info)
        if (z := npz("body2d", f"{s.name}.npz")) is not None:
            _body2d(z, s, T, n, info)
        if (z := npz("qc", f"{s.name}.npz")) is not None:
            T.set(f"{s.name}_qc_sharp", _floats(_rows(z["sharp"], n, f"qc/{s.name}"))); T.set(f"{s.name}_qc_bright", _floats(_rows(z["bright"], n, f"qc/{s.name}")))
            if int(z.get("schema", 1)) >= 2:
                T.set(f"{s.name}_qc_motion", _floats(_rows(z["motion"], n, f"qc/{s.name} motion")))
            else:
                info["qc_motion"] = "qc v1 motion (px/frame, row 0 = 0) not exported"
        if s.role == "ego":
            if (z := npz("hands", f"{s.name}.npz")) is not None:
                _hands(z, s, T, n, info)
            if (z := npz("objects", f"{s.name}.npz")) is not None:
                _objects(z, s, T, n)
            if (z := npz("headpose", f"{s.name}.npz")) is not None:
                M = _rows(z["T_world_cam"], n, f"headpose/{s.name}").reshape(n, 16)
                v = (z["valid"].astype(bool) if "valid" in z else True) & np.isfinite(M).all(1)
                T.set(f"{s.name}_T_world_cam", _lists(M, v)); info["headpose_backend"] = str(z["backend"])
        if s.imu and (z := npz("imu_arm", f"{s.name}.npz")) is not None:
            _imu(z, ep, s, T, n, t_frames, info)
    body_person = None
    if (z := npz("world3d", "world3d.npz")) is not None:
        body_person = _world(z, ep, T, n, objects, feats)
    body3d_stream = None
    if (z := npz("body3d", "body3d.npz")) is not None:
        w = _rows(z["world"], n, "body3d"); body3d_stream = str(z["stream"])
        for p in range(min(N_BODY3D, w.shape[1])):
            T.set(f"body3d_p{p}", _lists(w[:, p], np.isfinite(w[:, p]).any((1, 2))))
    meta = {"schema_version": SCHEMA_VERSION, "episode": ep.name, "reference": ep.reference, "proc_fps": ep.proc_fps, "proc_size": ep.proc_size,
            "common_start_s": ep.common_start_s, "common_end_s": ep.common_end_s, "n_frames": n,
            "time_model": "video: t_ref = (1 + drift_ppm*1e-6) * t_stream + offset_s (per stream; t_stream: 0 = the stream's first "
                          "video frame); IMU: t_stream = (1 + imu_drift_ppm*1e-6) * t_imu + imu_offset_s (per IMU stream, "
                          "streams.<s>.imu; t_imu = time_ms / 1000 on the IMU clock)",
            "nulls": "null = not available (stage not done, stream unusable, entity absent, no IMU sample within tolerance); "
                     "NaN inside a list = component missing within a present entity",
            "imu_tolerance_periods": IMU_TOL_PERIODS, "streams": streams, "body3d_stream": body3d_stream, "world_body_person": body_person, "object_names": objects,
            "world_features": feats, "rig_error": rig_error, "missing_outputs": missing, "frames": FRAMES,
            "stages": {k: {kk: v.get(kk) for kk in ("state", "fingerprint", "finished", "detail")} for k, v in ep.status.items()},
            "columns": T.doc}
    _extra_columns(ep, T, n, info if "info" in dir() else {})
    table = T.table(); out = ep.derived / "records.parquet"
    atomic_write_parquet(table, out, {"duet.schema_version": str(SCHEMA_VERSION), "duet.records": meta})
    nulls = [st for st in ("body2d", "hands", "objects", "headpose", "world3d", "body3d", "qc", "imu_arm") if not ok(st)]
    ep.set_status("export", "done", f"{table.num_rows} rows x {table.num_columns} columns -> {out.relative_to(ep.dir)}"
                  + (f"; null (stage not done): {nulls}" if nulls else "") + (f"; missing outputs: {missing}" if missing else "")
                  + (f"; {rig_error}" if rig_error else ""))


def _extra_columns(ep: Episode, T: _Table, n: int, info: dict) -> None:
    """Merge derived/<stage>/records_extra.parquet from every done stage: one row per common-timeline frame (extra rows
    are cut, missing rows padded with null), columns prefixed with the stage name. Types are inferred from pandas dtypes:
    bool, integer, float, string, or list<float32> for object columns holding sequences."""
    import pandas as pd
    pa = _pa()
    for extra in sorted(ep.derived.glob("*/records_extra.parquet")):
        stage = extra.parent.name
        if not ep.stage_ok(stage):
            continue
        try:
            df = pd.read_parquet(extra).iloc[:n].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            info[f"extra_{stage}"] = f"unreadable: {e}"; continue
        if len(df) < n:
            df = df.reindex(range(n))
        for c in df.columns:
            name = c if c.startswith(stage + "_") else f"{stage}_{c}"
            if name in T.types:
                info[f"extra_{stage}_{c}"] = "duplicate column skipped"; continue
            col = df[c]
            try:
                if pd.api.types.is_bool_dtype(col):
                    typ, arr = "bool", pa.array(col.astype(object).where(col.notna(), None).tolist(), pa.bool_())
                elif pd.api.types.is_integer_dtype(col):
                    typ, arr = "int64", pa.array(col.astype("Int64").tolist(), pa.int64())
                elif pd.api.types.is_float_dtype(col):
                    typ, arr = "float64", pa.array(col.to_numpy(np.float64), pa.float64())
                elif col.map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any():
                    typ, arr = "list<float32>", pa.array([None if v is None or (isinstance(v, float) and np.isnan(v)) else [float(x) for x in v] for v in col], pa.list_(pa.float32()))
                else:
                    typ, arr = "string", pa.array([None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in col], pa.string())
            except Exception as e:  # noqa: BLE001
                info[f"extra_{stage}_{c}"] = f"skipped: {e}"; continue
            T.add(name, typ, unit="", frame=f"records_extra from {stage}"); T.set(name, arr)
