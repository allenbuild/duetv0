"""Monocular depth for head cameras without a depth sensor (opt-in stage: runs only when named in --stages).

Model. Depth Anything V2 Small (relative checkpoint, pinned revision) predicts on the DISTORTED ego frame an
affine-invariant disparity d ~= s / Z + t with unknown s > 0 and t per image (larger d = closer; Z = depth along the
optical axis, metres). Metric depth needs a scale AND a shift: per sampled frame, 1/Z = a*d + b is fitted to depth
anchors by least squares on the relative inverse-depth residual r = (a*d + b)*Z - 1, with pairwise RANSAC when there
are more than 2 anchors (each anchor is kept or dropped on its own). A fit needs >= 2 inlier anchors whose depths
differ by >= min_depth_ratio; without one the frame stays relative and is never labelled metric. A frame without its
own fit may borrow the nearest own fit within borrow_s seconds (fit_source "borrowed"; approximate, the model
normalises every image separately). rig.json {"depth_mono": {"model": "metric_indoor"}} selects the metric indoor
checkpoint instead: it predicts metres (fitted as d = 1/Z_pred) and anchors only correct its scale (b = 0, >= 1
anchor); frames without any scale correction keep the model's metres (fit_source "model").

Anchors. AprilTags detected directly in the ego frame with calib.json's intrinsics (already at extracted-frame size).
Pixel = the tag centre (T_cam_tag translation) projected WITH the lens distortion, i.e. raw frame pixels, the pixels the
model saw; depth = the centre's camera-frame Z. Needs an explicit rig.json "tag_size_m" (black-border edge, metres);
cameras with nominal (FOV-guess) intrinsics give no anchors unless rig.json "allow_nominal" is true.

Output derived/depth_mono/<stream>.npz (schema 2), N sampled frames, maps [N, h, w] at out_w px wide:
  frame_idx [N] processed-frame index k; t_ref_s, t_src_s [N] from frames/<stream>/index.parquet
  depth_m [N,h,w] float16 metres along the optical axis; NaN where the frame has no fit or outside (0, max_depth_m]
  pred [N,h,w] float16 raw model output, units in pred_units ("disparity_affine" or "m")
  metric [N] bool; fit_source [N] "anchors" | "borrowed" | "model" | "none"; fit_from [N] row whose fit is used (-1)
  fit_a, fit_b [N] (1/Z = a*d + b; d = pred, or 1/pred for the metric model); fit_rms [N] RMS residual r of its inliers
  anchor_row, anchor_tag [M]; anchor_uv [M,2] raw frame px; anchor_z_m, anchor_d, anchor_resid [M]; anchor_used [M]
  units "m", model_id, model_revision, model_kind, model_weights ("hub" | "local:<dir>"), intrinsics_source, frame_wh [2]
Streams with real ZED depth (zed/<stream>/depth/*.png) or without a usable time offset are skipped.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G
from . import perception as P
from . import runtime
from .episode import Episode, Stream

SCHEMA = 2
# name -> (Hugging Face model id, pinned commit sha, kind). The shas are the repos' heads resolved from the Hub API on
# 2026-09-26 (Small-hf last modified 2024-07-05, Metric-Indoor-Small-hf 2024-08-27). Override with rig.json
# {"depth_mono": {"model_id": ..., "revision": ..., "kind": "relative" | "metric"}}.
MODELS = {"relative": ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99", "relative"),
          "metric_indoor": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "8078d68a9c75a972131914f6afd0c1723be0da7f", "metric")}
DEFAULTS = dict(model="relative", model_id=None, revision=None, kind=None, every_s=0.5, out_w=256, batch=8, min_depth_ratio=1.5, max_resid=0.10,
                max_tag_err_px=2.0, min_z_m=0.15, max_z_m=15.0, borrow_s=1.0, max_depth_m=20.0)
WEIGHT_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")  # what a pre-fetch must place in weights_dir()


def weights_dir(model_id: str, revision: str | None) -> Path:
    """Pre-fetched weights (`playground.py fetch-models`, SHA-256-checked): runtime.hf_weights_dir, i.e.
    <models_dir>/<model id, "/" -> "--">@<revision>/ holding WEIGHT_FILES. Used instead of the Hub when complete."""
    return runtime.hf_weights_dir(model_id, revision)


def depth_mono(ep: Episode, **overrides) -> None:
    """Opt-in stage (module docstring). overrides (tests, CLI) take precedence over rig.json "depth_mono"."""
    ep.set_status("depth_mono", "running"); cfg = _config(ep, overrides)
    if not ep.stage_ok("frames"):
        ep.set_status("depth_mono", "skipped", "needs frames"); return
    todo, notes = [], []
    for s in ep.egos():
        if not ep.usable(s):
            notes.append(f"{s.name}: no usable time offset")
        elif _has_zed_depth(ep, s):
            notes.append(f"{s.name}: has ZED depth")
        else:
            todo.append(s)
    if not todo:  # the model is loaded only when some stream needs it
        ep.set_status("depth_mono", "skipped", "; ".join(notes) or "no ego stream"); return
    try:
        model = _DepthModel(cfg, runtime.pick_device(ep.device))
    except Exception as e:  # noqa: BLE001 - torch/torchvision/transformers missing, or an offline host without the weights cached
        why = " ".join(str(e).split())[:300]  # transformers' backend errors start with a newline
        ep.set_status("depth_mono", "skipped", f"depth model {cfg['model_id']}@{cfg['revision']} unavailable: {type(e).__name__}: {why}"); return
    out = ep.derived / "depth_mono"
    for s in todo:
        arrays, summary = _stream_depth(ep, s, model, cfg)
        out.mkdir(parents=True, exist_ok=True); runtime.atomic_savez(out / f"{s.name}.npz", **arrays); notes.append(summary)
    ep.set_status("depth_mono", "done", "; ".join(notes))


def _config(ep: Episode, overrides: dict) -> dict:
    p = ep.dir / "rig.json"; rig = json.loads(p.read_text()) if p.exists() else {}
    cfg = {**DEFAULTS, **rig.get("depth_mono", {}), **{k: v for k, v in overrides.items() if v is not None}}
    bad = sorted(set(cfg) - set(DEFAULTS))
    if bad:
        raise ValueError(f"depth_mono: unknown option(s) {bad}")
    mid, rev, kind = MODELS.get(cfg["model"], (None, None, None))
    cfg.update(model_id=cfg["model_id"] or mid, revision=cfg["revision"] or rev, kind=cfg["kind"] or kind)
    if not cfg["model_id"] or cfg["kind"] not in ("relative", "metric"):
        raise ValueError(f"depth_mono: model must be one of {sorted(MODELS)} or model_id + kind ('relative'|'metric'), got {cfg['model']!r}")
    tag = rig.get("tag_size_m")  # explicit only, never defaulted: it sets the scale of every anchor
    assert tag is None or 0.005 <= float(tag) <= 1.0, f"rig.json tag_size_m must be metres (black-border edge), got {tag}"
    assert cfg["every_s"] > 0 and cfg["out_w"] >= 8 and cfg["batch"] >= 1 and cfg["min_depth_ratio"] >= 1.0 and cfg["max_depth_m"] > 0
    return {**cfg, "tag_size_m": None if tag is None else float(tag), "allow_nominal": bool(rig.get("allow_nominal", False))}


def _has_zed_depth(ep: Episode, s: Stream) -> bool:
    """Only a depth/ directory holding depth PNGs counts (older zed_export.py created an empty one with --no-depth)."""
    d = ep.dir / "zed" / s.name / "depth"
    return d.is_dir() and next(d.glob("*.png"), None) is not None


class _DepthModel:
    """Depth Anything through transformers at the pinned revision (pre-fetched weights_dir() if complete, else the Hub),
    eval mode, fp16 on CUDA. predict() takes a batch of RGB frames of one size and returns the raw prediction [B, H, W]
    float32 resized (bilinear) to that frame size."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device, self.half = torch, device, device.startswith("cuda")
        local = weights_dir(cfg["model_id"], cfg["revision"])
        if all((local / f).is_file() for f in WEIGHT_FILES):
            src, kw, self.weights = str(local), {"local_files_only": True}, f"local:{local.name}"
        else:
            src, kw, self.weights = cfg["model_id"], {"revision": cfg["revision"]}, "hub"
        self.proc = AutoImageProcessor.from_pretrained(src, **kw)
        model = AutoModelForDepthEstimation.from_pretrained(src, **kw).to(device).eval()
        self.model = model.half() if self.half else model

    def predict(self, rgb: list[np.ndarray]) -> np.ndarray:
        torch = self.torch; h, w = rgb[0].shape[:2]
        with torch.inference_mode():
            px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.device, torch.float16 if self.half else torch.float32)
            d = self.model(pixel_values=px).predicted_depth.float()
            d = torch.nn.functional.interpolate(d[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        out = d.cpu().numpy(); assert out.shape == (len(rgb), h, w), f"depth prediction {out.shape} for {len(rgb)} frames of {w}x{h}"
        return out


def _anchor_intrinsics(ep: Episode, s: Stream, cfg: dict, wh: tuple[int, int]) -> tuple[G.Intrinsics | None, str]:
    """calib.json intrinsics of this ego camera at frame size, or None and the reason anchors are off."""
    if cfg["tag_size_m"] is None:
        return None, "no tag_size_m in rig.json"
    if not ep.stage_ok("calib"):
        return None, "calib not done"
    try:
        import pupil_apriltags  # noqa: F401
    except ImportError:
        return None, "pupil_apriltags not installed"
    c = json.loads((ep.derived / "calib" / "calib.json").read_text())["cameras"].get(s.name)
    if c is None:
        return None, "camera not in calib.json"
    src = str(c.get("intrinsics_source") or "unknown")
    if src == "nominal_fov" and not cfg["allow_nominal"]:
        return None, "nominal-FOV intrinsics (rig.json allow_nominal is off)"
    intr = G.Intrinsics.from_json(c["intrinsics"])
    if (intr.width, intr.height) != wh:  # calib.json should already be at frame size; rescale deterministically if not
        try:
            src += f" (rescaled {intr.width}x{intr.height} -> {wh[0]}x{wh[1]})"; intr = intr.scaled_to(*wh)
        except G.GeometryError as e:  # a crop or another sensor mode: K cannot be trusted for these frames
            return None, f"calib.json intrinsics do not fit the {wh[0]}x{wh[1]} frames: {e}"
    return intr, src


def tag_anchors(gray: np.ndarray, intr: G.Intrinsics, tag_size_m: float, *, max_err_px: float = 2.0, z_range_m: tuple[float, float] = (0.15, 15.0)) -> list[dict]:
    """Depth anchors from AprilTags in one DISTORTED frame whose size matches intr: tag_id, u, v (raw frame pixels of
    the tag centre, projected through K and the distortion), z_m (centre depth along the optical axis, metres), err_px.
    Detections with reprojection error > max_err_px, depth outside z_range_m or centre outside the frame are dropped."""
    out = []
    for d in G.detect_tags(gray, intr, tag_size_m):
        if d.T_cam_tag is None or d.err_px is None or not d.err_px <= max_err_px:
            continue
        c = np.asarray(d.T_cam_tag, float)[:3, 3]
        if not z_range_m[0] <= c[2] <= z_range_m[1]:
            continue
        u, v = G.project_cam(intr, c[None])[0]  # NaN beyond the lens model's valid range -> dropped below
        if 0 <= u <= intr.width - 1 and 0 <= v <= intr.height - 1:
            out.append({"tag_id": int(d.tag_id), "u": float(u), "v": float(v), "z_m": float(c[2]), "err_px": float(d.err_px)})
    return out


def _sample(m: np.ndarray, u: float, v: float, r: int = 2) -> float:
    """Median of m in a (2r+1)^2 window around pixel (u, v) (robust to the tag's black/white edges)."""
    ui, vi = int(round(u)), int(round(v)); win = m[max(0, vi - r): vi + r + 1, max(0, ui - r): ui + r + 1]
    return float(np.median(win)) if win.size else np.nan


def fit_inverse_depth(d: np.ndarray, z: np.ndarray, *, scale_only: bool = False, min_depth_ratio: float = 1.5, max_resid: float = 0.10):
    """Fit 1/z = a*d + b (b = 0 when scale_only) to anchors: d model disparity (> 0, larger = closer), z metres.

    Residual r = (a*d + b)*z - 1 (relative inverse-depth error). Every minimal subset (pairs whose depths differ by
    >= min_depth_ratio; single anchors when scale_only) is a hypothesis; the one with the most inliers (|r| <=
    max_resid; ties -> lower median |r|) is refitted by least squares on its inliers, so one bad anchor drops alone.
    Returns (a, b, rms of inlier r, inlier mask, r [n]) or None: too few valid anchors, depths too similar, a <= 0."""
    d, z = np.asarray(d, float), np.asarray(z, float); ok = np.isfinite(d) & np.isfinite(z) & (d > 0) & (z > 0)
    idx, need = np.flatnonzero(ok), 1 if scale_only else 2

    def solve(sel):
        A = (d[sel] * z[sel])[:, None] if scale_only else np.c_[d[sel] * z[sel], z[sel]]
        x = np.linalg.lstsq(A, np.ones(len(sel)), rcond=None)[0]; return float(x[0]), 0.0 if scale_only else float(x[1])

    def spread(sel):
        return scale_only or z[sel].max() >= min_depth_ratio * z[sel].min()

    best = None
    for sub in combinations(idx, need):
        if not spread(list(sub)):
            continue
        a, b = solve(list(sub)); r = (a * d + b) * z - 1; inl = ok & (np.abs(r) <= max_resid)
        if a > 0 and inl.any() and (best is None or (inl.sum(), -np.median(np.abs(r[inl]))) > best[0]):
            best = ((inl.sum(), -np.median(np.abs(r[inl]))), inl)
    if best is None or best[1].sum() < need or not spread(np.flatnonzero(best[1])):
        return None
    a, b = solve(np.flatnonzero(best[1])); r = np.where(ok, (a * d + b) * z - 1, np.nan); inl = ok & (np.abs(r) <= max_resid)
    if not a > 0 or inl.sum() < need or not spread(np.flatnonzero(inl)):
        return None
    return a, b, float(np.sqrt(np.mean(r[inl] ** 2))), inl, r


def _borrow(t: np.ndarray, own: np.ndarray, max_dt: float) -> np.ndarray:
    """Row whose fit each row uses: itself when it has its own fit, else the nearest own-fit row within max_dt seconds,
    else -1. t must be increasing."""
    src = np.where(own, np.arange(len(t)), -1); j = np.flatnonzero(own); miss = np.flatnonzero(~own)
    if max_dt <= 0 or not len(j) or not len(miss):
        return src
    pos = np.searchsorted(t[j], t[miss]); lo, hi = j[np.clip(pos - 1, 0, len(j) - 1)], j[np.clip(pos, 0, len(j) - 1)]
    near = np.where(np.abs(t[lo] - t[miss]) <= np.abs(t[hi] - t[miss]), lo, hi)
    src[miss] = np.where(np.abs(t[near] - t[miss]) <= max_dt, near, -1)
    return src


def _stream_depth(ep: Episode, s: Stream, model: _DepthModel, cfg: dict) -> tuple[dict, str]:
    """Run the model on every step-th frame of one ego stream (frame list, calib and rig read once; each JPEG decoded
    once), fit per-frame anchors, borrow, convert to metres. Returns (npz arrays, status summary)."""
    paths = P.frame_paths(ep, s); step = max(1, int(round(cfg["every_s"] * ep.proc_fps))); ks = np.arange(0, len(paths), step)
    if not len(ks):
        raise RuntimeError(f"no frames for {s.name}")
    idx = P.frame_index(ep, s).set_index("k"); t_ref = idx.loc[ks, "t_ref_s"].to_numpy(float); t_src = idx.loc[ks, "t_src_s"].to_numpy(float)
    assert np.all(np.diff(t_ref) > 0), f"{s.name}: sampled frame times are not increasing"
    first = cv2.imread(str(paths[ks[0]]))
    if first is None:
        raise RuntimeError(f"unreadable frame {paths[ks[0]]}")
    H, W = first.shape[:2]; ow = int(cfg["out_w"]); oh = max(1, int(round(ow * H / W))); n = len(ks)
    intr, isrc = _anchor_intrinsics(ep, s, cfg, (W, H)); scale_only = cfg["kind"] == "metric"
    pred = np.empty((n, oh, ow), np.float16); fit = np.full((n, 3), np.nan); own = np.zeros(n, bool); anchors = []
    for i0 in range(0, n, int(cfg["batch"])):
        bgr = [first if k == ks[0] else cv2.imread(str(paths[k])) for k in ks[i0: i0 + int(cfg["batch"])]]
        if any(b is None or b.shape[:2] != (H, W) for b in bgr):
            raise RuntimeError(f"{s.name}: unreadable frame or size != {W}x{H} in rows {i0}..{i0 + len(bgr) - 1}")
        full = model.predict([cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr])
        for i, (b, m) in enumerate(zip(bgr, full, strict=True), start=i0):
            pred[i] = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)
            anc = [] if intr is None else tag_anchors(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), intr, cfg["tag_size_m"], max_err_px=cfg["max_tag_err_px"],
                                                      z_range_m=(cfg["min_z_m"], cfg["max_z_m"]))
            if not anc:
                continue
            raw = np.array([_sample(m, a["u"], a["v"]) for a in anc]); z = np.array([a["z_m"] for a in anc])
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = 1.0 / raw if scale_only else raw
            f = fit_inverse_depth(dd, z, scale_only=scale_only, min_depth_ratio=cfg["min_depth_ratio"], max_resid=cfg["max_resid"])
            used, r = (f[3], f[4]) if f else (np.zeros(len(anc), bool), np.full(len(anc), np.nan))
            if f:
                fit[i], own[i] = f[:3], True
            anchors += [(i, a["tag_id"], a["u"], a["v"], a["z_m"], raw[j], r[j], used[j]) for j, a in enumerate(anc)]
    src = _borrow(t_ref, own, float(cfg["borrow_s"])); has = src >= 0
    a_b = np.where(has[:, None], fit[np.maximum(src, 0), :2], np.nan)
    source = np.where(own, "anchors", np.where(has, "borrowed", "none")).astype("<U8")
    if scale_only:  # the metric checkpoint is metric without anchors, just not scale-corrected
        a_b[~has] = (1.0, 0.0); source[~has] = "model"
    depth = np.full(pred.shape, np.nan, np.float16)
    for i in np.flatnonzero(np.isfinite(a_b[:, 0])):
        dm = pred[i].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = a_b[i, 0] * (1.0 / dm if scale_only else dm) + a_b[i, 1]
            depth[i] = np.where(inv > 1.0 / cfg["max_depth_m"], 1.0 / inv, np.nan)
    A = np.array([x[:6] for x in anchors], float).reshape(-1, 6)
    arrays = {"schema": np.int64(SCHEMA), "frame_idx": ks.astype(np.int64), "t_ref_s": t_ref, "t_src_s": t_src, "depth_m": depth, "pred": pred,
              "pred_units": np.array("m" if scale_only else "disparity_affine"), "units": np.array("m"), "metric": np.isfinite(a_b[:, 0]),
              "fit_source": source, "fit_from": src.astype(np.int64), "fit_a": a_b[:, 0], "fit_b": a_b[:, 1], "fit_rms": np.where(has, fit[np.maximum(src, 0), 2], np.nan),
              "anchor_row": A[:, 0].astype(np.int64), "anchor_tag": A[:, 1].astype(np.int64), "anchor_uv": A[:, 2:4].astype(np.float32),
              "anchor_z_m": A[:, 4].astype(np.float32), "anchor_d": A[:, 5].astype(np.float32),
              "anchor_resid": np.array([x[6] for x in anchors], np.float32), "anchor_used": np.array([x[7] for x in anchors], bool),
              "model_id": np.array(cfg["model_id"]), "model_revision": np.array(str(cfg["revision"])), "model_kind": np.array(cfg["kind"]),
              "model_weights": np.array(model.weights),
              "intrinsics_source": np.array(isrc if intr is not None else f"none: {isrc}"), "frame_wh": np.array([W, H], np.int64)}
    n_b = int((source == "borrowed").sum())
    anc_txt = (f"no anchors: {isrc}" if intr is None else f"{len(anchors)} tag anchors, no usable fit" if not own.any()
               else f"anchors {int(arrays['anchor_used'].sum())}/{len(anchors)} used, median fit rms {np.median(fit[own, 2]):.1%}")
    kind = (f"{int(own.sum())} scale-corrected, {n_b} borrowed, {int((source == 'model').sum())} uncorrected model metres" if scale_only
            else f"{int(arrays['metric'].sum())}/{n} frames metric ({int(own.sum())} anchored, {n_b} borrowed)")
    return arrays, f"{s.name}: {kind}; {anc_txt}"
