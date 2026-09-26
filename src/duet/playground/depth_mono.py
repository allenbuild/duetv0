"""Monocular depth for head cameras that have no depth sensor (the cheap rig).

Runs Depth Anything (relative depth) on ego frames at a low rate and converts it to metres by scaling
against anchors of known distance in the same frame: AprilTag corners (tag pose gives the tag centre
depth) or, when present, ZED depth. Without any anchor the map is left relative and flagged.

Output derived/depth_mono/<stream>.npz: depth [N_sub, h, w] float16 metres (or relative), frame_idx,
scale_source, and per-frame scale factor. Downscaled (default 256 px wide) to keep the file small; this
is for "how far is the bowl from the hand", not for surface reconstruction.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .episode import Episode
from .perception import _frame_list

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def depth_mono(ep: Episode, every_s: float = 0.5, out_w: int = 256) -> None:
    ep.set_status("depth_mono", "running")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except Exception as e:  # noqa: BLE001
        ep.set_status("depth_mono", "skipped", f"transformers/torch unavailable: {e}"); return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID); model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device).eval()
    out_dir = ep.derived / "depth_mono"; out_dir.mkdir(exist_ok=True); done = []
    step = max(1, int(round(every_s * ep.proc_fps)))
    tags_dir = ep.derived / "tags"
    for s in ep.egos():
        zed_depth = (ep.dir / "zed" / s.name / "depth")
        if zed_depth.exists():
            continue  # real depth exists; nothing to estimate
        frames = _frame_list(ep, s); idx = list(range(0, len(frames), step)); maps = []; scales = []; sources = []
        for k in idx:
            img = cv2.cvtColor(cv2.imread(str(frames[k])), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                inp = proc(images=img, return_tensors="pt").to(device); pred = model(**inp).predicted_depth[0].float().cpu().numpy()
            # Depth Anything V2 (hf) outputs relative inverse-ish depth; larger = closer. Convert to a relative depth ordering in [0, 1].
            rel = pred.max() - pred; rel = rel / (rel.max() + 1e-6)
            rel = cv2.resize(rel, (out_w, int(out_w * img.shape[0] / img.shape[1])))
            scale, source = np.nan, "relative"
            # anchor: wall/object tags visible in this ego frame with known size -> metric distance from tag geometry
            det = _ego_tag_depths(ep, s.name, k, img.shape)
            if det:
                us, vs, zs = zip(*det); sy, sx = rel.shape[0] / img.shape[0], rel.shape[1] / img.shape[1]
                r = np.array([rel[int(v * sy), int(u * sx)] for u, v in zip(us, vs)]); z = np.array(zs)
                if (r > 0.02).all():
                    scale, source = float(np.median(z * r)), "apriltag"  # metres = scale / rel  (rel ~ 1/depth)
            maps.append(rel.astype(np.float16)); scales.append(scale); sources.append(source)
        np.savez_compressed(out_dir / f"{s.name}.npz", rel=np.stack(maps) if maps else np.zeros((0, 1, 1), np.float16), frame_idx=np.array(idx), scale=np.array(scales, np.float32),
                            source=np.array(sources), note="metres ~= scale / rel where scale is finite; rel is relative inverse depth otherwise")
        done.append(f"{s.name} ({sum(np.isfinite(scales))}/{len(scales)} frames metric)")
    ep.set_status("depth_mono", "done" if done else "skipped", "; ".join(done) if done else "no ego stream without ZED depth")


def _ego_tag_depths(ep: Episode, stream: str, k: int, shape) -> list[tuple[float, float, float]]:
    """Tags detected directly in this ego frame (with the ego intrinsics) give (u, v, z_m) anchors."""
    from . import geometry as G
    from .world import load_calib, rig
    cal = load_calib(ep)
    if stream not in cal:
        return []
    intr = cal[stream][0]; r = rig(ep); f = _frame_list(ep, ep.stream(stream))[k]
    out = []
    for d in G.detect_tags(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE), intr, r["tag_size_m"]):
        if d.T_cam_tag is not None and d.T_cam_tag[2, 3] > 0.1:
            out.append((float(d.corners_px[:, 0].mean()), float(d.corners_px[:, 1].mean()), float(d.T_cam_tag[2, 3])))
    return out
