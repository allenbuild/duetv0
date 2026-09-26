"""Gaze proxy from head pose: forward ray, 'looking at partner / object' features.

There is no eye tracker on the playground rigs, so "gaze" is the ego camera's optical axis. Two paths,
chosen per frame:

  3D (headpose)  derived/headpose/<ego>.npz gives T_world_cam (OpenCV: +Z forward). The gaze ray starts at the
                 camera centre along the camera +Z column; the partner's head is world3d/head_<partner> (or the
                 partner's own camera centre); objects are world3d/object_<name> (tag positions).
  2D (image)     no head pose: the ray is the image centre. The partner's head is the mean of the COCO nose/eye
                 keypoints of the best person track in this ego's body2d; its angular offset from the optical
                 axis comes from the calib intrinsics (nominal FOV pinhole when the board was never seen).
                 Objects are the YOLO-World boxes in the ego frame (box centre).

Per frame and ego stream: angle_to_partner_deg, looking_at_partner (< LOOK_DEG), angle_to_nearest_object_deg,
nearest_object, looking_at_object (name or ""), method ("headpose" | "image" | "none"); plus mutual_gaze (both
egos looking at each other). Written to derived/gaze_proxy/<ego>.npz and records_extra.parquet (export merges
it with the prefix "gaze_proxy_").

Validated against real Aria eye tracking on CoMind by scripts/eval_gaze_proxy_comind.py; numbers in
docs/design/stage_gaze_proxy.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import geometry as G
from .episode import Episode, Stream
from .perception import _frame_list

LOOK_DEG = 15.0
HEAD_KPTS = (0, 1, 2)  # COCO nose, left eye, right eye
HEAD_CONF = 0.3
NOMINAL_HFOV_DEG = 110.0


def _intrinsics(ep: Episode, s: Stream, width: int, height: int) -> G.Intrinsics:
    p = ep.derived / "calib" / "calib.json"
    if p.exists():
        cams = json.load(open(p)).get("cameras", {})
        if s.name in cams:
            intr = G.Intrinsics.from_json(cams[s.name]["intrinsics"])
            if intr.width == width and intr.height == height:
                return intr
            sc = width / intr.width; K = intr.K.copy(); K[:2] *= sc
            return G.Intrinsics(K, intr.dist, width, height, intr.rms_px)
    return G.Intrinsics.nominal(width, height, NOMINAL_HFOV_DEG)


def _image_size(ep: Episode, s: Stream) -> tuple[int, int]:
    z = ep.derived / "body2d" / f"{s.name}.npz"
    if z.exists():
        b = np.load(z)
        if "img_w" in b.files:
            return int(b["img_w"]), int(b["img_h"])
    frames = _frame_list(ep, s)
    if frames:
        import cv2
        im = cv2.imread(str(frames[0]))
        if im is not None:
            return im.shape[1], im.shape[0]
    if s.width and s.height:  # proc frames are scaled so the long side is proc_size
        sc = ep.proc_size / max(s.width, s.height)
        return int(round(s.width * sc)), int(round(s.height * sc))
    return ep.proc_size, ep.proc_size


def partner_head_px(kpts: np.ndarray, conf_min: float = HEAD_CONF) -> np.ndarray:
    """body2d kpts [N, P, 17, 3] -> partner head pixel [N, 2] (NaN when no head is visible).

    Per frame, the person slot with the highest mean nose/eye confidence is the partner (the wearer's own head is
    never in the ego image)."""
    n = kpts.shape[0]
    head = kpts[:, :, list(HEAD_KPTS), :]  # [N,P,3,3]
    conf = np.nan_to_num(head[..., 2], nan=0.0)
    good = conf >= conf_min
    score = np.where(good, conf, 0.0).sum(-1)  # [N,P]
    best = score.argmax(1)
    out = np.full((n, 2), np.nan, np.float32)
    for k in range(n):
        p = best[k]
        if score[k, p] <= 0:
            continue
        g = good[k, p]
        out[k] = head[k, p, g, :2].mean(0)
    return out


def pixel_angle_deg(intr: G.Intrinsics, px: np.ndarray) -> np.ndarray:
    """Angle (deg) between the optical axis and the ray through pixel px [N, 2] (pinhole, no distortion)."""
    fx, fy, cx, cy = intr.K[0, 0], intr.K[1, 1], intr.K[0, 2], intr.K[1, 2]
    x = (px[:, 0] - cx) / fx; y = (px[:, 1] - cy) / fy
    return np.degrees(np.arctan(np.hypot(x, y))).astype(np.float32)


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (deg) between row vectors a, b [N, 3]; NaN where either is NaN or zero."""
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    c = (a * b).sum(1) / np.where((na > 0) & (nb > 0), na * nb, np.nan)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0))).astype(np.float32)


def _load_headpose(ep: Episode, s: Stream, n: int):
    z = ep.derived / "headpose" / f"{s.name}.npz"
    if not z.exists():
        return None
    T = np.load(z)["T_world_cam"][:n].astype(np.float64)
    if T.shape[0] < n:
        T = np.concatenate([T, np.full((n - T.shape[0], 4, 4), np.nan)])
    return T if np.isfinite(T[:, 0, 0]).any() else None


def _world3d(ep: Episode):
    z = ep.derived / "world3d" / "world3d.npz"
    return np.load(z) if z.exists() else None


def _objects_3d(ep: Episode, n: int) -> dict[str, np.ndarray]:
    """name -> world position [N, 3] (NaN when unseen): world3d object_<name> if present, else straight from tags."""
    out = {}
    w3 = _world3d(ep)
    if w3 is not None:
        for key in w3.files:
            if key.startswith("object_"):
                v = w3[key][:n].astype(np.float64)
                out[key[len("object_"):]] = np.concatenate([v, np.full((n - len(v), 3), np.nan)]) if len(v) < n else v
    if out or not (ep.derived / "tags").exists():
        return out
    from .world import rig, tag_world_poses
    for name, tid in rig(ep).get("object_tags", {}).items():
        pos = np.full((n, 3), np.nan)
        for k, T in tag_world_poses(ep, int(tid)).items():
            if k < n:
                pos[k] = T[:3, 3]
        if np.isfinite(pos[:, 0]).any():
            out[name] = pos
    return out


def _objects_2d(ep: Episode, s: Stream, n: int):
    z = ep.derived / "objects" / f"{s.name}.npz"
    if not z.exists():
        return None
    o = np.load(z); return o["boxes"][:n], o["names"][:n]


def _stream_features(ep: Episode, s: Stream, partner: Stream | None, n: int, headposes: dict, w3) -> dict:
    width, height = _image_size(ep, s); intr = _intrinsics(ep, s, width, height)
    angle = np.full(n, np.nan, np.float32); method = np.full(n, "none", dtype="<U8"); head_px = np.full((n, 2), np.nan, np.float32)
    # ---- partner, 3D path
    T = headposes.get(s.name)
    if T is not None and partner is not None:
        cam_c = T[:, :3, 3]; fwd = T[:, :3, 2]
        ph = np.full((n, 3), np.nan)
        if w3 is not None and f"head_{partner.name}" in w3.files:
            v = w3[f"head_{partner.name}"][:n].astype(np.float64); ph[: len(v)] = v
        Tp = headposes.get(partner.name)
        if Tp is not None:
            miss = ~np.isfinite(ph[:, 0]); ph[miss] = Tp[miss, :3, 3]
        ok = np.isfinite(fwd[:, 0]) & np.isfinite(ph[:, 0])
        if ok.any():
            angle[ok] = angle_between_deg(fwd[ok], ph[ok] - cam_c[ok]); method[ok] = "headpose"
    # ---- partner, 2D path (frames the 3D path did not cover)
    z = ep.derived / "body2d" / f"{s.name}.npz"
    if z.exists():
        kp = np.load(z)["kpts"][:n]
        hp = partner_head_px(kp); head_px[: len(hp)] = hp
        ok = np.isfinite(head_px[:, 0]) & (method == "none")
        if ok.any():
            angle[ok] = pixel_angle_deg(intr, head_px[ok]); method[ok] = "image"
    looking = np.isfinite(angle) & (angle < LOOK_DEG)
    # ---- objects
    obj_angle = np.full(n, np.nan, np.float32); nearest = np.full(n, "", dtype="<U24")
    objs3 = _objects_3d(ep, n) if T is not None else {}
    if objs3 and T is not None:
        cam_c = T[:, :3, 3]; fwd = T[:, :3, 2]
        for name, pos in objs3.items():
            ok = np.isfinite(fwd[:, 0]) & np.isfinite(pos[:, 0])
            if not ok.any():
                continue
            a = np.full(n, np.nan, np.float32); a[ok] = angle_between_deg(fwd[ok], pos[ok] - cam_c[ok])
            better = np.isfinite(a) & (~np.isfinite(obj_angle) | (a < obj_angle))
            obj_angle[better] = a[better]; nearest[better] = name
    o2 = _objects_2d(ep, s, n)
    if o2 is not None:
        boxes, names = o2; m = len(boxes)
        todo = ~np.isfinite(obj_angle[:m])
        if todo.any():
            cx = 0.5 * (boxes[:, :, 0] + boxes[:, :, 2]); cy = 0.5 * (boxes[:, :, 1] + boxes[:, :, 3])
            a = pixel_angle_deg(intr, np.stack([cx.ravel(), cy.ravel()], 1)).reshape(cx.shape)
            a[~np.isfinite(boxes[:, :, 4])] = np.inf
            j = a.argmin(1); best = a[np.arange(m), j]
            hit = todo & np.isfinite(best)
            idx = np.flatnonzero(hit); obj_angle[idx] = best[idx]; nearest[idx] = names[idx, j[idx]]
    looking_obj = np.where(np.isfinite(obj_angle) & (obj_angle < LOOK_DEG), nearest, "")
    return {"angle_to_partner_deg": angle, "looking_at_partner": looking, "angle_to_nearest_object_deg": obj_angle, "nearest_object": nearest,
            "looking_at_object": looking_obj, "method": method, "partner_head_px": head_px,
            "intrinsics_fx_px": np.float32(intr.K[0, 0]), "image_width": np.int32(width), "image_height": np.int32(height)}


def gaze_proxy(ep: Episode) -> None:
    ep.set_status("gaze_proxy", "running")
    egos = ep.egos()
    if not egos:
        ep.set_status("gaze_proxy", "skipped", "no ego streams"); return
    counts = [len(_frame_list(ep, s)) for s in ep.streams]
    n = min(counts) if counts else 0
    if n == 0:
        ep.set_status("gaze_proxy", "skipped", "no extracted frames"); return
    headposes = {s.name: hp for s in egos if (hp := _load_headpose(ep, s, n)) is not None}
    w3 = _world3d(ep)
    feats = {}
    for s in egos:
        partner = next((o for o in egos if o.name != s.name), None)
        feats[s.name] = _stream_features(ep, s, partner, n, headposes, w3)
    mutual = np.zeros(n, bool)
    if len(egos) >= 2:
        mutual = feats[egos[0].name]["looking_at_partner"] & feats[egos[1].name]["looking_at_partner"]
    out = ep.derived / "gaze_proxy"; out.mkdir(exist_ok=True)
    cols = {}
    for s in egos:
        f = feats[s.name]; f["mutual_gaze"] = mutual
        np.savez_compressed(out / f"{s.name}.npz", **f)
        for key in ("angle_to_partner_deg", "looking_at_partner", "angle_to_nearest_object_deg", "nearest_object", "looking_at_object", "method"):
            cols[f"{s.name}_{key}"] = f[key]
    cols["mutual_gaze"] = mutual
    pd.DataFrame(cols).to_parquet(out / "records_extra.parquet", index=False)
    detail = []
    for s in egos:
        m = feats[s.name]["method"]; cov = float((m != "none").mean())
        src = "headpose" if (m == "headpose").any() else ("image" if (m == "image").any() else "none")
        detail.append(f"{s.name}: {src}, partner seen {cov:.0%}, looking {float(feats[s.name]['looking_at_partner'].mean()):.0%}")
    detail.append(f"mutual {float(mutual.mean()):.0%}")
    any_cov = any((feats[s.name]["method"] != "none").any() for s in egos)
    ep.set_status("gaze_proxy", "done" if any_cov else "skipped", "; ".join(detail) if any_cov else "no head pose and no partner head in any ego view")
