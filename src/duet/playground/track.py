"""Person identity tracking: stable ids across frames (and across fixed views).

body2d fills up to 4 slots per frame and links them frame to frame by IoU only, so slot identities swap whenever
people cross or one is briefly lost. This stage re-links the body2d detections into tracks:

  cost(track, detection) = W_IOU * (1 - IoU(predicted box, detection box)) + W_APP * appearance distance
  appearance              = HSV (hue x saturation) histogram of the torso (between shoulders and hips from the
                            keypoints; the middle of the box when the keypoints are missing), read from the frame
                            jpg, kept as an exponential moving average per track
  prediction              = constant-velocity box (velocity = EMA of consecutive box displacements)
  assignment              = Hungarian (scipy.optimize.linear_sum_assignment), matches above COST_MAX rejected
  memory                  = a track survives MEMORY_S seconds without a detection, then closes

Duplicate detections of one person within a frame (IoU > DEDUP_OVERLAP, or near-identical keypoints) are suppressed
first, keeping the higher confidence. Tracks shorter than MIN_TRACK_S are dropped; the remaining tracks are ranked by total presence and the top MAX_TRACKS
kept. Outputs, per stream with body2d, ``derived/track/<stream>.npz``:

  ids       [N, 4]      stable track id of each body2d slot, -1 when the slot is empty or its track was dropped
  kpts      [N, P, 17, 3], boxes [N, P, 5]   body2d re-ordered by stable id (P = number of kept tracks)
  n_tracks  scalar, track_len_s [P], presence [P]
  switch    [N] bool    residual identity-switch flag: a kept track's box centre jumps > SWITCH_FRAC of the image
                        width between consecutive frames (the same test the validation uses on raw slots)
  slot_remap [N] bool   a track id sits in a different body2d slot than in the previous frame (a raw swap corrected)
  global_ids [P]        cross-view person id (-1 when not matched), method in cross_view.json

Cross-view identity (fixed views only; ego wearers are not visible in their own stream, so ego tracks are other people
and are left unmatched):
  world3d    when derived/world3d/world3d.npz has triangulated bodies and derived/calib gives T_world_cam for the
             fixed camera, each track is matched to the world body whose reprojection agrees best (mean keypoint
             distance, Hungarian).
  fallback   without world bodies: tracks are matched across the fixed views by appearance distance plus temporal
             co-occurrence; the left/right ordering agreement of the chosen assignment is reported together with
             the orientation it implies (same / mirrored), which is undetermined without calibration.

``derived/track/report.json`` carries the validation numbers (switches before/after, track counts, lengths) and
``derived/track/records_extra.parquet`` the per-frame columns for the export.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .episode import Episode

W_IOU, W_APP = 1.0, 1.0          # cost weights
COST_MAX = 1.4                   # reject assignments above this cost
MEMORY_S = 2.0                   # keep a lost track alive this long
MIN_TRACK_S = 1.0                # drop shorter tracks
MAX_TRACKS = 4
SWITCH_FRAC = 0.4                # centre jump > this fraction of the image width = identity switch
APP_EMA = 0.15                   # appearance histogram update rate
VEL_EMA = 0.5                    # box velocity update rate
MAX_PRED_GAP = 5                 # constant-velocity prediction is capped at this many frames of extrapolation
H_BINS, S_BINS = 16, 8
KPT_CONF = 0.3
DEDUP_OVERLAP = 0.55             # two detections with IoU above this (or near-identical keypoints) are one person: keep the higher confidence
DEDUP_APP_MAX = 0.4              # ... unless their torso appearance distance exceeds this (two different people overlapping)
SHOULDERS, HIPS = (5, 6), (11, 12)


# ----------------------------------------------------------------------------- helpers

def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))


def dedup(boxes_k: np.ndarray, det: list[int], overlap: float, kpts_k: np.ndarray | None = None, hists: dict | None = None) -> tuple[list[int], list[int]]:
    """Suppress duplicate detections of one person within a frame (YOLO often returns a counter-cut box and a full box
    for the same person). Duplicate = IoU > overlap, or the confident keypoints they share sit within 10 % of the box
    height of each other; never when the torso appearances clearly differ (two people overlapping). Containment alone
    is NOT used: a small person can sit entirely inside a bigger person's box. Returns (kept slots, dropped slots)."""
    order = sorted(det, key=lambda s: -float(boxes_k[s, 4])); kept: list[int] = []; dropped: list[int] = []
    for s in order:
        a = boxes_k[s, :4]; dup = False
        for t in kept:
            b = boxes_k[t, :4]
            if hists is not None and hists.get(s) is not None and hists.get(t) is not None and appearance_distance(hists[s], hists[t]) > DEDUP_APP_MAX:
                continue
            if _iou(a, b) > overlap:
                dup = True; break
            if kpts_k is not None and np.isfinite(kpts_k[s, 0, 0]) and np.isfinite(kpts_k[t, 0, 0]):
                good = (kpts_k[s, :, 2] > KPT_CONF) & (kpts_k[t, :, 2] > KPT_CONF)
                if good.sum() >= 5:
                    d = np.linalg.norm(kpts_k[s, good, :2] - kpts_k[t, good, :2], axis=1); h = max(a[3] - a[1], b[3] - b[1], 1.0)
                    if np.median(d) < 0.1 * h:
                        dup = True; break
        (dropped if dup else kept).append(s)
    return sorted(kept), dropped


def torso_region(box, kp, img_w, img_h) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) pixel rectangle of the torso: shoulders-to-hips from the keypoints, else the box middle."""
    pts = None
    if kp is not None and np.isfinite(kp[:, 0]).all():
        sh = kp[list(SHOULDERS)]; hp = kp[list(HIPS)]
        if (sh[:, 2] > KPT_CONF).all() and (hp[:, 2] > KPT_CONF).all():
            xs = np.concatenate([sh[:, 0], hp[:, 0]]); y0 = sh[:, 1].mean(); y1 = hp[:, 1].mean()
            if y1 - y0 > 4 and xs.max() - xs.min() > 4:
                pts = (xs.min(), y0, xs.max(), y1)
    if pts is None:
        w, h = box[2] - box[0], box[3] - box[1]
        pts = (box[0] + 0.25 * w, box[1] + 0.2 * h, box[2] - 0.25 * w, box[1] + 0.6 * h)
    x0, y0, x1, y1 = (int(round(float(v))) for v in pts)
    x0, x1 = max(0, min(x0, img_w - 2)), max(1, min(x1, img_w - 1)); y0, y1 = max(0, min(y0, img_h - 2)), max(1, min(y1, img_h - 1))
    if x1 <= x0: x1 = x0 + 1
    if y1 <= y0: y1 = y0 + 1
    return x0, y0, x1, y1


def appearance(img_bgr: np.ndarray, region) -> np.ndarray:
    """Normalised hue x saturation histogram of a region (value channel ignored: lighting-invariant-ish)."""
    x0, y0, x1, y1 = region
    hsv = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [H_BINS, S_BINS], [0, 180, 0, 256]).astype(np.float32)
    return hist / (hist.sum() + 1e-9)


def appearance_distance(h1: np.ndarray | None, h2: np.ndarray | None) -> float:
    if h1 is None or h2 is None:
        return 0.5  # unknown: neutral
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))  # 0 identical .. 1 disjoint


@dataclass
class _Track:
    tid: int
    box: np.ndarray                # last matched box [4]
    vel: np.ndarray                # px / frame [4]
    hist: np.ndarray | None
    last: int                      # last matched frame
    frames: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)

    def predict(self, k: int) -> np.ndarray:
        g = min(k - self.last, MAX_PRED_GAP)
        return self.box + self.vel * g


# ----------------------------------------------------------------------------- core tracker

def link_tracks(boxes: np.ndarray, kpts: np.ndarray | None, images, img_w: int, img_h: int, fps: float,
                memory_s: float = MEMORY_S, min_track_s: float = MIN_TRACK_S, max_tracks: int = MAX_TRACKS, dedup_overlap: float = DEDUP_OVERLAP) -> dict:
    """Link body2d slots into stable tracks.

    boxes [N, S, 5], kpts [N, S, 17, 3] or None, images: callable k -> BGR image (or None for no appearance).
    Returns dict(ids [N,S], kpts [N,P,17,3], boxes [N,P,5], n_tracks, track_len_s [P], presence [P], switch [N],
    slot_remap [N], hists [P]).
    """
    n, S = boxes.shape[:2]; memory = int(round(memory_s * fps)); active: list[_Track] = []; closed: list[_Track] = []; next_id = 0
    raw_ids = np.full((n, S), -1, np.int64)
    dup = np.zeros((n, S), bool)
    for k in range(n):
        det = [s for s in range(S) if np.isfinite(boxes[k, s, 4])]
        img = images(k) if (images is not None and det) else None
        hists = {}
        for s in det:
            hists[s] = appearance(img, torso_region(boxes[k, s, :4], kpts[k, s] if kpts is not None else None, img_w, img_h)) if img is not None else None
        det, dropped = dedup(boxes[k], det, dedup_overlap, kpts[k] if kpts is not None else None, hists); dup[k, dropped] = True
        for t in list(active):
            if k - t.last > memory:
                active.remove(t); closed.append(t)
        assigned: dict[int, int] = {}
        if active and det:
            C = np.zeros((len(active), len(det)))
            for i, t in enumerate(active):
                pb = t.predict(k)
                for j, s in enumerate(det):
                    C[i, j] = W_IOU * (1.0 - _iou(pb, boxes[k, s, :4])) + W_APP * appearance_distance(t.hist, hists[s])
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] <= COST_MAX:
                    assigned[det[j]] = i
        for s in det:
            b = boxes[k, s, :4].astype(np.float64)
            if s in assigned:
                t = active[assigned[s]]
                if k - t.last == 1:
                    t.vel = (1 - VEL_EMA) * t.vel + VEL_EMA * (b - t.box)
                else:
                    t.vel = np.zeros(4)
                t.box = b; t.last = k
                if hists[s] is not None:
                    t.hist = hists[s] if t.hist is None else (1 - APP_EMA) * t.hist + APP_EMA * hists[s]
            else:
                t = _Track(next_id, b, np.zeros(4), hists[s], k); next_id += 1; active.append(t)
            t.frames.append(k); t.slots.append(s); raw_ids[k, s] = t.tid
    tracks = sorted(active + closed, key=lambda t: (-len(t.frames), t.frames[0]))
    kept = [t for t in tracks if len(t.frames) >= int(round(min_track_s * fps))][:max_tracks]
    P = len(kept); remap = {t.tid: i for i, t in enumerate(kept)}
    ids = np.full((n, S), -1, np.int64)
    out_k = np.full((n, P, 17, 3), np.nan, np.float32); out_b = np.full((n, P, 5), np.nan, np.float32)
    for i, t in enumerate(kept):
        for k, s in zip(t.frames, t.slots):
            ids[k, s] = i; out_b[k, i] = boxes[k, s]
            if kpts is not None:
                out_k[k, i] = kpts[k, s]
    switch = switch_flags(out_b, img_w) if P else np.zeros(n, bool)
    slot_remap = np.zeros(n, bool)
    for k in range(1, n):
        for i in range(P):
            a = np.where(ids[k - 1] == i)[0]; b = np.where(ids[k] == i)[0]
            if len(a) and len(b) and a[0] != b[0]:
                slot_remap[k] = True
    return {"ids": ids, "kpts": out_k, "boxes": out_b, "n_tracks": P, "track_len_s": np.array([len(t.frames) / fps for t in kept], np.float32),
            "presence": np.array([len(t.frames) for t in kept], np.int64), "switch": switch, "slot_remap": slot_remap,
            "hists": [t.hist for t in kept], "raw_ids": raw_ids, "n_raw_tracks": next_id, "duplicates": dup}


def switch_flags(boxes: np.ndarray, img_w: int, frac: float = SWITCH_FRAC, require_two: bool = True) -> np.ndarray:
    """Per-frame flag: some slot's box centre jumps > frac * img_w between consecutive frames (both frames with 2 people
    when require_two). boxes [N, S, 5]."""
    n = boxes.shape[0]; cx = (boxes[..., 0] + boxes[..., 2]) / 2.0; present = np.isfinite(boxes[..., 4]); npeople = present.sum(1)
    jump = np.abs(cx[1:] - cx[:-1]) > frac * img_w
    both = present[1:] & present[:-1]
    flag = (jump & both).any(1)
    if require_two:
        flag &= (npeople[1:] == 2) & (npeople[:-1] == 2)
    return np.concatenate([[False], flag])


def count_switches(boxes: np.ndarray, img_w: int) -> int:
    return int(switch_flags(boxes, img_w).sum())


# ----------------------------------------------------------------------------- cross-view identity

def _project(intr, T_world_cam, pts_world):
    from . import geometry as G
    return G.project(intr, G.inv(T_world_cam), pts_world)


def match_tracks_to_world(track_kpts: np.ndarray, world_bodies: np.ndarray, intr, T_world_cam: np.ndarray, img_w: int) -> tuple[np.ndarray, np.ndarray, int]:
    """track_kpts [N,P,17,3] (image px), world_bodies [N,B,17,3] (m). Returns (global id per track [P], mean normalised
    reprojection distance per track, number of frames used). Unmatched -> -1."""
    n = min(track_kpts.shape[0], world_bodies.shape[0]); P, B = track_kpts.shape[1], world_bodies.shape[1]
    D = np.full((P, B), np.nan); cnt = np.zeros((P, B), int); used = 0
    acc = np.zeros((P, B));
    for k in range(n):
        wb = world_bodies[k]; ok_b = np.isfinite(wb[:, :, 0]).sum(1) >= 5
        if not ok_b.any():
            continue
        proj = {b: _project(intr, T_world_cam, wb[b]) for b in range(B) if ok_b[b]}
        for p in range(P):
            kp = track_kpts[k, p]
            if not np.isfinite(kp[0, 0]):
                continue
            for b, pr in proj.items():
                good = np.isfinite(pr[:, 0]) & np.isfinite(wb[b][:, 0]) & (kp[:, 2] > KPT_CONF)
                if good.sum() < 5:
                    continue
                acc[p, b] += np.linalg.norm(pr[good] - kp[good, :2], axis=1).mean() / img_w; cnt[p, b] += 1
        used += 1
    D = np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0)
    gid = np.full(P, -1, np.int64); dist = np.full(P, np.nan)
    if P and B:
        ri, ci = linear_sum_assignment(D)
        for p, b in zip(ri, ci):
            if cnt[p, b] >= 5 and D[p, b] < 0.1:  # agree within 10 % of the image width on >= 5 frames
                gid[p] = b; dist[p] = D[p, b]
    return gid, dist, used


def match_tracks_across_views(views: dict[str, dict]) -> tuple[dict[str, np.ndarray], dict]:
    """Fallback without world bodies. views: stream -> dict(boxes [N,P,5], hists [P], img_w).
    Global ids are the track indices of the first (reference) view; other views are assigned by Hungarian on
    cost = appearance distance + (1 - temporal co-occurrence). Reports the left/right rank agreement of the chosen
    assignment and the orientation it implies."""
    names = list(views); info = {"method": "cooccurrence+appearance (no triangulated bodies)", "pairs": {}}
    if not names:
        return {}, info
    ref = names[0]; out = {ref: np.arange(views[ref]["boxes"].shape[1])}

    def ranks(v):
        b = v["boxes"]; cx = (b[..., 0] + b[..., 2]) / 2.0; pres = np.isfinite(b[..., 4])
        r = np.full(cx.shape, np.nan)
        for k in range(cx.shape[0]):
            idx = np.where(pres[k])[0]
            if len(idx) >= 2:
                order = np.argsort(cx[k, idx]); r[k, idx[order]] = np.arange(len(idx)) / (len(idx) - 1)
        return pres, r

    pres_r, rank_r = ranks(views[ref])
    for name in names[1:]:
        v = views[name]; pres_v, rank_v = ranks(v); Pr, Pv = pres_r.shape[1], pres_v.shape[1]; n = min(len(pres_r), len(pres_v))
        C = np.zeros((Pr, Pv)); cooc = np.zeros((Pr, Pv)); app = np.zeros((Pr, Pv))
        for i in range(Pr):
            for j in range(Pv):
                both = (pres_r[:n, i] & pres_v[:n, j]).sum(); either = (pres_r[:n, i] | pres_v[:n, j]).sum()
                cooc[i, j] = both / max(either, 1); app[i, j] = appearance_distance(views[ref]["hists"][i], v["hists"][j])
                C[i, j] = app[i, j] + (1.0 - cooc[i, j])
        gid = np.full(Pv, -1, np.int64); margin = None
        if Pr and Pv:
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                gid[j] = i
            # how decisive is it? best total cost vs the best alternative assignment (enumerated: P <= 4)
            from itertools import permutations
            m = min(Pr, Pv); costs = sorted([sum(C[i, j] for i, j in zip(range(m), perm)) for perm in permutations(range(Pv), m)] if Pr <= Pv else
                                 [sum(C[i, j] for i, j in zip(perm, range(m))) for perm in permutations(range(Pr), m)])
            margin = round(float(costs[1] - costs[0]), 4) if len(costs) > 1 else None
        # left/right agreement of the assignment: fraction of co-visible frames where the rank order matches (same) or is mirrored
        same = mirr = tot = 0
        for k in range(n):
            pairs = [(i, j) for j, i in enumerate(gid) if i >= 0 and np.isfinite(rank_r[k, i]) and np.isfinite(rank_v[k, j])]
            if len(pairs) < 2:
                continue
            a = np.array([rank_r[k, i] for i, _ in pairs]); b = np.array([rank_v[k, j] for _, j in pairs]); tot += 1
            same += int(np.array_equal(np.argsort(a), np.argsort(b))); mirr += int(np.array_equal(np.argsort(a), np.argsort(-b)))
        info["pairs"][f"{ref}->{name}"] = {"assignment": {int(j): int(i) for j, i in enumerate(gid)}, "cost": C.round(3).tolist(), "appearance": app.round(3).tolist(),
                                            "cooccurrence": cooc.round(3).tolist(), "frames_covisible_2plus": tot,
                                            "lr_agreement_same": round(same / tot, 3) if tot else None, "lr_agreement_mirrored": round(mirr / tot, 3) if tot else None,
                                            "orientation": ("same" if same >= mirr else "mirrored") if tot else "unknown",
                                            "cost_margin": margin, "ambiguous": bool(margin is not None and margin < 0.1)}
        out[name] = gid
    return out, info


# ----------------------------------------------------------------------------- stage

def _image_reader(frames: list[Path]):
    cache: dict[int, np.ndarray] = {}

    def read(k):
        if k not in cache:
            cache.clear(); cache[k] = cv2.imread(str(frames[k]))
        return cache[k]
    return read


def track(ep: Episode) -> None:
    from .perception import _frame_list
    ep.set_status("track", "running")
    out = ep.derived / "track"; out.mkdir(exist_ok=True)
    report: dict = {"streams": {}, "params": {"w_iou": W_IOU, "w_app": W_APP, "cost_max": COST_MAX, "memory_s": MEMORY_S, "min_track_s": MIN_TRACK_S, "switch_frac": SWITCH_FRAC}}
    results: dict[str, dict] = {}; n_ref = None
    for s in ep.streams:
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if not z.exists():
            continue
        d = np.load(z); boxes, kpts = d["boxes"], d["kpts"]; img_w, img_h = int(d["img_w"]), int(d["img_h"])
        frames = _frame_list(ep, s); n = min(len(boxes), len(frames)) if frames else len(boxes)
        boxes, kpts = boxes[:n], kpts[:n]; n_ref = n if n_ref is None else min(n_ref, n)
        r = link_tracks(boxes, kpts, _image_reader(frames) if frames else None, img_w, img_h, ep.proc_fps)
        r["img_w"], r["img_h"] = img_w, img_h; results[s.name] = r
        report["streams"][s.name] = {"role": s.role, "switches_before": count_switches(boxes, img_w), "switches_after": count_switches(r["boxes"], img_w),
                                     "n_tracks": int(r["n_tracks"]), "n_raw_tracks": int(r["n_raw_tracks"]), "mean_track_len_s": float(r["track_len_s"].mean()) if r["n_tracks"] else 0.0,
                                     "track_len_s": r["track_len_s"].round(1).tolist(), "slot_remaps": int(r["slot_remap"].sum()), "duplicate_detections": int(r["duplicates"].sum()),
                                     "frames_with_2_people": int((np.isfinite(boxes[..., 4]).sum(1) == 2).sum())}
    if not results:
        ep.set_status("track", "skipped", "no body2d results"); return
    # cross-view identity on the fixed views
    exo = [s.name for s in ep.exos() if s.name in results]; global_ids = {name: np.full(results[name]["n_tracks"], -1, np.int64) for name in results}
    cross: dict = {"method": "none", "streams": exo}
    w3 = ep.derived / "world3d" / "world3d.npz"; cal = {}
    try:
        from .world import load_calib; cal = load_calib(ep)
    except Exception as e:  # noqa: BLE001
        cross["calib_error"] = str(e)
    have_bodies = False
    if w3.exists():
        bodies = np.load(w3)["bodies"]; have_bodies = bool(np.isfinite(bodies[:, :, :, 0]).any())
    if have_bodies and any(name in cal and cal[name][1] is not None for name in exo):
        cross["method"] = "world3d reprojection"; cross["per_stream"] = {}
        for name in exo:
            if name in cal and cal[name][1] is not None:
                intr, T_wc = cal[name]; gid, dist, used = match_tracks_to_world(results[name]["kpts"], bodies, intr, T_wc, results[name]["img_w"])
                global_ids[name] = gid; cross["per_stream"][name] = {"global_ids": gid.tolist(), "mean_reproj_frac_img_w": [None if not np.isfinite(v) else round(float(v), 4) for v in dist], "frames_used": used}
            else:
                cross["per_stream"][name] = {"global_ids": None, "note": "no T_world_cam for this camera"}
    elif len(exo) >= 1:
        views = {name: {"boxes": results[name]["boxes"], "hists": results[name]["hists"], "img_w": results[name]["img_w"]} for name in exo}
        gids, info = match_tracks_across_views(views); cross.update(info); cross["reference_view"] = exo[0]
        cross["note"] = ("world3d has no triangulated bodies" if w3.exists() else "no world3d") + "; global id = track index in the reference view; orientation (same/mirrored) is not determined without calibration"
        for name, g in gids.items():
            global_ids[name] = g
    cross["global_ids"] = {name: g.tolist() for name, g in global_ids.items()}
    report["cross_view"] = cross
    # write per-stream npz + records_extra
    cols: dict = {}
    for name, r in results.items():
        np.savez_compressed(out / f"{name}.npz", ids=r["ids"], kpts=r["kpts"], boxes=r["boxes"], n_tracks=r["n_tracks"], track_len_s=r["track_len_s"], presence=r["presence"],
                            switch=r["switch"], slot_remap=r["slot_remap"], global_ids=global_ids[name], img_w=r["img_w"], img_h=r["img_h"])
        n = r["ids"].shape[0]
        cols[f"{name}_n_people"] = (r["ids"] >= 0).sum(1); cols[f"{name}_switch"] = r["switch"]; cols[f"{name}_slot_ids"] = [json.dumps(row.tolist()) for row in r["ids"]]
        for i in range(r["n_tracks"]):
            b = r["boxes"][:, i]; cols[f"{name}_t{i}_cx"] = (b[:, 0] + b[:, 2]) / (2.0 * r["img_w"]); cols[f"{name}_t{i}_cy"] = (b[:, 1] + b[:, 3]) / (2.0 * r["img_h"])
            cols[f"{name}_t{i}_global"] = np.full(n, int(global_ids[name][i]), np.int64)
    n_rows = max(len(v) for v in cols.values())
    pd.DataFrame({k: (list(v) + [np.nan] * (n_rows - len(v))) if len(v) < n_rows else v for k, v in cols.items()}).to_parquet(out / "records_extra.parquet", index=False)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    summ = "; ".join(f"{k}: {v['n_tracks']} tracks, switches {v['switches_before']}->{v['switches_after']}" for k, v in report["streams"].items())
    ep.set_status("track", "done", f"{summ}; cross-view: {cross['method']}")
