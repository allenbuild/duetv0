"""Hand-object contact: which object is in which hand, per frame, on the ego views.

Inputs (per ego stream): derived/hands/<stream>.npz (MediaPipe 21 landmarks, index 0 = "Left", 1 = "Right" as MediaPipe
labels them) and derived/objects/<stream>.npz (YOLO-World boxes + names).

Per frame, per hand:
  candidates   objects whose box (enlarged by BOX_GROW) contains >= MIN_TIPS of the 5 fingertips (landmarks 4, 8, 12,
               16, 20) or contains the palm centre (mean of landmarks 0, 5, 9, 13, 17)
  tie-break    smallest box area
  raw score    fraction of fingertips inside x object confidence (a palm-only candidate with no fingertip inside gets
               PALM_ONLY_FRAC instead of 0 so the score column is never 0 for a held object)
  depth check  when episode/zed/<stream>/depth/*.png exist: the median hand depth (at the 21 landmarks) and the median
               depth of the object box must agree within DEPTH_TOL_M, otherwise the candidate is rejected
Temporal:
  vote         majority vote of the raw label over a VOTE_S window (centred)
  hysteresis   a contact starts after START_S of the same voted label and ends after END_S of a different one; the
               held label is back-filled to the start of the run, so held[] and the events agree

Outputs
  derived/contact/<stream>.npz      held [N,2] (object name or ""), score [N,2], raw_held [N,2], raw_score [N,2],
                                    depth_used (bool)
  derived/contact/events.json       [{"t": reference seconds, "stream", "person", "hand": "L"|"R", "object", "type":
                                    "grasp"|"release"}, ...]
  derived/contact/records_extra.parquet   <stream>_held_L/R, <stream>_contact_score_L/R (export prefixes "contact_")
  derived/contact/report.json       per-stream contact fraction, event count, depth usage
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import Episode, Stream

TIPS = (4, 8, 12, 16, 20)
PALM = (0, 5, 9, 13, 17)
MIN_TIPS = 3
BOX_GROW = 0.10        # enlarge each object box by 10 % (about its centre) before the containment tests
PALM_ONLY_FRAC = 0.2   # score fraction for a palm-only candidate (no fingertip inside)
VOTE_S, START_S, END_S = 0.5, 0.3, 0.5
DEPTH_TOL_M = 0.15
HANDS = ("L", "R")


# ----------------------------------------------------------------------------- per-frame geometry

def grow(box: np.ndarray, frac: float = BOX_GROW) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2; w, h = (box[2] - box[0]) * (1 + frac), (box[3] - box[1]) * (1 + frac)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def inside(pts: np.ndarray, box: np.ndarray) -> np.ndarray:
    return (pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) & (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])


def frame_contact(lm: np.ndarray, boxes: np.ndarray, names: np.ndarray, depth: np.ndarray | None = None, depth_scale: tuple[float, float] = (1.0, 1.0)) -> tuple[str, float, int]:
    """One hand, one frame. lm [21,2] px (NaN when absent), boxes [M,5], names [M].
    Returns (object name or "", raw score, index of the chosen object or -1)."""
    if lm is None or not np.isfinite(lm[:, 0]).all():
        return "", 0.0, -1
    tips = lm[list(TIPS)]; palm = lm[list(PALM)].mean(0, keepdims=True)
    hand_z = None
    if depth is not None:
        hand_z = _median_depth_at(depth, lm, depth_scale)
    best = None
    for j in range(boxes.shape[0]):
        if not np.isfinite(boxes[j, 4]) or not names[j]:
            continue
        b = grow(boxes[j, :4]); n_in = int(inside(tips, b).sum()); palm_in = bool(inside(palm, b)[0])
        if n_in < MIN_TIPS and not palm_in:
            continue
        if depth is not None and hand_z is not None:
            obj_z = _median_depth_box(depth, boxes[j, :4], depth_scale)
            if obj_z is not None and abs(obj_z - hand_z) > DEPTH_TOL_M:
                continue
        area = float((boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1]))
        score = max(n_in / 5.0, PALM_ONLY_FRAC if palm_in else 0.0) * float(boxes[j, 4])
        if best is None or area < best[0]:
            best = (area, j, score)
    if best is None:
        return "", 0.0, -1
    return str(names[best[1]]), float(best[2]), int(best[1])


def _median_depth_at(depth: np.ndarray, pts: np.ndarray, scale) -> float | None:
    u = np.clip((pts[:, 0] * scale[0]).astype(int), 0, depth.shape[1] - 1); v = np.clip((pts[:, 1] * scale[1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]; z = z[z > 0.05]
    return float(np.median(z)) if len(z) else None


def _median_depth_box(depth: np.ndarray, box: np.ndarray, scale) -> float | None:
    x0, x1 = sorted(np.clip((box[[0, 2]] * scale[0]).astype(int), 0, depth.shape[1] - 1)); y0, y1 = sorted(np.clip((box[[1, 3]] * scale[1]).astype(int), 0, depth.shape[0] - 1))
    z = depth[y0:y1 + 1, x0:x1 + 1]; z = z[z > 0.05]
    return float(np.median(z)) if z.size else None


# ----------------------------------------------------------------------------- temporal

def majority_vote(labels: list[str], win: int) -> list[str]:
    n = len(labels); half = win // 2; out = []
    for k in range(n):
        seg = labels[max(0, k - half): k + half + 1]; c = Counter(seg); top = c.most_common()
        best = [l for l, v in top if v == top[0][1]]
        out.append(labels[k] if labels[k] in best else best[0])  # ties go to the current frame's label
    return out


def hysteresis(voted: list[str], start_n: int, end_n: int) -> tuple[list[str], list[tuple[int, str, str]]]:
    """State machine over the voted labels. Returns (held per frame, events [(frame, type, object)]). The held label is
    back-filled to the first frame of the run that triggered the transition."""
    n = len(voted); held = [""] * n; events: list[tuple[int, str, str]] = []; cur = ""; run_label, run_start = None, 0
    for k in range(n):
        if voted[k] != run_label:
            run_label, run_start = voted[k], k
        run = k - run_start + 1
        if cur and run_label != cur and run >= end_n:   # release the current object
            events.append((run_start, "release", cur)); cur = ""
            for i in range(run_start, k + 1):
                held[i] = ""
        if not cur and run_label and run >= start_n:    # grasp a new object
            cur = run_label; events.append((run_start, "grasp", cur))
            for i in range(run_start, k + 1):
                held[i] = cur
        held[k] = cur
    return held, events


# ----------------------------------------------------------------------------- depth (ZED) support

def _zed_depth(ep: Episode, s: Stream):
    """Returns (callable k -> depth [H,W] metres or None, (sx, sy) px scale from extracted frame to depth image) or None."""
    import cv2
    d = ep.dir / "zed" / s.name / "depth"
    if not d.exists() or not any(d.glob("*.png")) or not (ep.dir / "zed" / s.name / "export.json").exists():
        return None
    fps = json.load(open(ep.dir / "zed" / s.name / "export.json"))["fps"]; first = sorted(d.glob("*.png"))[0]
    im = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    frames = sorted((ep.derived / "frames" / s.name).glob("*.jpg")); f0 = cv2.imread(str(frames[0])) if frames else None
    scale = (im.shape[1] / f0.shape[1], im.shape[0] / f0.shape[0]) if f0 is not None else (1.0, 1.0)

    def read(k: int):
        kz = int(round((ep.common_start_s + k / ep.proc_fps - s.offset_s) * fps)); p = d / f"{kz:06d}.png"
        if not p.exists():
            return None
        z = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        return None if z is None else z.astype(np.float32) / 1000.0
    return read, scale


# ----------------------------------------------------------------------------- stage

def contact_stream(lm2d: np.ndarray, boxes: np.ndarray, names: np.ndarray, fps: float, depth_reader=None, depth_scale=(1.0, 1.0)) -> dict:
    """lm2d [N,2,21,2], boxes [N,M,5], names [N,M]. Returns dict(held [N,2] str, score [N,2], raw_held, raw_score, events)."""
    n = lm2d.shape[0]; raw_held = np.full((n, 2), "", dtype="<U24"); raw_score = np.zeros((n, 2), np.float32); depth_frames = 0
    for k in range(n):
        depth = depth_reader(k) if depth_reader is not None else None; depth_frames += depth is not None
        for h in range(2):
            name, sc, _ = frame_contact(lm2d[k, h], boxes[k], names[k], depth, depth_scale); raw_held[k, h] = name; raw_score[k, h] = sc
    win = max(1, int(round(VOTE_S * fps)) | 1); start_n = max(1, int(round(START_S * fps))); end_n = max(1, int(round(END_S * fps)))
    held = np.full((n, 2), "", dtype="<U24"); score = np.zeros((n, 2), np.float32); events = []
    for h in range(2):
        voted = majority_vote([str(x) for x in raw_held[:, h]], win); hh, ev = hysteresis(voted, start_n, end_n)
        held[:, h] = hh; events += [(k, typ, obj, HANDS[h]) for k, typ, obj in ev]
        half = win // 2
        for k in range(n):
            if hh[k]:
                seg = slice(max(0, k - half), k + half + 1); m = raw_held[seg, h] == hh[k]
                score[k, h] = float(raw_score[seg, h][m].mean()) if m.any() else 0.0
    return {"held": held, "score": score, "raw_held": raw_held, "raw_score": raw_score, "events": sorted(events), "depth_frames": depth_frames}


def contact(ep: Episode) -> None:
    ep.set_status("contact", "running")
    out = ep.derived / "contact"; out.mkdir(exist_ok=True)
    events: list[dict] = []; cols: dict = {}; report: dict = {"streams": {}, "params": {"min_tips": MIN_TIPS, "box_grow": BOX_GROW, "vote_s": VOTE_S, "start_s": START_S, "end_s": END_S, "depth_tol_m": DEPTH_TOL_M}}
    n_ref = None
    for s in ep.egos():
        hz, oz = ep.derived / "hands" / f"{s.name}.npz", ep.derived / "objects" / f"{s.name}.npz"
        if not (hz.exists() and oz.exists()):
            continue
        h, o = np.load(hz), np.load(oz); n = min(len(h["lm2d"]), len(o["boxes"])); n_ref = n if n_ref is None else min(n_ref, n)
        zed = _zed_depth(ep, s); reader, scale = zed if zed else (None, (1.0, 1.0))
        r = contact_stream(h["lm2d"][:n], o["boxes"][:n], o["names"][:n], ep.proc_fps, reader, scale)
        depth_used = bool(zed) and r["depth_frames"] > 0
        np.savez_compressed(out / f"{s.name}.npz", held=r["held"], score=r["score"], raw_held=r["raw_held"], raw_score=r["raw_score"], depth_used=depth_used)
        for k, typ, obj, hand in r["events"]:
            events.append({"t": round(float(ep.common_start_s + k / ep.proc_fps), 3), "frame": int(k), "stream": s.name, "person": s.person, "hand": hand, "object": obj, "type": typ})
        for i, hand in enumerate(HANDS):
            cols[f"{s.name}_held_{hand}"] = list(r["held"][:, i]); cols[f"{s.name}_contact_score_{hand}"] = r["score"][:, i]
        any_contact = (r["held"] != "").any(1)
        report["streams"][s.name] = {"person": s.person, "frames": int(n), "contact_fraction_any_hand": round(float(any_contact.mean()), 4),
                                     "contact_fraction_L": round(float((r["held"][:, 0] != "").mean()), 4), "contact_fraction_R": round(float((r["held"][:, 1] != "").mean()), 4),
                                     "raw_contact_fraction_any_hand": round(float((r["raw_held"] != "").any(1).mean()), 4),
                                     "hand_present_L": round(float(np.isfinite(h["lm2d"][:n, 0, 0, 0]).mean()), 4), "hand_present_R": round(float(np.isfinite(h["lm2d"][:n, 1, 0, 0]).mean()), 4),
                                     "n_events": sum(1 for e in events if e["stream"] == s.name), "objects_held": dict(Counter(x for x in r["held"].ravel() if x)),
                                     "depth_check": "zed depth used on %d frames" % r["depth_frames"] if depth_used else "not used (no episode/zed/<stream>/depth)"}
    if not report["streams"]:
        ep.set_status("contact", "skipped", "no ego stream with both hands and objects"); return
    events.sort(key=lambda e: (e["t"], e["stream"], e["hand"]))
    json.dump(events, open(out / "events.json", "w"), indent=1)
    n_rows = max(len(v) for v in cols.values())
    pd.DataFrame({k: (list(v) + [np.nan] * (n_rows - len(v))) if len(v) < n_rows else v for k, v in cols.items()}).to_parquet(out / "records_extra.parquet", index=False)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    ep.set_status("contact", "done", "; ".join(f"{k}: contact {v['contact_fraction_any_hand']:.0%} of frames, {v['n_events']} events, {v['depth_check']}" for k, v in report["streams"].items()))
