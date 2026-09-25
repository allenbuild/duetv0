"""Quality scores, per stream and per episode, in the style Eidon used to gate their 1,274 h release
(metadata.parquet: good_frame_percent, hand_presence_ratio, stability_score, lighting_score,
average_brightness, average_optical_flow). Ours are computed on the extracted frames so they are
cheap and reproducible; thresholds are ours and are written into the output so they can be tuned.

Per frame:  sharpness (variance of Laplacian), brightness (mean grey), motion (median optical-flow
            magnitude vs previous frame, px/frame at proc_size), hand_present (ego: MediaPipe found a
            hand; exo: a person box exists), n_people (exo).
Per stream: good_frame_percent (sharp & not too dark/bright), hand_presence_ratio, stability_score
            (1 - clipped normalised motion), lighting_score, plus alignment confidence.
Episode:    both_visible_ratio (exo frames with >= 2 people), hands_both_views_ratio (all ego streams
            show a hand), and the minimum stream score.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from .episode import Episode
from .perception import _frame_list

THR = {"sharp_min": 40.0, "bright_min": 35.0, "bright_max": 225.0, "motion_norm_px": 12.0}


def _per_frame(frames):
    n = len(frames); sharp = np.zeros(n); bright = np.zeros(n); motion = np.zeros(n); prev = None
    for k, f in enumerate(frames):
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        sharp[k] = cv2.Laplacian(g, cv2.CV_64F).var(); bright[k] = g.mean()
        small = cv2.resize(g, (160, 160 * g.shape[0] // g.shape[1]))
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, small, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            motion[k] = np.median(np.linalg.norm(flow, axis=-1)) * (g.shape[1] / 160)
        prev = small
    return sharp, bright, motion


def qc(ep: Episode) -> None:
    ep.set_status("qc", "running")
    out_dir = ep.derived / "qc"; out_dir.mkdir(exist_ok=True)
    report = {"thresholds": THR, "streams": {}}
    n_people_exo, hands_ego = [], []
    for s in ep.streams:
        frames = _frame_list(ep, s); sharp, bright, motion = _per_frame(frames); n = len(frames)
        good = (sharp > THR["sharp_min"]) & (bright > THR["bright_min"]) & (bright < THR["bright_max"])
        present = np.zeros(n, bool); n_people = np.zeros(n, int)
        if s.role == "ego":
            z = ep.derived / "hands" / f"{s.name}.npz"
            if z.exists():
                present = np.isfinite(np.load(z)["lm2d"][:, :, 0, 0]).any(1)[:n]; hands_ego.append(present)
        else:
            z = ep.derived / "body2d" / f"{s.name}.npz"
            if z.exists():
                n_people = np.isfinite(np.load(z)["boxes"][:, :, 4]).sum(1)[:n]; present = n_people > 0; n_people_exo.append(n_people)
        stab = float(np.clip(1 - np.median(motion[1:]) / THR["motion_norm_px"], 0, 1)) if n > 1 else 0.0
        light = float(np.clip(1 - np.abs(bright.mean() - 120) / 120, 0, 1))
        report["streams"][s.name] = {
            "role": s.role, "n_frames": int(n), "good_frame_percent": float(good.mean()), "hand_presence_ratio": float(present.mean()) if s.role == "ego" else None,
            "person_presence_ratio": float(present.mean()) if s.role == "exo" else None, "stability_score": stab, "lighting_score": light,
            "average_brightness": float(bright.mean()), "average_sharpness": float(sharp.mean()), "average_optical_flow_px": float(motion[1:].mean()) if n > 1 else 0.0,
            "alignment_confidence": s.offset_confidence, "offset_s": s.offset_s}
        np.savez_compressed(out_dir / f"{s.name}.npz", sharp=sharp, bright=bright, motion=motion, present=present, n_people=n_people)
    L = min([len(x) for x in n_people_exo + hands_ego] or [0])
    report["episode"] = {
        "both_visible_ratio": float((np.stack([x[:L] for x in n_people_exo]).max(0) >= 2).mean()) if n_people_exo and L else None,
        "hands_all_egos_ratio": float(np.stack([x[:L] for x in hands_ego]).all(0).mean()) if hands_ego and L else None,
        "min_good_frame_percent": min(v["good_frame_percent"] for v in report["streams"].values()),
        "min_alignment_confidence": min([v["alignment_confidence"] for v in report["streams"].values() if v["alignment_confidence"] is not None] or [None]),
        "duration_s": ep.common_end_s - ep.common_start_s}
    json.dump(report, open(out_dir / "report.json", "w"), indent=1)
    ep.set_status("qc", "done", f"min good-frame {report['episode']['min_good_frame_percent']:.2f}")
