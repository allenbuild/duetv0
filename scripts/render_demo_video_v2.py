#!/usr/bin/env python3
"""Demo clip v2: both ego views with live 3D hand skeletons, gaze, detected objects and the
partner's body skeleton drawn on the frames; a 3D panel with both people's hands; and a
scrolling signal strip (wrist speeds + model meters). Everything drawn comes from data we
actually have for this recording:

  hands     Aria hand tracking, all 21 landmarks per hand, projected through the official
            online camera calibration (native->MP4 rotation, 1408->640 scale)
  gaze      Aria eye gaze, converted CPF->device with the measured T_Device_CPF, projected
  objects   cached YOLO-World detections (3 Hz), the box a hand holds is highlighted
  body      YOLOv8-pose 2D keypoints on the frame (whoever is visible: mostly the partner)
  meters    held-out predictions of the paired tree model (percentile within recording)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from build_comind_objects import CATS, NAME2CAT, load_rgb_calibs, project_points  # noqa: E402
from duet.adapters.comind.kinematics import HAND_DIM, frame_device_times_ns, parse_mp4_tail, person_slice  # noqa: E402
from duet.adapters.comind.shared_world_features import cpf_to_device  # noqa: E402

FPS, OUT_FPS, IMG = 30, 15, 640
# Aria landmark ids: 0-4 fingertips (thumb..pinky), 5 wrist, 6-7 thumb, 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky, 20 palm
BONES = [(5, 6), (6, 7), (7, 0), (5, 8), (8, 9), (9, 10), (10, 1), (5, 11), (11, 12), (12, 13), (13, 2),
         (5, 14), (14, 15), (15, 16), (16, 3), (5, 17), (17, 18), (18, 19), (19, 4), (5, 20)]
COCO_LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
C = {"bg": "#0B0F0E", "panel": "#121817", "ink": "#E6ECE9", "mute": "#7C8783", "grid": "#1F2725",
     "left": "#57E39B", "right": "#FF6B6B", "gaze": "#FFD447", "box": "#3EA7FF", "held": "#FFD447", "body": "#B58CFF",
     "leader": "#8FA3CB", "helper": "#E39468", "onset": "#FF8A3D", "active": "#3FA58C", "ja": "#B58CFF"}


def load_hands_full(csv: Path, query_ts_ns: np.ndarray):
    df = pd.read_csv(csv)
    ts = df["tracking_timestamp_us"].to_numpy(np.int64); order = np.argsort(ts); df = df.iloc[order]; ts = ts[order]
    idx = np.clip(np.searchsorted(ts, query_ts_ns // 1000), 1, len(ts) - 1)
    pick = np.where(np.abs(ts[idx - 1] - query_ts_ns // 1000) <= np.abs(ts[idx] - query_ts_ns // 1000), idx - 1, idx)
    ok = np.abs(ts[pick] - query_ts_ns // 1000) <= 20_000
    out = {}
    for side in ("left", "right"):
        conf = df[f"{side}_tracking_confidence"].to_numpy(np.float32)[pick]
        pts = np.stack([df[[f"tx_{side}_landmark_{k}_device", f"ty_{side}_landmark_{k}_device", f"tz_{side}_landmark_{k}_device"]].to_numpy(np.float64)[pick] for k in range(21)], axis=1)
        out[side] = (pts, ok & (conf > 0.5) & np.isfinite(pts).all(axis=(1, 2)))
    return out


def extract(mp4: Path, start: float, dur: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(mp4),
                    "-vf", f"fps={OUT_FPS},scale={IMG}:{IMG}", "-q:v", "3", str(out_dir / "%05d.jpg")], check=True)
    return sorted(out_dir.glob("*.jpg"))


def draw_view(ax, img, hands_px, gaze_px, boxes, held_idx, body_kpts, title):
    ax.imshow(img); ax.set_xlim(0, IMG); ax.set_ylim(IMG, 0); ax.set_axis_off()
    for j, (x0, y0, x1, y1, name) in enumerate(boxes):
        col = C["held"] if j in held_idx else C["box"]; lw = 1.8 if j in held_idx else 0.8
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=col, lw=lw, alpha=0.95 if j in held_idx else 0.6))
        ax.text(x0 + 2, y0 - 3, name, color=col, fontsize=6.5, family="monospace", alpha=0.95 if j in held_idx else 0.7)
    if body_kpts is not None:
        for kp, kc in body_kpts:
            for a, b in COCO_LINKS:
                if kc[a] > 0.4 and kc[b] > 0.4:
                    ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]], color=C["body"], lw=1.4, alpha=0.85)
            m = kc > 0.4; ax.plot(kp[m, 0], kp[m, 1], "o", ms=2.5, color=C["body"], alpha=0.9)
    for side, px in hands_px.items():
        if px is None: continue
        col = C["left"] if side == "left" else C["right"]
        for a, b in BONES:
            if np.isfinite(px[[a, b]]).all():
                ax.plot(px[[a, b], 0], px[[a, b], 1], color=col, lw=1.6, alpha=0.95, solid_capstyle="round")
        m = np.isfinite(px).all(1); ax.plot(px[m, 0], px[m, 1], "o", ms=2.2, color=col)
    if gaze_px is not None and np.isfinite(gaze_px).all():
        ax.plot(gaze_px[0], gaze_px[1], "o", ms=14, mfc="none", mec=C["gaze"], mew=1.6)
        ax.plot(gaze_px[0], gaze_px[1], "+", ms=10, color=C["gaze"], mew=1.2)
    ax.set_title(title, color=C["ink"], fontsize=10, loc="left", family="monospace", pad=4)


def draw_hands3d(ax, hands_dev, title):
    ax.set_facecolor(C["panel"]); ax.set_title(title, color=C["mute"], fontsize=8, family="monospace", loc="left", pad=2)
    for side, (pts, ok) in hands_dev.items():
        if not ok: continue
        col = C["left"] if side == "left" else C["right"]
        for a, b in BONES:
            ax.plot(pts[[a, b], 0], pts[[a, b], 2], pts[[a, b], 1] * -1, color=col, lw=2.2)
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=9, c=col)
    ax.scatter([0], [0], [0], s=30, c=C["ink"], marker="^")  # the headset
    ax.set_xlim(-0.3, 0.3); ax.set_ylim(0.05, 0.65); ax.set_zlim(-0.5, 0.1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=18, azim=-60)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.set_facecolor(C["panel"]); pane.set_edgecolor(C["grid"])
    ax.grid(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="43276420-701f-4731-b9ab-bebc7fd14994")
    ap.add_argument("--start-s", type=float, default=596.0)
    ap.add_argument("--dur-s", type=float, default=30.0)
    ap.add_argument("--preds", default="outputs/paired_benchmark/gbdt_preds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rid = a.recording; rid8 = rid[:8]
    rec = ROOT / "data/raw/comind/recordings" / rid; proc = ROOT / "data/processed/comind" / rid
    meta = json.load(open(proc / "kinematics_v0.json")); kin = np.load(proc / "kinematics_v0.npz")["features"]; n = len(kin)
    objs = np.load(proc / "objects_v0.npz")["objects"]
    pinfo = json.load(open(ROOT / a.preds / "results.json"))["info"]["tasks"]
    probs = np.load(ROOT / a.preds / f"preds_both_{rid8}.npz")["probs"].astype(np.float32)
    k = 15; sm = np.stack([np.convolve(probs[:, i], np.ones(k) / k, mode="same") for i in range(probs.shape[1])], 1)
    rank = np.stack([np.argsort(np.argsort(sm[:, i])) / (n - 1) for i in range(sm.shape[1])], 1)
    ti = {t: pinfo.index(t) for t in ("onset_within_5s", "handover_active", "ja_active") if t in pinfo}

    from ultralytics import YOLO
    pose = YOLO("yolov8n-pose.pt")
    tmp = Path(tempfile.mkdtemp(prefix="duet_demo2_"))
    per = {}
    fi0 = int(round(a.start_s * FPS)); nfr = int(a.dur_s * OUT_FPS)
    frame_idx = [min(n - 1, fi0 + int(round(i * FPS / OUT_FPS))) for i in range(nfr)]
    for role in ("leader", "helper"):
        frames = extract(rec / "mp4s" / f"{role}_trimmed_sync.mp4", a.start_s, a.dur_s, tmp / role)
        anchor = parse_mp4_tail((proc / f"mp4_tail_{role}.bin").read_bytes()); ts = frame_device_times_ns(anchor)
        cts, rgbs = load_rgb_calibs(rec / f"mps_{role}_trimmed_vrs/slam/online_calibration.jsonl")
        hands = load_hands_full(rec / f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv", ts[frame_idx])
        dz = np.load(proc / f"detections_{role}_v0.npz", allow_pickle=True); dets, names = list(dz["dets"]), list(dz["names"])
        s = person_slice(role)
        pose_res = pose.predict([str(p) for p in frames[:nfr]], device="mps", verbose=False, conf=0.3)
        per[role] = []
        for i, fi in enumerate(frame_idx[: len(frames)]):
            ci = int(np.argmin(np.abs(cts - ts[fi] / 1000))); rgb = rgbs[ci]
            hp, hd = {}, {}
            for side in ("left", "right"):
                pts, ok = hands[side]; hd[side] = (pts[i], bool(ok[i]))
                hp[side] = project_points(rgb, pts[i]) if ok[i] else None
            gv = kin[fi, s.start + 2 * HAND_DIM + 6] > 0
            gp = project_points(rgb, cpf_to_device(kin[fi, s.start + 2 * HAND_DIM: s.start + 2 * HAND_DIM + 3][None].astype(np.float64)))[0] if gv else None
            kd = min(len(dets) - 1, int(round(fi / FPS * 3))); bx, cf, cl = dets[kd]
            keep = (cf > 0.4) & np.array([names[c] != "hand" for c in cl], bool)
            top = np.flatnonzero(keep); top = top[np.argsort(-cf[top])][:8]; keep = np.zeros_like(keep); keep[top] = True
            boxes = [(*bx[j], names[cl[j]]) for j in np.flatnonzero(keep)]
            held = set()
            for j, jj in enumerate(np.flatnonzero(keep)):
                for side in ("left", "right"):
                    if hp[side] is None: continue
                    px = hp[side]; inside = ((px[:, 0] >= bx[jj, 0]) & (px[:, 0] <= bx[jj, 2]) & (px[:, 1] >= bx[jj, 1]) & (px[:, 1] <= bx[jj, 3])).sum()
                    if inside >= 3 and NAME2CAT.get(names[cl[jj]], -1) not in (-1, CATS.index("hand")): held.add(j)
            r = pose_res[i]; body = None
            if r.keypoints is not None and r.keypoints.conf is not None and len(r.keypoints.xy):
                body = []
                for q in range(len(r.keypoints.xy)):
                    kp, kc = r.keypoints.xy[q].cpu().numpy(), r.keypoints.conf[q].cpu().numpy()
                    good = kc > 0.5
                    if good.sum() >= 6 and (np.ptp(kp[good, 0]) > 40 or np.ptp(kp[good, 1]) > 40) and not (good[[5, 6]].sum() == 0):
                        body.append((kp, np.where(good, kc, 0.0)))
            per[role].append(dict(img=plt.imread(frames[i]), hp=hp, hd=hd, gp=gp, boxes=boxes, held=held, body=body))
    # signals
    def wspeed(role):
        s = person_slice(role); out = np.zeros(n)
        for h in range(2):
            b = s.start + h * HAND_DIM; w = kin[:, b + 15: b + 18]; v = kin[:, b + 25] > 0
            sp = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=1) * FPS; sp[~v] = 0; out = np.maximum(out, sp)
        return np.convolve(out, np.ones(5) / 5, mode="same")
    ws = {r: wspeed(r) for r in ("leader", "helper")}; t_axis = np.arange(n) / FPS
    png = tmp / "comp"; png.mkdir()
    nfr = min(len(per["leader"]), len(per["helper"]))
    for i in range(nfr):
        fi = frame_idx[i]; t = fi / FPS
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100); fig.patch.set_facecolor(C["bg"])
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.9], height_ratios=[1.35, 1.35, 1.0], left=0.015, right=0.985, top=0.93, bottom=0.06, hspace=0.18, wspace=0.04)
        for j, role in enumerate(("leader", "helper")):
            d = per[role][i]; ax = fig.add_subplot(gs[0:2, j])
            draw_view(ax, d["img"], d["hp"], d["gp"], d["boxes"], d["held"], d["body"], f"{role.upper()}  ego · hands 21j · gaze · objects · body")
        for j, role in enumerate(("leader", "helper")):
            ax3 = fig.add_subplot(gs[j, 2], projection="3d"); draw_hands3d(ax3, per[role][i]["hd"], f"{role} hands, 3D (device frame, m)")
        ax = fig.add_subplot(gs[2, :]); ax.set_facecolor(C["panel"])
        lo, hi = t - 12, t + 4
        for h in meta["handovers"]:
            s0, e0 = h["start_frame"] / FPS, h["end_frame"] / FPS
            if e0 > lo and s0 < hi: ax.axvspan(s0, e0, color=C["gaze"], alpha=0.12, lw=0); ax.axvline(s0, color=C["gaze"], ls=":", lw=1)
        m = (t_axis >= lo) & (t_axis <= t)
        ax.plot(t_axis[m], np.clip(ws["leader"][m] / 1.5, 0, 1), color=C["leader"], lw=0.9, label="leader wrist speed")
        ax.plot(t_axis[m], np.clip(ws["helper"][m] / 1.5, 0, 1), color=C["helper"], lw=0.9, label="helper wrist speed")
        for name, col, lab in (("onset_within_5s", C["onset"], "handover coming (<5 s)"), ("handover_active", C["active"], "handover in progress"), ("ja_active", C["ja"], "joint attention")):
            if name in ti: ax.plot(t_axis[m], rank[m, ti[name]], color=col, lw=1.6, label=lab)
        ax.axvline(t, color=C["ink"], lw=0.8); ax.set_xlim(lo, hi); ax.set_ylim(0, 1.02)
        ax.tick_params(colors=C["mute"], labelsize=7); ax.set_yticks([0, 0.5, 1])
        for sp in ax.spines.values(): sp.set_color(C["grid"])
        ax.grid(True, color=C["grid"], lw=0.5); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7, frameon=False, labelcolor=C["ink"])
        ax.set_xlabel("time in recording (s) · gold = annotated handover · model meters are percentiles within this recording, causal", color=C["mute"], fontsize=8)
        vals = "   ".join(f"{lab} {rank[fi, ti[name]]:.2f}" for name, lab in (("onset_within_5s", "HANDOVER<5s"), ("handover_active", "HANDOVER"), ("ja_active", "JOINT ATTN")) if name in ti)
        fig.text(0.015, 0.965, f"DUET V0  ·  CoMind {rid8}  ·  held-out pair  ·  t = {t:7.2f} s", color=C["ink"], fontsize=11, family="monospace")
        fig.text(0.985, 0.965, vals, color=C["gaze"], fontsize=10, family="monospace", ha="right")
        fig.savefig(png / f"{i:05d}.png", facecolor=fig.get_facecolor()); plt.close(fig)
        if i % 50 == 0: print(f"  frame {i}/{nfr}", flush=True)
    out = Path(a.out) if a.out else ROOT / "outputs/paired_benchmark/demo" / f"{rid8}_{int(a.start_s)}s_v2.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(OUT_FPS), "-i", str(png / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)
    shutil.rmtree(tmp, ignore_errors=True); print(out)


if __name__ == "__main__":
    main()
