#!/usr/bin/env python3
"""Object-level features from ego video: what each hand holds and what each person looks at.

Per recording and role (needs <role>_trimmed_sync.mp4 and mps_<role>_trimmed_vrs/slam/online_calibration.jsonl):
  1. extract frames at DET_FPS (3 Hz), 640x640, with hardware decode;
  2. YOLO-World (open vocabulary) with a CoMind kitchen vocabulary -> boxes;
  3. project the wearer's Aria 3D hand landmarks and gaze point into the MP4 image through the official
     online camera calibration (Fisheye624, native->MP4 clockwise rotation verified in v0);
  4. associate: a hand "holds" the highest-confidence box that contains >= 3 of its 7 keypoints;
     gaze "is on" the smallest box containing the gaze pixel; gaze on a `hand` box that does not contain
     the wearer's own landmarks = looking at the partner's hand;
  5. write objects_v0.npz [N, 2 * OBJ_DIM] at 30 Hz (sample-and-hold from 3 Hz), leader block first.

OBJ_DIM = 2 hands * (8 categories + holding flag) + gaze (8 categories + on-object + on-partner-hand)
          + n objects visible + n objects near hands + has_video = 31
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, KEY_LANDMARKS, frame_device_times_ns, parse_mp4_tail, person_slice  # noqa: E402

FPS, DET_FPS, IMG, NATIVE = 30, 3, 640, 1408
VOCAB = {
    "cookware": ["pan", "pot", "lid", "baking tray"],
    "container": ["bowl", "plate", "cup", "glass", "jar", "container", "bag", "package", "carton"],
    "utensil": ["knife", "spoon", "spatula", "ladle", "whisk", "strainer", "cutting board", "tongs", "peeler", "grater", "rolling pin", "scissors"],
    "ingredient": ["onion", "garlic", "tomato", "mushroom", "carrot", "pepper", "vegetable", "meat", "chicken", "egg", "dough", "flour", "rice", "pasta", "cheese", "bread", "lemon", "herbs"],
    "seasoning": ["salt shaker", "spice jar", "oil bottle", "sauce bottle", "bottle"],
    "fixture": ["sink", "oven", "microwave", "fridge", "stove knob", "faucet", "cupboard"],
    "towel": ["towel", "sponge"],
    "hand": ["hand"],
}
CATS = list(VOCAB)
NAMES = [n for c in CATS for n in VOCAB[c]]
NAME2CAT = {n: CATS.index(c) for c in CATS for n in VOCAB[c]}
OBJ_DIM = 2 * (len(CATS) + 1) + (len(CATS) + 2) + 3


def mp4_complete(mp4: Path) -> bool:
    """CoMind MP4s put the moov atom at the end; a partially downloaded file has none."""
    try:
        with open(mp4, "rb") as f:
            f.seek(max(0, mp4.stat().st_size - 12_000_000)); return b"moov" in f.read()
    except OSError:
        return False


def extract_frames(mp4: Path, out_dir: Path) -> list[Path]:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox", "-i", str(mp4),
                    "-vf", f"fps={DET_FPS},scale={IMG}:{IMG}", "-q:v", "4", str(out_dir / "%06d.jpg")], check=True)
    return sorted(out_dir.glob("*.jpg"))


def find_calibration(rec: Path, role: str) -> Path | None:
    """Standard MPS calibration, else the Multi-SLAM copy (same device, same clock)."""
    cands = [rec / f"mps_{role}_trimmed_vrs/slam/online_calibration.jsonl", rec / f"mps_{role}_gapfill_vrs/slam/online_calibration.jsonl",
             rec / f"multislam_output/{role}_trimmed/slam/online_calibration.jsonl"]
    vmap = rec / "multislam_output/vrs_to_multi_slam.json"
    if vmap.exists():
        import json
        for k, v in json.load(open(vmap)).items():
            if k.endswith(f"{role}_trimmed.vrs"):
                cands.append(rec / f"multislam_output/{v}/slam/online_calibration.jsonl")
    for c in cands:
        if c.exists() and c.stat().st_size > 50_000_000:
            return c
    # Fallback: no calibration exported for this participant. Aria RGB intrinsics/extrinsics are near-identical
    # across units, and association here is box containment at 640 px, so use the partner's calibration from the
    # same recording, else a nominal one. Timestamps are ignored in that case (nearest record is arbitrary).
    other = "helper" if role == "leader" else "leader"
    for c in [rec / f"mps_{other}_trimmed_vrs/slam/online_calibration.jsonl", rec / f"mps_{other}_gapfill_vrs/slam/online_calibration.jsonl",
              ROOT / "data/raw/comind/recordings/43276420-701f-4731-b9ab-bebc7fd14994/mps_leader_trimmed_vrs/slam/online_calibration.jsonl"]:
        if c.exists() and c.stat().st_size > 50_000_000:
            return c
    return None


def load_rgb_calibs(path: Path):
    from projectaria_tools.core import mps
    cal = mps.read_online_calibration(str(path))
    ts = np.array([c.tracking_timestamp.total_seconds() * 1e6 for c in cal])
    rgb = [[cc for cc in c.camera_calibs if cc.get_label() == "camera-rgb"][0] for c in cal]
    return ts, rgb


def project_points(rgb_calib, pts_device: np.ndarray) -> np.ndarray:
    """[K,3] device-frame points -> [K,2] MP4 pixel coords at IMG resolution (NaN if not projectable)."""
    T_cd = rgb_calib.get_transform_device_camera().inverse()
    out = np.full((len(pts_device), 2), np.nan)
    sc = IMG / NATIVE
    for i, p in enumerate(pts_device):
        uv = rgb_calib.project(T_cd @ p.astype(np.float64))
        if uv is not None:
            out[i] = ((NATIVE - 1 - uv[1]) * sc, uv[0] * sc)  # native (u,v) -> MP4 (x,y): clockwise 90 deg, then scale
    return out


def build_role(rec: Path, proc: Path, role: str, feats: np.ndarray, model, tmp: Path, log) -> np.ndarray:
    n = len(feats)
    out = np.zeros((n, OBJ_DIM), np.float32)
    mp4 = rec / "mp4s" / f"{role}_trimmed_sync.mp4"; calp = find_calibration(rec, role)
    if not (mp4.exists() and calp is not None):
        return out
    t0 = time.time()
    frames = extract_frames(mp4, tmp)
    log(f"    {role}: {len(frames)} frames extracted ({time.time() - t0:.0f}s)")
    cts, rgbs = load_rgb_calibs(calp)
    anchor = parse_mp4_tail((proc / f"mp4_tail_{role}.bin").read_bytes()); dev_ts = frame_device_times_ns(anchor)
    s = person_slice(role)
    t0 = time.time()
    for start in range(0, len(frames), 64):
        batch = frames[start: start + 64]
        res = model.predict([str(p) for p in batch], imgsz=IMG, conf=0.2, device="mps", verbose=False)
        for j, r in enumerate(res):
            k = start + j
            fi = min(n - 1, int(round(k / DET_FPS * FPS)))
            boxes = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
            names = [model.names[c] for c in cls]; cats = np.array([NAME2CAT.get(nm, -1) for nm in names])
            f = np.zeros(OBJ_DIM, np.float32); f[-1] = 1.0
            ci = int(np.argmin(np.abs(cts - dev_ts[min(fi, len(dev_ts) - 1)] / 1000)))
            rgb = rgbs[ci]
            own_pts = []
            near_hands = 0
            for h in range(2):
                b = s.start + h * HAND_DIM
                if feats[fi, b + 25] <= 0:
                    continue
                pts = feats[fi, b: b + 3 * len(KEY_LANDMARKS)].reshape(-1, 3)
                px = project_points(rgb, pts); px = px[np.isfinite(px).all(1)]
                own_pts.append(px)
                if len(px) == 0 or len(boxes) == 0:
                    continue
                inside = ((px[:, None, 0] >= boxes[None, :, 0]) & (px[:, None, 0] <= boxes[None, :, 2]) & (px[:, None, 1] >= boxes[None, :, 1]) & (px[:, None, 1] <= boxes[None, :, 3])).sum(0)
                cand = np.flatnonzero((inside >= 3) & (cats >= 0) & (cats != CATS.index("hand")))
                near_hands += int((inside >= 1).sum())
                if len(cand):
                    best = cand[np.argmax(conf[cand])]
                    f[h * (len(CATS) + 1) + cats[best]] = 1.0; f[h * (len(CATS) + 1) + len(CATS)] = 1.0
            g0 = 2 * (len(CATS) + 1)
            if feats[fi, s.start + 2 * HAND_DIM + 6] > 0 and len(boxes):
                gz = project_points(rgb, feats[fi, s.start + 2 * HAND_DIM: s.start + 2 * HAND_DIM + 3][None])[0]
                if np.isfinite(gz).all():
                    inb = (gz[0] >= boxes[:, 0]) & (gz[0] <= boxes[:, 2]) & (gz[1] >= boxes[:, 1]) & (gz[1] <= boxes[:, 3]) & (cats >= 0)
                    if inb.any():
                        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]); area[~inb] = np.inf
                        best = int(np.argmin(area)); f[g0 + cats[best]] = 1.0; f[g0 + len(CATS)] = 1.0
                        if cats[best] == CATS.index("hand"):
                            own_in = any(((p[:, 0] >= boxes[best, 0]) & (p[:, 0] <= boxes[best, 2]) & (p[:, 1] >= boxes[best, 1]) & (p[:, 1] <= boxes[best, 3])).any() for p in own_pts if len(p))
                            f[g0 + len(CATS) + 1] = 0.0 if own_in else 1.0
            f[-3] = min(len(boxes), 20) / 20.0; f[-2] = min(near_hands, 10) / 10.0
            lo = fi; hi = min(n, int(round((k + 1) / DET_FPS * FPS)))
            out[lo:hi] = f
    log(f"    {role}: detection+association done ({time.time() - t0:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default=None, help="one recording id (default: all with kinematics)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--roles", default="leader,helper")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO("yolov8s-worldv2.pt"); model.set_classes(NAMES)
    proc_root = ROOT / "data/processed/comind"; raw_root = ROOT / "data/raw/comind/recordings"
    ids = [a.recording] if a.recording else sorted(p.parent.name for p in proc_root.glob("*/kinematics_v0.npz"))
    log = lambda s: print(s, flush=True)
    done = 0
    for rid in ids:
        proc = proc_root / rid; rec = raw_root / rid; out = proc / "objects_v0.npz"
        feats = np.load(proc / "kinematics_v0.npz")["features"]
        prev = np.load(out)["objects"] if out.exists() else np.zeros((len(feats), 2 * OBJ_DIM), np.float32)
        changed = False
        for role in a.roles.split(","):
            i = ("leader", "helper").index(role)
            if prev[:, (i + 1) * OBJ_DIM - 1].max() > 0 and not a.force:
                continue  # already built for this role
            mp4 = rec / "mp4s" / f"{role}_trimmed_sync.mp4"; calp = find_calibration(rec, role)
            if not (mp4.exists() and calp is not None and mp4_complete(mp4)):
                continue
            log(f"{rid[:8]} {role}")
            tmp = Path(tempfile.mkdtemp(prefix="duet_frames_"))
            try:
                prev[:, i * OBJ_DIM: (i + 1) * OBJ_DIM] = build_role(rec, proc, role, feats, model, tmp, log); changed = True
            except Exception as e:  # noqa: BLE001 - keep the batch going; the watcher retries next pass
                log(f"  {rid[:8]} {role} FAILED: {str(e)[:120]}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if changed:
            np.savez_compressed(out, objects=prev); done += 1
            L, H = prev[:, :OBJ_DIM], prev[:, OBJ_DIM:]
            log(f"  saved; leader holding L/R {L[:, 8].mean():.2f}/{L[:, 17].mean():.2f} gaze-on-object {L[:, 26].mean():.2f} | helper holding {H[:, 8].mean():.2f}/{H[:, 17].mean():.2f} gaze-on-object {H[:, 26].mean():.2f}")
    log(f"built/updated {done} recordings")


if __name__ == "__main__":
    main()
