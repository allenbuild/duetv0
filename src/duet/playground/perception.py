"""Per-frame perception on the common timeline.

frames    ffmpeg extracts every stream at ``proc_fps`` on the COMMON reference timeline, so
          frame k of every stream is the same instant t_k = common_start + k / proc_fps.
body2d    YOLOv8-pose (COCO-17) on every view; people tracked by IoU so ids are stable-ish.
hands     MediaPipe HandLandmarker on ego views: 21 landmarks, image-normalised + metric
          "world" coordinates (wrist-relative, metres), handedness.
objects   YOLO-World with the CoMind kitchen vocabulary on ego views.
body3d    MediaPipe PoseLandmarker world landmarks (33 joints, metres, hip-centred) on the exo
          view that sees the largest person. Monocular: a plausible 3D skeleton, not a
          triangulated one; multi-view triangulation needs calibrated cameras (next step).

Everything is written under derived/<stage>/<stream>.npz with a frames axis matching the
extracted frames, so the viewer and the exporter can index by frame.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .episode import Episode, Stream

MODELS = Path(__file__).resolve().parents[3] / "data/playground/models"
VOCAB = ["pan", "pot", "lid", "bowl", "plate", "cup", "glass", "jar", "container", "bag", "package", "box", "knife", "spoon", "spatula", "ladle",
         "whisk", "strainer", "cutting board", "tongs", "scissors", "onion", "garlic", "tomato", "mushroom", "carrot", "pepper", "vegetable", "meat",
         "chicken", "egg", "dough", "flour", "rice", "pasta", "cheese", "bread", "salt shaker", "spice jar", "oil bottle", "sauce bottle", "bottle",
         "towel", "sponge", "sink", "oven", "microwave", "fridge", "stove knob", "cupboard", "laundry", "shirt", "towel", "basket", "phone", "tool"]


def frames_dir(ep: Episode, s: Stream) -> Path:
    return ep.derived / "frames" / s.name


def extract_frames(ep: Episode) -> None:
    ep.set_status("frames", "running")
    dur = ep.common_end_s - ep.common_start_s
    n_expected = int(dur * ep.proc_fps)
    for s in ep.streams:
        out = frames_dir(ep, s); out.mkdir(parents=True, exist_ok=True)
        if len(list(out.glob("*.jpg"))) >= n_expected - 1:
            continue
        ss = ep.common_start_s - s.offset_s  # stream time at the common start
        scale = f"scale={ep.proc_size}:-2" if s.width >= s.height else f"scale=-2:{ep.proc_size}"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{max(0.0, ss):.3f}", "-t", f"{dur:.3f}", "-i", str(ep.dir / s.path),
                        "-vf", f"fps={ep.proc_fps},{scale}", "-q:v", "3", str(out / "%06d.jpg")], check=True)
    ep.set_status("frames", "done", f"{n_expected} frames x {len(ep.streams)} streams at {ep.proc_fps} fps")


def _frame_list(ep: Episode, s: Stream) -> list[Path]:
    return sorted(frames_dir(ep, s).glob("*.jpg"))


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def body2d(ep: Episode, device: str = "mps") -> None:
    from ultralytics import YOLO
    ep.set_status("body2d", "running")
    model = YOLO("yolov8n-pose.pt"); out_dir = ep.derived / "body2d"; out_dir.mkdir(exist_ok=True)
    for s in ep.streams:
        frames = _frame_list(ep, s); n = len(frames)
        kpts = np.full((n, 4, 17, 3), np.nan, np.float32); boxes = np.full((n, 4, 5), np.nan, np.float32)  # up to 4 people, tracked slots
        tracks: list[list[float] | None] = [None] * 4
        for start in range(0, n, 64):
            res = model.predict([str(p) for p in frames[start: start + 64]], device=device, verbose=False, conf=0.35, imgsz=ep.proc_size)
            for j, r in enumerate(res):
                k = start + j
                if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                    continue
                bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
                kp = np.concatenate([r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()[..., None]], -1)
                order = np.argsort(-cf)[:4]; used = set()
                for i in order:
                    best, best_iou = None, 0.3
                    for t, tb in enumerate(tracks):
                        if t in used or tb is None: continue
                        v = _iou(tb, bx[i])
                        if v > best_iou: best, best_iou = t, v
                    if best is None:
                        free = [t for t in range(4) if t not in used and (tracks[t] is None)]
                        if not free: continue
                        best = free[0]
                    used.add(best); tracks[best] = list(bx[i]); kpts[k, best] = kp[i]; boxes[k, best, :4] = bx[i]; boxes[k, best, 4] = cf[i]
                for t in range(4):
                    if t not in used and tracks[t] is not None:
                        tracks[t] = None if k % 15 == 0 else tracks[t]  # drop stale tracks every 1.5 s
        np.savez_compressed(out_dir / f"{s.name}.npz", kpts=kpts, boxes=boxes, img_w=_img_size(frames)[0], img_h=_img_size(frames)[1])
    ep.set_status("body2d", "done", "YOLOv8n-pose, up to 4 tracked people per view")


def _img_size(frames):
    import cv2
    im = cv2.imread(str(frames[0])); return im.shape[1], im.shape[0]


def hands(ep: Episode) -> None:
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    ep.set_status("hands", "running")
    out_dir = ep.derived / "hands"; out_dir.mkdir(exist_ok=True)
    for s in ep.egos():
        hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")), num_hands=2,
            running_mode=vision.RunningMode.VIDEO, min_hand_detection_confidence=0.4, min_tracking_confidence=0.4))
        frames = _frame_list(ep, s); n = len(frames)
        lm2d = np.full((n, 2, 21, 2), np.nan, np.float32); lm3d = np.full((n, 2, 21, 3), np.nan, np.float32); score = np.zeros((n, 2), np.float32)
        for k, f in enumerate(frames):
            img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
            r = hl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), int(k / ep.proc_fps * 1000))
            for h in range(len(r.hand_landmarks)):
                side = 0 if r.handedness[h][0].category_name == "Left" else 1
                lm2d[k, side] = [(p.x * img.shape[1], p.y * img.shape[0]) for p in r.hand_landmarks[h]]
                lm3d[k, side] = [(p.x, p.y, p.z) for p in r.hand_world_landmarks[h]]
                score[k, side] = r.handedness[h][0].score
        np.savez_compressed(out_dir / f"{s.name}.npz", lm2d=lm2d, lm3d=lm3d, score=score)
    ep.set_status("hands", "done", "MediaPipe 21-landmark hands on ego views")


def objects(ep: Episode, device: str = "mps") -> None:
    from ultralytics import YOLO
    ep.set_status("objects", "running")
    model = YOLO("yolov8s-worldv2.pt"); model.set_classes(sorted(set(VOCAB)))
    out_dir = ep.derived / "objects"; out_dir.mkdir(exist_ok=True)
    for s in ep.egos():
        frames = _frame_list(ep, s); n = len(frames)
        boxes = np.full((n, 12, 5), np.nan, np.float32); names = np.full((n, 12), "", dtype="<U24")
        for start in range(0, n, 64):
            res = model.predict([str(p) for p in frames[start: start + 64]], device=device, verbose=False, conf=0.3, imgsz=ep.proc_size)
            for j, r in enumerate(res):
                k = start + j; cf = r.boxes.conf.cpu().numpy(); order = np.argsort(-cf)[:12]
                for i, idx in enumerate(order):
                    boxes[k, i, :4] = r.boxes.xyxy.cpu().numpy()[idx]; boxes[k, i, 4] = cf[idx]; names[k, i] = model.names[int(r.boxes.cls[idx])]
        np.savez_compressed(out_dir / f"{s.name}.npz", boxes=boxes, names=names)
    ep.set_status("objects", "done", "YOLO-World, kitchen/household vocabulary, ego views")


def body3d(ep: Episode) -> None:
    """Monocular 3D body (MediaPipe world landmarks) on the exo view with the largest person; falls back to ego."""
    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    ep.set_status("body3d", "running")
    # choose view: exo with the largest median person box from body2d
    best, best_area = None, 0.0
    for s in ep.exos() or ep.streams:
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if not z.exists(): continue
        b = np.load(z)["boxes"]; area = np.nanmedian((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]))
        if np.isfinite(area) and area > best_area: best, best_area = s, float(area)
    if best is None:
        ep.set_status("body3d", "failed", "no body2d results"); return
    po = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_lite.task")), running_mode=vision.RunningMode.VIDEO, num_poses=2))
    frames = _frame_list(ep, best); n = len(frames)
    world = np.full((n, 2, 33, 3), np.nan, np.float32); img2d = np.full((n, 2, 33, 3), np.nan, np.float32)
    for k, f in enumerate(frames):
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        r = po.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), int(k / ep.proc_fps * 1000))
        for p in range(min(2, len(r.pose_landmarks))):
            world[k, p] = [(l.x, l.y, l.z) for l in r.pose_world_landmarks[p]]
            img2d[k, p] = [(l.x * img.shape[1], l.y * img.shape[0], l.visibility) for l in r.pose_landmarks[p]]
    out_dir = ep.derived / "body3d"; out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / "body3d.npz", world=world, img2d=img2d, stream=best.name)
    ep.set_status("body3d", "done", f"monocular MediaPipe world landmarks on {best.name} (not triangulated)")
