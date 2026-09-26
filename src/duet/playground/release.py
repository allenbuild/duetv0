"""Release packaging: what turns an episode into something that can leave the building.

    releases/<version>/<episode>/
        streams/<stream>.mp4           face-blurred re-encode of EVERY stream (audio kept, NOT anonymised)
        derived/records.parquet        the per-frame record table (copied verbatim)
        derived/<stage>/<file>.json    transcript / annotations / proposals / contact events / metrics / qc, if present
        episode.json                   streams, offsets, stage status
        schema.json                    every column of records.parquet: name, dtype, shape, unit, meaning
        manifest.json                  sha256 + size of every file, code rev, stage snapshot, package versions, face-blur stats
        LICENSE                        CC BY-NC 4.0 notice (DRAFT, founders to confirm)
        consent.json                   one entry per person, consent_recorded: false until a human fills it in
        README.md

Face blurring. Faces are found per frame by a union of detectors, because no single one was
reliable on this footage (see docs/design/release.md for the measurements):
  * MediaPipe BlazeFace full-range (bundled with the mediapipe wheel, legacy solutions API)
  * MediaPipe BlazeFace short-range (tasks API, blaze_face_short_range.tflite downloaded once to
    data/playground/models/) run on 3x2 overlapping tiles upscaled 2x, so small far faces are in range
  * YOLOv8n-pose face keypoints (nose/eyes/ears) turned into a head box, which also covers profile and
    partly turned-away heads that face detectors miss
  * OpenCV Haar cascade only when mediapipe cannot be imported.
Boxes are tracked across frames by IoU; a track keeps blurring for 0.5 s after its last detection so
the blur does not flicker. Boxes are enlarged 30 percent and Gaussian-blurred with a kernel
proportional to the box. False positives cost nothing but blur; misses are the failure that matters.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _meta
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import Episode

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "data/playground/models"
RELEASES = ROOT / "data/playground/releases"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FACE_MODEL = MODELS / "blaze_face_short_range.tflite"

ENLARGE = 0.30       # box enlargement before blurring
PERSIST_S = 0.5      # a track keeps blurring this long after its last detection
IOU_MATCH = 0.3      # detection <-> track association
TILE_MIN_SCORE = 0.6 # the tiled short-range detector produces false positives below this
TILE_SOLO_SCORE = 0.85  # a tile detection that no other detector corroborates needs at least this score
MAX_FACE_FRAC = 0.35    # no face box is wider/taller than this fraction of the frame's shorter side (kills counter/bowl boxes)
EXTRA_FILES = ["derived/speech/transcript.json", "derived/annotate/annotations.json", "derived/autolabel/proposals.json",
               "derived/contact/events.json", "derived/metrics/session.json", "derived/qc/report.json"]
KEY_PACKAGES = ["mediapipe", "ultralytics", "opencv-python", "numpy", "pandas", "pyarrow", "torch", "fastapi"]


# ----------------------------------------------------------------------------- face detection

def ensure_face_model(download: bool = True) -> Path | None:
    """The short-range BlazeFace model, downloaded once from Google's mediapipe-models bucket."""
    if FACE_MODEL.exists():
        return FACE_MODEL
    if not download:
        return None
    MODELS.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    return FACE_MODEL if FACE_MODEL.exists() and FACE_MODEL.stat().st_size > 100_000 else None


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _touches(a, b) -> bool:
    """a's centre inside b, b's centre inside a, or IoU > 0.1."""
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2); cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return (b[0] <= ca[0] <= b[2] and b[1] <= ca[1] <= b[3]) or (a[0] <= cb[0] <= a[2] and a[1] <= cb[1] <= a[3]) or _iou(a, b) > 0.1


def filter_boxes(boxes: list[tuple], w: int, h: int) -> list[tuple]:
    """Drop implausibly large boxes, and tile detections that neither another detector corroborates nor a high score supports.
    The tiled short-range detector fires on bowls, counters and torsos; on its own it blurred the hands and the object in the
    exo view of a handover, which is the one thing the data is for."""
    lim = MAX_FACE_FRAC * min(w, h)
    boxes = [b for b in boxes if max(b[2] - b[0], b[3] - b[1]) <= lim]
    anchors = [b for b in boxes if b[5] != "mp_short_tiles"]
    return anchors + [b for b in boxes if b[5] == "mp_short_tiles" and (b[4] >= TILE_SOLO_SCORE or any(_touches(b, a) for a in anchors))]


def _nms(boxes: list[tuple], thr: float = 0.4) -> list[tuple]:
    out: list[tuple] = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if all(_iou(b, o) < thr for o in out):
            out.append(b)
    return out


class FaceDetectors:
    """Union of face detectors. Each returns boxes (x0, y0, x1, y1, score, source) in full-frame pixels."""

    def __init__(self, backends: tuple[str, ...] = ("mp_full", "mp_short_tiles", "pose"), device: str = "mps"):
        self.backends = list(backends); self.device = device; self._full = self._short = self._pose = self._haar = None
        self.name = "+".join(self.backends)
        try:
            import mediapipe  # noqa: F401
        except Exception:  # noqa: BLE001
            self.backends = ["haar"]; self.name = "haar (mediapipe unavailable)"

    # lazy constructors so tests that never detect do not load models
    def _get_full(self):
        if self._full is None:
            import mediapipe as mp
            self._full = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        return self._full

    def _get_short(self):
        if self._short is None:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            model = ensure_face_model()
            if model is None:
                raise RuntimeError(f"face model missing and could not be downloaded from {FACE_MODEL_URL}")
            self._short = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5))
        return self._short

    def _get_pose(self):
        if self._pose is None:
            from ultralytics import YOLO
            self._pose = YOLO("yolov8n-pose.pt")
        return self._pose

    def _get_haar(self):
        if self._haar is None:
            import cv2
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return self._haar

    def detect(self, bgr: np.ndarray) -> list[tuple]:
        import cv2
        h, w = bgr.shape[:2]; out: list[tuple] = []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if "mp_full" in self.backends:
            r = self._get_full().process(rgb)
            for d in (r.detections or []):
                b = d.location_data.relative_bounding_box
                out.append((b.xmin * w, b.ymin * h, (b.xmin + b.width) * w, (b.ymin + b.height) * h, float(d.score[0]), "mp_full"))
        if "mp_short_tiles" in self.backends:
            import mediapipe as mp
            det = self._get_short(); nx, ny, up = 3, 2, 2.0; tw, th = w / nx, h / ny
            for i in range(nx):
                for j in range(ny):
                    x0 = int(max(0, i * tw - tw * 0.25)); y0 = int(max(0, j * th - th * 0.25)); x1 = int(min(w, (i + 1) * tw + tw * 0.25)); y1 = int(min(h, (j + 1) * th + th * 0.25))
                    crop = cv2.resize(rgb[y0:y1, x0:x1], None, fx=up, fy=up)
                    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
                    for d in r.detections:
                        bb = d.bounding_box; sc = float(d.categories[0].score)
                        if sc >= TILE_MIN_SCORE:
                            out.append((x0 + bb.origin_x / up, y0 + bb.origin_y / up, x0 + (bb.origin_x + bb.width) / up, y0 + (bb.origin_y + bb.height) / up, sc, "mp_short_tiles"))
        if "pose" in self.backends:
            out += self._pose_boxes(bgr)
        if "haar" in self.backends:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for (x, y, bw, bh) in self._get_haar().detectMultiScale(g, 1.1, 5, minSize=(30, 30)):
                out.append((float(x), float(y), float(x + bw), float(y + bh), 1.0, "haar"))
        return _nms(filter_boxes(out, w, h))

    def _pose_boxes(self, bgr) -> list[tuple]:
        r = self._get_pose().predict(bgr, device=self.device, verbose=False, conf=0.35)[0]
        if r.keypoints is None or r.keypoints.conf is None:
            return []
        kp = r.keypoints.xy.cpu().numpy(); cf = r.keypoints.conf.cpu().numpy(); out = []
        for p in range(len(kp)):
            ok = cf[p, :5] > 0.3
            if ok.sum() < 2:
                continue
            pts = kp[p, :5][ok]; cx, cy = pts.mean(0)
            spread = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
            sh = float(np.linalg.norm(kp[p, 5] - kp[p, 6])) if (cf[p, 5] > 0.3 and cf[p, 6] > 0.3) else 0.0
            s = max(spread * 2.2, sh * 0.7, 30.0)
            out.append((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, float(cf[p, :5].max()), "pose"))
        return out


# ----------------------------------------------------------------------------- tracking + blur (pure, testable)

@dataclass
class _Track:
    box: np.ndarray
    last_seen: int
    hits: int = 1


class FaceTracker:
    """IoU association with persistence: a box keeps being blurred for ``persist_s`` after its last detection."""

    def __init__(self, fps: float, persist_s: float = PERSIST_S, iou_thr: float = IOU_MATCH):
        self.persist = max(1, int(round(persist_s * fps))); self.iou_thr = iou_thr; self.tracks: list[_Track] = []

    def update(self, k: int, boxes: list[tuple]) -> list[np.ndarray]:
        matched: set[int] = set()
        for b in boxes:
            b = np.asarray(b[:4], float); best, best_iou = None, self.iou_thr
            for i, t in enumerate(self.tracks):
                if i in matched:
                    continue
                v = _iou(t.box, b)
                if v > best_iou:
                    best, best_iou = i, v
            if best is None:
                self.tracks.append(_Track(b, k)); matched.add(len(self.tracks) - 1)
            else:
                t = self.tracks[best]; t.box = 0.6 * b + 0.4 * t.box; t.last_seen = k; t.hits += 1; matched.add(best)
        self.tracks = [t for t in self.tracks if k - t.last_seen <= self.persist]
        return [t.box.copy() for t in self.tracks]


def enlarge_box(box, frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; bw, bh = (x1 - x0) * (1 + frac), (y1 - y0) * (1 + frac)
    return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))


def blur_kernel(box) -> int:
    k = int(0.6 * max(box[2] - box[0], box[3] - box[1])); k = max(15, k)
    return k if k % 2 == 1 else k + 1


def blur_boxes(img: np.ndarray, boxes, frac: float = ENLARGE) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    for b in boxes:
        x0, y0, x1, y1 = enlarge_box(b, frac, w, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        k = blur_kernel((x0, y0, x1, y1)); img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (k, k), 0)
    return img


# ----------------------------------------------------------------------------- video re-encode

def blur_stream(src: Path, dst: Path, detectors: FaceDetectors, example_png: Path | None = None, example_frame: int = 0, log=print) -> dict:
    """Detect + track + blur every frame of ``src``; write ``dst`` with libx264 video and the original audio track."""
    import cv2
    cap = cv2.VideoCapture(str(src)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-i", str(src),
           "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(dst)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    tracker = FaceTracker(fps); stats = {"frames": 0, "frames_with_detection": 0, "frames_blurred": 0, "detections": 0, "by_source": {}, "fps": fps, "width": w, "height": h}
    k = 0; t0 = time.time()
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                break
            dets = detectors.detect(im); active = tracker.update(k, dets)
            stats["frames"] += 1; stats["frames_with_detection"] += bool(dets); stats["frames_blurred"] += bool(active); stats["detections"] += len(dets)
            for d in dets:
                stats["by_source"][d[5]] = stats["by_source"].get(d[5], 0) + 1
            if active:
                im = blur_boxes(im, active)
            if example_png is not None and k == example_frame:
                cv2.imwrite(str(example_png), im)
            proc.stdin.write(im.tobytes()); k += 1
            if k % 300 == 0:
                log(f"    {src.name}: {k}/{n} frames, {stats['frames_blurred']} blurred, {(time.time() - t0) / k * 1000:.0f} ms/frame")
    finally:
        proc.stdin.close(); err = proc.stderr.read().decode(errors="replace"); proc.wait(); cap.release()
    if proc.returncode != 0:
        if "-c:a" in cmd and "copy" in cmd:  # audio codec not muxable into mp4 as-is: retry re-encoding the audio
            cmd[cmd.index("copy")] = "aac"
            return _blur_stream_retry(src, dst, detectors, cmd, example_png, example_frame, log)
        raise RuntimeError(f"ffmpeg failed for {src.name}: {err[-400:]}")
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def _blur_stream_retry(src, dst, detectors, cmd, example_png, example_frame, log):
    import cv2
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; tracker = FaceTracker(fps)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stats = {"frames": 0, "frames_with_detection": 0, "frames_blurred": 0, "detections": 0, "by_source": {}, "fps": fps, "audio": "re-encoded aac"}
    k = 0
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                break
            dets = detectors.detect(im); active = tracker.update(k, dets)
            stats["frames"] += 1; stats["frames_with_detection"] += bool(dets); stats["frames_blurred"] += bool(active); stats["detections"] += len(dets)
            for d in dets:
                stats["by_source"][d[5]] = stats["by_source"].get(d[5], 0) + 1
            if active:
                im = blur_boxes(im, active)
            if example_png is not None and k == example_frame:
                cv2.imwrite(str(example_png), im)
            proc.stdin.write(im.tobytes()); k += 1
    finally:
        proc.stdin.close(); err = proc.stderr.read().decode(errors="replace"); proc.wait(); cap.release()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src.name}: {err[-400:]}")
    return stats


# ----------------------------------------------------------------------------- schema

# (regex on the column name, unit, meaning). {stream}/{p}/{side}/{tail} come from named groups. Order matters.
def _families(streams: list[str], persons: list[str]) -> list[tuple[re.Pattern, str, str]]:
    S = "(?P<stream>" + "|".join(map(re.escape, streams)) + ")" if streams else "(?P<stream>[A-Za-z0-9]+)"
    P = "(?P<person>" + "|".join(map(re.escape, persons)) + ")" if persons else "(?P<person>[A-Za-z0-9]+)"
    fam = [
        (r"^t_s$", "s", "time on the common reference timeline: frame k is common_start_s + k / proc_fps in the reference stream's clock"),
        (S + r"_body2d_p(?P<p>\d+)$", "px", "COCO-17 body keypoints (x, y, confidence) x 17 of tracked person slot {p} in stream {stream}, pixels of the proc_size frame; NaN when the slot is empty. Slots are IoU-tracked per view, not identities"),
        (S + r"_hand_(?P<side>[LR])_present$", "bool", "MediaPipe found the {side} hand of the {stream} wearer in this frame"),
        (S + r"_hand_(?P<side>[LR])_score$", "0-1", "MediaPipe handedness confidence for the {side} hand in stream {stream}"),
        (S + r"_hand_(?P<side>[LR])_lm2d$", "px", "21 MediaPipe hand landmarks (x, y) of the {side} hand in stream {stream}, pixels of the proc_size frame; landmark 0 is the wrist"),
        (S + r"_hand_(?P<side>[LR])_lm3d$", "m", "21 MediaPipe world landmarks (x, y, z) of the {side} hand in stream {stream}, metres, wrist-relative (not in any world frame)"),
        (S + r"_objects$", "json", "JSON list of [name, confidence, x0, y0, x1, y1] YOLO-World detections in stream {stream} (kitchen/household vocabulary), pixels of the proc_size frame"),
        (S + r"_imu_arm$", "m", "IMU-harness arm chain for stream {stream}: 2 sides x 4 joints (chest, shoulder, elbow, hand) x xyz, chest-relative metres"),
        (S + r"_qc_sharp$", "var(Laplacian)", "sharpness of the {stream} frame (variance of the Laplacian); low = blurred"),
        (S + r"_qc_bright$", "0-255", "mean grey level of the {stream} frame"),
        (S + r"_qc_motion$", "px/frame", "median optical-flow magnitude of the {stream} frame vs the previous one, pixels at proc_size"),
        (r"^world_body_p(?P<p>\d+)$", "m", "triangulated COCO-17 body of person slot {p}: 17 x (x, y, z) in the world (board) frame; NaN without >= 2 calibrated cameras"),
        (r"^world_hands3d_" + S + r"$", "m", "21 hand landmarks x 2 hands x (x, y, z) of the {stream} wearer in the world frame (ZED depth + head pose); NaN when unavailable"),
        (r"^world_head_" + S + r"$", "m", "head (ego camera) position of the {stream} wearer in the world frame"),
        (r"^world_object_(?P<tail>.+)$", "m", "position of tagged object '{tail}' in the world frame (AprilTag)"),
        (r"^head_dist_m$", "m", "distance between the two wearers' heads in the world frame"),
        (r"^min_wrist_dist_m$", "m", "minimum distance between any wrist of person 0 and any wrist of person 1 (triangulated bodies); NaN without world 3D"),
        (S + r"_facing_partner_cos$", "cos", "cosine between the {stream} wearer's head forward axis and the direction to the partner's head (1 = facing them)"),
        (S + r"_T_world_cam$", "m, unitless", "4x4 pose (row-major) of the {stream} ego camera in the world frame"),
        (S + r"_headpose_backend$", "str", "which backend produced the {stream} head pose (zed_tracking+board, head_tag, None)"),
        (r"^body3d_p(?P<p>\d+)$", "m", "monocular MediaPipe pose world landmarks of person {p}: 33 x (x, y, z), hip-centred metres, from the exo view named in body3d_stream; NOT triangulated"),
        (r"^body3d_stream$", "str", "the exo stream the monocular body3d came from"),
        # per-frame columns contributed by later stages via derived/<stage>/records_extra.parquet (prefixed with the stage name)
        (r"^track_(?P<tail>.+)$", "id", "person identity tracking (track stage): {tail}; stable person ids across frames/views"),
        (r"^contact_(?P<tail>.+)$", "label", "hand-object contact (contact stage): {tail}; which object is in which hand"),
        (r"^gaze_proxy_(?P<tail>.+)$", "unitless", "gaze proxy from head pose (gaze_proxy stage): {tail}; forward-ray 'looking at partner/object' features"),
        (r"^speech_speaking_" + P + r"$", "bool", "{person} is speaking in this frame (speech stage: transcript + speaker attribution across ego mics)"),
        (r"^speech_(?P<tail>.+)$", "unitless", "speech stage per-frame feature: {tail}"),
        (r"^autolabel_(?P<tail>.+)$", "label", "proposed event labels for review (autolabel stage): {tail}"),
        (r"^metrics_hand_speed_" + P + r"$", "m/s or frame-widths/s (see derived/metrics/session.json hand_speed.unit)", "smoothed wrist speed of {person} (max over both hands); NaN when no hand is visible"),
        (r"^metrics_idle_" + P + r"$", "bool", "{person} is idle: hand speed below threshold for >= 2 s (metrics stage)"),
        (r"^metrics_hand_visible_" + P + r"$", "bool", "at least one hand of {person} is detected in their ego view (metrics stage)"),
        (r"^metrics_(?P<tail>.+)$", "unitless", "metrics stage per-frame feature: {tail}"),
        (r"^stereo_depth_(?P<tail>.+)$", "m", "stereo depth stage per-frame feature: {tail}"),
        (r"^depth_mono_(?P<tail>.+)$", "m", "monocular depth stage per-frame feature: {tail}"),
        (r"^annotate_(?P<tail>.+)$", "label", "dense VLM annotation feature (annotate stage): {tail}"),
    ]
    return [(re.compile(rx), unit, meaning) for rx, unit, meaning in fam]


def _dtype_shape(col: pd.Series) -> tuple[str, list | str]:
    v = col.iloc[0] if len(col) else None
    if isinstance(v, (list, np.ndarray)):
        lens = {len(x) for x in col if isinstance(x, (list, np.ndarray))}
        inner = np.asarray(v).dtype.name if len(v) else "float"
        return f"list[{inner}]", [lens.pop()] if len(lens) == 1 else "variable"
    if isinstance(v, str):
        return "str", []
    return str(col.dtype), []


def build_schema(df: pd.DataFrame, streams: list[str], persons: list[str], ep_meta: dict | None = None) -> dict:
    fams = _families(streams, persons); cols = []; unknown = 0
    for name in df.columns:
        dtype, shape = _dtype_shape(df[name]); unit, meaning = "unknown", "unknown"
        for rx, u, m in fams:
            mt = rx.match(name)
            if mt:
                unit = u; meaning = m.format(**{k: (v or "") for k, v in mt.groupdict().items()}); break
        unknown += meaning == "unknown"
        cols.append({"name": name, "dtype": dtype, "shape": shape, "unit": unit, "meaning": meaning})
    return {"rows": int(len(df)), "n_columns": len(cols), "n_unknown": unknown, "unknown_fraction": unknown / max(1, len(cols)),
            "row_meaning": "one row per frame of the common timeline at proc_fps; list columns hold fixed-length vectors (row-major flattening of the shapes given in 'meaning')",
            "episode": ep_meta or {}, "columns": cols}


# ----------------------------------------------------------------------------- manifest / licence / consent / readme

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in KEY_PACKAGES:
        try:
            out[p] = _meta.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    try:
        out["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:  # noqa: BLE001
        out["ffmpeg"] = None
    return out


def write_manifest(out_dir: Path, ep: Episode, version: str, extra: dict | None = None) -> dict:
    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files.append({"path": str(p.relative_to(out_dir)), "sha256": sha256(p), "bytes": p.stat().st_size})
    man = {"release_version": version, "episode": ep.name, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_git_rev": git_rev(),
           "stage_status": ep.status, "streams": [{"name": s.name, "role": s.role, "person": s.person, "offset_s": s.offset_s} for s in ep.streams],
           "versions": package_versions(), "files": files, **(extra or {})}
    json.dump(man, open(out_dir / "manifest.json", "w"), indent=1)
    return man


def verify_release(out_dir: Path) -> dict:
    """Re-hash every file listed in manifest.json; report mismatches, missing and untracked files."""
    man = json.load(open(out_dir / "manifest.json")); listed = {f["path"]: f for f in man["files"]}
    mismatched, missing = [], []
    for rel, f in listed.items():
        p = out_dir / rel
        if not p.exists():
            missing.append(rel); continue
        if sha256(p) != f["sha256"] or p.stat().st_size != f["bytes"]:
            mismatched.append(rel)
    untracked = [str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file() and p.name != "manifest.json" and str(p.relative_to(out_dir)) not in listed]
    return {"ok": not (mismatched or missing), "checked": len(listed), "mismatched": mismatched, "missing": missing, "untracked": untracked,
            "release_version": man.get("release_version"), "episode": man.get("episode"), "created_utc": man.get("created_utc")}


LICENSE_TEXT = """DRAFT, founders to confirm. This notice is a placeholder until Duet's founders sign off on the licence.

Duet paired-interaction dataset release {version}, episode {episode}.

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0). To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

You are free to share (copy and redistribute the material in any medium or format) and adapt (remix,
transform, and build upon the material) under the following terms:
  Attribution    You must give appropriate credit, provide a link to the license, and indicate if changes were made.
  NonCommercial  You may not use the material for commercial purposes.
  No additional restrictions may be applied that legally restrict others from doing what the license permits.

Faces in the video streams are blurred. Audio is NOT anonymised: voices, names and anything said are
present as recorded. Consent status per person is in consent.json; do not redistribute an episode whose
consent entries are not all consent_recorded: true.
"""


def persons_in(ep: Episode) -> list[str]:
    seen: list[str] = []
    for s in ep.streams:
        if s.person and s.person not in seen:
            seen.append(s.person)
    if not seen:
        seen = [f"person_{i}" for i in range(max(1, len(ep.egos())))]
    return seen


def consent_template(ep: Episode) -> dict:
    return {"episode": ep.name, "template": True, "instructions": "one entry per person appearing in the episode; set consent_recorded to true only after a signed consent form exists and is on file",
            "persons": [{"name": p, "date": "", "consent_recorded": False, "contact": "", "scope": "research dataset"} for p in persons_in(ep)]}


def readme_text(ep: Episode, version: str, blur: dict, copied: list[str]) -> str:
    streams = "\n".join(f"| {s.name} | {s.role} | {s.person or ''} | {s.width}x{s.height} @ {s.fps:.0f} fps | {s.offset_s:+.2f} s |" for s in ep.streams)
    blur_rows = "\n".join(f"| {k} | {v.get('frames', 0)} | {v.get('frames_with_detection', 0)} | {v.get('frames_blurred', 0)} |" for k, v in blur.get("streams", {}).items())
    return f"""# Duet episode release: {ep.name} ({version})

Generated by `scripts/release_episode.py` on {time.strftime('%Y-%m-%d')}. Verify integrity with
`.venv/bin/python scripts/release_episode.py {ep.name} --version {version} --verify`.

## Contents

| file | what |
|---|---|
| `streams/<stream>.mp4` | every camera, re-encoded with faces blurred (H.264, original audio track kept) |
| `derived/records.parquet` | one row per frame of the common timeline ({ep.proc_fps:g} fps): hands, bodies, objects, 3D, QC and later-stage columns |
| `schema.json` | name, dtype, shape, unit and meaning of every column of records.parquet |
| `episode.json` | streams, time offsets (`t_ref = t_stream + offset_s`), stage status |
| `manifest.json` | sha256 and size of every file, code revision, package versions, face-blur statistics |
| `consent.json` | one entry per person; a template until a human records consent |
| `LICENSE` | CC BY-NC 4.0 notice, DRAFT |
{chr(10).join(f'| `{c}` | copied from the episode |' for c in copied if c not in ('derived/records.parquet', 'episode.json'))}

## Streams

| stream | role | person | size | offset to reference ({ep.reference}) |
|---|---|---|---|---|
{streams}

Common timeline: {ep.common_start_s:.2f} s to {ep.common_end_s:.2f} s of the reference stream ({ep.common_end_s - ep.common_start_s:.1f} s).

## Anonymisation

Faces were blurred with: {blur.get('backend', 'none')}. Boxes are tracked across frames (IoU, {PERSIST_S} s persistence),
enlarged {int(ENLARGE * 100)} percent and Gaussian-blurred. Detection statistics per stream (30 fps video frames):

| stream | frames | frames with >= 1 detection | frames blurred (incl. persistence) |
|---|---|---|---|
{blur_rows}

What is NOT anonymised: audio (voices, names, anything said), body shape, clothing, tattoos, the room and its
contents, on-screen text, and any face the detectors missed (partial faces at the fisheye edge of the ego
cameras are the known weak spot). Review the videos before sharing outside the consent scope.

## Licence and consent

See `LICENSE` (DRAFT, founders to confirm) and `consent.json`. Do not redistribute until every person's entry
says `consent_recorded: true`.
"""


# ----------------------------------------------------------------------------- release

def release(ep: Episode, version: str, out_root: Path = RELEASES, blur: bool = True, detectors: FaceDetectors | None = None,
            example_png: Path | None = None, example_frame: int = 0, example_stream: str | None = None, log=print) -> Path:
    out = Path(out_root) / version / ep.name
    if out.exists():
        shutil.rmtree(out)
    (out / "streams").mkdir(parents=True); (out / "derived").mkdir()
    blur_info: dict = {"backend": "none (NOT ANONYMISED)", "streams": {}}
    if blur:
        detectors = detectors or FaceDetectors(); blur_info["backend"] = detectors.name
    for s in ep.streams:
        src = (ep.dir / s.path).resolve(); dst = out / "streams" / f"{s.name}.mp4"
        if blur:
            log(f"  blurring {s.name}")
            png = example_png if (example_png is not None and s.name == (example_stream or ep.streams[0].name)) else None
            blur_info["streams"][s.name] = blur_stream(src, dst, detectors, png, example_frame, log)
            log(f"    {s.name}: {blur_info['streams'][s.name]}")
        else:
            shutil.copy2(src, dst)
    copied = []
    for rel in ["derived/records.parquet", "episode.json", *EXTRA_FILES]:
        p = ep.dir / rel
        if p.exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, out / rel); copied.append(rel)
    if (out / "derived/records.parquet").exists():
        df = pd.read_parquet(out / "derived/records.parquet")
        schema = build_schema(df, [s.name for s in ep.streams], persons_in(ep), {"name": ep.name, "proc_fps": ep.proc_fps, "proc_size": ep.proc_size, "common_start_s": ep.common_start_s, "reference": ep.reference})
        json.dump(schema, open(out / "schema.json", "w"), indent=1)
        log(f"  schema: {schema['n_columns']} columns, {schema['n_unknown']} unknown")
    (out / "LICENSE").write_text(LICENSE_TEXT.format(version=version, episode=ep.name))
    json.dump(consent_template(ep), open(out / "consent.json", "w"), indent=1)
    (out / "README.md").write_text(readme_text(ep, version, blur_info, copied))
    write_manifest(out, ep, version, {"face_blur": blur_info, "copied": copied})
    log(f"  release written to {out}")
    return out
