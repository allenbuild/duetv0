"""Per-frame perception on the common timeline (contracts C4/C5).

frames    PyAV decodes every usable stream once (streams in parallel) and keeps, for processed frame k, the
          source frame whose pts is NEAREST to ``Episode.stream_time(s, t_ref_k)`` with
          t_ref_k = common_start_s + k / proc_fps (ties -> earlier frame). The first frame's display matrix (container
          or SEI) is applied like ffmpeg's autorotate (runtime.display_transform -> rotation, flip: the values probe
          records), the long side is resized to proc_size (aspect kept, even size) and the frame written as
          derived/frames/<s>/NNNNNN.jpg (1-based name, 0-based k). derived/frames/<s>/index.parquet records k, file,
          t_ref_s (nominal), t_src_s (actual pts on the stream timeline: container pts minus video_start_s) and
          src_frame (0-based decoded-frame index); derived/frames/manifest.json the frame size, scale (= extracted
          width / display width), source identity and the offsets/window/proc_fps used.
body2d    YOLOv8-pose (COCO-17, extracted-frame px) on every usable view. People are kept in 4 slots by
          ``SlotTracker`` (duplicate boxes merged, Hungarian IoU/centre matching, per-track miss counters);
          slots are stable within ONE stream only.
hands     MediaPipe HandLandmarker (VIDEO mode, up to 4 hands) on ego views. Every detection is kept: it is
          attributed to the WEARER or the PARTNER and to a left/right slot jointly (``attribute_hands``: forearm
          direction/entry, wrist height, gated partner arms from body2d, temporal continuity; a wearer prior only
          when the episode declares nobody else and nobody is seen in the view), so two hands never overwrite each other. lm3d is wrist-relative (MediaPipe world landmarks are hand-centred:
          the wrist sits ~8 cm from their origin, so landmark 0 is subtracted), metres. ``score`` is the MediaPipe
          handedness probability of the slot's side, NOT a detection confidence.
objects   YOLO-World with the kitchen/household vocabulary on ego views, class-agnostic NMS (the vocabulary
          overlaps: "spice jar"/"oil bottle", "bowl"/"pan").
body3d    MediaPipe PoseLandmarker world landmarks (33 joints, metres, hip-centred) on the exo view with the
          largest person; monocular, not triangulated. Skipped (no file) when there is no usable exo view.

Model weights come from ``runtime.models_dir()`` (``scripts/playground.py fetch-models``; YOLO-World with the vocabulary
baked in by ``bake_world_vocab``); ultralytics runs with YOLO_AUTOINSTALL=False and YOLO_OFFLINE=True (no pip installs,
downloads or telemetry at run time). Devices come from ``runtime.pick_device(ep.device)`` (fp16 on CUDA only).
"""
from __future__ import annotations

import hashlib
import io
import itertools
import os
import shutil
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import runtime
from .episode import Episode, Stream

VOCAB = ["pan", "pot", "lid", "bowl", "plate", "cup", "glass", "jar", "container", "bag", "package", "box", "knife", "spoon", "spatula", "ladle",
         "whisk", "strainer", "cutting board", "tongs", "scissors", "onion", "garlic", "tomato", "mushroom", "carrot", "pepper", "vegetable", "meat",
         "chicken", "egg", "dough", "flour", "rice", "pasta", "cheese", "bread", "salt shaker", "spice jar", "oil bottle", "sauce bottle", "bottle",
         "towel", "sponge", "sink", "oven", "microwave", "fridge", "stove knob", "cupboard", "laundry", "shirt", "basket", "phone", "tool"]
WORLD_VOCAB = sorted(set(VOCAB))
POSE_WEIGHTS, WORLD_WEIGHTS = "yolov8n-pose.pt", "yolov8s-worldv2.pt"
HAND_TASK, POSE_TASK = "hand_landmarker.task", "pose_landmarker_lite.task"
JPEG_QUALITY = 85  # ~same size as the old ffmpeg -q:v 3 frames, PSNR ~40 dB
MAX_PEOPLE, MAX_OBJECTS, MAX_HANDS = 4, 12, 4
FRAMES_VERSION = 1


# ============================================================================================== frames (C4)

def frames_dir(ep: Episode, s: Stream | str) -> Path:
    return ep.derived / "frames" / (s if isinstance(s, str) else s.name)


def proc_dims(width: int, height: int, proc_size: int) -> tuple[int, int]:
    """Extracted (w, h) for a DISPLAY size: long side = proc_size, other side keeps the aspect ratio rounded to an even
    number of pixels (as ffmpeg ``scale=proc_size:-2``)."""
    assert width > 0 and height > 0 and proc_size > 0, (width, height, proc_size)
    if width >= height:
        return proc_size, max(2, int(np.floor(height * proc_size / width / 2 + 0.5)) * 2)
    return max(2, int(np.floor(width * proc_size / height / 2 + 0.5)) * 2), proc_size


def select_nearest(frames: Iterable[tuple[float, Any]], targets: np.ndarray, tol: float) -> Iterator[tuple[int, int, float, Any]]:
    """Nearest-frame selection over a decoded stream.

    ``frames`` yields (t, payload) with t strictly increasing (seconds); ``targets`` are ascending times on the same
    clock. Yields (k, i, t_i, payload_i) for every target k in order, where frame i (enumeration index) is the one
    whose time is nearest to targets[k]; ties go to the earlier frame. Consumption stops once every target is served.
    ValueError when a target lies more than ``tol`` before the first or after the last frame (window not covered)."""
    targets = np.asarray(targets, dtype=float); n = len(targets); k = 0; prev: tuple[int, float, Any] | None = None
    assert n == 0 or np.all(np.diff(targets) > 0), "targets must be strictly increasing"
    for i, (t, payload) in enumerate(frames):
        if prev is not None and not t > prev[1]:
            raise ValueError(f"frame times not strictly increasing at frame {i}: {t} after {prev[1]}")
        while k < n and targets[k] <= t:
            if prev is None:
                if t - targets[k] > tol:
                    raise ValueError(f"target t={targets[k]:.4f}s precedes the first frame (t={t:.4f}s) by more than {tol:.4f}s")
                yield k, i, t, payload
            elif targets[k] - prev[1] <= t - targets[k]:
                yield (k, *prev)
            else:
                yield k, i, t, payload
            k += 1
        if k == n:
            return
        prev = (i, t, payload)
    if k < n:
        if prev is None:
            raise ValueError("no decodable frames")
        if targets[n - 1] - prev[1] > tol:
            raise ValueError(f"target t={targets[n - 1]:.4f}s is past the last frame (t={prev[1]:.4f}s) by more than {tol:.4f}s")
        for kk in range(k, n):
            yield (kk, *prev)


def apply_display(a: np.ndarray, rotation: int, flip: bool) -> np.ndarray:
    """Decoded (H, W[, C]) frame -> display orientation for runtime.display_transform(matrix) = (rotation, flip), the
    values probe stores in Stream.rotation/flip: ``np.rot90(a[::-1] if flip else a, rotation // 90)`` (vertical flip
    first, then counter-clockwise rotation). Pixel-identical to ffmpeg's autorotate."""
    if rotation % 90 or not isinstance(flip, (bool, np.bool_)):
        raise ValueError(f"bad display transform rotation={rotation!r} flip={flip!r}")
    a = a[::-1] if flip else a
    return np.rot90(a, (rotation // 90) % 4) if rotation % 360 else a


def _frame_display_matrix(fr: Any) -> np.ndarray | None:
    """The frame's DISPLAYMATRIX side data (container matrix or H.264/HEVC display-orientation SEI), or None."""
    from av.sidedata.sidedata import Type
    for sd in fr.side_data:
        if sd.type == Type.DISPLAYMATRIX:
            return np.frombuffer(bytes(sd), "<i4").copy()
    return None


def _source_identity(p: Path) -> dict:
    """size, mtime and sha256 of the first + last MiB (cheap, content-based)."""
    st = p.stat(); h = hashlib.sha256(str(st.st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END); h.update(f.read(1 << 20))
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256_head_tail": h.hexdigest()}


def _extract_stream(ep: Episode, s: Stream, out: Path, t_ref: np.ndarray, threads: int, stop: threading.Event | None = None) -> dict:
    """Decode stream ``s`` once and write the frames nearest to the common grid ``t_ref`` into ``out``.
    Returns the manifest entry; writes out/index.parquet. ``stop`` (set when another stream failed) aborts."""
    import pandas as pd
    from PIL import Image

    src = ep.dir / s.path
    if not src.exists():
        raise FileNotFoundError(f"{s.name}: source video {s.path} is missing (dangling link?)")
    t_tgt = np.asarray(ep.stream_time(s, t_ref), dtype=float)  # stream seconds, 0 = first video frame
    assert t_tgt.shape == t_ref.shape and np.all(np.isfinite(t_tgt)) and np.all(np.diff(t_tgt) > 0), f"{s.name}: bad target times"
    # file: protocol + FFMPEG_SAFE_INPUT whitelists, probe's video stream, ep.hwaccel (software for auto/none)
    with runtime.pyav_open(src, video_index=getattr(s, "video_index", None), hwaccel=getattr(ep, "hwaccel", "auto")) as inp:
        c, vs, hw_label = inp.container, inp.stream, inp.hwaccel
        vs.thread_type = "AUTO"; vs.thread_count = threads
        tb = vs.time_base; fps = float(vs.average_rate or vs.guessed_rate or 0) or None
        start = float(vs.start_time * tb) if vs.start_time is not None else None
        if start is not None and abs(start - s.video_start_s) > 1e-3:
            raise ValueError(f"{s.name}: video start_time {start:.6f}s != probed video_start_s {s.video_start_s:.6f}s (probe is stale: re-run probe)")
        origin = s.video_start_s
        tol = 1.0 / fps + 1e-3 if fps else 0.1  # a target may sit up to one source frame outside the decoded span
        geo: dict = {}; dropped = [0]; last_i = [-1]

        def decoded() -> Iterator[tuple[float, Any]]:
            prev_t = -np.inf
            for fr in c.decode(vs):
                if stop is not None and stop.is_set():
                    raise RuntimeError(f"{s.name}: cancelled (another stream failed or the stage was cancelled)")
                last_i[0] += 1
                if fr.pts is None:
                    raise ValueError(f"{s.name}: decoded frame without pts; original timestamps cannot be preserved")
                t = float(fr.pts * tb) - origin
                if not geo:  # display geometry from the first frame's matrix (container or SEI), as ffmpeg's autorotate uses it
                    m = _frame_display_matrix(fr); rot, flip = runtime.display_transform(None if m is None else [int(x) for x in m])
                    disp = (fr.height, fr.width) if rot in (90, 270) else (fr.width, fr.height)
                    if (rot, bool(flip), disp) != (int(s.rotation) % 360, bool(s.flip), (s.width, s.height)):  # same mapping as probe
                        raise ValueError(f"{s.name}: file displays {disp[0]}x{disp[1]} rot {rot} flip {flip!r} but episode says "
                                         f"{s.width}x{s.height} rot {s.rotation} flip {s.flip!r} (probe is stale: re-run probe)")
                    geo.update(rot=rot, flip=flip, matrix=None if m is None else [int(x) for x in m], coded=(fr.width, fr.height), disp=disp,
                               size=proc_dims(*disp, ep.proc_size))
                if t <= prev_t:  # duplicate / non-monotonic pts (broken muxing): dropped, counted in the manifest
                    dropped[0] += 1; continue
                prev_t = t
                yield t, (last_i[0], fr)

        def render(fr: Any, names: list[str]) -> None:
            if stop is not None and stop.is_set():
                return  # cancelled: write nothing more
            w, h = geo["size"]; cw, ch = (h, w) if geo["rot"] in (90, 270) else (w, h)
            a = fr.to_ndarray(width=cw, height=ch, format="rgb24", interpolation="AREA")
            a = np.ascontiguousarray(apply_display(a, geo["rot"], geo["flip"]))  # coded -> display orientation
            assert a.shape == (h, w, 3), (a.shape, w, h)
            buf = io.BytesIO(); Image.fromarray(a).save(buf, format="JPEG", quality=JPEG_QUALITY); data = buf.getvalue()
            for nm in names:
                (out / nm).write_bytes(data)

        rows: list[tuple[int, int, float]] = []; t0 = time.time()
        with ThreadPoolExecutor(2) as pool:
            futs: deque = deque(); pend: list | None = None

            def flush(p: list) -> None:
                futs.append(pool.submit(render, p[1], p[2]))
                while len(futs) > 16:
                    futs.popleft().result()

            for k, _, t, (src_i, fr) in select_nearest(decoded(), t_tgt, tol):
                rows.append((k, src_i, t)); nm = f"{k + 1:06d}.jpg"
                if pend is not None and pend[0] == src_i:  # same source frame serves consecutive k (proc_fps > source fps)
                    pend[2].append(nm); continue
                if pend is not None:
                    flush(pend)
                pend = [src_i, fr, [nm]]
            if pend is not None:
                flush(pend)
            while futs:
                futs.popleft().result()
        codec = vs.codec_context.name; sar = vs.sample_aspect_ratio
    ks = np.array([r[0] for r in rows]); src_frame = np.array([r[1] for r in rows], np.int64); t_src = np.array([r[2] for r in rows])
    n = len(t_ref)
    assert len(rows) == n and np.array_equal(ks, np.arange(n)), f"{s.name}: {len(rows)} frames selected, expected {n}"
    files = [f"{k + 1:06d}.jpg" for k in range(n)]
    missing = [f for f in files if not (out / f).is_file()]
    assert not missing, f"{s.name}: {len(missing)} frames not written"
    dt = t_src - t_tgt
    idx = pd.DataFrame({"k": np.arange(n, dtype=np.int64), "file": files, "t_ref_s": t_ref, "t_src_s": t_src, "src_frame": src_frame})
    runtime.atomic_write_parquet(idx, out / "index.parquet", metadata={
        "stream": s.name, "t_src_s": "actual pts of the source frame used, stream seconds (container pts - video_start_s)",
        "t_ref_s": "nominal common-timeline time of processed frame k", "selection": "nearest source frame, ties -> earlier"})
    w, h = geo["size"]
    return {"n_frames": n, "width": w, "height": h, "scale": w / geo["disp"][0], "display_width": geo["disp"][0], "display_height": geo["disp"][1],
            "coded_width": geo["coded"][0], "coded_height": geo["coded"][1], "rotation": geo["rot"], "flip": geo["flip"], "display_matrix": geo["matrix"],
            "sample_aspect_ratio": str(sar) if sar else None,
            "codec": codec, "src_fps": fps, "time_base": str(tb), "video_start_s": origin, "offset_s": s.offset_s, "drift_ppm": s.drift_ppm,
            "offset_status": s.offset_status, "hwaccel": hw_label, "source": {"path": s.path, **_source_identity(src)},
            "n_src_decoded": last_i[0] + 1, "n_nonmonotonic_dropped": dropped[0], "dt_max_abs_ms": float(np.abs(dt).max() * 1e3),
            "dt_mean_ms": float(dt.mean() * 1e3), "decode_s": round(time.time() - t0, 2)}


def extract_frames(ep: Episode) -> None:
    """Stage "frames" (C4): exactly ``ep.n_frames()`` nearest-source frames per usable stream + index + manifest."""
    n = ep.n_frames()  # ValueError on an empty/negative window or bad proc_fps
    t_ref = ep.frame_times_ref(); assert len(t_ref) == n
    root = ep.derived / "frames"
    if root.exists():
        shutil.rmtree(root)  # run.py clears it too; a direct call must not mix old and new frames
    use = [s for s in ep.streams if ep.usable(s)]; skipped = {s.name: "unaligned (offset unknown)" for s in ep.streams if not ep.usable(s)}
    assert use, "no usable stream"
    for s in use:
        frames_dir(ep, s).mkdir(parents=True)
    workers = min(len(use), 4); threads = max(2, (os.cpu_count() or 4) // workers); stop = threading.Event()

    def one(s: Stream) -> dict:
        try:
            return _extract_stream(ep, s, frames_dir(ep, s), t_ref, threads, stop)
        except BaseException:
            stop.set(); raise

    pool = ThreadPoolExecutor(workers); futs: dict = {}  # streams in parallel (decoders release the GIL)
    try:
        futs = {s.name: pool.submit(one, s) for s in use}
        errs = [f.exception() for f in futs.values()]
    except BaseException:  # cancelled (SIGTERM -> SystemExit, Ctrl-C): stop the decoders before re-raising, so no worker
        stop.set()          # is still writing JPEGs while run.py clears derived/frames
        _drain(pool, futs.values(), timeout=10.0)
        raise
    pool.shutdown(wait=True)
    first = next((e for e in errs if e is not None and "cancelled (another stream" not in str(e)), next((e for e in errs if e), None))
    if first is not None:
        raise first
    entries = {k: f.result() for k, f in futs.items()}
    runtime.atomic_write_json(root / "manifest.json", {
        "version": FRAMES_VERSION, "n_frames": n, "proc_fps": ep.proc_fps, "proc_size": ep.proc_size, "common_start_s": ep.common_start_s,
        "common_end_s": ep.common_end_s, "reference": ep.reference, "jpeg_quality": JPEG_QUALITY,
        "selection": "frame k = source frame nearest to Episode.stream_time(s, common_start_s + k / proc_fps); ties -> earlier; display rotation applied",
        "streams": entries, "skipped": skipped})
    _FRAME_CACHE.clear()
    worst = max(e["dt_max_abs_ms"] for e in entries.values())
    notes = [f"{k}: {e['n_nonmonotonic_dropped']} non-monotonic pts dropped" for k, e in entries.items() if e["n_nonmonotonic_dropped"]]
    notes += [f"WARNING {k}: nearest source frame up to {e['dt_max_abs_ms']:.0f} ms from the grid (gaps / variable frame rate; see index t_src_s)"
              for k, e in entries.items() if e["dt_max_abs_ms"] > 500 / ep.proc_fps]
    ep.set_status("frames", "done", f"{n} frames x {len(entries)} streams at {ep.proc_fps:g} fps (nearest source frame, max |dt| {worst:.1f} ms; "
                  + ", ".join(f"{k} {e['width']}x{e['height']}" for k, e in entries.items()) + ")"
                  + (f"; skipped {', '.join(f'{k} ({v})' for k, v in skipped.items())}" if skipped else "") + (f"; {'; '.join(notes)}" if notes else ""))


def _drain(pool: ThreadPoolExecutor, futs: Iterable[Any], timeout: float) -> None:
    """Cancel queued work and wait (<= ``timeout`` s) for running workers, which check the stop event every frame.
    A repeated cancel signal during the wait is absorbed (the first one is re-raised by the caller)."""
    from concurrent.futures import wait
    pool.shutdown(wait=False, cancel_futures=True); futs = list(futs); end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if not wait(futs, timeout=max(0.0, end - time.monotonic())).not_done:
                return
        except BaseException:  # noqa: BLE001, S112 - second SIGTERM/Ctrl-C while draining: keep draining
            continue


_FRAME_CACHE: dict[tuple, Any] = {}


def frame_manifest(ep: Episode) -> dict | None:
    """derived/frames/manifest.json (None for episodes extracted by the old code)."""
    import json
    p = ep.derived / "frames" / "manifest.json"
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    key = ("manifest", str(p), st.st_mtime_ns, st.st_size)
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = p.read_text()
    return json.loads(_FRAME_CACHE[key])  # a fresh dict per call (callers may modify it)


def _index_entry(ep: Episode, s: Stream | str) -> tuple[Any, list[Path]]:
    """(index DataFrame, frame paths), cached on index.parquet's (mtime, size) or, for old episodes, the dir mtime."""
    import pandas as pd
    d = frames_dir(ep, s); p = d / "index.parquet"
    try:
        st = p.stat(); key: tuple = ("index", str(p), st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        st = None; key = ("legacy", str(d), d.stat().st_mtime_ns if d.exists() else 0)
    if key not in _FRAME_CACHE:
        if st is not None:
            df = pd.read_parquet(p)
            assert list(df.columns[:5]) == ["k", "file", "t_ref_s", "t_src_s", "src_frame"] and np.array_equal(df["k"], np.arange(len(df))), p
        else:
            files = sorted(f.name for f in d.glob("*.jpg")) if d.exists() else []
            df = pd.DataFrame({"k": np.arange(len(files), dtype=np.int64), "file": files, "t_ref_s": ep.common_start_s + np.arange(len(files)) / ep.proc_fps,
                               "t_src_s": np.full(len(files), np.nan), "src_frame": np.full(len(files), -1, np.int64)})
        if len(_FRAME_CACHE) > 256:
            _FRAME_CACHE.clear()
        _FRAME_CACHE[key] = (df, [d / f for f in df["file"]])
    return _FRAME_CACHE[key]


def frame_index(ep: Episode, s: Stream | str) -> Any:
    """pandas DataFrame (k, file, t_ref_s, t_src_s, src_frame) of stream ``s``'s extracted frames (C4).

    Episodes extracted by the old code have no index.parquet: the JPEGs on disk are listed, t_ref_s is nominal and
    t_src_s = NaN, src_frame = -1 (their true source times are unknown; they lag nominal time by ~33 ms)."""
    return _index_entry(ep, s)[0].copy()


def frame_paths(ep: Episode, s: Stream | str) -> list[Path]:
    """Paths of stream ``s``'s extracted frames in k order (cached on index.parquet's mtime; no directory glob)."""
    return list(_index_entry(ep, s)[1])


_frame_list = frame_paths  # old name, still imported by world/qc/depth_mono/scene_scan


def frame_size(ep: Episode, s: Stream | str) -> tuple[int, int]:
    """(width, height) of stream ``s``'s extracted frames, px."""
    name = s if isinstance(s, str) else s.name; m = frame_manifest(ep)
    if m and name in m.get("streams", {}):
        e = m["streams"][name]; return int(e["width"]), int(e["height"])
    from PIL import Image
    paths = frame_paths(ep, name)
    if not paths:
        raise FileNotFoundError(f"no extracted frames for {name}")
    with Image.open(paths[0]) as im:
        return im.size


def frame_scale(ep: Episode, s: Stream | str) -> float:
    """Extracted width / DISPLAY width of the source video (multiply native-resolution pixel quantities by this)."""
    s = ep.stream(s) if isinstance(s, str) else s; m = frame_manifest(ep)
    if m and s.name in m.get("streams", {}):
        return float(m["streams"][s.name]["scale"])
    if not s.width:
        raise ValueError(f"{s.name}: display width unknown (run probe)")
    return frame_size(ep, s)[0] / s.width


def _require_frames(ep: Episode, s: Stream) -> list[Path]:
    paths = frame_paths(ep, s); n = ep.n_frames()
    if len(paths) != n:
        raise RuntimeError(f"{s.name}: {len(paths)} extracted frames but the common grid has {n} (frames are stale: re-run frames)")
    return paths


def _iter_images(paths: list[Path], rgb: bool, workers: int = 3, ahead: int = 12) -> Iterator[np.ndarray]:
    """Decode JPEGs in a small thread pool ahead of the consumer (cv2 releases the GIL). BGR unless ``rgb``."""
    import cv2

    def load(p: Path) -> np.ndarray:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"cannot read frame {p.name}")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if rgb else im

    with ThreadPoolExecutor(workers) as pool:
        it = iter(paths); futs = deque(pool.submit(load, p) for p in itertools.islice(it, ahead))
        while futs:
            im = futs.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                futs.append(pool.submit(load, nxt))
            yield im


# ============================================================================================== models

def _model_file(name: str) -> Path:
    p = runtime.models_dir() / name
    if not p.is_file():
        raise FileNotFoundError(f"model weights {name} not found in the models directory (run `python scripts/playground.py fetch-models`, "
                                "or point PLAYGROUND_MODELS at a directory that has them)")
    return p


def _yolo() -> Any:
    """ultralytics.YOLO with run-time pip installs, downloads and telemetry disabled."""
    os.environ["YOLO_AUTOINSTALL"] = "False"; os.environ["YOLO_OFFLINE"] = "True"  # read by ultralytics at import
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is not installed (pip install 'ultralytics>=8.2'; model weights via scripts/playground.py fetch-models)") from e
    import ultralytics.utils as uu
    uu.AUTOINSTALL = False  # in case ultralytics was imported earlier in this process
    try:
        import ultralytics.utils.checks as uc
        uc.AUTOINSTALL = False
    except ImportError:  # pragma: no cover - layout differs between ultralytics versions
        pass
    return YOLO


def _fp16(device: str) -> dict[str, Any]:
    """fp16 inference arguments, CUDA only (ultralytics >= 8.4 spells it quantize=16; older versions half=True)."""
    if not device.startswith("cuda"):
        return {}
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
    except ImportError:
        pass
    return {"half": True}


def _predict(model: Any, paths: list[Path], device: str, **kw: Any) -> Iterator[Any]:
    """One ultralytics Results per frame, frames decoded ahead and run ``batch`` at a time (fp16 on CUDA only)."""
    batch = 32 if device.startswith("cuda") else 8; kw = {**kw, **_fp16(device)}; buf: list[np.ndarray] = []
    for im in _iter_images(paths, rgb=False):
        buf.append(im)
        if len(buf) == batch:
            yield from model.predict(buf, device=device, batch=batch, stream=True, verbose=False, **kw); buf = []
    if buf:
        yield from model.predict(buf, device=device, batch=len(buf), stream=True, verbose=False, **kw)


def _mediapipe() -> tuple[Any, Any, Any]:
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError as e:
        raise RuntimeError("mediapipe is not installed (pip install mediapipe==0.10.21; 1.0.x crashes on macOS Metal)") from e
    return mp, mpp, vision


# ============================================================================================== body2d

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a[m,4] and b[n,4] (x0, y0, x1, y1)."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def dedupe_people(boxes: np.ndarray, conf: np.ndarray, kpts: np.ndarray | None = None, iou_max: float = 0.8, contain_min: float = 0.85,
                  kp_conf: float = 0.3, frag_conf: float = 0.65) -> np.ndarray:
    """Indices (confidence order) of detections that are not duplicates of a more confident one: IoU > ``iou_max``, or
    the box lies mostly (> ``contain_min`` of its area) inside a kept box AND either their commonly visible keypoints
    (>= 3) coincide (median distance < 15 % of the smaller box's size: one person detected twice, e.g. a full and a
    truncated box) or it is a weak faceless fragment (conf < ``frag_conf``, > 90 % inside, < half the area, < 2 face
    keypoints: e.g. a forearm detected as a person). Nested boxes of two people (a weak one with a visible face
    included) are both kept; without ``kpts`` only the IoU rule applies."""
    order = np.argsort(-conf, kind="stable"); keep: list[int] = []
    for i in order:
        dup = False
        for j in keep:
            b1, b2 = boxes[i], boxes[j]
            iw = max(0.0, min(b1[2], b2[2]) - max(b1[0], b2[0])); ih = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]); a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]); inter = iw * ih
            if inter / (a1 + a2 - inter + 1e-9) > iou_max:
                dup = True
            elif kpts is not None and inter / (min(a1, a2) + 1e-9) > contain_min:
                ok = (kpts[i, :, 2] > kp_conf) & (kpts[j, :, 2] > kp_conf)
                if ok.sum() >= 3:
                    d = np.median(np.linalg.norm(kpts[i, ok, :2] - kpts[j, ok, :2], axis=1))
                    dup = d < 0.15 * np.sqrt(min(a1, a2))
                dup = dup or (conf[i] < frag_conf and inter / (a1 + 1e-9) > 0.9 and a1 < 0.5 * a2 and (kpts[i, :5, 2] > 0.5).sum() < 2)
            if dup:
                break
        if not dup:
            keep.append(int(i))
    return np.array(keep, dtype=int)


class SlotTracker:
    """Keeps up to ``n_slots`` people in stable slots over the frames of ONE stream (call ``update`` for EVERY frame).

    Per frame: duplicates are merged (``dedupe_people``) and only the ``n_slots`` most confident people kept. Tracks
    seen in the previous frame are matched first, then older ones (so a stale box never steals a live person), each
    level by Hungarian assignment on (1 - IoU) + 0.5 * centre distance / box diagonal against a constant-velocity
    prediction, gated by IoU >= ``min_iou`` or centre distance <= ``max_center`` diagonals. Unmatched detections
    open a track in a free slot, else replace the longest-unseen track; tracks unseen for more than ``max_miss``
    frames free their slot."""

    def __init__(self, n_slots: int = MAX_PEOPLE, max_miss: int = 15, min_iou: float = 0.1, max_center: float = 0.5) -> None:
        self.n, self.max_miss, self.min_iou, self.max_center = n_slots, max_miss, min_iou, max_center
        self.tracks: dict[int, dict] = {}  # slot -> {box, vel, miss}

    @staticmethod
    def _pred(t: dict) -> np.ndarray:
        return t["box"] + np.r_[t["vel"], t["vel"]] * (t["miss"] + 1)

    def update(self, boxes: np.ndarray, conf: np.ndarray, kpts: np.ndarray | None = None) -> np.ndarray:
        """boxes[m,4] (x0,y0,x1,y1 px), conf[m], kpts[m,K,3] or None -> slot per detection (-1 = dropped)."""
        from scipy.optimize import linear_sum_assignment
        boxes = np.asarray(boxes, float).reshape(-1, 4); conf = np.asarray(conf, float).reshape(-1); m = len(boxes)
        slots = np.full(m, -1, int)
        cand = dedupe_people(boxes, conf, kpts)[: self.n] if m else np.zeros(0, int)
        free = list(cand)
        for level in (lambda t: t["miss"] == 0, lambda t: t["miss"] > 0):
            ts = [k for k, t in self.tracks.items() if level(t)]
            if not ts or not free:
                continue
            P = np.array([self._pred(self.tracks[k]) for k in ts]); D = boxes[free]
            iou = _iou_matrix(P, D)
            cp = (P[:, None, :2] + P[:, None, 2:]) / 2; cd = (D[None, :, :2] + D[None, :, 2:]) / 2
            diag = np.hypot(P[:, 2] - P[:, 0], P[:, 3] - P[:, 1])[:, None] + 1e-9
            dc = np.linalg.norm(cp - cd, axis=-1) / diag
            ok = (iou >= self.min_iou) | (dc <= self.max_center)
            cost = np.where(ok, 1 - iou + 0.5 * np.minimum(dc, 2.0), 1e6)
            used = set()
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] >= 1e6:
                    continue
                k, i = ts[r], free[c]; t = self.tracks[k]
                ctr_old = (t["box"][:2] + t["box"][2:]) / 2; ctr = (boxes[i, :2] + boxes[i, 2:]) / 2
                t["vel"] = 0.5 * t["vel"] + 0.5 * (ctr - ctr_old) / (t["miss"] + 1)
                t["box"] = boxes[i].copy(); t["miss"] = -1; slots[i] = k; used.add(i)
            free = [i for i in free if i not in used]
        for t in self.tracks.values():
            t["miss"] += 1  # matched tracks go back to 0
        for i in free:  # new people, most confident first
            empty = [k for k in range(self.n) if k not in self.tracks]
            if empty:
                k = empty[0]
            else:
                stale = [k for k, t in self.tracks.items() if t["miss"] > 0]
                if not stale:
                    continue
                k = max(stale, key=lambda q: (self.tracks[q]["miss"], -q))
            self.tracks[k] = {"box": boxes[i].copy(), "vel": np.zeros(2), "miss": 0}; slots[i] = k
        for k in [k for k, t in self.tracks.items() if t["miss"] > self.max_miss]:
            del self.tracks[k]
        return slots


def body2d(ep: Episode, device: str | None = None) -> None:
    """Stage "body2d": YOLOv8n-pose on every usable stream -> body2d/<s>.npz (C5: kpts[n,4,17,3] x px, y px, conf;
    boxes[n,4,5] x0,y0,x1,y1,conf; NaN where a slot is empty; extracted-frame pixels; img_w, img_h, schema=2)."""
    YOLO = _yolo(); dev = runtime.pick_device(device or ep.device); n = ep.n_frames()
    model = YOLO(str(_model_file(POSE_WEIGHTS)))
    out_dir = ep.derived / "body2d"; out_dir.mkdir(parents=True, exist_ok=True); stats = []
    for s in [s for s in ep.streams if ep.usable(s)]:
        paths = _require_frames(ep, s); w, h = frame_size(ep, s)
        kpts = np.full((n, MAX_PEOPLE, 17, 3), np.nan, np.float32); boxes = np.full((n, MAX_PEOPLE, 5), np.nan, np.float32)
        tr = SlotTracker(MAX_PEOPLE, max_miss=max(1, round(1.5 * ep.proc_fps))); k = -1
        for k, r in enumerate(_predict(model, paths, dev, conf=0.35, imgsz=ep.proc_size)):
            assert r.orig_shape == (h, w), (r.orig_shape, w, h)
            if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
                tr.update(np.zeros((0, 4)), np.zeros(0)); continue
            bx = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); xy = r.keypoints.xy.cpu().numpy()
            kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2], np.float32)
            kp = np.concatenate([xy, kc[..., None]], -1)
            for i, slot in enumerate(tr.update(bx, cf, kp)):
                if slot >= 0:
                    kpts[k, slot] = kp[i]; boxes[k, slot, :4] = bx[i]; boxes[k, slot, 4] = cf[i]
        assert k == n - 1, f"{s.name}: {k + 1} results for {n} frames"
        runtime.atomic_savez(out_dir / f"{s.name}.npz", kpts=kpts, boxes=boxes, img_w=w, img_h=h, schema=2)
        stats.append(f"{s.name} {np.isfinite(boxes[..., 4]).sum(1).mean():.2f}")
    ep.set_status("body2d", "done", f"YOLOv8n-pose on {dev}, <= {MAX_PEOPLE} tracked people per view; mean people/frame: " + ", ".join(stats))


# ============================================================================================== hands

@dataclass(frozen=True)
class HandParams:
    """Owner/side attribution weights (log-odds / nats), fitted on CoMind Aria ego views against the dataset's MPS
    wearer-hand tracking and rounded (see ``attribute_hands``)."""
    w_dir: float = 4.0  # forearm (wrist - knuckles) pointing to below the camera: cos with the ray to (W/2, 1.5 H)
    w_y: float = 14.0  # wrist height in the image ...
    y0: float = 0.78  # ... relative to 78 % of the image height (own hands enter from the bottom)
    w_partner: float = 2.0  # a partner's YOLO arm (face visible, elbow not towards the bottom) ends at this wrist
    w_exit: float = 16.0  # wearer side: where the forearm ray meets the bottom edge, left/right of the body midline
    side_calib_min: int = 30  # confident hands needed to estimate the midline per stream (else 0.5 W)
    w_side_t: float = 2.0  # continuity of a hand's SIDE (handedness labels flicker) ...
    w_owner_t: float = 0.5  # ... and, weakly, of its OWNER (never enough to overrule clear owner evidence)
    d_temporal: float = 1.5  # a detection continues a slot's track within this many palm lengths
    max_gap_s: float = 0.3  # a slot's track is remembered this long
    dedupe: float = 0.3  # detections whose landmarks coincide within this many palm lengths are one hand
    p_side_clip: float = 0.02
    solo_prior: float = 8.0  # owner log-odds added when the wearer is alone: no other person declared, none ever seen
    solo_max_s: float = 1.0  # ... i.e. a face-visible person persists (>= 0.3 s runs) for less than this in total


HAND_PARAMS = HandParams()


def _hand_geometry(lm: np.ndarray, w: int, h: int, partner_arms: list[tuple[np.ndarray, np.ndarray]], P: HandParams) -> tuple[float, float, float]:
    """(log-odds that hand lm[21,2] px is the wearer's, palm length px, forearm exit x / W).

    The forearm ray starts at the wrist and points away from the knuckles; its exit x is where it meets the bottom
    edge line (clipped to [-1, 2] W; a ray pointing upwards exits at -1 or 2 by its horizontal direction)."""
    wrist = lm[0]; knuckles = lm[[5, 9, 13, 17]].mean(0); L = float(np.linalg.norm(knuckles - wrist)) + 1e-6
    u = (wrist - knuckles) / L; v = np.array([w / 2, 1.5 * h]) - wrist; v = v / (np.linalg.norm(v) + 1e-9)
    s = P.w_dir * float(u @ v) + P.w_y * float(np.clip(wrist[1] / h - P.y0, -0.3, 0.2))  # bounded: fitted range only
    for q, e in partner_arms:  # q = partner wrist, e = partner elbow (px)
        if np.linalg.norm(q - wrist) < L:
            d = e - q; vb = np.array([w / 2, 1.5 * h]) - q
            if (d / (np.linalg.norm(d) + 1e-9)) @ (vb / (np.linalg.norm(vb) + 1e-9)) < -0.2:
                s -= P.w_partner; break
    ex = (wrist[0] + (h - wrist[1]) / u[1] * u[0]) / w if u[1] > 1e-3 else (2.0 if u[0] > 0 else -1.0)
    return s, L, float(np.clip(ex, -1.0, 2.0))


def partner_arms_from_body2d(kpts: np.ndarray, conf: float = 0.3, face_conf: float = 0.5) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Per frame, (wrist, elbow) px pairs of body2d people in an EGO view that show >= 2 face keypoints (the wearer's
    own arms, often detected as a faceless 'person' at the bottom of the frame, are excluded)."""
    out = []
    for fr in kpts:
        arms = []
        for p in fr:
            if not np.isfinite(p[0, 0]) or (p[:5, 2] > face_conf).sum() < 2:
                continue
            for j in (9, 10):
                if p[j, 2] > conf and p[j - 2, 2] > conf:
                    arms.append((p[j, :2].astype(float), p[j - 2, :2].astype(float)))
        out.append(arms)
    return out


def declares_partner(ep: Episode, s: Stream) -> bool:
    """True when the episode names someone besides the wearer of ego stream ``s``: another ego stream (unless labelled
    with the same person: one wearer, two cameras) or any stream labelled with a different person."""
    for x in ep.streams:
        same = bool(x.person and s.person and x.person == s.person)
        if x.name != s.name and not same and (x.role == "ego" or x.person):
            return True
    return False


def partner_seen_s(kpts: np.ndarray, fps: float, run_s: float = 0.3, face_conf: float = 0.5) -> float:
    """Seconds in which a body2d person with >= 2 face keypoints is visible in an ego view, counting only runs of at
    least ``run_s`` (isolated face detections are false positives). ~0 when the wearer is alone (e.g. a single-person
    Eidon recording) and >= 1 s whenever a partner faces the wearer at some point."""
    f = (((kpts[..., :5, 2] > face_conf).sum(-1) >= 2) & np.isfinite(kpts[..., 0, 0])).any(1)
    edges = np.flatnonzero(np.diff(np.r_[0, f.astype(np.int8), 0]))
    runs = edges[1::2] - edges[::2]
    return float(runs[runs >= max(1, round(run_s * fps))].sum() / fps)


def _side_midline(exits: list[float], is_left: list[bool], P: HandParams) -> float:
    """Body midline (x / W) that best separates confidently-labelled wearer hands by forearm exit (left arms enter
    left of it); 0.5 without enough evidence. Head cameras are not always centred (Aria's RGB camera sits at the left
    temple): the midline is estimated per stream, within [0.35, 0.65]."""
    if len(exits) < P.side_calib_min:
        return 0.5
    x, lab = np.asarray(exits), np.asarray(is_left); c = np.round(np.arange(0.35, 0.6501, 0.01), 2)
    agree = ((x[None, :] < c[:, None]) == lab[None, :]).mean(1)
    best = np.flatnonzero(agree >= agree.max() - 1e-12)
    return float(c[best[np.argmin(np.abs(c[best] - 0.5))]])


def attribute_hands(dets: list[dict], img_w: int, img_h: int, fps: float, partner_arms: list[list] | None = None,
                    P: HandParams = HAND_PARAMS, prior: float = 0.0) -> dict[str, np.ndarray]:
    """Assign every hand detection of one ego stream to one of 4 slots: wearer L/R, partner L/R.

    dets[k] = {"lm2d": [m,21,2] px, "lm3d": [m,21,3] m (MediaPipe world landmarks), "p_left": [m]} (P(anatomical
    left hand) from MediaPipe handedness; non-mirrored ego video needs no swap). Per frame, (1) owners are assigned
    jointly (Hungarian, <= 2 hands per owner, no hand dropped unless > 4 remain after merging duplicates) on
    -log p_wearer | -log(1 - p_wearer) minus a weak continuity bonus (w_owner_t) for staying with the owner of the
    nearby hand of the previous frames; (2) within each owner, left/right is assigned jointly on -log P(side) minus a
    strong continuity bonus (w_side_t) for keeping the side of the nearby previous hand.
    p_wearer = sigmoid(w_dir * cos(forearm, ray to below the camera) + w_y * (wrist_y / H - y0) - w_partner * [a
    partner's YOLO arm ends at this wrist]). Wearer side: logit P(left) = MediaPipe handedness logit + w_exit *
    (midline - forearm exit x / W), the midline estimated per stream (``_side_midline``); partner side: handedness.
    ``prior`` (log-odds) is added to every owner logit: hands() passes solo_prior when no partner is ever seen.

    Validation (CoMind clip, Aria MPS wearer-hand ground truth, 466 frames x 2 ego views): the geometric owner cues
    separate wearer from partner hands with AUC 0.93-0.99; the forearm exit gives the wearer's side in 93 % of
    cases alone vs 65-79 % for MediaPipe handedness. YOLO wrists alone were unreliable in ego views (the wearer's own
    arms get merged into the partner's person box), hence only the gated arm cue.

    Returns C5 v2 arrays: lm2d/lm3d/score/present (wearer, slot 0 = left), partner_* (same), n_detected, plus
    p_wearer[n,2] / partner_p_wearer[n,2] (owner probability of the stored hand), n_merged (duplicates merged),
    side_midline (x / W)."""
    from scipy.optimize import linear_sum_assignment
    n = len(dets); A: dict = {}
    for pre in ("", "partner_"):
        A[pre + "lm2d"] = np.full((n, 2, 21, 2), np.nan, np.float32); A[pre + "lm3d"] = np.full((n, 2, 21, 3), np.nan, np.float32)
        A[pre + "score"] = np.full((n, 2), np.nan, np.float32); A[pre + "present"] = np.zeros((n, 2), bool)
        A[pre + "p_wearer"] = np.full((n, 2), np.nan, np.float32)
    A["n_detected"] = np.zeros(n, np.int16); A["n_merged"] = np.zeros(n, np.int16)
    frames = []; cal_x: list[float] = []; cal_l: list[bool] = []
    for k, d in enumerate(dets):  # pass 1: per-detection geometry, duplicates merged, side-midline evidence
        lm2 = np.asarray(d["lm2d"], float).reshape(-1, 21, 2); lm3 = np.asarray(d["lm3d"], float).reshape(-1, 21, 3)
        pl = np.asarray(d["p_left"], float).reshape(-1); m = len(lm2); A["n_detected"][k] = m
        assert lm3.shape[0] == m and pl.shape[0] == m, f"frame {k}: inconsistent detections"
        if m == 0:
            frames.append(None); continue
        arms = partner_arms[k] if partner_arms is not None else []
        g = [(q[0] + prior, q[1], q[2]) for q in (_hand_geometry(lm2[i], img_w, img_h, arms, P) for i in range(m))]
        keep: list[int] = []  # merge duplicate detections of one hand (keep the more certain handedness)
        for i in sorted(range(m), key=lambda i: -abs(pl[i] - 0.5)):
            if all(np.median(np.linalg.norm(lm2[i] - lm2[j], axis=1)) > P.dedupe * min(g[i][1], g[j][1]) for j in keep):
                keep.append(i)
        A["n_merged"][k] = m - len(keep); keep = keep[:MAX_HANDS]
        for i in keep:
            if g[i][0] > np.log(0.7 / 0.3) and abs(pl[i] - 0.5) > 0.4:
                cal_x.append(g[i][2]); cal_l.append(bool(pl[i] > 0.5))
        frames.append((lm2, lm3, pl, g, keep))
    mid = _side_midline(cal_x, cal_l, P); A["side_midline"] = np.float32(mid); A["owner_prior"] = np.float32(prior)
    gap = max(1, round(P.max_gap_s * fps)); tracks: dict[int, tuple[np.ndarray, float, int]] = {}  # slot -> (wrist, L, k last seen)
    for k, f in enumerate(frames):  # pass 2: owner assignment, then left/right within each owner
        if f is None:
            continue
        lm2, lm3, pl, g, keep = f
        logit = np.array([g[i][0] for i in keep]); L = [g[i][1] for i in keep]
        pw = 1 / (1 + np.exp(-logit)); pwc = np.clip(pw, 1e-3, 1 - 1e-3)
        hand = np.clip(pl[keep], P.p_side_clip, 1 - P.p_side_clip)
        wl = 1 / (1 + np.exp(-(np.log(hand / (1 - hand)) + P.w_exit * (mid - np.array([g[i][2] for i in keep])))))
        wl = np.clip(wl, P.p_side_clip, 1 - P.p_side_clip)
        near = np.zeros((len(keep), 4))  # closeness (0..1) of each detection to each slot's previous hand
        for j, t in tracks.items():
            if k - t[2] <= gap:
                near[:, j] = np.clip(1 - np.array([np.linalg.norm(lm2[i, 0] - t[0]) / max(L[r], t[1]) for r, i in enumerate(keep)]) / P.d_temporal, 0, 1)
        own = np.stack([-np.log(pwc), -np.log(1 - pwc)], 1) - P.w_owner_t * np.stack([near[:, :2].max(1), near[:, 2:].max(1)], 1)
        rows, cols = linear_sum_assignment(np.repeat(own, 2, axis=1))  # <= 2 hands per owner
        for grp in (0, 1):
            mine = [r for r, c in zip(rows, cols) if c // 2 == grp]
            if not mine:
                continue
            pls = (wl if grp == 0 else hand)[mine]  # wearer: handedness + forearm-exit prior; partner: handedness
            side_cost = np.stack([-np.log(pls), -np.log(1 - pls)], 1) - P.w_side_t * near[mine][:, 2 * grp:2 * grp + 2]
            for r_, sl in zip(*linear_sum_assignment(side_cost)):
                r = mine[r_]; i = keep[r]; pre = "" if grp == 0 else "partner_"
                A[pre + "lm2d"][k, sl] = lm2[i]; A[pre + "lm3d"][k, sl] = lm3[i] - lm3[i, :1]
                A[pre + "score"][k, sl] = pl[i] if sl == 0 else 1 - pl[i]; A[pre + "present"][k, sl] = True
                A[pre + "p_wearer"][k, sl] = pw[r]; tracks[2 * grp + sl] = (lm2[i, 0].copy(), L[r], k)
    return A


def hands(ep: Episode) -> None:
    """Stage "hands" (C5 v2): MediaPipe HandLandmarker on usable ego views, wearer/partner + left/right attribution.

    hands/<s>.npz: lm2d[n,2,21,2] (px), lm3d[n,2,21,3] (m, wrist-relative), score[n,2] (handedness probability of the
    slot's side; NaN when absent), present[n,2], the same with the partner_ prefix, n_detected[n], p_wearer /
    partner_p_wearer, n_merged, side_midline, owner_prior, partner_seen_s, partner_declared, img_w, img_h, wearer,
    partner, lm3d_origin="wrist", schema=2. The owner model's image-position cues were fitted on Aria glasses; a strong
    wearer prior is used only when the wearer is alone: the episode declares no other person (``declares_partner``)
    AND body2d never shows a face in this view (``partner_seen_s`` < 1 s)."""
    mp, mpp, vision = _mediapipe(); n = ep.n_frames(); task = _model_file(HAND_TASK)
    egos = [s for s in ep.egos() if ep.usable(s)]
    if not egos:
        ep.set_status("hands", "skipped", "no usable ego stream"); return
    out_dir = ep.derived / "hands"; out_dir.mkdir(parents=True, exist_ok=True); use_body = ep.stage_ok("body2d"); stats = []
    opts = {"num_hands": MAX_HANDS, "running_mode": vision.RunningMode.VIDEO, "min_hand_detection_confidence": 0.4,
            "min_hand_presence_confidence": 0.5, "min_tracking_confidence": 0.4}
    for s in egos:
        paths = _require_frames(ep, s); w, h = frame_size(ep, s); dets = []
        with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(task)), **opts)) as hl:
            for k, img in enumerate(_iter_images(paths, rgb=True)):
                assert img.shape[:2] == (h, w), (img.shape, w, h)
                r = hl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), round(k * 1000 / ep.proc_fps))
                m = len(r.hand_landmarks)
                dets.append({"lm2d": np.array([[(p.x * w, p.y * h) for p in lm] for lm in r.hand_landmarks], float).reshape(m, 21, 2),
                             "lm3d": np.array([[(p.x, p.y, p.z) for p in lm] for lm in r.hand_world_landmarks], float).reshape(m, 21, 3),
                             "p_left": np.array([c[0].score if c[0].category_name == "Left" else 1 - c[0].score for c in r.handedness], float)})
        assert len(dets) == n
        arms = None; prior = 0.0; seen = None; declared = declares_partner(ep, s); bz = ep.derived / "body2d" / f"{s.name}.npz"
        if use_body and bz.exists():
            kp = runtime.load_npz(bz)["kpts"]; assert kp.shape[0] == n, f"{s.name}: body2d has {kp.shape[0]} frames, expected {n}"
            arms = partner_arms_from_body2d(kp); seen = partner_seen_s(kp, ep.proc_fps)
            # wearer prior only for a wearer who is alone: nobody else declared in the episode AND nobody seen in the view
            prior = HAND_PARAMS.solo_prior if not declared and seen < HAND_PARAMS.solo_max_s else 0.0
        A = attribute_hands(dets, w, h, ep.proc_fps, arms, prior=prior)
        others = [x.person for x in ep.egos() if x.name != s.name and x.person and x.person != s.person]
        runtime.atomic_savez(out_dir / f"{s.name}.npz", **A, img_w=w, img_h=h, wearer=s.person or "", partner=others[0] if len(others) == 1 else "",
                             partner_seen_s=np.float32(np.nan if seen is None else seen), partner_declared=np.bool_(declared),
                             lm3d_origin="wrist", schema=2)
        stats.append(f"{s.name}: {int(A['n_detected'].sum())} detected, wearer {int(A['present'].sum())}, partner {int(A['partner_present'].sum())}"
                     + (" (alone: no other person declared or seen, wearer prior)" if prior else ""))
    ep.set_status("hands", "done", "MediaPipe hands (VIDEO, <= 4) with wearer/partner + L/R attribution" + ("" if use_body else " (no body2d: geometric cues only)")
                  + "; " + "; ".join(stats))


# ============================================================================================== objects

def world_vocab_weights() -> str:
    """File name (in the models directory) of yolov8s-worldv2 with VOCAB's text embeddings baked in."""
    return f"yolov8s-worldv2-vocab-{hashlib.sha256(chr(10).join(WORLD_VOCAB).encode()).hexdigest()[:10]}.pt"


def bake_world_vocab(models: Path | None = None) -> Path:
    """Write <models>/world_vocab_weights(): YOLO-World with VOCAB set and the CLIP text model dropped, so ``objects``
    needs neither CLIP nor its weights at run time. Needs CLIP once (``scripts/playground.py fetch-models``)."""
    YOLO = _yolo(); d = Path(models) if models else runtime.models_dir()
    src = d / WORLD_WEIGHTS
    if not src.is_file():
        raise FileNotFoundError(f"{WORLD_WEIGHTS} not found in the models directory (download it first)")
    m = YOLO(str(src)); m.set_classes(list(WORLD_VOCAB))
    if hasattr(m.model, "clip_model"):
        del m.model.clip_model  # 300 MB of CLIP weights that would otherwise be pickled into the checkpoint
    out = d / world_vocab_weights(); tmp = d / f".{out.name}.tmp"
    m.save(str(tmp)); os.replace(tmp, out)
    return out


def _world_model(YOLO: Any) -> tuple[Any, str]:
    """(YOLO-World with VOCAB, how). Prefers the baked checkpoint; otherwise set_classes, which needs the CLIP package
    and its ViT-B/32 weights already on disk: ultralytics would pip-install CLIP (disabled) and clip.load would
    download 338 MB at run time, so both fail with a clear message instead."""
    baked = runtime.models_dir() / world_vocab_weights()
    if baked.is_file():
        model = YOLO(str(baked)); names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if names != WORLD_VOCAB:
            raise RuntimeError(f"{baked.name} holds a different vocabulary; re-run scripts/playground.py fetch-models")
        return model, "baked vocabulary"
    model = YOLO(str(_model_file(WORLD_WEIGHTS)))
    hint = "; or run `python scripts/playground.py fetch-models`, which bakes the vocabulary into " + world_vocab_weights() + " (no CLIP at run time)"
    try:
        import clip  # noqa: F401
    except ImportError as e:
        raise RuntimeError("objects: YOLO-World needs the CLIP package to embed the vocabulary (`pip install "
                           "git+https://github.com/ultralytics/CLIP.git`; run-time auto-install is disabled)" + hint) from e
    from ultralytics.utils import WEIGHTS_DIR
    if not (Path(WEIGHTS_DIR) / "clip" / "ViT-B-32.pt").is_file():
        raise FileNotFoundError("objects: CLIP ViT-B/32 weights are not in ultralytics' weights directory (no downloads at run time)" + hint)
    model.set_classes(list(WORLD_VOCAB))
    return model, "CLIP set_classes"


def objects(ep: Episode, device: str | None = None) -> None:
    """Stage "objects": YOLO-World on usable ego views, class-agnostic NMS -> objects/<s>.npz (boxes[n,12,5]
    x0,y0,x1,y1,conf in extracted-frame px, most confident first; names[n,12])."""
    YOLO = _yolo(); dev = runtime.pick_device(device or ep.device); n = ep.n_frames()
    egos = [s for s in ep.egos() if ep.usable(s)]
    if not egos:
        ep.set_status("objects", "skipped", "no usable ego stream"); return
    model, how = _world_model(YOLO); out_dir = ep.derived / "objects"; out_dir.mkdir(parents=True, exist_ok=True); stats = []
    for s in egos:
        paths = _require_frames(ep, s)
        boxes = np.full((n, MAX_OBJECTS, 5), np.nan, np.float32); names = np.full((n, MAX_OBJECTS), "", dtype="<U24"); k = -1
        for k, r in enumerate(_predict(model, paths, dev, conf=0.3, imgsz=ep.proc_size, agnostic_nms=True)):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            cf = r.boxes.conf.cpu().numpy(); xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
            for i, idx in enumerate(np.argsort(-cf, kind="stable")[:MAX_OBJECTS]):
                boxes[k, i, :4] = xy[idx]; boxes[k, i, 4] = cf[idx]; names[k, i] = str(r.names[cl[idx]])[:24]
        assert k == n - 1, f"{s.name}: {k + 1} results for {n} frames"
        runtime.atomic_savez(out_dir / f"{s.name}.npz", boxes=boxes, names=names)
        stats.append(f"{s.name} {np.isfinite(boxes[..., 4]).sum(1).mean():.1f}/frame")
    ep.set_status("objects", "done", f"YOLO-World ({len(WORLD_VOCAB)} classes, {how}, class-agnostic NMS) on {dev}; " + ", ".join(stats))


# ============================================================================================== body3d

def body3d(ep: Episode) -> None:
    """Stage "body3d": monocular MediaPipe world landmarks (33 joints, m, hip-centred) on the usable EXO view with the
    largest median person box. body3d/body3d.npz: world[n,2,33,3], img2d[n,2,33,3] (x px, y px, visibility), vis[n,2,33],
    stream/view (the view used), schema=2; person slots kept by SlotTracker on the landmark boxes. SKIPPED (no file)
    without an exo view: on a head camera it invents skeletons of the partner/wearer far outside the frame."""
    exos = [s for s in ep.exos() if ep.usable(s)]
    if not exos:
        ep.set_status("body3d", "skipped", "no usable exo (fixed) view; monocular body3d is not estimated from head cameras"); return
    if not ep.stage_ok("body2d"):
        raise RuntimeError("body3d needs body2d (not done)")
    best, best_area = None, 0.0
    for s in exos:
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if not z.exists():
            continue
        b = runtime.load_npz(z)["boxes"]; area = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        med = float(np.nanmedian(area)) if np.isfinite(area).any() else float("nan")
        if np.isfinite(med) and med > best_area:
            best, best_area = s, med
    if best is None:
        ep.set_status("body3d", "skipped", "no person detected in any exo view"); return
    mp, mpp, vision = _mediapipe(); n = ep.n_frames(); paths = _require_frames(ep, best); w, h = frame_size(ep, best)
    world = np.full((n, 2, 33, 3), np.nan, np.float32); img2d = np.full((n, 2, 33, 3), np.nan, np.float32); vis = np.full((n, 2, 33), np.nan, np.float32)
    tr = SlotTracker(2, max_miss=max(1, round(1.5 * ep.proc_fps)))
    opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(_model_file(POSE_TASK))), running_mode=vision.RunningMode.VIDEO, num_poses=2)
    with vision.PoseLandmarker.create_from_options(opts) as po:
        for k, img in enumerate(_iter_images(paths, rgb=True)):
            r = po.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img), round(k * 1000 / ep.proc_fps))
            m = min(2, len(r.pose_landmarks))
            if m == 0:
                tr.update(np.zeros((0, 4)), np.zeros(0)); continue
            i2 = np.array([[(p.x * w, p.y * h, p.visibility) for p in r.pose_landmarks[q]] for q in range(m)], float)
            wl = np.array([[(p.x, p.y, p.z) for p in r.pose_world_landmarks[q]] for q in range(m)], float)
            bx = np.array([[*np.nanmin(q[:, :2], 0), *np.nanmax(q[:, :2], 0)] for q in i2]); cf = np.nanmean(i2[..., 2], 1)
            for q, slot in enumerate(tr.update(bx, cf, None)):
                if slot >= 0:
                    world[k, slot] = wl[q]; img2d[k, slot] = i2[q]; vis[k, slot] = i2[q, :, 2]
    runtime.atomic_savez(ep.derived / "body3d" / "body3d.npz", world=world, img2d=img2d, vis=vis, stream=best.name, view=best.name, schema=2)
    ep.set_status("body3d", "done", f"monocular MediaPipe world landmarks on {best.name} (not triangulated); "
                  f"{np.isfinite(world[:, :, 0, 0]).any(1).mean():.0%} of frames with a person")
