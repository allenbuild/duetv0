"""Playground perception: frames (C4) nearest-frame timing, rotation, counts, index/manifest; the body2d slot
tracker; wearer/partner + L/R hand attribution; YOLO/MediaPipe stage plumbing with fake model modules.

Synthetic clips encode their own frame index as 12 binary blocks, so every extracted JPEG can be decoded back to the
source frame it came from. Nothing under data/ is touched: episodes are built in tmp_path."""
from __future__ import annotations

import importlib.util
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from duet.playground import perception as P
from duet.playground.episode import Episode, probe

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAVE_AV = importlib.util.find_spec("av") is not None
needs_video = pytest.mark.skipif(not (HAVE_FFMPEG and HAVE_AV), reason="needs ffmpeg/ffprobe on PATH and PyAV")


# ------------------------------------------------------------------------------------------ synthetic video

def _encoder() -> list[str]:
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"] if " libx264 " in enc else ["-c:v", "mpeg4", "-q:v", "2"]


def _clip(path: Path, n: int, fps: float, W: int = 160, H: int = 120, extra: tuple = (), pattern=None) -> Path:
    """n frames at ``fps``; frame i shows i in binary (12 blocks, 3 rows x 4 cols) unless ``pattern(i)`` is given.
    GOP 12 with 2 B-frames, so decode order != presentation order."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           *_encoder(), "-g", "12", "-bf", "2", *extra, str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        im = np.zeros((H, W), np.uint8)
        if pattern is None:
            for b in range(12):
                if (i >> b) & 1:
                    r, c = divmod(b, 4); im[r * H // 3:(r + 1) * H // 3, c * W // 4:(c + 1) * W // 4] = 255
        else:
            im = pattern(i)
        p.stdin.write(im.tobytes())
    p.stdin.close(); assert p.wait() == 0
    return path


def _code(path: Path) -> int:
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L"), float); H, W = g.shape; v = 0
    for b in range(12):
        r, c = divmod(b, 4)
        blk = g[int((r + .25) * H / 3):int((r + .75) * H / 3), int((c + .25) * W / 4):int((c + .75) * W / 4)]
        v |= int(blk.mean() > 128) << b
    return v


def _pts(path: Path) -> np.ndarray:
    """Presentation times (container seconds) of every frame, decoded independently of the code under test."""
    import av
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        return np.array([float(f.pts * vs.time_base) for f in c.decode(vs)])


def _episode(tmp: Path, videos: list, ref: str, proc_size: int = 80, proc_fps: float = 10.0) -> Episode:
    ep = Episode.create(tmp / "eps", "ep", videos, reference=ref, proc_fps=proc_fps, proc_size=proc_size)
    probe(ep)
    return ep


# ------------------------------------------------------------------------------------------ select_nearest

def test_select_nearest_ties_bounds_and_lazy_consumption():
    consumed = []

    def frames():
        for i, t in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
            consumed.append(i); yield t, f"f{i}"

    got = list(P.select_nearest(frames(), np.array([0.0, 0.05, 0.149, 0.151, 0.3]), tol=0.1))
    assert [(k, i) for k, i, _, _ in got] == [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3)]  # 0.05 is a tie -> earlier frame
    assert got[3][3] == "f2" and consumed == [0, 1, 2, 3]  # stops once the last target is served
    # targets just outside the decoded span (<= tol) map to the first/last frame; farther ones are an error
    assert [i for _, i, _, _ in P.select_nearest(iter([(0.0, 0), (0.1, 1)]), np.array([-0.03, 0.12]), tol=0.034)] == [0, 1]
    with pytest.raises(ValueError, match="precedes the first frame"):
        list(P.select_nearest(iter([(0.0, 0), (0.1, 1)]), np.array([-0.05]), tol=0.034))
    with pytest.raises(ValueError, match="past the last frame"):
        list(P.select_nearest(iter([(0.0, 0), (0.1, 1)]), np.array([0.05, 0.2]), tol=0.034))
    with pytest.raises(ValueError, match="strictly increasing"):
        list(P.select_nearest(iter([(0.0, 0), (0.0, 1), (0.2, 2)]), np.array([0.19]), tol=0.1))
    # proc_fps above the source rate: consecutive targets share a source frame
    assert [i for _, i, _, _ in P.select_nearest(iter([(0.0, 0), (0.1, 1)]), np.array([0.0, 0.04, 0.06, 0.1]), tol=0.1)] == [0, 0, 1, 1]


def test_proc_dims_long_side_even():
    assert P.proc_dims(1280, 720, 640) == (640, 360)
    assert P.proc_dims(720, 1280, 640) == (360, 640)
    assert P.proc_dims(1408, 1408, 640) == (640, 640)
    assert P.proc_dims(2704, 1520, 640) == (640, 360)  # 359.76 -> even 360 (ffmpeg scale=640:-2)
    assert P.proc_dims(4000, 3000, 640) == (640, 480)


# ------------------------------------------------------------------------------------------ frames stage (C4)

@needs_video
@pytest.mark.parametrize("fps,ext,start,offset,drift", [
    (25, ".mp4", 0.0, 0.37, 0.0),       # later stream, positive offset
    (30, ".mkv", 1.25, -0.52, 0.0),     # nonzero container start, 1 ms timebase, negative offset
    (60, ".mp4", 0.8, 0.013, 400.0),    # nonzero start, sub-frame offset, clock drift
])
def test_frames_pick_nearest_source_frame(tmp_path, fps, ext, start, offset, drift):
    a = _clip(tmp_path / "a.mp4", 150, 30)
    b = _clip(tmp_path / f"b{ext}", int(5.5 * fps), fps, extra=("-output_ts_offset", str(start)) if start else ())
    ep = _episode(tmp_path, [("a", "ego", a, "alice"), ("b", "exo", b, None)], "a")
    sb = ep.stream("b"); assert abs(sb.video_start_s - start) < 1e-6
    sb.offset_s, sb.offset_status, sb.drift_ppm = offset, "manual", drift
    ep.common_start_s, ep.common_end_s = 0.6, 3.6; ep.save()
    P.extract_frames(ep)
    n = ep.n_frames(); assert n == 30 and ep.status["frames"]["state"] == "done"
    man = json.loads((ep.derived / "frames" / "manifest.json").read_text())
    assert man["n_frames"] == n and man["proc_fps"] == 10 and (man["common_start_s"], man["common_end_s"]) == (0.6, 3.6)
    for s, src in (("a", a), ("b", b)):
        idx = P.frame_index(ep, s); paths = P.frame_paths(ep, s)
        assert len(idx) == len(paths) == n and sorted(p.name for p in P.frames_dir(ep, s).glob("*.jpg")) == [f"{k:06d}.jpg" for k in range(1, n + 1)]
        assert list(idx.columns) == ["k", "file", "t_ref_s", "t_src_s", "src_frame"] and [p.name for p in paths] == list(idx.file)
        np.testing.assert_allclose(idx.t_ref_s, 0.6 + np.arange(n) / 10, atol=1e-12)
        t_tgt = ep.stream_time(s, idx.t_ref_s.to_numpy())
        pts = _pts(src) - ep.stream(s).video_start_s  # stream time of every source frame
        want = np.array([int(np.argmin(np.abs(pts - t))) for t in t_tgt])  # nearest, ties -> first (earlier)
        assert np.array_equal(idx.src_frame, want)
        assert np.array_equal([_code(p) for p in paths], want)  # the pixels really are that source frame
        np.testing.assert_allclose(idx.t_src_s, pts[want], atol=1e-9)
        assert np.all(np.abs(idx.t_src_s - t_tgt) <= 0.5 / {"a": 30, "b": fps}[s] + 1e-3)
        e = man["streams"][s]
        assert (e["width"], e["height"]) == (80, 60) and e["scale"] == 0.5 and e["source"]["path"] == ep.stream(s).path
        assert e["offset_s"] == ep.stream(s).offset_s and e["drift_ppm"] == ep.stream(s).drift_ppm
        assert P.frame_scale(ep, s) == 0.5 and P.frame_size(ep, s) == (80, 60)


@needs_video
def test_frames_rotation_matches_ffmpeg_autorotate(tmp_path):
    def pattern(i):
        im = np.zeros((96, 160), np.uint8); im[:32, :40] = 255; im[-20:, -30:] = 120; return im  # white top-left of the STORED frame
    base = _clip(tmp_path / "base.mp4", 20, 30, W=160, H=96, pattern=pattern)
    rot = tmp_path / "rot.mp4"
    if subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(base), "-c", "copy", str(rot)], check=False).returncode:
        pytest.skip("this ffmpeg has no -display_rotation (ffmpeg >= 6.1)")
    ep = _episode(tmp_path, [("a", "ego", rot, None)], "a")
    s = ep.stream("a"); assert (s.width, s.height, s.rotation) == (96, 160, 90)  # display size (portrait)
    ep.common_start_s, ep.common_end_s = 0.0, 0.5; ep.save()
    P.extract_frames(ep)
    from PIL import Image
    got = np.asarray(Image.open(P.frame_paths(ep, "a")[0]).convert("L"), float)
    assert got.shape == (80, 48)  # long side = proc_size on the DISPLAY orientation (was 80x48 before the fix)
    cli = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(rot), "-frames:v", "1", "-vf", "scale=48:80", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    ref = np.frombuffer(cli, np.uint8).reshape(80, 48).astype(float)
    assert np.abs(got - ref).mean() < 12  # same orientation as ffmpeg's autorotate (a wrong turn differs by ~100)
    assert got[-20:, :10].mean() > 200  # stored top-left -> display bottom-left for a 90 deg CCW display matrix
    s.rotation = 0; s.width, s.height = 160, 96; ep.save()  # stale probe data must not be silently used
    with pytest.raises(ValueError, match="probe is stale"):
        P.extract_frames(ep)


@needs_video
@pytest.mark.parametrize("rot,hflip,vflip,sei", [(r, h, 0, None) for r in (0, 90, 180, 270) for h in (0, 1)]  # the 8 distinct transforms
                         + [(90, 0, 1, None), (None, 0, 0, "rotate=90"), (None, 0, 0, "rotate=90:flip=horizontal"), (None, 0, 0, "flip=vertical")])
def test_frames_display_matrix_flips_match_ffmpeg(tmp_path, rot, hflip, vflip, sei):
    def pattern(i):
        im = np.tile(np.linspace(0, 200, 96, dtype=np.uint8), (64, 1)); im[:20, :24] = 255; im[-12:, -40:] = 40; return im  # asymmetric
    base = _clip(tmp_path / "base.mp4", 6, 30, W=96, H=64, pattern=pattern)
    src = tmp_path / "t.mp4"
    if sei:  # rotation only in the H.264 display-orientation SEI (frame side data), not in the container
        if "libx264" not in " ".join(_encoder()):
            pytest.skip("needs an H.264 encoder for the SEI case")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(base), "-c", "copy", "-bsf:v", f"h264_metadata=display_orientation=insert:{sei}", str(src)]
    else:
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", str(rot)] + ["-display_hflip"] * hflip + ["-display_vflip"] * vflip + ["-i", str(base), "-c", "copy", str(src)]
    if subprocess.run(cmd, check=False).returncode:
        pytest.skip("this ffmpeg lacks -display_rotation/-display_hflip or h264_metadata")
    ref = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    import av

    from duet.playground import runtime
    with av.open(str(src)) as c:
        m = P._frame_display_matrix(next(c.decode(c.streams.video[0])))
    r, flip = runtime.display_transform(None if m is None else [int(x) for x in m])
    dw, dh = (64, 96) if r in (90, 270) else (96, 64)
    ref = np.frombuffer(ref, np.uint8).reshape(dh, dw).astype(float)  # ffmpeg's autorotated frame (display orientation)
    ep = Episode.create(tmp_path / "eps", "ep", [("a", "ego", src, None)], reference="a", proc_size=96)
    probe(ep); s = ep.stream("a")
    assert (s.width, s.height, s.rotation, s.flip) == (dw, dh, r, flip)  # probe = runtime.display_transform of the first frame (SEI too)
    ep.common_start_s, ep.common_end_s = 0.0, 0.1; ep.save()
    P.extract_frames(ep)
    from PIL import Image
    got = np.asarray(Image.open(P.frame_paths(ep, "a")[0]).convert("L"), float)
    assert got.shape == ref.shape and np.abs(got - ref).mean() < 4  # a wrong flip/turn differs by ~50
    man = P.frame_manifest(ep)["streams"]["a"]
    assert (man["rotation"], man["flip"]) == (r, flip) and man["display_matrix"] == (None if m is None else [int(x) for x in m])
    if flip:  # stale probe data (flip lost) must not be silently used
        s.flip = type(flip)(); ep.save()
        with pytest.raises(ValueError, match="probe is stale"):
            P.extract_frames(ep)


@needs_video
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX interval timers")
def test_frames_cancel_stops_decoders_promptly(tmp_path, monkeypatch):
    a = _clip(tmp_path / "a.mp4", 150, 30); b = _clip(tmp_path / "b.mp4", 150, 30)
    ep = _episode(tmp_path, [("a", "ego", a, None), ("b", "ego", b, None)], "a")
    ep.stream("b").offset_status = "manual"; ep.common_start_s, ep.common_end_s = 0.0, 4.5; ep.save()
    orig = P.select_nearest

    def slow(*args, **kw):  # ~2.2 s of work per stream
        for x in orig(*args, **kw):
            time.sleep(0.05); yield x

    def cancel(*_):  # what run.py's SIGTERM handler does in the main thread
        raise SystemExit("cancelled")

    threads0 = threading.active_count()
    monkeypatch.setattr(P, "select_nearest", slow); old = signal.signal(signal.SIGALRM, cancel); t0 = time.monotonic()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.4)
        with pytest.raises(SystemExit):
            P.extract_frames(ep)
        dt = time.monotonic() - t0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
    assert dt < 1.5, dt  # before the fix the executor join waited for both decoders (>= 2.2 s)
    jpgs = lambda: sorted((ep.derived / "frames").rglob("*.jpg"))  # noqa: E731
    n0 = len(jpgs()); time.sleep(0.4)
    assert len(jpgs()) == n0 and threading.active_count() <= threads0  # no decoder/writer left running after the raise


@needs_video
def test_frames_window_errors_and_unaligned_streams(tmp_path):
    a = _clip(tmp_path / "a.mp4", 60, 30); b = _clip(tmp_path / "b.mp4", 60, 30)
    ep = _episode(tmp_path, [("a", "ego", a, None), ("b", "ego", b, None)], "a")
    ep.common_start_s, ep.common_end_s = 1.0, 0.5; ep.save()
    with pytest.raises(ValueError, match="empty common window"):
        P.extract_frames(ep)
    assert not (ep.derived / "frames").exists()
    ep.common_start_s, ep.common_end_s = 0.0, 3.0; ep.save()  # stream b is 2 s long: window not covered
    b_ = ep.stream("b"); b_.offset_status = "manual"; ep.save()
    with pytest.raises(ValueError, match="past the last frame"):
        P.extract_frames(ep)
    b_.offset_s = 0.8; ep.common_start_s, ep.common_end_s = 0.5, 1.5; ep.save()  # b starts 0.3 s after the window: no silent clamp
    with pytest.raises(ValueError, match="precedes the first frame"):
        P.extract_frames(ep)
    b_.offset_s = 0.0; ep.common_start_s = 0.0
    b_.offset_status = "unaligned"; ep.common_end_s = 1.5; ep.save()  # unaligned: skipped, not extracted with offset 0
    P.extract_frames(ep)
    assert len(P.frame_paths(ep, "a")) == 15 and not P.frames_dir(ep, "b").exists()
    assert "skipped b (unaligned" in ep.status["frames"]["detail"]
    assert json.loads((ep.derived / "frames" / "manifest.json").read_text())["skipped"] == {"b": "unaligned (offset unknown)"}


@needs_video
def test_frames_explicit_hwaccel_same_selection(tmp_path):
    import av.codec.hwaccel as hw
    a = _clip(tmp_path / "a.mp4", 45, 30)
    ep = _episode(tmp_path, [("a", "ego", a, None)], "a")
    ep.common_start_s, ep.common_end_s = 0.2, 1.2; ep.hwaccel = "videotoolbox"; ep.save()
    P.extract_frames(ep)
    label = json.loads((ep.derived / "frames" / "manifest.json").read_text())["streams"]["a"]["hwaccel"]
    assert label == ("videotoolbox" if "videotoolbox" in hw.hwdevices_available() else "none (videotoolbox not available in this PyAV build)")
    idx = P.frame_index(ep, "a")
    assert np.array_equal(idx.src_frame, np.round(ep.stream_time("a", idx.t_ref_s.to_numpy()) * 30).astype(int))
    assert np.array_equal([_code(p) for p in P.frame_paths(ep, "a")], idx.src_frame)


def test_frame_paths_legacy_layout_and_cache(tmp_path, monkeypatch):
    from PIL import Image
    ep = Episode(name="old", root=str(tmp_path), proc_fps=10.0, common_start_s=1.67, common_end_s=2.07)
    from duet.playground.episode import Stream
    ep.streams = [Stream(name="cam", role="exo", path="streams/cam.mp4", width=1280, height=720)]
    d = P.frames_dir(ep, "cam"); d.mkdir(parents=True)
    for k in (3, 1, 4, 2):
        Image.new("RGB", (640, 360)).save(d / f"{k:06d}.jpg")
    idx = P.frame_index(ep, "cam")  # no index.parquet: old episodes still load
    assert list(idx.file) == [f"{k:06d}.jpg" for k in range(1, 5)] and idx.t_src_s.isna().all() and (idx.src_frame == -1).all()
    np.testing.assert_allclose(idx.t_ref_s, [1.67, 1.77, 1.87, 1.97])
    assert P.frame_scale(ep, "cam") == 0.5 and P.frame_size(ep, "cam") == (640, 360)
    monkeypatch.setattr(Path, "glob", lambda *a, **k: (_ for _ in ()).throw(AssertionError("glob called again")))
    assert len(P._frame_list(ep, "cam")) == 4  # cached on the directory mtime: no re-listing


# ------------------------------------------------------------------------------------------ body2d tracker

def _person(cx: float, cy: float, w: float = 60, h: float = 150, kp_shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    kp = np.zeros((17, 3)); kp[:, 0] = cx + kp_shift[0] + np.linspace(-20, 20, 17); kp[:, 1] = cy + kp_shift[1] + np.linspace(-60, 60, 17); kp[:, 2] = 0.9
    return box, kp


def _frame(people, conf=None):
    if not people:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
    b, k = zip(*people)
    return np.array(b), np.array(conf if conf is not None else [0.9] * len(people)), np.array(k)


def test_tracker_keeps_ids_through_crossing_and_occlusion():
    tr = P.SlotTracker(4, max_miss=15); ids = []
    for t in range(21):
        xa, xb = 100 + 20 * t, 500 - 20 * t
        people = [_person(xa, 200), _person(xb, 200, kp_shift=(0, 5))]
        if t == 10:  # full occlusion at the crossing: one detection only (A in front)
            people = people[:1]
        slots = tr.update(*_frame(people))
        ids.append(tuple(slots))
    assert all(s[0] == ids[0][0] for s in ids) and all(s[1] == ids[0][1] for s in ids if len(s) == 2)
    assert ids[0][0] != ids[0][1]


def test_tracker_merges_duplicate_box_keeps_nested_people():
    full, kp = _person(300, 200); trunc = np.array([270, 125, 330, 210.0])  # upper body of the same person
    boxes = np.array([full, trunc]); kps = np.array([kp, kp]); conf = np.array([0.9, 0.6])
    assert list(P.dedupe_people(boxes, conf, kps)) == [0]
    kp2 = kp.copy(); kp2[:, :2] += [0, 40]  # a different (nested, e.g. seated child) person
    assert sorted(P.dedupe_people(boxes, conf, np.array([kp, kp2]))) == [0, 1]
    tr = P.SlotTracker(4); s = tr.update(boxes, conf, kps)
    assert s[0] >= 0 and s[1] == -1 and len(tr.tracks) == 1  # no phantom slot


def test_dedupe_drops_faceless_fragment_keeps_weak_nested_person():
    big, kp_big = _person(300, 200, w=300, h=300)
    frag = np.array([330, 220, 400, 330.0]); kp_frag = np.zeros((17, 3)); kp_frag[:, 0] = np.linspace(335, 395, 17)
    kp_frag[:, 1] = np.linspace(225, 325, 17); kp_frag[:, 2] = 0.8; kp_frag[:5, 2] = 0.1  # a forearm: no face
    boxes = np.array([big, frag]); conf = np.array([0.9, 0.5])
    assert list(P.dedupe_people(boxes, conf, np.array([kp_big, kp_frag]))) == [0]
    kp_face = kp_frag.copy(); kp_face[:5, 2] = 0.9  # same box, but a face is visible: a real (partly hidden) person
    assert sorted(P.dedupe_people(boxes, conf, np.array([kp_big, kp_face]))) == [0, 1]


def test_tracker_misses_expiry_recycling_and_more_than_four_people():
    tr = P.SlotTracker(4, max_miss=5)
    s0 = tr.update(*_frame([_person(100, 200)]))[0]
    for _ in range(4):
        tr.update(*_frame([]))  # expiry counters advance on EMPTY frames too
    assert tr.update(*_frame([_person(105, 200)]))[0] == s0  # back within max_miss: same slot
    for _ in range(6):
        tr.update(*_frame([]))
    assert not tr.tracks  # expired
    # 4 slots filled, then everyone leaves and 4 new people arrive far away: stale slots are recycled
    tr = P.SlotTracker(4, max_miss=50)
    tr.update(*_frame([_person(80 + 150 * i, 200) for i in range(4)]))
    new = tr.update(*_frame([_person(80 + 150 * i, 600) for i in range(4)]))
    assert sorted(new) == [0, 1, 2, 3]
    # 6 people: the 4 most confident are kept
    tr = P.SlotTracker(4)
    conf = [0.5, 0.95, 0.4, 0.9, 0.85, 0.8]
    s = tr.update(*_frame([_person(80 + 120 * i, 200) for i in range(6)], conf))
    assert list(s >= 0) == [False, True, False, True, True, True]


def test_tracker_live_track_beats_stale_box():
    tr = P.SlotTracker(4, max_miss=15)
    sa = tr.update(*_frame([_person(300, 200)]))[0]  # person A
    sb = tr.update(*_frame([_person(300, 200, w=62), _person(600, 200)]))  # A + a phantom-ish B far away
    for x in range(300, 330, 5):  # A drifts right; the old box of A is kept by nobody else
        s = tr.update(*_frame([_person(x, 200)]))
        assert s[0] == sa
    assert sb[0] == sa


# ------------------------------------------------------------------------------------------ hands attribution

def _hand(wrist, direction, L=40.0, spread=1.0):
    """21 landmarks px: fingers extend from ``wrist`` along ``direction`` (unit), palm length L."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
    lm = np.zeros((21, 2)); lm[0] = wrist
    for f, off in zip(range(5), (-1.5, -0.75, 0.0, 0.75, 1.5)):
        for j in range(4):
            idx = 1 + 4 * f + j
            lm[idx] = np.asarray(wrist) + d * L * (0.6 + 0.35 * (j + 1) if f else 0.3 + 0.2 * j) + n * off * 8 * spread
    for i, f in zip((5, 9, 13, 17), (-0.75, 0.0, 0.75, 1.5)):  # knuckles (MCPs) one palm length from the wrist
        lm[i] = np.asarray(wrist) + d * L + n * f * 8
    return lm


def _det(hands_):
    """hands_ = [(lm2d, p_left)] -> detector output for one frame (lm3d: world landmarks offset from the wrist)."""
    if not hands_:
        return {"lm2d": np.zeros((0, 21, 2)), "lm3d": np.zeros((0, 21, 3)), "p_left": np.zeros(0)}
    lm3 = np.stack([np.c_[(h[0] - h[0][9]) / 400.0, np.full(21, 0.01)] for h in hands_])  # hand-centred like MediaPipe
    return {"lm2d": np.stack([h[0] for h in hands_]), "lm3d": lm3, "p_left": np.array([h[1] for h in hands_])}


W, H = 640, 640
WEARER_L = _hand((220, 600), (0.2, -1)); WEARER_R = _hand((420, 600), (-0.2, -1))
PARTNER = _hand((330, 330), (0.1, 1), L=28)  # partner reaching towards the camera: fingers point down, forearm up


def test_hands_two_same_label_hands_are_both_kept():
    A = P.attribute_hands([_det([(WEARER_L, 0.1), (WEARER_R, 0.05)])], W, H, 10.0)  # both labelled "Right"
    assert A["present"][0].all() and not A["partner_present"][0].any() and A["n_detected"][0] == 2
    np.testing.assert_allclose(A["lm2d"][0, 0], WEARER_L); np.testing.assert_allclose(A["lm2d"][0, 1], WEARER_R)
    assert A["score"][0, 0] == pytest.approx(0.1) and A["score"][0, 1] == pytest.approx(0.95)  # P(side of the slot)


def test_hands_partner_hand_goes_to_partner_slots():
    A = P.attribute_hands([_det([(WEARER_L, 0.9), (PARTNER, 0.8), (WEARER_R, 0.2)])], W, H, 10.0)
    np.testing.assert_allclose(A["lm2d"][0, 0], WEARER_L); np.testing.assert_allclose(A["lm2d"][0, 1], WEARER_R)
    assert A["partner_present"][0].tolist() == [True, False]; np.testing.assert_allclose(A["partner_lm2d"][0, 0], PARTNER)
    assert A["p_wearer"][0].min() > 0.9 and A["partner_p_wearer"][0, 0] < 0.1
    # lm3d is wrist-relative (landmark 0 at the origin), NaN where absent
    assert np.all(A["lm3d"][0, :, 0] == 0) and np.all(A["partner_lm3d"][0, 0, 0] == 0) and np.isnan(A["partner_lm3d"][0, 1]).all()
    assert np.isnan(A["partner_score"][0, 1]) and A["n_detected"][0] == 3


def test_hands_partner_yolo_arm_cue():
    ambiguous = _hand((330, 520), (0.9, -0.3))  # low in the frame, pointing sideways: geometry says wearer
    arms = [[(np.array([330.0, 522.0]), np.array([420.0, 380.0]))]]  # partner's YOLO wrist here, elbow up-right
    A0 = P.attribute_hands([_det([(ambiguous, 0.3)])], W, H, 10.0)
    A1 = P.attribute_hands([_det([(ambiguous, 0.3)])], W, H, 10.0, partner_arms=arms)
    assert A0["present"][0].any() and A1["partner_present"][0].any() and not A1["present"][0].any()
    wearer_arm = [[(np.array([330.0, 522.0]), np.array([300.0, 700.0]))]]  # elbow towards the bottom: the wearer's arm
    assert P.attribute_hands([_det([(ambiguous, 0.3)])], W, H, 10.0, partner_arms=wearer_arm)["present"][0].any()
    kp = np.full((1, 4, 17, 3), np.nan); kp[0, 0] = 0; kp[0, 0, :, 2] = 0.9; kp[0, 0, 9, :2] = (330, 522); kp[0, 0, 7, :2] = (420, 380)
    assert len(P.partner_arms_from_body2d(kp)[0]) == 2  # face visible -> both arms offered
    kp[0, 0, :5, 2] = 0.1
    assert P.partner_arms_from_body2d(kp)[0] == []  # faceless 'person' (the wearer's own arms) is ignored


def test_hands_wearer_side_from_forearm_entry_beats_wrong_handedness():
    crossed = _hand((450, 560), (0.8, -0.6))  # LEFT arm from the bottom-left reaching across to the right side
    A = P.attribute_hands([_det([(crossed, 0.15), (WEARER_R, 0.2)])], W, H, 10.0)  # MediaPipe calls both "Right"
    np.testing.assert_allclose(A["lm2d"][0, 0], crossed); np.testing.assert_allclose(A["lm2d"][0, 1], WEARER_R)
    assert A["score"][0, 0] == pytest.approx(0.15)  # score stays MediaPipe's own P(left) (low): not a side confidence
    assert float(A["side_midline"]) == 0.5  # too little evidence to estimate the midline: centre


def test_hands_partner_side_continuity():
    frames, pl = [], [0.9, 0.85, 0.45, 0.9, 0.4, 0.88]  # the SAME partner hand; weak label flips at frames 2 and 4
    for k, p_left in enumerate(pl):
        frames.append(_det([(PARTNER + [k, 0], p_left)]))
    A = P.attribute_hands(frames, W, H, 10.0)
    assert A["partner_present"][:, 0].all() and not A["partner_present"][:, 1].any() and not A["present"].any()
    assert A["partner_score"][2, 0] == pytest.approx(0.45)
    assert P.attribute_hands([frames[2]], W, H, 10.0)["partner_present"][0, 1]  # without history the flip goes right


def test_hands_owner_follows_evidence_not_history():
    up = _hand((250, 490), (0, -1))  # frames 0-2: low, fingers up -> wearer
    side = _hand((250, 461), (1, 0))  # frames 3-5: same place (0.7 palm lengths), pointing sideways, higher -> partner
    frames = [_det([(up, 0.2)])] * 3 + [_det([(side, 0.1)])] * 3  # MediaPipe: "Right"; the forearm says left arm
    A = P.attribute_hands(frames, W, H, 10.0)
    assert A["present"][:3, 0].all() and A["partner_present"][3:].any(1).all() and not A["present"][3:].any()
    assert (A["partner_p_wearer"][3:][A["partner_present"][3:]] < 0.3).all()


def test_side_midline_estimated_per_stream():
    rng = np.random.default_rng(0); prm = P.HandParams()
    left, right = rng.uniform(0.30, 0.58, 40), rng.uniform(0.64, 0.90, 40)  # off-centre camera: midline ~0.6
    mid = P._side_midline(list(left) + list(right), [True] * 40 + [False] * 40, prm)
    assert 0.57 <= mid <= 0.64  # any separating boundary; ties go to the one nearest the centre
    assert P._side_midline([0.2, 0.8], [True, False], prm) == 0.5  # too little evidence
    assert P._side_midline(list(rng.uniform(0.0, 0.2, 40)), [True] * 40, prm) == 0.5  # uninformative: stays central


def test_partner_seen_and_solo_wearer_prior():
    kp = np.full((40, 4, 17, 3), np.nan); face = lambda k: kp[k, 0].__setitem__(slice(None), [100, 100, 0.9])  # noqa: E731
    for k in (3, 10, 11):  # isolated face detections (< 0.3 s runs) are false positives
        face(k)
    assert P.partner_seen_s(kp, 10.0) == 0.0
    for k in range(20, 32):  # a partner facing the wearer for 1.2 s
        face(k)
    assert P.partner_seen_s(kp, 10.0) == pytest.approx(1.2)
    high = _hand((320, 150), (0.3, -1))  # the wearer's hand near the top of a steeply down-looking camera
    assert P.attribute_hands([_det([(high, 0.5)])], W, H, 10.0)["partner_present"][0].any()
    A = P.attribute_hands([_det([(high, 0.5)])], W, H, 10.0, prior=P.HAND_PARAMS.solo_prior)
    assert A["present"][0].any() and float(A["owner_prior"]) == P.HAND_PARAMS.solo_prior


def test_hands_duplicates_merged_and_empty_frames():
    dup = WEARER_L + 2.0
    A = P.attribute_hands([_det([(WEARER_L, 0.9), (dup, 0.8)]), _det([])], W, H, 10.0)
    assert A["n_detected"].tolist() == [2, 0] and A["n_merged"].tolist() == [1, 0] and A["present"].sum() == 1
    assert not A["present"][1].any() and np.isnan(A["lm2d"][1]).all() and np.isnan(A["score"][1]).all()


# ------------------------------------------------------------------------------------------ stages with fake model packages

def _frames_episode(tmp_path: Path, n: int = 5, roles=("ego",), size=(64, 48)) -> Episode:
    """Episode with n extracted frames per stream written directly (index.parquet + manifest), no video needed."""
    import pandas as pd
    from PIL import Image

    from duet.playground import runtime
    from duet.playground.episode import Stream
    root = tmp_path / "ep"; root.mkdir()
    ep = Episode(name="ep", root=str(root), proc_fps=10.0, proc_size=64, common_start_s=0.0, common_end_s=n / 10)
    ep.streams = [Stream(name=f"s{i}", role=r, path=f"streams/s{i}.mp4", person=f"p{i}", width=size[0] * 2, height=size[1] * 2,
                         offset_status="reference" if i == 0 else "manual") for i, r in enumerate(roles)]
    ep.reference = "s0"; ep.save(); entries = {}
    for s in ep.streams:
        d = P.frames_dir(ep, s); d.mkdir(parents=True)
        for k in range(n):
            Image.new("RGB", size, (10 * k, 0, 0)).save(d / f"{k + 1:06d}.jpg")
        runtime.atomic_write_parquet(pd.DataFrame({"k": np.arange(n), "file": [f"{k + 1:06d}.jpg" for k in range(n)], "t_ref_s": np.arange(n) / 10,
                                                   "t_src_s": np.arange(n) / 10, "src_frame": 3 * np.arange(n)}), d / "index.parquet")
        entries[s.name] = {"width": size[0], "height": size[1], "scale": 0.5}
    runtime.atomic_write_json(ep.derived / "frames" / "manifest.json", {"version": 1, "streams": entries})
    ep.set_status("frames", "done", "test")
    return ep


class _T:  # minimal torch-like tensor
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _NS(types.SimpleNamespace):  # ultralytics Boxes/Keypoints stand-in (len = number of detections)
    def __len__(self):
        return len(next(iter(vars(self).values())).a)


def _fake_ultralytics(monkeypatch, record: dict, kind: str):
    mod = types.ModuleType("ultralytics"); utils = types.ModuleType("ultralytics.utils"); checks = types.ModuleType("ultralytics.utils.checks")
    utils.AUTOINSTALL = True; checks.AUTOINSTALL = True

    class Res:
        def __init__(self, shape, k):
            self.orig_shape = shape; self.names = {0: "bowl", 1: "pan", 2: "cup"}
            b = np.array([[5, 5, 30, 40], [6, 5, 31, 40], [40, 10, 60, 30]], float)[: (k % 3) + 1]
            self.boxes = _NS(xyxy=_T(b), conf=_T([0.5, 0.9, 0.7][: len(b)]), cls=_T([0, 1, 2][: len(b)]))
            kp = np.zeros((len(b), 17, 2)); kp[..., 0] = b[:, None, 0] + 3; kp[..., 1] = b[:, None, 1] + np.arange(17)
            self.keypoints = _NS(xy=_T(kp), conf=_T(np.full((len(b), 17), 0.8)))

    class YOLO:
        def __init__(self, path):
            record["path"] = path; record["env_autoinstall"] = __import__("os").environ.get("YOLO_AUTOINSTALL"); record["calls"] = []; self.k = 0
            self.names = dict(enumerate(P.WORLD_VOCAB)) if "-vocab-" in path else {0: "person"}
            self.model = types.SimpleNamespace(clip_model="300 MB of CLIP")

        def set_classes(self, classes):
            record["classes"] = list(classes); self.names = dict(enumerate(classes))

        def save(self, filename):
            record["saved_has_clip"] = hasattr(self.model, "clip_model"); Path(filename).write_bytes(b"baked")

        def predict(self, source, **kw):
            record["calls"].append({**kw, "n": len(source)})
            for im in source:
                yield Res(im.shape[:2], self.k); self.k += 1

    mod.YOLO = YOLO; mod.utils = utils; utils.checks = checks; utils.WEIGHTS_DIR = record.setdefault("weights_dir", Path("/nonexistent"))
    for name, m in (("ultralytics", mod), ("ultralytics.utils", utils), ("ultralytics.utils.checks", checks)):
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    return utils, checks


def _models(tmp_path, monkeypatch, *names):
    d = tmp_path / "models"; d.mkdir(exist_ok=True)
    for nm in names:
        (d / nm).write_bytes(b"x")
    monkeypatch.setenv("PLAYGROUND_MODELS", str(d))
    return d


def test_objects_agnostic_nms_batching_and_no_runtime_installs(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    rec: dict = {"weights_dir": tmp_path / "ulw"}; utils, checks = _fake_ultralytics(monkeypatch, rec, "world")
    ep = _frames_episode(tmp_path, n=5); monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "no_models"))
    with pytest.raises(FileNotFoundError, match="fetch-models"):
        P.objects(ep, device="cpu")  # no weights: clear error, nothing downloaded into the CWD
    d = _models(tmp_path, monkeypatch, P.WORLD_WEIGHTS)
    monkeypatch.setitem(sys.modules, "clip", None)
    with pytest.raises(RuntimeError, match="CLIP package.*fetch-models"):
        P.objects(ep, device="cpu")  # never a run-time pip install
    monkeypatch.setitem(sys.modules, "clip", types.ModuleType("clip"))
    with pytest.raises(FileNotFoundError, match="no downloads at run time"):
        P.objects(ep, device="cpu")  # clip.load would fetch 338 MB of weights
    (tmp_path / "ulw" / "clip").mkdir(parents=True); (tmp_path / "ulw" / "clip" / "ViT-B-32.pt").write_bytes(b"x")
    P.objects(ep, device="cpu")
    assert rec["path"] == str(d / P.WORLD_WEIGHTS) and rec["env_autoinstall"] == "False" and not utils.AUTOINSTALL and not checks.AUTOINSTALL
    assert rec["classes"] == P.WORLD_VOCAB and len(set(rec["classes"])) == len(rec["classes"]) and "CLIP set_classes" in ep.status["objects"]["detail"]
    call = rec["calls"][0]
    assert call["agnostic_nms"] is True and call["stream"] is True and call["device"] == "cpu" and call["batch"] == 5
    assert "half" not in call and "quantize" not in call  # fp32 off CUDA
    from duet.playground import runtime
    z = runtime.load_npz(ep.derived / "objects" / "s0.npz")
    assert z["boxes"].shape == (5, P.MAX_OBJECTS, 5) and z["names"].shape == (5, P.MAX_OBJECTS)
    assert list(z["names"][1, :2]) == ["pan", "bowl"] and z["boxes"][1, 0, 4] == pytest.approx(0.9)  # most confident first
    # the baked checkpoint (fetch-models) needs neither CLIP nor its weights at run time
    baked = P.bake_world_vocab()
    assert baked == d / P.world_vocab_weights() and baked.is_file() and rec["saved_has_clip"] is False
    monkeypatch.setitem(sys.modules, "clip", None); rec.pop("classes")
    P.objects(ep, device="cpu")
    assert rec["path"] == str(baked) and "classes" not in rec and "baked vocabulary" in ep.status["objects"]["detail"]


def test_body2d_layout_and_half_on_cuda_only(tmp_path, monkeypatch):
    pytest.importorskip("cv2"); pytest.importorskip("scipy")
    rec: dict = {}; _fake_ultralytics(monkeypatch, rec, "pose"); _models(tmp_path, monkeypatch, P.POSE_WEIGHTS)
    ep = _frames_episode(tmp_path, n=4, roles=("ego", "exo"))
    P.body2d(ep, device="cuda:0")
    assert all(c.get("half") is True and c["device"] == "cuda:0" and c["stream"] is True for c in rec["calls"])  # fp16 on CUDA
    from duet.playground import runtime
    z = runtime.load_npz(ep.derived / "body2d" / "s1.npz")
    assert z["kpts"].shape == (4, P.MAX_PEOPLE, 17, 3) and z["boxes"].shape == (4, P.MAX_PEOPLE, 5)
    assert int(z["img_w"]) == 64 and int(z["img_h"]) == 48 and int(z["schema"]) == 2
    z0 = runtime.load_npz(ep.derived / "body2d" / "s0.npz")  # s0 frame 1: two near-identical boxes (IoU 0.92) = one person
    assert np.isfinite(z0["boxes"][1, :, 4]).sum() == 1 and np.nanmax(z0["boxes"][1, :, 4]) == pytest.approx(0.9)


def _fake_mediapipe(monkeypatch, rec: dict, hands_per_frame=2):
    mp = types.ModuleType("mediapipe"); tasks = types.ModuleType("mediapipe.tasks"); py = types.ModuleType("mediapipe.tasks.python")
    vision = types.ModuleType("mediapipe.tasks.python.vision")
    mp.Image = lambda image_format, data: types.SimpleNamespace(data=data)
    mp.ImageFormat = types.SimpleNamespace(SRGB=1); py.BaseOptions = lambda model_asset_path: types.SimpleNamespace(path=model_asset_path)
    vision.RunningMode = types.SimpleNamespace(VIDEO="VIDEO", IMAGE="IMAGE")
    vision.HandLandmarkerOptions = lambda **kw: kw; vision.PoseLandmarkerOptions = lambda **kw: kw
    L = lambda x, y, z=0.0, v=1.0: types.SimpleNamespace(x=x, y=y, z=z, visibility=v)  # noqa: E731

    class Landmarker:
        def __init__(self, opts):
            rec["opts"] = opts; rec["ts"] = []; rec["closed"] = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            rec["closed"] = True

        def detect_for_video(self, img, ts):
            rec["ts"].append(ts); h = []
            for j in range(hands_per_frame):  # hands low in the frame, fingers up: the wearer's
                wx = 0.3 + 0.4 * j
                h.append([L(wx, 0.95)] + [L(wx + 0.01 * ((i % 4) - 2), 0.95 - 0.02 * (1 + i // 4)) for i in range(20)])
            cat = lambda nm: [types.SimpleNamespace(category_name=nm, score=0.9)]  # noqa: E731
            return types.SimpleNamespace(hand_landmarks=h, hand_world_landmarks=[[L(0.01 * i, 0.02, 0.03) for i in range(21)] for _ in h],
                                         handedness=[cat("Right"), cat("Right")][:len(h)],
                                         pose_landmarks=[[L(0.5, 0.5, 0, 0.9)] * 33], pose_world_landmarks=[[L(0.1, 0.2, 0.3)] * 33])

    vision.HandLandmarker = types.SimpleNamespace(create_from_options=Landmarker)
    vision.PoseLandmarker = types.SimpleNamespace(create_from_options=Landmarker)
    mp.tasks = tasks; tasks.python = py; py.vision = vision
    for name, m in (("mediapipe", mp), ("mediapipe.tasks", tasks), ("mediapipe.tasks.python", py), ("mediapipe.tasks.python.vision", vision)):
        monkeypatch.setitem(sys.modules, name, m)


def test_hands_stage_writes_v2_schema_and_closes_landmarker(tmp_path, monkeypatch):
    pytest.importorskip("cv2"); pytest.importorskip("scipy")
    rec: dict = {}; _fake_mediapipe(monkeypatch, rec); _models(tmp_path, monkeypatch, P.HAND_TASK)
    ep = _frames_episode(tmp_path, n=4, roles=("ego", "ego"))
    P.hands(ep)
    assert rec["closed"] and rec["opts"]["num_hands"] >= 4 and rec["ts"] == [0, 100, 200, 300]
    from duet.playground import runtime
    z = runtime.load_npz(ep.derived / "hands" / "s0.npz")
    assert int(z["schema"]) == 2 and str(z["lm3d_origin"]) == "wrist" and str(z["wearer"]) == "p0" and str(z["partner"]) == "p1"
    for pre in ("", "partner_"):
        assert z[pre + "lm2d"].shape == (4, 2, 21, 2) and z[pre + "lm3d"].shape == (4, 2, 21, 3) and z[pre + "score"].shape == (4, 2)
        assert z[pre + "present"].dtype == bool
    assert z["present"].all() and z["n_detected"].tolist() == [2, 2, 2, 2]  # two "Right" hands: both kept, one per slot
    assert np.all(z["lm3d"][:, :, 0] == 0) and "body2d" in ep.status["hands"]["detail"]  # no body2d: geometric cues only
    assert float(z["owner_prior"]) == 0.0
    faceless = {"kpts": np.full((4, 4, 17, 3), np.nan), "boxes": np.full((4, 4, 5), np.nan), "img_w": 64, "img_h": 48, "schema": 2}
    runtime.atomic_savez(ep.derived / "body2d" / "s0.npz", **faceless)
    ep.set_status("body2d", "done", "test"); P.hands(ep)  # no face ever seen, but a second ego wearer is declared: no prior
    z = runtime.load_npz(ep.derived / "hands" / "s0.npz")
    assert float(z["owner_prior"]) == 0.0 and bool(z["partner_declared"]) and "alone" not in ep.status["hands"]["detail"]


def test_hands_solo_prior_only_for_a_lone_wearer(tmp_path, monkeypatch):
    pytest.importorskip("cv2"); pytest.importorskip("scipy")
    from duet.playground import runtime
    rec: dict = {}; _fake_mediapipe(monkeypatch, rec); _models(tmp_path, monkeypatch, P.HAND_TASK)
    ep = _frames_episode(tmp_path, n=4, roles=("ego", "exo"))  # one wearer + an unlabelled fixed camera
    ep.stream("s1").person = None; ep.save()
    runtime.atomic_savez(ep.derived / "body2d" / "s0.npz", kpts=np.full((4, 4, 17, 3), np.nan), boxes=np.full((4, 4, 5), np.nan), img_w=64, img_h=48, schema=2)
    ep.set_status("body2d", "done", "test"); P.hands(ep)
    z = runtime.load_npz(ep.derived / "hands" / "s0.npz")
    assert float(z["owner_prior"]) == P.HAND_PARAMS.solo_prior and not bool(z["partner_declared"]) and "alone" in ep.status["hands"]["detail"]


def test_declares_partner():
    from duet.playground.episode import Stream
    ep = Episode(name="e", root="/nonexistent")
    mk = lambda *v: [Stream(name=n, role=r, path=f"{n}.mp4", person=p) for n, r, p in v]  # noqa: E731
    ep.streams = mk(("a", "ego", "alice"), ("b", "ego", "bob"))
    assert P.declares_partner(ep, ep.stream("a"))  # two wearers (e.g. CoMind)
    ep.streams = mk(("a", "ego", "alice"), ("b", "ego", None))
    assert P.declares_partner(ep, ep.stream("a"))  # a second ego camera of an unknown wearer
    ep.streams = mk(("a", "ego", "alice"), ("head2", "ego", "alice"), ("cam", "exo", None))
    assert not P.declares_partner(ep, ep.stream("a"))  # one person with two cameras + an unlabelled exo
    ep.streams = mk(("a", "ego", "alice"), ("cam", "exo", "bob"))
    assert P.declares_partner(ep, ep.stream("a"))  # someone else named on a fixed camera
    ep.streams = mk(("a", "ego", None))
    assert not P.declares_partner(ep, ep.stream("a"))  # single-person recording (e.g. Eidon)


def test_body3d_skipped_without_exo_view(tmp_path, monkeypatch):
    rec: dict = {}; _fake_mediapipe(monkeypatch, rec)
    ep = _frames_episode(tmp_path, n=3, roles=("ego",))
    P.body3d(ep)
    assert ep.status["body3d"]["state"] == "skipped" and not (ep.derived / "body3d").exists() and "opts" not in rec


def test_body3d_on_exo_records_view_and_visibility(tmp_path, monkeypatch):
    pytest.importorskip("cv2"); pytest.importorskip("scipy")
    rec: dict = {}; _fake_mediapipe(monkeypatch, rec); _models(tmp_path, monkeypatch, P.POSE_TASK)
    ep = _frames_episode(tmp_path, n=3, roles=("ego", "exo"))
    from duet.playground import runtime
    b = np.full((3, 4, 5), np.nan); b[:, 0] = [0, 0, 20, 40, 0.9]
    runtime.atomic_savez(ep.derived / "body2d" / "s1.npz", kpts=np.full((3, 4, 17, 3), np.nan), boxes=b, img_w=64, img_h=48, schema=2)
    ep.set_status("body2d", "done", "test")
    P.body3d(ep)
    z = runtime.load_npz(ep.derived / "body3d" / "body3d.npz")
    assert str(z["view"]) == str(z["stream"]) == "s1" and z["vis"].shape == (3, 2, 33) and z["world"].shape == (3, 2, 33, 3)
    assert rec["closed"] and np.allclose(z["vis"][:, 0], 0.9) and int(z["schema"]) == 2


def test_real_mediapipe_hands_smoke(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe"); pytest.importorskip("cv2"); pytest.importorskip("scipy")
    from duet.playground import runtime
    if not (runtime.models_dir() / P.HAND_TASK).is_file():
        pytest.skip("hand_landmarker.task not in the models directory")
    ep = _frames_episode(tmp_path, n=3, roles=("ego",))
    P.hands(ep)
    z = runtime.load_npz(ep.derived / "hands" / "s0.npz")
    assert int(z["schema"]) == 2 and z["lm2d"].shape == (3, 2, 21, 2) and z["n_detected"].tolist() == [0, 0, 0]  # blank frames
