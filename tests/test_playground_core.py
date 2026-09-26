"""Core playground infrastructure: Episode persistence/validation/time mapping, probe, runtime helpers, run.py
orchestration (fingerprints, invalidation, DEPS, output clearing, job queue in subprocesses, stale recovery) and
the CLI. Synthetic episodes only (tmp_path); the committed episodes are only READ (copied into tmp_path)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from duet.playground import run as R
from duet.playground import runtime
from duet.playground.episode import STAGES, Episode, EpisodeError, Stream, probe, run_lock

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
COMMITTED = sorted((REPO / "data/playground/episodes").glob("*/episode.json"))
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _video(path: Path, dur: float = 1.0, audio: float | None = None, size: str = "64x48", extra_in: list[str] | None = None,
           codec: list[str] | None = None) -> Path:
    """Tiny lavfi test clip (optionally with a sine track of a different length)."""
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *(extra_in or []), "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={dur}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=8000:duration={audio}"]
    cmd += codec or ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else [])
    subprocess.run(cmd + [str(path)], check=True)
    return path


def _dummy(path: Path, data: bytes = b"not really a video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data); return path


def _mk(tmp_path: Path, name: str = "E", n: int = 2, **kw) -> Episode:
    vids = [(f"s{i}", "ego" if i == 0 else "exo", _dummy(tmp_path / "media" / f"{name}_s{i}.mp4"), None) for i in range(n)]
    return Episode.create(tmp_path / "eps", name, vids, **kw)


# ============================================================================================ Episode persistence

@pytest.mark.parametrize("src", COMMITTED, ids=[p.parent.name for p in COMMITTED])
def test_committed_episode_json_loads_and_upgrades(tmp_path, src):
    d = tmp_path / src.parent.name; d.mkdir(); shutil.copy(src, d / "episode.json")
    raw = json.loads((d / "episode.json").read_text())
    ep = Episode.load(d)
    assert ep.name == raw["name"] and [s.name for s in ep.streams] == [s["name"] for s in raw["streams"]]
    assert ep.schema_version == 1 and ep.max_lag_s == 120.0 and ep.device == "auto" and ep.hwaccel == "auto"
    for s, rs in zip(ep.streams, raw["streams"]):
        assert s.offset_s == rs["offset_s"] and s.video_start_s == 0.0 and s.drift_ppm == 0.0 and s.offset_override_s is None
        assert s.imu_offset_override_s is None
        want = "reference" if s.name == raw["reference"] else ("audio" if (rs.get("offset_confidence") or 0) >= 1.3 else "unaligned")
        assert s.offset_status == want and ep.usable(s)
    assert ep.stage_ok("probe") and ep.n_frames() == {"comind_43276420_clip": 466, "eidon_10004": 617}.get(ep.name, ep.n_frames())
    assert ep.min_overlap_s == 5.0
    ep.set_status("qc", "stale", "x")  # first write upgrades the file, keeps everything else
    new = json.loads((d / "episode.json").read_text())
    assert new["schema_version"] == 2 and new["streams"][0]["offset_status"] in ("reference", "audio", "unaligned")
    assert {k: v for k, v in new["status"].items() if k != "qc"} == {k: v for k, v in raw["status"].items() if k != "qc"}
    assert not list(d.glob(".episode.json.*"))  # no temp file left behind


def test_legacy_low_confidence_offsets_are_unaligned(tmp_path):
    d = tmp_path / "L"; d.mkdir()
    streams = [{"name": n, "role": "exo", "path": f"streams/{n}.mp4", "offset_confidence": c} for n, c in
               (("r", None), ("good", 1.3), ("weak", 1.29), ("noaudio", 0.0), ("never", None))]
    (d / "episode.json").write_text(json.dumps({"name": "L", "streams": streams, "reference": "r"}))
    got = {s.name: s.offset_status for s in Episode.load(d).streams}
    assert got == {"r": "reference", "good": "audio", "weak": "unaligned", "noaudio": "unaligned", "never": "unaligned"}


def test_load_is_tolerant_and_keeps_unknown_keys(tmp_path):
    d = tmp_path / "X"; d.mkdir()
    (d / "episode.json").write_text(json.dumps({"name": "X", "streams": [{"name": "a", "role": "ego", "path": "streams/a.mp4", "future": 1}],
                                                "reference": "a", "status": {}, "zzz_new_top": {"k": [1, 2]}}))
    ep = Episode.load(d)
    assert ep.stream("a").offset_status == "reference" and ep.proc_fps == 10.0 and ep.notes == []
    ep.proc_fps = 5.0; ep.save()
    j = json.loads((d / "episode.json").read_text())
    assert j["zzz_new_top"] == {"k": [1, 2]} and j["streams"][0]["future"] == 1 and j["proc_fps"] == 5.0
    with pytest.raises(KeyError):
        ep.stream("nope")


def test_corrupt_or_missing_json_raises_episode_error(tmp_path):
    d = tmp_path / "C"; d.mkdir()
    with pytest.raises(EpisodeError):
        Episode.load(d)
    (d / "episode.json").write_text('{"name": "C", "streams": [')  # torn file written by the old code
    with pytest.raises(EpisodeError):
        Episode.load(d)
    assert issubclass(EpisodeError, ValueError)


def test_lost_update_scenario_from_review(tmp_path):
    """CLI re-align (offset) and a server thread with a stale copy (body2d status) must both survive."""
    ep = _mk(tmp_path)
    a, b = Episode.load(ep.dir), Episode.load(ep.dir)
    a.stream("s1").offset_s = 9.99; a.set_status("align", "done", "CLI re-align")
    b.set_status("body2d", "done", "server thread finishes body2d")
    c = Episode.load(ep.dir)
    assert c.stream("s1").offset_s == 9.99 and c.status["align"]["detail"] == "CLI re-align" and c.status["body2d"]["state"] == "done"
    assert b.stream("s1").offset_s == 9.99  # the stale copy was refreshed by its own write


def test_save_refreshes_in_place_and_update_is_read_modify_write(tmp_path):
    ep = _mk(tmp_path); s1 = ep.stream("s1"); other = Episode.load(ep.dir)
    other.update(lambda e: setattr(e.stream("s1"), "offset_s", 1.5))
    ep.common_end_s = 3.0; ep.save()
    assert ep.stream("s1") is s1 and s1.offset_s == 1.5 and Episode.load(ep.dir).common_end_s == 3.0
    ep.update(lambda e: e.notes.append("n1")); other.update(lambda e: e.notes.append("n2"))
    assert Episode.load(ep.dir).notes == ["n1", "n2"]


def test_threads_no_torn_reads_no_lost_updates(tmp_path):
    ep = _mk(tmp_path); stop = threading.Event(); errors: list[str] = []; reads = [0]

    def writer(i: int) -> None:
        e = Episode.load(ep.dir)
        for j in range(40):
            e.set_status(f"stage{i}", "running", f"step {j}")
        e.set_status(f"stage{i}", "done", f"final {i}")

    def reader() -> None:
        while not stop.is_set():
            try:
                Episode.load(ep.dir); reads[0] += 1
            except Exception as ex:  # noqa: BLE001
                errors.append(repr(ex))

    rt = threading.Thread(target=reader); rt.start()
    ws = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    [t.start() for t in ws]; [t.join() for t in ws]; stop.set(); rt.join()
    final = Episode.load(ep.dir)
    assert not errors and reads[0] > 0
    assert {f"stage{i}": ("done", f"final {i}") for i in range(6)} == {k: (v["state"], v["detail"]) for k, v in final.status.items()}


_WRITER = """
import sys; sys.path.insert(0, {src!r})
from duet.playground.episode import Episode
ep = Episode.load(sys.argv[1]); i = int(sys.argv[2])
for j in range(30):
    ep.set_status(f"p{{i}}", "running", f"step {{j}}")
ep.update(lambda e: setattr(e.stream("s1"), "offset_override_s", 0.25) if i == 0 else e.notes.append(f"w{{i}}"))
ep.set_status(f"p{{i}}", "done", f"final {{i}}")
"""


def test_processes_no_torn_reads_no_lost_updates(tmp_path):
    ep = _mk(tmp_path); code = _WRITER.format(src=str(SRC))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(ep.dir), str(i)]) for i in range(4)]
    errors = []
    while any(p.poll() is None for p in procs):
        try:
            Episode.load(ep.dir)
        except Exception as ex:  # noqa: BLE001
            errors.append(repr(ex))
    assert [p.returncode for p in procs] == [0] * 4 and not errors
    final = Episode.load(ep.dir)
    assert {k: v["detail"] for k, v in final.status.items()} == {f"p{i}": f"final {i}" for i in range(4)}
    assert final.stream("s1").offset_override_s == 0.25 and sorted(final.notes) == ["w1", "w2", "w3"]


def test_crash_mid_write_keeps_previous_file(tmp_path):
    target = tmp_path / "x.json"; runtime.atomic_write_json(target, {"a": 1})

    def boom(f):
        f.write(b'{"a": 2, "trunc'); raise OSError("disk full")
    with pytest.raises(OSError):
        runtime.atomic_write(target, boom)
    assert json.loads(target.read_text()) == {"a": 1} and [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_set_status_fields_detail_cleaning(tmp_path):
    ep = _mk(tmp_path)
    ep.set_status("frames", "running")
    st = ep.status["frames"]; assert st["pid"] == os.getpid() and st["host"] == socket.gethostname() and "finished" not in st
    long = f"boom at {ep.dir}/streams/s0.mp4 " + "x" * 5000 + " END"
    ep.set_status("frames", "failed", long)
    st = Episode.load(ep.dir).status["frames"]
    assert st["state"] == "failed" and st["finished"] >= st["started"] and len(st["detail"]) <= 2000
    assert str(ep.dir.parent) not in st["detail"] and "E/streams/s0.mp4" in st["detail"] and st["detail"].endswith("END")
    ep.set_status("calib", "skipped", "no board"); assert ep.status["calib"]["finished"] and ep.status["calib"]["started"]
    with pytest.raises(ValueError):
        ep.set_status("calib", "weird")


# ============================================================================================ time mapping

def test_time_mapping_n_frames_and_usable():
    ep = Episode(name="T", root="/nonexistent", streams=[Stream("r", "ego", "streams/r.mp4", offset_status="reference"),
                                                        Stream("a", "exo", "streams/a.mp4", offset_s=2.5, drift_ppm=100.0, offset_status="audio"),
                                                        Stream("u", "exo", "streams/u.mp4")], reference="r")
    t = np.array([0.0, 1.0, 3600.0])
    assert np.allclose(ep.ref_time("a", t), (1 + 1e-4) * t + 2.5) and np.allclose(ep.stream_time("a", ep.ref_time("a", t)), t)
    assert isinstance(ep.ref_time("r", 1.0), float) and ep.ref_time("r", 1.0) == 1.0
    ep.common_start_s, ep.common_end_s, ep.proc_fps = 2.5, 12.5, 10.0
    assert ep.n_frames() == 100 and np.allclose(ep.frame_times_ref()[[0, -1]], [2.5, 12.4])
    ep.common_end_s = 2.5 + 0.3 * 3; ep.proc_fps = 10.0 / 3  # 0.9 s * 3.333 fps = 3.0 frames despite float error
    assert ep.n_frames() == 3
    assert np.allclose(ep.frame_times_stream("a"), ep.stream_time("a", ep.frame_times_ref()))
    for start, end, fps in [(5.0, 5.0, 10.0), (5.0, 4.0, 10.0), (0.0, 0.05, 10.0), (0.0, float("nan"), 10.0), (0.0, 10.0, 0.0)]:
        ep.common_start_s, ep.common_end_s, ep.proc_fps = start, end, fps
        with pytest.raises(ValueError):
            ep.n_frames()
    assert ep.usable("r") and ep.usable("a") and not ep.usable("u")
    ep.status = {"frames": {"state": "done"}, "calib": {"state": "stale"}}
    assert ep.stage_ok("frames") and not ep.stage_ok("calib") and not ep.stage_ok("hands")


# ============================================================================================ create validation

def test_create_relative_symlinks_and_defaults(tmp_path):
    src = _dummy(tmp_path / "rec" / "cam 1.MP4")
    ep = Episode.create(tmp_path / "eps", "E", [("front", "exo", src, None), ("lead", "ego", _dummy(tmp_path / "rec/l.mp4"), "alice")],
                        offsets={"front": 1.25}, proc_fps=5)
    link = ep.dir / "streams/front.mp4"
    assert link.is_symlink() and not os.path.isabs(os.readlink(link)) and link.read_bytes() == src.read_bytes()
    assert ep.reference == "lead" and ep.stream("lead").offset_status == "reference" and ep.stream("lead").person == "alice"
    assert ep.stream("front").offset_override_s == 1.25 and ep.proc_fps == 5.0 and ep.schema_version == 2
    moved = tmp_path / "moved"; shutil.move(str(tmp_path / "eps"), moved); shutil.move(str(tmp_path / "rec"), tmp_path / "x" / "rec")
    # relative links survive moving the episodes root and the sources together (same relative layout)
    shutil.move(str(moved), tmp_path / "x" / "eps")
    assert (tmp_path / "x/eps/E/streams/front.mp4").read_bytes() == b"not really a video"
    assert not (tmp_path / "x/eps/.staging").exists() or not any((tmp_path / "x/eps/.staging").iterdir())


@pytest.mark.parametrize("bad", [
    dict(videos=[("a", "ego", "S0", None), ("a", "exo", "S1", None)], err=ValueError, match="duplicate"),
    dict(videos=[("../../escaped", "ego", "S0", None)], err=ValueError, match="stream name"),
    dict(videos=[("a b", "ego", "S0", None)], err=ValueError, match="stream name"),
    dict(videos=[("a", "top", "S0", None)], err=ValueError, match="role"),
    dict(videos=[("a", "ego", "MISSING", None)], err=FileNotFoundError, match="does not exist"),
    dict(videos=[("a", "ego", "S0", "<img src=x>")], err=ValueError, match="person"),
    dict(videos=[("a", "ego", "S0", None)], reference="zz", err=ValueError, match="reference"),
    dict(videos=[("a", "ego", "S0", None)], imus={"zz": "S1"}, err=ValueError, match="unknown stream"),
    dict(videos=[("a", "ego", "S0", None)], offsets={"zz": 1.0}, err=ValueError, match="unknown stream"),
    dict(videos=[("a", "ego", "S0", None)], offsets={"a": float("nan")}, err=ValueError, match="finite"),
    dict(videos=[("a", "ego", "S0", None)], imu_offsets={"a": 0.4}, err=ValueError, match="no IMU"),
    dict(videos=[("a", "ego", "S0", None)], min_overlap_s=0.0, err=ValueError, match="min_overlap_s"),
    dict(videos=[("a", "ego", "S0", None)], proc_fps=float("nan"), err=ValueError, match="proc_fps"),
    dict(videos=[("a", "ego", "S0", None)], proc_fps=1e6, err=ValueError, match="proc_fps"),
    dict(videos=[("a", "ego", "S0", None)], device="gpu", err=ValueError, match="device"),
    dict(videos=[("a", "ego", "S0", None)], name="../x", err=ValueError, match="episode name"),
    dict(videos=[], err=ValueError, match="at least one"),
])
def test_create_validation(tmp_path, bad):
    files = {"S0": _dummy(tmp_path / "m/s0.mp4"), "S1": _dummy(tmp_path / "m/s1.parquet"), "MISSING": tmp_path / "m/none.mp4"}
    vids = [(n, r, files.get(p, p), per) for n, r, p, per in bad["videos"]]
    kw = {k: v for k, v in bad.items() if k not in ("videos", "err", "match", "name")}
    if "imus" in kw:
        kw["imus"] = {k: files[v] for k, v in kw["imus"].items()}
    with pytest.raises(bad["err"], match=bad["match"]):
        Episode.create(tmp_path / "eps", bad.get("name", "E"), vids, **kw)
    assert not (tmp_path / "eps" / "E").exists() and not any((tmp_path / "eps").glob("*/episode.json"))


def test_create_refuses_existing_and_overwrite_replaces(tmp_path):
    ep = _mk(tmp_path)
    (ep.derived / "frames").mkdir(); (ep.dir / "rig.json").write_text("{}"); ep.set_status("frames", "done", "old")
    with pytest.raises(FileExistsError):
        _mk(tmp_path)
    ep2 = Episode.create(tmp_path / "eps", "E", [("new", "ego", _dummy(tmp_path / "m2/new.mov"), None)], overwrite=True)
    assert [s.name for s in ep2.streams] == ["new"] and ep2.status == {} and not (ep2.dir / "derived/frames").exists()
    assert sorted(p.name for p in (ep2.dir / "streams").iterdir()) == ["new.mov"] and (ep2.dir / "rig.json").exists()
    with run_lock(ep2.dir), pytest.raises(RuntimeError, match="running"):  # a running pipeline blocks overwriting
        Episode.create(tmp_path / "eps", "E", [("x", "ego", _dummy(tmp_path / "m3/x.mp4"), None)], overwrite=True)


def test_concurrent_create_of_the_same_episode(tmp_path):
    src = _dummy(tmp_path / "m/a.mp4"); out: list[str] = []; barrier = threading.Barrier(8)

    def go(i: int) -> None:
        barrier.wait()
        try:
            Episode.create(tmp_path / "eps", "E", [(f"cam{i}", "ego", src, None)], mode="copy"); out.append(f"ok{i}")
        except FileExistsError:
            out.append("exists")
    ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    ok = [x for x in out if x.startswith("ok")]
    assert len(ok) == 1 and out.count("exists") == 7
    ep = Episode.load(tmp_path / "eps/E")
    assert [s.name for s in ep.streams] == [f"cam{ok[0][2:]}"] and len(list((ep.dir / "streams").iterdir())) == 1
    assert not any((tmp_path / "eps/.staging").iterdir())


def test_create_copy_and_move_modes_with_rollback(tmp_path, monkeypatch):
    a, b, imu = _dummy(tmp_path / "m/a.mp4", b"A"), _dummy(tmp_path / "m/b.mp4", b"B"), _dummy(tmp_path / "m/a.parquet", b"I")
    ep = Episode.create(tmp_path / "eps", "C", [("a", "ego", a, None)], {"a": imu}, mode="copy", imu_offsets={"a": -0.42})
    assert not (ep.dir / "streams/a.mp4").is_symlink() and a.exists() and ep.stream("a").imu == "imu/a.parquet"
    assert ep.stream("a").imu_offset_override_s == -0.42 and (ep.dir / "imu/a.parquet").read_bytes() == b"I"
    real_rename = os.rename

    def failing_rename(src, dst):
        if Path(dst).name == "M":
            raise OSError("simulated failure")
        return real_rename(src, dst)
    monkeypatch.setattr(os, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated"):
        Episode.create(tmp_path / "eps", "M", [("a", "ego", a, None), ("b", "exo", b, None)], mode="move")
    assert a.read_bytes() == b"A" and b.read_bytes() == b"B" and not (tmp_path / "eps/M").exists()  # moved back
    monkeypatch.setattr(os, "rename", real_rename)
    ep = Episode.create(tmp_path / "eps", "M", [("a", "ego", a, None), ("b", "exo", b, None)], mode="move")
    assert not a.exists() and (ep.dir / "streams/b.mp4").read_bytes() == b"B"
    assert not any((tmp_path / "eps/.staging").iterdir())


# ============================================================================================ probe

@needs_ffmpeg
def test_probe_edge_cases(tmp_path):
    m = tmp_path / "m"; m.mkdir()
    base = _video(m / "base.mp4", 2.0, audio=2.0)
    clips = {
        "audlong": _video(m / "audlong.mp4", 1.0, audio=2.5),  # audio tail must not extend the video duration
        "noaud": _video(m / "noaud.mp4", 1.0),
        "live": _video(m / "live.webm", 1.0, audio=1.0, codec=["-c:v", "libvpx", "-c:a", "libopus", "-f", "webm", "-live", "1"]),
        "rot": m / "rot.mp4", "cover": m / "cover.mkv",
    }
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(base), "-c", "copy", str(clips["rot"])], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=32x32:duration=1", "-frames:v", "1", str(m / "c.png")], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(m / "c.png"), "-i", str(base), "-map", "0", "-map", "1", "-c", "copy",
                    "-c:v:0", "mjpeg", str(clips["cover"])], check=True)  # an image track BEFORE the real video
    ep = Episode.create(tmp_path / "eps", "P", [(k, "exo", v, None) for k, v in clips.items()])
    probe(ep)
    s = {x.name: x for x in Episode.load(ep.dir).streams}
    assert ep.status["probe"]["state"] == "done"
    assert s["audlong"].duration_s == pytest.approx(1.0, abs=0.05) and s["audlong"].has_audio and s["audlong"].nb_frames == 10
    assert not s["noaud"].has_audio and s["noaud"].audio_start_s is None
    assert s["live"].duration_s == pytest.approx(1.0, abs=0.15) and s["live"].fps == pytest.approx(10.0) and s["live"].nb_frames == 10
    assert (s["rot"].width, s["rot"].height, s["rot"].rotation) == (48, 64, 90)
    assert (s["cover"].width, s["cover"].height, s["cover"].video_index) == (64, 48, 1)
    for x in s.values():
        assert x.fps == pytest.approx(10.0) and x.video_start_s == pytest.approx(0.0, abs=0.05)


@needs_ffmpeg
def test_probe_missing_or_garbage_file_fails_with_stream_name(tmp_path):
    ep = _mk(tmp_path)  # dummy bytes, not videos
    with pytest.raises(RuntimeError, match=r"stream s0: .*ffprobe failed"):
        probe(ep)
    os.unlink(tmp_path / "media" / "E_s0.mp4")  # now a dangling relative symlink
    with pytest.raises(FileNotFoundError, match="dangling"):
        probe(Episode.load(ep.dir))


# ============================================================================================ runtime

def test_json_safe_and_atomic_writers(tmp_path):
    obj = {"a": np.float32(1.5), "b": [np.nan, np.inf, 2], "c": np.arange(3), "d": (np.int64(4), Path("/x")), "e": {1: np.bool_(True)}}
    assert runtime.json_safe(obj) == {"a": 1.5, "b": [None, None, 2], "c": [0, 1, 2], "d": [4, "/x"], "e": {"1": True}}
    runtime.atomic_write_json(tmp_path / "o.json", obj)
    assert "NaN" not in (tmp_path / "o.json").read_text()
    runtime.atomic_savez(tmp_path / "a.npz", x=np.ones((2, 3), np.float32), s=np.array("zed"))
    z = runtime.load_npz(tmp_path / "a.npz"); assert z["x"].shape == (2, 3) and str(z["s"]) == "zed"
    with pytest.raises(TypeError):
        runtime.atomic_savez(tmp_path / "b.npz", x=np.array([{"a": 1}], dtype=object))
    assert not (tmp_path / "b.npz").exists()
    pd = pytest.importorskip("pandas"); pq = pytest.importorskip("pyarrow.parquet")
    runtime.atomic_write_parquet(pd.DataFrame({"k": [1, 2]}), tmp_path / "r.parquet", {"units": "m", "offsets": {"a": 1.5, "b": np.nan}})
    md = pq.read_schema(tmp_path / "r.parquet").metadata
    assert md[b"units"] == b"m" and json.loads(md[b"offsets"]) == {"a": 1.5, "b": None}
    pa = pytest.importorskip("pyarrow")
    runtime.atomic_write_parquet(pa.table({"k": [1]}), tmp_path / "t.parquet", {"v": 2})
    assert pq.read_table(tmp_path / "t.parquet").num_rows == 1 and pq.read_schema(tmp_path / "t.parquet").metadata[b"v"] == b"2"


def test_pick_device_and_hwaccel(monkeypatch):
    assert runtime.pick_device("cpu") == "cpu" and runtime.pick_device("cuda:1") == "cuda:1"
    assert runtime.pick_device("auto") in ("cpu", "mps", "cuda:0")
    for bad in ("gpu", "cuda:x", ""):
        assert not runtime.valid_device(bad) or bad == ""
    with pytest.raises(ValueError):
        runtime.pick_device("gpu")
    monkeypatch.setitem(sys.modules, "torch", None)  # torch missing -> cpu
    assert runtime.pick_device("auto") == "cpu"
    assert runtime.hwaccel_args("none") == [] and runtime.hwaccel_args("cuda") == ["-hwaccel", "cuda"]
    with pytest.raises(ValueError):
        runtime.hwaccel_args("bogus")


@needs_ffmpeg
def test_ffmpeg_safe_input_blocks_playlists_and_run_ffmpeg_falls_back(tmp_path):
    playlist = "#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n"
    (tmp_path / "evil.m3u8").write_text(playlist); (tmp_path / "evil2.mp4").write_text("ffconcat version 1.0\nfile /etc/hosts\n")
    for f in ("evil.m3u8", "evil2.mp4"):
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            runtime.ffprobe_json(tmp_path / f, "-show_entries", "format=format_name")
    clip = _video(tmp_path / "c.mp4", 1.0)
    assert "mov" in runtime.ffprobe_json(clip, "-show_entries", "format=format_name")["format"]["format_name"]
    # a hwaccel that cannot work here (cuda on macOS / without NVIDIA) must be retried in software
    r = runtime.run_ffmpeg(["-i", str(clip), "-frames:v", "1", str(tmp_path / "f.jpg")], hwaccel="cuda" if sys.platform == "darwin" else "none")
    assert r.returncode == 0 and (tmp_path / "f.jpg").stat().st_size > 0
    with pytest.raises(RuntimeError, match=r"(?s)ffmpeg failed .*No such file"):
        runtime.run_ffmpeg(["-i", str(tmp_path / "nope.mp4"), str(tmp_path / "g.jpg")], hwaccel="none")
    assert runtime._guard_inputs(["-i", "a", "-i", "b"]).count("-protocol_whitelist") == 2


class _FakeResp:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n: int) -> bytes:
        out, self.data = self.data[:n], self.data[n:]
        return out


def test_fetch_models_verifies_sha256_without_network(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "models"))
    good = b"weights" * 1000
    monkeypatch.setitem(runtime.MODEL_FILES, "toy.bin", ("https://example.invalid/toy.bin", hashlib.sha256(good).hexdigest()))
    assert runtime.fetch_models(["toy.bin"], opener=lambda url, timeout: _FakeResp(good), log=lambda m: None) == {"toy.bin": "downloaded"}
    assert runtime.fetch_models(["toy.bin"], opener=None, log=lambda m: None) == {"toy.bin": "ok"}  # no download when present
    (tmp_path / "models/toy.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="exists with sha256"):
        runtime.fetch_models(["toy.bin"], log=lambda m: None)
    with pytest.raises(RuntimeError, match="mismatch"):
        runtime.fetch_models(["toy.bin"], force=True, opener=lambda url, timeout: _FakeResp(b"evil"), log=lambda m: None)
    assert (tmp_path / "models/toy.bin").read_bytes() == b"tampered" and len(list((tmp_path / "models").iterdir())) == 1
    with pytest.raises(ValueError):
        runtime.fetch_models(["nope"])


def test_fetch_models_groups_and_hf_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "models"))
    rel = runtime._DEPTH_RELATIVE; d = runtime.hf_weights_dir(*rel)
    assert d == tmp_path / "models" / "depth-anything--Depth-Anything-V2-Small-hf@5426e4f0f36572d16453bbda7a8389317b1bef99"
    assert {f"{d.name}/{f}" for f in ("config.json", "preprocessor_config.json", "model.safetensors")} <= set(runtime.MODEL_GROUPS["default"])
    assert not set(runtime.MODEL_GROUPS["depth-metric-indoor"]) & set(runtime.MODEL_GROUPS["default"])
    key = f"{d.name}/config.json"; body = b'{"model_type": "depth_anything"}'
    import hashlib
    monkeypatch.setitem(runtime.MODEL_FILES, key, (runtime.MODEL_FILES[key][0], hashlib.sha256(body).hexdigest()))
    urls = []
    assert runtime.fetch_models([key], opener=lambda url, timeout: (urls.append(url), _FakeResp(body))[1], log=lambda m: None) == {key: "downloaded"}
    assert (d / "config.json").read_bytes() == body and urls == [f"https://huggingface.co/{rel[0]}/resolve/{rel[1]}/config.json"]


def test_depth_mono_models_are_pinned_for_fetch_models():
    pytest.importorskip("cv2")
    dm = pytest.importorskip("duet.playground.depth_mono")
    if not hasattr(dm, "weights_dir"):
        pytest.skip("depth_mono without pre-fetch support")
    for repo, rev, _kind in dm.MODELS.values():
        assert set(runtime.HF_SNAPSHOTS[(repo, rev)]) == set(dm.WEIGHT_FILES)
        assert dm.weights_dir(repo, rev) == runtime.hf_weights_dir(repo, rev)


def test_fetch_models_bakes_world_vocab_once(tmp_path, monkeypatch, cli, capsys):
    import types
    monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path)); calls = []
    fake = types.SimpleNamespace(world_vocab_weights=lambda: "yolov8s-worldv2-vocab-abc.pt",
                                 bake_world_vocab=lambda: (calls.append(1), (tmp_path / "yolov8s-worldv2-vocab-abc.pt").write_bytes(b"b"))[0] or
                                 tmp_path / "yolov8s-worldv2-vocab-abc.pt")
    assert cli.bake_world_vocab(None, False, fake) == "not requested"  # base weights not fetched
    (tmp_path / "yolov8s-worldv2.pt").write_bytes(b"w")
    assert cli.bake_world_vocab(["depth-metric-indoor"], False, fake) == "not requested"
    monkeypatch.setattr(cli, "_clip_available", lambda: False)
    assert cli.bake_world_vocab(None, False, fake) == "no clip" and "CLIP is not importable" in capsys.readouterr().err and not calls
    monkeypatch.setattr(cli, "_clip_available", lambda: True)
    assert cli.bake_world_vocab(None, False, fake) == "baked" and calls == [1]
    assert cli.bake_world_vocab(["yolov8s-worldv2.pt"], False, fake) == "ok" and calls == [1]  # already baked
    assert cli.bake_world_vocab(None, True, fake) == "baked" and calls == [1, 1]  # --force re-bakes


def test_model_pins_match_committed_task_files():
    committed = sorted((REPO / "data/playground/models").glob("*.task"))
    if not committed:
        pytest.skip("no committed .task files")
    for p in committed:  # read-only hashing
        assert runtime.file_sha256(p) == runtime.MODEL_FILES[p.name][1]
    assert {"yolov8n-pose.pt", "yolov8s-worldv2.pt"} <= set(runtime.MODEL_FILES)


# ============================================================================================ orchestration (stub stages)

class Stubs:
    """Stub pipeline mirroring the real DEPS shape; counts calls; configurable failures."""

    def __init__(self):
        self.calls: list[str] = []; self.fail: set[str] = set(); self.skip: set[str] = set()

    def _stage(self, name: str, out: str | None):
        def fn(ep: Episode) -> None:
            self.calls.append(name); ep.set_status(name, "running")
            if out:
                d = ep.derived / out; d.mkdir(parents=True, exist_ok=True); (d / "k.txt").write_text(f"{ep.proc_fps}")
            if name in self.fail:
                raise RuntimeError(f"{name} exploded in {ep.dir}/streams")
            if name in self.skip:
                ep.set_status(name, "skipped", "nothing to do"); return
            if name == "align":
                ep.common_start_s, ep.common_end_s = 0.0, 2.0
                for s in ep.streams:
                    s.offset_status = "reference" if s.name == ep.reference else "audio"
            ep.set_status(name, "done", f"{name} ok")
        fn.__qualname__ = f"stub_{name}"
        return fn

    def specs(self) -> dict[str, R.StageSpec]:
        S = R.StageSpec
        return {"probe": S(self._stage("probe", None), config=("files",), reads_sources=True),
                "align": S(self._stage("align", None), deps=("probe",), config=("streams", "align"), reads_sources=True),
                "frames": S(self._stage("frames", "frames"), deps=("align",), outputs=("frames",), config=("streams", "timeline"), reads_sources=True),
                "body2d": S(self._stage("body2d", "body2d"), deps=("frames",), outputs=("body2d",), config=("streams", "timeline")),
                "hands": S(self._stage("hands", "hands"), deps=("frames",), outputs=("hands",), config=("streams", "timeline", "persons")),
                "body3d": S(self._stage("body3d", "body3d"), deps=("frames", "body2d"), outputs=("body3d",), config=("streams", "timeline")),
                "scan": S(self._stage("scan", "scan"), deps=("frames",), outputs=("scan",), opt_in=True),
                "qc": S(self._stage("qc", "qc"), deps=("frames",), uses=("body2d", "hands"), outputs=("qc",), config=("streams", "timeline")),
                "export": S(self._stage("export", None), deps=("frames",), uses=("body2d", "hands", "body3d", "qc"),
                            outputs=("records.parquet",), config=("streams", "timeline"))}


def _run(ep, stubs, stages=None, **kw):
    return R.run_stages(ep, stages, log=lambda m: None, specs=stubs.specs(), **kw)


def test_registry_is_lazy_and_consistent():
    code = ("import sys; import duet.playground.run as r, duet.playground.episode as e; [r._source_hash(v.func) for v in r.SPECS.values()];"
            "print(int('cv2' in sys.modules), int('torch' in sys.modules), int('fastapi' in sys.modules))")  # code hashes: static
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)}, check=True)
    assert out.stdout.split() == ["0", "0", "0"]
    assert tuple(R.SPECS) == STAGES and R.DEFAULT_ORDER == ["probe", "align", "frames", "calib", "stereo_depth", "body2d", "hands", "objects", "contact",
                                                             "tags", "headpose", "body3d", "world3d", "track", "gaze_proxy", "speech", "qc", "imu_arm",
                                                             "annotate", "autolabel", "metrics", "export"]
    assert set(R.ALL_STAGES) - set(R.DEFAULT_ORDER) == {"depth_mono", "scene_scan"}
    for st, deps in R.DEPS.items():
        assert all(R.ALL_STAGES.index(d) < R.ALL_STAGES.index(st) for d in (*deps, *R.USES[st]))
    assert all(isinstance(f, str) and ":" in f for f in R.REGISTRY.values())
    assert R.STAGE_OUTPUTS["export"] == ("derived/records.parquet",) and R.STAGE_OUTPUTS["probe"] == ()
    with pytest.raises(ValueError, match="unknown stage"):
        R.parse_stages("probe,bogus")


def test_fingerprint_cache_cascade_and_stale_marking(tmp_path):
    ep = _mk(tmp_path); st = Stubs()
    res = _run(ep, st)
    assert set(res.values()) == {"done"} and "scan" not in res and st.calls == ["probe", "align", "frames", "body2d", "hands", "body3d", "qc", "export"]
    assert all(ep.status[s].get("fingerprint") and ep.status[s].get("inputs") for s in res)
    st.calls.clear(); assert set(_run(ep, st).values()) == {"cached"} and st.calls == []
    # proc_fps change: probe stays cached, align (drift check) and everything downstream re-run
    ep.update(lambda e: setattr(e, "proc_fps", 5.0)); st.calls.clear()
    res = _run(ep, st)
    assert res["probe"] == "cached" and st.calls == ["align", "frames", "body2d", "hands", "body3d", "qc", "export"]
    # person label only matters to hands (and its dependents via fingerprints)
    ep.update(lambda e: setattr(e.stream("s0"), "person", "alice")); st.calls.clear()
    _run(ep, st); assert st.calls == ["hands", "qc", "export"]
    # partial run: re-running align with a new override marks non-selected downstream stages stale
    ep.update(lambda e: setattr(e.stream("s1"), "offset_override_s", 0.5)); st.calls.clear()
    res = _run(ep, st, ["align"])
    s = Episode.load(ep.dir).status; down = ("frames", "body2d", "hands", "body3d", "qc", "export")
    assert res == {"align": "done"} and st.calls == ["align"] and s["probe"]["state"] == "done"
    assert {k: s[k]["state"] for k in down} == dict.fromkeys(down, "stale")
    assert "align changed" in s["frames"]["detail"] and "frames changed" in s["body2d"]["detail"]
    st.calls.clear(); res = _run(ep, st, ["qc"])  # qc cannot run on stale frames: its record and files are kept, still "stale"
    assert res == {"qc": "kept"} and ep.status["qc"]["state"] == "stale" and (ep.derived / "qc").exists() and st.calls == []
    st.calls.clear(); res = _run(ep, st)
    assert st.calls == ["frames", "body2d", "hands", "body3d", "qc", "export"] and set(res.values()) <= {"cached", "done"}
    # an edit of the timeline outside align (e.g. a hand-edited window) is detected as well
    ep.update(lambda e: setattr(e, "common_end_s", 1.5))
    assert R.refresh_stale(ep, st.specs()) == list(down)
    assert "timeline changed" in Episode.load(ep.dir).status["frames"]["detail"]
    st.calls.clear(); _run(ep, st); assert st.calls == list(down)
    # --force re-runs only the selected stages when the inputs did not change
    st.calls.clear(); _run(ep, st, ["frames"], force=True); assert st.calls == ["frames"]
    st.calls.clear(); _run(ep, st); assert st.calls == []


def test_imu_offset_override_invalidates_imu_stages_only(tmp_path, cli, monkeypatch):
    imu = _dummy(tmp_path / "m/imu.parquet", b"I")
    ep = Episode.create(tmp_path / "eps", "I", [("ego", "ego", _dummy(tmp_path / "m/e.mp4"), None)], {"ego": imu})
    st = Stubs(); specs = st.specs()
    specs["imu_arm"] = R.StageSpec(st._stage("imu_arm", "imu_arm"), deps=("align",), uses=("frames", "qc"), outputs=("imu_arm",),
                                   config=("streams", "timeline", "imu"))
    R.run_stages(ep, None, log=lambda m: None, specs=specs)
    monkeypatch.setattr(cli, "run_stages", lambda e, *a, configure=None, **k: (configure and e.update(configure)) and {})  # config only
    assert cli.main(["--root", str(tmp_path / "eps"), "run", "I", "--imu-offset", "ego=-0.42"]) == 0
    assert Episode.load(ep.dir).stream("ego").imu_offset_override_s == -0.42
    st.calls.clear(); R.run_stages(ep, None, log=lambda m: None, specs=specs)
    assert st.calls == ["imu_arm"]
    with pytest.raises(SystemExit):  # a stream without an IMU cannot take an IMU offset
        cli.main(["--root", str(tmp_path / "eps"), "run", "I", "--imu-offset", "nope=1"])


def test_model_weights_in_fingerprint_missing_file_keeps_results(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYGROUND_MODELS", str(tmp_path / "models")); (tmp_path / "models").mkdir()
    w = tmp_path / "models" / "toy-pose.pt"; w.write_bytes(b"v1")
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs()
    specs["body2d"] = R.StageSpec(st._stage("body2d", "body2d"), deps=("frames",), outputs=("body2d",), models=("toy-pose.pt",))
    run = lambda: (st.calls.clear(), R.run_stages(ep, None, log=lambda m: None, specs=specs))[1]  # noqa: E731
    run(); assert ep.status["body2d"]["models"] == {"toy-pose.pt": R._sha(b"v1")}
    assert run()["body2d"] == "cached"
    w.unlink()  # weights not fetched on this host: results are kept
    assert run()["body2d"] == "cached" and R.refresh_stale(ep, specs) == [] and st.calls == []
    w.write_bytes(b"v1"); assert run()["body2d"] == "cached"  # same content back (new mtime): still cached
    w.write_bytes(b"v2 retrained"); res = run()  # different weights: body2d and its dependents re-run
    assert st.calls == ["body2d", "body3d", "qc", "export"] and res["body2d"] == "done"
    assert R.SPECS["hands"].models == ("hand_landmarker.task",) and R.SPECS["objects"].models[0] == "yolov8s-worldv2.pt"
    assert R._model_name(R.SPECS["objects"].models[1]).startswith("yolov8s-worldv2-vocab-")  # perception's baked-vocab weights
    assert R._model_name("no_such_module_xyz:f") == "no_such_module_xyz:f"  # unresolvable entries do not crash


def test_code_hash_covers_transitive_duet_imports(tmp_path, monkeypatch):
    epi, rot = R._duet_file("duet.playground.episode"), R._duet_file("duet.geometry.rotations")
    for st in R.SPECS:  # every stage's code hash covers episode.py (time mapping) and runtime.py ...
        assert epi in R._source_files(R.SPECS[st].func), st
    assert rot in R._source_files(R.SPECS["world3d"].func)  # ... and shared duet.* helpers such as duet.geometry.rotations
    ep = _mk(tmp_path); st = Stubs(); _run(ep, st); real = R._file_sha
    monkeypatch.setattr(R, "_file_sha", lambda p: "edited" if Path(p) == epi else real(p))
    st.calls.clear(); _run(ep, st)  # e.g. a change to stream_time/ref_time: everything importing episode.py re-runs
    assert st.calls == ["probe", "align", "frames", "body2d", "hands", "body3d", "qc", "export"]
    assert set(R.USES["world3d"]) == {"body2d", "hands", "tags", "headpose"}  # world.py reads neither objects nor depth_mono


def test_code_change_invalidates_via_source_hash(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs(); _run(ep, st)
    mod = tmp_path / "stagemod.py"; mod.write_text("def f(ep):\n    ep.set_status('body2d', 'done', 'v1')\n")
    sys.path.insert(0, str(tmp_path))
    try:
        specs["body2d"] = R.StageSpec("stagemod:f", deps=("frames",), outputs=("body2d",))
        R.run_stages(ep, None, log=lambda m: None, specs=specs)
        fp1 = ep.status["body2d"]["fingerprint"]
        mod.write_text("def f(ep):\n    ep.set_status('body2d', 'done', 'v2')  # edited\n")
        res = R.run_stages(ep, None, log=lambda m: None, specs=specs)
        assert res["body2d"] == "done" and ep.status["body2d"]["fingerprint"] != fp1 and res["body3d"] == "done"
    finally:
        sys.path.remove(str(tmp_path)); sys.modules.pop("stagemod", None)


def test_stage_params_are_passed_and_fingerprinted_and_logs_pruned(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs(); got = []

    def f(e, k=0):
        got.append(k); e.set_status("hands", "done", f"k={k}")
    specs["hands"] = R.StageSpec(f, deps=("frames",), params={"k": 1})
    logs = ep.derived / "logs"; logs.mkdir()
    for i in range(60):
        (logs / f"old{i:02d}.log").write_text("x"); os.utime(logs / f"old{i:02d}.log", (1000 + i, 1000 + i))
    R.run_stages(ep, None, log=lambda m: None, specs=specs)
    specs["hands"] = R.StageSpec(f, deps=("frames",), params={"k": 2})
    res = R.run_stages(ep, None, log=lambda m: None, specs=specs)
    assert got == [1, 2] and res["hands"] == "done" and res["qc"] == "done" and res["body2d"] == "cached"
    assert len(list(logs.glob("*.log"))) == 50 and not (logs / "old00.log").exists()


def test_input_file_identity_invalidates_probe(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); _run(ep, st)
    src = tmp_path / "media" / "E_s0.mp4"; src.write_bytes(b"a different recording")
    st.calls.clear(); res = _run(ep, st)
    assert st.calls[:2] == ["probe", "align"] and res["probe"] == "done"
    (ep.dir / "rig.json").write_text('{"tag_size_m": 0.08}')  # rig is not in these stubs' config: nothing re-runs
    st.calls.clear(); _run(ep, st); assert st.calls == []
    # a temporarily unavailable video (unmounted drive, dangling link) keeps the results instead of invalidating them
    src.rename(tmp_path / "media" / "away.mp4")
    st.calls.clear(); res = _run(ep, st)
    assert st.calls == [] and set(res.values()) == {"cached"} and R.refresh_stale(ep, st.specs()) == []
    (tmp_path / "media" / "away.mp4").rename(src); st.calls.clear(); _run(ep, st); assert st.calls == []
    src.unlink()  # even a forced probe is not run without its video: nothing is rewritten, nothing goes stale
    res = _run(ep, st, ["probe"], force=True); s = Episode.load(ep.dir).status
    assert res == {"probe": "kept"} and st.calls == [] and s["probe"]["state"] == s["align"]["state"] == s["export"]["state"] == "done"


def test_deps_skip_output_clearing_and_failures(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); st.fail = {"body2d"}
    res = _run(ep, st)
    assert res["body2d"] == "failed" and res["body3d"] == "skipped" and res["hands"] == "done" and res["qc"] == "done"
    s = Episode.load(ep.dir).status
    assert s["body3d"]["detail"] == "needs body2d (failed)" and s["body3d"]["finished"]
    assert not (ep.derived / "body2d").exists() and not (ep.derived / "body3d").exists()  # failed/skipped leave nothing
    assert len(s["body2d"]["detail"]) <= 2000 and str(tmp_path) not in s["body2d"]["detail"] and "log: derived/logs/" in s["body2d"]["detail"]
    log = (ep.dir / s["body2d"]["log"]).read_text()
    assert "Traceback" in log and "exploded" in log
    # a stage that skips itself leaves no output either
    st.fail.clear(); st.skip = {"hands"}; res = _run(ep, st, ["body2d", "hands"], force=True)
    assert res == {"body2d": "done", "hands": "skipped"} and not (ep.derived / "hands").exists() and (ep.derived / "body2d").exists()
    # stale artifacts from an older run are removed before a stage runs
    (ep.derived / "qc" / "old_stream.npz").write_bytes(b"old")
    st.skip.clear(); _run(ep, st, ["qc"], force=True); assert not (ep.derived / "qc" / "old_stream.npz").exists()
    # stages before their prerequisites: nothing runs on zero data
    ep2 = _mk(tmp_path, "F"); st2 = Stubs()
    assert _run(ep2, st2, ["align", "frames"]) == {"align": "skipped", "frames": "skipped"} and st2.calls == []
    assert Episode.load(ep2.dir).status["align"]["detail"] == "needs probe (not run)"
    # opt-in stages only when named
    assert _run(ep2, st2, ["probe", "align", "frames", "scan"])["scan"] == "done"


def test_stage_returning_without_final_state_and_interrupt(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs()
    specs["hands"] = R.StageSpec(lambda e: e.set_status("hands", "running"), deps=("frames",), outputs=("hands",))

    def interrupted(e):
        e.set_status("qc", "running"); (e.derived / "qc").mkdir(); raise KeyboardInterrupt
    specs["qc"] = R.StageSpec(interrupted, deps=("frames",), outputs=("qc",))
    with pytest.raises(KeyboardInterrupt):
        R.run_stages(ep, None, log=lambda m: None, specs=specs)
    s = Episode.load(ep.dir).status
    assert s["hands"]["state"] == "failed" and "without recording" in s["hands"]["detail"]
    assert s["qc"]["state"] == "interrupted" and s["qc"]["finished"] and not (ep.derived / "qc").exists()
    assert not R.run_lock_held(ep.dir)


def test_legacy_status_without_fingerprints(tmp_path):
    ep = _mk(tmp_path); st = Stubs()
    legacy = {k: {"state": "done", "detail": "old", "started": 1.0, "finished": 2.0} for k in ("probe", "align", "frames", "body2d", "qc", "export")}
    legacy["hands"] = {"state": "skipped", "detail": "old"}
    ep.update(lambda e: (e.status.update(legacy), setattr(e, "common_end_s", 2.0)))
    assert R.refresh_stale(ep, st.specs()) == [] and Episode.load(ep.dir).stage_ok("export")  # old outputs stay usable
    res = _run(ep, st, ["export"])  # re-export on old outputs works
    assert res == {"export": "done"} and st.calls == ["export"] and ep.stage_ok("qc")
    st.calls.clear(); _run(ep, st, ["body2d"])  # an upstream re-ran: legacy dependents become stale
    s = Episode.load(ep.dir).status
    assert s["qc"]["state"] == "stale" and s["export"]["state"] == "stale" and s["frames"]["state"] == "done"


def test_drop_frames_keeps_downstream_cached(tmp_path):
    ep = _mk(tmp_path); st = Stubs(); _run(ep, st)
    for s in ep.streams:
        d = ep.derived / "frames" / s.name; d.mkdir(parents=True, exist_ok=True); (d / "000001.jpg").write_bytes(b"j")
    st.calls.clear(); res = _run(ep, st, drop_frames=True)
    s = Episode.load(ep.dir).status["frames"]
    assert set(res.values()) == {"cached"} and s["state"] == "stale" and s["dropped"] and s["fingerprint"]
    assert not list((ep.derived / "frames").glob("*/*.jpg")) and (ep.derived / "frames" / "k.txt").exists()
    assert R.refresh_stale(ep, st.specs()) == []  # body2d/qc/export stay valid
    st.calls.clear(); res = _run(ep, st, ["export"]); assert res == {"export": "cached"}
    # forcing a stage that needs the dropped JPEGs cannot run it, and must not destroy its still-valid result
    st.calls.clear(); res = _run(ep, st, ["body2d"], force=True)
    assert res == {"body2d": "kept"} and st.calls == [] and ep.stage_ok("body2d") and (ep.derived / "body2d/k.txt").exists()
    st.calls.clear(); res = _run(ep, st)  # full run: only frames is re-extracted
    assert st.calls == ["frames"] and res["body2d"] == res["export"] == "cached"


def test_run_lock_blocks_second_pipeline(tmp_path):
    ep = _mk(tmp_path); st = Stubs()
    with run_lock(ep.dir):
        assert R.run_lock_held(ep.dir) and R.is_running(ep.dir)
        with pytest.raises(BlockingIOError):
            _run(ep, st)
    assert not R.run_lock_held(ep.dir) and not R.is_running(ep.dir)


# ============================================================================================ job queue (subprocesses)

STUB_MODULE = textwrap.dedent('''
    import os, signal, time
    from duet.playground.run import StageSpec

    def probe(ep):
        ep.set_status("probe", "running"); ep.set_status("probe", "done", "stub")

    def slow(ep):
        ep.set_status("slow", "running")
        for _ in range(1200):
            if (ep.dir / "go").exists():
                break
            time.sleep(0.05)
        ep.set_status("slow", "done", "slow ok")

    def exit9(ep):
        ep.set_status("exit9", "running"); os._exit(9)

    def sigkill(ep):
        ep.set_status("sigkill", "running"); os.kill(os.getpid(), signal.SIGKILL)

    def stubborn(ep):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # ignores the cancel: the worker must escalate to SIGKILL
        ep.set_status("stubborn", "running")
        time.sleep(600)

    SLOW = {"probe": StageSpec(probe), "slow": StageSpec(slow, deps=("probe",))}
    STUBBORN = {"probe": StageSpec(probe), "stubborn": StageSpec(stubborn, deps=("probe",))}
    CRASH = {"probe": StageSpec(probe), "exit9": StageSpec(exit9, deps=("probe",))}
    KILL = {"probe": StageSpec(probe), "sigkill": StageSpec(sigkill, deps=("probe",))}
''')


@pytest.fixture
def stub_jobs(tmp_path, monkeypatch):
    (tmp_path / "pg_stub_stages.py").write_text(STUB_MODULE)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(SRC)]))

    def use(attr: str) -> None:
        monkeypatch.setattr(R, "JOB_EXTRA_ARGS", ["--specs", f"pg_stub_stages:{attr}"])
    return use


def _wait(pred, timeout: float = 60.0) -> None:
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.05)


def test_submit_dedupes_under_concurrency_and_runs_in_subprocess(tmp_path, stub_jobs):
    stub_jobs("SLOW"); ep = _mk(tmp_path); results: list[bool] = []; barrier = threading.Barrier(16)

    def go():
        barrier.wait(); results.append(R.submit(ep.dir))
    ts = [threading.Thread(target=go) for _ in range(16)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results.count(True) == 1 and R.is_running(ep.dir)
    _wait(lambda: (Episode.load(ep.dir).status.get("slow") or {}).get("state") == "running")
    q = R.queue_state()
    job = next(j for j in q["running"] if j["episode"] == "E")
    assert job["pid"] and job["pid"] != os.getpid() and q["max_jobs"] >= 1 and not R.submit(ep.dir)
    assert Episode.load(ep.dir).status["slow"]["pid"] == job["pid"]  # the stage ran in the job's process
    (ep.dir / "go").write_text("1")
    _wait(lambda: not R.is_running(ep.dir))
    s = Episode.load(ep.dir).status
    assert s["slow"]["state"] == "done" and s["probe"]["state"] == "done" and (ep.dir / s["slow"]["log"]).exists()


@pytest.mark.parametrize("attr,stage,why", [("CRASH", "exit9", "exited with code 9"), ("KILL", "sigkill", "killed by SIGKILL")])
def test_crashing_stage_does_not_kill_the_caller(tmp_path, stub_jobs, attr, stage, why):
    stub_jobs(attr); ep = _mk(tmp_path)
    assert R.submit(ep.dir)
    _wait(lambda: not R.is_running(ep.dir))
    st = Episode.load(ep.dir).status[stage]
    assert st["state"] == "failed" and why in st["detail"] and st["finished"]
    assert R.submit(ep.dir)  # the queue keeps working after a crash
    _wait(lambda: not R.is_running(ep.dir))


def test_cancel_running_and_queued_jobs(tmp_path, stub_jobs):
    stub_jobs("SLOW"); a = _mk(tmp_path, "A"); b = _mk(tmp_path, "B")
    assert R.submit(a.dir) and R.submit(b.dir)
    _wait(lambda: (Episode.load(a.dir).status.get("slow") or {}).get("state") == "running")
    if int(os.environ.get("PLAYGROUND_MAX_JOBS", "1")) == 1:
        assert [j["episode"] for j in R.queue_state()["queued"]] == ["B"]
        assert R.cancel(b.dir) and not R.is_running(b.dir)
    assert R.cancel(a.dir)
    _wait(lambda: not R.is_running(a.dir))
    st = Episode.load(a.dir).status["slow"]
    assert st["state"] == "interrupted" and not R.cancel(a.dir)
    (b.dir / "go").write_text("1"); _wait(lambda: not R.is_running(b.dir))


def test_recover_stale(tmp_path):
    host = socket.gethostname()
    dead = subprocess.Popen([sys.executable, "-c", "pass"]); dead.wait()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])  # stands in for a REUSED pid
    try:
        eps = {n: _mk(tmp_path, n, n=1) for n in ("dead", "pidreuse", "other", "legacy", "locked")}
        runs = {"dead": {"pid": dead.pid, "host": host}, "pidreuse": {"pid": alive.pid, "host": host}, "other": {"pid": dead.pid, "host": "elsewhere"},
                "legacy": {}, "locked": {"pid": dead.pid, "host": host}}
        for n, extra in runs.items():
            eps[n].update(lambda e, extra=extra: e.status.update({"frames": {"state": "running", "started": 1.0, **extra}, "probe": {"state": "done"},
                                                                  "_job": {"state": "running", "pid": extra.get("pid"), "host": extra.get("host")}}))
        with run_lock(eps["locked"].dir):
            out = R.recover_stale(tmp_path / "eps")
        assert sorted(out) == [(n, k) for n in ("dead", "legacy", "pidreuse") for k in ("_job", "frames")]
        states = {n: Episode.load(e.dir).status["frames"]["state"] for n, e in eps.items()}
        # liveness = the run lock, not pid + host: a live process with a recorded pid proves nothing (pid reuse)
        assert states == {"dead": "interrupted", "pidreuse": "interrupted", "other": "running", "legacy": "interrupted", "locked": "running"}
        assert "gone" in Episode.load(eps["dead"].dir).status["frames"]["detail"]
        assert Episode.load(eps["dead"].dir).status["probe"]["state"] == "done"
    finally:
        alive.kill(); alive.wait()


def test_submit_rejects_unknown_episode_and_stage(tmp_path):
    with pytest.raises(EpisodeError):
        R.submit(tmp_path / "nope")
    ep = _mk(tmp_path)
    with pytest.raises(ValueError):
        R.submit(ep.dir, ["bogus"])
    assert R.run_in_background is R.submit


# ============================================================================================ CLI

@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("playground_cli", REPO / "scripts" / "playground.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_cli_parses_colon_paths_and_persons(tmp_path, cli):
    p = _dummy(tmp_path / "rec" / "leader_2026-09-25T10:32:11.mp4")
    assert cli.parse_stream(f"leader={p}", "ego") == ("leader", "ego", p, None)
    assert cli.parse_stream(f"leader={p}:alice", "ego") == ("leader", "ego", p, "alice")
    plain = _dummy(tmp_path / "rec" / "helper.mp4")
    assert cli.parse_stream(f"helper={plain}:bob", "ego") == ("helper", "ego", plain, "bob")
    with pytest.raises(Exception, match="no such file"):
        cli.parse_stream(f"x={tmp_path}/missing.mp4:bob", "ego")
    with pytest.raises(Exception, match="NAME=VALUE"):
        cli.parse_stream("nopath", "exo")
    root = tmp_path / "eps"
    rc = cli.main(["--root", str(root), "create", "E", "--ego", f"leader={p}", "--exo", f"front={plain}", "--person", "leader=alice",
                   "--offset", "front=1.5", "--fps", "5", "--max-lag", "30", "--device", "cpu", "--hwaccel", "none"])
    ep = Episode.load(root / "E")
    assert rc == 0 and ep.stream("leader").person == "alice" and ep.stream("front").offset_override_s == 1.5
    assert (ep.proc_fps, ep.max_lag_s, ep.device, ep.hwaccel) == (5.0, 30.0, "cpu", "none")
    assert os.readlink(ep.dir / "streams/leader.mp4").endswith("leader_2026-09-25T10:32:11.mp4")
    assert cli.main(["--root", str(root), "create", "E", "--ego", f"leader={p}"]) == 2  # exists


@pytest.mark.parametrize("argv", [["create", "E", "--ego", "a=x", "--fps", "0"], ["create", "E", "--ego", "a=x", "--fps", "61"],
                                  ["create", "E", "--ego", "a=x", "--fps", "nan"], ["create", "E", "--ego", "a=x", "--max-lag", "-1"],
                                  ["create", "E", "--ego", "a=x", "--device", "gpu"], ["create", "E", "--ego", "a=x", "--hwaccel", "magic"],
                                  ["run", "E", "--stages", "probe,bogus"], ["create", "E", "--ego", "a=x", "--copy", "--move"]])
def test_cli_rejects_bad_arguments(cli, argv):
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(argv)
    assert e.value.code == 2


def test_cli_run_applies_config_and_validates(tmp_path, cli, monkeypatch, capsys):
    root = tmp_path / "eps"; ep = _mk(tmp_path)
    seen = {}
    def fake(e, stages, force, drop_frames=False, configure=None):
        seen.update(stages=stages, force=force, drop=drop_frames)
        if configure:
            e.update(configure)
        return {}
    monkeypatch.setattr(cli, "run_stages", fake)
    assert cli.main(["--root", str(root), "run", "E", "--stages", "export,probe", "--force", "--drop-frames", "--offset", "s1=2.0", "--fps", "4"]) == 0
    assert seen == {"stages": ["export", "probe"], "force": True, "drop": True}
    e = Episode.load(ep.dir); assert e.stream("s1").offset_override_s == 2.0 and e.proc_fps == 4.0
    assert cli.main(["--root", str(root), "run", "E", "--offset", "s1=none"]) == 0 and Episode.load(ep.dir).stream("s1").offset_override_s is None
    with pytest.raises(SystemExit):
        cli.main(["--root", str(root), "run", "E", "--offset", "zz=1"])
    assert cli.main(["--root", str(root), "run", "../etc"]) == 2


def test_cli_serve_refuses_public_bind_without_token(cli, monkeypatch, capsys):
    monkeypatch.delenv("PLAYGROUND_TOKEN", raising=False)
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2 and "PLAYGROUND_TOKEN" in capsys.readouterr().err
    monkeypatch.setenv("PLAYGROUND_TOKEN", "short")
    assert cli.main(["serve", "--host", "192.168.1.5"]) == 2
    monkeypatch.setenv("PLAYGROUND_TOKEN", "  short-token-12  \n")  # 18 chars raw, 14 after stripping (the server's rule)
    capsys.readouterr()
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2 and "whitespace" in capsys.readouterr().err
    assert cli._is_loopback("127.0.0.1") and cli._is_loopback("::1") and cli._is_loopback("localhost") and not cli._is_loopback("0.0.0.0")


# ============================================================================================ code-review fixes

def test_interrupted_rerun_leaves_no_stale_done_and_stage_fresh(tmp_path):
    """#1: a cancelled/killed re-run (e.g. --fps 10 -> 5, cancel during frames) must not leave downstream "done"."""
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs(); _run(ep, st)
    ep.update(lambda e: setattr(e, "proc_fps", 5.0))

    def killed(e):
        e.set_status("frames", "running"); raise KeyboardInterrupt
    specs["frames"] = R.StageSpec(killed, deps=("align",), outputs=("frames",), config=("streams", "timeline"))
    with pytest.raises(KeyboardInterrupt):
        R.run_stages(ep, None, log=lambda m: None, specs=specs)
    s = Episode.load(ep.dir).status
    assert s["frames"]["state"] == "interrupted"
    assert {k: s[k]["state"] for k in ("body2d", "hands", "body3d", "qc", "export")} == dict.fromkeys(("body2d", "hands", "body3d", "qc", "export"), "stale")
    # a change made OUTSIDE any run (rig.json): stage_ok still says done, stage_fresh does not, and nothing is written
    ep2 = _mk(tmp_path, "F"); st2 = Stubs(); sp2 = st2.specs()
    sp2["qc"] = R.StageSpec(st2._stage("qc", "qc"), deps=("frames",), uses=("body2d", "hands"), outputs=("qc",), config=("streams", "timeline", "rig"))
    R.run_stages(ep2, None, log=lambda m: None, specs=sp2)
    assert R.stage_fresh(ep2, "qc", sp2) and R.fresh_stages(ep2, sp2) >= {"probe", "frames", "qc", "export"}
    (ep2.dir / "rig.json").write_text('{"tag_size_m": 0.1}'); before = (ep2.dir / "episode.json").read_bytes()
    e2 = Episode.load(ep2.dir)
    assert e2.stage_ok("qc") and not R.stage_fresh(e2, "qc", sp2) and "export" not in R.fresh_stages(e2, sp2)  # cascades via qc
    assert R.stale_stages(e2, sp2) == {"qc": "rig changed since this ran", "export": "qc changed since this ran"}
    assert (ep2.dir / "episode.json").read_bytes() == before


def _legacy_copy(tmp_path: Path, src: Path) -> Episode:
    """A committed-format (legacy) episode in tmp_path: its episode.json, DANGLING stream links, dummy derived files."""
    d = tmp_path / "eps" / src.parent.name; (d / "streams").mkdir(parents=True); shutil.copy(src, d / "episode.json")
    ep = Episode.load(d)
    for s in ep.streams:
        os.symlink("/nonexistent/teammate/home/" + Path(s.path).name, d / s.path)
    for st in ("frames", "body2d", "hands", "objects", "body3d", "qc", "imu_arm"):
        _dummy(d / "derived" / st / "x.npz")
    return ep


def test_legacy_episode_with_dangling_links_keeps_everything(tmp_path):
    """#2: a Run-all on a committed episode whose stream links dangle must not hide (rewrite) any result."""
    src = next((p for p in COMMITTED if p.parent.name == "eidon_10004"), None)
    if src is None:
        pytest.skip("committed eidon_10004 episode not present")
    ep = _legacy_copy(tmp_path, src); before = {k: dict(v) for k, v in ep.status.items()}; calls = []

    def recorder(name):
        def fn(e):
            calls.append(name); e.set_status(name, "skipped", "no board")  # like calib/tags/headpose without a rig
        return fn
    import dataclasses
    specs = {k: dataclasses.replace(v, func=recorder(k)) for k, v in R.SPECS.items()}
    fresh0 = R.fresh_stages(ep, specs)
    res = R.run_stages(ep, None, log=lambda m: None, specs=specs)
    after = Episode.load(ep.dir).status
    assert R.missing_sources(ep) == ["ego"]
    assert all(after[k] == v for k, v in before.items() if v.get("state") == "done"), "a legacy result was rewritten"
    assert {k for k, v in res.items() if v == "cached"} == {k for k, v in before.items() if v.get("state") == "done"}
    NEW = {"stereo_depth", "track", "contact", "gaze_proxy", "speech", "annotate", "autolabel", "metrics"}  # stages added after this legacy episode was committed
    assert calls[0] == "calib" and set(calls[1:]) <= NEW and {k: after[k]["state"] for k in ("tags", "headpose", "world3d")} == dict.fromkeys(("tags", "headpose", "world3d"), "skipped")
    assert R.fresh_stages(ep, specs) == fresh0 >= {"probe", "align", "frames", "body2d", "hands", "export"}


def test_move_never_loses_originals(tmp_path):
    """#3: move = link/copy + commit + THEN unlink; data/raw refused; symlinks refused; kill before commit is harmless."""
    raw = _dummy(tmp_path / "data" / "raw" / "cam.mp4", b"RAW")
    with pytest.raises(ValueError, match="data/raw"):
        Episode.create(tmp_path / "eps", "R", [("a", "ego", raw, None)], mode="move")
    assert raw.read_bytes() == b"RAW" and Episode.create(tmp_path / "eps", "R", [("a", "ego", raw, None)], mode="copy")
    os.symlink(raw, tmp_path / "link.mp4")
    with pytest.raises(ValueError, match="symlink"):
        Episode.create(tmp_path / "eps", "L", [("a", "ego", tmp_path / "link.mp4", None)], mode="move")
    a = _dummy(tmp_path / "up" / "a.mp4", b"A" * 1000)
    ep = Episode.create(tmp_path / "eps", "M", [("a", "ego", a, None)], mode="move")
    assert not a.exists() and (ep.dir / "streams/a.mp4").read_bytes() == b"A" * 1000 and not (ep.dir / "streams/a.mp4").is_symlink()
    # killed right before the commit (SIGKILL: no finally/except runs): the original is intact, the staging dir is garbage
    b = _dummy(tmp_path / "up" / "b.mp4", b"B" * 1000)
    code = textwrap.dedent(f"""
        import os, signal, sys; sys.path.insert(0, {str(SRC)!r})
        from duet.playground import episode as E
        E.os.rename = lambda *a: os.kill(os.getpid(), signal.SIGKILL)
        E.Episode.create({str(tmp_path / 'eps')!r}, "K", [("b", "ego", {str(b)!r}, None)], mode="move")
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == -signal.SIGKILL
    staged = [p for p in (tmp_path / "eps/.staging").iterdir() if p.is_dir()]
    assert b.read_bytes() == b"B" * 1000 and not (tmp_path / "eps/K").exists() and len(staged) == 1
    from duet.playground.episode import _clean_staging
    assert _clean_staging(tmp_path / "eps/.staging", min_age_s=0) == [] and b.read_bytes() == b"B" * 1000
    assert not [p for p in (tmp_path / "eps/.staging").iterdir()]
    # defence in depth: a staged file whose original vanished is restored, never deleted
    d = tmp_path / "eps/.staging/X-deadbeef"; _dummy(d / "streams/x.mp4", b"ONLY COPY")
    runtime.atomic_write_json(d.with_name(d.name + ".json"), {"items": [[str(tmp_path / "gone/x.mp4"), "streams/x.mp4"]]})
    assert _clean_staging(tmp_path / "eps/.staging", min_age_s=0) == [str(tmp_path / "gone/x.mp4")]
    assert (tmp_path / "gone/x.mp4").read_bytes() == b"ONLY COPY"


def test_clean_staging_skips_an_active_create(tmp_path):
    from duet.playground.episode import _clean_staging, _staging_lock
    tmp = tmp_path / ".staging" / "E-1234"; _dummy(tmp / "streams/a.mp4")
    with _staging_lock(tmp):
        _clean_staging(tmp_path / ".staging", min_age_s=0); assert tmp.exists()
    _clean_staging(tmp_path / ".staging", min_age_s=0); assert not tmp.exists()


def test_overwrite_refuses_to_delete_real_stream_files(tmp_path):
    a = _dummy(tmp_path / "m/a.mp4", b"A"); ep = Episode.create(tmp_path / "eps", "E", [("a", "ego", a, None)], mode="move")
    b = _dummy(tmp_path / "m/b.mp4", b"B")
    with pytest.raises(FileExistsError, match="only"):
        Episode.create(tmp_path / "eps", "E", [("b", "ego", b, None)], overwrite=True)
    assert (ep.dir / "streams/a.mp4").read_bytes() == b"A"  # the moved-in original survived
    Episode.create(tmp_path / "eps", "E", [("b", "ego", b, None)], overwrite=True, discard_old_streams=True)
    assert not (ep.dir / "streams/a.mp4").exists()


def test_prune_logs_only_in_the_episode_log_dir(tmp_path):
    """#4: --log-file elsewhere never deletes unrelated *.log files."""
    ep = _mk(tmp_path); other = tmp_path / "batch_logs"
    for i in range(60):
        _dummy(other / f"other-{i:02d}.log")
    R.run_stages(ep, ["probe"], log=lambda m: None, specs=Stubs().specs(), log_file=other / "mine.log")
    assert len(list(other.glob("other-*.log"))) == 60 and (other / "mine.log").exists()


def test_unverifiable_is_per_stream(tmp_path):
    """#5: a missing stream b must not mask a CHANGED stream a."""
    ep = _mk(tmp_path); st = Stubs(); _run(ep, st)
    (tmp_path / "media/E_s0.mp4").write_bytes(b"a different recording")
    (tmp_path / "media/E_s1.mp4").rename(tmp_path / "media/away.mp4")
    st.calls.clear(); res = _run(ep, st)
    s = Episode.load(ep.dir).status
    assert res["probe"] == "kept" and s["probe"]["state"] == "stale" and "files" in s["probe"]["detail"] and st.calls == []
    assert s["export"]["state"] == "stale" and not ({"probe", "align", "frames", "export"} & R.fresh_stages(ep, st.specs()))


def test_hands_runs_without_body2d(tmp_path):
    """#6: body2d is optional for hands (geometric wearer cues)."""
    assert R.SPECS["hands"].deps == ("frames",) and "body2d" in R.SPECS["hands"].uses
    ep = _mk(tmp_path); st = Stubs(); st.fail = {"body2d"}; specs = st.specs()
    specs["hands"] = R.StageSpec(st._stage("hands", "hands"), deps=("frames",), uses=("body2d",), outputs=("hands",))
    res = R.run_stages(ep, None, log=lambda m: None, specs=specs)
    assert res["body2d"] == "failed" and res["hands"] == "done"


def test_cancel_before_spawn_and_escalation_and_job_record(tmp_path, stub_jobs, monkeypatch):
    """#8: (a) cancel between dequeue and spawn; (b) SIGTERM ignored -> SIGKILL; (c) a job that dies before any stage."""
    ep = _mk(tmp_path)
    job = {"key": R._key(ep.dir), "episode": "E", "stages": None, "force": False, "queued": time.time()}
    with R._QC:
        R._ACTIVE[job["key"]] = job
    try:
        assert R.cancel(ep.dir) and not R.cancel(ep.dir) and job["cancelled"]
        assert R._run_job(job) == -signal.SIGTERM and "pid" not in job  # never spawned
    finally:
        with R._QC:
            R._ACTIVE.pop(job["key"], None)
    assert Episode.load(ep.dir).status["_job"]["state"] == "interrupted"
    stub_jobs("STUBBORN"); monkeypatch.setenv("PLAYGROUND_KILL_GRACE_S", "0.5")
    ep2 = _mk(tmp_path, "S"); assert R.submit(ep2.dir)
    _wait(lambda: (Episode.load(ep2.dir).status.get("stubborn") or {}).get("state") == "running")
    t0 = time.time(); assert R.cancel(ep2.dir)
    _wait(lambda: not R.is_running(ep2.dir), timeout=30)
    s = Episode.load(ep2.dir).status
    assert s["stubborn"]["state"] == "interrupted" and s["_job"]["state"] == "interrupted" and time.time() - t0 < 20
    stub_jobs("SLOW"); ep3 = _mk(tmp_path, "V"); ep3.update(lambda e: setattr(e, "proc_fps", 0.0))  # invalid: fails before any stage
    assert R.submit(ep3.dir); _wait(lambda: not R.is_running(ep3.dir))
    j = Episode.load(ep3.dir).status["_job"]
    assert j["state"] == "failed" and j["returncode"] == 2 and "proc_fps" in j["detail"] and "slow" not in Episode.load(ep3.dir).status


def test_cli_run_config_waits_for_the_run_lock(tmp_path, cli):
    """#10: `run --fps` while another pipeline runs changes nothing."""
    ep = _mk(tmp_path); before = (ep.dir / "episode.json").read_bytes()
    with run_lock(ep.dir):
        assert cli.main(["--root", str(tmp_path / "eps"), "run", "E", "--fps", "5", "--device", "cpu"]) == 2
    assert (ep.dir / "episode.json").read_bytes() == before


def test_zed_flow_prepared_dir_and_zed_sources(tmp_path):
    """#11: export into <ep>/zed/<s>/ first, then create with <ep>/zed/<s>/left.mp4."""
    d = tmp_path / "eps" / "Z"; left = _dummy(d / "zed/leader/left.mp4", b"ZED"); _dummy(d / "zed/leader/pose.csv", b"t")
    (d / "rig.json").write_text("{}"); gp = _dummy(tmp_path / "m/gp.mp4")
    ep = Episode.create(tmp_path / "eps", "Z", [("leader", "ego", left, None), ("front", "exo", gp, None)])
    assert os.readlink(ep.dir / "streams/leader.mp4") == "../zed/leader/left.mp4" and (ep.dir / "streams/leader.mp4").read_bytes() == b"ZED"
    assert (ep.dir / "rig.json").exists() and (ep.dir / "zed/leader/pose.csv").exists()
    with pytest.raises(ValueError, match="zed"):
        Episode.create(tmp_path / "eps", "Z", [("leader", "ego", left, None)], mode="move", overwrite=True)
    Episode.create(tmp_path / "eps", "Z", [("leader", "ego", left, None)], overwrite=True)  # overwrite from zed/: allowed, zed kept
    assert (ep.dir / "zed/leader/pose.csv").exists()


def test_overwrite_moves_stale_zed_export_aside(tmp_path):
    """#12: a new take for stream s must not silently reuse the old take's zed/<s>/."""
    t1, t2 = _dummy(tmp_path / "m/take1.mp4", b"1"), _dummy(tmp_path / "m/take2.mp4", b"22")
    ep = Episode.create(tmp_path / "eps", "E", [("leader", "ego", t1, None)]); _dummy(ep.dir / "zed/leader/pose.csv")
    Episode.create(tmp_path / "eps", "E", [("leader", "ego", t1, None)], overwrite=True)  # same source: kept
    assert (ep.dir / "zed/leader/pose.csv").exists()
    Episode.create(tmp_path / "eps", "E", [("leader", "ego", t2, None)], overwrite=True, keep_zed=True)
    assert (ep.dir / "zed/leader/pose.csv").exists()
    Episode.create(tmp_path / "eps", "E", [("leader", "ego", t1, None)], overwrite=True)  # different source: aside
    assert not (ep.dir / "zed/leader").exists() and len(list((ep.dir / "zed").glob("leader.old-*/pose.csv"))) == 1
    assert "leader.old" not in str(R._zed(Episode.load(ep.dir)))


def test_reference_explicit(tmp_path, cli, monkeypatch):
    """#13: a user-chosen reference is an align input; an automatic one is align's to pick (an output)."""
    ep = _mk(tmp_path); assert not ep.reference_explicit and ep.reference == "s0"
    e2 = _mk(tmp_path, "X", reference="s1"); assert e2.reference_explicit and e2.reference == "s1"
    p0 = R.fingerprint_parts(ep, "align")["config.align"]
    ep.update(lambda e: setattr(e, "reference", "s1"))  # align picking another reference does not invalidate align itself
    assert R.fingerprint_parts(ep, "align")["config.align"] == p0 and R.fingerprint_parts(ep, "frames")["config.timeline"]
    monkeypatch.setattr(cli, "run_stages", lambda e, *a, configure=None, **k: (configure and e.update(configure)) and {})
    assert cli.main(["--root", str(tmp_path / "eps"), "run", "E", "--ref", "s1"]) == 0
    e = Episode.load(ep.dir); assert e.reference == "s1" and e.reference_explicit
    assert R.fingerprint_parts(e, "align")["config.align"] != p0


def test_names_use_fullmatch(tmp_path, cli):
    """#14: "ep\\n" is not a valid episode or stream name."""
    src = _dummy(tmp_path / "m/a.mp4")
    for name, stream in (("ep\n", "a"), ("ep", "a\n")):
        with pytest.raises(ValueError, match="must match"):
            Episode.create(tmp_path / "eps", name, [(stream, "ego", src, None)])
    import argparse
    with pytest.raises(EpisodeError):
        cli._episode_dir(argparse.Namespace(name="ep\n", root=str(tmp_path / "eps")))


def test_env_paths_expand_user(monkeypatch, tmp_path):
    """#15."""
    monkeypatch.setenv("HOME", str(tmp_path)); monkeypatch.setenv("PLAYGROUND_MODELS", "~/m"); monkeypatch.setenv("PLAYGROUND_EPISODES", "~/e")
    assert runtime.models_dir() == tmp_path / "m" and runtime.episodes_root() == tmp_path / "e"


@needs_ffmpeg
def test_pyav_open_is_safe_and_picks_the_video_stream(tmp_path):  # video_index None = the fallback picker (#19)
    """#16: one PyAV opener: file: URL + whitelists, probe's stream choice (cover art skipped)."""
    pytest.importorskip("av")
    assert runtime.pyav_options() == {"protocol_whitelist": "file", "format_whitelist": runtime.FFMPEG_DEMUXERS}
    base = _video(tmp_path / "b.mp4", 1.0, audio=1.0)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=32x32:duration=1", "-frames:v", "1", str(tmp_path / "c.png")], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_path / "c.png"), "-i", str(base), "-map", "0", "-map", "1", "-c", "copy",
                    "-c:v:0", "mjpeg", str(tmp_path / "cover.mkv")], check=True)
    with runtime.pyav_open(tmp_path / "cover.mkv") as src:
        assert src.stream.index == 1 and src.stream.codec_context.width == 64 and src.hwaccel == "none"
        assert sum(1 for _ in src.container.decode(src.stream)) == 10
    with runtime.pyav_open(tmp_path / "cover.mkv", video_index=0) as src:
        assert src.stream.index == 0
    (tmp_path / "evil.m3u8").write_text("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n")
    with pytest.raises(Exception):  # hls is not on the demuxer whitelist
        runtime.pyav_open(tmp_path / "evil.m3u8")


def test_pyproject_pins():
    """#17."""
    import tomllib
    d = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    pg = d["optional-dependencies"]["playground"]
    assert "fastapi>=0.115.2" in pg and "starlette>=0.39" in pg and "av>=14.1" in d["dependencies"]
    assert {f"opencv-{x}>=4.8" for x in ("python-headless", "python", "contrib-python")} <= set(pg)


def test_dropped_frames_with_changed_inputs_cascade(tmp_path):
    """F6: frames dropped after export, then its inputs change: downstream must not stay done."""
    ep = _mk(tmp_path); st = Stubs(); _run(ep, st, drop_frames=True)
    assert Episode.load(ep.dir).status["frames"]["dropped"]
    ep.update(lambda e: setattr(e, "proc_size", 320))
    marked = R.refresh_stale(ep, st.specs()); s = Episode.load(ep.dir).status
    assert "frames" in marked and not s["frames"]["dropped"] and s["qc"]["state"] == s["export"]["state"] == "stale"


def test_n_frames_rejects_non_numeric_config():
    ep = Episode(name="T", root="/nonexistent", common_start_s=0.0, common_end_s=10.0)
    for bad in (None, "10", float("nan")):
        ep.proc_fps = bad
        with pytest.raises(ValueError):
            ep.n_frames()


@needs_ffmpeg
def test_probe_records_the_first_frames_display_transform(tmp_path):
    """#18: rotation + flip from the FIRST FRAME's display matrix (also H.264 SEI-only orientation), pixel-exact against
    ffmpeg's autorotate: display = np.rot90(F(decoded), rotation // 90)."""
    W, H = 64, 40
    base = _video(tmp_path / "base.mp4", 1.0, size=f"{W}x{H}")
    cases = {"id": [], "rot90": ["-display_rotation", "90"], "rot180": ["-display_rotation", "180"], "rot270": ["-display_rotation", "270"],
             "hflip": ["-display_hflip"], "vflip": ["-display_vflip"], "rot90h": ["-display_rotation", "90", "-display_hflip"],
             "rot90v": ["-display_rotation", "90", "-display_vflip"]}
    clips = {}
    for k, opts in cases.items():
        clips[k] = tmp_path / f"{k}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *opts, "-i", str(base), "-c", "copy", str(clips[k])], check=True)
    for k, bsf in (("sei90", "insert:rotate=90"), ("sei90h", "insert:rotate=90:flip=horizontal"), ("seiv", "insert:flip=vertical")):
        clips[k] = tmp_path / f"{k}.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(base), "-c", "copy", "-bsf:v", f"h264_metadata=display_orientation={bsf}",
                        str(clips[k])], check=True)

    def gray(f, noauto):
        out = subprocess.run(["ffmpeg", "-loglevel", "error", *(["-noautorotate"] if noauto else []), "-i", str(f), "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(out, np.uint8)
    ep = Episode.create(tmp_path / "eps", "D", [(k, "exo", v, None) for k, v in clips.items()]); probe(ep)
    got = {s.name: (s.rotation, s.flip, s.width, s.height) for s in Episode.load(ep.dir).streams}
    assert got["id"] == (0, False, W, H) and got["hflip"] == (180, True, W, H) and got["vflip"] == (0, True, W, H)
    assert got["rot90"] == (90, False, H, W) and got["rot90h"][1] and got["rot90v"][1]
    assert got["sei90"] == (90, False, H, W) and got["seiv"][:2] == (0, True) and got["sei90h"][1]  # SEI: stream side data is empty
    for s in Episode.load(ep.dir).streams:
        nat = gray(clips[s.name], True).reshape(H, W)
        disp = np.rot90(nat[::-1] if s.flip else nat, s.rotation // 90)
        assert disp.shape == (s.height, s.width), s.name
        assert np.abs(disp.astype(int) - gray(clips[s.name], False).reshape(disp.shape)).mean() < 1.0, s.name
        assert (s.display_matrix is None) == (s.name == "id")


def test_display_transform_table_and_odd_angles():
    q = 65536
    mk = lambda a, b, c, d: [a * q, b * q, 0, c * q, d * q, 0, 0, 0, 1 << 30]  # noqa: E731
    assert runtime.display_transform(None) == (0, False) and runtime.display_transform(mk(1, 0, 0, 1)) == (0, False)
    assert runtime.display_transform(mk(0, -1, 1, 0)) == (90, False) and runtime.display_transform(mk(-1, 0, 0, 1)) == (180, True)
    for key, (rot, flip) in runtime._DISPLAY.items():  # rotation = ffprobe's (av_display_rotation_get), flip = det < 0
        a, b, c, d = key
        assert flip == (a * d - b * c < 0) and rot == round(-math.degrees(math.atan2(b, a))) % 360
    c, s_ = math.cos(math.radians(60)), math.sin(math.radians(60))
    assert runtime.display_transform([round(c * q), round(-s_ * q), 0, round(s_ * q), round(c * q), 0, 0, 0, 1 << 30]) == (0, False)
    assert runtime.parse_displaymatrix("\n00000000:  0  65536  0\n00000001:  -65536  0  0\n00000002:  0  0  1073741824\n") == \
        [0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824]
    assert runtime.parse_displaymatrix("garbage") is None


def test_force_is_passed_to_stages_that_take_it(tmp_path):
    """fx-depth-scene: --force reaches a stage's own cache (scene_scan station cache)."""
    ep = _mk(tmp_path); st = Stubs(); specs = st.specs(); seen = []

    def cached_stage(e, force: bool = False):
        seen.append(force); e.set_status("scan", "done", f"force={force}")
    specs["scan"] = R.StageSpec(cached_stage, deps=("frames",), outputs=("scan",), opt_in=True)
    R.run_stages(ep, None, log=lambda m: None, specs=specs)
    R.run_stages(ep, ["scan"], log=lambda m: None, specs=specs)
    R.run_stages(ep, ["scan"], force=True, log=lambda m: None, specs=specs)
    assert seen == [False, True] and st.calls.count("body2d") == 1  # other stages do not receive force
