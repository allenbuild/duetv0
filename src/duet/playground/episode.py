"""Episode model for the ego/exo playground.

An episode is a directory:

    <episodes_root>/<name>/
        episode.json          streams, offsets, stage status (written by us, atomically, under episode.json.lock)
        streams/<stream>.mp4  the videos (relative symlinks, copies or moved files)
        imu/<stream>.parquet  optional IMU (Eidon 7-slot schema)
        rig.json, zed/<s>/    optional user-provided rig config / ZED exports
        derived/...           everything the pipeline produces; per-run logs in derived/logs/<ts>-<pid>.log (CLI runs)
                              or derived/logs/<ts>-job.log (server jobs), newest 50 kept

Stream roles: "ego" (head-mounted) or "exo" (fixed). One stream is the time reference.

Time (seconds everywhere):
    t_stream  a stream's own timeline, 0 = its first VIDEO frame (container pts minus ``video_start_s``)
    t_ref     the reference stream's timeline
    t_ref = (1 + drift_ppm * 1e-6) * t_stream + offset_s        (``Episode.ref_time`` / ``Episode.stream_time``)
These two methods are the only sanctioned conversions. Processed frame k of every usable stream is nominally at
``t_ref = common_start_s + k / proc_fps`` (``frame_times_ref``), k < ``n_frames()``.

Persistence: ``save()`` writes atomically (temp file + fsync + os.replace) under an fcntl lock (episode.json.lock)
and merges this object's changes (fields changed since it was loaded/saved; per stage key for ``status``, per
stream field for ``streams``) onto the latest file, so concurrent writers (server, pipeline subprocess, CLI) never
tear the file or clobber each other. ``update(fn)`` applies ``fn`` to the fresh state under the lock. A running
pipeline holds <ep>/run.lock (``run_lock``). Only the probe and align stages write episode config (stream facts,
offsets, window); later stages have it in their fingerprints and must not change it.
"""
from __future__ import annotations

import contextlib
import copy
import errno
import fcntl
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import uuid
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import runtime

SCHEMA_VERSION = 2
# every stage in execution order; run.py builds its registry from this (single source of truth)
STAGES = ("probe", "align", "frames", "calib", "stereo_depth", "body2d", "hands", "objects", "contact", "tags", "headpose", "body3d",
          "depth_mono", "scene_scan", "world3d", "track", "gaze_proxy", "speech", "qc", "imu_arm", "annotate", "autolabel", "metrics", "export")
OPT_IN_STAGES = ("depth_mono", "scene_scan")  # run only when named explicitly
STATES = ("done", "skipped", "failed", "running", "interrupted", "stale")
FINISHED_STATES = ("done", "skipped", "failed", "interrupted")
OFFSET_STATUSES = ("reference", "audio", "manual", "unaligned")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_DETAIL = 2000
MAX_FPS = 60.0


class EpisodeError(ValueError):
    """episode.json missing, unreadable or invalid."""


@dataclass
class Stream:
    name: str
    role: str  # ego | exo
    path: str  # relative to episode dir
    person: str | None = None  # who wears it (ego) or label (exo)
    fps: float = 0.0
    width: int = 0  # DISPLAY size (after rotation)
    height: int = 0
    duration_s: float = 0.0  # video stream duration (first to past-last video frame)
    has_audio: bool = False
    offset_s: float = 0.0  # t_ref = (1 + drift_ppm*1e-6) * t_stream + offset_s
    offset_confidence: float | None = None
    imu: str | None = None  # relative path to IMU parquet, if any
    video_start_s: float = 0.0  # ffprobe start_time of the video stream (container seconds)
    audio_start_s: float | None = None  # ffprobe start_time of the first audio stream (container seconds)
    rotation: int = 0  # CCW degrees 0/90/180/270 of the display matrix (ffprobe "rotation"); runtime.display_transform
    flip: bool = False  # the display matrix is a reflection: display = np.rot90(decoded[::-1] if flip else decoded, rotation // 90)
    nb_frames: int | None = None
    offset_status: str = "unaligned"  # reference | audio | manual | unaligned
    offset_override_s: float | None = None  # manual offset (streams without usable audio); align must honour it
    drift_ppm: float = 0.0
    video_index: int | None = None  # ffprobe index of the probed video stream (attached pictures skipped)
    imu_offset_override_s: float | None = None  # manual IMU clock offset (s): t_stream = t_imu + this (imu_arm honours it)
    display_matrix: list[int] | None = None  # the first video frame's FFmpeg display matrix (9 int32), None = identity
    _extra: dict = field(default_factory=dict, repr=False, compare=False)  # unknown keys, preserved on save


@dataclass
class Episode:
    name: str
    root: str  # absolute episode directory (not persisted)
    streams: list[Stream] = field(default_factory=list)
    reference: str = ""
    reference_explicit: bool = False  # True when the user chose the reference (align may otherwise pick a better one)
    proc_fps: float = 10.0
    proc_size: int = 640
    status: dict = field(default_factory=dict)  # stage -> {state, started, finished, detail, fingerprint, pid, host, log}
    common_start_s: float = 0.0  # in reference time
    common_end_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    max_lag_s: float = 120.0  # align search window
    min_overlap_s: float = 5.0  # align: minimum audio overlap per lag / minimum common window
    device: str = "auto"  # torch device preference (runtime.pick_device)
    hwaccel: str = "auto"  # ffmpeg decode hwaccel preference (runtime.run_ffmpeg)
    schema_version: int = SCHEMA_VERSION
    _extra: dict = field(default_factory=dict, repr=False, compare=False)
    _base: dict | None = field(default=None, repr=False, compare=False)  # last state read from/written to disk (merge base)
    _fp: dict = field(default_factory=dict, repr=False, compare=False)  # stage -> fingerprint of the current run (run.py)

    # ------------------------------------------------------------------ paths / lookup
    @property
    def dir(self) -> Path:
        return Path(self.root)

    @property
    def derived(self) -> Path:
        d = self.dir / "derived"
        with contextlib.suppress(OSError):
            d.mkdir(exist_ok=True)
        return d

    @property
    def json_path(self) -> Path:
        return self.dir / "episode.json"

    def stream(self, name: str) -> Stream:
        for s in self.streams:
            if s.name == name:
                return s
        raise KeyError(f"no stream {name!r} in episode {self.name!r}")

    def _s(self, s: Stream | str) -> Stream:
        return self.stream(s) if isinstance(s, str) else s

    def egos(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "ego"]

    def exos(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "exo"]

    # ------------------------------------------------------------------ time mapping (C1)
    def ref_time(self, s: Stream | str, t_stream):
        """Stream time (s, 0 = first video frame) -> reference time (s). Vectorised."""
        s = self._s(s)
        return (1.0 + s.drift_ppm * 1e-6) * np.asarray(t_stream, dtype=float) + s.offset_s

    def stream_time(self, s: Stream | str, t_ref):
        """Reference time (s) -> stream time (s, 0 = first video frame). Vectorised inverse of ``ref_time``."""
        s = self._s(s)
        return (np.asarray(t_ref, dtype=float) - s.offset_s) / (1.0 + s.drift_ppm * 1e-6)

    def n_frames(self) -> int:
        """Processed frames on the common grid; ValueError when the window or proc_fps is invalid/empty."""
        if not (_finite(self.common_start_s) and _finite(self.common_end_s) and self.common_end_s > self.common_start_s):
            raise ValueError(f"empty common window [{self.common_start_s}, {self.common_end_s}] s (run align)")
        span, fps = self.common_end_s - self.common_start_s, self.proc_fps
        if not (_finite(fps) and 0 < fps <= MAX_FPS):
            raise ValueError(f"invalid proc_fps {fps!r}")
        n = int(np.floor(span * fps + 1e-9))
        if n < 1:
            raise ValueError(f"common window {span:.3f} s holds no frame at {fps} fps")
        return n

    def frame_times_ref(self) -> np.ndarray:
        """Nominal reference time (s) of processed frame k = common_start_s + k / proc_fps, k < n_frames()."""
        return self.common_start_s + np.arange(self.n_frames()) / self.proc_fps

    def frame_times_stream(self, s: Stream | str) -> np.ndarray:
        """Stream time (s) of processed frame k's nominal instant, for stream ``s``."""
        return self.stream_time(s, self.frame_times_ref())

    def usable(self, s: Stream | str) -> bool:
        """False for a non-reference stream whose offset is unknown ("unaligned"): consumers skip it."""
        s = self._s(s)
        return s.name == self.reference or s.offset_status != "unaligned"

    def stage_ok(self, name: str) -> bool:
        """True iff the stage's state is "done". Check this before reading an upstream artifact."""
        return (self.status.get(name) or {}).get("state") == "done"

    def validate_config(self) -> None:
        """Assert the processing config is sane (raises ValueError)."""
        if not (_finite(self.proc_fps) and 0 < self.proc_fps <= MAX_FPS):
            raise ValueError(f"proc_fps must be in (0, {MAX_FPS:g}], got {self.proc_fps!r}")
        if not (isinstance(self.proc_size, int) and not isinstance(self.proc_size, bool) and 64 <= self.proc_size <= 4096):
            raise ValueError(f"proc_size must be an int in [64, 4096], got {self.proc_size!r}")
        if not (_finite(self.max_lag_s) and 0 < self.max_lag_s <= 3600):
            raise ValueError(f"max_lag_s must be in (0, 3600] s, got {self.max_lag_s!r}")
        if not (_finite(self.min_overlap_s) and 0 < self.min_overlap_s <= 3600):
            raise ValueError(f"min_overlap_s must be in (0, 3600] s, got {self.min_overlap_s!r}")
        if not runtime.valid_device(self.device):
            raise ValueError(f"device must be auto|cpu|mps|cuda[:N], got {self.device!r}")
        if self.hwaccel not in runtime.HWACCELS:
            raise ValueError(f"hwaccel must be one of {runtime.HWACCELS}, got {self.hwaccel!r}")
        names = [s.name for s in self.streams]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stream names {names}")
        if self.streams and self.reference not in names:
            raise ValueError(f"reference {self.reference!r} is not a stream ({names})")
        for s in self.streams:
            if s.offset_override_s is not None and not _finite(s.offset_override_s):
                raise ValueError(f"{s.name}: offset override must be a finite number of seconds")
            if s.imu_offset_override_s is not None and not (_finite(s.imu_offset_override_s) and s.imu):
                raise ValueError(f"{s.name}: IMU offset override needs an IMU stream and a finite number of seconds")
            if s.offset_status not in OFFSET_STATUSES:
                raise ValueError(f"{s.name}: offset_status must be one of {OFFSET_STATUSES}, got {s.offset_status!r}")

    # ------------------------------------------------------------------ status
    def set_status(self, stage: str, state: str, detail: str = "", **extra: Any) -> None:
        """Record a stage state (merged per stage key under the lock; also persists this object's other changes).

        "running" starts a fresh entry with started/pid/host; terminal states keep started/pid/host/log of the
        running entry and set ``finished``. ``detail`` is truncated to 2,000 chars with local paths stripped."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}, got {state!r}")
        detail = self.clean_detail(detail)

        def fn(ep: Episode) -> None:
            old = ep.status.get(stage) or {}; now = time.time(); was_running = old.get("state") == "running"
            if state == "running":
                st = {"state": state, "detail": detail, "started": now, "pid": os.getpid(), "host": socket.gethostname()}
                if was_running and "log" in old:
                    st["log"] = old["log"]
            elif state == "stale":  # outputs kept but out of date: keep the record, change the state
                st = {**old, "state": state, "detail": detail}
            else:
                st = {k: old[k] for k in ("started", "pid", "host", "log") if k in old} if was_running else {"started": now}
                st.update(state=state, detail=detail, finished=now)
                if state == "done" and stage in ep._fp:
                    st["fingerprint"] = ep._fp[stage]
            st.update(runtime.json_safe(extra)); ep.status[stage] = st
        self.update(fn)

    def clean_detail(self, detail: Any) -> str:
        """str, local absolute paths (episodes root, home) stripped, <= 2,000 chars (head + tail kept)."""
        d = str(detail if detail is not None else "")
        for p in {str(self.dir.parent), os.path.realpath(self.dir.parent)}:
            d = d.replace(p.rstrip("/") + "/", "")
        home = os.path.expanduser("~")
        if home and home != "/":
            d = d.replace(home, "~")
        if len(d) > MAX_DETAIL:
            d = d[: MAX_DETAIL // 2 - 3] + " ... " + d[-(MAX_DETAIL // 2 - 2):]
        return d

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("root", "_extra", "_base", "_fp"):
            d.pop(k, None)
        d["streams"] = [_stream_to_dict(s) for s in self.streams]
        return {**copy.deepcopy(self._extra), **d}

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive inter-process (fcntl) + inter-thread lock on this episode's episode.json.lock."""
        with _file_lock(self.dir / "episode.json.lock"):
            yield

    def save(self) -> None:
        """Merge this object's changes since load/last save onto the latest episode.json and write it atomically."""
        with self.locked():
            self._sync(write=True)

    def update(self, fn: Callable[[Episode], Any]) -> Episode:
        """Under the lock: refresh from disk (keeping this object's unsaved changes), apply ``fn(self)``, write."""
        with self.locked():
            self._sync(write=False)
            fn(self)
            self._sync(write=True)
        return self

    def reload(self) -> Episode:
        """Refresh from disk, keeping this object's unsaved changes (no write)."""
        self._sync(write=False, strict=True)
        return self

    def _sync(self, write: bool, strict: bool = False) -> None:
        """Merge (base -> self) changes onto the file. write=False: refresh self, base := file. write=True: also
        write the merge, base := merge. The caller holds the lock when writing."""
        mine = self.to_dict()
        try:
            theirs = _read_json(self.json_path)
        except EpisodeError:
            if strict:
                raise
            theirs = None  # corrupt file (e.g. torn by the old non-atomic writer): our state replaces it
        merged = mine if theirs is None else _merge3(self._base, mine, theirs)
        if write:
            merged["schema_version"] = SCHEMA_VERSION  # everything we write is the v2 model (legacy files upgraded)
            runtime.atomic_write_json(self.json_path, merged)
            self._apply(merged); self._base = copy.deepcopy(merged)
        elif theirs is not None:
            self._apply(merged); self._base = theirs

    def _apply(self, d: dict) -> None:
        """Replace this object's persisted fields with ``d`` in place (existing Stream objects are updated, so
        references held by callers stay valid)."""
        known = {f.name for f in fields(Episode)} - {"root", "_extra", "_base", "streams"}
        for k in known:
            if k in d:
                setattr(self, k, copy.deepcopy(d[k]))
        self._extra = {k: copy.deepcopy(v) for k, v in d.items() if k not in known and k != "streams"}
        by_name = {s.name: s for s in self.streams}; new = []
        for sd in d.get("streams", []):
            s = by_name.get(sd.get("name"))
            fresh = _stream_from_dict(sd)
            if s is None:
                new.append(fresh); continue
            for f in fields(Stream):
                setattr(s, f.name, getattr(fresh, f.name))
            new.append(s)
        self.streams = new

    @classmethod
    def load(cls, path: str | os.PathLike) -> Episode:
        """Load <path>/episode.json. Tolerant: missing fields take defaults, unknown keys are kept, files written by
        the old code are upgraded in memory. Raises EpisodeError when the file is missing or not valid JSON."""
        root = Path(path).resolve()
        d = _read_json(root / "episode.json")
        if d is None:
            raise EpisodeError(f"no episode.json in {root.name}")
        ep = cls(name=str(d.get("name") or root.name), root=str(root))
        ep._apply(d); ep._base = copy.deepcopy(d)
        return ep

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, root: str | os.PathLike, name: str, videos: list[tuple[str, str, str | os.PathLike, str | None]],
               imus: dict[str, str | os.PathLike] | None = None, reference: str | None = None, link: bool | None = None, *,
               mode: str = "symlink", overwrite: bool = False, proc_fps: float = 10.0, proc_size: int = 640,
               max_lag_s: float = 120.0, min_overlap_s: float = 5.0, device: str = "auto", hwaccel: str = "auto",
               offsets: dict[str, float] | None = None, imu_offsets: dict[str, float] | None = None,
               keep_zed: bool = False, discard_old_streams: bool = False) -> Episode:
        """Create <root>/<name> from ``videos`` = [(stream_name, role, source_path, person)].

        mode: "symlink" (RELATIVE link), "copy" or "move" (``link=False`` is the old spelling of "copy"). "move" hard-links
        (same filesystem) or copies + fsyncs each source into the staging dir, commits the episode, and only THEN unlinks
        the originals, so a killed process never leaves a video only in staging; it needs regular files (no symlinks)
        and refuses sources under a data/raw directory (immutable, AGENTS.md).
        Validates names (``NAME_RE``, unique), roles, persons, sources (exist, regular files), ``reference``/``imus``/
        ``offsets`` keys (known streams) and the config. The episode is assembled in <root>/.staging/ and renamed (or,
        into an existing directory, moved) into place; episode.json is written last, so a failure leaves no episode.
        Existing directory: refused unless ``overwrite``, except a prepared one holding only rig.json and zed/ (the ZED
        flow: zed_export into <ep>/zed/<s>/, then create with <ep>/zed/<s>/left.mp4; sources inside the episode's zed/
        are allowed in symlink/copy mode). Overwrite replaces episode.json, derived/, streams/ and imu/ and keeps rig.json
        and zed/, but refuses when streams/ or imu/ hold regular files (moved/copied in: maybe the only copy) unless
        ``discard_old_streams``, and moves zed/<s>/ aside to zed/<s>.old-<time>/ when stream s gets a different source
        unless ``keep_zed``."""
        if link is not None:
            mode = "symlink" if link else "copy"
        reference = reference or None
        if mode not in ("symlink", "copy", "move"):
            raise ValueError(f"mode must be symlink|copy|move, got {mode!r}")
        if not NAME_RE.fullmatch(name or ""):
            raise ValueError(f"episode name {name!r} must match {NAME_RE.pattern}")
        root = Path(os.path.realpath(root)); final = root / name; zed_dir = final / "zed"
        if not videos:
            raise ValueError("at least one video is required")
        srcs: list[tuple[Stream, Path]] = []
        for v in videos:
            if len(v) != 4:
                raise ValueError(f"video spec must be (name, role, path, person), got {v!r}")
            sname, role, src, person = v; src = Path(src); person = str(person).strip() if person is not None and str(person).strip() else None
            if not NAME_RE.fullmatch(str(sname or "")):
                raise ValueError(f"stream name {sname!r} must match {NAME_RE.pattern}")
            if role not in ("ego", "exo"):
                raise ValueError(f"stream {sname}: role must be ego|exo, got {role!r}")
            if person is not None and (len(person) > 64 or re.search(r"[\x00-\x1f<>\"'`\\]", person)):
                raise ValueError(f"stream {sname}: person label {person!r} must be 1-64 chars without control chars or <>\"'`\\")
            if not src.is_file():
                raise FileNotFoundError(f"stream {sname}: source {src} does not exist or is not a file")
            suffix = src.suffix.lower() or ".mp4"
            if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
                raise ValueError(f"stream {sname}: unsupported file extension {src.suffix!r}")
            srcs.append((Stream(name=sname, role=role, path=f"streams/{sname}{suffix}", person=person), src))
        names = [s.name for s, _ in srcs]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate stream names {dup} (give each stream a unique name)")
        imus = {k: Path(v) for k, v in (imus or {}).items()}
        for k, v in imus.items():
            if k not in names:
                raise ValueError(f"--imu for unknown stream {k!r} (streams: {names})")
            if not v.is_file():
                raise FileNotFoundError(f"imu for {k}: {v} does not exist or is not a file")
        if reference is not None and reference not in names:
            raise ValueError(f"reference {reference!r} is not a stream ({names})")
        for what, d in (("offset", offsets), ("imu offset", imu_offsets)):
            for k, v in (d or {}).items():
                if k not in names:
                    raise ValueError(f"{what} for unknown stream {k!r} (streams: {names})")
                if not math.isfinite(float(v)):
                    raise ValueError(f"{what} for {k} must be finite")
        for k in imu_offsets or {}:
            if k not in imus:
                raise ValueError(f"imu offset for {k!r}, which has no IMU file")
        all_srcs = [p for _, p in srcs] + list(imus.values())
        for p in all_srcs:
            real = Path(os.path.realpath(p))
            if real.is_relative_to(final) and not (real.is_relative_to(zed_dir) and mode != "move"):
                raise ValueError(f"source {p} lies inside the episode directory" + (" (move is not allowed from zed/)" if real.is_relative_to(zed_dir) else ""))
            if mode == "move":
                if Path(p).is_symlink():
                    raise ValueError(f"--move needs regular files; {p} is a symlink (use --copy or the default symlink mode)")
                if _under_raw(real):
                    raise ValueError(f"refusing to move {p}: files under data/raw are immutable (AGENTS.md); use --copy or symlink")
        if len({os.path.realpath(p) for p in all_srcs}) != len(all_srcs) and mode == "move":
            raise ValueError("the same source file is given twice (mode=move)")
        existing = final.exists() or final.is_symlink()
        prepared = existing and final.is_dir() and not final.is_symlink() and _only_kept(final)
        if existing and not prepared:
            if not overwrite:
                raise FileExistsError(f"episode {name} exists (use overwrite)")
            if not final.is_dir() or final.is_symlink():
                raise FileExistsError(f"{name} exists and is not an episode directory")
            if not discard_old_streams:
                _refuse_real_files(final)
        if existing:
            _refuse_if_running(final)
        streams = [s for s, _ in srcs]
        ref = reference or (next((s.name for s in streams if s.role == "ego"), streams[0].name))
        for s in streams:
            if s.name == ref:
                s.offset_status = "reference"
            if offsets and s.name in offsets:
                s.offset_override_s = float(offsets[s.name])
        ep = cls(name=name, root=str(final), streams=streams, reference=ref, reference_explicit=reference is not None,
                 proc_fps=float(proc_fps), proc_size=int(proc_size), max_lag_s=float(max_lag_s), min_overlap_s=float(min_overlap_s),
                 device=device, hwaccel=hwaccel)
        for sname in imus:
            s = ep.stream(sname); s.imu = f"imu/{sname}.parquet"
            if imu_offsets and sname in imu_offsets:
                s.imu_offset_override_s = float(imu_offsets[sname])
        ep.validate_config()

        staging = root / ".staging"; staging.mkdir(parents=True, exist_ok=True)
        _clean_staging(staging)
        tmp = staging / f"{name}-{uuid.uuid4().hex[:8]}"
        items = [(tmp / s.path, final / s.path, src) for s, src in srcs] + [(tmp / ep.stream(k).imu, final / ep.stream(k).imu, src) for k, src in imus.items()]
        with _staging_lock(tmp):  # held while assembling: _clean_staging never touches an active staging dir
            try:
                runtime.atomic_write_json(tmp.with_name(tmp.name + ".json"),
                                          {"episode": name, "mode": mode, "items": [[os.path.abspath(s), str(t.relative_to(tmp))] for t, _, s in items]})
                (tmp / "streams").mkdir(parents=True); (tmp / "imu").mkdir()
                for dst_tmp, dst_final, src in items:
                    src_real = Path(os.path.realpath(src))
                    if mode == "symlink":
                        os.symlink(os.path.relpath(src_real, dst_final.parent), dst_tmp)
                    elif mode == "move":
                        try:
                            os.link(src_real, dst_tmp)  # same filesystem: instant, no extra space; original untouched until commit
                        except OSError:
                            _copy_fsync(src_real, dst_tmp)
                    else:
                        _copy_fsync(src_real, dst_tmp)
                runtime.atomic_write_json(tmp / "episode.json", ep.to_dict())
                if existing:
                    with _file_lock(final / "episode.json.lock"):
                        _refuse_if_running(final)
                        if not prepared and not discard_old_streams:
                            _refuse_real_files(final)
                        if not keep_zed:
                            _move_zed_aside(final, srcs)
                        for sub in ("derived", "streams", "imu"):
                            p = final / sub
                            if p.is_symlink() or p.is_file():
                                p.unlink()
                            elif p.exists():
                                shutil.rmtree(p)
                            if (tmp / sub).exists():
                                os.replace(tmp / sub, p)
                        os.replace(tmp / "episode.json", final / "episode.json")  # commit point
                else:
                    try:
                        os.rename(tmp, final)  # atomic commit; fails if a concurrent create won the race
                    except OSError as e:
                        if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
                            raise FileExistsError(f"episode {name} exists (created concurrently)") from None
                        raise
                runtime._fsync_dir(root)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                with contextlib.suppress(OSError):
                    tmp.with_name(tmp.name + ".json").unlink()
        if mode == "move":  # committed: now the originals can go
            for _, _, src in items:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(src)
        return cls.load(final)


# ---------------------------------------------------------------------------------------------- helpers

_STREAM_FIELDS = {f.name for f in fields(Stream)} - {"_extra"}


def _stream_to_dict(s: Stream) -> dict:
    d = asdict(s); extra = d.pop("_extra", {}) or {}
    return {**copy.deepcopy(extra), **d}


def _stream_from_dict(d: dict) -> Stream:
    s = Stream(**{k: copy.deepcopy(v) for k, v in d.items() if k in _STREAM_FIELDS})
    s._extra = {k: copy.deepcopy(v) for k, v in d.items() if k not in _STREAM_FIELDS}
    return s


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _normalize(d: dict) -> dict:
    """Complete a parsed episode.json with every default (so merges compare like with like). Files written by the
    old code (schema 1): offset_status = "reference" for the reference, "audio" if offset_confidence >= 1.3 (the old
    code's own check-by-eye threshold), else "unaligned" (0.0 was its "no audio, offset left at 0" marker)."""
    d = dict(d); d.setdefault("schema_version", 1)
    for f in fields(Episode):
        if f.name not in d and f.name not in ("name", "root", "streams") and not f.name.startswith("_"):
            d[f.name] = f.default_factory() if f.default is MISSING else f.default  # type: ignore[misc]
    ref = d.get("reference"); out = []
    for sd in d.get("streams") or []:
        if not isinstance(sd, dict) or not all(isinstance(sd.get(k), str) for k in ("name", "role", "path")):
            raise EpisodeError(f"stream entry without name/role/path: {str(sd)[:120]}")
        sd = dict(sd)
        if "offset_status" not in sd:
            conf = sd.get("offset_confidence")
            sd["offset_status"] = "reference" if sd.get("name") == ref else ("audio" if _finite(conf) and conf >= 1.3 else "unaligned")
        out.append(_stream_to_dict(_stream_from_dict(sd)))
    d["streams"] = out
    return d


def _read_json(p: Path) -> dict | None:
    """Parsed + normalised episode.json, None when absent; EpisodeError when unreadable."""
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as e:
        raise EpisodeError(f"cannot read {p.parent.name}/episode.json: {e}") from None
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise EpisodeError(f"corrupt {p.parent.name}/episode.json ({len(raw)} bytes): {e}") from None
    if not isinstance(d, dict) or not isinstance(d.get("streams", []), list):
        raise EpisodeError(f"{p.parent.name}/episode.json is not an episode object")
    return _normalize(d)


def _merge3(base: dict | None, mine: dict, theirs: dict) -> dict:
    """Three-way merge: a value changed in ``mine`` since ``base`` wins, otherwise ``theirs`` (the file) wins.

    Granularity: top-level keys; ``status`` per stage key; ``streams`` per stream (by name) and field."""
    if base is None:
        return copy.deepcopy(mine)
    out: dict = {}
    for k in dict.fromkeys([*theirs, *mine]):
        if k == "status":
            out[k] = _merge_map(base.get(k) or {}, mine.get(k) or {}, theirs.get(k) or {})
        elif k == "streams":
            out[k] = _merge_streams(base.get(k) or [], mine.get(k) or [], theirs.get(k) or [])
        elif k in mine and (k not in base or mine[k] != base[k]):
            out[k] = copy.deepcopy(mine[k])
        elif k in theirs:
            out[k] = copy.deepcopy(theirs[k])
        elif k in base:
            continue  # deleted on disk, unchanged here
        else:
            out[k] = copy.deepcopy(mine[k])
    return out


def _merge_map(base: dict, mine: dict, theirs: dict) -> dict:
    out = {}
    for k in dict.fromkeys([*theirs, *mine]):
        if mine.get(k, _MISSING) != base.get(k, _MISSING):  # changed (or deleted) here
            if k in mine:
                out[k] = copy.deepcopy(mine[k])
        elif k in theirs:
            out[k] = copy.deepcopy(theirs[k])
    return out


def _merge_streams(base: list, mine: list, theirs: list) -> list:
    b = {s.get("name"): s for s in base}; m = {s.get("name"): s for s in mine}; t = {s.get("name"): s for s in theirs}
    set_changed = [s.get("name") for s in mine] != [s.get("name") for s in base]
    order = [s.get("name") for s in (mine if set_changed else theirs)]
    order += [n for n in t if n not in order and n not in b]  # added on disk
    out = []
    for n in order:
        if n in m and n in t:
            out.append(_merge_map(b.get(n) or {}, m[n], t[n]))
        elif n in m:
            if n in b and not set_changed:
                continue  # deleted on disk
            out.append(copy.deepcopy(m[n]))
        elif n in t and n not in b:
            out.append(copy.deepcopy(t[n]))
    return out


_MISSING = object()
_TLOCKS: dict[str, threading.RLock] = {}
_TLOCKS_GUARD = threading.Lock()
_DEPTH: dict[str, int] = {}


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Re-entrant exclusive lock: a per-path threading.RLock (threads) + fcntl.flock on ``path`` (processes)."""
    key = os.path.realpath(path)
    with _TLOCKS_GUARD:
        rl = _TLOCKS.setdefault(key, threading.RLock())
    with rl:
        if _DEPTH.get(key):
            _DEPTH[key] += 1
            try:
                yield
            finally:
                _DEPTH[key] -= 1
            return
        fd = os.open(key, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX); _DEPTH[key] = 1
            try:
                yield
            finally:
                _DEPTH[key] = 0
        finally:
            os.close(fd)


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def run_lock(ep_dir: str | os.PathLike, wait_s: float = 2.0) -> Iterator[None]:
    """Exclusive per-episode pipeline lock (fcntl on <ep>/run.lock, released by the kernel when the holder dies).
    Raises BlockingIOError when another pipeline holds it for longer than ``wait_s``."""
    p = Path(ep_dir) / "run.lock"
    fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        t_end = time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                if time.monotonic() > t_end:
                    raise BlockingIOError(f"a pipeline is already running for episode {Path(ep_dir).name}") from None
                time.sleep(0.05)
        os.ftruncate(fd, 0); os.write(fd, f"{os.getpid()} {socket.gethostname()}\n".encode())
        yield
    finally:
        os.close(fd)


def run_lock_held(ep_dir: str | os.PathLike) -> bool:
    """True while some process (any, this one included) holds the episode's pipeline lock."""
    p = Path(ep_dir) / "run.lock"
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def _clean_staging(d: Path, min_age_s: float = 600) -> list[str]:
    """Remove creation temp dirs of killed processes (not locked, older than min_age_s). A staged file whose original
    source no longer exists (per the dir's manifest) is restored to that path first - by construction create never
    unlinks an original before committing, so this is defence in depth. Returns what was restored."""
    restored: list[str] = []
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        if not p.is_dir() or p.is_symlink():
            continue
        lock = p.with_name(p.name + ".lock")
        if lock.exists():
            fd = os.open(lock, os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a create is assembling it
            finally:
                os.close(fd)
        with contextlib.suppress(OSError):
            if time.time() - p.lstat().st_mtime < min_age_s:
                continue
            man = p.with_name(p.name + ".json")
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                for src, rel in json.loads(man.read_text())["items"]:
                    if not os.path.lexists(src) and (p / rel).is_file():
                        os.makedirs(os.path.dirname(src), exist_ok=True); shutil.move(str(p / rel), src); restored.append(src)
            shutil.rmtree(p)
            for f in (man, lock):
                with contextlib.suppress(OSError):
                    f.unlink()
    return restored


def _refuse_if_running(ep_dir: Path) -> None:
    """Every pipeline holds <ep>/run.lock (released by the kernel when its process dies), so the lock is the liveness test."""
    if run_lock_held(ep_dir):
        raise RuntimeError(f"episode {ep_dir.name}: a pipeline is running; cancel it first")


_KEPT = {"zed", "rig.json", "episode.json.lock", "run.lock"}  # what create keeps in (or accepts in) an existing directory


def _only_kept(d: Path) -> bool:
    """A prepared directory: no episode.json, nothing but rig.json / zed/ (and lock files)."""
    return all(p.name in _KEPT for p in d.iterdir())


def _under_raw(p: Path) -> bool:
    parts = Path(p).parts
    return any(a == "data" and b == "raw" for a, b in zip(parts, parts[1:]))


def _refuse_real_files(ep_dir: Path) -> None:
    """Refuse to delete stream/IMU files that are not links: moved or copied in, they may be the only copy."""
    real = [str(p.relative_to(ep_dir)) for sub in ("streams", "imu") if (ep_dir / sub).is_dir()
            for p in sorted((ep_dir / sub).iterdir()) if p.is_file() and not p.is_symlink()]
    if real:
        raise FileExistsError(f"episode {ep_dir.name}: overwrite would delete {len(real)} file(s) that are not links and may be the only "
                              f"copy ({', '.join(real[:4])}{' ...' if len(real) > 4 else ''}); move them elsewhere or pass "
                              "discard_old_streams (--discard-old-streams)")


def _copy_fsync(src: Path, dst: Path) -> None:
    shutil.copy2(src, dst)
    fd = os.open(dst, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _same_source(old: Path, new: Path) -> bool:
    """Does an episode's existing stream file ``old`` stand for the same recording as source ``new``?"""
    if old.is_symlink():
        return os.path.realpath(old) == os.path.realpath(new)
    try:
        a, b = old.stat(), Path(new).stat()
    except OSError:
        return False
    return (a.st_size, a.st_mtime_ns) == (b.st_size, b.st_mtime_ns)  # copy2 keeps size + mtime


def _move_zed_aside(final: Path, srcs: list[tuple[Stream, Path]]) -> None:
    """zed/<s>/ belongs to the recording it was exported from: move it to zed/<s>.old-<time>/ when stream s now gets a
    different source (a source inside zed/<s>/ is the ZED export itself: kept)."""
    old = {}
    with contextlib.suppress(EpisodeError):
        old = {sd["name"]: sd for sd in (_read_json(final / "episode.json") or {}).get("streams", [])}
    for s, src in srcs:
        z = final / "zed" / s.name
        if not z.is_dir() or Path(os.path.realpath(src)).is_relative_to(os.path.realpath(z)) or s.name not in old:
            continue
        if not _same_source(final / old[s.name]["path"], src):
            os.replace(z, z.with_name(f"{s.name}.old-{time.strftime('%Y%m%d-%H%M%S')}"))


@contextlib.contextmanager
def _staging_lock(tmp: Path) -> Iterator[None]:
    fd = os.open(tmp.with_name(tmp.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp.with_name(tmp.name + ".lock"))
        os.close(fd)


# ---------------------------------------------------------------------------------------------- probe

def _rate(s: str | None) -> float:
    try:
        num, _, den = str(s or "").partition("/")
        v = float(num) / float(den or 1)
        return v if math.isfinite(v) and v > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _num(x: Any) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _hms(x: Any) -> float | None:
    """Matroska DURATION tag "HH:MM:SS.fffffffff" -> seconds."""
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2}(?:\.\d+)?)", str(x or "").strip())
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else None


def ffprobe(path: Path) -> dict:
    """Video facts for one file (safe input options). The video stream is the largest non-cover-art one (recorded as
    ``video_index``); duration/start come from it (audio tails ignored); width/height are the DISPLAY size after the
    rotation side data/tag; audio_start_s is the first audio stream's start (container seconds)."""
    j = runtime.ffprobe_json(path, "-show_entries",
                             "format=duration,start_time:stream=index,codec_type,width,height,r_frame_rate,avg_frame_rate,"
                             "start_time,duration,nb_frames:stream_disposition=attached_pic:stream_tags=rotate,DURATION"
                             ":stream_side_data=rotation")
    streams = j.get("streams") or []; fmt = j.get("format") or {}
    vids = [s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")]
    auds = [s for s in streams if s.get("codec_type") == "audio"]
    if not vids:
        raise ValueError(f"{Path(path).name}: no video stream")
    v = max(vids, key=lambda x: (int(x.get("width") or 0) * int(x.get("height") or 0), -int(x.get("index", 0))))  # like ffmpeg's default pick
    fps = next((f for f in (_rate(v.get("avg_frame_rate")), _rate(v.get("r_frame_rate"))) if 0 < f <= 240), 0.0)
    start = _num(v.get("start_time")) or 0.0
    nb = int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() and int(v["nb_frames"]) > 0 else None
    dur = _num(v.get("duration")) or _hms((v.get("tags") or {}).get("DURATION"))
    if not dur and nb and fps:
        dur = nb / fps
    if not dur and _num(fmt.get("duration")):
        dur = _num(fmt["duration"]) - max(0.0, start - (_num(fmt.get("start_time")) or 0.0))
    if not dur:  # e.g. live-recorded webm: no durations anywhere -> scan packet timestamps
        pk = runtime.ffprobe_json(path, "-select_streams", str(v.get("index", 0)), "-show_entries", "packet=pts_time,duration_time")
        pts = [p for p in (_num(q.get("pts_time")) for q in pk.get("packets") or []) if p is not None]
        if pts:
            nb = nb or len(pts)
            if not dur:
                step = 1.0 / fps if fps else (np.median(np.diff(sorted(pts))) if len(pts) > 1 else 0.0)
                dur = max(pts) + step - min(start, min(pts))
    if not fps and nb and dur:
        fps = nb / dur
    if not (dur and dur > 0 and fps and 0 < fps <= 240):
        raise ValueError(f"{Path(path).name}: cannot determine video duration/fps (duration={dur}, fps={fps})")
    matrix = _first_frame_matrix(path, int(v.get("index", 0)))  # what ffmpeg's autorotate applies (incl. H.264/HEVC SEI)
    if matrix is None:
        matrix = next((m for sd in v.get("side_data_list") or [] if (m := runtime.parse_displaymatrix(sd.get("displaymatrix")))), None)
    rot, flip = runtime.display_transform(matrix)
    if matrix is None and _num((v.get("tags") or {}).get("rotate")) is not None:  # legacy tag: clockwise degrees
        rot = round(-_num(v["tags"]["rotate"]) / 90.0) * 90 % 360
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if rot in (90, 270):
        w, h = h, w
    if w <= 0 or h <= 0:
        raise ValueError(f"{Path(path).name}: invalid frame size {w}x{h}")
    return {"fps": float(fps), "width": w, "height": h, "duration_s": float(dur), "has_audio": bool(auds), "video_start_s": float(start),
            "audio_start_s": (_num(auds[0].get("start_time")) if auds else None), "rotation": rot, "flip": flip,
            "display_matrix": matrix, "nb_frames": nb, "video_index": int(v.get("index", 0))}


def _first_frame_matrix(path: Path, index: int) -> list[int] | None:
    """Display matrix of the first decoded frame of stream ``index`` (None if it has none or cannot be read)."""
    try:
        j = runtime.ffprobe_json(path, "-select_streams", str(index), "-read_intervals", "%+#1",
                                 "-show_entries", "frame_side_data=side_data_type,displaymatrix", timeout=60)
    except (RuntimeError, ValueError):
        return None
    for fr in j.get("frames") or []:
        for sd in fr.get("side_data_list") or []:
            if (m := runtime.parse_displaymatrix(sd.get("displaymatrix"))) is not None:
                return m
    return None


def probe(ep: Episode) -> None:
    """Stage "probe": ffprobe every stream (display size, video fps/start/duration, audio start, rotation)."""
    ep.set_status("probe", "running")
    info = {}
    try:
        for s in ep.streams:
            p = ep.dir / s.path
            if not p.exists():
                tgt = f" (dangling link -> {os.readlink(p)})" if p.is_symlink() else ""
                raise FileNotFoundError(f"stream {s.name}: {s.path} missing{tgt}")
            try:
                info[s.name] = ffprobe(p)
            except Exception as e:  # noqa: BLE001 - name the stream
                raise RuntimeError(f"stream {s.name}: {type(e).__name__}: {e}") from e
    except Exception as e:
        ep.set_status("probe", "failed", str(e))
        raise
    for s in ep.streams:
        for k, v in info[s.name].items():
            setattr(s, k, v)
    ep.set_status("probe", "done", "; ".join(f"{s.name} {s.width}x{s.height}@{s.fps:.2f} {s.duration_s:.1f}s" + (f" rot{s.rotation}" if s.rotation else "")
                                             + (" flip" if s.flip else "")
                                             + ("" if s.has_audio else " no-audio") for s in ep.streams))
