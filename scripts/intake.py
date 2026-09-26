#!/usr/bin/env python3
"""Batch import of SD-card dumps into playground episodes.

  intake.py ROOT --map cam_leader=ego:Idhant --map cam_helper=ego:Allen --map exoA=exo --map exoB=exo --ref cam_leader \
      [--prefix NAME] [--gap-min 5] [--dry-run] [--run] [--stages probe,align,...] [--episodes DIR] [--scratch DIR] [--no-drift]

ROOT holds one subfolder per camera, each with chunked recordings (VID_0001.MP4, VID_0002.MP4, ...). Steps:

1. ffprobe every file: duration + creation_time tag (falls back to file mtime - duration, flagged "mtime").
2. Order chunks per camera by name (natural sort); chunk start/end times come from the timestamps.
3. Split each camera's chunks into sessions where the gap between one chunk's end and the next start exceeds
   --gap-min (5 min). Sessions of the reference camera define the episodes; every other camera's sessions are
   matched to them by time overlap (by index if nothing overlaps but the counts agree).
4. Per session and camera: lossless concat (ffmpeg concat demuxer, -c copy) into a scratch folder.
5. Clock drift: audio envelope + cross-correlation (from playground.align) between each stream and the
   reference on a 60 s window at the START and at the END of their overlap. If the two offsets differ by more
   than --drift-ms (20 ms) the stream's clock runs at rate k = 1 + d_offset / span and the file is re-timed with
   ffmpeg (-filter:v setpts=PTS*k -af atempo=1/k, re-encoded) before the episode is created; the factor is
   recorded in the episode notes. Needs >= --drift-min-span (120 s) of overlap between the windows.
6. Episode.create(copy) named <prefix>_<session index> under --episodes (data/playground/episodes); --run starts
   the pipeline (--stages, default all).

--dry-run prints the plan (steps 1-3) and touches nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground.align import HOP, SR, _audio_envelope, _xcorr_lag
from duet.playground.episode import Episode

EPISODES = ROOT / "data/playground/episodes"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".mts", ".m4v"}
HOP_S = HOP / SR  # 10 ms


# ----------------------------------------------------------------------------- probing
@dataclass
class Chunk:
    path: Path
    duration_s: float
    start: dt.datetime | None  # UTC
    ts_source: str  # creation_time | mtime | none

    @property
    def end(self) -> dt.datetime | None:
        return self.start + dt.timedelta(seconds=self.duration_s) if self.start else None


@dataclass
class Camera:
    name: str
    role: str
    person: str | None
    chunks: list[Chunk] = field(default_factory=list)
    sessions: list[list[Chunk]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SessionPlan:
    index: int
    name: str
    cams: dict[str, list[Chunk]]  # camera -> chunks
    warnings: list[str] = field(default_factory=list)


def natural_key(p: Path) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def parse_time(s: str) -> dt.datetime | None:
    s = s.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            t = dt.datetime.strptime(s, fmt)
            return t if t.tzinfo else t.replace(tzinfo=dt.UTC)
        except ValueError:
            continue
    return None


def probe_chunk(path: Path) -> Chunk | None:
    """None when ffprobe cannot read the file (e.g. a chunk truncated when the camera died: no moov atom)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags=creation_time:stream=codec_type:stream_tags=creation_time",
                        "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    j = json.loads(r.stdout); dur = float(j.get("format", {}).get("duration", 0.0))
    if not j.get("streams") or dur <= 0:
        return None
    ct = j["format"].get("tags", {}).get("creation_time") or next((s.get("tags", {}).get("creation_time") for s in j.get("streams", []) if s.get("tags", {}).get("creation_time")), None)
    start = parse_time(ct) if ct else None
    if start is not None:
        return Chunk(path, dur, start, "creation_time")
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
    return Chunk(path, dur, mtime - dt.timedelta(seconds=dur), "mtime")


def list_chunks(cam_dir: Path) -> list[Path]:
    return sorted((p for p in cam_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.name.startswith(".")), key=natural_key)


# ----------------------------------------------------------------------------- sessions
def split_sessions(chunks: list[Chunk], gap_s: float) -> list[list[Chunk]]:
    """Chunks stay in the given (name) order; a new session starts when next.start - prev.end > gap_s.
    Chunks without timestamps never open a new session."""
    sessions: list[list[Chunk]] = []
    for c in chunks:
        if sessions and sessions[-1][-1].end is not None and c.start is not None and (c.start - sessions[-1][-1].end).total_seconds() > gap_s:
            sessions.append([c])
        elif sessions:
            sessions[-1].append(c)
        else:
            sessions.append([c])
    return sessions


def _span(sess: list[Chunk]) -> tuple[dt.datetime | None, dt.datetime | None]:
    starts = [c.start for c in sess if c.start]; ends = [c.end for c in sess if c.end]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _overlap_s(a: list[Chunk], b: list[Chunk]) -> float:
    a0, a1 = _span(a); b0, b1 = _span(b)
    if None in (a0, a1, b0, b1):
        return 0.0
    return (min(a1, b1) - max(a0, b0)).total_seconds()


def match_sessions(cams: dict[str, Camera], ref: str, prefix: str) -> list[SessionPlan]:
    plans = [SessionPlan(i + 1, f"{prefix}_{i + 1}", {ref: s}) for i, s in enumerate(cams[ref].sessions)]
    for name, cam in cams.items():
        if name == ref:
            continue
        assigned = {}
        for s in cam.sessions:
            ov = [(_overlap_s(s, p.cams[ref]), p) for p in plans]
            best = max(ov, key=lambda x: x[0]) if ov else (0.0, None)
            if best[0] > 0 and best[1].index not in assigned:
                assigned[best[1].index] = s
        if not assigned and len(cam.sessions) == len(plans):
            for p, s in zip(plans, cam.sessions):
                assigned[p.index] = s; p.warnings.append(f"{name}: no timestamp overlap with {ref}; sessions matched by index")
        for p in plans:
            if p.index in assigned:
                p.cams[name] = assigned[p.index]
            else:
                p.warnings.append(f"{name}: no session overlaps this one; camera omitted")
    return plans


def build_plan(root: Path, maps: dict[str, tuple[str, str | None]], ref: str, prefix: str, gap_s: float) -> tuple[dict[str, Camera], list[SessionPlan]]:
    cams: dict[str, Camera] = {}
    for cam_name, (role, person) in maps.items():
        d = root / cam_name
        if not d.is_dir():
            raise SystemExit(f"camera folder not found: {d}")
        cam = Camera(cam_name, role, person)
        for p in list_chunks(d):
            c = probe_chunk(p)
            if c is None:
                cam.warnings.append(f"{cam_name}: {p.name} unreadable (ffprobe failed), skipped")
            else:
                cam.chunks.append(c)
        if not cam.chunks:
            raise SystemExit(f"no readable video files in {d}")
        cam.sessions = split_sessions(cam.chunks, gap_s); cams[cam_name] = cam
    if ref not in cams:
        raise SystemExit(f"--ref {ref} is not in --map")
    plans = match_sessions(cams, ref, prefix)
    for p in plans:
        p.warnings += [w for cam in cams.values() for w in cam.warnings]
    return cams, plans


def print_plan(cams: dict[str, Camera], plans: list[SessionPlan], ref: str) -> None:
    for cam in cams.values():
        print(f"{cam.name} ({cam.role}{', ' + cam.person if cam.person else ''}{', reference' if cam.name == ref else ''}): {len(cam.chunks)} chunks, {len(cam.sessions)} session(s)")
        for c in cam.chunks:
            print(f"    {c.path.name:24s} {c.duration_s:8.2f}s  start {c.start.isoformat(timespec='seconds') if c.start else '-':25s} [{c.ts_source}]")
        for w in cam.warnings:
            print(f"    warning: {w}")
    for p in plans:
        print(f"episode {p.name}:")
        for cam_name, sess in p.cams.items():
            s0, s1 = _span(sess)
            print(f"    {cam_name:12s} {len(sess)} chunk(s), {sum(c.duration_s for c in sess):8.2f}s  {s0.isoformat(timespec='seconds') if s0 else '-'} -> {s1.isoformat(timespec='seconds') if s1 else '-'}")
        for w in p.warnings:
            print(f"    warning: {w}")


# ----------------------------------------------------------------------------- concat + drift
def concat_chunks(chunks: list[Chunk], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".txt")
    lst.write_text("".join("file '" + str(c.path.resolve()).replace("'", "'\\''") + "'\n" for c in chunks))
    # video + audio only: GoPro/Aria files carry data streams (tmcd timecode, GPMF) that the mp4 muxer rejects on copy
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v", "-map", "0:a?", "-c", "copy",
                    "-movflags", "+faststart", str(out)], check=True)
    return out


def _safe_max_lag(n: int, max_lag: int) -> int:
    """align._xcorr_lag pads to L = 2^ceil(log2(2n)) and slices lags -max_lag..max_lag out of it; keep that valid for short signals."""
    L = int(2 ** np.ceil(np.log2(max(2, 2 * n))))
    return max(1, min(max_lag, n - 1, (L - 1) // 2))


def estimate_drift(ref_env, sig_env, win_s: float = 60.0, max_lag_s: float = 120.0, min_span_s: float = 120.0, local_lag_s: float = 2.0) -> dict:
    """Offset of ``sig`` vs ``ref`` (t_ref = t_sig + offset) at the start and at the end of their overlap.

    Returns offset_start_s, offset_end_s, their confidences, the stream-time span between the two windows, the
    global lag, and k = 1 + (offset_end - offset_start) / span (the factor to multiply the stream's timestamps by).
    ``ok`` is False when the overlap is too short for two separated windows."""
    W = int(round(win_s / HOP_S))
    # global lag from the head of both recordings only (max_lag + 2 windows): over a whole hour a drifting clock smears the
    # correlation peak across many hops, and the head is enough to find a lag of up to max_lag_s
    n_g = min(len(ref_env), len(sig_env), int((max_lag_s + 2 * win_s) / HOP_S))
    lag, conf = _xcorr_lag(ref_env[:n_g], sig_env[:n_g], _safe_max_lag(n_g, int(max_lag_s / HOP_S)))
    i0 = max(0, -lag); i1 = min(len(sig_env), len(ref_env) - lag)
    res = {"lag_global_s": lag * HOP_S, "conf_global": conf, "k": 1.0, "ok": False, "span_s": 0.0}
    if i1 - W - i0 < min_span_s / HOP_S:
        res["reason"] = f"overlap {max(0, i1 - i0) * HOP_S:.0f}s too short for two {win_s:.0f}s windows {min_span_s:.0f}s apart"
        return res

    def local(a: int) -> tuple[float, float]:
        d, c = _xcorr_lag(ref_env[a + lag: a + lag + W], sig_env[a: a + W], int(local_lag_s / HOP_S))
        return (lag + d) * HOP_S, c

    (o0, c0), (o1, c1) = local(i0), local(i1 - W)
    span = (i1 - W - i0) * HOP_S
    res.update(offset_start_s=o0, offset_end_s=o1, conf_start=c0, conf_end=c1, span_s=span, k=1.0 + (o1 - o0) / span, ok=True)
    return res


def apply_drift(src: Path, dst: Path, k: float) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-filter:v", f"setpts=PTS*{k:.9f}", "-af", f"atempo={1 / k:.9f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)], check=True)
    return dst


def drift_correct(files: dict[str, Path], ref: str, scratch: Path, thr_ms: float, min_conf: float, win_s: float, min_span_s: float, log=print) -> tuple[dict[str, Path], list[str]]:
    notes = []
    ref_env = _audio_envelope(files[ref])
    if ref_env is None:
        return files, [f"intake: drift check skipped, reference {ref} has no audio"]
    out = dict(files)
    for name, path in files.items():
        if name == ref:
            continue
        env = _audio_envelope(path)
        if env is None:
            notes.append(f"intake: {name}: no audio, drift check skipped"); continue
        r = estimate_drift(ref_env, env, win_s=win_s, min_span_s=min_span_s)
        if not r["ok"]:
            notes.append(f"intake: {name}: drift not estimated ({r['reason']}); global offset {r['lag_global_s']:+.2f}s (x{r['conf_global']:.1f})"); continue
        delta_ms = (r["offset_end_s"] - r["offset_start_s"]) * 1000
        desc = f"offset {r['offset_start_s']:+.3f}s (x{r['conf_start']:.1f}) at start, {r['offset_end_s']:+.3f}s (x{r['conf_end']:.1f}) at end, {r['span_s']:.0f}s apart"
        if abs(delta_ms) <= thr_ms:
            notes.append(f"intake: {name}: {desc}: drift {delta_ms:+.0f}ms <= {thr_ms:.0f}ms, no correction")
        elif min(r["conf_start"], r["conf_end"]) < min_conf:
            notes.append(f"intake: {name}: {desc}: drift {delta_ms:+.0f}ms but window confidence < {min_conf}, NOT corrected")
        else:
            k = r["k"]; log(f"    {name}: drift {delta_ms:+.0f}ms over {r['span_s']:.0f}s -> k={k:.7f}, re-timing (re-encode)")
            out[name] = apply_drift(path, scratch / f"{name}_drift{path.suffix}", k)
            notes.append(f"intake: {name}: {desc}: drift {delta_ms:+.0f}ms -> k={k:.7f} applied (setpts=PTS*k, atempo=1/k, re-encoded)")
    return out, notes


# ----------------------------------------------------------------------------- execute
def execute(cams: dict[str, Camera], plans: list[SessionPlan], ref: str, episodes: Path, scratch: Path, drift: bool, run: bool, stages: list[str] | None,
            drift_ms: float = 20.0, drift_min_conf: float = 1.2, drift_win_s: float = 60.0, drift_min_span_s: float = 120.0, log=print) -> list[Episode]:
    eps = []
    for p in plans:
        if (episodes / p.name).exists():
            log(f"{p.name}: already exists, skipping"); continue
        log(f"{p.name}: concatenating {sum(len(s) for s in p.cams.values())} chunks from {len(p.cams)} cameras")
        sdir = scratch / p.name
        files = {cam: concat_chunks(chunks, sdir / f"{cam}{chunks[0].path.suffix.lower()}") for cam, chunks in p.cams.items()}
        notes = [f"intake: {cam}: {len(chunks)} chunk(s) concatenated: {', '.join(c.path.name for c in chunks)}" for cam, chunks in p.cams.items()]
        notes += [f"intake: warning: {w}" for w in p.warnings]
        if drift and ref in files and len(files) > 1:
            files, dn = drift_correct(files, ref, sdir, drift_ms, drift_min_conf, drift_win_s, drift_min_span_s, log); notes += dn
        elif drift:
            notes.append("intake: drift check skipped (reference missing from this session or single camera)")
        vids = [(cam, cams[cam].role, files[cam], cams[cam].person) for cam in p.cams]
        ep = Episode.create(episodes, p.name, vids, reference=ref if ref in files else None, link=False)
        ep.notes += notes; ep.save(); eps.append(ep)
        for n in notes:
            log(f"    {n}")
        log(f"    created {ep.dir}")
        if run:
            from duet.playground.run import run_stages
            run_stages(ep, stages, log=log)
    return eps


def parse_map(spec: str) -> tuple[str, tuple[str, str | None]]:
    cam, _, rest = spec.partition("=")
    role, _, person = rest.partition(":")
    if not cam or role not in ("ego", "exo"):
        raise SystemExit(f"bad --map {spec!r}: expected cam=ego:Person or cam=exo[:label]")
    return cam, (role, person or None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path); ap.add_argument("--map", action="append", default=[], required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--prefix", help="episode name prefix (default: ROOT folder name)"); ap.add_argument("--gap-min", type=float, default=5.0)
    ap.add_argument("--episodes", type=Path, default=EPISODES); ap.add_argument("--scratch", type=Path, help="concat/drift scratch folder (default: temp dir, deleted afterwards)")
    ap.add_argument("--no-drift", action="store_true"); ap.add_argument("--drift-ms", type=float, default=20.0); ap.add_argument("--drift-win", type=float, default=60.0)
    ap.add_argument("--drift-min-span", type=float, default=120.0)
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--stages", default=None, help="comma list for --run (default all)")
    a = ap.parse_args(argv)
    maps = dict(parse_map(m) for m in a.map)
    prefix = a.prefix or a.root.resolve().name
    cams, plans = build_plan(a.root, maps, a.ref, prefix, a.gap_min * 60)
    print_plan(cams, plans, a.ref)
    if a.dry_run:
        print("dry run: nothing written"); return
    tmp = None
    scratch = a.scratch or Path(tmp := tempfile.mkdtemp(prefix="intake_"))
    try:
        execute(cams, plans, a.ref, a.episodes, scratch, not a.no_drift, a.run, a.stages.split(",") if a.stages else None)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
