#!/usr/bin/env python3
"""Ego/exo playground CLI.

  playground.py create NAME --ego NAME=PATH ... --exo NAME=PATH ... [--person NAME=LABEL] [--imu NAME=PATH.parquet]
                [--ref NAME] [--offset NAME=SECONDS] [--imu-offset NAME=SECONDS] [--fps 10] [--max-lag 120]
                [--min-overlap 5] [--device auto] [--hwaccel auto] [--copy | --move] [--overwrite]
  playground.py run NAME [--stages probe,align,...] [--force] [--drop-frames]
                [--offset NAME=SECONDS|none] [--imu-offset NAME=SECONDS|none] [--fps F] [--max-lag S] [--min-overlap S]
                [--device D] [--hwaccel H]
  playground.py serve [--port 8765] [--host 127.0.0.1] [--root DIR]
  playground.py fetch-models [--force] [GROUP|FILE ...]      (default group: MediaPipe, YOLO, Depth Anything V2 Small)

Episodes live under --root, else $PLAYGROUND_EPISODES, else data/playground/episodes/. Stream specs are NAME=PATH;
the person wearing an ego camera is given with --person NAME=LABEL. The old NAME=PATH:LABEL suffix is still
accepted when unambiguous (PATH itself is not an existing file but the part before the last ':' is), so paths
containing ':' (e.g. leader_2026-09-25T10:32:11.mp4) work. --offset sets a manual offset (t_ref = t_stream + offset)
for streams whose audio cannot be aligned (e.g. ZED left.mp4); --imu-offset NAME=SECONDS sets a manual IMU clock offset
(t_video = t_imu + SECONDS, t_imu = time_ms/1000 of the IMU file); "none" clears either. `run` applies config options to
the episode before running.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground import runtime  # noqa: E402
from duet.playground.episode import MAX_FPS, NAME_RE, Episode, EpisodeError  # noqa: E402
from duet.playground.run import DEFAULT_ORDER, SPECS, parse_stages, run_stages  # noqa: E402


def episodes_root(a: argparse.Namespace | None = None) -> Path:
    return Path(a.root).expanduser() if a is not None and getattr(a, "root", None) else runtime.episodes_root()


def _kv(spec: str, what: str) -> tuple[str, str]:
    name, sep, val = spec.partition("=")
    if not sep or not name or not val:
        raise argparse.ArgumentTypeError(f"{what} must be NAME=VALUE, got {spec!r}")
    return name, val


def parse_stream(spec: str, role: str) -> tuple[str, str, Path, str | None]:
    """NAME=PATH (or the old NAME=PATH:PERSON when unambiguous) -> (name, role, path, person)."""
    name, rest = _kv(spec, f"--{role}")
    if Path(rest).expanduser().is_file() or ":" not in rest:
        return name, role, Path(rest).expanduser(), None
    path, _, person = rest.rpartition(":")
    if path and person and "/" not in person and Path(path).expanduser().is_file():
        return name, role, Path(path).expanduser(), person
    raise argparse.ArgumentTypeError(f"--{role} {spec!r}: no such file {rest!r}" + (f" (nor {path!r} with person {person!r})" if path else "")
                                     + "; give the person with --person NAME=LABEL")


def fps_arg(x: str) -> float:
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {x!r}") from None
    if not (math.isfinite(v) and 0 < v <= MAX_FPS):
        raise argparse.ArgumentTypeError(f"--fps must be in (0, {MAX_FPS:g}], got {x}")
    return v


def lag_arg(x: str) -> float:
    try:
        v = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {x!r}") from None
    if not (math.isfinite(v) and 0 < v <= 3600):
        raise argparse.ArgumentTypeError(f"--max-lag must be in (0, 3600] seconds, got {x}")
    return v


def device_arg(x: str) -> str:
    if not runtime.valid_device(x):
        raise argparse.ArgumentTypeError(f"--device must be auto|cpu|mps|cuda[:N], got {x!r}")
    return x


def stages_arg(x: str) -> list[str] | None:
    try:
        return parse_stages(x)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def offsets_arg(specs: list[str], allow_none: bool, flag: str = "--offset") -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for spec in specs:
        name, val = _kv(spec, flag)
        if allow_none and val.lower() in ("none", "auto", "clear"):
            out[name] = None; continue
        try:
            v = float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} {spec!r}: seconds must be a number") from None
        if not math.isfinite(v):
            raise argparse.ArgumentTypeError(f"{flag} {spec!r}: seconds must be finite")
        out[name] = v
    return out


def _config_args(p: argparse.ArgumentParser, defaults: bool) -> None:
    p.add_argument("--offset", action="append", default=[], metavar="NAME=SECONDS", help="manual offset: t_ref = t_stream + SECONDS")
    p.add_argument("--imu-offset", action="append", default=[], metavar="NAME=SECONDS",
                   help="manual IMU clock offset: t_video = t_imu + SECONDS (t_video: the stream's video time, 0 = first video "
                        "frame; t_imu = time_ms/1000 of the IMU file; negative = IMU stamps run ahead of the video, eidon_10004 "
                        "~ -0.41 s); 'none' = automatic estimate; drift 0 when set")
    p.add_argument("--fps", type=fps_arg, default=10.0 if defaults else None, help=f"processing frame rate (0, {MAX_FPS:g}]")
    p.add_argument("--max-lag", type=lag_arg, default=120.0 if defaults else None, help="align search window, seconds")
    p.add_argument("--min-overlap", type=lag_arg, default=5.0 if defaults else None, help="align: minimum audio overlap, seconds")
    p.add_argument("--device", type=device_arg, default="auto" if defaults else None, help="torch device: auto|cpu|mps|cuda[:N]")
    p.add_argument("--hwaccel", choices=runtime.HWACCELS, default="auto" if defaults else None, help="ffmpeg decode acceleration")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ego/exo playground: create episodes, run the pipeline, serve the viewer.")
    ap.add_argument("--root", help="episodes directory (default $PLAYGROUND_EPISODES or data/playground/episodes)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="create an episode from videos")
    c.add_argument("name")
    c.add_argument("--ego", action="append", default=[], metavar="NAME=PATH", help="head-mounted video (repeatable)")
    c.add_argument("--exo", action="append", default=[], metavar="NAME=PATH", help="fixed-camera video (repeatable)")
    c.add_argument("--person", action="append", default=[], metavar="NAME=LABEL", help="who wears / is labelled by stream NAME")
    c.add_argument("--imu", action="append", default=[], metavar="NAME=PATH", help="IMU parquet for stream NAME")
    c.add_argument("--ref", help="time-reference stream (default: first ego; align may then pick one with better audio)")
    _config_args(c, defaults=True)
    m = c.add_mutually_exclusive_group()
    m.add_argument("--copy", action="store_true", help="copy the videos (default: relative symlinks)")
    m.add_argument("--move", action="store_true", help="move the videos into the episode (regular files only, never from data/raw; "
                   "originals are removed only after the episode is complete)")
    c.add_argument("--overwrite", action="store_true", help="replace an existing episode (its derived data is deleted; rig.json and "
                   "zed/ are kept; zed/<s>/ is moved aside when stream s gets another source)")
    c.add_argument("--keep-zed", action="store_true", help="with --overwrite: keep zed/<s>/ even when stream s gets another source")
    c.add_argument("--discard-old-streams", action="store_true",
                   help="with --overwrite: delete old stream/IMU files that are not links (moved or copied in; maybe the only copy)")
    r = sub.add_parser("run", help="run pipeline stages on an episode")
    r.add_argument("name", help="episode name (or directory)")
    r.add_argument("--stages", type=stages_arg, default=None, help=f"comma-separated subset of {','.join(SPECS)}; default {','.join(DEFAULT_ORDER)}")
    r.add_argument("--force", action="store_true", help="re-run the selected stages even when cached")
    r.add_argument("--drop-frames", action="store_true", help="delete extracted JPEGs after a successful export")
    r.add_argument("--ref", help="make stream NAME the time reference (user-chosen; align keeps it)")
    _config_args(r, defaults=False)
    s = sub.add_parser("serve", help="serve the viewer / API")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--host", default="127.0.0.1", help="bind address; anything but loopback requires PLAYGROUND_TOKEN")
    s.add_argument("--root", dest="serve_root", help="episodes directory (same as the global --root)")
    f = sub.add_parser("fetch-models", help=f"download pinned model weights into {runtime.models_dir()} (SHA-256 verified)")
    f.add_argument("models", nargs="*", help=f"groups ({', '.join(runtime.MODEL_GROUPS)}; default: default) or file names")
    f.add_argument("--force", action="store_true", help="replace files whose hash differs from the pin")
    return ap


def _is_loopback(host: str) -> bool:
    import ipaddress
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h.split("%")[0]).is_loopback
    except ValueError:
        return False


def cmd_create(a: argparse.Namespace) -> int:
    vids = [parse_stream(x, "ego") for x in a.ego] + [parse_stream(x, "exo") for x in a.exo]
    persons = dict(_kv(x, "--person") for x in a.person)
    unknown = sorted(set(persons) - {v[0] for v in vids})
    if unknown:
        raise argparse.ArgumentTypeError(f"--person for unknown stream(s) {unknown}")
    vids = [(n, role, p, persons.get(n, person)) for n, role, p, person in vids]
    imus = {k: Path(v).expanduser() for k, v in (_kv(x, "--imu") for x in a.imu)}
    ep = Episode.create(episodes_root(a), a.name, vids, imus, a.ref, mode="copy" if a.copy else "move" if a.move else "symlink",
                        overwrite=a.overwrite, proc_fps=a.fps, max_lag_s=a.max_lag, min_overlap_s=a.min_overlap, device=a.device,
                        hwaccel=a.hwaccel, offsets=offsets_arg(a.offset, allow_none=False),
                        imu_offsets=offsets_arg(a.imu_offset, allow_none=False, flag="--imu-offset"), keep_zed=a.keep_zed,
                        discard_old_streams=a.discard_old_streams)
    print("created", ep.dir)
    return 0


def _episode_dir(a: argparse.Namespace) -> Path:
    """An episode name under the episodes root (names of old episodes may contain dots), or a directory path."""
    if "/" in a.name or a.name.startswith(("~", ".")):
        d = Path(a.name).expanduser()
        if not (d / "episode.json").exists():
            raise EpisodeError(f"no episode at {a.name}")
        return d
    if not (NAME_RE.fullmatch(a.name) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", a.name)):
        raise EpisodeError(f"invalid episode name {a.name!r}")
    return episodes_root(a) / a.name


def cmd_run(a: argparse.Namespace) -> int:
    ep = Episode.load(_episode_dir(a))
    offsets = offsets_arg(a.offset, allow_none=True); imu_offsets = offsets_arg(a.imu_offset, allow_none=True, flag="--imu-offset")
    unknown = sorted((set(offsets) | set(imu_offsets) | ({a.ref} if a.ref else set())) - {s.name for s in ep.streams})
    if unknown:
        raise argparse.ArgumentTypeError(f"--offset/--imu-offset/--ref for unknown stream(s) {unknown}")
    changes = {k: v for k, v in (("proc_fps", a.fps), ("max_lag_s", a.max_lag), ("min_overlap_s", a.min_overlap), ("device", a.device),
                                 ("hwaccel", a.hwaccel)) if v is not None}
    if a.ref:
        changes.update(reference=a.ref, reference_explicit=True)
    fn = None
    if changes or offsets or imu_offsets:
        def fn(e: Episode) -> None:  # applied by run_stages under the run lock: never changes a running job's config
            for k, v in changes.items():
                setattr(e, k, v)
            for n, v in offsets.items():
                e.stream(n).offset_override_s = v
            for n, v in imu_offsets.items():
                e.stream(n).imu_offset_override_s = v
        print("config:", {**changes, **{f"offset[{k}]": v for k, v in offsets.items()}, **{f"imu_offset[{k}]": v for k, v in imu_offsets.items()}})
    res = run_stages(ep, a.stages, a.force, drop_frames=a.drop_frames, configure=fn)
    bad = {k: v for k, v in res.items() if v in ("failed", "interrupted")}
    return 1 if bad else 0


def cmd_serve(a: argparse.Namespace) -> int:
    root = a.serve_root or a.root
    if root:
        os.environ["PLAYGROUND_EPISODES"] = str(Path(root).expanduser().resolve())
    raw = os.environ.get("PLAYGROUND_TOKEN", ""); tok = raw.strip()  # same rule as the server: strip, then >= 16 chars
    if tok != raw:
        print("WARNING: PLAYGROUND_TOKEN has leading/trailing whitespace; the stripped value is used", file=sys.stderr)
    if not _is_loopback(a.host):
        if len(tok) < 16:
            print(f"refusing to serve on {a.host!r}: binding a non-loopback address requires PLAYGROUND_TOKEN (>= 16 chars; "
                  "e.g. `export PLAYGROUND_TOKEN=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')`)", file=sys.stderr)
            return 2
        print(f"WARNING: serving on {a.host}:{a.port} — every API/video request needs the token; the token and participant video "
              "travel in clear text unless a TLS reverse proxy fronts this port.", file=sys.stderr)
    import uvicorn
    from duet.playground import server
    app = server.create_app(os.environ.get("PLAYGROUND_EPISODES") or None) if hasattr(server, "create_app") else server.app
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")  # access log on
    return 0


def _clip_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("clip") is not None


def bake_world_vocab(names: list[str] | None, force: bool, P=None) -> str:
    """After fetching yolov8s-worldv2.pt: bake the objects vocabulary into it once (perception.bake_world_vocab), so the
    objects stage needs no CLIP at run time. Needs ultralytics + CLIP at setup time; without CLIP only a warning."""
    if "yolov8s-worldv2.pt" not in runtime.expand_model_names(names) or not (runtime.models_dir() / "yolov8s-worldv2.pt").is_file():
        return "not requested"
    if P is None:
        from duet.playground import perception as P
    dst = runtime.models_dir() / P.world_vocab_weights()
    if dst.is_file() and not force:
        print(f"{dst.name}: ok"); return "ok"
    if not _clip_available():
        print(f"WARNING: {dst.name} not baked: CLIP is not importable. The objects stage will need it: "
              "pip install git+https://github.com/ultralytics/CLIP.git, then re-run fetch-models", file=sys.stderr)
        return "no clip"
    out = P.bake_world_vocab()
    print(f"{Path(out).name}: baked (objects vocabulary, no CLIP needed at run time)")
    return "baked"


def cmd_fetch_models(a: argparse.Namespace) -> int:
    runtime.fetch_models(a.models or None, force=a.force)
    try:
        bake_world_vocab(a.models or None, a.force)
    except ImportError as e:
        print(f"WARNING: YOLO-World vocabulary not baked ({e}); the objects stage will need it", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return {"create": cmd_create, "run": cmd_run, "serve": cmd_serve, "fetch-models": cmd_fetch_models}[a.cmd](a)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))
    except (ValueError, FileNotFoundError, FileExistsError, EpisodeError, BlockingIOError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
