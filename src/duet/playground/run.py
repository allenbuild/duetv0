"""Pipeline orchestration for the ego/exo playground.

Stages are registered lazily ("module:function"), so ``import duet.playground.run`` needs neither cv2 nor torch.
Each stage has required upstream stages (``DEPS``), optional upstream stages whose results it reads (``USES``),
the paths under derived/ it owns (``STAGE_OUTPUTS``) and the groups of episode config it depends on.

Caching and invalidation
    fingerprint = hash(episode config groups the stage depends on, (state, fingerprint) of every DEPS/USES stage,
    source hash of the stage's module and of every duet.* module it imports (transitively, found statically), input
    file identities (size+mtime of streams/IMU, content hash of rig.json and zed/*/*.json, size+mtime of other zed
    files), sha256 of the model weight files it loads, StageSpec.params). device/hwaccel are excluded.
    A stage is "cached" only if state == "done" and the fingerprint matches; otherwise it re-runs (``force`` re-runs
    the selected stages regardless). Before and after every stage, and on EVERY exit of ``run_stages`` (also
    interrupts), "done" stages whose fingerprint no longer matches are marked "stale" (cascading downstream).
    ``fresh_stages``/``stage_fresh`` answer the same question without writing (readers: the server, the exporter), e.g.
    after a rig.json edit made outside a run.
    Unverifiable inputs keep results: a missing stream file (dangling link, unmounted drive) or weight file never
    invalidates a result by itself; when a stream file is missing, the stages that read the videos (probe, align,
    frames) are not run at all and every "done" stage written by the old code (no fingerprint) is kept. Episodes
    written by the old code otherwise stay done until one of their upstream stages changes.
    A stage whose DEPS are not all "done" is not run: a stage without a result is recorded "skipped" ("needs <dep>"),
    an existing result is left as is (files and record) and is marked "stale" only if its inputs changed. STAGE_OUTPUTS
    are deleted before a stage runs, and again when it fails, skips itself or is interrupted.

Execution
    ``run_stages`` runs in-process under a per-episode fcntl lock (<ep>/run.lock, the liveness test for everything:
    the kernel releases it when the process dies), writing a per-run log with full tracebacks to
    derived/logs/<timestamp>-<pid>.log (server jobs: <timestamp>-job.log). The server uses ``submit``: ONE worker thread
    (env PLAYGROUND_MAX_JOBS, default 1) runs queued jobs sequentially, each in a SUBPROCESS
    (``python -m duet.playground.run <ep_dir> [--stages a,b] [--force]``), so a native crash cannot kill the server;
    status["_job"] records each job (running / done / failed with the log tail / interrupted). ``cancel`` SIGTERMs
    the job's process group and SIGKILLs it after PLAYGROUND_KILL_GRACE_S (10 s). ``recover_stale`` turns "running"
    records of episodes whose run.lock is free into "interrupted".
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import runtime
from .episode import OPT_IN_STAGES, STAGES, Episode, EpisodeError, run_lock, run_lock_held

SRC = Path(__file__).resolve().parents[2]  # .../src, put on PYTHONPATH of the job subprocess


@dataclass(frozen=True)
class StageSpec:
    func: str | Callable[[Episode], Any]  # "package.module:function" (lazy) or a callable (tests)
    deps: tuple[str, ...] = ()  # required upstream stages (must be "done")
    uses: tuple[str, ...] = ()  # optional upstream stages whose outputs are read when done
    outputs: tuple[str, ...] = ()  # paths under derived/ owned by this stage
    config: tuple[str, ...] = ()  # episode config groups in the fingerprint (see CONFIG_GROUPS)
    sources: tuple[str, ...] = ()  # extra modules whose source is part of the fingerprint
    opt_in: bool = False  # runs only when named explicitly
    params: dict = field(default_factory=dict)  # keyword arguments for func (part of the fingerprint)
    models: tuple[str, ...] = ()  # weight files it loads from runtime.models_dir() (or "module:function" -> file name); hashed
    reads_sources: bool = False  # decodes the stream videos: not run while a stream file is missing


_P = "duet.playground."
_GEO = (_P + "geometry",)
_TL = ("streams", "timeline")
SPECS: dict[str, StageSpec] = {
    "probe": StageSpec(_P + "episode:probe", config=("files",), reads_sources=True),
    "align": StageSpec(_P + "align:align", deps=("probe",), config=("streams", "align"), reads_sources=True),
    "frames": StageSpec(_P + "perception:extract_frames", deps=("align",), outputs=("frames",), config=_TL, reads_sources=True),
    "calib": StageSpec(_P + "world:calib", deps=("frames",), outputs=("calib",), config=(*_TL, "rig", "zed"), sources=_GEO),
    "body2d": StageSpec(_P + "perception:body2d", deps=("frames",), outputs=("body2d",), config=_TL, models=("yolov8n-pose.pt",)),
    "hands": StageSpec(_P + "perception:hands", deps=("frames",), uses=("body2d",), outputs=("hands",), config=(*_TL, "persons"),
                       models=("hand_landmarker.task",)),  # body2d optional: without it, geometric wearer cues only
    "objects": StageSpec(_P + "perception:objects", deps=("frames",), outputs=("objects",), config=_TL,
                         models=("yolov8s-worldv2.pt", _P + "perception:world_vocab_weights")),  # + the vocabulary-baked file
    "tags": StageSpec(_P + "world:tags", deps=("frames", "calib"), outputs=("tags",), config=(*_TL, "rig"), sources=_GEO),
    "headpose": StageSpec(_P + "world:headpose", deps=("frames", "calib"), uses=("tags",), outputs=("headpose",),
                          config=(*_TL, "persons", "rig", "zed"), sources=_GEO),
    "body3d": StageSpec(_P + "perception:body3d", deps=("frames", "body2d"), outputs=("body3d",), config=_TL,
                        models=("pose_landmarker_lite.task",)),
    "depth_mono": StageSpec(_P + "depth_mono:depth_mono", deps=("frames",), uses=("calib",), outputs=("depth_mono",),
                            config=(*_TL, "rig", "zed"), sources=_GEO, opt_in=True),
    "scene_scan": StageSpec(_P + "scene_scan:scene_scan", deps=("frames", "calib"), uses=("body2d",), outputs=("scene_scan",),
                            config=(*_TL, "rig"), sources=_GEO, opt_in=True),
    "world3d": StageSpec(_P + "world:world3d", deps=("frames", "calib"), uses=("body2d", "hands", "tags", "headpose"),
                         outputs=("world3d",), config=(*_TL, "persons", "rig", "zed"), sources=_GEO),
    "qc": StageSpec(_P + "qc:qc", deps=("frames",), uses=("body2d", "hands"), outputs=("qc",), config=_TL),
    "imu_arm": StageSpec(_P + "imu_arm:imu_arm", deps=("align",), uses=("frames", "qc"), outputs=("imu_arm",), config=(*_TL, "persons", "imu")),
    "export": StageSpec(_P + "export:export", deps=("frames",),
                        uses=("calib", "body2d", "hands", "objects", "tags", "headpose", "body3d", "depth_mono", "scene_scan", "world3d", "qc", "imu_arm"),
                        outputs=("records.parquet",), config=(*_TL, "persons", "rig", "imu")),
}
assert tuple(SPECS) == STAGES and {s for s, v in SPECS.items() if v.opt_in} == set(OPT_IN_STAGES), "run.SPECS out of sync with episode.STAGES"
REGISTRY: dict[str, str] = {k: v.func for k, v in SPECS.items()}  # type: ignore[misc]
DEPS: dict[str, tuple[str, ...]] = {k: v.deps for k, v in SPECS.items()}
USES: dict[str, tuple[str, ...]] = {k: v.uses for k, v in SPECS.items()}
STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {k: tuple(f"derived/{o}" for o in v.outputs) for k, v in SPECS.items()}
ALL_STAGES: list[str] = list(SPECS)
DEFAULT_ORDER: list[str] = [k for k, v in SPECS.items() if not v.opt_in]
ORDER = DEFAULT_ORDER  # backward-compatible name


def parse_stages(stages: str | list[str] | tuple[str, ...] | None, specs: dict[str, StageSpec] | None = None) -> list[str] | None:
    """"a,b" or [a, b] -> validated list (ValueError naming the valid stages); empty -> None (= default order)."""
    specs = specs or SPECS
    items = [x.strip() for x in (stages.split(",") if isinstance(stages, str) else (stages or [])) if x and x.strip()]
    bad = [x for x in items if x not in specs]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid: {', '.join(specs)}")
    return items or None


def resolve(func: str | Callable) -> Callable[[Episode], Any]:
    if callable(func):
        return func
    mod, _, name = func.partition(":")
    return getattr(importlib.import_module(mod), name)


# ---------------------------------------------------------------------------------------------- fingerprints

_SRC_CACHE: dict[tuple, str] = {}
_IMPORTS_CACHE: dict[tuple, list[str]] = {}
_DUET = Path(__file__).resolve().parents[1]  # src/duet


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha(p: Path) -> str | None:
    try:
        st = p.stat(); key = (str(p), st.st_size, st.st_mtime_ns)
    except OSError:
        return None
    if key not in _SRC_CACHE:
        _SRC_CACHE[key] = _sha(p.read_bytes())
    return _SRC_CACHE[key]


def _duet_file(mod: str) -> Path | None:
    """Source file of a duet.* module WITHOUT importing it (``find_spec`` of a dotted name imports its parents)."""
    parts = mod.split(".")
    if parts[0] != "duet":
        return None
    base = _DUET.joinpath(*parts[1:])
    for cand in (base.with_suffix(".py") if len(parts) > 1 else None, base / "__init__.py"):
        if cand is not None and cand.is_file():
            return cand
    return None


def _duet_imports(path: Path) -> list[str]:
    """duet.* modules (and their parent packages) named by any import statement in ``path``, relative ones resolved;
    imports inside functions count (stages import lazily)."""
    try:
        st = path.stat(); key = (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return []
    if key not in _IMPORTS_CACHE:
        try:
            rel = path.resolve().relative_to(_DUET.parent).with_suffix("")
            mod = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts); pkg = mod if rel.name == "__init__" else mod.rpartition(".")[0]
        except ValueError:
            pkg = ""
        out: list[str] = []
        try:
            tree = ast.parse(path.read_bytes())
        except (SyntaxError, ValueError):
            tree = ast.Module(body=[], type_ignores=[])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                out += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                try:
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), pkg) if node.level else (node.module or "")
                except (ImportError, ValueError):
                    continue
                out += [base, *[f"{base}.{a.name}" for a in node.names]]  # "from . import runtime" names a module
        names = {m for m in out if m == "duet" or m.startswith("duet.")}
        names |= {".".join(m.split(".")[:i]) for m in names for i in range(1, m.count(".") + 1)}  # parent packages run too
        _IMPORTS_CACHE[key] = sorted(names)
    return _IMPORTS_CACHE[key]


def _source_files(func_or_module: str | Callable) -> list[Path]:
    """The stage's source file plus every duet.* module it imports, transitively."""
    try:
        if callable(func_or_module):
            f = inspect.getsourcefile(func_or_module); todo = [Path(f)] if f else []
        else:
            f = _duet_file(func_or_module.partition(":")[0])
            if f is None:  # not under src/duet (e.g. a test module): locate it without importing its parents' code
                spec = importlib.util.find_spec(func_or_module.partition(":")[0])
                f = Path(spec.origin) if spec and spec.origin else None
            todo = [f] if f else []
    except (ImportError, ValueError, TypeError, OSError):
        return []
    seen: dict[Path, None] = {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen[f] = None
        todo += [m for m in (_duet_file(n) for n in _duet_imports(f)) if m is not None and m not in seen]
    return sorted(seen)


def _source_hash(func_or_module: str | Callable) -> str:
    """sha256 over the stage's source and its transitive duet.* imports (no module is imported to compute it)."""
    files = _source_files(func_or_module)
    if not files:
        return f"unresolved:{func_or_module!r}"
    return _sha(json.dumps([[str(f.relative_to(_DUET.parent)) if f.is_relative_to(_DUET.parent) else f.name, _file_sha(f)]
                            for f in files]).encode())


def _identity(p: Path) -> list | None:
    """size + mtime of a (possibly symlinked) input file; None when missing."""
    try:
        st = p.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _zed(ep: Episode) -> dict:
    out: dict = {}
    zroot = ep.dir / "zed"
    if zroot.is_dir():
        for d in sorted(p for p in zroot.iterdir() if p.is_dir() and ".old-" not in p.name):
            out[d.name] = {f.name: (_file_sha(f) if f.suffix == ".json" else _identity(f)) for f in sorted(d.iterdir()) if f.is_file()}
    return out


def _files_raw(ep: Episode) -> dict[str, list]:
    return {s.name: [s.path, _identity(ep.dir / s.path)] for s in ep.streams}


CONFIG_GROUPS: dict[str, Callable[[Episode], Any]] = {
    "files": lambda ep: [[n, *v] for n, v in _files_raw(ep).items()],
    "streams": lambda ep: {"streams": [[s.name, s.role, s.path] for s in ep.streams]},
    "persons": lambda ep: {s.name: s.person for s in ep.streams},
    # the reference is an align INPUT only when the user chose it; otherwise align may pick it (an output: timeline)
    "align": lambda ep: {"max_lag_s": ep.max_lag_s, "min_overlap_s": ep.min_overlap_s, "proc_fps": ep.proc_fps,
                         "reference": ep.reference if ep.reference_explicit else None, "override": {s.name: s.offset_override_s for s in ep.streams}},
    "timeline": lambda ep: {"reference": ep.reference, "proc_fps": ep.proc_fps, "proc_size": ep.proc_size,
                            "window": [ep.common_start_s, ep.common_end_s],
                            "streams": {s.name: [s.offset_s, s.drift_ppm, s.offset_status, s.video_start_s, s.rotation, s.width, s.height,
                                                 s.fps, s.duration_s, *([s.flip] if s.flip else [])] for s in ep.streams}},
    "rig": lambda ep: _file_sha(ep.dir / "rig.json"),
    "zed": _zed,
    "imu": lambda ep: {s.name: [s.imu, _identity(ep.dir / s.imu), s.imu_offset_override_s] for s in ep.streams if s.imu},
}


def model_file(name: str) -> Path | None:
    """A stage's weight file in runtime.models_dir(), or None when it is not there."""
    p = runtime.models_dir() / name
    return p if p.is_file() else None


def _model_name(entry: str) -> str:
    """File name of a StageSpec.models entry; "module:function" entries name a derived file (e.g. perception's
    vocabulary-baked YOLO-World weights, whose name hashes the vocabulary)."""
    if ":" not in entry:
        return entry
    try:
        return str(resolve(entry)())
    except Exception:  # noqa: BLE001 - unresolvable here: tracked under the entry itself
        return entry


def model_hashes(ep: Episode, stage: str, specs: dict[str, StageSpec] | None = None) -> dict[str, str | None]:
    """sha256 of every weight file ``stage`` loads. A missing file (not fetched on this host) keeps the hash recorded
    when the stage last ran, so it never invalidates a result; a different file does. (depth_mono's HF model id and
    pinned revision are code constants / rig.json settings, already covered by the source and rig hashes.)"""
    rec = (ep.status.get(stage) or {}).get("models") or {}
    names = [_model_name(m) for m in (specs or SPECS)[stage].models]
    return {n: (_file_sha(p) if (p := model_file(n)) else rec.get(n)) for n in names}


def _up_state(ep: Episode, u: str) -> tuple[str | None, str | None]:
    """(effective state, fingerprint) of an upstream stage; frames dropped after export count as done."""
    st = ep.status.get(u) or {}
    state = "done" if st.get("state") == "stale" and st.get("dropped") else st.get("state")
    return state, (st.get("fingerprint") if state == "done" else None)


def fingerprint_parts(ep: Episode, stage: str, specs: dict[str, StageSpec] | None = None) -> dict[str, str]:
    """Short hashes of each input group of ``stage`` (stored in status as ``inputs``, used to explain staleness)."""
    specs = specs or SPECS; sp = specs[stage]
    h = lambda o: _sha(json.dumps(runtime.json_safe(o), sort_keys=True).encode())[:12]  # noqa: E731
    parts = {f"config.{g}": h(CONFIG_GROUPS[g](ep)) for g in sp.config}
    parts.update({f"upstream.{u}": h(_up_state(ep, u)) for u in dict.fromkeys((*sp.deps, *sp.uses)) if u in specs})
    parts["source"] = h([_source_hash(sp.func), *[_source_hash(m) for m in sp.sources]])
    if sp.params:
        parts["params"] = h(sp.params)
    if sp.models:
        parts["models"] = h(model_hashes(ep, stage, specs))
    return parts


def _digest(parts: dict[str, str]) -> str:
    return _sha(json.dumps(parts, sort_keys=True).encode())[:16]


def fingerprint(ep: Episode, stage: str, specs: dict[str, StageSpec] | None = None) -> str:
    return _digest(fingerprint_parts(ep, stage, specs))


def missing_sources(ep: Episode) -> list[str]:
    """Streams whose video file is missing (dangling link, unmounted drive)."""
    return [s.name for s in ep.streams if not (ep.dir / s.path).exists()]


def _unverifiable(ep: Episode, parts: dict[str, str], rec: dict) -> bool:
    """True when the only changed input is the identity of stream files that are MISSING now (every present file
    unchanged): the recorded result cannot be re-checked or recomputed, so it is kept rather than invalidated."""
    old = rec.get("inputs") or {}
    if [k for k in parts if old.get(k) != parts[k]] != ["config.files"] or not isinstance(rec.get("files"), dict):
        return False
    then, now = rec["files"], _files_raw(ep)
    changed = [n for n in now if now[n] != then.get(n)]
    return set(now) == set(then) and bool(changed) and all(now[n][1] is None and now[n][0] == then[n][0] for n in changed)


def _stale_reason(ep: Episode, stage: str, specs: dict[str, StageSpec]) -> str | None:
    rec = ep.status.get(stage) or {}; sp = specs[stage]
    if rec.get("fingerprint") is None:  # written by the old code: trust until an upstream changes
        for u in sp.deps:
            if _up_state(ep, u)[0] != "done":
                return f"needs {u} ({_up_state(ep, u)[0] or 'not run'})"
        changed = [u for u in (*sp.deps, *sp.uses) if u in specs and ((ep.status.get(u) or {}).get("fingerprint")
                   or _up_state(ep, u)[0] in ("stale", "failed", "running", "interrupted"))]
        return f"{', '.join(changed)} changed since this ran" if changed else None
    parts = fingerprint_parts(ep, stage, specs)
    if _digest(parts) == rec["fingerprint"] or _unverifiable(ep, parts, rec):
        return None
    old = rec.get("inputs") or {}
    diff = [k.split(".", 1)[-1] for k in parts if old.get(k) != parts[k]] or ["inputs"]
    return f"{', '.join(diff)} changed since this ran"


def _sweep(e: Episode, specs: dict[str, StageSpec], skip: set[str] | frozenset, marked: dict[str, str]) -> None:
    """Mark, in stage order (so it cascades), "done" stages whose inputs changed as "stale"; a dropped-frames record
    (stale + dropped, counted as done downstream) whose own inputs changed loses ``dropped`` so the cascade continues."""
    for st in specs:
        rec = e.status.get(st) or {}
        if st in skip or not (rec.get("state") == "done" or (rec.get("state") == "stale" and rec.get("dropped"))):
            continue
        why = _stale_reason(e, st, specs)
        if why:
            e.status[st] = {**rec, "state": "stale", "dropped": False, "detail": e.clean_detail(f"out of date: {why}; re-run {st}")}
            marked[st] = why


def stale_stages(ep: Episode, specs: dict[str, StageSpec] | None = None) -> dict[str, str]:
    """{stage: reason} for every result that is out of date now (cascaded), WITHOUT writing anything."""
    specs = specs or SPECS; marked: dict[str, str] = {}
    saved = ep.status; ep.status = {k: dict(v) if isinstance(v, dict) else v for k, v in saved.items()}
    try:
        _sweep(ep, specs, frozenset(), marked)
    finally:
        ep.status = saved
    return marked


def fresh_stages(ep: Episode, specs: dict[str, StageSpec] | None = None) -> set[str]:
    """Stages whose result can be served: state "done" AND its inputs (config, upstream, sources, code, weights) still
    match, with the same tolerance for unverifiable (missing) sources as ``run_stages``. Non-writing; compute once per
    request and gate every read on membership."""
    specs = specs or SPECS; stale = stale_stages(ep, specs)
    return {st for st in specs if (ep.status.get(st) or {}).get("state") == "done" and st not in stale}


def stage_fresh(ep: Episode, name: str, specs: dict[str, StageSpec] | None = None) -> bool:
    """``ep.stage_ok(name)`` that also verifies the fingerprint against the current inputs (see ``fresh_stages``)."""
    return name in fresh_stages(ep, specs)


def refresh_stale(ep: Episode, specs: dict[str, StageSpec] | None = None, skip: set[str] | frozenset = frozenset()) -> list[str]:
    """Mark "done" stages (not in ``skip``) whose inputs changed as "stale", in stage order (cascades)."""
    specs = specs or SPECS; marked: dict[str, str] = {}
    saved = ep.status; ep.status = {k: dict(v) if isinstance(v, dict) else v for k, v in saved.items()}
    try:
        _sweep(ep, specs, skip, marked)  # dry run on a copy: take the lock and write only when something is stale
    finally:
        ep.status = saved
    if marked:
        marked.clear(); ep.update(lambda e: _sweep(e, specs, skip, marked))
    return list(marked)


# ---------------------------------------------------------------------------------------------- running

def _stamp() -> str:
    t = time.time()
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


class _RunLog:
    """Per-run log: lines go to ``log`` (stdout by default) and to derived/logs/<ts>-<pid>.log with timestamps."""

    def __init__(self, ep: Episode, log: Callable[[str], Any], log_file: str | os.PathLike | None, stdout_is_log: bool):
        self.log, self.f = log, None
        logs = ep.derived / "logs"
        if log_file == "auto":
            log_file = logs / f"{_stamp()}-{os.getpid()}.log"
        self.path = Path(log_file) if log_file else None
        if self.path and not stdout_is_log:
            self.path.parent.mkdir(parents=True, exist_ok=True); self.f = open(self.path, "a", buffering=1)  # noqa: SIM115
        if self.path and self.path.parent.resolve() == logs.resolve():
            _prune_logs(logs)  # only the episode's own log dir, never wherever --log-file points
        try:
            self.rel = str(self.path.resolve().relative_to(ep.dir.resolve())) if self.path else None
        except ValueError:
            self.rel = self.path.name if self.path else None

    def __call__(self, msg: str) -> None:
        self.log(msg)
        if self.f:
            self.f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

    def close(self) -> None:
        if self.f:
            self.f.close()


def _prune_logs(d: Path, keep: int = 50) -> None:
    """Keep the newest ``keep`` run logs of an episode (derived/logs/*.log; subdirectories are left alone)."""
    try:
        logs = sorted(d.glob("*.log"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    for p in logs[:-keep]:
        with contextlib.suppress(OSError):
            p.unlink()


def _accepts(fn: Callable, name: str) -> bool:
    """Does ``fn`` declare a keyword parameter ``name`` (stages with their own caches take ``force``)?"""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _clear_outputs(ep: Episode, sp: StageSpec) -> None:
    base = ep.derived.resolve()
    for rel in sp.outputs:
        assert ".." not in Path(rel).parts and not Path(rel).is_absolute(), rel
        p = ep.derived / rel
        if p.is_symlink() or p.is_file():
            p.unlink()
        elif p.is_dir():
            assert p.resolve().is_relative_to(base), p
            shutil.rmtree(p)


def run_stages(ep: Episode, stages: list[str] | str | None = None, force: bool = False, log: Callable[[str], Any] = print, *,
               specs: dict[str, StageSpec] | None = None, drop_frames: bool = False, log_file: str | os.PathLike | None = "auto",
               stdout_is_log: bool = False, configure: Callable[[Episode], Any] | None = None) -> dict[str, str]:
    """Run ``stages`` (default: DEFAULT_ORDER; always executed in canonical order) on ``ep`` in this process.

    ``configure(ep)`` is applied (and validated) under the run lock before anything runs: config changes never touch
    an episode another pipeline is running. Returns {stage: "cached" | "kept" | "done" | "skipped" | "failed" |
    "interrupted"} ("kept": not run, the previous result left as is). Raises BlockingIOError if another pipeline holds
    the episode's run lock (nothing written), ValueError for unknown stages / invalid config."""
    specs = specs or SPECS
    sel = parse_stages(stages, specs) or [k for k, v in specs.items() if not v.opt_in]
    sel = [k for k in specs if k in set(sel)]
    ep.reload()
    results: dict[str, str] = {}
    with run_lock(ep.dir):
        if configure is not None:
            ep.update(lambda e: (configure(e), e.validate_config()))
        ep.validate_config()
        L = _RunLog(ep, log, log_file, stdout_is_log)
        L(f"run {ep.name}: stages={','.join(sel)} force={force} pid={os.getpid()} host={socket.gethostname()}")
        try:
            gone = missing_sources(ep)
            if gone:
                L(f"  stream file(s) missing: {', '.join(gone)}: probe/align/frames are not run and old-format results are kept")
            refresh_stale(ep, specs, skip=set(sel))
            for i, st in enumerate(sel):
                sp = specs[st]; rest = set(sel[i + 1:])
                parts = fingerprint_parts(ep, st, specs); fp = _digest(parts)
                rec = ep.status.get(st) or {}; models = model_hashes(ep, st, specs); done = rec.get("state") == "done"
                if done and not force and (rec.get("fingerprint") == fp or _unverifiable(ep, parts, rec) or (gone and rec.get("fingerprint") is None)):
                    results[st] = "cached"; L(f"  {st}: cached" + ("" if rec.get("fingerprint") == fp else " (stream file missing: kept)")); continue
                if sp.reads_sources and gone:  # cannot decode missing videos: the record is left as is
                    why = f"stream file(s) missing: {', '.join(gone)}"
                    if done or rec.get("state") == "stale":
                        results[st] = "kept"; L(f"  {st}: not run ({why}); previous result kept")
                    else:
                        ep.set_status(st, "skipped", why); results[st] = "skipped"; L(f"  {st}: skipped ({why})")
                    continue
                missing = [d for d in sp.deps if not ep.stage_ok(d)]
                if missing:  # not run
                    why = "needs " + ", ".join(f"{d} ({(ep.status.get(d) or {}).get('state', 'not run')})" for d in missing)
                    if done or rec.get("state") == "stale":  # keep the result: marked stale below only if its inputs changed
                        results[st] = "kept"; L(f"  {st}: not run ({why}); previous result kept")
                    else:
                        ep.set_status(st, "skipped", why); results[st] = "skipped"; L(f"  {st}: skipped ({why})")
                    refresh_stale(ep, specs, skip=rest); continue
                _clear_outputs(ep, sp)
                ep._fp[st] = fp
                ep.set_status(st, "running", **({"log": L.rel} if L.rel else {}))
                L(f"  {st}: running"); t0 = time.time()
                try:
                    fn = resolve(sp.func)
                    fn(ep, **sp.params, **({"force": True} if force and _accepts(fn, "force") else {}))
                except (KeyboardInterrupt, SystemExit):
                    ep.set_status(st, "interrupted", "interrupted (cancelled or killed)"); _clear_outputs(ep, sp)
                    results[st] = "interrupted"; L(f"  {st}: INTERRUPTED"); raise
                except Exception as e:  # noqa: BLE001 - recorded, pipeline continues with independent stages
                    L(traceback.format_exc().rstrip())
                    ep.set_status(st, "failed", f"{type(e).__name__}: {e}" + (f" [log: {L.rel}]" if L.rel else ""))
                finally:
                    ep._fp.pop(st, None)
                ep.reload(); rec = ep.status.get(st) or {}
                if rec.get("state") == "running":
                    ep.set_status(st, "failed", "stage returned without recording a final state"); rec = ep.status.get(st) or {}
                if rec.get("state") == "done":
                    extra = {"fingerprint": fp, "inputs": parts, **({"models": models} if models else {}),
                             **({"files": _files_raw(ep)} if "files" in sp.config else {})}
                    ep.update(lambda e, st=st, extra=extra: e.status[st].update(extra))
                else:
                    _clear_outputs(ep, sp)  # failed / skipped: no (partial) artifact is left for consumers to misread
                rec = ep.status.get(st) or {}
                results[st] = rec.get("state", "failed")
                L(f"  {st}: {results[st]} in {time.time() - t0:.1f}s  {rec.get('detail', '')}")
                refresh_stale(ep, specs, skip=rest)
            if drop_frames:
                _drop_frames(ep, L)
        finally:
            try:  # every exit (also interrupts): no "done" record may outlive a change of its inputs
                marked = refresh_stale(ep, specs)
                if marked:
                    L(f"  marked stale: {', '.join(marked)}")
            except Exception as e:  # noqa: BLE001 - never mask the original exception
                L(f"  final staleness check failed: {type(e).__name__}: {e}")
            L(f"run {ep.name}: finished {results}"); L.close()
    return results


def _drop_frames(ep: Episode, L: Callable[[str], Any]) -> None:
    """Delete extracted JPEGs after a successful export; frames becomes "stale" (dropped) with its fingerprint kept."""
    if not ep.stage_ok("export") or not ep.stage_ok("frames"):
        L("  drop-frames: export/frames not done; frames kept"); return
    n = 0
    for f in (ep.derived / "frames").glob("*/*.jpg"):
        f.unlink(); n += 1
    ep.set_status("frames", "stale", f"{n} JPEGs deleted after export (--drop-frames); re-run frames to restore them", dropped=True)
    L(f"  drop-frames: deleted {n} JPEGs")


# ---------------------------------------------------------------------------------------------- background queue

JOB_EXTRA_ARGS: list[str] = []  # appended to every job command (tests use ["--specs", "module:ATTR"])
_QC = threading.Condition()
_QUEUE: list[dict] = []
_ACTIVE: dict[str, dict] = {}
_WORKERS: list[threading.Thread] = []


def _key(ep_dir: str | os.PathLike) -> str:
    return os.path.realpath(ep_dir)


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _max_jobs() -> int:
    try:
        return max(1, int(os.environ.get("PLAYGROUND_MAX_JOBS", "1")))
    except ValueError:
        return 1


def submit(ep_dir: str | os.PathLike, stages: list[str] | str | None = None, force: bool = False, *, drop_frames: bool = False) -> bool:
    """Queue a pipeline run (executed by the worker thread in a subprocess). False if this episode is already queued
    or running (in this process's queue, or any process holding its run lock)."""
    key = _key(ep_dir); sel = parse_stages(stages)
    if not (Path(key) / "episode.json").exists():
        raise EpisodeError(f"no episode at {Path(key).name}")
    with _QC:
        if key in _ACTIVE or any(j["key"] == key for j in _QUEUE) or run_lock_held(key):
            return False
        _QUEUE.append({"key": key, "episode": Path(key).name, "stages": sel, "force": bool(force), "drop_frames": bool(drop_frames),
                       "queued": time.time()})
        while len([t for t in _WORKERS if t.is_alive()]) < _max_jobs():
            t = threading.Thread(target=_worker, name="playground-worker", daemon=True); _WORKERS.append(t); t.start()
        _QC.notify_all()
    return True


run_in_background = submit  # old name


def is_running(ep_dir: str | os.PathLike) -> bool:
    """Queued or running here, or another process (CLI) holds the episode's run lock."""
    key = _key(ep_dir)
    with _QC:
        if key in _ACTIVE or any(j["key"] == key for j in _QUEUE):
            return True
    return run_lock_held(key)


def queue_state() -> dict:
    """{"max_jobs", "running": [...], "queued": [...]}; jobs carry episode name, stages, force, times, pid."""
    pub = lambda j: {k: j.get(k) for k in ("episode", "stages", "force", "queued", "started", "pid", "log")}  # noqa: E731
    with _QC:
        return {"max_jobs": _max_jobs(), "running": [pub(j) for j in _ACTIVE.values()], "queued": [pub(j) for j in _QUEUE]}


def cancel(ep_dir: str | os.PathLike) -> bool:
    """Drop a queued job, or cancel a running one: SIGTERM to its process group (the stage ends "interrupted"), SIGKILL
    after PLAYGROUND_KILL_GRACE_S (default 10 s). A job taken off the queue but not spawned yet is cancelled before it
    starts. True if something was cancelled."""
    key = _key(ep_dir)
    with _QC:
        for j in _QUEUE:
            if j["key"] == key:
                _QUEUE.remove(j); return True
        job = _ACTIVE.get(key)
        if not job or job.get("cancelled"):
            return False
        job["cancelled"] = time.time(); proc = job.get("proc")
    if proc is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
    return True


def _worker() -> None:
    while True:
        with _QC:
            while not _QUEUE:
                _QC.wait()
            job = _QUEUE.pop(0); _ACTIVE[job["key"]] = job
        try:
            _run_job(job)
        except Exception:  # noqa: BLE001 - never let the worker die
            traceback.print_exc()
        finally:
            with _QC:
                _ACTIVE.pop(job["key"], None)


def _set_job(ep_dir: Path, rec: dict) -> None:
    """status["_job"]: the latest background job of this episode (running / done / failed / interrupted)."""
    with contextlib.suppress(EpisodeError, OSError):
        Episode.load(ep_dir).update(lambda e: e.status.__setitem__("_job", runtime.json_safe(rec)))


def _log_tail(p: Path, n: int = 1500) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(max(0, p.stat().st_size - n)); return f.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _run_job(job: dict) -> int:
    ep_dir = Path(job["key"]); logs = ep_dir / "derived" / "logs"; logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{_stamp()}-job.log"; rel = str(log_path.relative_to(ep_dir))
    base = {"stages": job["stages"], "force": job["force"], "queued": job["queued"], "log": rel, "host": socket.gethostname()}
    with _QC:
        cancelled_early = bool(job.get("cancelled"))
    if cancelled_early:  # cancelled between leaving the queue and spawning
        _set_job(ep_dir, {**base, "state": "interrupted", "detail": "cancelled before it started", "finished": time.time()}); return -signal.SIGTERM
    cmd = [sys.executable, "-m", "duet.playground.run", str(ep_dir), "--log-file", str(log_path), "--stdout-is-log"]
    cmd += (["--stages", ",".join(job["stages"])] if job["stages"] else []) + (["--force"] if job["force"] else [])
    cmd += (["--drop-frames"] if job.get("drop_frames") else []) + list(JOB_EXTRA_ARGS)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]), "PYTHONUNBUFFERED": "1"}
    grace = _env_float("PLAYGROUND_KILL_GRACE_S", 10.0)
    with open(log_path, "ab") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        with _QC:
            job.update(proc=p, pid=p.pid, started=time.time(), log=rel); cancelled = job.get("cancelled")
        if cancelled:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGTERM)
        _set_job(ep_dir, {**base, "state": "running", "pid": p.pid, "started": job["started"]})
        rc, killed_at = None, None
        while rc is None:
            try:
                rc = p.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                c = job.get("cancelled")
                if c and killed_at is None and time.time() - c > grace:  # SIGTERM ignored or stuck: escalate
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(p.pid, signal.SIGKILL)
                    killed_at = time.time()
                elif killed_at is not None and time.time() - killed_at > 30:  # unkillable (e.g. stuck in I/O): stop waiting
                    rc = -signal.SIGKILL
    job["returncode"] = rc
    end = {**base, "pid": p.pid, "started": job["started"], "finished": time.time(), "returncode": rc}
    if rc == 0:
        _set_job(ep_dir, {**end, "state": "done"})
    else:
        why = ("cancelled" if job.get("cancelled") else f"pipeline process killed by {_signame(-rc)}" if rc < 0 else
               f"pipeline process exited with code {rc}")
        state = "interrupted" if job.get("cancelled") else "failed"
        _mark_running(ep_dir, state, f"{why}; see {rel}", pids={p.pid})
        _set_job(ep_dir, {**end, "state": state, "detail": f"{why}: {_log_tail(log_path)}"})
    return rc


def _signame(n: int) -> str:
    try:
        return signal.Signals(n).name
    except ValueError:
        return f"signal {n}"


def _mark_running(ep_dir: Path, state: str, detail: str, pids: set | None = None) -> list[str]:
    """Set every "running" stage (optionally only those of ``pids``) to ``state``."""
    marked: list[str] = []
    try:
        ep = Episode.load(ep_dir)
    except EpisodeError:
        return marked

    def fn(e: Episode) -> None:
        for st, v in e.status.items():
            if isinstance(v, dict) and v.get("state") == "running" and (pids is None or v.get("pid") in pids):
                e.status[st] = {**v, "state": state, "detail": e.clean_detail(detail), "finished": time.time()}; marked.append(st)
    ep.update(fn)
    return marked


def recover_stale(episodes_root: str | os.PathLike | None = None) -> list[tuple[str, str]]:
    """Mark "running" records (stages and "_job") of episodes nobody is processing as "interrupted". Liveness is the
    fcntl run.lock that every pipeline holds (released by the kernel when its process dies; immune to pid reuse):
    episodes queued here or whose lock is held are left alone, as are records written on another host."""
    root = Path(episodes_root or runtime.episodes_root()); host = socket.gethostname(); out: list[tuple[str, str]] = []
    for j in sorted(root.glob("*/episode.json")):
        d = j.parent
        if is_running(d):
            continue
        try:
            ep = Episode.load(d)
        except EpisodeError:
            continue
        dead = {st for st, v in ep.status.items() if isinstance(v, dict) and v.get("state") == "running" and v.get("host") in (None, host)}
        if not dead:
            continue

        def fn(e: Episode, dead=dead) -> None:
            for st in dead:
                v = e.status.get(st) or {}
                if v.get("state") == "running":
                    e.status[st] = {**v, "state": "interrupted", "finished": time.time(),
                                    "detail": f"interrupted: process {v.get('pid')} on {v.get('host') or '?'} is gone (restart or crash)"}
                    out.append((e.name, st))
        ep.update(fn)
    return out


# ---------------------------------------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m duet.playground.run", description="Run playground stages on one episode.")
    ap.add_argument("episode", help="episode directory")
    ap.add_argument("--stages", default="", help=f"comma-separated subset of: {','.join(SPECS)} (default: {','.join(DEFAULT_ORDER)})")
    ap.add_argument("--force", action="store_true", help="re-run the selected stages even if cached")
    ap.add_argument("--drop-frames", action="store_true", help="delete extracted JPEGs after a successful export")
    ap.add_argument("--log-file", default="auto", help="per-run log path (default derived/logs/<ts>-<pid>.log)")
    ap.add_argument("--stdout-is-log", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--specs", help=argparse.SUPPRESS)  # "module:ATTR" = dict of StageSpec replacing SPECS (tests)
    a = ap.parse_args(argv)

    def _term(signum, frame):  # cancel -> SystemExit inside the running stage -> "interrupted"
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    try:
        specs = None
        if a.specs:
            mod, _, attr = a.specs.partition(":"); specs = getattr(importlib.import_module(mod), attr)
        stages = parse_stages(a.stages, specs)
        ep = Episode.load(a.episode)
        run_stages(ep, stages, a.force, specs=specs, drop_frames=a.drop_frames, log_file=a.log_file, stdout_is_log=a.stdout_is_log)
    except BlockingIOError as e:
        print(f"error: {e}", file=sys.stderr); return 3
    except (ValueError, EpisodeError) as e:
        print(f"error: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
