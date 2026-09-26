"""Pipeline orchestration for the ego/exo playground."""
from __future__ import annotations

import threading
import traceback
from pathlib import Path

from . import align as _align, depth_mono as _dm, episode as _ep, export as _export, imu_arm as _imu, perception as _p, qc as _qc, scene_scan as _sc, world as _w

STAGE_FUNCS = {
    "probe": _ep.probe,
    "align": _align.align,
    "frames": _p.extract_frames,
    "calib": _w.calib,          # intrinsics + board registration (world = board)
    "body2d": _p.body2d,
    "hands": _p.hands,
    "objects": _p.objects,
    "tags": _w.tags,            # AprilTags in fixed views -> world poses
    "headpose": _w.headpose,    # ego camera pose per frame: ZED tracking or head tag
    "body3d": _p.body3d,        # monocular fallback
    "world3d": _w.world3d,      # triangulated bodies, metric hands, objects, cross-person features
    "imu_arm": _imu.imu_arm,
    "depth_mono": _dm.depth_mono,
    "scene_scan": _sc.scene_scan,
    "qc": _qc.qc,
    "export": _export.export,
}
ORDER = list(STAGE_FUNCS)
_running: dict[str, threading.Thread] = {}


def run_stages(ep: _ep.Episode, stages: list[str] | None = None, force: bool = False, log=print) -> None:
    for st in stages or ORDER:
        if not force and ep.status.get(st, {}).get("state") == "done":
            log(f"  {st}: cached"); continue
        log(f"  {st}: running")
        try:
            STAGE_FUNCS[st](ep)
            log(f"  {st}: {ep.status[st].get('state')} {ep.status[st].get('detail', '')}")
        except Exception as e:  # noqa: BLE001
            ep.set_status(st, "failed", f"{type(e).__name__}: {str(e)[:200]}")
            log(f"  {st}: FAILED {e}\n{traceback.format_exc()[-800:]}")
            if st in ("probe", "align", "frames"):
                break  # downstream stages need these


def run_in_background(ep_dir: Path, stages: list[str] | None, force: bool = False) -> bool:
    key = str(ep_dir)
    if key in _running and _running[key].is_alive():
        return False

    def target():
        ep = _ep.Episode.load(ep_dir)
        run_stages(ep, stages, force)

    t = threading.Thread(target=target, daemon=True); _running[key] = t; t.start()
    return True


def is_running(ep_dir: Path) -> bool:
    t = _running.get(str(ep_dir)); return bool(t and t.is_alive())
