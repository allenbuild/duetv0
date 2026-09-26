"""Ingest a ZED export (folder or zip from scripts/zed_kit.py) into an episode.

Bundle layout (what zed_export.py writes):
    left.mp4  pose.csv  imu.csv  calibration.json  export.json  depth/000000.png ...

Ingest copies it to <episode>/zed/<stream>/ and returns the path of left.mp4 to register as the
stream's video. world.py picks the folder up automatically: factory intrinsics for `calib`, ZED
positional tracking for `headpose`, depth PNGs for metric hands in `world3d`.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REQUIRED = ("left.mp4", "calibration.json")


def is_bundle(path: Path) -> bool:
    p = Path(path)
    if p.is_dir():
        return all((p / r).exists() for r in REQUIRED)
    if p.suffix.lower() == ".zip" and p.is_file():
        try:
            names = set(zipfile.ZipFile(p).namelist())
        except zipfile.BadZipFile:
            return False
        return all(r in names for r in REQUIRED)
    return False


def ingest(bundle: Path, episode_dir: Path, stream: str) -> Path:
    """Copy/unzip the bundle to <episode_dir>/zed/<stream>/ and return the left.mp4 path."""
    dst = Path(episode_dir) / "zed" / stream
    dst.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle)
    if bundle.is_dir():
        for p in bundle.rglob("*"):
            if p.is_file():
                t = dst / p.relative_to(bundle); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    else:
        with zipfile.ZipFile(bundle) as zf:
            for m in zf.namelist():
                if m.endswith("/") or ".." in m or m.startswith("/"):
                    continue
                zf.extract(m, dst)
    if not all((dst / r).exists() for r in REQUIRED):
        shutil.rmtree(dst, ignore_errors=True)
        raise ValueError(f"{bundle} is not a ZED export bundle (needs {REQUIRED})")
    return dst / "left.mp4"
