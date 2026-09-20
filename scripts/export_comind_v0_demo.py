#!/usr/bin/env python3
"""Render the completed CoMind V0 handover from verified caches and two MP4 clips.

No VRS, CSV, matching, selection, scan registration, or independent video clocks
are used. Only compact canonical geometry is retained; RGB frames stream into
PyAV and only the three requested decoded screenshots are saved.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from duet.adapters.comind.indexed_video import iter_indexed_rgb
from duet.adapters.comind.playback import load_playback
from duet.visualization.comind_handover import assemble_index_frame
from duet.visualization.demo_compositor import (
    HEIGHT,
    WIDTH,
    WORLD_PANEL,
    annotation_active,
    iter_composed_frames,
    make_thumbnail,
)
from duet.visualization.demo_encoding import encode_video, inspect_video
from duet.visualization.demo_geometry import fit_view

ROOT = Path(__file__).resolve().parents[1]
RECORDING_ID = "43276420-701f-4731-b9ab-bebc7fd14994"


def _outside_raw(path: Path, dataset_root: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to((ROOT / "data/raw").resolve()) or path.is_relative_to(
        dataset_root.resolve()
    ):
        raise ValueError("derived inputs and outputs must stay outside immutable raw data")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default=RECORDING_ID)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/comind")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed/comind")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/v0_demo")
    args = parser.parse_args(argv)
    recording_id = str(UUID(args.recording_id))
    output = _outside_raw(args.output_dir, args.dataset_root)
    report = _outside_raw(
        args.validation_report or ROOT / "outputs/comind_frame_validation" / f"{recording_id}.json",
        args.dataset_root,
    )
    bundle = load_playback(
        _outside_raw(args.processed_root, args.dataset_root), report, recording_id
    )
    if bundle.selection["annotation_id"] != "018240":
        raise ValueError("this V0 presentation requires the already-selected bowl handover 018240")
    frame_count = bundle.end_index - bundle.start_index + 1
    if frame_count != 253:
        raise ValueError("the verified selected V0 window must contain exactly 253 paired frames")
    snapshots, status_frames = [], []
    for frame in bundle.iter_frames():
        updates, status = assemble_index_frame(frame, bundle.bindings)
        snapshots.append((frame.frame_index, updates))
        status_frames.append(status)
    view = fit_view([updates for _, updates in snapshots], WORLD_PANEL[2:])
    trajectories = {
        role: np.asarray(
            [
                update.value.matrix[:3, 3]
                for _, updates in snapshots
                for update in updates
                if update.path == f"world/{role}/device" and update.kind == "pose"
            ]
        )
        for role in ("helper", "leader")
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        name: _outside_raw(output / f"duet_v0_demo{suffix}", args.dataset_root)
        for name, suffix in (
            ("video", ".mp4"),
            ("first", "_first.png"),
            ("middle", "_middle.png"),
            ("last", "_last.png"),
            ("thumbnail", "_thumbnail.png"),
            ("validation", "_validation.json"),
        )
    }
    videos = {
        role: iter_indexed_rgb(
            args.dataset_root / "recordings" / recording_id / "mp4s" / f"{role}_trimmed_sync.mp4",
            bundle.frame_maps[role].mp4_pts,
            bundle.frame_maps[role].mp4_time_base,
            start_index=bundle.start_index,
            end_index=bundle.end_index,
        )
        for role in ("helper", "leader")
    }
    start, end = int(bundle.selection["start_frame"]), int(bundle.selection["end_frame"])
    frames = iter_composed_frames(
        snapshots,
        videos,
        view=view,
        trajectories=trajectories,
        annotation_start=start,
        annotation_end=end,
    )
    print(f"Rendering {frame_count} paired frames with one fixed 3D viewpoint…", flush=True)
    try:
        encoded = encode_video(
            artifacts["video"], frames, size=(WIDTH, HEIGHT), fps=30, expected_frames=frame_count
        )
    finally:
        frames.close()
        for video in videos.values():
            video.close()
    verified = inspect_video(
        artifacts["video"],
        expected_size=(WIDTH, HEIGHT),
        expected_fps=30,
        expected_frames=frame_count,
        screenshot_paths={
            0: artifacts["first"],
            frame_count // 2: artifacts["middle"],
            frame_count - 1: artifacts["last"],
        },
    )
    with Image.open(artifacts["middle"]) as middle:
        make_thumbnail(middle.convert("RGB")).save(artifacts["thumbnail"])
    active_indices = [i for i, _ in snapshots if annotation_active(i, start, end)]
    hand_coverage = {
        role: {
            side: sum(
                bool(status["participants"][role]["hands"][side]["present"])
                for status in status_frames
            )
            for side in ("left", "right")
        }
        for role in ("helper", "leader")
    }
    result = {
        "recording_id": recording_id,
        "annotation_id": "018240",
        "timeline": "comind_sync_frame_index",
        "start_frame": bundle.start_index,
        "end_frame": bundle.end_index,
        "annotation_active_indices": active_indices,
        "relative_time_origin": "selected annotation start frame; presentation seconds at 30 fps",
        "encoding": encoded,
        "verified_video": verified,
        "fixed_view": {
            "size": view.size,
            "basis": view.basis.tolist(),
            "world_center_m": view.world_center.tolist(),
            "plane_center_m": view.plane_center.tolist(),
            "pixels_per_meter": view.pixels_per_meter,
            "padding_pixels": view.padding_pixels,
        },
        "rendered_hand_frames": hand_coverage,
        "scene_included": False,
        "object_pose_included": False,
        "hand_representation": "verified landmarks only; no invented connectivity",
        "validation_report": str(report),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "limitations": [
            "Scan registration unresolved; scene omitted",
            "No object 3D pose or occlusion-aware visibility",
            "Orthographic display uses one fixed view fitted to the whole clip",
        ],
    }
    artifacts["validation"].write_text(
        json.dumps(result, indent=2, default=lambda item: item.tolist()) + "\n"
    )
    print(json.dumps({"video": str(artifacts["video"]), **verified}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
