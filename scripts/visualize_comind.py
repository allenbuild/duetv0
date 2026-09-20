#!/usr/bin/env python3
"""Save a verified paired-index handover or a static spatial diagnostic.

Reproduce without reading raw data or launching a GUI::

    .venv/bin/python scripts/visualize_comind.py --recording-id UUID --diagnostic

Use --synchronized after building and validating per-frame MP4/VRS maps.
There is no force option. Diagnostic mode reads only existing spatial caches
and reports; it never follows raw source paths recorded in their provenance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import UUID
from zipfile import BadZipFile

import numpy as np

from duet.schemas.common import FrameId, Provenance
from duet.visualization.rerun_episode import save_spatial_diagnostic

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("helper", "leader")
DISABLED_LAYERS = (
    "current participant poses",
    "hands at a common instant",
    "synchronized ego video",
    "camera frustums",
    "active handover",
    "annotation bounding boxes",
    "source-label participant assignments",
    "scan registration",
)
SYNCHRONIZATION_BLOCKED = (
    "Select --synchronized for a handover with VERIFIED per-frame MP4/VRS mappings on the "
    "common video index, or --diagnostic for static spatial evidence. The official MP4 "
    "extractor supplied only one DEVICE_TIME value per video; validated frame maps are "
    "required for synchronized mode. UTC and fitted offsets cannot replace them."
)


def _outside_raw(path: Path) -> Path:
    """Resolve symlinks before any input read or output creation."""
    resolved = path.resolve()
    raw = (REPOSITORY_ROOT / "data" / "raw").resolve()
    if resolved.is_relative_to(raw):
        raise ValueError("diagnostic inputs and outputs must stay outside immutable data/raw")
    return resolved


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path) -> dict[str, object]:
    with _outside_raw(path).open(encoding="utf-8") as stream:
        document = json.load(
            stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    if not isinstance(document, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return document


def _world(document: Mapping[str, object], recording_id: str) -> tuple[str, FrameId]:
    if document.get("recording_id") != recording_id:
        raise ValueError("cache/report recording identity does not match --recording-id")
    shared = document.get("shared_world")
    if not isinstance(shared, Mapping) or shared.get("status") != "VERIFIED":
        raise ValueError("shared-world graph/frame metadata is not VERIFIED")
    graph = shared.get("graph_uid")
    if not isinstance(graph, str) or not graph.strip():
        raise ValueError("shared-world graph UID must be present")
    expected_frame = f"comind/{recording_id}/multislam/{graph}"
    if shared.get("frame") != expected_frame:
        raise ValueError("shared-world frame does not match recording and graph UID")
    return graph, FrameId(expected_frame)


def _participant_metadata(document: Mapping[str, object], role: str, graph: str) -> Mapping:
    participants = document.get("participants")
    if not isinstance(participants, Mapping) or set(participants) != set(ROLES):
        raise ValueError("metadata must identify exactly helper and leader")
    participant = participants[role]
    if not isinstance(participant, Mapping):
        raise TypeError(f"invalid {role} metadata")
    source, cache = participant.get("source"), participant.get("cache")
    if not isinstance(source, Mapping) or source.get("graph_uid") != graph:
        raise ValueError(f"{role} source graph does not match shared-world graph")
    invalid_count = source.get("transform_invalid_count")
    if (
        type(invalid_count) is not int
        or invalid_count != 0
        or source.get("unchanged_during_read") is not True
    ):
        raise ValueError(f"{role} source transform/integrity audit is not valid")
    if not isinstance(cache, Mapping) or cache.get("participant") != role:
        raise ValueError(f"{role} cache participant identity does not match")
    return participant


def load_diagnostic(
    processed_root: Path, reports_root: Path, recording_id: str
) -> tuple[FrameId, dict[str, np.ndarray], dict[str, object], Provenance]:
    """Validate cached spatial identities without reading any raw source file.

    Position units are explicit in the cache field ``trajectory_translation_m``.
    Cache evidence is historical; source CSVs are deliberately not reopened or
    revalidated. Only static point sets are returned, never current poses.
    """
    if str(UUID(recording_id)) != recording_id:
        raise ValueError("recording-id must be a canonical UUID")
    cache_root = _outside_raw(processed_root) / recording_id / "semantics"
    reports_root = _outside_raw(reports_root)
    manifest_path = _outside_raw(cache_root / "manifest.json")
    clock_path = _outside_raw(reports_root / f"{recording_id}.json")
    video_path = _outside_raw(reports_root / f"{recording_id}_video_annotations.json")
    manifest = _read_json(manifest_path)
    clock_report = _read_json(clock_path)
    video_report = _read_json(video_path)
    graph, world = _world(manifest, recording_id)
    if _world(clock_report, recording_id) != (graph, world):
        raise ValueError("semantic report and cache shared-world identities differ")
    if video_report.get("recording_id") != recording_id:
        raise ValueError("video/annotation report recording identity differs")
    if not isinstance(video_report.get("mappings"), Mapping):
        raise TypeError("video/annotation semantic evidence is missing")
    trajectories = {}
    cache_checks = {}
    provenance_sources = [manifest_path, clock_path, video_path]
    for role in ROLES:
        metadata = _participant_metadata(manifest, role, graph)
        reported = _participant_metadata(clock_report, role, graph)
        if metadata["cache"] != reported["cache"] or metadata["source"] != reported["source"]:
            raise ValueError(f"{role} cache and report provenance disagree")
        path = _outside_raw(cache_root / f"{role}.npz")
        expected_bytes = metadata["cache"].get("bytes")
        if type(expected_bytes) is not int or expected_bytes != path.stat().st_size:
            raise ValueError(f"{role} cache byte size differs from audited manifest")
        with np.load(path, allow_pickle=False) as cached:
            cached_graph = cached["graph_uid"]
            if cached_graph.shape != () or cached_graph.dtype.kind not in ("U", "S"):
                raise ValueError(f"{role} cached graph UID must be a string scalar")
            if cached_graph.item() != graph:
                raise ValueError(f"{role} cached graph differs from verified shared graph")
            points = cached["trajectory_translation_m"]
        if points.dtype.kind not in ("f", "i", "u"):
            raise ValueError(f"{role} trajectory positions must be real numeric meters")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"{role} trajectory positions must be finite (N, 3) meters")
        expected_count = metadata["cache"].get("trajectory_sample_count")
        if type(expected_count) is not int or expected_count != len(points) or not len(points):
            raise ValueError(f"{role} trajectory sample count differs from audited manifest")
        trajectories[role] = points
        cache_checks[role] = {
            "path": str(path),
            "point_count": len(points),
            "graph_uid": graph,
            "distance_unit": "meters",
            "cache_byte_size_matches_manifest": True,
            "source_csv_reopened": False,
        }
        provenance_sources.append(path)
    status = {
        "mode": "static spatial diagnostic; no common timeline",
        "disabled_layers": list(DISABLED_LAYERS),
        "cache_checks": cache_checks,
        "cache_provenance_scope": "Historical audit reused; raw sources were not reopened.",
        "clock_and_motion_evidence": clock_report,
        "video_and_annotation_evidence": video_report,
    }
    provenance = Provenance(
        str(manifest_path),
        detail="Static derived trajectory points only, in meters",
        parents=tuple(Provenance(str(path)) for path in provenance_sources),
    )
    return world, trajectories, status, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True, help="Recording UUID")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostic", action="store_true", help="Static spatial evidence only")
    mode.add_argument("--synchronized", action="store_true", help="Verified selected handover")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument("--reports-root", type=Path, default=Path("outputs/comind_semantics"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--frame-validation-report", type=Path)
    parser.add_argument("--output", type=Path, help="Destination .rrd outside raw data")
    args = parser.parse_args(argv)
    if not args.diagnostic and not args.synchronized:
        parser.error(SYNCHRONIZATION_BLOCKED)
    try:
        recording_id = str(UUID(args.recording_id))
        if args.synchronized:
            return _synchronized(args, recording_id)
        output = _outside_raw(
            args.output
            or (Path("outputs/comind_visualization") / f"{recording_id}_spatial_diagnostic.rrd")
        )
        if output.suffix != ".rrd":
            raise ValueError("diagnostic output must end in .rrd")
        world, trajectories, status, provenance = load_diagnostic(
            args.processed_root, args.reports_root, recording_id
        )
        saved = save_spatial_diagnostic(
            output,
            world_frame=world,
            trajectories=trajectories,
            status=status,
            provenance=provenance,
        )
    except (ValueError, TypeError, KeyError, OSError, BadZipFile) as error:
        parser.error(str(error))
    print(f"Saved static spatial diagnostic (no verified common timeline): {saved}")
    return 0


def _synchronized(args: argparse.Namespace, recording_id: str) -> int:
    """Use only the verified selected window; no VRS images or CSVs are reread."""
    from duet.adapters.comind.indexed_video import iter_indexed_rgb
    from duet.adapters.comind.playback import load_playback
    from duet.visualization.comind_handover import save_index_episode

    report = _outside_raw(
        args.frame_validation_report
        or Path("outputs/comind_frame_validation") / f"{recording_id}.json"
    )
    output = _outside_raw(
        args.output
        or Path("outputs/comind_visualization") / f"{recording_id}_synchronized_handover.rrd"
    )
    if output.suffix != ".rrd":
        raise ValueError("synchronized output must end in .rrd")
    bundle = load_playback(_outside_raw(args.processed_root), report, recording_id)
    videos = {
        role: iter_indexed_rgb(
            args.dataset_root / "recordings" / recording_id / "mp4s" / f"{role}_trimmed_sync.mp4",
            bundle.frame_maps[role].mp4_pts,
            bundle.frame_maps[role].mp4_time_base,
            start_index=bundle.start_index,
            end_index=bundle.end_index,
        )
        for role in ROLES
    }
    try:
        saved = save_index_episode(
            output,
            bundle.iter_frames(),
            bindings=bundle.bindings,
            video_frames=videos,
            trajectories=bundle.trajectories,
        )
    finally:
        for iterator in videos.values():
            iterator.close()
    print(
        json.dumps(
            {
                "saved": str(saved),
                "timeline": "comind_sync_frame_index",
                "start_frame": bundle.start_index,
                "end_frame": bundle.end_index,
                "annotation_id": bundle.selection["annotation_id"],
                "object_category": bundle.selection["object_category_level_1"],
                "validation_report": str(report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
