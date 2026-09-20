#!/usr/bin/env python3
"""Build existing Duet frame maps from bounded remote VRS RGB records.

Explicit --fetch is mandatory. Every selected handover context and sparse global
anchors is visually matched; the remaining MP4 rows stay UNRESOLVED. No local
VRS, RGB payload cache, generated capture timestamps, or whole-file fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import numpy as np
from build_comind_frame_map import (
    FingerprintCache,
    audit_pixel_geometry,
    cache_mp4,
    read_vrs_fingerprints,
    sparse_anchors,
)
from comind_download import DEFAULT_BASE_URL, build_url

from duet.adapters.comind.annotations import load_handover_annotations
from duet.adapters.comind.frame_map import VideoVrsFrameMap, match_fingerprints
from duet.adapters.comind.remote_vrs import (
    HttpRangeReader,
    RemoteVrs,
    atomic_json,
    atomic_npz,
    processed_directory,
)
from duet.adapters.comind.validation import ensure_output_outside_raw
from duet.schemas.common import Provenance


def requested_frames(count: int, annotations: list, anchors: list[dict]) -> np.ndarray:
    """Select inclusive handovers with the renderer's 120-frame context each side."""
    if count <= 0:
        raise ValueError("MP4 frame count must be positive")
    if any(not 0 <= row["mp4_frame_index"] < count for row in anchors):
        raise ValueError("anchor frame bounds must lie in the current MP4")
    requested = {0, (count - 1) // 2, count - 1}
    requested.update(int(row["mp4_frame_index"]) for row in anchors)
    for annotation in annotations:
        start, end = annotation.start_frame, annotation.end_frame
        if start is None or end is None or not 0 <= start <= end < count:
            raise ValueError("handover frame bounds must lie in the current MP4")
        requested.update(range(max(0, start - 120), min(count, end + 121)))
    return np.array(sorted(requested), dtype=np.int64)


def match_requested(
    mp4: np.ndarray,
    native: np.ndarray,
    valid: np.ndarray,
    *,
    requested: np.ndarray,
    centers: np.ndarray,
    radius: int = 8,
) -> dict[str, np.ndarray]:
    """Reuse direct matching while preventing unobserved frames acquiring timestamps."""
    subset = match_fingerprints(
        mp4[requested],
        native,
        candidate_centers=centers[requested],
        vrs_valid=valid,
        radius=radius,
    )
    for row, frame in enumerate(requested):
        window = valid[
            max(0, int(centers[frame]) - radius) : min(len(valid), int(centers[frame]) + radius + 1)
        ]
        if not len(window) or not window.all():
            subset["status"][row] = "UNRESOLVED"
            subset["vrs_rgb_frame_index"][row] = -1
            subset["match_confidence"][row] = 0
            subset["mapping_reason"][row] = "incomplete_candidate_window"
    result = {}
    for name, values in subset.items():
        if name == "status":
            fill = "UNRESOLVED"
        elif name == "mapping_reason":
            fill = "outside_requested_windows"
        elif name == "match_confidence":
            fill = 0
        elif values.dtype.kind in "iu":
            fill = -1
        else:
            fill = np.nan
        result[name] = np.full(len(mp4), fill, dtype=values.dtype)
        result[name][requested] = values
    return result


def build_remote_role(args: argparse.Namespace, role: str, entry: dict) -> dict:
    """Use participant-local exact timestamps; shared MP4 indices remain the timeline."""
    root = processed_directory(args.processed_root / args.recording_id)
    output, vrs_directory = root / "frame_maps", root / "vrs"
    output.mkdir(parents=True, exist_ok=True)
    source = (
        args.dataset_root / "recordings" / args.recording_id / "mp4s" / f"{role}_trimmed_sync.mp4"
    )
    mp4, video = cache_mp4(source, output / "fingerprints" / f"{role}_mp4")
    annotations = load_handover_annotations(
        args.dataset_root / "annotations/dataset_handover_consolidated.json",
        recording_uuid=args.recording_id,
    )
    if args.annotation_id:
        requested_ids = set(args.annotation_id)
        annotations = [a for a in annotations if a.annotation_id in requested_ids]
        if {a.annotation_id for a in annotations} != requested_ids:
            raise ValueError("requested annotation IDs are absent from this recording")
    if not annotations:
        raise ValueError("no handover annotations available")
    url = build_url(DEFAULT_BASE_URL, args.recording_id, entry["path"])
    with HttpRangeReader(
        url,
        size=int(entry["size"]),
        budget_bytes=args.range_budget_bytes,
        ledger=vrs_directory / f"{role}_range_ledger.json",
    ) as reader:
        remote = (
            RemoteVrs.from_cache(reader, vrs_directory, role=role)
            if (vrs_directory / f"{role}_remote_vrs.json").exists()
            else RemoteVrs.open(reader)
        )
        try:
            native = FingerprintCache(
                output / "fingerprints" / f"{role}_remote_vrs",
                source=None,
                count=remote.rgb_count,
                kind="remote_vrs",
                source_identity={
                    "source": url,
                    "size": entry["size"],
                    "manifest_hash": entry["hash"],
                    "etag": reader.etag,
                },
            )
            # Compact remote cache is authoritative after an interrupted run.
            # Never reuse a fingerprint whose exact capture metadata was lost.
            native.done[:] &= remote.metadata_known
            native.valid[:] &= native.done
            boundaries = [i for a in annotations for i in (a.start_frame, a.end_frame)]
            anchors = sparse_anchors(
                mp4,
                native,
                remote,
                remote.stream_id,
                handover_boundaries=boundaries,
            )
            atomic_json(output / f"{role}_anchors.json", anchors)
            reliable = [
                row
                for row in anchors
                if row["best_rmse"] <= 1.5 and row["second_rmse"] - row["best_rmse"] >= 0.08
            ]
            rotations = {row["clockwise_quarter_turns"] for row in reliable}
            if len(reliable) < 10 or len(rotations) != 1:
                raise ValueError("sparse anchors do not establish consistent visual orientation")
            rotation = rotations.pop()
            centers = np.rint(
                np.interp(
                    np.arange(mp4.count),
                    [row["mp4_frame_index"] for row in reliable],
                    [row["vrs_index"] for row in reliable],
                )
            ).astype(np.int64)
            requested = requested_frames(mp4.count, annotations, anchors)
            # Fetch the union of candidate windows exactly once, not every RGB record.
            candidates = np.unique(
                np.concatenate(
                    [
                        np.arange(
                            max(0, int(centers[i]) - 8), min(remote.rgb_count, int(centers[i]) + 9)
                        )
                        for i in requested
                    ]
                )
            )
            for batch in np.array_split(candidates, max(1, len(candidates) // 128)):
                read_vrs_fingerprints(remote, remote.stream_id, native, batch)
                remote.save_metadata(vrs_directory, role=role, recording_id=args.recording_id)
            matches = match_requested(
                mp4.fingerprints,
                np.rot90(native.fingerprints, -rotation, axes=(1, 2)),
                native.valid,
                requested=requested,
                centers=centers,
            )
            accepted = matches["vrs_rgb_frame_index"] >= 0
            timestamps = np.full(mp4.count, -1, dtype=np.int64)
            indices = matches["vrs_rgb_frame_index"][accepted]
            if not remote.metadata_known[indices].all():
                raise ValueError("a visual match lacks exact native DEVICE_TIME")
            timestamps[accepted] = remote.device_timestamps_ns[indices]
            VideoVrsFrameMap(
                args.recording_id,
                role,
                np.arange(mp4.count),
                matches["vrs_rgb_frame_index"],
                timestamps,
                matches["status"],
                matches["match_confidence"],
                mp4.pts,
                Fraction(video["time_base_numerator"], video["time_base_denominator"]),
                Provenance(url, "Direct sparse RGB visual matches with exact capture_timestamp_ns"),
            )
            adjacent = accepted[1:] & accepted[:-1]
            delta = np.diff(matches["vrs_rgb_frame_index"])
            matches["source_repeat"] = np.r_[False, adjacent & (delta == 0)]
            matches["skipped_vrs_before"] = np.r_[
                0, np.where(adjacent, np.maximum(0, delta - 1), -1)
            ]
            mp4_identity = json.loads((mp4.directory / "source.json").read_text())
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (
                mp4_identity["size"],
                mp4_identity["mtime_ns"],
            ):
                raise ValueError("raw MP4 changed during remote visual matching")
            path = output / f"{role}_frame_map.npz"
            atomic_npz(
                path,
                mp4_frame_index=np.arange(mp4.count, dtype=np.int64),
                vrs_device_timestamp_ns=timestamps,
                mp4_pts=mp4.pts,
                mp4_time_base_numerator=video["time_base_numerator"],
                mp4_time_base_denominator=video["time_base_denominator"],
                **matches,
            )
            status, counts = np.unique(matches["status"], return_counts=True)
            metadata = {
                "recording_id": args.recording_id,
                "participant": role,
                "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "clockwise_quarter_turns": rotation,
                "mp4_frame_count": mp4.count,
                "vrs_rgb_frame_count": remote.rgb_count,
                "status_counts": dict(zip(status.tolist(), counts.tolist(), strict=True)),
                "mp4_dimensions": [video["width"], video["height"]],
                "mapping_evidence": "Direct image matching only for selected handover contexts and sparse anchors; exact native DEVICE_TIME from RGB data records. Other rows remain UNRESOLVED. No inferred capture timestamps.",
                "annotation_ids": [a.annotation_id for a in annotations],
                "requested_mp4_frames": requested.tolist(),
                "all_native_fingerprints_cached": bool(native.done.all()),
                "remote_range_body_bytes": reader.report["body_bytes_read"],
                "remote_range_reserved_bytes": reader.report["reserved_bytes"],
            }
            atomic_json(path.with_suffix(".json"), metadata)
            if args.audit_image_geometry:
                audit_pixel_geometry(
                    (str(args.dataset_root), str(args.processed_root), args.recording_id, role),
                    rgb_provider=remote.rgb_image,
                )
            return metadata
        finally:
            remote.save_metadata(vrs_directory, role=role, recording_id=args.recording_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="saved official healthcheck JSON"
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument(
        "--fetch", action="store_true", help="explicitly allow bounded VRS requests"
    )
    parser.add_argument("--annotation-id", action="append", help="default: all handover windows")
    parser.add_argument(
        "--range-budget-bytes",
        type=int,
        default=4_000_000_000,
        help="persistent conservative budget per role, including request allowance",
    )
    parser.add_argument("--audit-image-geometry", action="store_true")
    args = parser.parse_args(argv)
    args.recording_id = str(UUID(args.recording_id))
    if not args.fetch:
        parser.error(
            "no network access without --fetch; use comind_download_minimal.py --dry-run for planning"
        )
    ensure_output_outside_raw(args.processed_root, args.dataset_root)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("recording_id") != args.recording_id:
        raise ValueError("manifest belongs to another recording")
    results = []
    for role in ("helper", "leader"):
        entries = [
            entry
            for entry in manifest["files"]
            if entry["path"] == f"trimmed_vrs/{role}_trimmed.vrs"
        ]
        if len(entries) != 1:
            raise ValueError("manifest must identify exactly one trimmed VRS per participant")
        results.append(build_remote_role(args, role, entries[0]))
    atomic_json(args.processed_root / args.recording_id / "frame_maps/summary.json", results)


if __name__ == "__main__":
    main()
