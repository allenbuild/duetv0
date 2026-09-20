#!/usr/bin/env python3
"""Validate mapped videos against local poses/hands and rank annotated handovers.

Example (derived outputs only; raw sources opened read-only):
    .venv/bin/python scripts/validate_comind_frame_map.py \
        --recording-id 43276420-701f-4731-b9ab-bebc7fd14994

Use --prepare-only once while visual matching runs. This builds compact full-rate
pose caches in one pass per participant; subsequent validation reuses them and
the existing complete hand cache. It never decodes images or uses UTC alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from duet.adapters.comind.annotations import load_handover_annotations
from duet.adapters.comind.frame_map import load_frame_map
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.adapters.comind.temporal_validation import (
    ParticipantStreams,
    TemporalThresholds,
    bind_annotations,
    rank_handovers,
    read_full_trajectory_once,
    validate_participant_frames,
)


def outside_raw(path: Path) -> Path:
    resolved = path.resolve()
    if any(a == "data" and b == "raw" for a, b in zip(resolved.parts, resolved.parts[1:])):
        raise ValueError("derived outputs must remain outside data/raw")
    return resolved


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError(f"expected JSON object: {path}")
    return document


def prepare_poses(
    directory: Path, recording_id: str, manifest: dict, *, cache_directory: Path
) -> dict[str, dict]:
    """Reuse generation-matched full-rate caches or read each trajectory once."""
    outside_raw(cache_directory).mkdir(parents=True, exist_ok=True)
    shared = manifest["shared_world"]
    if manifest.get("recording_id") != recording_id or shared.get("status") != "VERIFIED":
        raise ValueError("semantics manifest must identify the recording and verified shared world")
    graph = shared["graph_uid"]
    if shared["frame"] != f"comind/{recording_id}/multislam/{graph}":
        raise ValueError("shared world frame does not match the recording and graph")
    results = {}
    for role in ("helper", "leader"):
        source = manifest["participants"][role]["source"]
        if source["graph_uid"] != graph:
            raise ValueError("participant trajectory belongs to a different shared graph")
        source_path = Path(source["path"])
        identity = {
            "recording_id": recording_id,
            "participant": role,
            "graph_uid": graph,
            "source_path": str(source_path.resolve()),
            "source_bytes": source["bytes"],
            "source_mtime_ns": source["mtime_ns"],
            "source_row_count": source["row_count"],
            "source_generation_crc32": source["crc32"],
        }
        cache = cache_directory / f"{role}_full_trajectory.npz"
        metadata = cache.with_suffix(".json")
        if cache.exists() and metadata.exists():
            existing = read_json(metadata)
            if existing["source_identity"] != identity:
                raise ValueError(
                    "full trajectory cache source identity differs from source manifest"
                )
            if existing.get("npz_sha256") != sha256(cache):
                raise ValueError("full trajectory cache payload checksum differs from its metadata")
        else:
            stat = source_path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
                raise ValueError("source trajectory changed since geometry verification")
            arrays = read_full_trajectory_once(source_path, expected_graph_uid=graph)
            if len(arrays["device_us"]) != source["row_count"]:
                raise ValueError("full-rate trajectory row count differs from manifest")
            np.savez_compressed(cache, **arrays)
            metadata.write_text(
                json.dumps(
                    {
                        "source_identity": identity,
                        "npz_sha256": sha256(cache),
                        "sampling": "Every source trajectory row; no downsampling or interpolation",
                        "source_integrity": "Source size/mtime unchanged; all graph IDs and rotations checked; prior CRC retained, not recomputed",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        results[role] = {"path": str(cache), "metadata": str(metadata), **identity}
    return results


def load_streams(
    directory: Path,
    recording_id: str,
    role: str,
    manifest: dict,
    pose_cache: Path,
) -> ParticipantStreams:
    with np.load(pose_cache, allow_pickle=False) as source:
        pose = {name: source[name] for name in source.files}
    with np.load(directory / "semantics" / f"{role}.npz", allow_pickle=False) as source:
        hands = {
            name: source[name]
            for name in (
                "hand_device_us",
                "hand_points_shared_m",
                "hand_points_device_m",
                "hand_confidence",
                "hand_pose_residual_us",
                "hand_pose_accepted",
                "graph_uid",
            )
        }
    graph = manifest["shared_world"]["graph_uid"]
    if str(hands["graph_uid"].item()) != graph:
        raise ValueError("hand cache does not belong to the verified shared graph")
    role_metadata = manifest["participants"][role]
    if role_metadata["cache"]["participant"] != role or role_metadata["device_clock"] != (
        device_clock(recording_id, role).name
    ):
        raise ValueError("hand cache participant/device clock metadata differs")
    return ParticipantStreams(
        recording_id,
        role,
        graph,
        role_metadata["device_clock"],
        pose["device_us"],
        pose["translation_m"],
        pose["quaternion_xyzw"],
        hands["hand_device_us"],
        hands["hand_points_shared_m"],
        hands["hand_confidence"],
        hands["hand_pose_residual_us"],
        hands["hand_pose_accepted"],
        hands["hand_points_device_m"],
    )


def load_map_arrays(directory: Path, recording_id: str, role: str) -> dict[str, np.ndarray]:
    """Validate the serialized map and exact native timestamp/index pairs."""
    path = directory / "frame_maps" / f"{role}_frame_map.npz"
    metadata = read_json(path.with_suffix(".json"))
    if metadata.get("recording_id") != recording_id or metadata.get("participant") != role:
        raise ValueError("frame map metadata belongs to another recording or participant")
    validated = load_frame_map(path)
    if (validated.recording_id, validated.participant) != (recording_id, role):
        raise ValueError("validated frame map belongs to another participant")
    with np.load(path, allow_pickle=False) as source:
        result = {name: source[name] for name in source.files}
    with np.load(directory / "vrs" / f"{role}_rgb_metadata.npz", allow_pickle=False) as source:
        native = source["device_timestamps_ns"]
    assigned = result["status"] != "UNRESOLVED"
    indices = result["vrs_rgb_frame_index"][assigned]
    if indices.dtype.kind != "i" or np.any(indices < 0) or np.any(indices >= len(native)):
        raise ValueError("assigned VRS indices outside the exact native metadata cache")
    if not np.array_equal(native[indices], result["vrs_device_timestamp_ns"][assigned]):
        raise ValueError("assigned frame times do not equal exact native VRS capture timestamps")
    return result


def json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def sha256(path: Path) -> str:
    """Hash compact derived artifacts, never the raw videos or VRS payloads."""
    with outside_raw(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/raw/comind/annotations/dataset_handover_consolidated.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pose-max-gap-ns", type=int, default=2_000_000)
    parser.add_argument("--hand-max-gap-ns", type=int, default=20_000_000)
    parser.add_argument("--minimum-hand-confidence", type=float, default=0.5)
    parser.add_argument("--context-frames", type=int, default=120)
    args = parser.parse_args(argv)
    directory = outside_raw(args.processed_root / args.recording_id)
    cache_directory = outside_raw(directory / "frame_validation")
    output = outside_raw(
        args.output or Path(f"outputs/comind_frame_validation/{args.recording_id}.json")
    )
    thresholds = TemporalThresholds(
        pose_max_gap_ns=args.pose_max_gap_ns,
        hand_max_gap_ns=args.hand_max_gap_ns,
        minimum_hand_confidence=args.minimum_hand_confidence,
    )
    manifest = read_json(directory / "semantics" / "manifest.json")
    full_poses = prepare_poses(
        directory, args.recording_id, manifest, cache_directory=cache_directory
    )
    if args.prepare_only:
        print(json.dumps({"full_rate_pose_caches": full_poses}, indent=2))
        return 0
    validations = {}
    generation = uuid.uuid4().hex
    cache_identities, map_inputs = {}, {}
    for role in ("helper", "leader"):
        streams = load_streams(
            directory, args.recording_id, role, manifest, Path(full_poses[role]["path"])
        )
        validation = validate_participant_frames(
            load_map_arrays(directory, args.recording_id, role),
            streams,
            recording_id=args.recording_id,
            participant=role,
            thresholds=thresholds,
        )
        validations[role] = validation
        aligned = cache_directory / f"{role}_aligned.npz"
        np.savez_compressed(aligned, **validation.arrays, generation_id=np.asarray(generation))
        cache_identities[role] = {"path": str(aligned), "sha256": sha256(aligned)}
        map_path = directory / "frame_maps" / f"{role}_frame_map.npz"
        native_path = directory / "vrs" / f"{role}_rgb_metadata.npz"
        map_inputs[role] = {
            "path": str(map_path),
            "sha256": sha256(map_path),
            "metadata_path": str(map_path.with_suffix(".json")),
            "metadata_sha256": sha256(map_path.with_suffix(".json")),
            "native_metadata_path": str(native_path),
            "native_metadata_sha256": sha256(native_path),
        }
    annotations = load_handover_annotations(args.annotations, recording_uuid=args.recording_id)
    binding = MappingEvidence(
        "annotation_to_comind_sync_frame_index",
        EvidenceStatus.VERIFIED,
        "User-authorized direct annotation frame binding to the CoMind paired-video frame timeline; independently checked source frame/time arithmetic",
        (
            "User workstream 5: bind annotation frame indices directly to comind_sync_frame_index",
            "outputs/comind_timestamp_probe/annotation frame/time audit; repeated by this validator",
        ),
    )
    alignment = MappingEvidence(
        "paired_video_frame_alignment",
        EvidenceStatus.VERIFIED,
        "CoMind documents both ego videos as frame-aligned; individual device clocks remain separate",
        ("https://comind.ethz.ch/",),
    )
    bound = bind_annotations(
        annotations,
        recording_id=args.recording_id,
        frame_count=len(validations["helper"].arrays["mp4_frame_index"]),
        binding_evidence=binding,
    )
    ranking = rank_handovers(
        validations["helper"],
        validations["leader"],
        bound,
        alignment_evidence=alignment,
        context_frames=args.context_frames,
        thresholds=thresholds,
    )
    report = {
        "recording_id": args.recording_id,
        "generation_id": generation,
        "timeline": "comind_sync_frame_index",
        "shared_world": manifest["shared_world"],
        "thresholds": asdict(thresholds),
        "participants": {role: value.summary for role, value in validations.items()},
        "full_rate_pose_caches": full_poses,
        "aligned_array_caches": {
            role: str(cache_directory / f"{role}_aligned.npz") for role in validations
        },
        "aligned_cache_identities": cache_identities,
        "frame_map_inputs": map_inputs,
        "annotation_binding": binding.to_dict(),
        "annotation_count": len(bound),
        "handover_ranking": ranking,
        "provenance": {
            "hand_world_coordinates": "Reused full hand cache, originally transformed by nearest full-rate own-device pose; secondary residual gates retained",
            "pose_matching": "Full-rate trajectory source arrays, exact nanoseconds, earlier-time/first-duplicate tie policy",
            "raw_data_writes": False,
            "images_decoded": False,
            "source_hand_csv_reread": False,
            "utc_used": False,
            "missing_residual_sentinel": int(np.iinfo(np.int64).min),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=json_default, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "selection": ranking["selected_annotation_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
