#!/usr/bin/env python3
"""Stream compact visual fingerprints once, then match every MP4 frame to native VRS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import av
import numpy as np

from duet.adapters.comind.annotations import load_handover_annotations
from duet.adapters.comind.frame_map import (
    VideoVrsFrameMap,
    image_errors,
    image_fingerprint,
    load_frame_map,
    match_fingerprints,
    pixel_geometry_scores,
)
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.validation import ensure_output_outside_raw
from duet.schemas.common import Provenance


class FingerprintCache:
    """Compact derived uint8 arrays, resumable validity masks, source identity guard."""

    def __init__(
        self,
        directory: Path,
        *,
        source: Path | None,
        count: int,
        kind: str,
        source_identity: dict | None = None,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        if source_identity is None:
            if source is None:
                raise ValueError("a local source or explicit remote identity is required")
            stat = source.stat()
            source_identity = {
                "source": str(source.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        identity = {
            **source_identity,
            "frame_count": count,
            "fingerprint_size": 32,
            "kind": kind,
            "method": "FFmpeg gray 32x32, native orientation for VRS",
        }
        metadata = directory / "source.json"
        if metadata.exists() and json.loads(metadata.read_text()) != identity:
            raise ValueError("fingerprint cache source identity changed")
        metadata.write_text(json.dumps(identity, indent=2) + "\n")
        self.count = count
        self.fingerprints = self.array("fingerprints", np.uint8, (count, 32, 32), 0)
        self.done = self.array("done", np.bool_, (count,), False)
        self.valid = self.array("valid", np.bool_, (count,), False)
        self.pts = self.array("pts", np.int64, (count,), -1)

    def array(self, name: str, dtype, shape: tuple[int, ...], fill):
        path = self.directory / f"{name}.npy"
        if path.exists():
            result = np.load(path, mmap_mode="r+", allow_pickle=False)
            if result.dtype != np.dtype(dtype) or result.shape != shape:
                raise ValueError("fingerprint cache array dimensions changed")
        else:
            result = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            result[:] = fill
            result.flush()
        return result

    def flush(self) -> None:
        for array in (self.fingerprints, self.pts, self.valid, self.done):
            array.flush()


def cache_mp4(path: Path, directory: Path) -> tuple[FingerprintCache, dict[str, object]]:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        cache = FingerprintCache(directory, source=path, count=stream.frames, kind="mp4")
        metadata = {
            "frame_count": stream.frames,
            "width": stream.width,
            "height": stream.height,
            "time_base_numerator": stream.time_base.numerator,
            "time_base_denominator": stream.time_base.denominator,
        }
        if not cache.done.all():
            # One sequential decode establishes true output ordinals and exact PTS.
            # Resumption still decodes once but skips expensive completed reductions.
            count = 0
            previous_pts = None
            for index, frame in enumerate(container.decode(stream)):
                if index >= cache.count or frame.pts is None:
                    raise ValueError("MP4 frame count/PTS are inconsistent with headers")
                if previous_pts is not None and frame.pts <= previous_pts:
                    raise ValueError("MP4 output presentation order is not strictly increasing")
                previous_pts = frame.pts
                if cache.done[index] and cache.pts[index] != frame.pts:
                    raise ValueError("cached MP4 PTS changed")
                if not cache.done[index]:
                    cache.fingerprints[index] = frame.reformat(
                        width=32, height=32, format="gray"
                    ).to_ndarray()
                    cache.pts[index] = frame.pts
                    cache.valid[index] = True
                    cache.done[index] = True
                count += 1
                if count % 3000 == 0:
                    cache.flush()
                    print(f"{path.name}: MP4 fingerprints {count}/{cache.count}", flush=True)
            if count != cache.count:
                raise ValueError("decoded MP4 count differs from header")
            cache.flush()
        (directory / "video.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return cache, metadata


def read_vrs_fingerprints(provider, stream, cache: FingerprintCache, indices: np.ndarray) -> None:
    for index in indices:
        index = int(index)
        if cache.done[index]:
            continue
        if hasattr(provider, "fingerprint"):
            cache.fingerprints[index] = provider.fingerprint(index)
            cache.valid[index] = True
            cache.done[index] = True
            continue
        image, _ = provider.get_image_data_by_index(stream, index)
        if image.is_valid():
            cache.fingerprints[index] = image_fingerprint(image.to_numpy_array())
            cache.valid[index] = True
        cache.done[index] = True
        if index % 3000 == 0:
            cache.flush()
            print(f"VRS fingerprints {index}/{cache.count}", flush=True)
    cache.flush()


def sparse_anchors(
    mp4: FingerprintCache,
    vrs: FingerprintCache,
    provider,
    stream,
    *,
    handover_boundaries: list[int],
) -> list[dict[str, object]]:
    # Locate the first frame over a broad beginning window under all rotations.
    read_vrs_fingerprints(provider, stream, vrs, np.arange(min(100, vrs.count)))
    first_candidates = np.flatnonzero(vrs.valid[:100])
    first_scores = []
    for rotation in range(4):
        images = np.rot90(vrs.fingerprints[first_candidates], -rotation, axes=(1, 2))
        errors = image_errors(mp4.fingerprints[0], images)
        winner = int(np.argmin(errors))
        first_scores.append((float(errors[winner]), rotation, int(first_candidates[winner])))
    _, _, initial_offset = min(first_scores)
    requested = sorted(
        {
            0,
            (mp4.count - 1) // 2,
            mp4.count - 1,
            *range(900, mp4.count, 900),
            *(i for b in handover_boundaries for i in (b - 120, b, b + 120) if 0 <= i < mp4.count),
        }
    )
    anchors = []
    for index in requested:
        center = min(vrs.count - 1, index + initial_offset)
        candidates = np.arange(max(0, center - 16), min(vrs.count, center + 17))
        read_vrs_fingerprints(provider, stream, vrs, candidates)
        candidates = candidates[vrs.valid[candidates]]
        rotation_results = []
        for rotation in range(4):
            errors = image_errors(
                mp4.fingerprints[index],
                np.rot90(vrs.fingerprints[candidates], -rotation, axes=(1, 2)),
            )
            ranked = np.argsort(errors, kind="stable")
            rotation_results.append(
                {
                    "clockwise_quarter_turns": rotation,
                    "vrs_index": int(candidates[ranked[0]]),
                    "best_rmse": float(errors[ranked[0]]),
                    "second_rmse": float(errors[ranked[1]]),
                }
            )
        winner = min(rotation_results, key=lambda value: value["best_rmse"])
        anchors.append({"mp4_frame_index": index, **winner, "all_rotations": rotation_results})
    return anchors


def build_role(arguments: tuple[str, str, str, str]) -> dict[str, object]:
    dataset_root, processed_root, recording_id, role = map(str, arguments)
    started = time.monotonic()
    base = Path(dataset_root) / "recordings" / recording_id
    processed = Path(processed_root) / recording_id
    annotations = load_handover_annotations(
        Path(dataset_root) / "annotations" / "dataset_handover_consolidated.json",
        recording_uuid=recording_id,
    )
    boundaries = [
        index
        for item in annotations
        for index in (item.start_frame, item.end_frame)
        if index is not None
    ]
    output = processed / "frame_maps"
    ensure_output_outside_raw(output, Path(dataset_root))
    output.mkdir(parents=True, exist_ok=True)
    mp4, video = cache_mp4(
        base / "mp4s" / f"{role}_trimmed_sync.mp4", output / "fingerprints" / f"{role}_mp4"
    )
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    vrs_path = base / "trimmed_vrs" / f"{role}_trimmed.vrs"
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    stream = StreamId("214-1")
    native_count = provider.get_num_data(stream)
    vrs = FingerprintCache(
        output / "fingerprints" / f"{role}_vrs", source=vrs_path, count=native_count, kind="vrs"
    )
    anchors = sparse_anchors(mp4, vrs, provider, stream, handover_boundaries=boundaries)
    reliable = [
        row
        for row in anchors
        if row["best_rmse"] <= 1.5 and row["second_rmse"] - row["best_rmse"] >= 0.08
    ]
    rotations = {row["clockwise_quarter_turns"] for row in reliable}
    if len(reliable) < 10 or len(rotations) != 1:
        raise ValueError("sparse anchors do not establish one consistent image orientation")
    rotation = rotations.pop()
    (output / f"{role}_anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    print(
        f"{role}: {len(reliable)}/{len(anchors)} strong sparse anchors, clockwise quarter turns={rotation}",
        flush=True,
    )
    # A direct decision for every MP4 frame requires its neighboring source
    # images. Cache each native image at most once; retain only 32x32 fingerprints.
    read_vrs_fingerprints(provider, stream, vrs, np.flatnonzero(~vrs.done))
    centers = np.rint(
        np.interp(
            np.arange(mp4.count),
            [row["mp4_frame_index"] for row in reliable],
            [row["vrs_index"] for row in reliable],
        )
    ).astype(np.int64)
    native_fingerprints = np.rot90(vrs.fingerprints, -rotation, axes=(1, 2))
    matches = match_fingerprints(
        mp4.fingerprints,
        native_fingerprints,
        candidate_centers=centers,
        vrs_valid=vrs.valid,
        radius=8,
    )
    # Expand weak searches using existing compact arrays, never decode again.
    retry = np.flatnonzero(
        np.isin(matches["mapping_reason"], ["search_boundary", "high_image_error"])
    )
    if len(retry):
        broader = match_fingerprints(
            mp4.fingerprints[retry],
            native_fingerprints,
            candidate_centers=centers[retry],
            vrs_valid=vrs.valid,
            radius=96,
        )
        for key in matches:
            matches[key][retry] = broader[key]
    # Recheck monotonicity after combining expanded and original direct matches.
    while True:
        assigned = np.flatnonzero(matches["vrs_rgb_frame_index"] >= 0)
        bad = np.flatnonzero(np.diff(matches["vrs_rgb_frame_index"][assigned]) < 0)
        if not len(bad):
            break
        rejected = np.unique(np.r_[assigned[bad], assigned[bad + 1]])
        matches["vrs_rgb_frame_index"][rejected] = -1
        matches["status"][rejected] = "UNRESOLVED"
        matches["mapping_reason"][rejected] = "nonmonotonic_visual_candidates"
    timestamp_path = processed / "vrs" / f"{role}_rgb_metadata.npz"
    with np.load(timestamp_path, allow_pickle=False) as timestamps:
        device_ns = timestamps["device_timestamps_ns"]
        if (
            str(timestamps["clock_domain"]) != device_clock(recording_id, role).name
            or str(timestamps["stream_id"]) != "214-1"
            or not np.array_equal(timestamps["vrs_rgb_frame_index"], np.arange(native_count))
        ):
            raise ValueError(
                "native timestamp cache has an incompatible participant clock, stream, or index inventory"
            )
    if len(device_ns) != native_count:
        raise ValueError("native timestamp count differs from decoded stream inventory")
    mapped_ns = np.full(mp4.count, -1, dtype=np.int64)
    accepted = matches["vrs_rgb_frame_index"] >= 0
    mapped_ns[accepted] = device_ns[matches["vrs_rgb_frame_index"][accepted]]
    VideoVrsFrameMap(
        recording_id,
        role,
        np.arange(mp4.count),
        matches["vrs_rgb_frame_index"],
        mapped_ns,
        matches["status"],
        matches["match_confidence"],
        mp4.pts,
        Fraction(video["time_base_numerator"], video["time_base_denominator"]),
        Provenance(str(vrs_path), detail="Direct visual matches and exact SDK device timestamps"),
    )
    adjacent_assigned = accepted[1:] & accepted[:-1]
    delta = np.diff(matches["vrs_rgb_frame_index"])
    matches["source_repeat"] = np.r_[False, adjacent_assigned & (delta == 0)]
    matches["skipped_vrs_before"] = np.r_[
        0, np.where(adjacent_assigned, np.maximum(0, delta - 1), -1)
    ].astype(np.int64)
    statuses, counts = np.unique(matches["status"], return_counts=True)
    metadata = {
        "recording_id": recording_id,
        "participant": role,
        "mp4_source": str(mp4.directory / "source.json"),
        "vrs_source": str(vrs_path),
        "native_timestamp_source": str(timestamp_path),
        "mp4_frame_count": mp4.count,
        "vrs_rgb_frame_count": native_count,
        "clockwise_quarter_turns": rotation,
        "native_dimensions": [
            int(provider.get_image_configuration(stream).image_width),
            int(provider.get_image_configuration(stream).image_height),
        ],
        "mp4_dimensions": [video["width"], video["height"]],
        "status_counts": {
            str(key): int(value) for key, value in zip(statuses, counts, strict=True)
        },
        "mapping_evidence": "Every VERIFIED row has a unique low-error direct rotated-image match in a local candidate window; exact native DEVICE_TIME is selected by VRS RGB index. Ambiguous/nonmonotonic rows remain UNRESOLVED; no FPS/PTS/UTC-derived capture times.",
        "confidence_semantics": "1 - best_RMSE/second_RMSE; image separation, not calibrated probability",
        "thresholds": {"maximum_rmse": 1.5, "minimum_margin": 0.08, "minimum_ratio": 1.15},
        "sparse_anchor_count": len(anchors),
        "strong_anchor_count": len(reliable),
        "handovers_with_anchor_boundaries": len(annotations),
        "matching_model": "Monotonic direct frame correspondence with explicitly repeated/skipped VRS sources; not a fixed offset or generated timestamps",
        "repeated_source_transitions": int(matches["source_repeat"].sum()),
        "skipped_native_frames_between_matches": int(
            matches["skipped_vrs_before"].clip(min=0).sum()
        ),
        "unique_native_frames_matched": len(np.unique(matches["vrs_rgb_frame_index"][accepted])),
        "first_matched_native_index": int(
            matches["vrs_rgb_frame_index"][np.flatnonzero(accepted)[0]]
        )
        if accepted.any()
        else None,
        "last_matched_native_index": int(
            matches["vrs_rgb_frame_index"][np.flatnonzero(accepted)[-1]]
        )
        if accepted.any()
        else None,
        "best_rmse_percentiles": dict(
            zip(
                ("median", "p95", "max"),
                np.percentile(matches["best_rmse"][accepted], [50, 95, 100]).tolist(),
                strict=True,
            )
        )
        if accepted.any()
        else None,
        "minimum_best_vs_second_margin": float(
            np.min((matches["second_rmse"] - matches["best_rmse"])[accepted])
        )
        if accepted.any()
        else None,
        "minimum_second_over_best_ratio": float(
            np.min((matches["second_rmse"] / matches["best_rmse"])[accepted])
        )
        if accepted.any()
        else None,
        "all_native_fingerprints_cached": bool(vrs.done.all()),
        "unresolved_reasons": {
            str(reason): int(np.count_nonzero(matches["mapping_reason"] == reason))
            for reason in np.unique(matches["mapping_reason"])
        },
        "elapsed_seconds": time.monotonic() - started,
        "raw_immutability_check": "source sizes and modification times, not a repeated VRS checksum pass",
    }
    identities = [json.loads((cache.directory / "source.json").read_text()) for cache in (mp4, vrs)]
    metadata["raw_sources_unmodified"] = all(
        (Path(identity["source"]).stat().st_size, Path(identity["source"]).stat().st_mtime_ns)
        == (identity["size"], identity["mtime_ns"])
        for identity in identities
    )
    if not metadata["raw_sources_unmodified"]:
        raise ValueError(
            "raw source changed during visual extraction; refusing to publish verified mappings"
        )
    with tempfile.NamedTemporaryFile(
        dir=output, prefix=f".{role}_", suffix=".npz", delete=False
    ) as temporary:
        np.savez(
            temporary,
            mp4_frame_index=np.arange(mp4.count, dtype=np.int64),
            vrs_device_timestamp_ns=mapped_ns,
            mp4_pts=mp4.pts,
            mp4_time_base_numerator=video["time_base_numerator"],
            mp4_time_base_denominator=video["time_base_denominator"],
            **matches,
        )
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    metadata["npz_sha256"] = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
    temporary_json = temporary_path.with_suffix(".json")
    temporary_json.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_path.replace(output / f"{role}_frame_map.npz")
    temporary_json.replace(output / f"{role}_frame_map.json")
    print(
        json.dumps(
            {
                "participant": role,
                "status_counts": metadata["status_counts"],
                "elapsed_seconds": metadata["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return metadata


def audit_pixel_geometry(
    arguments: tuple[str, str, str, str], *, rgb_provider=None
) -> dict[str, object]:
    """Use three already mapped image pairs to check full-resolution export geometry."""
    dataset_root, processed_root, recording_id, role = arguments
    directory = Path(processed_root) / recording_id / "frame_maps"
    path = directory / f"{role}_frame_map.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    output = directory / f"{role}_pixel_geometry.json"
    if output.exists():
        cached = json.loads(output.read_text())
        if (
            cached.get("frame_map_npz_sha256") == metadata["npz_sha256"]
            and cached.get("pixel_geometry_method_version") == 2
        ):
            return cached
    mapping = load_frame_map(path)
    base = Path(dataset_root) / "recordings" / recording_id
    if rgb_provider is None:
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.stream_id import StreamId

        provider = data_provider.create_vrs_data_provider(
            str(base / "trimmed_vrs" / f"{role}_trimmed.vrs")
        )

        def rgb_provider(index):
            image, _ = provider.get_image_data_by_index(StreamId("214-1"), index)
            return image.to_numpy_array()

    samples = []
    with av.open(str(base / "mp4s" / f"{role}_trimmed_sync.mp4"), mode="r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for index in (0, (mapping.frame_count - 1) // 2, mapping.frame_count - 1):
            if mapping.timestamp_at(index) is None:
                raise ValueError("pixel geometry audit requires a verified visual frame match")
            requested_pts = int(mapping.mp4_pts[index])
            container.seek(requested_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts == requested_pts:
                    target = frame.to_ndarray(format="gray")
                    break
                if frame.pts is None or frame.pts > requested_pts:
                    raise ValueError("pixel geometry audit could not locate exact cached MP4 PTS")
            else:
                raise ValueError("pixel geometry audit reached EOF before requested frame")
            vrs_index = int(mapping.vrs_rgb_frame_index[index])
            native = av.VideoFrame.from_ndarray(rgb_provider(vrs_index), format="rgb24").to_ndarray(
                format="gray"
            )
            upright = np.ascontiguousarray(np.rot90(native, -metadata["clockwise_quarter_turns"]))
            samples.append(
                {
                    "mp4_frame_index": index,
                    "vrs_rgb_frame_index": vrs_index,
                    "device_timestamp_ns": int(mapping.vrs_device_timestamp_ns[index]),
                    "image_dimensions": [target.shape[1], target.shape[0]],
                    **pixel_geometry_scores(target, upright),
                    "symmetric_lowpass_control": pixel_geometry_scores(
                        target, upright, prefilter_passes=3
                    ),
                }
            )
    result = {
        "recording_id": recording_id,
        "participant": role,
        "frame_map_npz_sha256": metadata["npz_sha256"],
        "pixel_geometry_method_version": 2,
        "clockwise_quarter_turns": metadata["clockwise_quarter_turns"],
        "samples": samples,
        "zero_shift_unit_scale_wins_all_samples": all(
            sample["zero_shift_unit_scale_wins"] for sample in samples
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    # The remote path reuses matching/cache/audit functions in this module, while
    # bounding decoded VRS records to selected handover windows and anchors.
    import sys

    if "--remote-vrs" in sys.argv:
        from build_comind_remote_frame_map import main as remote_main

        remote_main([argument for argument in sys.argv[1:] if argument != "--remote-vrs"])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--audit-image-geometry",
        action="store_true",
        help="Check three full-resolution paired images per role, reusing saved evidence when unchanged",
    )
    args = parser.parse_args()
    UUID(args.recording_id)
    ensure_output_outside_raw(args.processed_root, args.dataset_root)
    arguments = [
        (str(args.dataset_root), str(args.processed_root), args.recording_id, role)
        for role in ("helper", "leader")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(build_role, arguments))
        if args.audit_image_geometry:
            list(executor.map(audit_pixel_geometry, arguments))
    report = args.processed_root / args.recording_id / "frame_maps" / "summary.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
