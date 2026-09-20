#!/usr/bin/env python3
"""Bounded decoded-image evidence for trimmed versus untrimmed CoMind videos.

Frame locations use each MP4's own constant-rate presentation grid. Cross-video
correspondence is selected only from image errors, never description timestamps.
Sampled agreement cannot prove a mapping at unsampled frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from uuid import UUID

import av
import numpy as np

from duet.adapters.comind.validation import ensure_output_outside_raw


class FingerprintReader:
    """Seek/decode bounded windows; retain tiny fingerprints, never a video in RAM."""

    def __init__(self, path: Path, *, size: int = 64):
        self.path = path
        self.container = av.open(str(path), mode="r")
        if len(self.container.streams.video) != 1:
            self.close()
            raise ValueError("expected one video stream")
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = 2
        if not self.stream.average_rate or not self.stream.time_base:
            self.close()
            raise ValueError("frame-grid location needs explicit frame rate and time_base")
        step = Fraction(1, self.stream.average_rate) / self.stream.time_base
        if step.denominator != 1:
            self.close()
            raise ValueError("video does not have an integer constant-rate presentation grid")
        self.step = int(step)
        self.origin = self.stream.start_time or 0
        self.count = self.stream.frames
        if self.count <= 0 or self.stream.duration != self.count * self.step:
            self.close()
            raise ValueError("frame count and stream duration do not agree on a constant-rate grid")
        self.size = size
        self.decoded_count = 0
        self.seeks = 0
        self.grid_steps_checked = 0
        self.ordinal_prefix_frames_verified = 0
        self.eof_last_index = None
        self.rgb_hashes: dict[int, str] = {}

    def close(self) -> None:
        self.container.close()

    def headers(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "frame_count": self.count,
            "width": self.stream.codec_context.width,
            "height": self.stream.codec_context.height,
            "start_pts": self.origin,
            "pts_step_per_frame": self.step,
            "time_base": str(self.stream.time_base),
            "average_rate": str(self.stream.average_rate),
            "duration_pts": self.stream.duration,
        }

    def read(self, start: int, end: int, *, full_rgb_hashes: bool = False) -> dict[int, np.ndarray]:
        """Read inclusive CFR-grid indices, verifying every decoded PTS increment.

        The initial prefix additionally verifies grid indices against sequential
        decoder ordinals. Seeked windows validate local continuity; unsampled
        intervening PTS remain uninspected and are reported as such.
        """
        start, end = max(0, start), min(self.count - 1, end)
        if start > end:
            return {}
        self.container.seek(self.origin + start * self.step, stream=self.stream, backward=True)
        self.seeks += 1
        result = {}
        previous = None
        prefix_count = 0
        for frame in self.container.decode(self.stream):
            self.decoded_count += 1
            if frame.pts is None or (frame.pts - self.origin) % self.step:
                raise ValueError("decoded frame has no exact integer presentation-grid index")
            index = (frame.pts - self.origin) // self.step
            if not 0 <= index < self.count:
                raise ValueError("decoded frame lies outside header frame count")
            if previous is not None:
                self.grid_steps_checked += 1
                if index != previous + 1:
                    raise ValueError("decoded presentation grid has a gap or duplicate PTS")
            previous = index
            if start == 0:
                if index != prefix_count:
                    raise ValueError("presentation-grid index differs from decoded prefix ordinal")
                prefix_count += 1
            if index > end:
                break
            if index >= start:
                result[index] = frame.reformat(
                    width=self.size, height=self.size, format="gray"
                ).to_ndarray()
                if full_rgb_hashes:
                    self.rgb_hashes[index] = hashlib.sha256(
                        frame.to_ndarray(format="rgb24").tobytes()
                    ).hexdigest()
        else:
            self.eof_last_index = previous
        self.ordinal_prefix_frames_verified = max(self.ordinal_prefix_frames_verified, prefix_count)
        if len(result) != end - start + 1:
            raise ValueError("requested frame window is incomplete")
        return result


def fingerprint_error(first: np.ndarray, second: np.ndarray) -> float:
    """Grayscale RMSE in 0..255 intensity units, without learned representations."""
    if first.shape != second.shape or first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("fingerprints must be equal-sized uint8 arrays")
    difference = first.astype(np.float32) - second.astype(np.float32)
    return float(np.sqrt(np.mean(difference * difference)))


def rank_fingerprints(
    target: np.ndarray, candidates: dict[int, np.ndarray]
) -> list[tuple[float, int]]:
    """Rank every candidate by visual error; exact ties prefer the earlier index."""
    return sorted((fingerprint_error(target, value), index) for index, value in candidates.items())


def consecutive_offset_search(
    targets: dict[int, np.ndarray], candidates: dict[int, np.ndarray]
) -> list[tuple[float, int]]:
    """Rank candidate offsets by the mean error across consecutive target images."""
    if not targets or not candidates:
        raise ValueError("initial image search requires nonempty windows")
    result = []
    for offset in candidates:
        if all(index + offset in candidates for index in targets):
            errors = [
                fingerprint_error(frame, candidates[index + offset])
                for index, frame in targets.items()
            ]
            result.append((float(np.mean(errors)), offset))
    return sorted(result)


def merged_windows(centers: list[int], *, count: int) -> list[tuple[int, int]]:
    intervals = [(max(0, i - 1), min(count - 1, i + 1)) for i in centers]
    intervals.extend([(0, min(7, count - 1)), (max(0, count - 8), count - 1)])
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 1:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def identical_suffix_run(hashes: dict[int, str]) -> dict[str, object]:
    """Describe exact decoded-image repetition, without inferring capture timestamps."""
    indices = sorted(hashes)
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("image hash evidence must have consecutive frame indices")
    final_hash = hashes[indices[-1]]
    first = indices[-1]
    for index in reversed(indices[:-1]):
        if hashes[index] != final_hash:
            break
        first = index
    return {
        "start_index": first,
        "end_index": indices[-1],
        "length": indices[-1] - first + 1,
        "boundary_observed_inside_window": first > indices[0],
        "preceding_index": first - 1 if first > indices[0] else None,
        "image_sha256": final_hash,
        "capture_timestamp_duplication_verified": False,
    }


def probe_tail_repetition(
    *, mp4_directory: Path, role: str, window_frames: int = 120
) -> dict[str, object]:
    """Inspect one small tail window when the initial sample has repeated RGB images."""
    result = {}
    for suffix in ("trimmed_sync", "sync"):
        reader = FingerprintReader(mp4_directory / f"{role}_{suffix}.mp4")
        before = reader.path.stat()
        try:
            fingerprints = reader.read(
                reader.count - window_frames, reader.count - 1, full_rgb_hashes=True
            )
            fingerprint_hashes = {
                index: hashlib.sha256(array.tobytes()).hexdigest()
                for index, array in fingerprints.items()
            }
            result[suffix] = {
                "inspected_range": [min(fingerprints), max(fingerprints)],
                "full_rgb_identical_suffix": identical_suffix_run(reader.rgb_hashes),
                "fingerprint_identical_suffix": identical_suffix_run(fingerprint_hashes),
                "full_rgb_unique_images": len(set(reader.rgb_hashes.values())),
                "full_rgb_sha256": reader.rgb_hashes,
                "eof_last_index": reader.eof_last_index,
                "decoded_frames_including_seek_preroll": reader.decoded_count,
                "size_and_mtime_unchanged": (before.st_size, before.st_mtime_ns)
                == (reader.path.stat().st_size, reader.path.stat().st_mtime_ns),
            }
        finally:
            reader.close()
    return result


def summarize_correspondences(result: dict[str, object]) -> None:
    """Keep frame0 evidence separate from an aggregate constant-offset candidate."""
    records = result["matches"]
    confident = [row for row in records if row["confident_image_match"]]
    start = records[0]
    result["first_frame_visual_correspondence"] = {
        key: start[key]
        for key in (
            "trimmed_index",
            "best_untrimmed_index",
            "best_offset",
            "best_rmse",
            "second_rmse",
            "second_over_best_rmse",
            "confident_image_match",
        )
    }
    implied_end = start["best_offset"] + result["trimmed"]["frame_count"] - 1
    result["first_frame_offset_implied_end_untrimmed_index"] = implied_end
    result["first_frame_offset_implied_end_is_out_of_range"] = (
        implied_end >= result["untrimmed"]["frame_count"]
    )
    result["constant_offset_rejected_by_confident_visual_correspondences"] = (
        len({row["best_offset"] for row in confident}) > 1
    )
    result["confident_best_rmse"] = {
        key: float(np.percentile([row["best_rmse"] for row in confident], quantile))
        if confident
        else None
        for key, quantile in (("median", 50), ("p95", 95), ("max", 100))
    }
    consecutive_anomalies = []
    for first, second in pairwise(records):
        if (
            first["confident_image_match"]
            and second["confident_image_match"]
            and second["trimmed_index"] == first["trimmed_index"] + 1
            and second["best_untrimmed_index"] != first["best_untrimmed_index"] + 1
        ):
            consecutive_anomalies.append(
                {
                    "trimmed_indices": [first["trimmed_index"], second["trimmed_index"]],
                    "best_untrimmed_indices": [
                        first["best_untrimmed_index"],
                        second["best_untrimmed_index"],
                    ],
                    "image_rmse": [first["best_rmse"], second["best_rmse"]],
                    "interpretation": "Observed non-bijective image correspondence; capture duplication or export cause is unknown",
                }
            )
    result["consecutive_visual_index_irregularities"] = consecutive_anomalies


def probe_participant(
    *,
    mp4_directory: Path,
    role: str,
    anchor_count: int = 26,
    search_padding_frames: int = 30,
    neighbor_radius_frames: int = 6,
    max_rmse: float = 2.0,
    minimum_rank_margin: float = 0.1,
) -> dict[str, object]:
    """Find image-based initial offset, then check spaced consecutive windows/endpoints."""
    if anchor_count < 20 or neighbor_radius_frames < 1 or search_padding_frames < 1:
        raise ValueError("require at least20 anchors and positive search radii")
    if (
        not np.isfinite([max_rmse, minimum_rank_margin]).all()
        or min(max_rmse, minimum_rank_margin) < 0
    ):
        raise ValueError("image thresholds must be finite and nonnegative")
    start_time = time.monotonic()
    trimmed = FingerprintReader(mp4_directory / f"{role}_trimmed_sync.mp4")
    try:
        original = FingerprintReader(mp4_directory / f"{role}_sync.mp4")
    except (OSError, ValueError):
        trimmed.close()
        raise
    initial_stats = {reader.path: reader.path.stat() for reader in (trimmed, original)}
    try:
        if trimmed.count < anchor_count or original.count < 3:
            raise ValueError("videos are too short for the requested distinct anchor count")
        first = trimmed.read(0, 2)
        # Search all starts that could fit a complete equal-rate contiguous trim,
        # plus a small extension to detect endpoint irregularities. No timestamp
        # description/offset is used to generate or rank these candidates.
        search_end = min(
            original.count - 1, max(0, original.count - trimmed.count) + search_padding_frames + 2
        )
        prefix = original.read(0, search_end)
        ranked_offsets = consecutive_offset_search(first, prefix)
        if len(ranked_offsets) < 2:
            raise ValueError(
                "initial image search requires at least two complete candidate offsets"
            )
        initial_error, offset = ranked_offsets[0]
        if initial_error > max_rmse:
            raise ValueError("no convincing initial image match in the bounded prefix search")
        anchors = sorted(set(np.linspace(0, trimmed.count - 1, anchor_count, dtype=int).tolist()))
        records = []
        windows = []
        for start, end in merged_windows(anchors, count=trimmed.count):
            tail = end == trimmed.count - 1
            targets = trimmed.read(start, end, full_rgb_hashes=tail)
            candidate_start = max(0, start + offset - neighbor_radius_frames)
            candidate_end = min(original.count - 1, end + offset + neighbor_radius_frames)
            if not tail and candidate_end <= search_end:
                candidates = {i: prefix[i] for i in range(candidate_start, candidate_end + 1)}
            else:
                candidates = original.read(candidate_start, candidate_end, full_rgb_hashes=tail)
            windows.append(
                {"trimmed_range": [start, end], "candidate_range": [candidate_start, candidate_end]}
            )
            for index, frame in targets.items():
                scores = rank_fingerprints(frame, candidates)
                best_error, best_index = scores[0]
                second_error, second_index = scores[1]
                margin = second_error - best_error
                predicted_index = index + offset
                predicted_error = (
                    fingerprint_error(frame, candidates[predicted_index])
                    if predicted_index in candidates
                    else None
                )
                records.append(
                    {
                        "trimmed_index": index,
                        "best_untrimmed_index": best_index,
                        "best_offset": best_index - index,
                        "best_rmse": best_error,
                        "second_untrimmed_index": second_index,
                        "second_rmse": second_error,
                        "second_over_best_rmse": None
                        if best_error == 0
                        else second_error / best_error,
                        "rank_margin_rmse": margin,
                        "equal_best_candidates": [i for score, i in scores if score == best_error],
                        "confident_image_match": best_error <= max_rmse
                        and margin >= minimum_rank_margin,
                        "predicted_untrimmed_index": predicted_index,
                        "predicted_index_within_header": predicted_index < original.count,
                        "predicted_rmse": predicted_error,
                        "predicted_passes_visual_threshold": predicted_error is not None
                        and predicted_error <= max_rmse,
                        "top_candidates": [{"index": i, "rmse": score} for score, i in scores[:4]],
                    }
                )
        confident = [record for record in records if record["confident_image_match"]]
        off_model = [record for record in confident if record["best_offset"] != offset]
        invalid = [record for record in records if not record["predicted_index_within_header"]]
        mismatches = [
            record for record in records if not record["predicted_passes_visual_threshold"]
        ]
        result = {
            "participant": role,
            "trimmed": trimmed.headers(),
            "untrimmed": original.headers(),
            "method": "64x64 grayscale image RMSE; no description timestamps or cross-video PTS offsets used",
            "thresholds": {
                "maximum_rmse": max_rmse,
                "minimum_best_vs_second_margin_rmse": minimum_rank_margin,
            },
            "initial_visual_search": {
                "trimmed_frames": list(first),
                "untrimmed_range": [0, search_end],
                "candidate_offsets_checked": len(ranked_offsets),
                "best_consecutive_mean_rmse": initial_error,
                "best_vs_second_mean_rmse_margin": ranked_offsets[1][0] - initial_error,
                "minimum_margin_threshold_passed": ranked_offsets[1][0] - initial_error
                >= minimum_rank_margin,
                "top_offsets": [
                    {"offset": i, "mean_rmse": error} for error, i in ranked_offsets[:8]
                ],
            },
            "candidate_integer_offset": offset,
            "candidate_start_untrimmed_index": offset,
            "candidate_end_untrimmed_index": offset + trimmed.count - 1,
            "candidate_end_inside_untrimmed_video": offset + trimmed.count <= original.count,
            "well_spaced_anchor_count": len(anchors),
            "anchor_indices": anchors,
            "sampled_trimmed_frame_count": len(records),
            "sampled_windows": windows,
            "confident_match_count": len(confident),
            "confident_offset_counts": dict(
                Counter(str(record["best_offset"]) for record in confident)
            ),
            "ambiguous_or_high_error_count": len(records) - len(confident),
            "confident_off_model_matches": off_model,
            "predicted_mapping_visual_mismatches": mismatches,
            "predicted_indices_outside_source": invalid,
            "exact_global_fixed_offset_verified": False,
            "candidate_fixed_offset_contradicted_by_endpoint": bool(invalid),
            "unsampled_drops_insertions_or_duplicates_ruled_out": False,
            "index_evidence": {
                "convention": "zero-based MP4 presentation-grid indices",
                "headers_agree_count_times_pts_step_equals_duration": True,
                "all_decoded_frames_have_integer_grid_indices": True,
                "all_decoded_consecutive_pts_steps_checked": True,
                "initial_untrimmed_prefix_decoder_ordinals_verified": original.ordinal_prefix_frames_verified,
                "uninspected_intervening_pts_grid_assumed_from_headers": True,
                "trimmed_eof_last_index": trimmed.eof_last_index,
                "untrimmed_eof_last_index": original.eof_last_index,
            },
            "tail_full_decoded_rgb_sha256": {
                "trimmed": trimmed.rgb_hashes,
                "untrimmed": original.rgb_hashes,
                "trimmed_unique_images": len(set(trimmed.rgb_hashes.values())),
                "untrimmed_unique_images": len(set(original.rgb_hashes.values())),
                "capture_time_duplication_inferred": False,
            },
            "matches": records,
            "decoded_frame_work": {
                "trimmed": trimmed.decoded_count,
                "untrimmed": original.decoded_count,
                "trimmed_seeks": trimmed.seeks,
                "untrimmed_seeks": original.seeks,
                "largest_retained_fingerprint_window_bytes": sum(
                    value.nbytes for value in prefix.values()
                ),
                "full_video_images_retained": False,
            },
            "elapsed_seconds": time.monotonic() - start_time,
            "limitations": (
                "Sampled image agreement supports candidate correspondences only. Re-encoding and "
                "repeated/static images can make adjacent frames indistinguishable. No unique tail "
                "mapping is assigned from tied fingerprints. This does not bind any video frame "
                "to Aria DEVICE_TIME, and cannot exclude irregularities in unsampled windows."
            ),
        }
        result["raw_sizes_and_mtimes_unchanged"] = all(
            (before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns)
            for path, before in initial_stats.items()
        )
        summarize_correspondences(result)
        return result
    finally:
        trimmed.close()
        original.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/comind_sync_forensics"))
    parser.add_argument("--anchors", type=int, default=26)
    args = parser.parse_args()
    UUID(args.recording_id)
    output = args.output_root / f"{args.recording_id}_video_trim.json"
    ensure_output_outside_raw(output, args.dataset_root)
    directory = args.dataset_root / "recordings" / args.recording_id / "mp4s"
    report = {"recording_id": args.recording_id, "participants": {}}
    for role in ("helper", "leader"):
        report["participants"][role] = probe_participant(
            mp4_directory=directory, role=role, anchor_count=args.anchors
        )
        result = report["participants"][role]
        if result["tail_full_decoded_rgb_sha256"]["trimmed_unique_images"] == 1:
            result["expanded_repeated_tail_check"] = probe_tail_repetition(
                mp4_directory=directory, role=role
            )
        print(
            json.dumps(
                {
                    "participant": role,
                    "candidate_offset": result["candidate_integer_offset"],
                    "confident_offsets": result["confident_offset_counts"],
                    "sampled_frames": result["sampled_trimmed_frame_count"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            ),
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
