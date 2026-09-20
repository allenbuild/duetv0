#!/usr/bin/env python3
"""Call Project Aria's official timestamp API using MP4 headers, without decoding."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

from duet.adapters.comind.mp4_timestamps import (
    _load_official_extractor,
    probe_mp4_device_timestamps,
)
from duet.adapters.comind.validation import ensure_output_outside_raw


def run_probe(
    *,
    dataset_root: Path,
    recording_id: str,
    output_root: Path,
    validation_report: Path | None,
    expected_trimmed_frame_count: int = 21109,
) -> dict[str, object]:
    """Reuse existing trajectory ranges; write only a small derived JSON report."""
    UUID(recording_id)
    dataset_root = dataset_root.resolve()
    output_path = output_root / f"{recording_id}.json"
    ensure_output_outside_raw(output_path, dataset_root)
    ranges: dict[str, dict[str, tuple[int, int]]] = {role: {} for role in ("helper", "leader")}
    if validation_report is not None:
        audit = json.loads(validation_report.read_text())
        if audit["recording_id"] != recording_id:
            raise ValueError("trajectory range audit belongs to another recording")
        for role, role_ranges in ranges.items():
            for source in ("standard_mps", "multislam", "hands"):
                bounds = audit["participants"][role][source]["tracking_timestamp_us_range"]
                if bounds is not None:
                    role_ranges[source] = tuple(bounds)
    report: dict[str, object] = {
        "recording_id": recording_id,
        "method": "Unmodified installed official function; MP4 headers only; no video decode",
        "trajectory_range_source": None if validation_report is None else str(validation_report),
        "trajectory_sources_reread": False,
        "frame_index_to_device_time_verified": False,
        "cross_participant_video_to_device_assembly_available": False,
        "videos": {},
    }
    ffprobe = shutil.which("ffprobe")
    report["ffprobe_executable"] = ffprobe
    if ffprobe:
        completed = subprocess.run(
            [ffprobe, "-version"], capture_output=True, text=True, check=True
        )
        report["ffprobe_version"] = completed.stdout.splitlines()[0]
    try:
        extractor, function_path, package_version = _load_official_extractor()
        source_file = Path(inspect.getfile(extractor))
        report["official_implementation"] = {
            "function": function_path,
            "projectaria_tools_version": package_version,
            "source_path": str(source_file),
            "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
            "parser_behavior": (
                "Reads format.tags.description, removes square brackets, splits commas, "
                "converts each token with int(), returns numpy.array; a bare scalar becomes "
                "a one-element array without a frame-count check."
            ),
        }
    except (ImportError, AttributeError, TypeError, OSError) as error:
        report["official_import_error"] = f"{type(error).__name__}: {error}"
    packages = {}
    for package in (
        "projectaria-tools",
        "moviepy",
        "imageio",
        "decorator",
        "proglog",
        "python-dotenv",
        "tqdm",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report["optional_runtime_packages"] = packages
    paths = []
    for role in ("helper", "leader"):
        for suffix in ("trimmed_sync", "sync"):
            path = dataset_root / "recordings" / recording_id / "mp4s" / f"{role}_{suffix}.mp4"
            if not path.exists():
                report["videos"][path.name] = {"missing_file": True}
                continue
            before = path.stat()
            paths.append(path)
            outcome = probe_mp4_device_timestamps(
                path,
                recording_id=recording_id,
                participant=role,
                expected_frame_count=expected_trimmed_frame_count
                if suffix == "trimmed_sync"
                else None,
                trajectory_ranges_us=ranges[role],
            )
            after = path.stat()
            outcome["raw_file_size_and_mtime_unchanged"] = (before.st_size, before.st_mtime_ns) == (
                after.st_size,
                after.st_mtime_ns,
            )
            report["videos"][path.name] = outcome
    complete = all(
        report["videos"]
        .get(f"{role}_trimmed_sync.mp4", {})
        .get("per_frame_device_mapping_verified")
        for role in ("helper", "leader")
    )
    report["frame_index_to_device_time_verified"] = complete
    report["cross_participant_video_to_device_assembly_available"] = False
    report["additional_frame_alignment_evidence_required"] = True
    report["interpretation"] = (
        "Both trimmed local mappings are complete; explicit CoMind paired-frame evidence is "
        "still required to assemble a common frame-index timeline."
        if complete
        else "Existing MP4 metadata does not supply one device timestamp per frame. Scalar tags "
        "must not be expanded using FPS, PTS, UTC, or inferred offsets. An exporter-produced "
        "per-frame timestamp sidecar with verified trimmed-frame correspondence could resolve "
        "this without VRS; no such sidecar is established by this probe."
    )
    report["raw_files_opened"] = [str(path) for path in paths]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/comind_timestamp_probe"))
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--expected-trimmed-frame-count", type=int, default=21109)
    parser.add_argument(
        "--ffmpeg-directory",
        type=Path,
        help="Existing directory containing ffprobe and ffmpeg; no binaries are downloaded",
    )
    args = parser.parse_args()
    if args.ffmpeg_directory is not None:
        directory = args.ffmpeg_directory.resolve()
        if not (directory / "ffprobe").is_file() or not (directory / "ffmpeg").is_file():
            parser.error("--ffmpeg-directory must contain both ffprobe and ffmpeg")
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        # MoviePy imports the export API even though this probe only extracts metadata.
        os.environ["FFMPEG_BINARY"] = str(directory / "ffmpeg")
    report = run_probe(
        dataset_root=args.dataset_root,
        recording_id=args.recording_id,
        output_root=args.output_root,
        validation_report=args.validation_report,
        expected_trimmed_frame_count=args.expected_trimmed_frame_count,
    )
    print(
        json.dumps(
            {
                "report": str(args.output_root / f"{args.recording_id}.json"),
                "frame_index_to_device_time_verified": report[
                    "frame_index_to_device_time_verified"
                ],
                "timestamp_counts": {
                    name: result.get("extracted_timestamp_count")
                    for name, result in report["videos"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
