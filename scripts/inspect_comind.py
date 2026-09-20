#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import av
except ImportError:
    av = None


DEFAULT_RECORDING_ID = "43276420-701f-4731-b9ab-bebc7fd14994"

MAX_JSON_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_NAMES = 200

KEYWORDS = {
    "timestamp": [
        "timestamp",
        "time",
        "tracking_timestamp",
        "device_timestamp",
        "capture_timestamp",
    ],
    "transform": [
        "transform",
        "pose",
        "trajectory",
        "translation",
        "rotation",
        "quaternion",
        "t_world",
        "t_device",
        "t_camera",
    ],
    "calibration": [
        "calibration",
        "intrinsic",
        "extrinsic",
        "camera_model",
        "focal",
        "principal",
        "distortion",
    ],
    "hand": [
        "hand",
        "wrist",
        "palm",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    ],
    "gaze": [
        "gaze",
        "eye",
    ],
    "slam": [
        "slam",
        "pointcloud",
        "point_cloud",
        "semidense",
        "global",
        "world",
    ],
}


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def is_partial(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower.endswith(".part")
        or lower.endswith(".partial")
        or lower.endswith(".tmp")
        or ".part." in lower
    )


def keyword_matches(values: list[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}

    for category, words in KEYWORDS.items():
        hits = []
        for value in values:
            lower = value.lower()
            if any(word in lower for word in words):
                hits.append(value)

        if hits:
            matches[category] = sorted(set(hits))

    return matches


def collect_json_fields(
    obj: Any,
    prefix: str = "",
    depth: int = 0,
    max_depth: int = 3,
) -> set[str]:
    fields: set[str] = set()

    if depth > max_depth:
        return fields

    if isinstance(obj, dict):
        for key, value in list(obj.items())[:100]:
            current = f"{prefix}.{key}" if prefix else str(key)
            fields.add(current)
            fields |= collect_json_fields(
                value,
                prefix=current,
                depth=depth + 1,
                max_depth=max_depth,
            )

    elif isinstance(obj, list) and obj:
        fields |= collect_json_fields(
            obj[0],
            prefix=prefix,
            depth=depth + 1,
            max_depth=max_depth,
        )

    return fields


def inspect_json(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "json",
        "size_bytes": path.stat().st_size,
    }

    if path.stat().st_size > MAX_JSON_BYTES:
        info["status"] = "skipped_large_json"
        return info

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        info["status"] = "ok"
        info["top_level_type"] = type(data).__name__

        if isinstance(data, dict):
            info["top_level_keys"] = list(data.keys())[:100]

        elif isinstance(data, list):
            info["length"] = len(data)

        fields = sorted(collect_json_fields(data))
        info["fields"] = fields[:500]
        info["keyword_matches"] = keyword_matches(fields)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_jsonl(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "jsonl",
        "size_bytes": path.stat().st_size,
    }

    try:
        with path.open("r", encoding="utf-8") as f:
            line = f.readline()

        obj = json.loads(line)
        fields = sorted(collect_json_fields(obj))

        info["status"] = "ok"
        info["first_record_fields"] = fields[:500]
        info["keyword_matches"] = keyword_matches(fields)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def open_text(path: Path):
    lower = path.name.lower()

    if lower.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return path.open("r", encoding="utf-8", errors="replace")


def inspect_delimited(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "delimited_text",
        "size_bytes": path.stat().st_size,
    }

    try:
        with open_text(path) as f:
            first_line = f.readline()

            if not first_line:
                info["status"] = "empty"
                return info

            if "\t" in first_line and first_line.count("\t") > first_line.count(","):
                delimiter = "\t"
            else:
                delimiter = ","

            header = next(csv.reader([first_line], delimiter=delimiter))

        header = [item.strip() for item in header]

        info["status"] = "ok"
        info["delimiter"] = "\\t" if delimiter == "\t" else delimiter
        info["columns"] = header
        info["keyword_matches"] = keyword_matches(header)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_video(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "video",
        "size_bytes": path.stat().st_size,
    }

    if av is None:
        info["status"] = "pyav_not_installed"
        return info

    try:
        container = av.open(str(path))

        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]

        info["status"] = "ok"
        info["video_stream_count"] = len(video_streams)
        info["audio_stream_count"] = len(audio_streams)

        streams = []

        for stream in video_streams:
            codec_name = None
            try:
                codec_name = stream.codec_context.name
            except Exception:
                pass

            average_rate = None
            if stream.average_rate is not None:
                try:
                    average_rate = float(stream.average_rate)
                except Exception:
                    average_rate = str(stream.average_rate)

            stream_duration_s = None
            if stream.duration is not None and stream.time_base is not None:
                try:
                    stream_duration_s = float(stream.duration * stream.time_base)
                except Exception:
                    pass

            streams.append(
                {
                    "index": stream.index,
                    "codec": codec_name,
                    "width": stream.width,
                    "height": stream.height,
                    "average_rate_fps": average_rate,
                    "frames": stream.frames,
                    "time_base": str(stream.time_base),
                    "start_time": stream.start_time,
                    "duration_seconds": stream_duration_s,
                }
            )

        info["video_streams"] = streams

        if container.duration is not None:
            try:
                info["container_duration_seconds"] = (
                    float(container.duration) / 1_000_000.0
                )
            except Exception:
                pass

        container.close()

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_pcd(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "pcd",
        "size_bytes": path.stat().st_size,
    }

    try:
        header = []

        with path.open("rb") as f:
            for _ in range(100):
                line = f.readline()

                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace").strip()
                header.append(decoded)

                if decoded.upper().startswith("DATA"):
                    break

        info["status"] = "ok"
        info["header"] = header
        info["keyword_matches"] = keyword_matches(header)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_zip(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "zip",
        "size_bytes": path.stat().st_size,
    }

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()

        info["status"] = "ok"
        info["file_count"] = len(names)
        info["files"] = names[:MAX_ARCHIVE_NAMES]
        info["keyword_matches"] = keyword_matches(names)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_tar(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "type": "tar",
        "size_bytes": path.stat().st_size,
    }

    try:
        with tarfile.open(path) as archive:
            names = archive.getnames()

        info["status"] = "ok"
        info["file_count"] = len(names)
        info["files"] = names[:MAX_ARCHIVE_NAMES]
        info["keyword_matches"] = keyword_matches(names)

    except Exception as exc:
        info["status"] = "error"
        info["error"] = str(exc)

    return info


def inspect_file(path: Path) -> dict[str, Any] | None:
    lower = path.name.lower()

    if is_partial(path):
        return {
            "type": "partial_download",
            "status": "skipped",
            "size_bytes": path.stat().st_size,
        }

    if lower.endswith(".mp4") or lower.endswith(".mov"):
        return inspect_video(path)

    if lower.endswith(".json"):
        return inspect_json(path)

    if lower.endswith(".jsonl"):
        return inspect_jsonl(path)

    if (
        lower.endswith(".csv")
        or lower.endswith(".tsv")
        or lower.endswith(".csv.gz")
        or lower.endswith(".tsv.gz")
    ):
        return inspect_delimited(path)

    if lower.endswith(".pcd"):
        return inspect_pcd(path)

    if lower.endswith(".zip"):
        return inspect_zip(path)

    if (
        lower.endswith(".tar")
        or lower.endswith(".tar.gz")
        or lower.endswith(".tgz")
    ):
        return inspect_tar(path)

    return None


def make_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []

    lines.append("# CoMind Automatic Inspection")
    lines.append("")
    lines.append(f"Recording: `{report['recording_id']}`")
    lines.append("")
    lines.append(f"Recording root: `{report['recording_root']}`")
    lines.append("")

    summary = report["summary"]

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Files found: {summary['file_count']}")
    lines.append(f"- Total size currently present: {summary['total_size_human']}")
    lines.append(f"- Partial/incomplete files detected: {summary['partial_file_count']}")
    lines.append("")

    lines.append("## Extension counts")
    lines.append("")

    for extension, count in summary["extension_counts"].items():
        lines.append(f"- `{extension}`: {count}")

    lines.append("")
    lines.append("## File inventory")
    lines.append("")

    for item in report["files"]:
        status = ""

        if item.get("partial"):
            status = " [PARTIAL]"

        lines.append(
            f"- `{item['relative_path']}` "
            f"({item['size_human']}){status}"
        )

    lines.append("")

    if report["videos"]:
        lines.append("## Video inspection")
        lines.append("")

        for item in report["videos"]:
            lines.append(f"### `{item['relative_path']}`")
            lines.append("")
            lines.append(f"- Status: `{item['status']}`")

            if item["status"] == "ok":
                lines.append(
                    f"- Video streams: {item.get('video_stream_count', 0)}"
                )
                lines.append(
                    f"- Audio streams: {item.get('audio_stream_count', 0)}"
                )

                duration = item.get("container_duration_seconds")
                if duration is not None:
                    lines.append(f"- Container duration: {duration:.3f} s")

                for stream in item.get("video_streams", []):
                    lines.append(
                        "- Stream "
                        f"{stream['index']}: "
                        f"{stream.get('width')}x{stream.get('height')}, "
                        f"codec={stream.get('codec')}, "
                        f"fps={stream.get('average_rate_fps')}, "
                        f"time_base={stream.get('time_base')}, "
                        f"duration={stream.get('duration_seconds')}"
                    )

            else:
                lines.append(f"- Error: `{item.get('error', '')}`")

            lines.append("")

    if report["structured_files"]:
        lines.append("## Structured metadata inspection")
        lines.append("")

        for item in report["structured_files"]:
            lines.append(f"### `{item['relative_path']}`")
            lines.append("")
            lines.append(f"- Type: `{item['type']}`")
            lines.append(f"- Status: `{item['status']}`")

            if item.get("columns"):
                lines.append("- Columns:")
                for column in item["columns"]:
                    lines.append(f"  - `{column}`")

            if item.get("top_level_keys"):
                lines.append("- Top-level JSON keys:")
                for key in item["top_level_keys"]:
                    lines.append(f"  - `{key}`")

            if item.get("fields"):
                lines.append("- Sample JSON field paths:")
                for field in item["fields"][:100]:
                    lines.append(f"  - `{field}`")

            matches = item.get("keyword_matches", {})
            if matches:
                lines.append("- Potentially relevant fields:")

                for category, values in matches.items():
                    lines.append(f"  - {category}:")
                    for value in values[:50]:
                        lines.append(f"    - `{value}`")

            if item.get("error"):
                lines.append(f"- Error: `{item['error']}`")

            lines.append("")

    if report["pcd_files"]:
        lines.append("## PCD scan inspection")
        lines.append("")

        for item in report["pcd_files"]:
            lines.append(f"### `{item['relative_path']}`")
            lines.append("")
            lines.append(f"- Status: `{item['status']}`")

            for header_line in item.get("header", []):
                lines.append(f"- `{header_line}`")

            lines.append("")

    if report["archives"]:
        lines.append("## Archive inspection")
        lines.append("")

        for item in report["archives"]:
            lines.append(f"### `{item['relative_path']}`")
            lines.append("")
            lines.append(f"- Status: `{item['status']}`")
            lines.append(f"- Files inside: {item.get('file_count', 'unknown')}")

            for name in item.get("files", [])[:100]:
                lines.append(f"  - `{name}`")

            lines.append("")

    lines.append("## Filename keyword candidates")
    lines.append("")

    filename_matches = report.get("filename_keyword_matches", {})

    if filename_matches:
        for category, values in filename_matches.items():
            lines.append(f"### {category}")
            lines.append("")

            for value in values:
                lines.append(f"- `{value}`")

            lines.append("")
    else:
        lines.append("No keyword candidates found.")
        lines.append("")

    lines.append("## Important note")
    lines.append("")
    lines.append(
        "This report inventories and summarizes files. "
        "It does not establish coordinate-frame semantics. "
        "Transform directions, units, clock domains, and participant identities "
        "must still be verified from dataset documentation and geometry checks."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a downloaded CoMind recording without modifying raw data."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/comind"),
        help="CoMind dataset root",
    )

    parser.add_argument(
        "--recording-id",
        default=DEFAULT_RECORDING_ID,
        help="Recording UUID to inspect",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comind_inspection"),
        help="Directory for generated reports",
    )

    args = parser.parse_args()

    root = args.root
    recording_root = root / "recordings" / args.recording_id

    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    if not recording_root.exists():
        print(f"Recording directory does not exist yet: {recording_root}")
        print("The download may not have created it yet.")
        print("Rerun this script later.")
        return

    files = sorted(
        path
        for path in recording_root.rglob("*")
        if path.is_file()
    )

    inventory = []
    videos = []
    structured_files = []
    pcd_files = []
    archives = []

    total_size = 0
    partial_count = 0

    extension_counter: Counter[str] = Counter()

    relative_names = []

    for path in files:
        size = path.stat().st_size
        total_size += size

        relative = path.relative_to(recording_root).as_posix()
        relative_names.append(relative)

        partial = is_partial(path)

        if partial:
            partial_count += 1

        suffixes = "".join(path.suffixes).lower()
        extension = suffixes if suffixes else "<no extension>"

        extension_counter[extension] += 1

        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "size_human": human_size(size),
                "partial": partial,
            }
        )

        inspected = inspect_file(path)

        if inspected is None:
            continue

        inspected["relative_path"] = relative
        inspected["size_human"] = human_size(size)

        if inspected["type"] == "video":
            videos.append(inspected)

        elif inspected["type"] in {
            "json",
            "jsonl",
            "delimited_text",
        }:
            structured_files.append(inspected)

        elif inspected["type"] == "pcd":
            pcd_files.append(inspected)

        elif inspected["type"] in {"zip", "tar"}:
            archives.append(inspected)

    common_annotation_files = []

    annotations_root = root / "annotations"

    if annotations_root.exists():
        for path in sorted(annotations_root.rglob("*")):
            if path.is_file():
                common_annotation_files.append(
                    {
                        "relative_path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "size_human": human_size(path.stat().st_size),
                    }
                )

    report = {
        "recording_id": args.recording_id,
        "dataset_root": str(root),
        "recording_root": str(recording_root),
        "summary": {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "total_size_human": human_size(total_size),
            "partial_file_count": partial_count,
            "extension_counts": dict(sorted(extension_counter.items())),
        },
        "files": inventory,
        "videos": videos,
        "structured_files": structured_files,
        "pcd_files": pcd_files,
        "archives": archives,
        "common_annotation_files": common_annotation_files,
        "filename_keyword_matches": keyword_matches(relative_names),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / f"{args.recording_id}.json"
    md_path = args.output_dir / f"{args.recording_id}.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_path.write_text(
        make_markdown(report),
        encoding="utf-8",
    )

    print("")
    print("CoMind inspection complete")
    print("==========================")
    print(f"Recording: {args.recording_id}")
    print(f"Files found: {len(files)}")
    print(f"Size currently present: {human_size(total_size)}")
    print(f"Partial files: {partial_count}")
    print("")
    print(f"Markdown report: {md_path}")
    print(f"JSON report:     {json_path}")
    print("")

    if partial_count:
        print(
            "Download is still incomplete. "
            "Run this command again after the downloader finishes."
        )


if __name__ == "__main__":
    main()
