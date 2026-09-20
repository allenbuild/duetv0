#!/usr/bin/env python3
"""Render all verified CoMind handovers and metric-ranked V0 compilations.

Reuse existing full-recording mappings/aligned arrays; only selected source MP4
windows are decoded. Four seconds of context are included on each side. With
--qc-passing-only, each event is gated independently, with a three-second
fallback only when the four-second mapping context is unavailable. Use
--skip-compilations to write individual clips and rejection reports only. Source
annotations are read only when an existing observation report is unavailable;
VRS, raw trajectories and scans are never read.

Use --qc-profile demo for the separately disclosed partial-hand demonstration
profile. Its clips, manifest, validation and montage use distinct demo_qc paths;
it never overwrites strict-profile artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

import numpy as np
from PIL import Image

from duet.adapters.comind import demo_metrics as demo_metric_api
from duet.adapters.comind.annotations import load_handover_annotations
from duet.adapters.comind.demo_metrics import annotation_metrics, rank_examples
from duet.adapters.comind.frame_map import VideoVrsFrameMap, load_frame_map
from duet.adapters.comind.indexed_video import iter_indexed_rgb
from duet.adapters.comind.playback import PlaybackBundle, load_playback
from duet.visualization.comind_handover import assemble_index_frame
from duet.visualization.demo_compilation import choose_segment, export_compilation
from duet.visualization.demo_compositor import (
    HEIGHT,
    WIDTH,
    WORLD_PANEL,
    iter_composed_frames,
    make_thumbnail,
)
from duet.visualization.demo_encoding import encode_video, inspect_video
from duet.visualization.demo_geometry import evaluate_view, fit_view

ROOT = Path(__file__).resolve().parents[1]
RECORDING_ID = "43276420-701f-4731-b9ab-bebc7fd14994"
FPS = 30


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _derived(path: Path, dataset_root: Path) -> Path:
    path = path.resolve()
    if any(path.is_relative_to(raw.resolve()) for raw in (ROOT / "data/raw", dataset_root)):
        raise ValueError("derived inputs and outputs must stay outside immutable raw data")
    return path


def bind_numeric_ids(rows: Sequence[dict], source_report: Mapping, recording_id: str) -> None:
    """Keep raw numeric_id distinct from the opaque zero-padded annotation key."""
    if source_report.get("recording_id") != recording_id:
        raise ValueError("source annotation observations belong to another recording")
    segments = source_report["annotation_observations"]["segments"]
    for row in rows:
        if row.get("recording_id") != recording_id:
            raise ValueError("metric row belongs to another recording")
        matches = [
            item
            for item in segments
            if item["source_key"] == row["annotation_id"] and item.get("skip") is False
        ]
        if len(matches) != 1:
            raise ValueError("numeric_id requires exactly one usable source annotation")
        source = matches[0]
        if (source["start_frame"], source["end_frame"]) != (row["start_frame"], row["end_frame"]):
            raise ValueError(
                "source numeric_id annotation boundaries differ from the verified binding"
            )
        if source["object_category_level_1"] != row["object_category"]:
            raise ValueError("source annotation category differs from the verified binding")
        row["numeric_id"] = source["numeric_id"]


def load_annotation_source_report(
    report_path: Path, *, dataset_root: Path, recording_id: str
) -> Mapping:
    """Reuse prior observations or read only the existing annotation JSON.

    These fields preserve source labels and IDs. They establish no temporal,
    spatial, image-orientation, or physical participant correspondence.
    """
    if report_path.is_file():
        return json.loads(report_path.read_text())
    source = dataset_root / "annotations" / "dataset_handover_consolidated.json"
    annotations = load_handover_annotations(source, recording_uuid=recording_id)
    return {
        "recording_id": recording_id,
        "inspection_scope": "Source annotation observations only; no mapping verification",
        "annotation_source": str(source.resolve()),
        "annotation_observations": {
            "segments": [
                {
                    "source_key": annotation.annotation_id,
                    "skip": annotation.source_fields["skip"],
                    "numeric_id": annotation.source_fields["numeric_id"],
                    "start_frame": annotation.start_frame,
                    "end_frame": annotation.end_frame,
                    "object_category_level_1": annotation.object_category_level_1,
                }
                for annotation in annotations
            ]
        },
    }


def _write_json(path: Path, data: Mapping) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def plan_annotations(
    source_report: Mapping,
    maps: Mapping[str, VideoVrsFrameMap],
    recording_id: str,
    *,
    mapping_reasons: Mapping[str, np.ndarray] | None = None,
) -> list[dict]:
    """Plan independent inclusive source intervals without filling mapping gaps.

    Source frame labels are observations here. Canonical playback subsequently
    requires the explicit verified annotation binding from the temporal report.
    Only literal VERIFIED rows pass, even when an inferred row has a timestamp.
    """
    if source_report.get("recording_id") != recording_id:
        raise ValueError("source annotation observations belong to another recording")
    count = maps["helper"].frame_count
    for role in ("helper", "leader"):
        mapping = maps[role]
        if (mapping.recording_id, mapping.participant, mapping.frame_count) != (
            recording_id,
            role,
            count,
        ):
            raise ValueError("paired frame map identities or counts differ")
        reasons = (mapping_reasons or {}).get(role)
        if reasons is not None and reasons.shape != (count,):
            raise ValueError("mapping reasons must follow the complete frame map")
    segments = [
        item
        for item in source_report["annotation_observations"]["segments"]
        if item.get("skip") is False
    ]
    ids = [item["source_key"] for item in segments]
    if len(set(ids)) != len(ids):
        raise ValueError("source annotation IDs must be unique")
    records = []
    for item in sorted(segments, key=lambda value: value["start_frame"]):
        start, end = item["start_frame"], item["end_frame"]
        record = {
            "recording_id": recording_id,
            "annotation_id": item["source_key"],
            "numeric_id": item["numeric_id"],
            "object_category": item["object_category_level_1"],
            "start_frame": start,
            "end_frame": end,
            "context_frames_each_side": None,
            "clip_start_frame": None,
            "clip_end_frame": None,
            "actual_context_before_frames": None,
            "actual_context_after_frames": None,
            "context_policy": "120 frames each side; 90 only if 120 fails the mapping gate",
            "export_status": "pending",
            "rejection_stage": "",
            "rejection_reason": "",
            "mapping_rejections": [],
            "public_recommendation": None,
            "public_recommendation_reason": None,
            "event_all_four_hands_coverage": None,
            "context_all_four_hands_coverage": None,
            "video_path": None,
            "thumbnail_path": None,
        }
        records.append(record)
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < count:
            _reject(record, "annotation_bounds", "annotation bounds are outside paired videos")
            continue
        for context in (0, 120, 90):
            left, right = max(0, start - context), min(count - 1, end + context)
            failed = []
            for role, mapping in maps.items():
                indices = np.flatnonzero(mapping.status[left : right + 1] != "VERIFIED") + left
                reasons = (mapping_reasons or {}).get(role)
                for index in indices:
                    failed.append(
                        {
                            "participant": role,
                            "frame_index": int(index),
                            "status": str(mapping.status[index]),
                            "mapping_reason": str(reasons[index]) if reasons is not None else None,
                        }
                    )
            if failed:
                record["mapping_rejections"].append({"context_frames": context, "frames": failed})
                if context == 0:
                    _reject(record, "event_mapping", "event contains a non-VERIFIED mapping")
                    break
            elif context:
                record.update(
                    context_frames_each_side=context,
                    clip_start_frame=left,
                    clip_end_frame=right,
                    actual_context_before_frames=start - left,
                    actual_context_after_frames=right - end,
                )
                break
        if record["export_status"] != "rejected" and record["context_frames_each_side"] is None:
            _reject(record, "context_mapping", "neither 4s nor 3s context is fully VERIFIED")
    return records


def _reject(record: dict, stage: str, reason: object) -> None:
    record.update(export_status="rejected", rejection_stage=stage, rejection_reason=str(reason))


def _write_csv(path: Path, rows: Sequence[Mapping], *, empty_fields: Sequence[str]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, allow_nan=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _write_batch_reports(
    output: Path,
    recording_id: str,
    records: list[dict],
    rows: list[dict],
    *,
    dataset_root: Path,
    compilations: Mapping | None = None,
) -> dict:
    fields = ("recording_id", "annotation_id", "numeric_id", "export_status", "rejection_reason")
    rejected = [row for row in records if row["export_status"] == "rejected"]
    manifest = _derived(output / "examples_manifest.csv", dataset_root)
    _write_csv(manifest, rows, empty_fields=fields)
    _write_csv(
        _derived(output / "examples_eligibility.csv", dataset_root), records, empty_fields=fields
    )
    rejected_manifest = _derived(output / "examples_rejected.csv", dataset_root)
    _write_csv(rejected_manifest, rejected, empty_fields=fields)
    summary = {
        "recording_id": recording_id,
        "qc_passing_only": True,
        "status": "EXPORTED" if rows else "NO_ELIGIBLE_EXPORTS",
        "context_policy": "4s each side preferred; 3s only when the full 4s mapping context fails",
        "endpoint_policy": "inclusive event and context frame bounds",
        "gates": [
            "both event/context maps VERIFIED",
            "both calibrated frustums VERIFIED",
            "canonical hash/generation/native timestamp checks",
            "public tracking QC not excluded",
        ],
        "rows": rows,
        "annotations": records,
        "rejected_annotations": rejected,
        "compilations": dict(compilations or {}),
        "manifest_path": str(manifest),
        "rejected_manifest_path": str(rejected_manifest),
        "scene_included": False,
        "vrs_matching_performed": False,
    }
    _write_json(_derived(output / "examples_validation.json", dataset_root), summary)
    return summary


def _qc_passing_export(
    args,
    *,
    recording_id: str,
    output: Path,
    examples: Path,
    report_path: Path,
    projection_path: Path,
    source_path: Path,
) -> int:
    processed = _derived(args.processed_root, args.dataset_root)
    directory = processed / recording_id
    source = load_annotation_source_report(
        source_path, dataset_root=args.dataset_root, recording_id=recording_id
    )
    maps, reasons = {}, {}
    for role in ("helper", "leader"):
        path = _derived(directory / "frame_maps" / f"{role}_frame_map.npz", args.dataset_root)
        maps[role] = load_frame_map(path)
        with np.load(path, allow_pickle=False) as arrays:
            if "mapping_reason" in arrays.files:
                reasons[role] = arrays["mapping_reason"]
    records = plan_annotations(source, maps, recording_id, mapping_reasons=reasons)
    # Persist map rejections before loading heavier derived validation caches.
    _write_batch_reports(output, recording_id, records, [], dataset_root=args.dataset_root)
    candidates = [record for record in records if record["export_status"] != "rejected"]
    base, projection = None, None
    if candidates:
        try:
            for role in ("helper", "leader"):
                calibration = json.loads(
                    (directory / "vrs" / f"{role}_export_rgb_calibration.json").read_text()
                )
                if (
                    calibration.get("recording_id"),
                    calibration.get("participant"),
                    calibration.get("status"),
                ) != (recording_id, role, "VERIFIED"):
                    raise ValueError(
                        f"{role} calibrated export frustum is not VERIFIED for this recording"
                    )
            projection = json.loads(projection_path.read_text())
            if projection["validation_report_sha256"] != _digest(report_path):
                raise ValueError(
                    "projection QC does not refer to the current temporal validation report"
                )
            first = candidates[0]
            base = load_playback(
                processed,
                report_path,
                recording_id,
                annotation_id=first["annotation_id"],
                context_frames=first["context_frames_each_side"],
            )
            if any(
                role not in base.bindings.camera_extrinsics
                or role not in base.bindings.image_geometry
                for role in ("helper", "leader")
            ):
                raise ValueError("both calibrated export frustums must be VERIFIED")
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            for record in candidates:
                _reject(record, "derived_validation", exc)
            candidates = []
    prepared = []
    for record in candidates:
        try:
            bundle = base.with_annotation(
                record["annotation_id"], context_frames=record["context_frames_each_side"]
            )
            row = annotation_metrics(bundle, projection)
            bind_numeric_ids([row], source, recording_id)
            for key in (
                "public_recommendation",
                "public_recommendation_reason",
                "event_all_four_hands_coverage",
                "context_all_four_hands_coverage",
            ):
                record[key] = row.get(key)
            if row["public_recommendation"] == "exclude_from_public_demo":
                _reject(
                    record,
                    "tracking_qc",
                    row.get("public_recommendation_reason")
                    or "public_recommendation=exclude_from_public_demo",
                )
                continue
            if row["public_recommendation"] not in ("acceptable", "tracking_caution"):
                raise ValueError("unrecognized public tracking recommendation")
            row.update(
                context_frames_each_side=record["context_frames_each_side"],
                actual_context_before_frames=record["actual_context_before_frames"],
                actual_context_after_frames=record["actual_context_after_frames"],
            )
            prepared.append((record, bundle, row))
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            _reject(record, "annotation_validation", exc)
    rows = []
    if prepared:
        input_identity = {
            "validation_report_sha256": _digest(report_path),
            "renderer_sha256": _digest(ROOT / "src/duet/visualization/demo_geometry.py"),
            "compositor_sha256": _digest(ROOT / "src/duet/visualization/demo_compositor.py"),
            "timeline": "comind_sync_frame_index",
            "fps": FPS,
        }
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _render_example,
                    bundle,
                    row,
                    dataset_root=args.dataset_root,
                    output_dir=examples,
                    input_identity=input_identity,
                    reuse_clips=args.reuse_clips,
                )
                for _, bundle, row in prepared
            ]
            for (record, _, row), future in zip(prepared, futures, strict=True):
                try:
                    artifact = future.result()
                    row.update(
                        video_path=artifact["artifacts"]["video"],
                        thumbnail_path=artifact["artifacts"]["thumbnail"],
                        file_size_bytes=artifact["video"]["file_size_bytes"],
                        frame_count=artifact["video"]["frame_count"],
                    )
                    record.update(
                        export_status="exported",
                        video_path=row["video_path"],
                        thumbnail_path=row["thumbnail_path"],
                    )
                    rows.append(row)
                except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                    _reject(record, "render_validation", exc)
    rows = rank_examples(rows) if rows else []
    compilations = {}
    _write_batch_reports(output, recording_id, records, rows, dataset_root=args.dataset_root)
    if not args.skip_compilations and rows:
        for name, selected, best in (
            ("duet_v0_montage", rows, False),
            ("duet_v0_best_examples", rows[:3], True),
        ):
            if best and len(selected) < 3:
                compilations[name] = {
                    "status": "skipped",
                    "reason": "fewer than three QC-passing exports",
                }
                continue
            try:
                compilations[name] = export_compilation(
                    _derived(output / f"{name}.mp4", args.dataset_root),
                    _segments(selected, best=best),
                    fps=FPS,
                )
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                compilations[name] = {"status": "failed", "reason": str(exc)}
    summary = _write_batch_reports(
        output,
        recording_id,
        records,
        rows,
        dataset_root=args.dataset_root,
        compilations=compilations,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "exported": len(rows),
                "rejected": len(summary["rejected_annotations"]),
                "manifest": summary["manifest_path"],
                "rejections": summary["rejected_manifest_path"],
            },
            indent=2,
        )
    )
    return 0


def _write_demo_reports(
    output: Path,
    recording_id: str,
    rows: list[dict],
    *,
    dataset_root: Path,
    montage: Mapping | None = None,
) -> dict:
    """Write only the separate demonstration profile's artifacts, including failures."""
    manifest = _derived(output / "demo_qc_manifest.csv", dataset_root)
    _write_csv(
        manifest,
        rows,
        empty_fields=(
            "recording_id",
            "annotation_id",
            "numeric_id",
            "strict_qc_pass",
            "demo_qc_pass",
            "export_status",
        ),
    )
    summary = {
        "recording_id": recording_id,
        "qc_profile": "demo",
        "strict_policy_unchanged": True,
        "annotation_count": len(rows),
        "strict_qc_pass_count": sum(row.get("strict_qc_pass") is True for row in rows),
        "demo_qc_pass_count": sum(row.get("demo_qc_pass") is True for row in rows),
        "exported_count": sum(row.get("export_status") == "exported" for row in rows),
        "context_policy": "4s each side preferred; 3s only when the full 4s mapping context fails",
        "endpoint_policy": "inclusive event and context frame bounds",
        "tracking_display": "Only accepted per-hand observations are drawn; missing hands remain absent",
        "qc_label": "Partial hand tracking when at least one displayed hand observation is missing",
        "mapping_policy": "Both full event/context maps must remain VERIFIED; diagnostics do not grant playback",
        "geometry_policy": "Both canonical calibrated export frustums required",
        "rejected_context_metrics": "Unplayable contexts are measured from validated caches, never interpolated or rendered",
        "montage_segment_frames": 4 * FPS,
        "montage": dict(montage or {}),
        "rows": rows,
        "manifest_path": str(manifest),
        "scene_included": False,
        "vrs_matching_performed": False,
    }
    _write_json(_derived(output / "demo_qc_validation.json", dataset_root), summary)
    return summary


def _demo_qc_export(
    args,
    *,
    recording_id: str,
    output: Path,
    examples: Path,
    report_path: Path,
    projection_path: Path,
    source_path: Path,
) -> int:
    processed = _derived(args.processed_root, args.dataset_root)
    directory = processed / recording_id
    source = load_annotation_source_report(
        source_path, dataset_root=args.dataset_root, recording_id=recording_id
    )
    maps, reasons = {}, {}
    for role in ("helper", "leader"):
        path = _derived(directory / "frame_maps" / f"{role}_frame_map.npz", args.dataset_root)
        maps[role] = load_frame_map(path)
        with np.load(path, allow_pickle=False) as arrays:
            if "mapping_reason" in arrays.files:
                reasons[role] = arrays["mapping_reason"]
    plans = plan_annotations(source, maps, recording_id, mapping_reasons=reasons)
    candidates = [row for row in plans if row["export_status"] != "rejected"]
    base, projection, common_error = None, None, None
    if candidates:
        try:
            projection = json.loads(projection_path.read_text())
            if projection["validation_report_sha256"] != _digest(report_path):
                raise ValueError(
                    "projection QC does not refer to the current temporal validation report"
                )
            first = candidates[0]
            base = load_playback(
                processed,
                report_path,
                recording_id,
                annotation_id=first["annotation_id"],
                context_frames=first["context_frames_each_side"],
            )
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            common_error = str(exc)
    else:
        common_error = "No fully VERIFIED playback context is available to load canonical caches"
    calibrated = base is not None and all(
        role in base.bindings.camera_extrinsics and role in base.bindings.image_geometry
        for role in ("helper", "leader")
    )
    rows, prepared = [], []
    for plan in plans:
        row = dict(plan)
        row.update(
            qc_profile="demo",
            strict_qc_pass=None,
            demo_qc_pass=False,
            demo_qc_reason=common_error,
            calibrated_frustums_verified=calibrated,
        )
        rows.append(row)
        if base is None:
            _reject(row, "derived_validation", common_error)
            continue
        context = plan["context_frames_each_side"] or 120
        try:
            assessment = demo_metric_api.assess_demo_qc(
                base, plan["annotation_id"], projection, context_frames=context
            )
            row.update(assessment)
            row["qc_profile"] = "demo"
            row["strict_qc_pass"] = row["public_recommendation"] in (
                "acceptable",
                "tracking_caution",
            )
            bind_numeric_ids([row], source, recording_id)
            if plan["export_status"] == "rejected":
                row["demo_qc_pass"] = False
                _reject(row, plan["rejection_stage"], plan["rejection_reason"])
                continue
            if not calibrated:
                row["demo_qc_pass"] = False
                _reject(row, "calibration", "both calibrated export frustums must be VERIFIED")
                continue
            if row.get("demo_qc_pass") is not True:
                _reject(row, "demo_tracking_qc", row.get("demo_qc_reason") or "DEMO_QC failed")
                continue
            bundle = base.with_annotation(plan["annotation_id"], context_frames=context)
            # The metrics-only assessor cannot authorize or bypass this canonical gate.
            if (bundle.start_index, bundle.end_index) != (
                row["clip_start_frame"],
                row["clip_end_frame"],
            ):
                raise ValueError("demo assessment bounds differ from canonical playback")
            row.update(
                context_frames_each_side=context,
                actual_context_before_frames=row["start_frame"] - bundle.start_index,
                actual_context_after_frames=bundle.end_index - row["end_frame"],
            )
            prepared.append((bundle, row))
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            row["demo_qc_pass"] = False
            _reject(row, "annotation_validation", exc)
    _write_demo_reports(output, recording_id, rows, dataset_root=args.dataset_root)
    if prepared:
        identity = {
            "validation_report_sha256": _digest(report_path),
            "renderer_sha256": _digest(ROOT / "src/duet/visualization/demo_geometry.py"),
            "compositor_sha256": _digest(ROOT / "src/duet/visualization/demo_compositor.py"),
            "qc_profile": "demo",
            "timeline": "comind_sync_frame_index",
            "fps": FPS,
        }
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    _render_example,
                    bundle,
                    row,
                    dataset_root=args.dataset_root,
                    output_dir=examples,
                    input_identity=identity,
                    reuse_clips=args.reuse_clips,
                )
                for bundle, row in prepared
            ]
            for (_, row), future in zip(prepared, futures, strict=True):
                try:
                    artifact = future.result()
                    row.update(
                        export_status="exported",
                        video_path=artifact["artifacts"]["video"],
                        thumbnail_path=artifact["artifacts"]["thumbnail"],
                        file_size_bytes=artifact["video"]["file_size_bytes"],
                        frame_count=artifact["video"]["frame_count"],
                    )
                except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                    _reject(row, "render_validation", exc)
    _write_demo_reports(output, recording_id, rows, dataset_root=args.dataset_root)
    exported = [row for row in rows if row["export_status"] == "exported"]
    montage = {"status": "skipped", "reason": "No exported clips"}
    if args.skip_compilations:
        montage["reason"] = "--skip-compilations requested"
    elif exported:
        try:
            montage = export_compilation(
                _derived(output / "duet_v0_montage_demo_qc.mp4", args.dataset_root),
                _demo_segments(exported),
                fps=FPS,
            )
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            montage = {"status": "failed", "reason": str(exc)}
    summary = _write_demo_reports(
        output, recording_id, rows, dataset_root=args.dataset_root, montage=montage
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "qc_profile",
                    "annotation_count",
                    "strict_qc_pass_count",
                    "demo_qc_pass_count",
                    "exported_count",
                    "manifest_path",
                )
            },
            indent=2,
        )
    )
    return 0


def _render_example(
    bundle: PlaybackBundle,
    row: dict,
    *,
    dataset_root: Path,
    output_dir: Path,
    input_identity: Mapping,
    reuse_clips: bool,
) -> dict:
    annotation_id = str(bundle.selection["annotation_id"])
    category = str(bundle.selection["object_category_level_1"])
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "object"
    stem = f"handover_{annotation_id}_{slug}"
    artifacts = {
        key: _derived(output_dir / f"{stem}{suffix}", dataset_root)
        for key, suffix in (
            ("video", ".mp4"),
            ("thumbnail", "_thumbnail.png"),
            ("first", "_first.png"),
            ("middle", "_middle.png"),
            ("last", "_last.png"),
            ("validation", "_validation.json"),
        )
    }
    count = bundle.end_index - bundle.start_index + 1
    gaps = any(
        not bundle.aligned_arrays[role]["hand_high_confidence"][
            bundle.start_index : bundle.end_index + 1
        ].all()
        for role in ("helper", "leader")
    )
    identity = {
        **input_identity,
        "annotation_id": annotation_id,
        "object_category": category,
        "start_frame": bundle.start_index,
        "end_frame": bundle.end_index,
        "qc_label": (
            "Partial hand tracking" if row.get("qc_profile") == "demo" else "Tracking gaps"
        )
        if gaps
        else "",
        "render_recipe_version": 1,
    }
    if row.get("qc_profile") == "demo":
        identity["qc_profile"] = "demo"
    if reuse_clips and all(path.is_file() for path in artifacts.values()):
        previous = json.loads(artifacts["validation"].read_text())
        if previous.get("render_identity") == identity and (
            previous.get("video_sha256") == _digest(artifacts["video"])
        ):
            inspect_video(artifacts["video"], expected_frames=count)
            print(f"Reused validated {stem}", flush=True)
            return previous
    snapshots = []
    for frame in bundle.iter_frames():
        updates, _ = assemble_index_frame(frame, bundle.bindings)
        snapshots.append((frame.frame_index, updates))
    updates_only = [updates for _, updates in snapshots]
    view = fit_view(updates_only, WORLD_PANEL[2:])
    trajectories = {}
    for role in ("helper", "leader"):
        origins = []
        for updates in updates_only:
            pose = next(
                (u.value for u in updates if u.path == f"world/{role}/device" and u.kind == "pose"),
                None,
            )
            origins.append(np.full(3, np.nan) if pose is None else pose.matrix[:3, 3])
        trajectories[role] = np.asarray(origins)
    video_iterators = {
        role: iter_indexed_rgb(
            dataset_root
            / "recordings"
            / bundle.bindings.recording_id
            / "mp4s"
            / f"{role}_trimmed_sync.mp4",
            bundle.frame_maps[role].mp4_pts,
            bundle.frame_maps[role].mp4_time_base,
            start_index=bundle.start_index,
            end_index=bundle.end_index,
        )
        for role in ("helper", "leader")
    }
    frames = iter_composed_frames(
        snapshots,
        video_iterators,
        view=view,
        trajectories=trajectories,
        annotation_start=int(bundle.selection["start_frame"]),
        annotation_end=int(bundle.selection["end_frame"]),
        annotation_id=annotation_id,
        object_category=category,
        qc_label=identity["qc_label"],
    )
    print(f"Rendering {stem}: {count} frames ({count / FPS:.3f}s)", flush=True)
    try:
        encode_video(
            artifacts["video"], frames, size=(WIDTH, HEIGHT), fps=FPS, expected_frames=count
        )
    finally:
        frames.close()
        for iterator in video_iterators.values():
            iterator.close()
    midpoint = (int(bundle.selection["start_frame"]) + int(bundle.selection["end_frame"])) // 2
    verified = inspect_video(
        artifacts["video"],
        expected_frames=count,
        screenshot_paths={
            0: artifacts["first"],
            midpoint - bundle.start_index: artifacts["middle"],
            count - 1: artifacts["last"],
        },
    )
    with Image.open(artifacts["middle"]) as middle:
        make_thumbnail(middle.convert("RGB")).save(artifacts["thumbnail"])
    result = {
        "render_identity": identity,
        "video_sha256": _digest(artifacts["video"]),
        "video": verified,
        "view_qc": evaluate_view(updates_only, view),
        "fixed_view": {
            "basis": view.basis.tolist(),
            "world_center_m": view.world_center.tolist(),
            "plane_center_m": view.plane_center.tolist(),
            "pixels_per_meter": view.pixels_per_meter,
        },
        "scene_included": False,
        "object_pose_included": False,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    _write_json(artifacts["validation"], result)
    print(f"Validated {stem}: {verified['file_size_bytes']:,} bytes", flush=True)
    return result


def _segments(rows: Sequence[Mapping], *, best: bool) -> list[dict]:
    segments = []
    for row in rows:
        # Preserve the long carrot-peel interval in best-of; montage excerpts
        # intentionally show five seconds from each example.
        requested = min(240, max(225, row["end_frame"] - row["start_frame"] + 1)) if best else 150
        start, end = choose_segment(
            row["clip_start_frame"],
            row["clip_end_frame"],
            row["start_frame"],
            row["end_frame"],
            requested,
        )
        segments.append(
            {
                "path": row["video_path"],
                "start_ordinal": start - row["clip_start_frame"],
                "end_ordinal": end - row["clip_start_frame"],
                "annotation_id": row["annotation_id"],
                "object_category": row["object_category"],
            }
        )
    return segments


def _demo_segments(rows: Sequence[Mapping]) -> list[dict]:
    """Use four presentation seconds per passing demo, retaining source clip order."""
    result = []
    for row in rows:
        start, end = choose_segment(
            row["clip_start_frame"],
            row["clip_end_frame"],
            row["start_frame"],
            row["end_frame"],
            4 * FPS,
        )
        result.append(
            {
                "path": row["video_path"],
                "start_ordinal": start - row["clip_start_frame"],
                "end_ordinal": end - row["clip_start_frame"],
                "annotation_id": row["annotation_id"],
                "object_category": row["object_category"],
            }
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default=RECORDING_ID)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/comind")
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed/comind")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--qc-profile",
        choices=("strict", "demo"),
        default="strict",
        help="Separate demo profile permits disclosed partial-hand tracking; strict is unchanged",
    )
    parser.add_argument(
        "--reuse-clips", action="store_true", help="Reuse clips only after identity/hash validation"
    )
    parser.add_argument(
        "--qc-passing-only",
        action="store_true",
        help="Gate each annotation independently; write exclusions and export only QC-passing clips",
    )
    parser.add_argument(
        "--skip-compilations",
        action="store_true",
        help="Export individual clips and reports without montage or best-of compilations",
    )
    args = parser.parse_args(argv)
    recording_id = str(UUID(args.recording_id))
    default_output = ROOT / "outputs/v0_demo"
    if args.qc_profile == "demo":
        default_output /= recording_id
    output = _derived(args.output_dir or default_output, args.dataset_root)
    examples = _derived(
        output / ("examples_demo_qc" if args.qc_profile == "demo" else "examples"),
        args.dataset_root,
    )
    examples.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "outputs/comind_frame_validation" / f"{recording_id}.json"
    projection_path = ROOT / "outputs/comind_v0" / f"{recording_id}_projection_qc.json"
    source_path = ROOT / "outputs/comind_semantics" / f"{recording_id}_video_annotations.json"
    if args.qc_profile == "demo":
        return _demo_qc_export(
            args,
            recording_id=recording_id,
            output=output,
            examples=examples,
            report_path=report_path,
            projection_path=projection_path,
            source_path=source_path,
        )
    if args.qc_passing_only:
        return _qc_passing_export(
            args,
            recording_id=recording_id,
            output=output,
            examples=examples,
            report_path=report_path,
            projection_path=projection_path,
            source_path=source_path,
        )
    projection = json.loads(projection_path.read_text())
    if projection["validation_report_sha256"] != _digest(report_path):
        raise ValueError("projection QC does not refer to the current temporal validation report")
    base = load_playback(
        _derived(args.processed_root, args.dataset_root), report_path, recording_id
    )
    bundles = [
        base.with_annotation(str(item["annotation_id"]), context_frames=120)
        for item in sorted(base.annotations, key=lambda item: item["start_frame"])
    ]
    rows = [annotation_metrics(bundle, projection) for bundle in bundles]
    bind_numeric_ids(
        rows,
        load_annotation_source_report(
            source_path, dataset_root=args.dataset_root, recording_id=recording_id
        ),
        recording_id,
    )
    ranked = rank_examples(rows)
    rank_lookup = {row["annotation_id"]: row for row in ranked}
    rows = [rank_lookup[str(bundle.selection["annotation_id"])] for bundle in bundles]
    input_identity = {
        "validation_report_sha256": _digest(report_path),
        "renderer_sha256": _digest(ROOT / "src/duet/visualization/demo_geometry.py"),
        "compositor_sha256": _digest(ROOT / "src/duet/visualization/demo_compositor.py"),
        "timeline": "comind_sync_frame_index",
        "fps": FPS,
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _render_example,
                bundle,
                row,
                dataset_root=args.dataset_root,
                output_dir=examples,
                input_identity=input_identity,
                reuse_clips=args.reuse_clips,
            )
            for bundle, row in zip(bundles, rows, strict=True)
        ]
        rendered = [future.result() for future in futures]
    for row, artifact in zip(rows, rendered, strict=True):
        row.update(
            video_path=artifact["artifacts"]["video"],
            thumbnail_path=artifact["artifacts"]["thumbnail"],
            file_size_bytes=artifact["video"]["file_size_bytes"],
            frame_count=artifact["video"]["frame_count"],
        )
    manifest = _derived(output / "examples_manifest.csv", args.dataset_root)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = rank_examples(rows)
    best_rows = [
        row for row in ranked if row["public_recommendation"] != "exclude_from_public_demo"
    ][:3]
    if not args.skip_compilations and len(best_rows) != 3:
        raise ValueError("fewer than three examples pass the explicit event tracking gate")
    aggregate_reports = {}
    for name, selected, best in (
        ("duet_v0_montage", rows, False),
        ("duet_v0_best_examples", best_rows, True),
    ):
        if args.skip_compilations:
            continue
        path = _derived(output / f"{name}.mp4", args.dataset_root)
        print(f"Rendering {name}: {[row['annotation_id'] for row in selected]}", flush=True)
        aggregate_reports[name] = export_compilation(path, _segments(selected, best=best), fps=FPS)
    summary = {
        "recording_id": recording_id,
        "context_frames_each_side": 120,
        "ranking_policy": [
            "verified mappings",
            "pose coverage",
            "minimum of event/context all-four-hand coverage",
            "event all-four-hand coverage",
            "measured wrist projection coverage",
            "median accepted hand confidence",
            "maximum accepted local timestamp residual",
        ],
        "projection_interpretation": "Calibrated wrist image-domain coverage; not occlusion-aware visibility",
        "distance_interpretation": "Minimum simultaneous cross-person landmark distance in meters; not object contact",
        "numeric_id_interpretation": "Original source numeric_id; annotation_id is the distinct zero-padded source segment key",
        "rows": rows,
        "best_three_annotation_ids": [row["annotation_id"] for row in best_rows],
        "compilations": aggregate_reports,
        "manifest_path": str(manifest),
        "scene_included": False,
        "vrs_matching_performed": False,
    }
    _write_json(_derived(output / "examples_validation.json", args.dataset_root), summary)
    print(
        json.dumps(
            {"manifest": str(manifest), "best_three": summary["best_three_annotation_ids"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
