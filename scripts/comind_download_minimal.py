#!/usr/bin/env python3
"""Plan seven individual CoMind demo inputs; transfer only with explicit --download.

--dry-run --manifest PATH uses no network. Without --manifest only the official
healthcheck JSON is fetched during planning. VRS range reads are a separate,
explicit processing command; this CLI never opens a VRS asset URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

from comind_download import (
    DEFAULT_BASE_URL,
    build_url,
    fetch_manifest,
    human_bytes,
    is_non_trimmed_vrs,
)

from duet.adapters.comind.minimal_download import (
    MAPPING_PATH,
    Manifest,
    SelectedFile,
    download_selected_file,
    parse_manifest,
    safe_destination,
    select_handover_annotation,
    select_recording_files,
    verify_existing,
)
from duet.adapters.comind.multislam import parse_vrs_mapping
from duet.adapters.comind.validation import ensure_output_outside_raw

ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path: Path | None, *, url_id: str, base_url: str, timeout: float) -> Manifest:
    """Reuse the official fetcher when online; validate the same manifest offline."""
    raw = (
        path.read_bytes() if path is not None else fetch_manifest(base_url, url_id, timeout)["raw"]
    )
    return parse_manifest(raw, recording_id=url_id, is_excluded=is_non_trimmed_vrs)


def verified_local_mapping(manifest: Manifest, target: Path) -> dict[str, str] | None:
    entry = next((entry for entry in manifest.files if entry.path == MAPPING_PATH), None)
    if entry is None:
        raise ValueError("official manifest lacks vrs_to_multi_slam.json")
    selected = SelectedFile(
        manifest.recording_id,
        entry.path,
        f"recordings/{manifest.recording_id}/{entry.path}",
        entry.size,
        entry.hash,
        manifest.hash_algorithm,
        "mapping verification",
    )
    path = safe_destination(target, selected.destination)
    if not verify_existing(path, selected):
        return None
    return parse_vrs_mapping(path, recording_id=manifest.recording_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_id", help="one exact CoMind recording UUID")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="plan only (the default)")
    mode.add_argument("--download", action="store_true", help="explicitly transfer selected files")
    parser.add_argument("--manifest", type=Path, help="saved recording healthcheck JSON; no fetch")
    parser.add_argument("--target", type=Path, default=ROOT / "data/raw/comind")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--annotations-manifest",
        type=Path,
        help="saved annotations healthcheck; defaults to existing target/annotations/healthcheck.json",
    )
    parser.add_argument(
        "--remote-vrs-estimate-bytes",
        type=int,
        help="planning estimate only; does not initiate any VRS requests",
    )
    parser.add_argument("--report", type=Path, help="optional derived JSON report, outside raw")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    recording_id = str(UUID(args.recording_id))
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.remote_vrs_estimate_bytes is not None and args.remote_vrs_estimate_bytes < 0:
        raise ValueError("remote estimate must be non-negative")
    if not args.base_url.startswith("https://"):
        raise ValueError("the CoMind server URL must use HTTPS")
    if args.report is not None:
        ensure_output_outside_raw(args.report, ROOT / "data/raw")
        ensure_output_outside_raw(args.report, args.target)
    manifest = read_manifest(
        args.manifest, url_id=recording_id, base_url=args.base_url, timeout=args.timeout
    )
    mapping = verified_local_mapping(manifest, args.target)
    recording_files = select_recording_files(manifest, verified_mapping=mapping)
    annotation_manifest_path = args.annotations_manifest or (
        args.target / "annotations/healthcheck.json"
    )
    if not annotation_manifest_path.is_file():
        raise ValueError(
            "a saved annotations healthcheck.json is required to verify/reuse the handover "
            "JSON; provide --annotations-manifest (only that JSON will be selected)"
        )
    annotations = read_manifest(
        annotation_manifest_path,
        url_id="annotations",
        base_url=args.base_url,
        timeout=args.timeout,
    )
    selected_files = (*recording_files, select_handover_annotation(annotations))
    planned = []
    for entry in selected_files:
        exists = verify_existing(safe_destination(args.target, entry.destination), entry)
        planned.append({**entry.as_dict(), "status": "verified_existing" if exists else "needed"})
    recording_bytes = sum(entry.size for entry in recording_files)
    remaining_bytes = sum(item["size"] for item in planned if item["status"] == "needed")
    remote_estimate = args.remote_vrs_estimate_bytes
    expected_total = None if remote_estimate is None else remaining_bytes + remote_estimate
    report = {
        "recording_id": recording_id,
        "mode": "download" if args.download else "dry_run",
        "manifest_sha256": manifest.source_sha256,
        "annotations_manifest_sha256": annotations.source_sha256,
        "official_all_recording_bytes": manifest.total_size_bytes,
        "minimal_recording_bytes": recording_bytes,
        "remaining_local_download_bytes": remaining_bytes,
        "remote_vrs_estimate_bytes": remote_estimate,
        "expected_total_transfer_bytes": expected_total,
        "minimum_effective_mbps_for_three_hours": (
            None if expected_total is None else expected_total * 8 / (3 * 3600 * 1_000_000)
        ),
        "mapping_status": "locally_verified" if mapping else "verify_after_download",
        "calibration": "native RGB VRS calibration from bounded remote configuration records",
        "graph_verification": "complete trajectory CSV graph_uid fields; no extra asset needed",
        "vrs_requests_performed": 0,
        "selected_files": planned,
    }
    print(f"{'DOWNLOAD' if args.download else 'DRY RUN'}: {recording_id}")
    print("Exact official-layout files (all byte sizes and hashes are from healthcheck.json):")
    for item in planned:
        print(f"  {item['destination']}\n    {item['size']:,} bytes; {item['status']}")
        print(f"    {item['hash_algorithm']}={item['hash']}")
    for label, value in (
        (
            "Old full recording (official exclusions applied; common assets excluded)",
            manifest.total_size_bytes,
        ),
        ("Minimal recording files", recording_bytes),
        ("New local download (verified existing files skipped)", remaining_bytes),
        ("Estimated remote VRS ranges (separate command; not fetched)", remote_estimate),
        ("Expected total transfer", expected_total),
    ):
        print(
            f"{label}: "
            + ("unknown" if value is None else f"{value:,} bytes ({human_bytes(value)})")
        )
    if expected_total is not None:
        print(f"Effective throughput for <3 hours: >{expected_total * 8 / 10_800_000_000:.2f} Mbps")
    print("Calibration: native RGB VRS configuration; no online_calibration.jsonl selected.")
    print("Graph metadata: complete trajectory graph_uid columns plus vrs_to_multi_slam.json.")
    print("Participant mapping and shared graph membership must pass processing verification.")
    if args.download:
        for entry in selected_files:
            status = download_selected_file(
                entry,
                target=args.target,
                url=build_url(args.base_url, entry.url_id, entry.path),
                timeout=args.timeout,
            )
            print(f"{status}: {entry.destination}")
    else:
        print("No dataset asset requests made. Actual transfer requires explicit --download.")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
