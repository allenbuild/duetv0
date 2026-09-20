#!/usr/bin/env python3
"""Audit a real CoMind recording without launching a viewer or modifying raw data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from duet.adapters.comind.validation import (
    ensure_output_outside_raw,
    raw_inventory,
    run_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--recording-id", default="43276420-701f-4731-b9ab-bebc7fd14994")
    parser.add_argument("--max-hand-gap-seconds", type=float, default=0.01)
    parser.add_argument("--recover-incomplete-zip", action="store_true")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument("--output", type=Path, help="Optional JSON report path outside raw data")
    args = parser.parse_args()
    if args.output is not None:
        ensure_output_outside_raw(args.output, args.root)
    ensure_output_outside_raw(args.processed_root, args.root)
    before = raw_inventory(args.root)
    try:
        report = run_validation(
            args.root,
            args.recording_id,
            max_hand_gap_seconds=args.max_hand_gap_seconds,
            recover_incomplete_zip=args.recover_incomplete_zip,
            processed_root=args.processed_root,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    finally:
        after = raw_inventory(args.root)
        if before != after:
            changed = sorted(
                key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
            )
            raise RuntimeError(f"raw dataset inventory changed during validation: {changed}")
    report["raw_immutability"] = {
        "unchanged": True,
        "file_count": len(before),
        "verification": "file names, byte sizes and mtime_ns unchanged; payloads were not hashed",
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
