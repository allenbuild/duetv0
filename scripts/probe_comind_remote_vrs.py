#!/usr/bin/env python3
"""One explicit bounded HTTP HEAD/range probe, with a persistent 20 MB budget.

No redirects, retries, full GET fallback, decompression, or VRS reader invocation.
Bodies are read only after 206 and the exact requested Content-Range are verified.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import http.client
import json
import re
import time
from pathlib import Path

from duet.adapters.comind.validation import ensure_output_outside_raw

RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"
SIZES = {"helper": 12_572_405_231, "leader": 11_901_082_187}
LIMIT = 20_000_000
# Conservatively reserve this per request for possible transport buffering.
TRANSPORT_ALLOWANCE = 65_536
OUTPUT_ROOT = Path("outputs/comind_sync_forensics/remote_vrs")
REPORT_PATH = Path(f"outputs/comind_sync_forensics/{RECORDING}_remote_vrs.json")


def probe(role: str, *, offset: int | None, length: int | None, label: str) -> dict:
    """Validate and exclusively lock the cumulative budget before any HTTP."""
    if role not in SIZES or not re.fullmatch(r"[a-zA-Z0-9_-]+", label):
        raise ValueError("invalid role or artifact label")
    if (offset is None) != (length is None):
        raise ValueError("offset and length must both be supplied, or neither for HEAD")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (offset, length)
        if value is not None
    ):
        raise TypeError("byte offset and length must be integers")
    if offset is not None and (
        offset < 0 or length is None or length <= 0 or offset + length > SIZES[role]
    ):
        raise ValueError("invalid bounded byte range")
    for destination in (OUTPUT_ROOT, REPORT_PATH):
        ensure_output_outside_raw(destination, Path("data/raw/comind"))
    lock_path = REPORT_PATH.with_suffix(".lock")
    ensure_output_outside_raw(lock_path, Path("data/raw/comind"))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        # Reject a concurrent run rather than risk two processes spending one budget.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _probe_locked(role, offset=offset, length=length, label=label)


def _probe_locked(role: str, *, offset: int | None, length: int | None, label: str) -> dict:
    """Reserve the entire request before opening a connection; caller owns the lock."""
    report = (
        json.loads(REPORT_PATH.read_text())
        if REPORT_PATH.exists()
        else {"recording_id": RECORDING, "hard_budget_bytes": LIMIT, "requests": []}
    )
    if any(
        isinstance(item.get("reserved_budget_bytes"), bool)
        or not isinstance(item.get("reserved_budget_bytes"), int)
        or item["reserved_budget_bytes"] < TRANSPORT_ALLOWANCE
        for item in report["requests"]
    ):
        raise ValueError("invalid persistent request budget")
    reserved = sum(item["reserved_budget_bytes"] for item in report["requests"])
    reservation = (length or 0) + TRANSPORT_ALLOWANCE
    if reserved + reservation > LIMIT:
        raise ValueError("request would exceed cumulative conservative 20 MB budget")
    destination = OUTPUT_ROOT / f"{role}_{label}_{offset}_{length}.bin"
    ensure_output_outside_raw(destination, Path("data/raw/comind"))
    if offset is not None and destination.exists():
        raise FileExistsError("probe output exists; refusing duplicate download")
    method = "HEAD" if offset is None else "GET"
    remote_path = f"/dataset/{RECORDING}/trimmed_vrs/{role}_trimmed.vrs"
    entry = {
        "role": role,
        "method": method,
        "url": f"https://comind.ethz.ch{remote_path}",
        "requested_range": None if offset is None else f"bytes={offset}-{offset + length - 1}",
        "reserved_budget_bytes": reservation,
        "body_bytes_read": 0,
        "status": "PENDING",
    }
    report["requests"].append(entry)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    headers = {
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": "Duet-range-probe/1",
    }
    if offset is not None:
        headers["Range"] = entry["requested_range"]
    connection = http.client.HTTPSConnection("comind.ethz.ch", timeout=15)
    started = time.monotonic()
    try:
        connection.request(method, remote_path, headers=headers)
        response = connection.getresponse()
        entry["http_status"] = response.status
        entry["response_headers"] = dict(response.getheaders())
        if method == "HEAD":
            entry["status"] = "HEAD_ONLY_NO_BODY"
            return entry
        expected = f"bytes {offset}-{offset + length - 1}/{SIZES[role]}"
        if (
            response.status != 206
            or response.getheader("Content-Range") != expected
            or response.getheader("Content-Encoding", "identity") != "identity"
            or response.getheader("Content-Length") != str(length)
        ):
            entry["status"] = "REJECTED_HEADERS_CLOSED_WITHOUT_BODY_READ"
            return entry
        chunks = []
        while entry["body_bytes_read"] < length:
            if time.monotonic() - started > 30:
                raise TimeoutError("total probe deadline exceeded")
            block = response.read(min(65_536, length - entry["body_bytes_read"]))
            if not block:
                raise ValueError("range response ended before its declared length")
            entry["body_bytes_read"] += len(block)
            chunks.append(block)
        payload = b"".join(chunks)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("probe output exists; refusing silent overwrite")
        destination.write_bytes(payload)
        entry["artifact"] = str(destination.resolve())
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["status"] = "EXACT_RANGE_SAVED"
        return entry
    except Exception as exc:
        entry["status"] = "ERROR"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        connection.close()
        entry["elapsed_seconds"] = time.monotonic() - started
        report["body_bytes_read_total"] = sum(
            item["body_bytes_read"] for item in report["requests"]
        )
        report["conservative_reserved_budget_bytes"] = sum(
            item["reserved_budget_bytes"] for item in report["requests"]
        )
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=SIZES, required=True)
    parser.add_argument("--offset", type=int)
    parser.add_argument("--length", type=int)
    parser.add_argument("--label", default="header")
    args = parser.parse_args()
    if (args.offset is None) != (args.length is None):
        parser.error("offset and length must both be supplied, or neither for HEAD")
    print(
        json.dumps(
            probe(args.role, offset=args.offset, length=args.length, label=args.label), indent=2
        )
    )


if __name__ == "__main__":
    main()
