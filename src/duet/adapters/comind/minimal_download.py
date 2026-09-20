"""Exact CoMind assets and immutable, verified publication of new raw files.

Native RGB calibration/timestamps are obtained separately by bounded VRS range
reads. Complete shared-world trajectories carry the graph IDs needed for graph
verification; MPS SLAM, online calibration, ZIPs and scans are not inputs here.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROLES = ("helper", "leader")
HANDOVER_ANNOTATION = "dataset_handover_consolidated.json"
MAPPING_PATH = "multislam_output/vrs_to_multi_slam.json"
TRAJECTORY_RE = re.compile(r"multislam_output/([0-9]+)/slam/closed_loop_trajectory\.csv")
CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ManifestFile:
    path: str
    size: int
    hash: str


@dataclass(frozen=True)
class Manifest:
    recording_id: str
    hash_algorithm: str
    files: tuple[ManifestFile, ...]
    source_sha256: str

    @property
    def total_size_bytes(self) -> int:
        return sum(entry.size for entry in self.files)


@dataclass(frozen=True)
class SelectedFile:
    """One exact server asset, preserving the official target directory layout."""

    url_id: str
    path: str
    destination: str
    size: int
    hash: str
    hash_algorithm: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_relative_path(value: str) -> str:
    """Reject ambiguous paths before joining a manifest entry to the raw root."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or PurePosixPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"unsafe manifest path: {value!r}")
    return value


def parse_manifest(
    raw: bytes,
    *,
    recording_id: str,
    is_excluded: Callable[[str], bool],
) -> Manifest:
    """Validate official healthcheck fields and apply its non-trimmed VRS filter.

    The CLI reuses the official downloader's fetcher and exclusion predicate. This
    offline entry point applies the same path/size/hash field semantics with stricter
    identity, duplicate, integer-size and required-digest checks before planning.
    """
    data = json.loads(raw)
    if data.get("recording_id") != recording_id:
        raise ValueError("manifest recording_id does not match the requested recording")
    algorithm = data.get("hash_algorithm", "sha256")
    try:
        digest_length = hashlib.new(algorithm).digest_size * 2
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported manifest hash algorithm") from exc
    if digest_length <= 0:
        raise ValueError("variable-length manifest hash algorithms are unsupported")
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest contains no file list")
    seen = set()
    files = []
    for entry in entries:
        path = validate_relative_path(entry["path"])
        size = entry["size"]
        digest = entry.get("hash")
        if path in seen:
            raise ValueError(f"duplicate manifest path: {path}")
        seen.add(path)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid byte size for {path}")
        if not isinstance(digest, str) or not re.fullmatch(
            rf"[a-fA-F0-9]{{{digest_length}}}", digest
        ):
            raise ValueError(f"missing or invalid {algorithm} hash for {path}")
        if not is_excluded(path):
            files.append(ManifestFile(path, size, digest.lower()))
    if not files:
        raise ValueError("manifest has no downloadable files after exclusions")
    return Manifest(recording_id, algorithm, tuple(files), hashlib.sha256(raw).hexdigest())


def select_recording_files(
    manifest: Manifest, *, verified_mapping: Mapping[str, str] | None = None
) -> tuple[SelectedFile, ...]:
    """Select individual inputs without inferring role identity from output IDs.

    With exactly two trajectory directories both are selected. Downstream processing
    must parse vrs_to_multi_slam.json and verify all trajectory graph IDs. More than
    two directories require a mapping already verified against this manifest.
    """
    entries = {entry.path: entry for entry in manifest.files}
    identifiers = {
        match[1] for path in entries if (match := TRAJECTORY_RE.fullmatch(path)) is not None
    }
    if verified_mapping is not None:
        if set(verified_mapping) != set(ROLES):
            raise ValueError("verified mapping must name helper and leader")
        selected_ids = set(verified_mapping.values())
        if len(selected_ids) != 2 or not selected_ids <= identifiers:
            raise ValueError("verified mapping does not identify two available trajectories")
    elif len(identifiers) == 2:
        selected_ids = identifiers
    else:
        raise ValueError(
            "expected exactly two trajectory directories; provide a manifest-verified "
            "local vrs_to_multi_slam.json rather than guessing participant roles"
        )
    requests = [(f"mp4s/{role}_trimmed_sync.mp4", f"{role} synchronized video") for role in ROLES]
    requests.extend(
        (
            f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
            f"{role} device-frame hand annotations",
        )
        for role in ROLES
    )
    requests.extend(
        (
            f"multislam_output/{identifier}/slam/closed_loop_trajectory.csv",
            "shared-world trajectory and complete graph_uid verification",
        )
        for identifier in sorted(selected_ids, key=int)
    )
    requests.append((MAPPING_PATH, "verified VRS role to Multi-SLAM directory mapping"))
    selected = []
    for path, reason in requests:
        if path not in entries:
            raise ValueError(f"required individual file is absent from manifest: {path}")
        entry = entries[path]
        selected.append(
            SelectedFile(
                manifest.recording_id,
                path,
                f"recordings/{manifest.recording_id}/{path}",
                entry.size,
                entry.hash,
                manifest.hash_algorithm,
                reason,
            )
        )
    return tuple(selected)


def select_handover_annotation(manifest: Manifest) -> SelectedFile:
    """Select the handover JSON only, never the complete annotation part."""
    if manifest.recording_id != "annotations":
        raise ValueError("handover annotations require the annotations manifest")
    entry = next((entry for entry in manifest.files if entry.path == HANDOVER_ANNOTATION), None)
    if entry is None:
        raise ValueError("handover JSON is absent from annotations manifest")
    return SelectedFile(
        "annotations",
        entry.path,
        f"annotations/{entry.path}",
        entry.size,
        entry.hash,
        manifest.hash_algorithm,
        "handover intervals, reused across recordings",
    )


def safe_destination(target: Path, relative_path: str) -> Path:
    """Reject every existing symlink in the destination chain, including the root."""
    validate_relative_path(relative_path)
    root = Path(os.path.abspath(target))
    destination = root / relative_path
    for component in (*reversed(destination.parents), destination):
        if component.is_symlink():
            raise ValueError(f"symlink in immutable raw destination: {component}")
    return destination


def verify_existing(path: Path, entry: SelectedFile) -> bool:
    """Return false only when absent; existing mismatches always fail closed."""
    if path.is_symlink():
        raise ValueError(f"refusing existing symlink: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return False
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != entry.size:
            raise ValueError(f"existing raw file has wrong type/size; will not overwrite: {path}")
        digest = hashlib.file_digest(stream, entry.hash_algorithm).hexdigest()
    if digest != entry.hash:
        raise ValueError(f"existing raw file hash mismatch; will not overwrite: {path}")
    return True


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename using the OS no-clobber flag, including concurrent writers."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        arguments = (os.fsencode(source), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise RuntimeError("atomic no-overwrite rename requires macOS or Linux renameat2")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"destination appeared during transfer: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("asset download redirects are disabled")


def download_selected_file(
    entry: SelectedFile,
    *,
    target: Path,
    url: str,
    timeout: float,
    opener: Any = None,
) -> str:
    """Stream one selected asset, verify size/hash, then atomically publish once.

    There is deliberately no resume or replacement of an existing .part file. Failed
    new partial files are removed; existing raw data is never opened for writing.
    """
    destination = safe_destination(target, entry.destination)
    if verify_existing(destination, entry):
        return "verified_existing"
    partial = safe_destination(target, entry.destination + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_destination(target, entry.destination)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "duet-comind-minimal/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            open_request = opener or urllib.request.build_opener(_NoRedirect()).open
            with open_request(request, timeout=timeout) as response:
                if response.status != 200:
                    raise ValueError(
                        f"expected HTTP 200 for a selected asset, got {response.status}"
                    )
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("encoded asset response is not byte-verifiable")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and int(declared_length) != entry.size:
                    raise ValueError("HTTP Content-Length differs from official manifest size")
                digest = hashlib.new(entry.hash_algorithm)
                transferred = 0
                while True:
                    block = response.read(min(CHUNK_BYTES, entry.size - transferred + 1))
                    if not block:
                        break
                    transferred += len(block)
                    if transferred > entry.size:
                        raise ValueError("asset response exceeded official manifest size")
                    output.write(block)
                    digest.update(block)
                if transferred != entry.size:
                    raise ValueError("asset response truncated before official manifest size")
                if digest.hexdigest() != entry.hash:
                    raise ValueError("asset hash differs from official manifest hash")
            output.flush()
            os.fsync(output.fileno())
        safe_destination(target, entry.destination)
        _rename_no_replace(partial, destination)
        return "downloaded_verified"
    finally:
        # Only this invocation's exclusively-created partial can reach this block.
        partial.unlink(missing_ok=True)
