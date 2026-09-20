"""Explicit recovery of complete ZIP members from a truncated download.

This does not repair or certify the archive. It supports only complete local
headers with known sizes, no encryption/data descriptor/ZIP64, and stored/deflated
members. Recovered bytes are published outside raw only after size and CRC checks.
"""

import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from duet.schemas.common import Provenance


@dataclass(frozen=True)
class RecoveredMember:
    path: Path
    archive: Path
    member: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    provenance: Provenance


def recover_complete_zip_member(
    archive: Path,
    member: str,
    *,
    output_directory: Path,
    raw_root: Path,
    max_uncompressed_bytes: int = 512 * 1024 * 1024,
) -> RecoveredMember:
    """Explicitly recover one flat member, never modifying the raw source.

    Call only after an integrity failure has been reported to the caller. Earlier
    complete members can be recovered even if a later member or directory is cut
    off. Arbitrary corrupt data is not searched for plausible ZIP signatures.
    """
    archive, output_directory, raw_root = Path(archive), Path(output_directory), Path(raw_root)
    if Path(member).name != member or member in ("", ".", "..") or "\\" in member:
        raise ValueError("recovery accepts only an explicit flat member name")
    destination = (output_directory / member).resolve()
    resolved_raw = raw_root.resolve()
    if (
        output_directory.resolve().is_relative_to(resolved_raw)
        or destination.is_relative_to(resolved_raw)
        or destination == archive.resolve()
    ):
        raise ValueError("recovery output must be outside immutable raw data")
    if max_uncompressed_bytes <= 0:
        raise ValueError("recovery size limit must be positive")
    source_size = archive.stat().st_size
    with archive.open("rb") as source:
        while True:
            header = source.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError(f"complete local member {member!r} not found")
            fields = struct.unpack("<IHHHHHIIIHH", header)
            _, _, flags, method, _, _, crc, compressed, uncompressed, name_size, extra_size = fields
            if flags & ~0x800 or compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
                raise ValueError("unsupported ZIP flags, data descriptor, encryption, or ZIP64")
            name_bytes = source.read(name_size)
            extra = source.read(extra_size)
            if len(name_bytes) != name_size or len(extra) != extra_size:
                raise ValueError("truncated local ZIP header")
            name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
            if source.tell() + compressed > source_size:
                raise ValueError(f"ZIP member {name!r} is truncated; cannot recover {member!r}")
            if name != member:
                source.seek(compressed, 1)
                continue
            if method not in (0, 8) or uncompressed > max_uncompressed_bytes:
                raise ValueError("unsupported compression or member exceeds recovery size limit")
            output_directory.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=output_directory, delete=False) as target:
                    temporary = Path(target.name)
                    inflater = zlib.decompressobj(-15) if method == 8 else None
                    remaining, size, actual_crc = compressed, 0, 0
                    while remaining:
                        block = source.read(min(65536, remaining))
                        if not block:
                            raise ValueError("ZIP member ended during recovery")
                        remaining -= len(block)
                        data = (
                            inflater.decompress(block, max_uncompressed_bytes - size + 1)
                            if inflater
                            else block
                        )
                        size += len(data)
                        if size > uncompressed or size > max_uncompressed_bytes:
                            raise ValueError("recovered member exceeds declared size")
                        actual_crc = zlib.crc32(data, actual_crc)
                        target.write(data)
                    if inflater is not None and (
                        not inflater.eof or inflater.unused_data or inflater.unconsumed_tail
                    ):
                        raise ValueError("deflate stream did not end exactly at declared boundary")
                    if size != uncompressed or actual_crc != crc:
                        raise ValueError("recovered ZIP member failed length/CRC verification")
                os.replace(temporary, destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return RecoveredMember(
                destination,
                archive,
                member,
                compressed,
                uncompressed,
                crc,
                Provenance(
                    f"{archive}!/{member}",
                    f"Explicit local-header recovery; {uncompressed} bytes; CRC32={crc:08x}; "
                    "whole archive remains incomplete/unverified",
                ),
            )
