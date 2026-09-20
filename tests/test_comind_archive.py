"""CRC-checked recovery uses only tiny synthetic, intentionally truncated ZIPs."""

import io
import struct
import zipfile
import zlib

import pytest

from duet.adapters.comind.archive import recover_complete_zip_member

MEMBER = "closed_loop_trajectory.csv"
PAYLOAD = b"graph_uid,tracking_timestamp_us\nsynthetic,1000\n"


def _truncated_archive(tmp_path, *, compression=zipfile.ZIP_DEFLATED):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.txt", b"synthetic fixture", compress_type=compression)
        archive.writestr(MEMBER, PAYLOAD, compress_type=compression)
        archive.writestr("later_member.bin", b"later-data" * 20, compress_type=zipfile.ZIP_STORED)
        target_info = archive.getinfo(MEMBER)
        later_info = archive.getinfo("later_member.bin")
    # Keep the complete target member and only half of the later member.
    cut = later_info.header_offset + 30 + len(later_info.filename) + later_info.compress_size // 2
    raw_root = tmp_path / "synthetic_raw"
    raw_root.mkdir()
    path = raw_root / "slam.zip"
    path.write_bytes(buffer.getvalue()[:cut])
    return path, raw_root, target_info


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_complete_member_recovered_from_truncated_container_without_source_mutation(
    tmp_path, compression
):
    archive, raw_root, info = _truncated_archive(tmp_path, compression=compression)
    original = archive.read_bytes()
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(archive)
    result = recover_complete_zip_member(
        archive, MEMBER, output_directory=tmp_path / "processed", raw_root=raw_root
    )
    assert result.path.read_bytes() == PAYLOAD
    assert result.compressed_bytes == info.compress_size
    assert result.uncompressed_bytes == len(PAYLOAD)
    assert result.crc32 == zlib.crc32(PAYLOAD)
    assert "CRC32=" in result.provenance.detail
    assert "incomplete/unverified" in result.provenance.detail
    assert archive.read_bytes() == original
    assert sorted(path.name for path in raw_root.iterdir()) == ["slam.zip"]


def test_corrupted_crc_does_not_publish_or_replace_output(tmp_path):
    archive, raw_root, info = _truncated_archive(tmp_path)
    damaged = bytearray(archive.read_bytes())
    damaged[info.header_offset + 14] ^= 1
    archive.write_bytes(damaged)
    output = tmp_path / "processed"
    output.mkdir()
    prior = output / MEMBER
    prior.write_bytes(b"existing processed result")
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="length/CRC"):
        recover_complete_zip_member(archive, MEMBER, output_directory=output, raw_root=raw_root)
    assert prior.read_bytes() == b"existing processed result"
    assert list(output.iterdir()) == [prior]
    assert archive.read_bytes() == original


def test_incomplete_member_is_not_recovered(tmp_path):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match="truncated"):
        recover_complete_zip_member(
            archive, "later_member.bin", output_directory=output, raw_root=raw_root
        )
    assert not output.exists()


def test_recovery_rejects_raw_output_directory(tmp_path):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    with pytest.raises(ValueError, match="outside immutable raw"):
        recover_complete_zip_member(
            archive, MEMBER, output_directory=raw_root / "extract", raw_root=raw_root
        )
    assert not (raw_root / "extract").exists()


def test_recovery_rejects_symlink_into_raw_output(tmp_path):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    alias = tmp_path / "processed_alias"
    alias.symlink_to(raw_root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside immutable raw"):
        recover_complete_zip_member(archive, MEMBER, output_directory=alias, raw_root=raw_root)
    assert not (raw_root / MEMBER).exists()


def test_raw_directory_rejected_even_when_target_symlink_points_outside(tmp_path):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"existing processed file")
    (raw_root / MEMBER).symlink_to(outside)
    with pytest.raises(ValueError, match="outside immutable raw"):
        recover_complete_zip_member(archive, MEMBER, output_directory=raw_root, raw_root=raw_root)
    assert outside.read_bytes() == b"existing processed file"
    assert (raw_root / MEMBER).is_symlink()
    assert sorted(path.name for path in raw_root.iterdir()) == [MEMBER, "slam.zip"]


@pytest.mark.parametrize("member", ["../trajectory.csv", "nested/trajectory.csv", "x\\y", "", "."])
def test_recovery_rejects_nonflat_member_names(tmp_path, member):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    with pytest.raises(ValueError, match="flat member"):
        recover_complete_zip_member(
            archive, member, output_directory=tmp_path / "processed", raw_root=raw_root
        )


def test_recovery_enforces_explicit_size_limit(tmp_path):
    archive, raw_root, _ = _truncated_archive(tmp_path)
    with pytest.raises(ValueError, match="size limit"):
        recover_complete_zip_member(
            archive,
            MEMBER,
            output_directory=tmp_path / "processed",
            raw_root=raw_root,
            max_uncompressed_bytes=len(PAYLOAD) - 1,
        )


def test_recovery_rejects_data_descriptor_flag(tmp_path):
    archive, raw_root, info = _truncated_archive(tmp_path)
    damaged = bytearray(archive.read_bytes())
    struct.pack_into("<H", damaged, info.header_offset + 6, 8)
    archive.write_bytes(damaged)
    with pytest.raises(ValueError, match="data descriptor"):
        recover_complete_zip_member(
            archive, MEMBER, output_directory=tmp_path / "processed", raw_root=raw_root
        )
