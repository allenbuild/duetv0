"""Entirely synthetic VRS/range tests. No network or original dataset is needed."""

import io
import json
import struct

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image

from duet.adapters.comind.remote_vrs import (
    DISK_INFO,
    RECORD_HEADER,
    HttpRangeReader,
    RemoteVrs,
    estimate_sparse_transfer,
    processed_directory,
)


def packed_string(value):
    value = value.encode()
    return struct.pack("<I", len(value)) + value


def packed_mapping(value):
    return struct.pack("<I", len(value)) + b"".join(
        packed_string(key) + packed_string(item) for key, item in value.items()
    )


def record(payload, kind, version, timestamp, record_type, *, compression=0, uncompressed=0):
    return (
        RECORD_HEADER.pack(
            32 + len(payload),
            0,
            kind,
            version,
            timestamp,
            1,
            record_type,
            compression,
            uncompressed,
        )
        + payload
    )


def fixture_bytes(*, malformed_jpeg=False, compressed_config=False):
    metadata = [
        {"name": "frame_number", "type": "DataPieceValue<uint64_t>", "offset": 0},
        {"name": "capture_timestamp_ns", "type": "DataPieceValue<int64_t>", "offset": 8},
        {"name": "image_metadata", "type": "DataPieceVector<uint8_t>", "index": 0},
    ]
    configuration = [
        {"name": "image_width", "type": "DataPieceValue<uint32_t>", "offset": 0},
        {"name": "image_height", "type": "DataPieceValue<uint32_t>", "offset": 4},
        {"name": "nominal_rate", "type": "DataPieceValue<double>", "offset": 8},
    ]
    tags = {
        "VRS_Original_Recordable_Name": "RGB Camera Class",
        "RF:Data:2": "data_layout+image/jpg",
        "DL:Data:2:0": json.dumps({"data_layout": metadata}),
        "DL:Configuration:2:0": json.dumps({"data_layout": configuration}),
    }
    desc = struct.pack("<IiH", 1, 214, 1) + packed_mapping({}) + packed_mapping(tags)
    desc += packed_mapping({})
    description = record(desc, 2, 2, 0, 0)
    config_payload = struct.pack("<IId", 16, 12, 30)
    configuration_bytes = record(
        pa.compress(config_payload, codec="zstd").to_pybytes()
        if compressed_config
        else config_payload,
        214,
        2,
        0,
        2,
        compression=2 if compressed_config else 0,
        uncompressed=len(config_payload) if compressed_config else 0,
    )
    out = io.BytesIO()
    Image.fromarray(np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)).save(out, "JPEG")
    jpeg = b"bad-jpeg" if malformed_jpeg else out.getvalue()
    times = [2**53 + 1, 2**53 + 51]  # Preserve ints beyond IEEE754 integer precision.
    records = [configuration_bytes] + [
        record(struct.pack("<QqII", 900 + i, timestamp, 0, 0) + jpeg, 214, 2, timestamp / 1e9, 3)
        for i, timestamp in enumerate(times)
    ]
    infos = np.array(
        [(0, len(records[0]), 2, 214, 1)]
        + [(timestamp / 1e9, len(records[i + 1]), 3, 214, 1) for i, timestamp in enumerate(times)],
        dtype=DISK_INFO,
    )
    prelude = struct.pack("<IiHI", 1, 214, 1, len(infos))
    payload = prelude + pa.compress(infos.tobytes(), codec="zstd").to_pybytes()
    index = record(payload, 1, 2, 0, 0, compression=2, uncompressed=len(prelude) + infos.nbytes)
    index_offset = 80 + len(description)
    first_user = index_offset + len(index)
    header = bytearray(80)
    header[:8], header[72:] = b"VisionRe", b"cordVRS2"
    struct.pack_into("<IIqqq", header, 16, 80, 32, index_offset, 80, first_user)
    return bytes(header) + description + index + b"".join(records), times


class MemoryReader:
    identity = "fixture://synthetic.vrs"

    def __init__(self, payload):
        self.payload, self.size, self.calls = payload, len(payload), []

    def read(self, offset, length):
        self.calls.append((offset, length))
        assert 0 <= offset < offset + length <= self.size
        return self.payload[offset : offset + length]


def test_rgb_index_and_exact_device_time_are_distinct():
    payload, times = fixture_bytes()
    reader = MemoryReader(payload)
    vrs = RemoteVrs.open(reader)
    assert vrs.rgb_count == 2
    assert vrs.image_size == (16, 12)
    assert not vrs.metadata_known.any()
    np.testing.assert_array_equal(vrs.record_time_seconds, np.array(times) / 1e9)
    np.testing.assert_array_equal(vrs.capture_metadata([1, 0]), times[::-1])
    assert all(length < len(payload) for _, length in reader.calls)
    before = len(reader.calls)
    vrs.capture_metadata([0, 1])
    assert len(reader.calls) == before


def test_sparse_image_decode_caches_only_fingerprint_and_metadata(tmp_path):
    payload, times = fixture_bytes()
    reader = MemoryReader(payload)
    vrs = RemoteVrs.open(reader)
    image = vrs.rgb_image(1)
    assert image.shape == (12, 16, 3)
    assert vrs.metadata_known.tolist() == [False, True]
    assert vrs.device_timestamps_ns.tolist() == [-1, times[1]]
    before = len(reader.calls)
    assert vrs.fingerprint(1).shape == (32, 32)
    assert len(reader.calls) == before
    cache = tmp_path / "data/processed/comind/fixture/vrs"
    vrs.save_metadata(cache, role="helper", recording_id="fixture")
    assert {p.suffix for p in cache.iterdir()} == {".json", ".npz"}
    restored = RemoteVrs.from_cache(reader, cache, role="helper")
    assert restored.image_size == (16, 12)
    assert restored.metadata_known.tolist() == [False, True]
    np.testing.assert_array_equal(restored.fingerprint(1), vrs.fingerprint(1))
    assert len(reader.calls) == before
    (cache / "helper_rgb_metadata.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        RemoteVrs.from_cache(reader, cache, role="helper")


def test_wrong_jpeg_boundaries_rejected():
    payload, _ = fixture_bytes(malformed_jpeg=True)
    vrs = RemoteVrs.open(MemoryReader(payload))
    with pytest.raises(ValueError, match="JPEG boundaries"):
        vrs.rgb_image(0)


def test_unsupported_metadata_and_mismatched_record_are_rejected():
    payload, _ = fixture_bytes()
    reader = MemoryReader(payload)
    vrs = RemoteVrs.open(reader)
    altered = bytearray(payload)
    struct.pack_into("<d", altered, int(vrs.record_offsets[0]) + 16, 7)
    reader.payload = bytes(altered)
    with pytest.raises(ValueError, match="index disagree"):
        vrs.capture_metadata([0])


def test_only_processed_tree_can_hold_remote_derived_cache(tmp_path):
    with pytest.raises(ValueError, match="data/processed"):
        processed_directory(tmp_path / "outputs/vrs")
    with pytest.raises(ValueError, match="data/raw"):
        processed_directory(tmp_path / "data/processed/x/data/raw/vrs")


class Response:
    def __init__(self, *, status=206, payload=b"test", etag='"fixed"'):
        self.status, self.payload, self.read_calls = status, payload, 0
        self.headers = {"Content-Range": "bytes 0-3/1000", "Content-Length": "4", "ETag": etag}

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read(self, size):
        self.read_calls += 1
        block, self.payload = self.payload[:size], self.payload[size:]
        return block


def http_fixture(tmp_path, monkeypatch, response, *, budget=1_000_000):
    calls = []

    class Connection:
        def __init__(self, host, **kwargs):
            calls.append(("connect", host, kwargs))

        def request(self, *args, **kwargs):
            calls.append(("request", args, kwargs))

        def getresponse(self):
            return response

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr("duet.adapters.comind.remote_vrs.http.client.HTTPSConnection", Connection)
    reader = HttpRangeReader(
        "https://example.invalid/test.vrs",
        size=1000,
        budget_bytes=budget,
        ledger=tmp_path / "data/processed/ledger.json",
    )
    return reader, calls


@pytest.mark.parametrize(
    "failure",
    [
        "http200",
        "redirect",
        "etag",
        "weak_etag",
        "range",
        "length",
        "encoding",
        "transfer_encoding",
    ],
)
def test_http_rejections_do_not_read_body(tmp_path, monkeypatch, failure):
    response = Response()
    if failure == "http200":
        response.status = 200
    elif failure == "redirect":
        response.status = 302
    else:
        key, value = {
            "etag": ("ETag", None),
            "weak_etag": ("ETag", 'W/"weak"'),
            "range": ("Content-Range", "bytes 1-4/1000"),
            "length": ("Content-Length", "1000"),
            "encoding": ("Content-Encoding", "gzip"),
            "transfer_encoding": ("Transfer-Encoding", "chunked"),
        }[failure]
        response.headers[key] = value
    reader, calls = http_fixture(tmp_path, monkeypatch, response)
    with reader, pytest.raises(ValueError, match="without reading body"):
        reader.read(0, 4)
    assert response.read_calls == 0
    assert sum(call[0] == "request" for call in calls) == 1


def test_budget_is_reserved_before_any_connection_and_full_get_is_forbidden(tmp_path, monkeypatch):
    reader, calls = http_fixture(tmp_path, monkeypatch, Response(), budget=65_539)
    with reader:
        with pytest.raises(ValueError, match="full-file"):
            reader.read(0, 1000)
        with pytest.raises(ValueError, match="budget exceeded"):
            reader.read(0, 4)
    assert not calls


def test_etag_pinned_request_and_persistent_cumulative_budget(tmp_path, monkeypatch):
    response = Response()
    reader, calls = http_fixture(tmp_path, monkeypatch, response)
    with reader:
        assert reader.read(0, 4) == b"test"
        response.payload, response.headers["ETag"] = b"test", '"changed"'
        with pytest.raises(ValueError, match="ETag"):
            reader.read(0, 4)
    requests = [call for call in calls if call[0] == "request"]
    assert requests[1][2]["headers"]["If-Match"] == '"fixed"'
    ledger = json.loads(reader.ledger.read_text())
    assert ledger["reserved_bytes"] == 2 * (4 + 65_536)
    assert ledger["body_bytes_read"] == 4
    assert response.read_calls == 1


def test_truncated_body_preserves_spent_budget_and_never_saves_payload(tmp_path, monkeypatch):
    reader, _ = http_fixture(tmp_path, monkeypatch, Response(payload=b"te"))
    with reader, pytest.raises(ValueError, match="truncated"):
        reader.read(0, 4)
    ledger = json.loads(reader.ledger.read_text())
    assert ledger["body_bytes_read"] == 2
    assert ledger["reserved_bytes"] == 65_540
    assert ledger["requests"][0]["status"] == "FAILED"
    assert not list(reader.ledger.parent.glob("*.vrs"))


def test_zstd_rgb_configuration_observed_in_real_comind():
    payload, _ = fixture_bytes(compressed_config=True)
    vrs = RemoteVrs.open(MemoryReader(payload))
    assert vrs.image_size == (16, 12)
    assert vrs.rgb_image(0).shape == (12, 16, 3)


def test_offline_sparse_estimate_unions_overlapping_candidates(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline estimator must never make HTTP requests")

    monkeypatch.setattr("duet.adapters.comind.remote_vrs.http.client.HTTPSConnection", forbidden)
    reference = tmp_path / "prior_index.npz"
    np.savez(reference, record_sizes=np.full(1000, 400_000))
    result = estimate_sparse_transfer(
        target_size_bytes=800_000_000,
        reference_size_bytes=400_000_000,
        reference_rgb_index=reference,
        reference_index_bytes=1_000_000,
        handover_bounds=[(100, 110), (105, 120)],
        context_frames=10,
    )
    assert result["estimated_rgb_frame_count"] == 2000
    assert result["estimated_unique_rgb_records"] < 2000
    assert result["estimated_index_description_config_bytes"] == 2_000_000
    assert result["estimated_conservative_reserved_bytes"] > result["estimated_body_bytes"]
    assert result["recommended_budget_bytes"] > result["estimated_conservative_reserved_bytes"]
