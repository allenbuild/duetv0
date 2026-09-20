"""Synthetic HTTP and cached-format tests; never access the network or raw data."""

import fcntl
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, status=206, content_range="bytes 0-3/12572405231", payload=b"VRS!"):
        self.status, self.payload, self.read_calls = status, payload, 0
        self.headers = {"Content-Range": content_range, "Content-Length": str(len(payload))}

    def getheaders(self):
        return list(self.headers.items())

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, count):
        self.read_calls += 1
        result, self.payload = self.payload[:count], self.payload[count:]
        return result


def configured_probe(tmp_path, monkeypatch, response):
    module = script("probe_comind_remote_vrs.py")
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path / "bytes")
    monkeypatch.setattr(module, "REPORT_PATH", tmp_path / "report.json")
    calls = []

    class Connection:
        closed = False

        def __init__(self, host, timeout):
            calls.append((host, timeout))

        def request(self, method, path, headers):
            calls.append((method, path, headers))

        def getresponse(self):
            return response

        def close(self):
            self.closed = True

    monkeypatch.setattr(module.http.client, "HTTPSConnection", Connection)
    return module, calls


def test_exact_range_is_saved_and_budget_persisted(tmp_path, monkeypatch):
    response = Response()
    module, calls = configured_probe(tmp_path, monkeypatch, response)
    result = module.probe("helper", offset=0, length=4, label="fixture")
    assert result["status"] == "EXACT_RANGE_SAVED"
    assert Path(result["artifact"]).read_bytes() == b"VRS!"
    report = json.loads(module.REPORT_PATH.read_text())
    assert report["body_bytes_read_total"] == 4
    assert report["conservative_reserved_budget_bytes"] == 4 + 65536
    assert calls[1][2]["Range"] == "bytes=0-3"
    assert calls[1][2]["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("status,content_range", [(200, None), (302, None), (206, "bytes 0-7/9")])
def test_ignored_redirected_or_wrong_range_never_reads_body(
    tmp_path, monkeypatch, status, content_range
):
    response = Response(status=status, content_range=content_range)
    module, calls = configured_probe(tmp_path, monkeypatch, response)
    result = module.probe("helper", offset=0, length=4, label="fixture")
    assert result["status"] == "REJECTED_HEADERS_CLOSED_WITHOUT_BODY_READ"
    assert response.read_calls == 0
    assert len(calls) == 2  # one connection, one request, no redirect or retry


def test_head_never_reads_body(tmp_path, monkeypatch):
    response = Response(status=200)
    module, calls = configured_probe(tmp_path, monkeypatch, response)
    result = module.probe("leader", offset=None, length=None, label="fixture")
    assert result["status"] == "HEAD_ONLY_NO_BODY"
    assert response.read_calls == 0
    assert calls[1][0] == "HEAD"


def test_cumulative_budget_rejects_before_opening_connection(tmp_path, monkeypatch):
    module, calls = configured_probe(tmp_path, monkeypatch, Response())
    module.REPORT_PATH.write_text(json.dumps({"requests": [{"reserved_budget_bytes": 19_999_999}]}))
    with pytest.raises(ValueError, match="cumulative"):
        module.probe("helper", offset=0, length=4, label="fixture")
    assert not calls


def test_content_encoding_rejected_without_reading(tmp_path, monkeypatch):
    response = Response()
    response.headers["Content-Encoding"] = "gzip"
    module, _ = configured_probe(tmp_path, monkeypatch, response)
    assert module.probe("helper", offset=0, length=4, label="fixture")["body_bytes_read"] == 0
    assert response.read_calls == 0


@pytest.mark.parametrize(
    "offset,length", [(None, -100_000), (0, None), (True, 4), (0, False), (0.5, 4), (-1, 4)]
)
def test_invalid_range_cannot_reduce_budget_or_open_connection(
    tmp_path, monkeypatch, offset, length
):
    module, calls = configured_probe(tmp_path, monkeypatch, Response())
    with pytest.raises((ValueError, TypeError)):
        module.probe("helper", offset=offset, length=length, label="fixture")
    assert not calls


def test_concurrent_probe_cannot_spend_the_same_budget(tmp_path, monkeypatch):
    module, calls = configured_probe(tmp_path, monkeypatch, Response())
    with module.REPORT_PATH.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module.probe("helper", offset=0, length=4, label="fixture")
    assert not calls


def test_existing_artifact_prevents_duplicate_download(tmp_path, monkeypatch):
    module, calls = configured_probe(tmp_path, monkeypatch, Response())
    module.OUTPUT_ROOT.mkdir()
    (module.OUTPUT_ROOT / "helper_fixture_0_4.bin").write_bytes(b"VRS!")
    with pytest.raises(FileExistsError, match="duplicate"):
        module.probe("helper", offset=0, length=4, label="fixture")
    assert not calls


def test_offline_parser_detects_changed_cached_artifact(tmp_path, monkeypatch):
    module = script("inspect_comind_vrs_probe.py")
    artifact_path = tmp_path / "probe.bin"
    artifact_path.write_bytes(b"changed")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "status": "EXACT_RANGE_SAVED",
                        "artifact": str(artifact_path),
                        "body_bytes_read": 7,
                        "sha256": "incorrect",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "REPORT", report_path)
    with pytest.raises(ValueError, match="HTTP read log"):
        module.main()


def packed_string(value):
    data = value.encode()
    return struct.pack("<I", len(data)) + data


def test_description_parses_explicit_rgb_stream_and_calibration_tag():
    module = script("inspect_comind_vrs_probe.py")
    data = struct.pack("<IiHI", 1, 214, 1, 0)
    data += (
        struct.pack("<I", 1)
        + packed_string("VRS_Original_Recordable_Name")
        + packed_string("RGB Camera Class")
    )
    data += struct.pack("<I", 1) + packed_string("calib_json") + packed_string("{}")
    parsed = module.description(data)
    assert parsed["streams"]["214-1"]["vrs"]["VRS_Original_Recordable_Name"] == "RGB Camera Class"
    assert parsed["file_tags"]["calib_json"] == "{}"
    with pytest.raises(ValueError, match="truncated"):
        module.description(data[:-1])
    with pytest.raises(ValueError, match="unconsumed"):
        module.description(data + b"extra")


def make_index(module):
    records = np.array([(10.1, 160, 3, 214, 1), (10.2, 200, 3, 214, 1)], dtype=module.DISK_INFO)
    prelude = struct.pack("<IiHI", 1, 214, 1, len(records))
    payload = prelude + pa.compress(records.tobytes(), codec="zstd").to_pybytes() + bytes(32)
    return payload, len(prelude) + records.nbytes


def test_complete_padded_index_preserves_record_doubles_and_exact_offsets():
    module = script("inspect_comind_vrs_probe.py")
    payload, expected = make_index(module)
    records, offsets = module.index(
        payload, uncompressed_size=expected, first_user_offset=1000, file_size=1360
    )
    np.testing.assert_array_equal(offsets, [1000, 1160, 1360])
    np.testing.assert_array_equal(records["timestamp"], [10.1, 10.2])
    assert len(records) == 2


def test_index_size_or_record_extent_mismatch_rejected():
    module = script("inspect_comind_vrs_probe.py")
    payload, expected = make_index(module)
    with pytest.raises(ValueError, match="size"):
        module.index(
            payload, uncompressed_size=expected + 1, first_user_offset=1000, file_size=1360
        )
    with pytest.raises(ValueError, match="file size"):
        module.index(payload, uncompressed_size=expected, first_user_offset=1000, file_size=1361)
