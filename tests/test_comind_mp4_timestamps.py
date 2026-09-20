"""Synthetic metadata checks do not require Project Aria or real MP4 files."""

import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from duet.adapters.comind import mp4_timestamps as module
from duet.adapters.comind.mp4_timestamps import Mp4DeviceTimestamps
from duet.adapters.comind.mps import device_clock
from duet.schemas.common import Provenance
from duet.schemas.time import TimeUnit

PROVENANCE = Provenance("synthetic official export")


def mapping(values, frame_count=None, participant="helper"):
    return Mp4DeviceTimestamps(
        "synthetic",
        participant,
        values,
        len(values) if frame_count is None else frame_count,
        PROVENANCE,
    )


def install_fake(monkeypatch, values, frames=3, description="[10,20,30]"):
    calls = []

    def extract(path):
        calls.append(path)
        return values

    monkeypatch.setattr(module, "_video_headers", lambda path: (frames, description))
    monkeypatch.setattr(
        module, "_load_official_extractor", lambda: (extract, "official.fn", "2.test")
    )
    return calls


def test_exact_local_timestamp_above_float_precision_and_immutable_copy():
    values = np.array([2**53 + 1, 2**53 + 2], dtype=np.int64)
    result = mapping(values)
    values[0] = 1
    stamp = result.timestamp_at(0)
    assert stamp.raw_value == 2**53 + 1
    assert stamp.unit is TimeUnit.NANOSECONDS
    assert stamp.seconds == Fraction(2**53 + 1, 10**9)
    assert stamp.clock_domain == device_clock("synthetic", "helper")
    assert mapping([1], participant="leader").timestamp_at(0).clock_domain != stamp.clock_domain
    with pytest.raises(ValueError):
        result.device_timestamps_ns[0] = 2
    with pytest.raises(ValueError):
        result.device_timestamps_ns.setflags(write=True)


@pytest.mark.parametrize("values", [[], [1], [1, 2, 3, 4]])
def test_count_must_equal_every_video_frame(values):
    with pytest.raises(ValueError, match="count"):
        mapping(values, frame_count=3)


@pytest.mark.parametrize("values", [10, [[10, 20]], [1.0, 2.0], [True, False], [None, 2], [1j, 2j]])
def test_scalar_noninteger_or_nested_values_are_not_repaired(values):
    with pytest.raises(ValueError):
        mapping(values, frame_count=2)


@pytest.mark.parametrize("count", [False, True, 0, -1, 2.0])
def test_invalid_frame_counts(count):
    with pytest.raises(ValueError):
        mapping([1, 2], frame_count=count)


def test_decreasing_values_rejected_but_exporter_duplicates_preserved():
    with pytest.raises(ValueError, match="nondecreasing"):
        mapping([20, 10])
    repeated = mapping([10, 10, 20])
    assert repeated.timestamp_at(0).raw_value == repeated.timestamp_at(1).raw_value
    assert repeated.frame_count == 3


@pytest.mark.parametrize("index", [True, 1.0, None, "1"])
def test_frame_index_must_be_integer(index):
    with pytest.raises(TypeError):
        mapping([1, 2]).timestamp_at(index)


@pytest.mark.parametrize("index", [-1, 2])
def test_frame_index_bounds(index):
    with pytest.raises(IndexError):
        mapping([1, 2]).timestamp_at(index)


def test_load_calls_official_function_and_keeps_version_provenance(monkeypatch):
    calls = install_fake(monkeypatch, np.array([10, 20, 30]))
    result = module.load_mp4_device_timestamps(
        "synthetic.mp4", recording_id="synthetic", participant="helper", expected_frame_count=3
    )
    assert calls == ["synthetic.mp4"]
    assert result.timestamp_at(2).raw_value == 30
    assert "official.fn" in result.provenance.detail
    assert "2.test" in result.provenance.detail
    assert result.provenance.parents[0].source == module.TIMESTAMP_EVIDENCE


def test_actual_header_frame_count_controls_validation(monkeypatch):
    calls = install_fake(monkeypatch, np.array([10, 20, 30]), frames=4)
    with pytest.raises(ValueError, match="expected"):
        module.load_mp4_device_timestamps(
            "synthetic.mp4", recording_id="synthetic", participant="helper", expected_frame_count=3
        )
    assert not calls
    with pytest.raises(ValueError, match="count 3.*count 4"):
        module.load_mp4_device_timestamps(
            "synthetic.mp4", recording_id="synthetic", participant="helper"
        )


def test_scalar_metadata_official_success_is_not_a_per_frame_mapping(monkeypatch):
    install_fake(monkeypatch, np.array([919623571487]), frames=21109, description="919623571487")
    result = module.probe_mp4_device_timestamps(
        "synthetic.mp4",
        recording_id="synthetic",
        participant="helper",
        expected_frame_count=21109,
        trajectory_ranges_us={"multislam": (920356883, 1620641622)},
    )
    assert result["official_call_succeeded"]
    assert result["returned_type"] == "numpy.ndarray"
    assert result["returned_shape"] == [1]
    assert result["returned_dtype"] == "int64"
    assert result["returned_values_ns"] == [919623571487]
    assert result["extracted_timestamp_count"] == 1
    assert result["nondecreasing"] is None
    assert result["timestamp_interval_ns"] == {"median": None, "p95": None, "max": None}
    assert not result["per_frame_device_mapping_verified"]
    coverage = result["extracted_value_range_checks"]["multislam"]
    assert coverage["first_minus_range_start_ns"] == -733311513
    assert coverage["extracted_values_inside_range"] == 0
    assert coverage["not_video_coverage"]


def test_complete_arrays_report_duplicates_and_precise_intervals(monkeypatch):
    base = 2**53 + 1
    install_fake(monkeypatch, np.array([base, base, base + 2], dtype=np.int64))
    result = module.probe_mp4_device_timestamps(
        "synthetic.mp4", recording_id="synthetic", participant="helper"
    )
    assert result["per_frame_device_mapping_verified"]
    assert result["nondecreasing"]
    assert not result["strictly_increasing"]
    assert result["duplicate_timestamp_count"] == 1
    assert result["timestamp_interval_ns"]["median"] == 1
    assert result["timestamp_interval_ns"]["max"] == 2


def test_official_errors_preserved_without_a_fallback_parser(monkeypatch):
    install_fake(monkeypatch, np.array([10]))

    def missing(path):
        raise KeyError("description")

    monkeypatch.setattr(
        module, "_load_official_extractor", lambda: (missing, "official.fn", "2.test")
    )
    result = module.probe_mp4_device_timestamps(
        "synthetic.mp4", recording_id="synthetic", participant="helper"
    )
    assert not result["official_call_succeeded"]
    assert result["error_type"] == "KeyError"
    assert result["error"] == "'description'"
    with pytest.raises(KeyError):
        module.load_mp4_device_timestamps(
            "synthetic.mp4", recording_id="synthetic", participant="helper"
        )


def test_historical_official_import_path_is_supported(monkeypatch):
    calls = []

    def fake(path):
        return np.array([1])

    def importing(name):
        calls.append(name)
        if name == module.OFFICIAL_MODULES[0]:
            raise ModuleNotFoundError("absent new path", name=name)
        return SimpleNamespace(get_timestamp_from_mp4=fake)

    monkeypatch.setattr(module.importlib, "import_module", importing)
    monkeypatch.setattr(module, "version", lambda package: "old.test")
    function, path, package_version = module._load_official_extractor()
    assert function is fake
    assert path.startswith(module.OFFICIAL_MODULES[1])
    assert package_version == "old.test"
    assert calls == list(module.OFFICIAL_MODULES)


def test_missing_import_dependency_not_hidden_by_historical_fallback(monkeypatch):
    calls = []

    def importing(name):
        calls.append(name)
        raise ModuleNotFoundError("No module named moviepy", name="moviepy")

    monkeypatch.setattr(module.importlib, "import_module", importing)
    with pytest.raises(ModuleNotFoundError, match="moviepy"):
        module._load_official_extractor()
    assert calls == [module.OFFICIAL_MODULES[0]]


def test_header_read_never_decodes(monkeypatch):
    class Container:
        def __init__(self):
            self.metadata = {"description": "[1,2,3]"}
            self.streams = SimpleNamespace(video=[SimpleNamespace(frames=3)])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self):
            raise AssertionError("header probe must not decode")

    import av

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    assert module._video_headers(Path("synthetic.mp4")) == (3, "[1,2,3]")


@pytest.mark.parametrize("bounds", [(True, 2), (1.0, 2), (2, 1), (1,), (1, 2, 3)])
def test_probe_rejects_range_coercion(monkeypatch, bounds):
    install_fake(monkeypatch, np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="integer microsecond bounds"):
        module.probe_mp4_device_timestamps(
            "synthetic.mp4",
            recording_id="synthetic",
            participant="helper",
            trajectory_ranges_us={"multislam": bounds},
        )


@pytest.mark.parametrize("expected", [True, 3.0, 0, -1])
def test_expected_frame_count_rejects_coercion(monkeypatch, expected):
    install_fake(monkeypatch, np.array([1, 2, 3]))
    for operation in (module.load_mp4_device_timestamps, module.probe_mp4_device_timestamps):
        with pytest.raises(ValueError, match="positive integer"):
            operation(
                "synthetic.mp4",
                recording_id="synthetic",
                participant="helper",
                expected_frame_count=expected,
            )


def probe_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_comind_mp4_timestamps.py"
    spec = importlib.util.spec_from_file_location("duet_timestamp_probe_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_probe_cli_writes_small_report_and_keeps_scalar_gate_closed(tmp_path, monkeypatch):
    script = probe_script()
    recording = "43276420-701f-4731-b9ab-bebc7fd14994"
    raw_root = tmp_path / "data" / "raw" / "comind"
    directory = raw_root / "recordings" / recording / "mp4s"
    directory.mkdir(parents=True)
    originals = {}
    for role in ("helper", "leader"):
        for suffix in ("trimmed_sync", "sync"):
            path = directory / f"{role}_{suffix}.mp4"
            path.write_bytes(b"synthetic placeholder; metadata reader is mocked")
            originals[path] = (path.read_bytes(), path.stat().st_mtime_ns)
    install_fake(monkeypatch, np.array([10], dtype=np.int64), frames=3, description="10")
    monkeypatch.setattr(script, "_load_official_extractor", module._load_official_extractor)
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    output = tmp_path / "outputs"
    result = script.run_probe(
        dataset_root=raw_root,
        recording_id=recording,
        output_root=output,
        validation_report=None,
        expected_trimmed_frame_count=3,
    )
    saved = json.loads((output / f"{recording}.json").read_text())
    assert saved == result
    assert len(result["videos"]) == 4
    assert not result["frame_index_to_device_time_verified"]
    assert not result["trajectory_sources_reread"]
    assert all(item["extracted_timestamp_count"] == 1 for item in result["videos"].values())
    assert all(item["raw_file_size_and_mtime_unchanged"] for item in result["videos"].values())
    for path, original in originals.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original


def test_probe_cli_rejects_raw_output_before_reading_files(tmp_path):
    script = probe_script()
    raw_root = tmp_path / "data" / "raw" / "comind"
    with pytest.raises(ValueError, match="outside immutable raw"):
        script.run_probe(
            dataset_root=raw_root,
            recording_id="43276420-701f-4731-b9ab-bebc7fd14994",
            output_root=raw_root / "reports",
            validation_report=None,
        )
    assert not raw_root.exists()


def test_probe_cli_rejects_other_recordings_range_audit(tmp_path):
    script = probe_script()
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"recording_id": "another"}))
    with pytest.raises(ValueError, match="another recording"):
        script.run_probe(
            dataset_root=tmp_path / "raw",
            recording_id="43276420-701f-4731-b9ab-bebc7fd14994",
            output_root=tmp_path / "reports",
            validation_report=audit,
        )
