"""Synthetic metadata and projection tests; no raw VRS/video files are required."""

import struct
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from duet.adapters.comind.frame_map import VideoVrsFrameMap
from duet.adapters.comind.vrs import (
    ImageOrientation,
    VrsRgbMetadata,
    bind_exported_calibration,
    camera_from_summary,
    exported_calibration_evidence,
    extract_rgb_metadata,
    native_calibration_summary,
    read_source_frame_numbers,
    verify_cached_orientation_audit,
    verify_clockwise_export_anchors,
    verify_export_pixel_geometry,
)
from duet.schemas.common import Provenance


def metadata(times=(10, 20), **kwargs):
    values = {
        "path": Path("synthetic.vrs"),
        "recording_id": "synthetic",
        "participant": "helper",
        "stream_id": "214-1",
        "device_timestamps_ns": list(times),
        "image_size": (64, 48),
        "nominal_rate_hz": 30.0,
        "provenance": Provenance("synthetic VRS fixture"),
    }
    values.update(kwargs)
    return VrsRgbMetadata(**values)


def camera(model="FISHEYE624"):
    calibration = pytest.importorskip("projectaria_tools.core.calibration")
    sophus = pytest.importorskip("projectaria_tools.core.sophus")
    matrix = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    params = [40, 31.2, 23.8, 0.05, -0.01, 0, 0, 0, 0, 0.001, -0.002, 0.0003, 0, -0.0004, 0]
    if model == "LINEAR":
        params = [40, 43, 31.2, 23.8]
    return calibration.CameraCalibration(
        "camera-rgb",
        calibration.CameraModelType.__members__[model],
        params,
        sophus.SE3.from_matrix(matrix),
        64,
        48,
        None,
        1.2,
        "synthetic",
        0.001,
        0.002,
    )


def test_sdk_device_time_extraction_preserves_int64_and_never_accesses_images():
    times = [2**53 + 1, 2**53 + 37]
    c = camera()

    class Provider:
        def get_stream_id_from_label(self, label):
            assert label == "camera-rgb"
            return "214-1"

        def get_num_data(self, stream_id):
            return 2

        def get_timestamps_ns(self, stream_id, domain):
            from projectaria_tools.core.sensor_data import TimeDomain

            assert domain is TimeDomain.DEVICE_TIME
            return times

        def get_image_configuration(self, stream_id):
            return SimpleNamespace(image_width=64, image_height=48, nominal_rate_hz=30)

        def get_device_calibration(self):
            return SimpleNamespace(get_camera_calib=lambda label: c)

        def get_image_data_by_index(self, *args):
            raise AssertionError("metadata extraction must not decode RGB images")

    result, returned_camera = extract_rgb_metadata(
        "nonexistent-fixture.vrs",
        recording_id="synthetic",
        participant="helper",
        provider=Provider(),
    )
    assert returned_camera is c
    assert result.device_timestamps_ns.tolist() == times
    assert result.timestamp(1).raw_value == times[1]
    assert result.timestamp(0).clock_domain.name.endswith("/helper/device_time")
    with pytest.raises(ValueError):
        result.device_timestamps_ns.setflags(write=True)


def test_participant_clocks_and_frame_numbers_remain_distinct():
    helper = metadata(source_frame_numbers=[1927, 1929])
    leader = metadata(participant="leader", source_frame_numbers=[2197, 2198])
    assert helper.clock_domain != leader.clock_domain
    assert helper.summary()["source_frame_number_nonunit_steps"] == 1
    assert helper.timestamp(0).raw_value == 10
    with pytest.raises(IndexError):
        helper.timestamp(-1)
    with pytest.raises(TypeError):
        helper.timestamp(True)


def test_empty_and_duplicate_metadata_are_preserved_without_index_alignment():
    assert metadata(times=()).summary()["frame_count"] == 0
    duplicate = metadata(times=(10, 10, 11))
    assert duplicate.summary()["duplicate_timestamp_count"] == 1
    assert not duplicate.summary()["strictly_increasing"]
    with pytest.raises(ValueError, match="monotonic"):
        metadata(times=(20, 10))


@pytest.mark.parametrize("times", [[1.0, 2.0], [True, False], [-1, 2], [2**64 - 1]])
def test_unsafe_timestamp_conversion_is_rejected(times):
    with pytest.raises(ValueError):
        metadata(times=times)


LAYOUT = {
    "data_layout": [
        {"name": "capture_timestamp_ns", "offset": 60, "type": "DataPieceValue<int64_t>"},
        {"name": "frame_number", "offset": 24, "type": "DataPieceValue<uint64_t>"},
    ]
}


def write_metadata_fixture(path, times, *, compression=0):
    chunks = []
    for index, timestamp in enumerate(times):
        payload = bytearray(128)
        struct.pack_into("<q", payload, 60, timestamp)
        struct.pack_into("<Q", payload, 24, 1927 + index)
        header = struct.pack("<IIiIdHBBI", 160, 160, 214, 2, timestamp / 1e9, 1, 3, compression, 0)
        chunks.append(header + payload)
    path.write_bytes(b"".join(chunks))


def test_source_numbers_are_read_from_metadata_and_checked_against_sdk(tmp_path):
    path = tmp_path / "synthetic.vrs"
    times = [2**53 + 1, 2**53 + 2]
    write_metadata_fixture(path, times)
    observed = metadata(times=times, path=path)
    numbers = read_source_frame_numbers(
        path, record_offsets=[0, 160], metadata=observed, declared_data_layout=LAYOUT
    )
    assert numbers.tolist() == [1927, 1928]
    with pytest.raises(ValueError, match="SDK DEVICE_TIME"):
        read_source_frame_numbers(
            path,
            record_offsets=[0, 160],
            metadata=replace(observed, device_timestamps_ns=[1, 2]),
            declared_data_layout=LAYOUT,
        )


def test_compressed_or_wrong_metadata_offsets_fail_without_decoding(tmp_path):
    path = tmp_path / "synthetic.vrs"
    write_metadata_fixture(path, [10, 20], compression=2)
    with pytest.raises(ValueError, match="record header"):
        read_source_frame_numbers(
            path, record_offsets=[0, 160], metadata=metadata(), declared_data_layout=LAYOUT
        )
    with pytest.raises(ValueError, match="count/order"):
        read_source_frame_numbers(
            path, record_offsets=[160, 0], metadata=metadata(), declared_data_layout=LAYOUT
        )


@pytest.mark.parametrize("model", ["LINEAR", "FISHEYE624"])
def test_cw90_projection_retains_native_frame_and_matches_official_utility(model):
    c = camera(model)
    evidence = exported_calibration_evidence(
        c,
        participant="helper",
        orientation=ImageOrientation.CLOCKWISE_90,
        verification="Synthetic image rotation verified independently",
        provenance=Provenance("synthetic"),
        segments=16,
    )
    assert evidence["resolution"] == [48, 64]
    assert evidence["camera_frame"] == "helper/camera-rgb"
    assert evidence["projection_roundtrip_max_error_pixels"] < 1e-7
    assert evidence["official_rotation_max_error_pixels"] < 1e-7
    np.testing.assert_allclose(
        evidence["T_device_camera"], c.get_transform_device_camera().to_matrix()
    )
    np.testing.assert_allclose(np.linalg.norm(evidence["boundary_rays_camera"], axis=1), 1)


def test_asymmetric_pixel_rotation_project_and_unproject():
    c = camera()
    bound = bind_exported_calibration(
        c,
        participant="leader",
        orientation=ImageOrientation.CLOCKWISE_90,
        image_size=(48, 64),
        verification="Synthetic CW90 pixel correspondence",
        provenance=Provenance("synthetic"),
    )
    ray = np.array([0.1, -0.2, 1.0])
    native_pixel = c.project(ray)
    np.testing.assert_allclose(bound.project(ray), [47 - native_pixel[1], native_pixel[0]])
    np.testing.assert_allclose(
        bound.unproject(bound.project(ray)), ray / np.linalg.norm(ray), atol=1e-9
    )
    assert bound.unproject([-1, 4]) is None


def test_orientation_and_resize_require_explicit_evidence():
    c = camera()
    args = {
        "camera": c,
        "participant": "helper",
        "orientation": ImageOrientation.CLOCKWISE_90,
        "image_size": (48, 64),
        "provenance": Provenance("synthetic"),
    }
    with pytest.raises(ValueError, match="verification"):
        bind_exported_calibration(**args, verification="")
    with pytest.raises(ValueError, match="crop/resize"):
        bind_exported_calibration(**(args | {"image_size": (48, 63)}), verification="fixture")


def test_native_calibration_cache_reconstructs_identical_projection_and_extrinsics():
    c = camera()
    summary = native_calibration_summary(c, participant="helper", provenance=Provenance("fixture"))
    restored = camera_from_summary(summary)
    np.testing.assert_allclose(restored.get_projection_params(), c.get_projection_params())
    np.testing.assert_allclose(
        restored.get_transform_device_camera().to_matrix(),
        c.get_transform_device_camera().to_matrix(),
    )
    assert restored.get_time_offset_sec_device_camera() == c.get_time_offset_sec_device_camera()
    assert restored.get_readout_time_sec() == c.get_readout_time_sec()
    assert not summary["exported_mp4_orientation_verified"]


@pytest.mark.parametrize("value", [np.array([1j, 0, 1]), np.array([np.inf, 0, 1])])
def test_invalid_geometry_not_silently_coerced(value):
    bound = bind_exported_calibration(
        camera(),
        participant="helper",
        orientation=ImageOrientation.NATIVE,
        image_size=(64, 48),
        verification="native fixture",
        provenance=Provenance("synthetic"),
    )
    with pytest.raises(ValueError, match="finite real"):
        bound.project(value)


def orientation_anchors():
    return [
        {
            "mp4_frame_index": index * 100,
            "clockwise_quarter_turns": 1,
            "all_rotations": [
                {"clockwise_quarter_turns": turn, "best_rmse": 0.3 if turn == 1 else 12.0}
                for turn in range(4)
            ],
        }
        for index in range(21)
    ]


def test_orientation_anchors_establish_rotation_with_complete_span_and_explicit_thresholds():
    result = verify_clockwise_export_anchors(orientation_anchors(), frame_count=2001)
    assert result["status"] == "VERIFIED"
    assert result["anchor_count"] == 21
    assert result["first_last_mp4_frame"] == [0, 2000]
    assert result["max_cw90_rmse"] == 0.3
    assert result["min_other_rotation_rmse_margin"] == 11.7


@pytest.mark.parametrize(
    "mutation",
    ["missing_rotation", "mixed_rotation", "weak_margin", "high_rmse", "boolean_rotation"],
)
def test_orientation_gate_rejects_incomplete_or_weak_rotation_evidence(mutation):
    anchors = deepcopy(orientation_anchors())
    if mutation == "missing_rotation":
        anchors[4]["all_rotations"].pop()
    elif mutation == "mixed_rotation":
        anchors[4]["clockwise_quarter_turns"] = 0
    elif mutation == "weak_margin":
        anchors[4]["all_rotations"][0]["best_rmse"] = 0.31
    elif mutation == "high_rmse":
        anchors[4]["all_rotations"][1]["best_rmse"] = 2.0
    else:
        anchors[4]["clockwise_quarter_turns"] = True
    with pytest.raises(ValueError):
        verify_clockwise_export_anchors(anchors, frame_count=2001)


@pytest.mark.parametrize("indices", [list(range(1, 21)), list(range(20)), list(range(0, 21, 2))])
def test_orientation_gate_requires_sufficient_coverage(indices):
    anchors = [orientation_anchors()[index] for index in indices]
    with pytest.raises(ValueError, match="cover|insufficient"):
        verify_clockwise_export_anchors(anchors, frame_count=2001)


def pixel_audit():
    candidates = [
        {"scale": 1.0, "dx": dx, "dy": dy, "rmse": 1.0 if dx == dy == 0 else 4.0}
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
    ] + [{"scale": scale, "dx": 0, "dy": 0, "rmse": 2.0} for scale in (0.999, 1.001)]
    return {
        "recording_id": "fixture",
        "participant": "helper",
        "frame_map_npz_sha256": "fixturehash",
        "pixel_geometry_method_version": 2,
        "clockwise_quarter_turns": 1,
        "samples": [
            {
                "mp4_frame_index": index,
                "vrs_rgb_frame_index": index + 3,
                "device_timestamp_ns": 100 + index * 1000,
                "image_dimensions": [48, 64],
                "ranked_candidates": deepcopy(candidates),
                "symmetric_lowpass_control": {
                    "symmetric_binomial_prefilter_passes": 3,
                    "ranked_candidates": deepcopy(candidates),
                },
            }
            for index in (0, 10, 20)
        ],
    }


def check_pixel_audit(audit):
    return verify_export_pixel_geometry(
        audit,
        recording_id="fixture",
        participant="helper",
        frame_count=21,
        image_size=(48, 64),
        frame_map_sha256="fixturehash",
    )


def test_pixel_geometry_uses_symmetric_control_for_scale_and_raw_scores_for_shifts():
    audit = pixel_audit()
    # Raw interpolation can denoise re-encoding artifacts. It is not scale evidence.
    audit["samples"][1]["ranked_candidates"][-1]["rmse"] = 0.5
    result = check_pixel_audit(audit)
    assert result["status"] == "VERIFIED"
    assert len(result["samples"]) == 3


@pytest.mark.parametrize(
    "failure", ["hash", "coverage", "shift", "scale", "missing_control", "missing_hypothesis"]
)
def test_pixel_geometry_rejects_conflicting_or_incomplete_evidence(failure):
    audit = pixel_audit()
    sample = audit["samples"][1]
    if failure == "hash":
        audit["frame_map_npz_sha256"] = "different"
    elif failure == "coverage":
        sample["mp4_frame_index"] = 11
    elif failure == "shift":
        sample["ranked_candidates"][0]["rmse"] = 0.5
    elif failure == "scale":
        sample["symmetric_lowpass_control"]["ranked_candidates"][-1]["rmse"] = 0.5
    elif failure == "missing_control":
        sample["symmetric_lowpass_control"]["symmetric_binomial_prefilter_passes"] = 0
    else:
        sample["ranked_candidates"].pop()
    with pytest.raises(ValueError):
        check_pixel_audit(audit)


def cached_orientation_fixture():
    count = 4001
    verified = np.array([0, *range(1800, count, 100)])
    status = np.full(count, "UNRESOLVED", dtype="U10")
    status[verified] = "VERIFIED"
    indices = np.full(count, -1, dtype=np.int64)
    timestamps = indices.copy()
    indices[verified] = verified * 2
    timestamps[verified] = 2**53 + verified * 37
    mapping = VideoVrsFrameMap(
        "fixture",
        "helper",
        np.arange(count),
        indices,
        timestamps,
        status,
        np.ones(count),
        np.arange(count),
        Fraction(1, 30),
        Provenance("fixture"),
    )
    rows = []
    for index in verified:
        row = deepcopy(orientation_anchors()[0])
        row.update(
            mp4_frame_index=int(index),
            vrs_rgb_frame_index=int(indices[index]),
            device_timestamp_ns=int(timestamps[index]),
        )
        rows.append(row)
    bridges = []
    for index in (20, 904):
        row = deepcopy(orientation_anchors()[0])
        row.update(
            mp4_frame_index=index,
            orientation_candidate_vrs_rgb_frame_index=index * 2,
            temporal_assignment=False,
        )
        bridges.append(row)
    audit = {
        "recording_id": "fixture",
        "participant": "helper",
        "frame_map_npz_sha256": "npz",
        "frame_map_metadata_sha256": "metadata",
        "samples": rows,
        "orientation_only_samples": bridges,
    }
    return mapping, audit


def check_cached_orientation(mapping, audit):
    return verify_cached_orientation_audit(
        audit, mapping, frame_map_sha256="npz", frame_map_metadata_sha256="metadata"
    )


def test_cached_verified_pairs_and_orientation_only_bridges_preserve_temporal_map():
    mapping, audit = cached_orientation_fixture()
    before = mapping.vrs_device_timestamp_ns.copy()
    result = check_cached_orientation(mapping, audit)
    assert result["status"] == "VERIFIED"
    assert result["thresholds"]["maximum_anchor_gap_frames"] == 900
    assert result["thresholds"]["maximum_rmse"] == 1.5
    assert result["thresholds"]["minimum_rotation_margin"] == 5.0
    assert result["orientation_only_mp4_frames"] == [20, 904]
    assert not result["temporal_map_modified"]
    assert mapping.timestamp_at(20) is None
    assert mapping.timestamp_at(904) is None
    np.testing.assert_array_equal(before, mapping.vrs_device_timestamp_ns)


def test_verified_pair_only_audit_cannot_skip_excessive_coverage_gap():
    mapping, audit = cached_orientation_fixture()
    audit["orientation_only_samples"] = []
    with pytest.raises(ValueError, match="cover"):
        check_cached_orientation(mapping, audit)


@pytest.mark.parametrize(
    "failure",
    [
        "hash",
        "metadata_hash",
        "source_index",
        "float_timestamp",
        "omitted_pair",
        "bridge_timestamp",
        "weak_bridge",
        "bridge_duplicate",
        "unverified_pair",
    ],
)
def test_cached_orientation_audit_rejects_stale_or_promoted_evidence(failure):
    mapping, audit = cached_orientation_fixture()
    if failure == "hash":
        audit["frame_map_npz_sha256"] = "stale"
    elif failure == "metadata_hash":
        audit["frame_map_metadata_sha256"] = "stale"
    elif failure == "source_index":
        audit["samples"][0]["vrs_rgb_frame_index"] = 1
    elif failure == "float_timestamp":
        audit["samples"][0]["device_timestamp_ns"] = float(2**53)
    elif failure == "omitted_pair":
        audit["samples"].pop(3)
    elif failure == "bridge_timestamp":
        audit["orientation_only_samples"][0]["device_timestamp_ns"] = 123
    elif failure == "weak_bridge":
        audit["orientation_only_samples"][0]["all_rotations"][1]["best_rmse"] = 1.6
    elif failure == "bridge_duplicate":
        audit["orientation_only_samples"][0]["mp4_frame_index"] = 0
    else:
        audit["samples"][0]["mp4_frame_index"] = 20
    with pytest.raises(ValueError):
        check_cached_orientation(mapping, audit)
