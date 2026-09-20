"""Derived-only orientation evidence stays separate from capture-time mapping."""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import av
import numpy as np
import pytest

from duet.adapters.comind.mps import device_clock

RECORDING = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_comind_cached_orientation.py"
    spec = importlib.util.spec_from_file_location("cached_orientation_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, value):
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_candidates(*, control=False):
    rows = [
        {"scale": 1.0, "dx": dx, "dy": dy, "rmse": 1.0} for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    ]
    next(row for row in rows if row["dx"] == row["dy"] == 0)["rmse"] = 0.1 if control else 0.3
    rows += [{"scale": scale, "dx": 0, "dy": 0, "rmse": 0.2} for scale in (0.999, 1.001)]
    return sorted(rows, key=lambda row: row["rmse"])


@pytest.fixture
def cache(tmp_path):
    processed = tmp_path / "processed"
    root = processed / RECORDING
    directory = root / "frame_maps"
    directory.mkdir(parents=True)
    (root / "vrs").mkdir()
    count = 61
    verified = np.arange(0, count, 3)
    status = np.full(count, "UNRESOLVED", dtype="U10")
    status[verified] = "VERIFIED"
    native_indices = np.full(count, -1, dtype=np.int64)
    native_indices[verified] = verified
    times = 1_000_000_000 + np.arange(count, dtype=np.int64) * 30_000_000
    mapped_times = np.full(count, -1, dtype=np.int64)
    mapped_times[verified] = times[verified]
    path = directory / "helper_frame_map.npz"
    np.savez_compressed(
        path,
        mp4_frame_index=np.arange(count),
        vrs_rgb_frame_index=native_indices,
        vrs_device_timestamp_ns=mapped_times,
        status=status,
        match_confidence=(status == "VERIFIED").astype(float),
        mp4_pts=np.arange(count),
        mp4_time_base_numerator=np.array(1),
        mp4_time_base_denominator=np.array(30),
    )
    save_json(
        path.with_suffix(".json"),
        {
            "recording_id": RECORDING,
            "participant": "helper",
            "vrs_rgb_frame_count": count,
            "npz_sha256": digest(path),
            "mapping_evidence": "synthetic image matches",
            "mp4_dimensions": [1408, 1408],
        },
    )
    native = np.random.default_rng(26).integers(0, 256, size=(count, 32, 32), dtype=np.uint8)
    mp4 = np.rot90(native, -1, axes=(1, 2)).copy()
    for kind, fingerprints in (("mp4", mp4), ("remote_vrs", native)):
        folder = directory / "fingerprints" / f"helper_{kind}"
        folder.mkdir(parents=True)
        save_json(
            folder / "source.json",
            {
                "source": "https://example.invalid/private-token-not-to-copy",
                "kind": kind,
                "frame_count": count,
                "fingerprint_size": 32,
                "method": "FFmpeg gray 32x32, native orientation for VRS",
            },
        )
        for name, values in {
            "fingerprints": fingerprints,
            "done": np.ones(count, dtype=bool),
            "valid": np.ones(count, dtype=bool),
            "pts": np.arange(count, dtype=np.int64),
        }.items():
            np.save(folder / f"{name}.npy", values)
    np.savez_compressed(
        root / "vrs/helper_rgb_metadata.npz",
        device_timestamps_ns=times,
        vrs_rgb_frame_index=np.arange(count),
        clock_domain=np.array(device_clock(RECORDING, "helper").name),
        stream_id=np.array("214-1"),
        metadata_known=np.ones(count, dtype=bool),
    )
    save_json(
        directory / "helper_anchors.json",
        [
            {
                "mp4_frame_index": int(index),
                "vrs_index": int(index),
                "best_rmse": 0.0,
                "clockwise_quarter_turns": 1,
            }
            for index in verified
        ],
    )
    samples = []
    for index in (0, 30, 60):
        samples.append(
            {
                "mp4_frame_index": index,
                "vrs_rgb_frame_index": index,
                "device_timestamp_ns": int(times[index]),
                "image_dimensions": [1408, 1408],
                "ranked_candidates": pixel_candidates(),
                "symmetric_lowpass_control": {
                    "symmetric_binomial_prefilter_passes": 3,
                    "ranked_candidates": pixel_candidates(control=True),
                },
            }
        )
    save_json(
        directory / "helper_pixel_geometry.json",
        {
            "recording_id": RECORDING,
            "participant": "helper",
            "frame_map_npz_sha256": digest(path),
            "pixel_geometry_method_version": 2,
            "clockwise_quarter_turns": 1,
            "samples": samples,
            "zero_shift_unit_scale_wins_all_samples": False,
        },
    )
    return processed, directory


def test_all_verified_pairs_use_exact_source_identity_and_leave_caches_unchanged(
    cli, cache, monkeypatch
):
    processed, _ = cache
    before = {path: digest(path) for path in processed.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("orientation audit must not decode media")

    monkeypatch.setattr(av, "open", forbidden)
    result = cli.audit_role(processed, RECORDING, "helper")
    assert result["verified_pair_count"] == 21
    assert result["orientation_at_verified_pairs"] == "VERIFIED"
    assert result["combined_orientation_verification"]["status"] == "VERIFIED"
    assert result["rotation_summary"][1]["rmse"]["maximum"] == 0
    assert [row["mp4_frame_index"] for row in result["samples"]] == list(range(0, 61, 3))
    assert all(row["mp4_frame_index"] == row["vrs_rgb_frame_index"] for row in result["samples"])
    assert "private-token-not-to-copy" not in json.dumps(result, allow_nan=False)
    assert before == {path: digest(path) for path in processed.rglob("*") if path.is_file()}
    assert (
        result["cached_full_resolution_pixel_geometry"]["raw_all_hypotheses_identity_winner_flag"]
        is False
    )
    assert (
        result["cached_full_resolution_pixel_geometry"]["strict_existing_verifier"]["status"]
        == "VERIFIED"
    )


def test_orientation_only_rows_never_assign_device_time_or_modify_maps(cli, cache):
    processed, directory = cache
    map_path = directory / "helper_frame_map.npz"
    before = digest(map_path)
    result = cli.audit_role(processed, RECORDING, "helper", orientation_frames=[20, 22])
    rows = result["orientation_only_samples"]
    assert [row["orientation_candidate_vrs_rgb_frame_index"] for row in rows] == [20, 22]
    assert all(row["temporal_assignment"] is False for row in rows)
    assert all(row["original_map_status"] == "UNRESOLVED" for row in rows)
    assert all(
        "device_timestamp_ns" not in row and "vrs_rgb_frame_index" not in row for row in rows
    )
    assert result["verified_pair_count"] == 21
    assert result["combined_orientation_verification"]["anchor_count"] == 23
    assert digest(map_path) == before
    with np.load(map_path) as arrays:
        assert arrays["status"][20] == "UNRESOLVED"
        assert arrays["vrs_device_timestamp_ns"][20] == -1


@pytest.mark.parametrize("frames", [[3], [True], [-1], [61], [20, 20]])
def test_orientation_only_requests_are_explicit_unique_unresolved_indices(cli, cache, frames):
    with pytest.raises(ValueError):
        cli.audit_role(cache[0], RECORDING, "helper", orientation_frames=frames)


@pytest.mark.parametrize(
    "mode", ["missing_fingerprint", "wrong_rotation", "ambiguous_images", "wrong_pts"]
)
def test_orientation_only_pairs_fail_closed_without_reliable_rotation_evidence(cli, cache, mode):
    processed, directory = cache
    folder = directory / "fingerprints"
    if mode == "missing_fingerprint":
        path = folder / "helper_mp4/valid.npy"
        array = np.load(path)
        array[20] = False
        np.save(path, array)
    elif mode == "wrong_pts":
        path = folder / "helper_mp4/pts.npy"
        array = np.load(path)
        array[20] += 1
        np.save(path, array)
    elif mode == "wrong_rotation":
        path = folder / "helper_mp4/fingerprints.npy"
        array = np.load(path)
        array[20] = np.load(folder / "helper_remote_vrs/fingerprints.npy")[20]
        np.save(path, array)
    else:
        path = folder / "helper_remote_vrs/fingerprints.npy"
        array = np.load(path)
        array[19] = array[20]
        np.save(path, array)
    with pytest.raises(ValueError):
        cli.audit_role(processed, RECORDING, "helper", orientation_frames=[20])


@pytest.mark.parametrize(
    "mode",
    [
        "map_checksum",
        "cache_method",
        "valid_mask",
        "timestamp_clock",
        "pixel_identity",
        "pixel_gate",
    ],
)
def test_existing_cache_and_pixel_evidence_must_pass_unchanged_strict_gates(cli, cache, mode):
    processed, directory = cache
    if mode == "map_checksum":
        path = directory / "helper_frame_map.json"
        value = json.loads(path.read_text())
        value["npz_sha256"] = "0" * 64
        save_json(path, value)
    elif mode == "cache_method":
        path = directory / "fingerprints/helper_remote_vrs/source.json"
        value = json.loads(path.read_text())
        value["method"] = "already rotated"
        save_json(path, value)
    elif mode == "valid_mask":
        path = directory / "fingerprints/helper_remote_vrs/valid.npy"
        value = np.load(path)
        value[3] = False
        np.save(path, value)
    elif mode == "timestamp_clock":
        path = directory.parent / "vrs/helper_rgb_metadata.npz"
        with np.load(path) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        values["clock_domain"] = np.array(device_clock(RECORDING, "leader").name)
        np.savez_compressed(path, **values)
    else:
        path = directory / "helper_pixel_geometry.json"
        value = json.loads(path.read_text())
        if mode == "pixel_identity":
            value["samples"][0]["device_timestamp_ns"] += 1
        else:
            candidates = value["samples"][0]["symmetric_lowpass_control"]["ranked_candidates"]
            next(row for row in candidates if (row["scale"], row["dx"], row["dy"]) == (1, 0, 0))[
                "rmse"
            ] = 2
        save_json(path, value)
    with pytest.raises(ValueError):
        cli.audit_role(processed, RECORDING, "helper")


def test_coarse_bad_correspondence_is_preserved_as_diagnostic_not_used_for_rotation(cli, cache):
    processed, directory = cache
    path = directory / "helper_anchors.json"
    anchors = json.loads(path.read_text())
    anchors[1]["vrs_index"] += 20
    anchors[1]["best_rmse"] = 23
    save_json(path, anchors)
    before = deepcopy(anchors)
    result = cli.audit_role(processed, RECORDING, "helper")
    row = result["coarse_anchor_audit"][1]
    assert row["original_vrs_index"] == 23 and row["original_best_rmse"] == 23
    assert row["final_verified_pair"]["vrs_rgb_frame_index"] == 3
    assert row["final_verified_pair"]["best_rmse"] == 0
    assert result["coarse_anchor_summary"]["verified_source_mismatch_count"] == 1
    assert json.loads(path.read_text()) == before


def test_good_pair_rotation_does_not_silently_relax_full_span_coverage(cli, cache, monkeypatch):
    monkeypatch.setitem(cli.THRESHOLDS, "maximum_anchor_gap_frames", 2)
    result = cli.audit_role(cache[0], RECORDING, "helper")
    assert result["orientation_at_verified_pairs"] == "VERIFIED"
    assert result["coverage"]["strict_full_span_anchor_coverage"] == "INSUFFICIENT_DATA"
    assert result["coverage"]["maximum_verified_frame_gap"] == 3
    assert result["combined_orientation_verification"]["status"] == "FAIL"


def test_raw_path_guard_does_not_access_real_raw_data(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="data/raw"):
        cli.audit_role(tmp_path / "data/raw", RECORDING, "helper")
    assert not (tmp_path / "data/raw").exists()
