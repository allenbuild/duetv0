"""Playback assembly from tiny generated caches, without videos/VRS/GUI."""

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from duet.adapters.comind.annotations import parse_handover_segments
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.playback import load_playback
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.adapters.comind.temporal_validation import (
    ParticipantStreams,
    TemporalThresholds,
    bind_annotations,
    rank_handovers,
    validate_participant_frames,
)
from duet.schemas.common import Provenance
from duet.visualization.comind_handover import assemble_index_frame

RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(name):
    return MappingEvidence(name, EvidenceStatus.VERIFIED, "Synthetic authority", ("fixture",))


@pytest.fixture
def playback_cache(tmp_path):
    root = tmp_path / "processed"
    directory = root / RECORDING
    for folder in ("vrs", "frame_maps", "frame_validation"):
        (directory / folder).mkdir(parents=True)
    generation = "synthetic-generation"
    thresholds = TemporalThresholds()
    validations, identities, inputs = {}, {}, {}
    for role in ("helper", "leader"):
        times = np.arange(6, dtype=np.int64) * 33_333 + (
            1_000_000 if role == "helper" else 90_000_000
        )
        source = ParticipantStreams(
            RECORDING,
            role,
            "shared",
            device_clock(RECORDING, role).name,
            times,
            np.zeros((6, 3)),
            np.tile([0.0, 0.0, 0.0, 1.0], (6, 1)),
            times,
            np.ones((6, 2, 21, 3)),
            np.ones((6, 2)),
            np.zeros(6, dtype=np.int64),
            np.ones(6, dtype=bool),
        )
        confidence = source.hand_confidence.copy()
        confidence[0, 0] = 0
        source = replace(source, hand_confidence=confidence)
        mapping = {
            "mp4_frame_index": np.arange(6, dtype=np.int64),
            "vrs_rgb_frame_index": np.arange(6, dtype=np.int64),
            "vrs_device_timestamp_ns": times * 1000 + 1000,
            "status": np.full(6, "VERIFIED", dtype="U10"),
            "match_confidence": np.full(6, 0.9),
            "mp4_pts": np.arange(6, dtype=np.int64) * 512,
            "mp4_time_base_numerator": np.asarray(1),
            "mp4_time_base_denominator": np.asarray(15360),
        }
        path = directory / "frame_maps" / f"{role}_frame_map.npz"
        np.savez_compressed(path, **mapping)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "recording_id": RECORDING,
                    "participant": role,
                    "mapping_evidence": "synthetic",
                    "npz_sha256": digest(path),
                }
            )
        )
        native = directory / "vrs" / f"{role}_rgb_metadata.npz"
        np.savez_compressed(native, device_timestamps_ns=mapping["vrs_device_timestamp_ns"])
        result = validate_participant_frames(
            mapping, source, recording_id=RECORDING, participant=role
        )
        validations[role] = result
        aligned = directory / "frame_validation" / f"{role}_aligned.npz"
        np.savez_compressed(aligned, **result.arrays, generation_id=np.asarray(generation))
        identities[role] = {"path": str(aligned), "sha256": digest(aligned)}
        inputs[role] = {
            "path": str(path),
            "sha256": digest(path),
            "metadata_path": str(path.with_suffix(".json")),
            "metadata_sha256": digest(path.with_suffix(".json")),
            "native_metadata_path": str(native),
            "native_metadata_sha256": digest(native),
        }
    annotation_evidence = evidence("annotation_to_comind_sync_frame_index")
    annotations = parse_handover_segments(
        [
            {
                "skip": False,
                "start_frame": 1,
                "end_frame": 3,
                "start_time": 1 / 30,
                "end_time": 0.1,
                "initiator": "left",
                "delivering_flow": "rtl",
                "object_category_level_1": "bowl",
            }
        ],
        recording_uuid=RECORDING,
        provenance=Provenance("fixture"),
    )
    bound = bind_annotations(
        annotations,
        recording_id=RECORDING,
        frame_count=6,
        binding_evidence=annotation_evidence,
    )
    ranking = rank_handovers(
        validations["helper"],
        validations["leader"],
        bound,
        context_frames=1,
        alignment_evidence=evidence("paired_video_frame_alignment"),
    )
    report = {
        "recording_id": RECORDING,
        "timeline": "comind_sync_frame_index",
        "generation_id": generation,
        "thresholds": asdict(thresholds),
        "shared_world": {
            "status": "VERIFIED",
            "graph_uid": "shared",
            "frame": f"comind/{RECORDING}/multislam/shared",
        },
        "participants": {role: value.summary for role, value in validations.items()},
        "handover_ranking": ranking,
        "annotation_binding": annotation_evidence.to_dict(),
        "aligned_cache_identities": identities,
        "frame_map_inputs": inputs,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    return root, report_path, directory, report


def test_playback_exact_local_timestamps_and_missing_low_confidence(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    frames = list(bundle.iter_frames())
    assert [frame.frame_index for frame in frames] == [0, 1, 2, 3, 4]
    helper, leader = frames[0].participants
    assert helper.device_timestamp.raw_value == 1_000_001_000
    assert helper.pose_timestamp.raw_value == 1_000_000_000
    assert helper.hands[0].timestamp.raw_value == 1_000_000_000
    assert helper.hands[0].points is None and helper.hands[0].metadata.confidence.value == 0
    assert leader.device_timestamp.clock_domain != helper.device_timestamp.clock_domain
    updates, status = assemble_index_frame(frames[1], bundle.bindings)
    assert status["participants"]["helper"]["pose_residual_ns"] == -1000
    assert status["participants"]["leader"]["hands"]["right"]["present"]
    assert status["active_handovers"][0]["initiator_source_label"] == "left"
    assert frames[0].annotations == () and len(frames[3].annotations) == 1
    assert any(update.kind == "pose" for update in updates)


def test_with_annotation_reuses_loaded_arrays_and_preserves_source_key(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    first = dict(bundle.annotations[0], annotation_id="000001", numeric_id="7")
    second = dict(first, annotation_id="000004", numeric_id="11", start_frame=4, end_frame=5)
    bundle = replace(bundle, annotations=(first, second))
    selected = bundle.with_annotation("000004", context_frames=1)
    assert selected.selection["annotation_id"] == "000004"
    assert selected.selection["numeric_id"] == "11"
    assert (selected.start_index, selected.end_index) == (3, 5)
    assert selected.frame_maps is bundle.frame_maps
    assert selected.aligned_arrays is bundle.aligned_arrays
    assert (bundle.start_index, bundle.end_index) == (0, 4)
    assert [frame.frame_index for frame in selected.iter_frames()] == [3, 4, 5]
    np.testing.assert_array_equal(
        selected.trajectories["helper"], bundle.aligned_arrays["helper"]["pose_translation_m"][3:6]
    )


def test_with_annotation_does_not_reopen_deleted_reports_or_caches(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    bundle = replace(bundle, annotations=(dict(bundle.annotations[0], annotation_id="000001"),))
    path.unlink()
    assert bundle.with_annotation("000001", context_frames=0).start_index == 1


@pytest.mark.parametrize("context", [-1, True, 1.5])
def test_with_annotation_rejects_invalid_context(playback_cache, context):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    with pytest.raises(ValueError, match="context_frames"):
        bundle.with_annotation("000001", context_frames=context)


def test_with_annotation_rejects_unknown_or_duplicate_source_key(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    with pytest.raises(ValueError, match="exactly one"):
        bundle.with_annotation("000001")
    first = dict(bundle.annotations[0], annotation_id="000001")
    with pytest.raises(ValueError, match="exactly one"):
        replace(bundle, annotations=(first, first)).with_annotation("000001")


def test_with_annotation_rechecks_mapping_in_new_context(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    first = dict(bundle.annotations[0], annotation_id="000001")
    original = bundle.frame_maps["helper"]
    status, indices, timestamps = (
        original.status.copy(),
        original.vrs_rgb_frame_index.copy(),
        original.vrs_device_timestamp_ns.copy(),
    )
    status[5], indices[5], timestamps[5] = "UNRESOLVED", -1, -1
    changed = replace(
        original, status=status, vrs_rgb_frame_index=indices, vrs_device_timestamp_ns=timestamps
    )
    bundle = replace(
        bundle, annotations=(first,), frame_maps={**bundle.frame_maps, "helper": changed}
    )
    assert bundle.with_annotation("000001", context_frames=0).end_index == 3
    with pytest.raises(ValueError, match="VERIFIED"):
        bundle.with_annotation("000001", context_frames=2)


def test_with_annotation_rejects_bounds_outside_video(playback_cache):
    root, path, _, _ = playback_cache
    bundle = load_playback(root, path, RECORDING)
    bad = dict(bundle.annotations[0], annotation_id="000001", end_frame=6)
    with pytest.raises(ValueError, match="boundaries"):
        replace(bundle, annotations=(bad,)).with_annotation("000001")


def test_changed_derived_payload_rejected_by_hash(playback_cache):
    root, path, directory, _ = playback_cache
    with (directory / "frame_validation/helper_aligned.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        load_playback(root, path, RECORDING)


def rewrite_aligned(cache, role, mutation):
    _, path, directory, report = cache
    aligned = directory / "frame_validation" / f"{role}_aligned.npz"
    with np.load(aligned) as source:
        arrays = {key: source[key] for key in source.files}
    mutation(arrays)
    np.savez_compressed(aligned, **arrays)
    report["aligned_cache_identities"][role]["sha256"] = digest(aligned)
    path.write_text(json.dumps(report))


def test_generation_mismatch_rejected_even_with_updated_digest(playback_cache):
    root, path, _, _ = playback_cache
    rewrite_aligned(
        playback_cache, "helper", lambda arrays: arrays.update(generation_id=np.asarray("stale"))
    )
    with pytest.raises(ValueError, match="different validation generation"):
        load_playback(root, path, RECORDING)


def test_aligned_mapping_identity_must_equal_current_map(playback_cache):
    root, path, _, _ = playback_cache

    def mutate(arrays):
        arrays["device_timestamp_ns"][2] += 1

    rewrite_aligned(playback_cache, "helper", mutate)
    with pytest.raises(ValueError, match="differs from current"):
        load_playback(root, path, RECORDING)


def test_accepted_mask_cannot_override_temporal_gap(playback_cache):
    root, path, _, _ = playback_cache

    def mutate(arrays):
        arrays["pose_residual_ns"][2] = 10_000_000

    rewrite_aligned(playback_cache, "helper", mutate)
    with pytest.raises(ValueError, match="residual gate"):
        load_playback(root, path, RECORDING)


def test_no_verified_selection_rejected(playback_cache):
    root, path, _, report = playback_cache
    report["handover_ranking"]["selected_annotation_index"] = None
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="no handover"):
        load_playback(root, path, RECORDING)


def test_explicit_annotation_can_select_when_ranked_context_is_unavailable(playback_cache):
    root, path, _, report = playback_cache
    ranking = report["handover_ranking"]
    ranking["selected_annotation_index"] = None
    ranking["selection_status"] = "NO_VERIFIED_CONTEXT"
    ranking["handovers"][0].update(annotation_id="000001", eligible_verified_context=False)
    path.write_text(json.dumps(report))
    bundle = load_playback(root, path, RECORDING, annotation_id="000001", context_frames=0)
    assert (bundle.start_index, bundle.end_index) == (1, 3)
    assert bundle.selection["eligible_verified_context"] is True
    assert [frame.frame_index for frame in bundle.iter_frames()] == [1, 2, 3]


@pytest.mark.parametrize("context", [-1, True, 1.5])
def test_explicit_annotation_rejects_invalid_context(playback_cache, context):
    root, path, _, _ = playback_cache
    with pytest.raises(ValueError, match="context_frames"):
        load_playback(root, path, RECORDING, annotation_id="000001", context_frames=context)


def test_explicit_annotation_still_checks_payload_hash(playback_cache):
    root, path, directory, report = playback_cache
    report["handover_ranking"]["handovers"][0]["annotation_id"] = "000001"
    path.write_text(json.dumps(report))
    (directory / "frame_validation/helper_aligned.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        load_playback(root, path, RECORDING, annotation_id="000001", context_frames=0)


def test_explicit_annotation_rejects_unknown_id(playback_cache):
    root, path, _, _ = playback_cache
    with pytest.raises(ValueError, match="exactly one"):
        load_playback(root, path, RECORDING, annotation_id="000001", context_frames=0)


def test_wrong_participant_clock_rejected(playback_cache):
    root, path, _, report = playback_cache
    report["participants"]["helper"]["device_clock"] = device_clock(RECORDING, "leader").name
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="wrong local device clock"):
        load_playback(root, path, RECORDING)


def test_raw_cache_path_is_rejected_without_reading_it(playback_cache):
    root, path, _, report = playback_cache
    report["aligned_cache_identities"]["helper"]["path"] = str(root / "data/raw/file.npz")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="outside data/raw"):
        load_playback(root, path, RECORDING)


def test_verified_export_calibration_enables_camera_only_when_complete(playback_cache):
    root, path, directory, _ = playback_cache
    for role in ("helper", "leader"):
        rays = np.asarray([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]]) / np.sqrt(2)
        (directory / "vrs" / f"{role}_export_rgb_calibration.json").write_text(
            json.dumps(
                {
                    "recording_id": RECORDING,
                    "participant": role,
                    "status": "VERIFIED",
                    "camera_frame": f"{role}/camera-rgb",
                    "device_frame": f"{role}/device",
                    "T_device_camera": np.eye(4).tolist(),
                    "resolution": [4, 4],
                    "boundary_rays_camera": rays.tolist(),
                    "projection_model": "synthetic",
                    "verification": "synthetic calibrated rays",
                    "provenance_source": "fixture",
                    "provenance_detail": "synthetic test",
                }
            )
        )
    bundle = load_playback(root, path, RECORDING)
    _, status = assemble_index_frame(next(bundle.iter_frames()), bundle.bindings)
    assert status["participants"]["helper"]["camera_present"]
    assert status["participants"]["leader"]["frustum_verified"]
