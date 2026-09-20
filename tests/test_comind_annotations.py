"""Synthetic annotation tests; no dataset file or raw directory is accessed."""

import json
from decimal import Decimal

import pytest

from duet.adapters.comind.annotations import (
    load_handover_annotations,
    parse_handover_recording,
    parse_handover_segments,
)
from duet.schemas.common import Provenance
from duet.schemas.time import TimeUnit

RECORDING_UUID = "00000000-0000-0000-0000-000000000001"
OTHER_UUID = "00000000-0000-0000-0000-000000000002"
PROVENANCE = Provenance(source="synthetic handover fixture")


def segment(**updates):
    data = {
        "skip": False,
        "start_frame": 15,
        "end_frame": 31,
        "start_time": Decimal("1.234567890123456789"),
        "end_time": Decimal("1.900000000000000001"),
        "initiator": "left",
        "delivering_flow": "rtl",
        "initiation_type": ["verbal", "gestural"],
        "object_category_level_1": "bowl",
        "object_category_level_2": ["synthetic-category"],
        "object_category_level_3": None,
        "numeric_id": 99,
        "bbox": [1, 2, 3, 4],
    }
    data.update(updates)
    return data


def parse(segments):
    return parse_handover_segments(segments, recording_uuid=RECORDING_UUID, provenance=PROVENANCE)


def test_only_literal_false_skip_is_usable():
    missing_skip = segment()
    del missing_skip["skip"]
    result = parse(
        [segment(), segment(skip=True), segment(skip=0), segment(skip="false"), missing_skip]
    )
    assert len(result) == 1


def test_preserve_verified_values_and_uninterpreted_source_labels():
    annotation = parse([segment()])[0]
    assert annotation.recording_id == RECORDING_UUID
    assert annotation.annotation_id is None  # numeric_id semantics have not been established.
    assert (annotation.start_frame, annotation.end_frame) == (15, 31)
    assert annotation.start_time.raw_value == Decimal("1.234567890123456789")
    assert annotation.end_time.raw_value == Decimal("1.900000000000000001")
    assert annotation.start_time.unit is TimeUnit.SECONDS
    assert annotation.start_time.clock_domain.name == (
        f"comind:{RECORDING_UUID}:handover_annotations"
    )
    assert annotation.initiator == "left"
    assert annotation.delivering_flow == "rtl"
    assert annotation.initiation_type == ["verbal", "gestural"]
    assert annotation.object_category_level_1 == "bowl"
    assert annotation.object_category_level_2 == ["synthetic-category"]
    assert annotation.object_category_level_3 is None
    assert annotation.source_fields == {"numeric_id": 99, "bbox": [1, 2, 3, 4], "skip": False}
    assert annotation.provenance.parents == (PROVENANCE,)
    assert not hasattr(annotation, "participant_id")


def test_labels_are_not_restricted_to_observed_vocabulary():
    annotation = parse([segment(initiator="source-label", delivering_flow="unknown-flow")])[0]
    assert annotation.initiator == "source-label"
    assert annotation.delivering_flow == "unknown-flow"


def test_missing_boundaries_remain_missing():
    annotation = parse([{"skip": False}])[0]
    assert annotation.start_frame is None
    assert annotation.end_frame is None
    assert annotation.start_time.raw_value is None
    assert annotation.end_time.raw_value is None
    assert annotation.initiator is None


def test_recordings_have_distinct_unmapped_clock_domains():
    first = parse([segment()])[0]
    second = parse_handover_segments([segment()], recording_uuid=OTHER_UUID, provenance=PROVENANCE)[
        0
    ]
    assert first.start_time.clock_domain != second.start_time.clock_domain


def test_loader_accepts_explicit_alternative_recording_layout_selector(tmp_path):
    # This container is intentionally synthetic, not a claim about CoMind layout.
    path = tmp_path / "dataset_handover_consolidated.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version": "synthetic",
                "file_timestamp": "uninterpreted synthetic timestamp",
                "data": {
                    RECORDING_UUID: {"synthetic_container": [{"skip": False}]},
                    OTHER_UUID: {"synthetic_container": [{"skip": True}]},
                },
            }
        ),
        encoding="utf-8",
    )
    seen = []

    def select(recording):
        seen.append(recording)
        return recording["synthetic_container"]

    annotations = load_handover_annotations(
        path, recording_uuid=RECORDING_UUID, segment_selector=select
    )
    assert seen == [{"synthetic_container": [{"skip": False}]}]
    assert len(annotations) == 1
    assert annotations[0].provenance.source == str(path)
    with pytest.raises(TypeError, match="segment objects"):
        load_handover_annotations(path, recording_uuid=RECORDING_UUID)


def test_loader_preserves_decimal_json_timestamp_precision(tmp_path):
    path = tmp_path / "dataset_handover_consolidated.json"
    path.write_text(
        '{"data":{"' + RECORDING_UUID + '":[{"skip":false,'
        '"start_time":1.234567890123456789,"end_time":1.900000000000000001}]}}',
        encoding="utf-8",
    )
    annotation = load_handover_annotations(
        path, recording_uuid=RECORDING_UUID, segment_selector=lambda recording: recording
    )[0]
    assert annotation.start_time.raw_value == Decimal("1.234567890123456789")
    assert annotation.end_time.raw_value == Decimal("1.900000000000000001")


@pytest.mark.parametrize("document", [[], {}, {"data": []}])
def test_loader_rejects_unverified_root_layout(tmp_path, document):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TypeError):
        load_handover_annotations(
            path, recording_uuid=RECORDING_UUID, segment_selector=lambda recording: recording
        )


def test_loader_reports_missing_recording(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text('{"data":{}}', encoding="utf-8")
    with pytest.raises(KeyError, match="not present"):
        load_handover_annotations(
            path, recording_uuid=RECORDING_UUID, segment_selector=lambda recording: recording
        )


@pytest.mark.parametrize("text", ['{"data":{},"data":{}}', '{"data":NaN}', "not json"])
def test_loader_rejects_invalid_or_ambiguous_json(tmp_path, text):
    path = tmp_path / "annotations.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_handover_annotations(
            path, recording_uuid=RECORDING_UUID, segment_selector=lambda recording: recording
        )


@pytest.mark.parametrize("segments", [{"unverified_wrapper": []}, "not a sequence", [1]])
def test_parser_rejects_unverified_segment_layout(segments):
    with pytest.raises(TypeError):
        parse(segments)


@pytest.mark.parametrize(
    "updates",
    [{"start_time": "1.0"}, {"start_frame": True}, {"initiator": ["left"]}],
)
def test_parser_rejects_invalid_verified_field_types(updates):
    with pytest.raises(ValueError):
        parse([segment(**updates)])


def test_parser_does_not_propagate_unverified_extra_fields():
    annotation = parse([segment(guessed_participant_id="participant_a")])[0]
    assert "guessed_participant_id" not in annotation.source_fields


def test_empty_explicit_segments_are_supported():
    assert parse([]) == ()


def test_recording_identifier_must_be_uuid():
    with pytest.raises(ValueError, match="UUID"):
        parse_handover_segments([], recording_uuid="synthetic", provenance=PROVENANCE)


def test_native_recording_keeps_opaque_keys_original_order_and_labels():
    # Keys deliberately do not match frames: they must not manufacture times.
    annotations = parse_handover_recording(
        {"000099": segment(), "000001": segment(initiator="right"), "000050": segment(skip=True)},
        recording_uuid=RECORDING_UUID,
        provenance=PROVENANCE,
    )
    assert [item.annotation_id for item in annotations] == ["000099", "000001"]
    assert [item.start_frame for item in annotations] == [15, 15]
    assert [item.initiator for item in annotations] == ["left", "right"]
    assert annotations[0].start_time.raw_value == Decimal("1.234567890123456789")
    assert "source annotation key '000099'" in annotations[0].provenance.parents[0].detail


def test_loader_defaults_to_verified_native_recording_layout(tmp_path):
    path = tmp_path / "dataset_handover_consolidated.json"
    path.write_text(
        json.dumps(
            {
                "data": {
                    RECORDING_UUID: {
                        "002131": {"skip": False, "start_time": 71.0, "initiator": "right"},
                        "007676": {"skip": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result = load_handover_annotations(path, recording_uuid=RECORDING_UUID)
    assert len(result) == 1
    assert result[0].annotation_id == "002131"
    assert result[0].start_time.raw_value == Decimal("71.0")
    assert result[0].start_frame is None
    assert result[0].initiator == "right"


@pytest.mark.parametrize("recording", [[], "text", {"key": []}, {"": {"skip": False}}])
def test_native_recording_rejects_wrong_shapes(recording):
    with pytest.raises((ValueError, TypeError)):
        parse_handover_recording(recording, recording_uuid=RECORDING_UUID, provenance=PROVENANCE)
