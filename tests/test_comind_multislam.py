"""Tiny synthetic Multi-SLAM files; no real recording data is accessed."""

import csv
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from duet.adapters.comind.multislam import (
    SlamSource,
    iter_shared_trajectory,
    parse_vrs_mapping,
    trajectory_graph_uids,
    verify_shared_world,
)
from duet.schemas.common import FrameId, ParticipantId, Provenance

RECORDING_ID = "00000000-0000-0000-0000-000000000001"


def _csv_text(graph_uids):
    output = io.StringIO()
    fields = [
        "graph_uid",
        "tracking_timestamp_us",
        "utc_timestamp_ns",
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
        "quality_score",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for index, uid in enumerate(graph_uids):
        writer.writerow(dict(zip(fields, [uid, 1_000_000 + index, -1, 1, 2, 3, 0, 0, 0, 1, 0.8])))
    return output.getvalue()


def _source(tmp_path, role, graph_uids, *, zipped=False):
    if zipped:
        archive = tmp_path / f"{role}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr("closed_loop_trajectory.csv", _csv_text(graph_uids))
        return SlamSource(archive=archive)
    directory = tmp_path / role
    directory.mkdir(exist_ok=True)
    (directory / "closed_loop_trajectory.csv").write_text(_csv_text(graph_uids), encoding="utf-8")
    return SlamSource(directory=directory)


def _mapping(**updates):
    result = {
        f"{RECORDING_ID}/trimmed_vrs/helper_trimmed.vrs": "0",
        f"{RECORDING_ID}/trimmed_vrs/leader_trimmed.vrs": "1",
    }
    result.update(updates)
    return result


def test_parse_verified_mapping_shape_without_order_inference(tmp_path):
    path = tmp_path / "vrs_to_multi_slam.json"
    path.write_text(json.dumps(dict(reversed(list(_mapping().items())))), encoding="utf-8")
    assert parse_vrs_mapping(path, recording_id=RECORDING_ID) == {"helper": "0", "leader": "1"}


@pytest.mark.parametrize("identifier", ["../1", "/1", 1, "１", "", "0"])
def test_mapping_rejects_unsafe_or_duplicate_output_ids(tmp_path, identifier):
    mapping = _mapping()
    mapping[f"{RECORDING_ID}/trimmed_vrs/leader_trimmed.vrs"] = identifier
    path = tmp_path / "vrs_to_multi_slam.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_vrs_mapping(path, recording_id=RECORDING_ID)


def test_mapping_does_not_guess_role_from_arbitrary_basename(tmp_path):
    path = tmp_path / "vrs_to_multi_slam.json"
    path.write_text(json.dumps({"helper_trimmed.vrs": "0", "leader_trimmed.vrs": "1"}))
    with pytest.raises(ValueError, match="verified mapping key"):
        parse_vrs_mapping(path, recording_id=RECORDING_ID)


def test_duplicate_mapping_keys_are_rejected(tmp_path):
    path = tmp_path / "vrs_to_multi_slam.json"
    key = f"{RECORDING_ID}/trimmed_vrs/helper_trimmed.vrs"
    path.write_text('{"' + key + '":"0","' + key + '":"1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate VRS mapping key"):
        parse_vrs_mapping(path, recording_id=RECORDING_ID)


def test_directory_and_flat_zip_trajectories_share_explicit_world(tmp_path):
    sources = {
        "helper": _source(tmp_path, "helper", ["shared-graph", "shared-graph"], zipped=True),
        "leader": _source(tmp_path, "leader", ["shared-graph"]),
    }
    verification = verify_shared_world(sources, recording_id=RECORDING_ID)
    assert verification.graph_uid == "shared-graph"
    assert verification.world_frame == FrameId(f"comind/{RECORDING_ID}/multislam/shared-graph")
    helper = tuple(iter_shared_trajectory(verification, participant="helper"))
    leader = tuple(iter_shared_trajectory(verification, participant="leader"))
    assert helper[0].transform.destination == leader[0].transform.destination
    assert helper[0].transform.source == FrameId("helper/device")
    assert helper[0].participant_id == ParticipantId("participant/helper")
    assert helper[0].timestamp.clock_domain != leader[0].timestamp.clock_domain
    np.testing.assert_array_equal(helper[0].transform.matrix[:3, 3], [1, 2, 3])
    assert helper[0].utc_timestamp_ns_raw == -1
    assert helper[0].utc_timestamp.raw_value is None


@pytest.mark.parametrize(
    ("helper_graphs", "leader_graphs"),
    [(["one"], ["two"]), (["one", "two"], ["one"]), (["one", "two"], ["one", "two"])],
)
def test_reject_different_or_multiple_graph_islands(tmp_path, helper_graphs, leader_graphs):
    sources = {
        "helper": _source(tmp_path, "helper", helper_graphs),
        "leader": _source(tmp_path, "leader", leader_graphs),
    }
    with pytest.raises(ValueError, match="one shared world"):
        verify_shared_world(sources, recording_id=RECORDING_ID)


@pytest.mark.parametrize("graph_uids", [[], [""], [" "]])
def test_reject_empty_or_missing_graph_evidence(tmp_path, graph_uids):
    source = _source(tmp_path, "helper", graph_uids)
    with pytest.raises(ValueError):
        trajectory_graph_uids(source)


def test_shared_world_requires_both_roles(tmp_path):
    source = _source(tmp_path, "helper", ["one"])
    with pytest.raises(ValueError, match="helper and leader"):
        verify_shared_world({"helper": source}, recording_id=RECORDING_ID)


def test_existing_directory_preferred_to_archive_and_recovery_evidence_retained(tmp_path):
    source = _source(tmp_path, "helper", ["directory-graph"])
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a ZIP")
    evidence = Provenance("synthetic recovery", "CRC checked")
    selected = SlamSource(source.directory, archive, evidence)
    assert trajectory_graph_uids(selected) == frozenset({"directory-graph"})
    assert selected.provenance.parents == (evidence,)


@pytest.mark.parametrize("names", [["slam/closed_loop_trajectory.csv"], ["unrelated.csv"]])
def test_zip_does_not_guess_nested_or_other_trajectory_name(tmp_path, names):
    archive = tmp_path / "slam.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name in names:
            stream.writestr(name, _csv_text(["one"]))
    with pytest.raises(ValueError, match="exactly one"):
        trajectory_graph_uids(SlamSource(archive=archive))


def test_graph_change_after_verification_is_rejected(tmp_path):
    sources = {role: _source(tmp_path, role, ["one"]) for role in ("helper", "leader")}
    verification = verify_shared_world(sources, recording_id=RECORDING_ID)
    path = sources["helper"].directory / "closed_loop_trajectory.csv"
    path.write_text(_csv_text(["two"]), encoding="utf-8")
    with pytest.raises(ValueError, match="graph changed"):
        tuple(iter_shared_trajectory(verification, participant="helper"))


@pytest.mark.parametrize(
    "contents",
    [
        "graph_uid,graph_uid\none,two\n",
        "graph_uid,other\none\n",
        "graph_uid\none,extra\n",
    ],
)
def test_graph_audit_rejects_ambiguous_csv_headers_and_row_widths(tmp_path, contents):
    source = _source(tmp_path, "helper", ["one"])
    (source.directory / "closed_loop_trajectory.csv").write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="unique names|row width"):
        trajectory_graph_uids(source)


@pytest.mark.parametrize("change", ["world", "recording", "graph", "role", "source"])
def test_verification_constructor_rejects_inconsistent_recording_graph_and_sources(
    tmp_path, change
):
    sources = {role: _source(tmp_path, role, ["one"]) for role in ("helper", "leader")}
    verification = verify_shared_world(sources, recording_id=RECORDING_ID)
    updates = {
        "world": {"world_frame": FrameId("unverified-world")},
        "recording": {"recording_id": "00000000-0000-0000-0000-000000000002"},
        "graph": {"graph_uid": "different-graph"},
        "role": {"sources": {"helper": sources["helper"]}},
        "source": {"sources": {"helper": sources["helper"], "leader": "not-a-source"}},
    }
    with pytest.raises(ValueError):
        replace(verification, **updates[change])
