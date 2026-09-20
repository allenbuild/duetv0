"""Unverified semantic evidence must never enable a synchronized layer."""

import json

import pytest

from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence


@pytest.mark.parametrize("status", [EvidenceStatus.UNRESOLVED, EvidenceStatus.INFERRED])
def test_nonverified_evidence_rejects_layer_enablement(status):
    claim = MappingEvidence(
        "mp4_to_device",
        status,
        "No per-frame export correspondence.",
        evidence=("synthetic metadata",),
        missing_assets=("mp4_to_vrs_time_ns.csv",),
    )
    assert not claim.verified
    with pytest.raises(ValueError, match="mp4_to_device"):
        claim.require_verified()


def test_verified_scope_and_sources_survive_json_serialization():
    claim = MappingEvidence(
        "native_camera_pose",
        EvidenceStatus.VERIFIED,
        "Synthetic native physical camera pose only; exported pixels remain unresolved.",
        evidence=("synthetic calibration fixture",),
    )
    claim.require_verified()
    serialized = json.loads(json.dumps(claim.to_dict()))
    assert serialized["status"] == "verified"
    assert serialized["evidence"] == ["synthetic calibration fixture"]
    assert "exported pixels remain unresolved" in serialized["statement"]
    assert not hasattr(claim, "map_timestamp")


@pytest.mark.parametrize(
    "updates",
    [
        {"status": "verified"},
        {"evidence": []},
        {"name": ""},
        {"evidence": ()},
        {"missing_assets": ("required.csv",)},
    ],
)
def test_evidence_rejects_ambiguous_or_contradictory_claims(updates):
    arguments = {
        "name": "mapping",
        "status": EvidenceStatus.VERIFIED,
        "statement": "Explicit scoped claim",
        "evidence": ("source",),
    }
    arguments.update(updates)
    with pytest.raises((ValueError, TypeError)):
        MappingEvidence(**arguments)
