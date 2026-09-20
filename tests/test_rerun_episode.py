"""GUI-free viewer invariants and small real RRD serialization checks."""

import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from duet.adapters.comind.video import VideoFrameMatch, VideoFrameSample
from duet.geometry.transforms import RigidTransform
from duet.qc.result import QCResult, QCStatus
from duet.schemas.common import (
    Confidence,
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
)
from duet.schemas.episode import CameraSample, HandoverAnnotation, HandSample
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit, VerifiedClockMapping
from duet.synchronization.matching import MatchStatus
from duet.visualization.rerun_episode import (
    EpisodeRenderFrame,
    ParticipantRenderFrame,
    VerifiedImageGeometry,
    ViewerBindings,
    active_annotations,
    assemble_frame,
    lookup_video_on_common,
    save_episode,
    save_spatial_diagnostic,
)

PROVENANCE = Provenance("synthetic renderer fixture")
WORLD = FrameId("synthetic/shared_world")
COMMON = ClockDomain("synthetic/common")
EPOCH_NS = 1_800_000_000_000_000_001
CLOCKS = {role: ClockDomain(f"synthetic/{role}/device") for role in ("helper", "leader")}
DEVICE_US = {"helper": 1_000_000, "leader": 9_000_000}


def timestamp(value, *, role=None, unit=TimeUnit.MICROSECONDS):
    clock = COMMON if role is None else CLOCKS[role]
    return Timestamp(value, unit, clock, PROVENANCE)


def binding(**overrides):
    mappings = {
        role: VerifiedClockMapping(
            CLOCKS[role],
            COMMON,
            1,
            Fraction(EPOCH_NS, 1_000_000_000) - Fraction(DEVICE_US[role], 1_000_000),
            PROVENANCE,
            "Explicit synthetic clock correspondence",
        )
        for role in CLOCKS
    }
    values = {
        "world_frame": WORLD,
        "common_clock": COMMON,
        "device_time_mappings": mappings,
        "max_gap_seconds": Fraction(1, 100),
    }
    values.update(overrides)
    return ViewerBindings(**values)


def pose(role, *, camera=False):
    matrix = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    if role == "leader":
        matrix[:3, 3] = [-4, 5, 6]
    source = FrameId(f"{role}/camera-rgb" if camera else f"{role}/device")
    return RigidTransform(matrix, source, WORLD, DistanceUnit.METERS, PROVENANCE)


def hand(role, side="left", *, missing=False, confidence=0.0):
    return HandSample(
        ParticipantId(f"participant/{role}"),
        side,
        timestamp(DEVICE_US[role], role=role),
        WORLD,
        None if missing else [[1, 2, 3], [4, 5, 6]],
        DistanceUnit.METERS,
        PROVENANCE,
        metadata=SampleMetadata(SampleState.MISSING, reason="synthetic missing hand")
        if missing
        else SampleMetadata(confidence=Confidence(confidence, PROVENANCE)),
        qc=(QCResult("synthetic_hand_qc", QCStatus.PASS, {"count": 2}, {"max_gap_s": 0.01}),),
    )


def camera(role):
    return CameraSample(
        ParticipantId(f"participant/{role}"),
        FrameId(f"{role}/camera-rgb"),
        timestamp(DEVICE_US[role], role=role),
        pose(role, camera=True),
        PROVENANCE,
        qc=(QCResult("synthetic_camera_qc", QCStatus.PASS, {}, {}),),
    )


def render_frame(*, common_ns=EPOCH_NS):
    return EpisodeRenderFrame(
        timestamp(common_ns, unit=TimeUnit.NANOSECONDS),
        tuple(
            ParticipantRenderFrame(
                role,
                timestamp(DEVICE_US[role], role=role),
                pose(role),
                camera(role),
                (hand(role), hand(role, "right", missing=True)),
            )
            for role in CLOCKS
        ),
    )


def updates(plan):
    return {update.path: update for update in plan.updates}


def geometry(role="helper", **overrides):
    rays = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) / np.sqrt(3)
    values = {
        "camera_frame": FrameId(f"{role}/camera-rgb"),
        "resolution": (16, 12),
        "boundary_rays_camera": rays,
        "projection_model": "synthetic verified nonlinear model",
        "verification": "Synthetic source rays and export orientation verified",
        "provenance": PROVENANCE,
    }
    values.update(overrides)
    return VerifiedImageGeometry(**values)


def test_assembly_preserves_common_nanoseconds_above_float_precision_and_asymmetric_transform():
    frame = render_frame()
    plan = assemble_frame(frame, binding())
    assert plan.common_nanoseconds == EPOCH_NS
    assert plan.status["common_timestamp_ns"] == EPOCH_NS
    update = updates(plan)["world/helper/device"]
    assert update.kind == "pose"
    np.testing.assert_array_equal(update.value.matrix, frame.participants[0].device_pose.matrix)
    assert plan.status["participants"]["helper"]["device_residual_seconds"] == 0
    assert plan.status["participants"]["leader"]["device_residual_seconds"] == 0


def test_zero_confidence_stays_present_and_missing_hands_are_explicitly_cleared():
    plan = assemble_frame(render_frame(), binding())
    result = updates(plan)
    assert result["world/helper/hands/left"].kind == "points"
    assert result["world/helper/hands/right"].kind == "clear"
    left = plan.status["participants"]["helper"]["hands"]["left"]
    assert left["present"] and left["confidence"] == 0
    assert left["source_state"] == "present"
    assert "synthetic_hand_qc" in str(left)


def test_gap_rejection_clears_all_current_geometry_without_reusing_previous_data():
    plan = assemble_frame(render_frame(common_ns=EPOCH_NS + 20_000_000), binding())
    assert all(update.kind == "clear" for update in plan.updates)
    assert not plan.status["participants"]["helper"]["device_present"]


@pytest.mark.parametrize("nanoseconds", [Fraction(1, 10), 2**63])
def test_unrepresentable_common_nanosecond_timeline_is_rejected(nanoseconds):
    if isinstance(nanoseconds, Fraction):
        query = Timestamp(0.0000000001, TimeUnit.SECONDS, COMMON, PROVENANCE)
    else:
        query = timestamp(nanoseconds, unit=TimeUnit.NANOSECONDS)
    with pytest.raises(ValueError, match="nanoseconds"):
        assemble_frame(replace(render_frame(), common_timestamp=query), binding())


def test_wrong_participant_clock_and_frame_are_rejected():
    frame = render_frame()
    helper, leader = frame.participants
    with pytest.raises(ValueError, match="source clock"):
        assemble_frame(
            replace(
                frame,
                participants=(replace(helper, device_timestamp=leader.device_timestamp), leader),
            ),
            binding(),
        )
    with pytest.raises(ValueError, match="incompatible"):
        assemble_frame(
            replace(frame, participants=(replace(helper, device_pose=leader.device_pose), leader)),
            binding(),
        )
    wrong_hand = replace(helper.hands[0], participant_id=ParticipantId("participant/leader"))
    with pytest.raises(ValueError, match="hand participant"):
        assemble_frame(
            replace(frame, participants=(replace(helper, hands=(wrong_hand,)), leader)), binding()
        )


def test_both_explicit_verified_device_mappings_are_required():
    with pytest.raises(ValueError, match="both participant"):
        binding(device_time_mappings={})
    with pytest.raises(TypeError, match="explicitly verified"):
        binding(device_time_mappings={"helper": object(), "leader": object()})


def test_camera_pose_can_render_while_frustum_remains_disabled_without_image_proof():
    plan = assemble_frame(render_frame(), binding())
    assert updates(plan)["world/helper/camera_rgb"].kind == "pose"
    assert updates(plan)["world/helper/camera_rgb/frustum"].kind == "clear"
    assert "unresolved" in plan.status["participants"]["helper"]["frustum"]


def test_verified_perimeter_rays_enable_frustum_without_pinhole_approximation():
    proof = geometry()
    plan = assemble_frame(render_frame(), binding(image_geometry={"helper": proof}))
    update = updates(plan)["world/helper/camera_rgb/frustum"]
    assert update.kind == "frustum"
    assert update.value is proof
    with pytest.raises(ValueError):
        proof.boundary_rays_camera.setflags(write=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"verification": ""},
        {"boundary_rays_camera": np.zeros((4, 3))},
        {"boundary_rays_camera": np.ones((4, 3), dtype=complex) * 1j},
        {"resolution": (True, 12)},
    ],
)
def test_frustum_rays_require_evidence_and_valid_geometry(overrides):
    with pytest.raises((ValueError, TypeError)):
        geometry(**overrides)


def test_frustum_rejects_wrong_camera_frame():
    with pytest.raises(ValueError, match="wrong camera frame"):
        assemble_frame(render_frame(), binding(image_geometry={"helper": geometry("leader")}))


class FakeVideo:
    def __init__(self, *, pts=512, time_base=Fraction(1, 15360), candidate_clock=None):
        self.clock_domain = ClockDomain("synthetic/video/helper")
        self.pts, self.time_base = pts, time_base
        self.candidate_clock = candidate_clock or self.clock_domain
        self.requests = []

    def nearest_frame(self, seconds, *, max_gap_seconds):
        self.requests.append((seconds, max_gap_seconds))
        image = SimpleNamespace(to_ndarray=lambda **_: np.zeros((12, 16, 3), dtype=np.uint8))
        sample = VideoFrameSample(self.pts, self.time_base, self.candidate_clock, PROVENANCE, image)
        residual = sample.seconds - seconds
        status = (
            MatchStatus.MATCHED if abs(residual) <= max_gap_seconds else MatchStatus.GAP_EXCEEDED
        )
        return VideoFrameMatch(status, sample, seconds, residual, max_gap_seconds)


def video_mapping(video, *, scale=3):
    return VerifiedClockMapping(
        video.clock_domain, COMMON, scale, 100, PROVENANCE, "Synthetic exact affine video map"
    )


def test_video_common_lookup_preserves_exact_pts_and_scales_gap_to_source_clock():
    video = FakeVideo()
    query = timestamp(100_100_000_000, unit=TimeUnit.NANOSECONDS)
    match, residual = lookup_video_on_common(
        video, query, video_mapping(video), max_gap_seconds=Fraction(1, 1000)
    )
    assert match.accepted and residual == 0
    assert match.candidate.pts == 512
    assert match.candidate.time_base == Fraction(1, 15360)
    assert video.requests == [(Fraction(1, 30), Fraction(1, 3000))]


def test_video_common_lookup_rejects_excessive_gap_and_wrong_decoded_clock():
    video = FakeVideo(pts=11, time_base=Fraction(1, 10))
    query = timestamp(102_100_000_000, unit=TimeUnit.NANOSECONDS)
    match, residual = lookup_video_on_common(
        video, query, video_mapping(video, scale=2), max_gap_seconds=Fraction(1, 100)
    )
    assert not match.accepted and residual == Fraction(1, 10)
    wrong = FakeVideo(candidate_clock=ClockDomain("wrong-decoder-clock"))
    with pytest.raises(ValueError, match="different clock"):
        lookup_video_on_common(wrong, query, video_mapping(wrong), max_gap_seconds=1)


def annotation(*, clock=None):
    clock = clock or ClockDomain("synthetic/annotations")
    return HandoverAnnotation(
        "synthetic-recording",
        "event-1",
        0,
        30,
        Timestamp(0, TimeUnit.SECONDS, clock, PROVENANCE),
        Timestamp(1, TimeUnit.SECONDS, clock, PROVENANCE),
        "left",
        "rtl",
        ["gestural"],
        "bowl",
        None,
        None,
        PROVENANCE,
    )


def annotation_binding(*, inclusive=False):
    mapping = VerifiedClockMapping(
        ClockDomain("synthetic/annotations"),
        COMMON,
        1,
        Fraction(EPOCH_NS, 1_000_000_000),
        PROVENANCE,
        "Synthetic verified annotation time map",
    )
    return binding(
        annotation_mapping=mapping,
        annotation_end_inclusive=inclusive,
        annotation_interval_verification="Synthetic explicit endpoint convention",
    )


def test_annotations_stay_disabled_without_mapping_and_require_endpoint_policy():
    query = render_frame().common_timestamp
    assert active_annotations([annotation()], query, binding()) == ()
    mapped = annotation_binding()
    with pytest.raises(ValueError, match="inclusion"):
        replace(mapped, annotation_end_inclusive=None)
    with pytest.raises(ValueError, match="verification"):
        replace(mapped, annotation_interval_verification=None)


def test_annotation_interval_boundaries_follow_explicit_policy():
    item = annotation()
    start = timestamp(EPOCH_NS, unit=TimeUnit.NANOSECONDS)
    end = timestamp(EPOCH_NS + 1_000_000_000, unit=TimeUnit.NANOSECONDS)
    assert active_annotations([item], start, annotation_binding()) == (item,)
    assert active_annotations([item], end, annotation_binding(inclusive=False)) == ()
    assert active_annotations([item], end, annotation_binding(inclusive=True)) == (item,)


def test_annotation_cannot_bypass_mapping_by_claiming_the_common_clock():
    with pytest.raises(ValueError, match="clock"):
        active_annotations(
            [annotation(clock=COMMON)], render_frame().common_timestamp, annotation_binding()
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "scene_registration_verification": "synthetic",
            "scene_unit": DistanceUnit.CENTIMETERS,
            "scene_frame": WORLD,
        },
        {
            "scene_registration_verification": "synthetic",
            "scene_unit": DistanceUnit.METERS,
            "scene_frame": FrameId("other"),
        },
    ],
)
def test_scene_requires_registration_evidence_and_explicit_shared_meter_frame(tmp_path, options):
    with pytest.raises((ValueError, TypeError)):
        save_episode(
            tmp_path / "blocked.rrd",
            [],
            bindings=binding(),
            scene_points_world=[[0, 0, 0]],
            **options,
        )


def test_complex_scene_and_trajectory_are_not_silently_cast_to_real(tmp_path):
    with pytest.raises(ValueError, match="real"):
        save_episode(
            tmp_path / "scene.rrd",
            [],
            bindings=binding(),
            scene_points_world=[[1j, 0, 0]],
            scene_frame=WORLD,
            scene_unit=DistanceUnit.METERS,
            scene_registration_verification="Synthetic verified scene",
        )
    with pytest.raises(ValueError, match="real"):
        save_spatial_diagnostic(
            tmp_path / "trajectory.rrd",
            world_frame=WORLD,
            trajectories={"helper": [[1j, 0, 0]]},
            status={},
            provenance=PROVENANCE,
        )


def rerun_cli():
    executable = Path(sys.executable).parent / "rerun"
    if not executable.is_file():
        pytest.skip("Rerun CLI is unavailable in this Python environment")
    return str(executable)


@pytest.mark.parametrize("mode", ["episode", "diagnostic"])
def test_tiny_real_rrd_can_be_verified_without_gui(tmp_path, mode):
    path = tmp_path / f"{mode}.rrd"
    trajectories = {"helper": [[1, 2, 3], [1.1, 2.1, 3.1]], "leader": [[-4, 5, 6]]}
    if mode == "episode":
        save_episode(
            path,
            [render_frame(), render_frame(common_ns=EPOCH_NS + 20_000_000)],
            bindings=binding(image_geometry={"helper": geometry()}),
            trajectories=trajectories,
            scene_points_world=[[0, 0, 0]],
            scene_unit=DistanceUnit.METERS,
            scene_frame=WORLD,
            scene_registration_verification="Verified synthetic scene identity",
        )
    else:
        save_spatial_diagnostic(
            path,
            world_frame=WORLD,
            trajectories=trajectories,
            status={"shared_time_verified": False},
            provenance=PROVENANCE,
        )
    assert path.stat().st_size > 0
    checked = subprocess.run(
        [rerun_cli(), "rrd", "verify", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    printed = subprocess.run(
        [rerun_cli(), "rrd", "print", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    assert "world/helper/trajectory" in printed.stdout


@pytest.mark.parametrize("second_ns", [EPOCH_NS, EPOCH_NS - 1])
def test_stream_requires_strictly_increasing_common_timestamps(tmp_path, second_ns):
    with pytest.raises(ValueError, match="increasing"):
        save_episode(
            tmp_path / "unordered.rrd",
            [render_frame(), render_frame(common_ns=second_ns)],
            bindings=binding(),
        )
