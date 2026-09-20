"""Offline view tests require only synthetic geometry, with no datasets or GUI."""

import numpy as np
import pytest

from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, Provenance
from duet.visualization.demo_geometry import BACKGROUND, evaluate_view, fit_view, render_world
from duet.visualization.rerun_episode import EntityUpdate, VerifiedImageGeometry

PROVENANCE = Provenance("synthetic geometry")
WORLD = FrameId("synthetic/shared")


def snapshot(step=0, *, world=WORLD):
    updates = []
    for role, x in (("helper", -0.4), ("leader", 0.4)):
        for entity, source in (("device", "device"), ("camera_rgb", "camera-rgb")):
            matrix = np.eye(4)
            matrix[:3, 3] = [x + step * 0.2, 0.04 if entity == "camera_rgb" else 0, 1.2]
            pose = RigidTransform(
                matrix, FrameId(f"{role}/{source}"), world, DistanceUnit.METERS, PROVENANCE
            )
            updates.append(EntityUpdate(f"world/{role}/{entity}", "pose", pose))
        rays = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]], dtype=float)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        geometry = VerifiedImageGeometry(
            FrameId(f"{role}/camera-rgb"),
            (640, 480),
            rays,
            "synthetic nonlinear rays",
            "verified synthetic calibration",
            PROVENANCE,
        )
        updates.append(EntityUpdate(f"world/{role}/camera_rgb/frustum", "frustum", geometry))
        for side, shift in (("left", -0.1), ("right", 0.1)):
            points = np.array([[x + step * 0.3, shift, 0.7], [x + step * 0.3 + 0.1, shift, 0.72]])
            updates.append(EntityUpdate(f"world/{role}/hands/{side}", "points", points))
    return tuple(updates)


def paths(snapshots):
    return {
        role: np.array(
            [
                next(
                    update.value.matrix[:3, 3]
                    for update in frame
                    if update.path == f"world/{role}/device"
                )
                for frame in snapshots
            ]
        )
        for role in ("helper", "leader")
    }


def test_fit_is_deterministic_and_all_clip_geometry_is_in_view():
    snapshots = [snapshot(0), snapshot(1), snapshot(2)]
    first, second = fit_view(snapshots, (800, 600)), fit_view(snapshots, (800, 600))
    np.testing.assert_array_equal(first.basis, second.basis)
    np.testing.assert_array_equal(first.world_center, second.world_center)
    assert first.pixels_per_meter == second.pixels_per_meter
    projected = first.project(first.fitted_points)
    assert np.all(projected >= first.padding_pixels - 1e-8)
    assert np.all(projected <= np.array(first.size) - 1 - first.padding_pixels + 1e-8)
    with pytest.raises(ValueError):
        first.basis.setflags(write=True)


def test_render_returns_deterministic_rgb_at_requested_size():
    snapshots = [snapshot(0), snapshot(1)]
    view = fit_view(snapshots, (640, 480))
    first = render_world(snapshots[0], view, paths(snapshots))
    second = render_world(snapshots[0], view, paths(snapshots))
    assert first.mode == "RGB"
    assert first.size == (640, 480)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert np.any(np.asarray(first) != BACKGROUND)


def test_unresolved_scene_body_grid_and_objects_never_affect_fit_or_render():
    frame = snapshot()
    unverified = tuple(
        EntityUpdate(path, "points", [[1e6, 1e6, 1e6]])
        for path in ("world/scene", "world/helper/body", "world/object", "world/grid")
    )
    original, expanded = fit_view([frame], (640, 480)), fit_view([frame + unverified], (640, 480))
    np.testing.assert_array_equal(original.fitted_points, expanded.fitted_points)
    np.testing.assert_array_equal(
        np.asarray(render_world(frame, original, {})),
        np.asarray(render_world(frame + unverified, expanded, {})),
    )


def test_clear_updates_do_not_retain_stale_geometry():
    frame = snapshot()
    view = fit_view([frame], (640, 480))
    cleared = tuple(EntityUpdate(update.path, "clear", update.value) for update in frame)
    image = np.asarray(render_world(cleared, view, {}))
    assert np.all(image == BACKGROUND)


def test_unverified_or_wrong_camera_frustum_is_rejected():
    frame = snapshot()
    invalid = tuple(
        EntityUpdate(update.path, "frustum", object()) if update.kind == "frustum" else update
        for update in frame
    )
    with pytest.raises(ValueError, match="verified"):
        fit_view([invalid], (640, 480))
    missing_camera = tuple(update for update in frame if update.path != "world/helper/camera_rgb")
    with pytest.raises(ValueError, match="matching camera"):
        fit_view([missing_camera], (640, 480))


def test_unselected_trajectory_positions_rejected_and_nan_gaps_allowed():
    frames = [snapshot(0), snapshot(1)]
    view = fit_view(frames, (640, 480))
    with pytest.raises(ValueError, match="selected fitted snapshots"):
        render_world(frames[0], view, {"helper": np.array([[10, 20, 30]])})
    path = paths(frames)["helper"]
    render_world(frames[0], view, {"helper": np.vstack((path[:1], [[np.nan] * 3], path[1:]))})
    with pytest.raises(ValueError, match="all-NaN"):
        render_world(frames[0], view, {"helper": np.array([[np.nan, 0, 0]])})


def test_mixed_worlds_empty_fit_and_bad_dimensions_rejected():
    with pytest.raises(ValueError, match="world frames"):
        fit_view([snapshot(), snapshot(world=FrameId("other/world"))], (640, 480))
    with pytest.raises(ValueError, match="without verified geometry"):
        fit_view([], (640, 480))
    with pytest.raises(ValueError, match="dimensions"):
        fit_view([snapshot()], (True, 480))


def test_view_evaluation_reports_bounds_missing_layers_and_pixel_footprints():
    frame = snapshot()
    partly_missing = tuple(
        EntityUpdate(update.path, "clear", update.value)
        if update.path == "world/helper/hands/right"
        else update
        for update in frame
    )
    view = fit_view([frame, partly_missing], (912, 606))
    result = evaluate_view([frame, partly_missing], view)
    assert result["outside_panel_point_count"] == 0
    assert result["minimum_conservative_glyph_edge_margin_pixels"] > 50
    assert result["hands"]["helper/right"]["missing_count"] == 1
    assert result["hands"]["helper/left"]["bbox_diagonal_pixels"]["min"] > 0
    assert result["pose_present_counts"]["leader"] == {"device": 2, "camera_rgb": 2, "frustum": 2}
    assert not result["scene_drawn"]
    assert not result["hand_connectivity_drawn"]


def test_view_evaluation_detects_geometry_outside_a_different_clip_fit():
    view = fit_view([snapshot()], (912, 606))
    result = evaluate_view([snapshot(100)], view)
    assert result["outside_panel_point_count"] > 0
    assert result["minimum_conservative_glyph_edge_margin_pixels"] < 0


def test_empty_evaluation_has_no_invented_bounds():
    result = evaluate_view([], fit_view([snapshot()], (912, 606)))
    assert result["projected_geometry_bounds_pixels"] is None
    assert result["minimum_point_edge_margin_pixels"] is None
    assert result["hands"]["leader/left"]["bbox_diagonal_pixels"]["count"] == 0
