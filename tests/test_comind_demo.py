"""Synthetic demo panels and annotation timing; no recordings or GUI access."""

import numpy as np
import pytest
from PIL import Image

from duet.visualization import demo_compositor
from duet.visualization.demo_compositor import (
    HELPER_PANEL,
    LEADER_PANEL,
    WORLD_PANEL,
    annotation_active,
    compose_frame,
    make_thumbnail,
)


@pytest.mark.parametrize(
    ("frame_index", "expected"),
    ((18239, False), (18240, True), (18246, True), (18252, True), (18253, False)),
)
def test_selected_handover_badge_has_inclusive_global_frame_boundaries(frame_index, expected):
    assert annotation_active(frame_index, 18240, 18252) is expected


def synthetic_panels():
    helper = np.full((40, 40, 3), [201, 19, 33], dtype=np.uint8)
    leader = np.full((40, 40, 3), [25, 41, 203], dtype=np.uint8)
    world = Image.new("RGB", WORLD_PANEL[2:], (31, 199, 57))
    return helper, world, leader


def compose(helper, world, leader, *, thumbnail=False):
    return compose_frame(
        helper,
        world,
        leader,
        frame_index=18246,
        start_frame=18120,
        end_frame=18372,
        annotation_start=18240,
        annotation_end=18252,
        fps=30,
        thumbnail=thumbnail,
    )


@pytest.mark.parametrize("thumbnail", (False, True))
def test_composed_frame_is_full_hd_and_preserves_all_three_panel_order(thumbnail):
    panels = synthetic_panels()
    result = compose(*panels, thumbnail=thumbnail)
    assert isinstance(result, Image.Image)
    assert result.size == (1920, 1080)
    assert result.mode == "RGB"
    pixels = np.asarray(result)
    horizontal_centers = []
    for color in ([201, 19, 33], [31, 199, 57], [25, 41, 203]):
        mask = np.all(pixels == color, axis=2)
        assert mask.sum() > 10_000
        horizontal_centers.append(float(np.nonzero(mask)[1].mean()))
    assert horizontal_centers[0] < horizontal_centers[1] < horizontal_centers[2]


def test_composition_does_not_modify_source_images():
    helper, world, leader = synthetic_panels()
    helper_before, leader_before, world_before = helper.copy(), leader.copy(), world.tobytes()
    compose(helper, world, leader)
    np.testing.assert_array_equal(helper, helper_before)
    np.testing.assert_array_equal(leader, leader_before)
    assert world.tobytes() == world_before


@pytest.mark.parametrize(
    "frame_index, expected", [(18239, False), (18240, True), (18252, True), (18253, False)]
)
def test_composed_badge_is_absent_outside_selected_annotation(frame_index, expected):
    result = compose_frame(
        *synthetic_panels(),
        frame_index=frame_index,
        start_frame=18120,
        end_frame=18372,
        annotation_start=18240,
        annotation_end=18252,
    )
    badge_region = np.asarray(result)[910:951, 342:494]
    badge_fill_count = int(np.all(badge_region == [66, 54, 35], axis=2).sum())
    assert (badge_fill_count > 1000) is expected


def test_thumbnail_preserves_three_panels_and_original_frame():
    source = compose(*synthetic_panels())
    before = source.tobytes()
    thumbnail = make_thumbnail(source)
    assert thumbnail.size == (1920, 1080) and thumbnail.mode == "RGB"
    assert source.tobytes() == before
    for x, y, width, height in (HELPER_PANEL, WORLD_PANEL, LEADER_PANEL):
        assert (
            source.crop((x, y, x + width, y + height)).tobytes()
            == thumbnail.crop((x, y, x + width, y + height)).tobytes()
        )


@pytest.mark.parametrize("arguments", [(True, 1, 2), (1, -1, 2), (1, 2, 1)])
def test_invalid_annotation_interval_is_rejected(arguments):
    with pytest.raises(ValueError):
        annotation_active(*arguments)


def test_ego_panels_contain_nonsquare_images_without_stretching():
    helper, world, leader = synthetic_panels()
    helper = helper[:20]
    result = np.asarray(compose(helper, world, leader))
    ys, xs = np.nonzero(np.all(result == [201, 19, 33], axis=2))
    assert len(xs)
    assert (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) == pytest.approx(2, abs=0.02)


def test_selected_annotation_active_for_thirteen_of_253_context_frames():
    indices = list(range(18120, 18372 + 1))
    assert len(indices) == 253
    active = [index for index in indices if annotation_active(index, 18240, 18252)]
    assert active == list(range(18240, 18252 + 1))
    assert not annotation_active(18260, 18240, 18252)  # another annotation in this context


@pytest.fixture
def cheap_compositor(monkeypatch):
    calls = []
    world = Image.new("RGB", (2, 2))
    monkeypatch.setattr(demo_compositor, "render_world", lambda *args, **kwargs: world)

    def record_compose(helper, world, leader, **kwargs):
        calls.append(kwargs["frame_index"])
        return Image.new("RGB", (2, 2))

    monkeypatch.setattr(demo_compositor, "compose_frame", record_compose)
    return calls


def frame_iterator(indices, helper_indices=None, leader_indices=None):
    pixel = np.zeros((2, 2, 3), dtype=np.uint8)
    return demo_compositor.iter_composed_frames(
        [(index, ()) for index in indices],
        {
            "helper": iter(
                (index, pixel) for index in (indices if helper_indices is None else helper_indices)
            ),
            "leader": iter(
                (index, pixel) for index in (indices if leader_indices is None else leader_indices)
            ),
        },
        view=object(),
        trajectories={},
        annotation_start=18240,
        annotation_end=18252,
        fps=30,
    )


def test_composed_sequence_has_exactly_253_global_indices(cheap_compositor):
    indices = list(range(18120, 18373))
    count = 0
    for pixels in frame_iterator(indices):
        assert isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8
        count += 1
    assert count == 253
    assert cheap_compositor == indices


@pytest.mark.parametrize("participant", ("helper", "leader"))
def test_composed_sequence_rejects_wrong_video_index(cheap_compositor, participant):
    options = {f"{participant}_indices": [18121]}
    with pytest.raises(ValueError):
        list(frame_iterator([18120], **options))


@pytest.mark.parametrize("participant", ("helper", "leader"))
def test_composed_sequence_rejects_short_video(cheap_compositor, participant):
    options = {f"{participant}_indices": []}
    with pytest.raises(ValueError):
        list(frame_iterator([18120], **options))


@pytest.mark.parametrize("participant", ("helper", "leader"))
def test_composed_sequence_rejects_extra_video_images(cheap_compositor, participant):
    options = {f"{participant}_indices": [18120, 18121]}
    with pytest.raises(ValueError):
        list(frame_iterator([18120], **options))


@pytest.mark.parametrize("indices", ([18120, 18122], [18120, 18120], [18121, 18120]))
def test_composed_sequence_rejects_nonconsecutive_snapshots(cheap_compositor, indices):
    with pytest.raises(ValueError):
        list(frame_iterator(indices))
