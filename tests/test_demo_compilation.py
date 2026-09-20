"""Small rendered MP4 fixtures only: no recordings, frame maps, or VRS reads."""

import json
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image, ImageDraw

from duet.visualization import demo_compilation, demo_compositor
from duet.visualization.demo_compilation import choose_segment, export_compilation
from duet.visualization.demo_compositor import WORLD_PANEL, compose_frame, make_thumbnail
from duet.visualization.demo_encoding import encode_video


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ((100, 399, 240, 260, 150), (175, 324)),
        ((100, 399, 100, 110, 150), (100, 249)),
        ((100, 399, 390, 399, 150), (250, 399)),
        ((100, 399, 130, 350, 10), (235, 244)),
        ((100, 399, 200, 200, 1), (200, 200)),
        ((100, 399, 200, 201, 1), (200, 200)),
        ((100, 399, 100, 399, 300), (100, 399)),
    ],
)
def test_segment_centering_and_clip_bounds(arguments, expected):
    assert choose_segment(*arguments) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        (-1, 299, 20, 30, 150),
        (100, 99, 100, 100, 1),
        (100, 399, 99, 101, 150),
        (100, 399, 390, 400, 150),
        (100, 399, 200, 199, 150),
        (100, 399, 200, 201, 301),
        (100, 399, 200, 201, 0),
        (100, 399, 200, 201, True),
        (100, 399.0, 200, 201, 150),
    ],
)
def test_invalid_segment_requests_are_not_silently_adjusted(arguments):
    with pytest.raises(ValueError):
        choose_segment(*arguments)


def source_video(path, *, frames=12, fps=6, size=(64, 48), color=None):
    if color is None:
        pixels = (np.full((size[1], size[0], 3), index * 15, np.uint8) for index in range(frames))
    else:
        pixels = (np.full((size[1], size[0], 3), color, np.uint8) for _ in range(frames))
    encode_video(path, pixels, size=size, fps=fps, expected_frames=frames)
    return path


def specification(path, start=0, end=5, annotation_id="001", object_category="bowl"):
    return {
        "path": path,
        "start_ordinal": start,
        "end_ordinal": end,
        "annotation_id": annotation_id,
        "object_category": object_category,
    }


def decoded_pixels(path):
    with av.open(str(path)) as container:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]


def test_compilation_preserves_selected_ordinals_and_labels_and_is_json_safe(tmp_path):
    source = source_video(tmp_path / "source.mp4")
    output = tmp_path / "compilation.mp4"
    result = export_compilation(
        output,
        [specification(source, 2, 5, "002131", "paper plate"), specification(source, 8, 11)],
        fps=6,
        transition_frames=0,
    )
    assert result["path"] == str(output)
    assert result["codec"] == "h264" and result["pixel_format"] == "yuv420p"
    assert result["width"] == 64 and result["height"] == 48
    assert result["fps"] == 6 and result["decoded_frame_count"] == 8
    assert result["duration_seconds"] == pytest.approx(8 / 6)
    assert result["audio_stream_count"] == 0
    assert result["segments"][0]["annotation_id"] == "002131"
    assert result["segments"][0]["object_category"] == "paper plate"
    assert result["segments"][1]["output_start_ordinal"] == 4
    assert result["segments"][1]["output_end_ordinal"] == 7
    json.dumps(result, allow_nan=False)
    means = [float(pixels.mean()) for pixels in decoded_pixels(output)]
    np.testing.assert_allclose(means, np.array([2, 3, 4, 5, 8, 9, 10, 11]) * 15, atol=3)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["compilation.mp4", "source.mp4"]


def test_internal_cut_fades_to_dark_without_mixing_scenes_or_changing_count(tmp_path):
    red = source_video(tmp_path / "red.mp4", frames=6, color=[200, 0, 0])
    blue = source_video(tmp_path / "blue.mp4", frames=6, color=[0, 0, 200])
    output = tmp_path / "faded.mp4"
    result = export_compilation(
        output, [specification(red), specification(blue)], fps=6, transition_frames=8
    )
    pixels = decoded_pixels(output)
    assert len(pixels) == result["decoded_frame_count"] == 12
    means = np.array([image.mean(axis=(0, 1)) for image in pixels])
    np.testing.assert_allclose(means[:6, 0], [200, 200, 150, 100, 50, 0], atol=5)
    np.testing.assert_allclose(means[6:, 2], [0, 50, 100, 150, 200, 200], atol=5)
    assert means[:6, 2].max() < 5 and means[6:, 0].max() < 5
    assert result["duration_seconds"] == 2


@pytest.mark.parametrize("count, segment_frames, expected_count", [(6, 150, 900), (3, 225, 675)])
def test_batch_lengths_have_exact_requested_duration(
    tmp_path, count, segment_frames, expected_count
):
    source = source_video(
        tmp_path / "source.mp4", frames=segment_frames, fps=30, color=[80, 90, 100]
    )
    result = export_compilation(
        tmp_path / "batch.mp4",
        [specification(source, 0, segment_frames - 1, str(index)) for index in range(count)],
    )
    assert result["decoded_frame_count"] == expected_count
    assert result["duration_seconds"] == expected_count / 30


def test_single_segment_has_no_fade_even_when_transition_exceeds_its_length(tmp_path):
    source = source_video(tmp_path / "source.mp4", frames=2, color=[100, 100, 100])
    output = tmp_path / "single.mp4"
    export_compilation(output, [specification(source, 0, 1)], fps=6)
    np.testing.assert_allclose([frame.mean() for frame in decoded_pixels(output)], 100, atol=3)


@pytest.mark.parametrize("options", [{"fps": True}, {"fps": 6.0}, {"transition_frames": -1}])
def test_invalid_compilation_configuration_is_rejected_before_opening_sources(tmp_path, options):
    with pytest.raises(ValueError):
        export_compilation(
            tmp_path / "out.mp4", [specification(tmp_path / "absent.mp4")], **options
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"start_ordinal": True},
        {"start_ordinal": -1},
        {"end_ordinal": 12},
        {"start_ordinal": 6, "end_ordinal": 5},
        {"annotation_id": ""},
        {"object_category": None},
    ],
)
def test_invalid_segment_specification_preserves_existing_output(tmp_path, changes):
    source = source_video(tmp_path / "source.mp4")
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous")
    with pytest.raises(ValueError):
        export_compilation(output, [{**specification(source), **changes}], fps=6)
    assert output.read_bytes() == b"previous"


def test_empty_and_missing_segment_fields_are_rejected(tmp_path):
    for segments in ([], [{"path": "not-opened"}]):
        with pytest.raises(ValueError):
            export_compilation(tmp_path / "out.mp4", segments)


def test_sources_must_match_size_and_fps_and_not_be_overwritten(tmp_path):
    first = source_video(tmp_path / "one.mp4")
    second = source_video(tmp_path / "two.mp4", size=(32, 24))
    with pytest.raises(ValueError, match="dimensions"):
        export_compilation(
            tmp_path / "out.mp4", [specification(first), specification(second)], fps=6
        )
    with pytest.raises(ValueError, match="FPS"):
        export_compilation(tmp_path / "out.mp4", [specification(first)], fps=30)
    with pytest.raises(ValueError, match="differ"):
        export_compilation(first, [specification(first)], fps=6)


def test_transition_fades_cannot_overlap_inside_short_segment(tmp_path):
    source = source_video(tmp_path / "source.mp4")
    with pytest.raises(ValueError, match="overlap"):
        export_compilation(
            tmp_path / "out.mp4", [specification(source, 0, 5)] * 3, fps=6, transition_frames=8
        )


def test_late_output_validation_failure_preserves_existing_destination(tmp_path, monkeypatch):
    source = source_video(tmp_path / "source.mp4")
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous")

    def fail(*args, **kwargs):
        raise ValueError("decoded validation failed")

    monkeypatch.setattr(demo_compilation, "inspect_video", fail)
    with pytest.raises(ValueError, match="decoded validation"):
        export_compilation(output, [specification(source)], fps=6)
    assert output.read_bytes() == b"previous"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.mp4", "source.mp4"]


def test_raw_source_and_destination_guards(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr(demo_compilation, "_RAW_ROOT", raw)
    with pytest.raises(ValueError, match="data/raw"):
        export_compilation(raw / "out.mp4", [])
    with pytest.raises(ValueError, match="data/raw"):
        export_compilation(tmp_path / "out.mp4", [specification(raw / "source.mp4")])
    assert not raw.exists()


@pytest.mark.parametrize("ordinals", [[0, 2, 3], [0, 0, 1], [2, 3], [0, 1]])
def test_decoder_rejects_gaps_duplicates_wrong_seek_and_early_end(tmp_path, monkeypatch, ordinals):
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"fixture")
    source = demo_compilation._Source(
        source_path, (2, 2), 4, Fraction(1, 6), demo_compilation._signature(source_path)
    )
    segment = demo_compilation._Segment(source, 0, 2, "001", "bowl")
    seek_calls = []

    class Container:
        streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **kwargs):
            seek_calls.append(offset)

        def decode(self, stream):
            for ordinal in ordinals:
                yield SimpleNamespace(
                    pts=ordinal,
                    time_base=Fraction(1, 6),
                    width=2,
                    height=2,
                    format=SimpleNamespace(name="yuv420p"),
                    to_ndarray=lambda **kwargs: np.zeros((2, 2, 3), np.uint8),
                )

    monkeypatch.setattr(demo_compilation.av, "open", lambda *args, **kwargs: Container())
    with pytest.raises(ValueError):
        list(demo_compilation._decode_segment(segment, 6))
    assert seek_calls == [0]


def panels():
    return (
        np.zeros((8, 8, 3), np.uint8),
        Image.new("RGB", WORLD_PANEL[2:]),
        np.zeros((8, 8, 3), np.uint8),
    )


def test_dynamic_footer_labels_and_tracking_badge_do_not_overlap(monkeypatch):
    texts = []
    original = ImageDraw.ImageDraw.text

    def record(self, xy, text, *args, **kwargs):
        texts.append((xy, text))
        return original(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    image = compose_frame(
        *panels(),
        frame_index=20,
        start_frame=0,
        end_frame=30,
        annotation_start=19,
        annotation_end=21,
        annotation_id="018941",
        object_category="carrot peel",
        qc_label="Tracking gaps",
    )
    positions = {text: xy for xy, text in texts}
    assert positions["CARROT PEEL HANDOVER"] == (48, 909)
    assert positions["018941"] == (1872, 918)
    assert positions["Tracking gaps"][0] < positions["018941"][0]
    label_width = ImageDraw.Draw(image).textlength(
        "CARROT PEEL HANDOVER", font=demo_compositor._font(28, True)
    )
    assert positions["HANDOVER"][0] > 48 + label_width + 20
    thumbnail = make_thumbnail(image)
    assert (
        thumbnail.crop((40, 900, 1749, 960)).tobytes() == image.crop((40, 900, 1749, 960)).tobytes()
    )


def test_iterator_forwards_explicit_source_and_qc_labels(monkeypatch):
    calls = []
    monkeypatch.setattr(demo_compositor, "render_world", lambda *args: Image.new("RGB", (2, 2)))

    def compose(*args, **kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (2, 2))

    monkeypatch.setattr(demo_compositor, "compose_frame", compose)
    pixel = np.zeros((2, 2, 3), np.uint8)
    frames = demo_compositor.iter_composed_frames(
        [(7, ())],
        {role: iter([(7, pixel)]) for role in ("helper", "leader")},
        view=object(),
        trajectories={},
        annotation_start=7,
        annotation_end=7,
        annotation_id="018941",
        object_category="carrot peel",
        qc_label="Tracking gaps",
    )
    assert len(list(frames)) == 1
    assert {key: calls[0][key] for key in ("annotation_id", "object_category", "qc_label")} == {
        "annotation_id": "018941",
        "object_category": "carrot peel",
        "qc_label": "Tracking gaps",
    }
