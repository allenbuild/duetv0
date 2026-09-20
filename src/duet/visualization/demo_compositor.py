"""Deterministic three-panel presentation of one verified paired-index clip.

FPS controls the presentation rate only. Geometry has already passed the
canonical local DEVICE_TIME gates before it enters this presentation layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from duet.visualization.demo_geometry import FixedView, render_world
from duet.visualization.rerun_episode import EntityUpdate

WIDTH, HEIGHT = 1920, 1080
HELPER_PANEL = (48, 258, 432, 606)
WORLD_PANEL = (504, 258, 912, 606)
LEADER_PANEL = (1440, 258, 432, 606)
BACKGROUND = (12, 17, 24)
PANEL_BACKGROUND = (19, 25, 34)
TEXT = (238, 242, 248)
MUTED = (150, 162, 180)
HELPER = (113, 194, 255)
LEADER = (255, 143, 151)
ACCENT = (238, 204, 135)


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ("/System/Library/Fonts/Avenir Next.ttc", 0 if bold else 7),
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            0,
        ),
    )
    for path, index in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size, index=index)
    return ImageFont.load_default(size=size)


def annotation_active(frame_index: int, start: int, end: int) -> bool:
    """Follow the existing viewer's explicit inclusive annotation display policy."""
    if any(type(value) is not int for value in (frame_index, start, end)):
        raise ValueError("annotation frame indices must be integers")
    if start < 0 or end < start:
        raise ValueError("annotation interval must have ordered nonnegative indices")
    return start <= frame_index <= end


def _rgb_image(pixels: np.ndarray) -> Image.Image:
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or min(pixels.shape[:2]) < 1
    ):
        raise ValueError("ego panels require nonempty uint8 RGB images")
    return Image.fromarray(pixels)


def _ego_panel(canvas: Image.Image, pixels: np.ndarray, panel: tuple[int, ...]) -> None:
    x, y, width, height = panel
    image = ImageOps.contain(_rgb_image(pixels), (width, width), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))


def compose_frame(
    helper_rgb: np.ndarray,
    world_rgb: Image.Image,
    leader_rgb: np.ndarray,
    *,
    frame_index: int,
    start_frame: int,
    end_frame: int,
    annotation_start: int,
    annotation_end: int,
    fps: int = 30,
    thumbnail: bool = False,
    annotation_id: str = "018240",
    object_category: str = "bowl",
    qc_label: str = "",
) -> Image.Image:
    """Compose 1920×1080 RGB without cropping or stretching either ego image."""
    if (
        type(fps) is not int
        or fps <= 0
        or any(type(value) is not int for value in (frame_index, start_frame, end_frame))
        or not 0 <= start_frame <= frame_index <= end_frame
    ):
        raise ValueError("invalid presentation FPS or inclusive clip indices")
    if world_rgb.size != WORLD_PANEL[2:] or world_rgb.mode != "RGB":
        raise ValueError("world renderer must provide the exact RGB center-panel dimensions")
    if any(
        not isinstance(label, str) or not label.strip()
        for label in (annotation_id, object_category)
    ):
        raise ValueError("annotation ID and object category must be nonempty source labels")
    if not isinstance(qc_label, str):
        raise TypeError("QC label must be a string")
    active = annotation_active(frame_index, annotation_start, annotation_end)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 35), "DUET V0", fill=TEXT, font=_font(57, True))
    subtitle = (
        "Human-Human Interaction in Shared 3D"
        if thumbnail
        else "Synchronized Human-Human Interaction"
    )
    draw.text((49, 111), subtitle, fill=TEXT, font=_font(30))
    if not thumbnail:
        draw.text(
            (50, 157),
            "Paired egocentric video + hands + shared 3D motion",
            fill=MUTED,
            font=_font(20),
        )
    draw.rounded_rectangle((1520, 55, 1872, 103), radius=24, fill=(25, 41, 42))
    draw.ellipse((1539, 75, 1547, 83), fill=(130, 216, 179))
    draw.text((1561, 64), "Verified synchronization", font=_font(21), fill=(182, 230, 208))
    draw.line((48, 200, 1872, 200), fill=(42, 51, 65), width=1)
    for panel, label, color in (
        (HELPER_PANEL, "HELPER EGO", HELPER),
        (WORLD_PANEL, "SHARED 3D WORLD", TEXT),
        (LEADER_PANEL, "LEADER EGO", LEADER),
    ):
        x, y, width, height = panel
        draw.text((x, 219), label, font=_font(20, True), fill=color)
        draw.rounded_rectangle(
            (x, y, x + width - 1, y + height - 1), radius=16, fill=PANEL_BACKGROUND
        )
    _ego_panel(canvas, helper_rgb, HELPER_PANEL)
    _ego_panel(canvas, leader_rgb, LEADER_PANEL)
    center_mask = Image.new("L", WORLD_PANEL[2:], 0)
    ImageDraw.Draw(center_mask).rounded_rectangle(
        (0, 0, WORLD_PANEL[2] - 1, WORLD_PANEL[3] - 1), radius=16, fill=255
    )
    canvas.paste(world_rgb, WORLD_PANEL[:2], center_mask)
    draw = ImageDraw.Draw(canvas)
    for x, label, color in ((795, "Helper", HELPER), (993, "Leader", LEADER)):
        draw.ellipse((x, 830, x + 9, 839), fill=color)
        draw.text((x + 20, 820), label, font=_font(18), fill=MUTED)
    object_label = f"{object_category.upper()} HANDOVER"
    draw.text((48, 909), object_label, font=_font(28, True), fill=TEXT)
    if active:
        badge_x = max(342, 72 + round(draw.textlength(object_label, font=_font(28, True))))
        draw.rounded_rectangle((badge_x, 910, badge_x + 151, 950), radius=20, fill=(66, 54, 35))
        draw.text((badge_x + 22, 918), "HANDOVER", font=_font(17, True), fill=ACCENT)
    draw.text((1872, 918), annotation_id, anchor="rt", font=_font(20), fill=MUTED)
    if qc_label:
        qc_width = round(draw.textlength(qc_label, font=_font(18)))
        draw.rounded_rectangle((1738 - qc_width - 24, 910, 1750, 950), radius=20, fill=(66, 54, 35))
        draw.text((1738, 919), qc_label, anchor="rt", font=_font(18), fill=ACCENT)
    # A playback progress track is a presentation aid, never a device clock.
    denominator = max(1, end_frame - start_frame)
    progress_x = 48 + round(1824 * (frame_index - start_frame) / denominator)
    draw.rounded_rectangle((48, 979, 1872, 983), radius=2, fill=(40, 50, 64))
    lo = max(start_frame, annotation_start)
    hi = min(end_frame, annotation_end)
    if lo <= hi:
        x0 = 48 + round(1824 * (lo - start_frame) / denominator)
        x1 = 48 + round(1824 * (hi - start_frame) / denominator)
        draw.rectangle((x0, 979, max(x0 + 2, x1), 983), fill=(131, 116, 83))
    draw.ellipse((progress_x - 5, 976, progress_x + 5, 986), fill=TEXT)
    elapsed = (frame_index - annotation_start) / fps
    draw.text(
        (48, 1010),
        f"Frame {frame_index}  ·  {elapsed:+.2f}s  ·  DEVICE_TIME synchronized",
        font=_font(21),
        fill=MUTED,
    )
    draw.text(
        (1872, 1011),
        "Time relative to handover start",
        anchor="rt",
        font=_font(18),
        fill=(113, 127, 147),
    )
    return canvas


def make_thumbnail(frame: Image.Image) -> Image.Image:
    """Retitle a decoded handover frame, preserving its verified three panels."""
    if frame.size != (WIDTH, HEIGHT) or frame.mode != "RGB":
        raise ValueError("thumbnail source must be an RGB Full HD demo frame")
    result = frame.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((48, 104, 1430, 190), fill=BACKGROUND)
    draw.text((49, 111), "Human-Human Interaction in Shared 3D", fill=TEXT, font=_font(30))
    draw.rectangle((40, 966, 1880, 1079), fill=BACKGROUND)
    draw.rectangle((1750, 900, 1879, 958), fill=BACKGROUND)
    return result


def iter_composed_frames(
    snapshots: Sequence[tuple[int, tuple[EntityUpdate, ...]]],
    video_frames: Mapping[str, Iterator[tuple[int, np.ndarray]]],
    *,
    view: FixedView,
    trajectories: Mapping[str, np.ndarray],
    annotation_start: int,
    annotation_end: int,
    fps: int = 30,
    annotation_id: str = "018240",
    object_category: str = "bowl",
    qc_label: str = "",
) -> Iterator[np.ndarray]:
    """Advance geometry and both actual MP4 iterators from one common sequence index."""
    if not snapshots or set(video_frames) != {"helper", "leader"}:
        raise ValueError("nonempty snapshots and both ego streams are required")
    start, end = snapshots[0][0], snapshots[-1][0]
    if [index for index, _ in snapshots] != list(range(start, end + 1)):
        raise ValueError("snapshots must have contiguous synchronized frame indices")
    for index, updates in snapshots:
        images = {}
        for role in ("helper", "leader"):
            try:
                decoded_index, image = next(video_frames[role])
            except StopIteration as error:
                raise ValueError(f"{role} ego video ended before synchronized geometry") from error
            if decoded_index != index:
                raise ValueError(f"{role} ego index does not match synchronized geometry")
            images[role] = image
        world = render_world(updates, view, trajectories)
        composed = compose_frame(
            images["helper"],
            world,
            images["leader"],
            frame_index=index,
            start_frame=start,
            end_frame=end,
            annotation_start=annotation_start,
            annotation_end=annotation_end,
            fps=fps,
            annotation_id=annotation_id,
            object_category=object_category,
            qc_label=qc_label,
        )
        yield np.asarray(composed)
    for role, iterator in video_frames.items():
        if next(iterator, None) is not None:
            raise ValueError(f"{role} ego stream contains frames outside the selected clip")
