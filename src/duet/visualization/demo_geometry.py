"""Fixed orthographic drawing of verified snapshots for an offline demo.

Only device/camera poses, calibrated frustum rays, and supplied hand points are
drawn. Camera frustums use a fixed 0.24 m display length, not observed depth. The
world z axis chooses display-up only; no floor, body, object, scene, or hand
connectivity is inferred. The caller supplies already verified frame snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, pairwise
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image, ImageDraw

from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId
from duet.visualization.rerun_episode import EntityUpdate, VerifiedImageGeometry

BACKGROUND = (19, 25, 34)
ROLE_COLORS = {"helper": (113, 194, 255), "leader": (255, 143, 151)}
FRUSTUM_LENGTH_M = 0.24
AXIS_LENGTH_M = 0.035
_ROLES = tuple(ROLE_COLORS)
_LAYERS = {
    "device": "pose",
    "camera_rgb": "pose",
    "hands/left": "points",
    "hands/right": "points",
    "camera_rgb/frustum": "frustum",
}


def _points(value: ArrayLike, *, allow_missing: bool = False) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError("geometry must contain real meter coordinates")
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("geometry must have shape Nx3")
    valid = np.isfinite(points).all(axis=1)
    if not valid.all() and (not allow_missing or not np.isnan(points[~valid]).all()):
        raise ValueError("geometry must be finite; trajectory gaps must be all-NaN rows")
    return points


def _immutable(value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _snapshot(updates: tuple[EntityUpdate, ...]) -> dict[tuple[str, str], object]:
    """Allowlist the geometry emitted by assemble_index_frame; ignore other layers."""
    result: dict[tuple[str, str], object] = {}
    seen = set()
    for update in updates:
        parts = update.path.split("/", 2)
        if len(parts) != 3 or parts[0] != "world" or parts[1] not in _ROLES:
            continue
        role, layer = parts[1:]
        if layer not in _LAYERS:
            continue
        key = (role, layer)
        if key in seen:
            raise ValueError("snapshot contains duplicate geometry updates")
        seen.add(key)
        if update.kind == "clear":
            continue
        if update.kind != _LAYERS[layer]:
            raise ValueError("snapshot layer has an incompatible geometry kind")
        value = update.value
        if update.kind == "pose":
            expected = FrameId(f"{role}/device" if layer == "device" else f"{role}/camera-rgb")
            if not isinstance(value, RigidTransform) or value.source != expected:
                raise ValueError("pose must use the corresponding participant device/camera frame")
        elif update.kind == "points":
            value = _points(value)
        elif not isinstance(value, VerifiedImageGeometry):
            raise ValueError("frustum requires verified image geometry")
        result[key] = value
    for role in _ROLES:
        geometry = result.get((role, "camera_rgb/frustum"))
        pose = result.get((role, "camera_rgb"))
        if geometry is not None and (pose is None or geometry.camera_frame != pose.source):
            raise ValueError("verified frustum rays require their matching camera pose")
    return result


def _axes(pose: RigidTransform) -> np.ndarray:
    return pose.apply(np.vstack((np.zeros(3), np.eye(3) * AXIS_LENGTH_M)), unit=DistanceUnit.METERS)


def _frustum(pose: RigidTransform, geometry: VerifiedImageGeometry) -> np.ndarray:
    return pose.apply(
        np.asarray(geometry.boundary_rays_camera) * FRUSTUM_LENGTH_M, unit=DistanceUnit.METERS
    )


@dataclass(frozen=True, eq=False)
class FixedView:
    """One deterministic orthographic view fitted to every selected snapshot.

    Basis rows are display right, display up, and direction toward the viewer.
    Point arrays are immutable. Supplied trajectories may contain only device
    origins included during fitting; all-NaN rows explicitly break a path.
    """

    size: tuple[int, int]
    basis: np.ndarray = field(repr=False)
    world_center: np.ndarray = field(repr=False)
    plane_center: np.ndarray = field(repr=False)
    pixels_per_meter: float
    fitted_points: np.ndarray = field(repr=False)
    world_frame: FrameId | None
    trajectory_points: Mapping[str, np.ndarray] = field(repr=False)
    padding_pixels: float

    def __post_init__(self) -> None:
        for name in ("basis", "world_center", "plane_center", "fitted_points"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(
            self,
            "trajectory_points",
            MappingProxyType(
                {role: _immutable(points) for role, points in self.trajectory_points.items()}
            ),
        )

    def project(self, points_world_m: ArrayLike) -> np.ndarray:
        """Project Nx3 meter coordinates to Nx2 pixel coordinates without clipping."""
        points = _points(points_world_m)
        plane = ((points - self.world_center) @ self.basis.T)[:, :2] - self.plane_center
        pixels = plane * np.array([self.pixels_per_meter, -self.pixels_per_meter])
        pixels += (np.asarray(self.size) - 1) / 2
        if not np.isfinite(pixels).all():
            raise ValueError("geometry projection overflowed")
        return pixels


def fit_view(snapshots: Sequence[tuple[EntityUpdate, ...]], size: tuple[int, int]) -> FixedView:
    """Fit all allowed layers across the selected clip, with a stable elevated view."""
    if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 64 for v in size):
        raise ValueError("view dimensions must be integer pixel sizes of at least 64")
    all_points, world_frames = [], set()
    origins: dict[str, list[np.ndarray]] = {role: [] for role in _ROLES}
    for updates in snapshots:
        layers = _snapshot(updates)
        for (role, layer), value in layers.items():
            if isinstance(value, RigidTransform):
                world_frames.add(value.destination)
                all_points.append(_axes(value))
                if layer == "device":
                    origins[role].append(value.matrix[:3, 3])
            elif layer.startswith("hands/"):
                all_points.append(value)
            elif layer == "camera_rgb/frustum":
                all_points.append(_frustum(layers[(role, "camera_rgb")], value))
    if len(world_frames) > 1:
        raise ValueError("snapshots mix incompatible world frames")
    if not all_points or not sum(len(points) for points in all_points):
        raise ValueError("cannot fit a view without verified geometry")
    points = np.concatenate(all_points)
    baseline = np.array([1.0, 0.0, 0.0])
    if all(origins.values()):
        delta = np.mean(origins["leader"], axis=0) - np.mean(origins["helper"], axis=0)
        horizontal = np.array([delta[0], delta[1], 0.0])
        if np.linalg.norm(horizontal) > 1e-8:
            baseline = horizontal / np.linalg.norm(horizontal)
    display_z = np.array([0.0, 0.0, 1.0])
    outward = np.cross(baseline, display_z) + 0.28 * baseline + 0.58 * display_z
    outward /= np.linalg.norm(outward)
    right = np.cross(display_z, outward)
    right /= np.linalg.norm(right)
    up = np.cross(outward, right)
    basis = np.vstack((right, up, outward))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    plane = ((points - center) @ basis.T)[:, :2]
    low, high = plane.min(axis=0), plane.max(axis=0)
    extent = np.maximum(high - low, 0.10)
    padding = max(18.0, min(size) * 0.10)
    scale = float(np.min((np.asarray(size) - 1 - 2 * padding) / extent))
    return FixedView(
        tuple(size),
        basis,
        center,
        (low + high) / 2,
        scale,
        points,
        next(iter(world_frames), None),
        {role: np.asarray(values).reshape(-1, 3) for role, values in origins.items()},
        padding,
    )


def _color(role: str, strength: float = 1.0, *, pale: bool = False) -> tuple[int, int, int]:
    base = np.asarray(ROLE_COLORS[role], dtype=float)
    if pale:
        base = base * 0.72 + 255 * 0.28
    return tuple(np.rint(np.asarray(BACKGROUND) * (1 - strength) + base * strength).astype(int))


def evaluate_view(
    snapshots: Sequence[tuple[EntityUpdate, ...]], view: FixedView
) -> dict[str, object]:
    """Measure display bounds and hand footprints without inferring visibility.

    Pixel distances describe this fixed orthographic drawing only. Overlapping
    markers or projected hand centroids do not establish contact or occlusion.
    Missing counts refer to cleared/absent renderer layers after upstream QC.
    The conservative glyph margin includes the largest 5.2px device marker.
    """
    labels = [f"{role}/{side}" for role in _ROLES for side in ("left", "right")]
    hand_metrics = {label: {"diagonal": [], "spacing": [], "overlap": []} for label in labels}
    separations: dict[str, list[float]] = {}
    all_pixels = []
    pose_counts = {role: {"device": 0, "camera_rgb": 0, "frustum": 0} for role in _ROLES}
    for updates in snapshots:
        layers = _snapshot(updates)
        centroids = {}
        for (role, layer), value in layers.items():
            if isinstance(value, RigidTransform):
                if value.destination != view.world_frame:
                    raise ValueError("snapshot pose targets a different world from the fitted view")
                all_pixels.append(view.project(_axes(value)))
                pose_counts[role][layer] += 1
            elif layer == "camera_rgb/frustum":
                all_pixels.append(view.project(_frustum(layers[(role, "camera_rgb")], value)))
                pose_counts[role]["frustum"] += 1
            elif layer.startswith("hands/") and len(value):
                pixels = view.project(value)
                all_pixels.append(pixels)
                label = f"{role}/{layer.removeprefix('hands/')}"
                centroids[label] = pixels.mean(axis=0)
                hand_metrics[label]["diagonal"].append(
                    float(np.linalg.norm(np.ptp(pixels, axis=0)))
                )
                if len(pixels) > 1:
                    distances = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=2)
                    np.fill_diagonal(distances, np.inf)
                    closest = distances.min(axis=1)
                    hand_metrics[label]["spacing"].append(float(closest.min()))
                    hand_metrics[label]["overlap"].append(float(np.mean(closest < 7.2)))
        for first, second in combinations(sorted(centroids), 2):
            name = f"{first} to {second}"
            separations.setdefault(name, []).append(
                float(np.linalg.norm(centroids[first] - centroids[second]))
            )

    def statistics(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
        return {
            "count": len(values),
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    pixels = np.concatenate(all_pixels) if all_pixels else np.empty((0, 2))
    bounds = (
        None
        if not len(pixels)
        else {"min": pixels.min(axis=0).tolist(), "max": pixels.max(axis=0).tolist()}
    )
    edge_distance = np.minimum(pixels, np.asarray(view.size) - 1 - pixels)
    outside_count = int(np.any(edge_distance < 0, axis=1).sum())
    margin = float(edge_distance.min()) if len(pixels) else None
    return {
        "snapshot_count": len(snapshots),
        "panel_size": list(view.size),
        "pixels_per_meter": view.pixels_per_meter,
        "projected_geometry_bounds_pixels": bounds,
        "outside_panel_point_count": outside_count,
        "minimum_point_edge_margin_pixels": margin,
        "minimum_conservative_glyph_edge_margin_pixels": None if margin is None else margin - 5.2,
        "pose_present_counts": pose_counts,
        "hands": {
            label: {
                "present_count": len(metrics["diagonal"]),
                "missing_count": len(snapshots) - len(metrics["diagonal"]),
                "bbox_diagonal_pixels": statistics(metrics["diagonal"]),
                "minimum_landmark_separation_pixels": statistics(metrics["spacing"]),
                "fraction_of_landmarks_with_overlapping_7_2px_markers": statistics(
                    metrics["overlap"]
                ),
            }
            for label, metrics in hand_metrics.items()
        },
        "hand_centroid_separation_pixels": {
            label: statistics(values) for label, values in separations.items()
        },
        "interpretation": "Display geometry only: marker overlaps and centroid separation do not establish anatomical contact, occlusion, tracking quality, or actual visibility. Missing counts describe renderer layers after upstream QC.",
        "scene_drawn": False,
        "hand_connectivity_drawn": False,
    }


def render_world(
    updates: tuple[EntityUpdate, ...], view: FixedView, trajectories: Mapping[str, np.ndarray]
) -> Image.Image:
    """Draw one snapshot without carrying stale poses or hands across frames."""
    layers = _snapshot(updates)
    primitives = []

    def line(points: np.ndarray, color: tuple[int, int, int], width: float) -> None:
        depth = float(np.mean((points - view.world_center) @ view.basis[2]))
        primitives.append((depth, "line", view.project(points), color, width))

    def marker(
        point: np.ndarray, color: tuple[int, int, int], radius: float, *, ring: bool = False
    ) -> None:
        depth = float((point - view.world_center) @ view.basis[2])
        primitives.append(
            (depth, "ring" if ring else "point", view.project(point[None, :])[0], color, radius)
        )

    if set(trajectories) - set(_ROLES):
        raise ValueError("trajectory contains an unknown participant")
    for role, values in trajectories.items():
        path = _points(values, allow_missing=True)
        support = {tuple(point) for point in view.trajectory_points[role]}
        if any(tuple(point) not in support for point in path if np.isfinite(point).all()):
            raise ValueError("trajectory contains positions outside the selected fitted snapshots")
        for start, end in pairwise(path):
            if np.isfinite([start, end]).all():
                line(np.vstack((start, end)), _color(role, 0.35), 1.5)
    for (role, layer), value in layers.items():
        if isinstance(value, RigidTransform):
            if value.destination != view.world_frame:
                raise ValueError("current pose targets a different world from the fitted view")
            axes = _axes(value)
            for endpoint in axes[1:]:
                line(np.vstack((axes[0], endpoint)), _color(role, 0.36), 1.15)
            marker(axes[0], _color(role), 5.2 if layer == "device" else 3.7, ring=layer == "device")
        elif layer.startswith("hands/"):
            for point in value:
                marker(point, _color(role, pale=layer.endswith("right")), 3.6)
        elif layer == "camera_rgb/frustum":
            pose = layers[(role, "camera_rgb")]
            perimeter = _frustum(pose, value)
            line(np.vstack((perimeter, perimeter[:1])), _color(role, 0.48), 1.15)
            for index in np.linspace(0, len(perimeter), 4, endpoint=False, dtype=int):
                line(np.vstack((pose.matrix[:3, 3], perimeter[index])), _color(role, 0.24), 1.0)
    supersampling = 2
    image = Image.new("RGB", tuple(v * supersampling for v in view.size), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for _, kind, pixels, color, radius in sorted(primitives, key=lambda item: item[0]):
        if kind == "line":
            draw.line(
                [tuple(point * supersampling) for point in pixels],
                fill=color,
                width=max(1, round(radius * supersampling)),
                joint="curve",
            )
        else:
            x, y = pixels * supersampling
            r = radius * supersampling
            bounds = (x - r, y - r, x + r, y + r)
            if kind == "ring":
                draw.ellipse(bounds, fill=BACKGROUND, outline=color, width=2 * supersampling)
            else:
                draw.ellipse(bounds, fill=color, outline=BACKGROUND, width=1)
    return image.resize(view.size, resample=Image.Resampling.LANCZOS)
