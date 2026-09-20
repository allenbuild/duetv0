"""Dataset-independent quaternion conversion tests."""

from math import sqrt

import numpy as np
import pytest

from duet.geometry.rotations import (
    quaternion_xyzw_batch_to_rotations,
    quaternion_xyzw_to_rotation,
)


def test_xyzw_identity_and_quarter_turn_convention() -> None:
    np.testing.assert_array_equal(quaternion_xyzw_to_rotation([0, 0, 0, 1]), np.eye(3))
    rotation = quaternion_xyzw_to_rotation([0, 0, sqrt(0.5), sqrt(0.5)])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)


@pytest.mark.parametrize(
    "quaternion",
    [
        [0, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 0],
        [0, 0, 0, float("nan")],
        [0, 0, 0, float("inf")],
        [0, 0, 0, 1j],
    ],
)
def test_invalid_quaternion_rejected_without_normalization(quaternion: list[object]) -> None:
    with pytest.raises(ValueError):
        quaternion_xyzw_to_rotation(quaternion)


def test_small_serialization_error_is_retained_not_normalized() -> None:
    values = np.array([0, 0, sqrt(0.5), sqrt(0.5)]) * (1 + 1e-9)
    original = values.copy()
    rotation = quaternion_xyzw_to_rotation(values, norm_tolerance=1e-7)
    np.testing.assert_array_equal(values, original)
    assert not np.array_equal(
        rotation, quaternion_xyzw_to_rotation(values / np.linalg.norm(values))
    )
    with pytest.raises(ValueError, match="unit norm"):
        quaternion_xyzw_to_rotation(values, norm_tolerance=1e-10)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError):
        quaternion_xyzw_to_rotation([0, 0, 0, 1], norm_tolerance=tolerance)


def test_vectorized_rotation_batch_matches_single_conversion_and_handles_empty() -> None:
    values = np.array([[0, 0, 0, 1], [0, 0, sqrt(0.5), sqrt(0.5)]])
    rotations = quaternion_xyzw_batch_to_rotations(values)
    assert rotations.shape == (2, 3, 3)
    for quaternion, rotation in zip(values, rotations, strict=True):
        np.testing.assert_array_equal(rotation, quaternion_xyzw_to_rotation(quaternion))
    assert quaternion_xyzw_batch_to_rotations(np.empty((0, 4))).shape == (0, 3, 3)


@pytest.mark.parametrize(
    "batch",
    [
        [[0, 0, 0, 1], [0, 0, 0, 2]],
        [[0, 0, 0, 1], [0, 0, 0, float("nan")]],
        [[0, 0, 0, float("inf")]],
        [[0, 0, 0, 1j]],
        [[0, 0, 1]],
        [0, 0, 0, 1],
    ],
)
def test_invalid_quaternion_batch_rejects_entire_batch(batch: object) -> None:
    with pytest.raises(ValueError):
        quaternion_xyzw_batch_to_rotations(batch)
