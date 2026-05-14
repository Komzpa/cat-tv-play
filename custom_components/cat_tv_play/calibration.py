"""Wall-plane calibration helpers for Cat TV Play.

The camera that watches a projected wall is usually not square to the wall.
Jump height therefore must be measured through a planar homography, not by a
single pixel-per-centimeter scale.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationPoint:
    """A known point on the play wall."""

    image_x: float
    image_y: float
    wall_x_cm: float
    wall_y_cm: float


def _solve_linear_system(matrix: list[list[float]], values: list[float]) -> list[float]:
    """Solve a small dense linear system with Gaussian elimination."""

    size = len(values)
    rows = [matrix[index][:] + [values[index]] for index in range(size)]

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda row: abs(rows[row][pivot_index]))
        rows[pivot_index], rows[pivot_row] = rows[pivot_row], rows[pivot_index]
        pivot = rows[pivot_index][pivot_index]
        if abs(pivot) < 1e-12:
            raise ValueError("calibration points do not define a stable wall plane")

        for column in range(pivot_index, size + 1):
            rows[pivot_index][column] /= pivot

        for row in range(size):
            if row == pivot_index:
                continue
            factor = rows[row][pivot_index]
            if factor == 0:
                continue
            for column in range(pivot_index, size + 1):
                rows[row][column] -= factor * rows[pivot_index][column]

    return [rows[index][size] for index in range(size)]


def image_to_wall_homography(points: Iterable[CalibrationPoint]) -> tuple[float, ...]:
    """Return an image-to-wall homography from at least four wall points.

    The returned tuple contains eight coefficients. The ninth homography
    coefficient is fixed to one:

    wall_x = (h0*x + h1*y + h2) / (h6*x + h7*y + 1)
    wall_y = (h3*x + h4*y + h5) / (h6*x + h7*y + 1)
    """

    calibration_points = list(points)
    if len(calibration_points) < 4:
        raise ValueError("at least four calibration points are required")

    normal_matrix = [[0.0 for _ in range(8)] for _ in range(8)]
    normal_values = [0.0 for _ in range(8)]

    for point in calibration_points:
        x = point.image_x
        y = point.image_y
        wx = point.wall_x_cm
        wy = point.wall_y_cm
        rows = (
            ([x, y, 1.0, 0.0, 0.0, 0.0, -wx * x, -wx * y], wx),
            ([0.0, 0.0, 0.0, x, y, 1.0, -wy * x, -wy * y], wy),
        )
        for row, value in rows:
            for i, left in enumerate(row):
                normal_values[i] += left * value
                for j, right in enumerate(row):
                    normal_matrix[i][j] += left * right

    return tuple(_solve_linear_system(normal_matrix, normal_values))


def transform_image_point(homography: tuple[float, ...], image_x: float, image_y: float) -> tuple[float, float]:
    """Transform an image point into wall centimeters."""

    if len(homography) != 8:
        raise ValueError("homography must contain exactly eight coefficients")

    h0, h1, h2, h3, h4, h5, h6, h7 = homography
    denominator = h6 * image_x + h7 * image_y + 1.0
    if abs(denominator) < 1e-12:
        raise ValueError("image point maps to infinity for this calibration")

    wall_x = (h0 * image_x + h1 * image_y + h2) / denominator
    wall_y = (h3 * image_x + h4 * image_y + h5) / denominator
    return wall_x, wall_y


def points_from_service_data(points: Iterable[dict]) -> list[CalibrationPoint]:
    """Parse service data dictionaries into calibration points."""

    parsed: list[CalibrationPoint] = []
    for point in points:
        parsed.append(
            CalibrationPoint(
                image_x=float(point["image_x"]),
                image_y=float(point["image_y"]),
                wall_x_cm=float(point["wall_x_cm"]),
                wall_y_cm=float(point["wall_y_cm"]),
            )
        )
    return parsed

