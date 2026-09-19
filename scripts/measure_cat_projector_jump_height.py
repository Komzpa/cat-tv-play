#!/usr/bin/env python3
"""Measure cat jump height from calibrated projector-camera frame points.

The script deliberately measures geometry only. It expects a wall-plane
calibration and human/model observations that identify the cat's upper point or
bounding box in image pixels; pose detection and leaderboard actions stay
outside this measurement layer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Homography:
    values: tuple[float, float, float, float, float, float, float, float]

    def project(self, x_px: float, y_px: float) -> tuple[float, float]:
        h11, h12, h13, h21, h22, h23, h31, h32 = self.values
        denom = h31 * x_px + h32 * y_px + 1.0
        if abs(denom) < 1e-9:
            raise ValueError("calibrated point projects to infinity")
        x_mm = (h11 * x_px + h12 * y_px + h13) / denom
        y_mm = (h21 * x_px + h22 * y_px + h23) / denom
        return x_mm, y_mm

    def unproject(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        h11, h12, h13, h21, h22, h23, h31, h32 = self.values
        inverse = _invert_3x3(
            (
                (h11, h12, h13),
                (h21, h22, h23),
                (h31, h32, 1.0),
            )
        )
        denom = inverse[2][0] * x_mm + inverse[2][1] * y_mm + inverse[2][2]
        if abs(denom) < 1e-9:
            raise ValueError("calibrated wall point unprojects to infinity")
        x_px = (inverse[0][0] * x_mm + inverse[0][1] * y_mm + inverse[0][2]) / denom
        y_px = (inverse[1][0] * x_mm + inverse[1][1] * y_mm + inverse[1][2]) / denom
        return x_px, y_px


@dataclass(frozen=True)
class Observation:
    frame_id: str
    x_px: float
    y_px: float
    confidence: str
    source: str
    measurement_role: str
    include_in_max: bool
    exclusion_reason: str


def _float(value: object, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


Matrix3x3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _invert_3x3(matrix: Matrix3x3) -> Matrix3x3:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1e-12:
        raise ValueError("homography matrix is not invertible")
    inv_det = 1.0 / determinant
    return (
        ((e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det),
        ((f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det),
        ((d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det),
    )


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]

    for pivot_index in range(size):
        best_index = max(range(pivot_index, size), key=lambda row_index: abs(augmented[row_index][pivot_index]))
        if abs(augmented[best_index][pivot_index]) < 1e-12:
            raise ValueError("calibration points do not define a stable homography")
        augmented[pivot_index], augmented[best_index] = augmented[best_index], augmented[pivot_index]

        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]

        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0:
                continue
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row_index], augmented[pivot_index], strict=True)
            ]

    return [row[-1] for row in augmented]


def compute_homography(
    image_points_px: Iterable[Iterable[float]],
    wall_points_mm: Iterable[Iterable[float]],
) -> Homography:
    image_points = [tuple(point) for point in image_points_px]
    wall_points = [tuple(point) for point in wall_points_mm]
    if len(image_points) != len(wall_points):
        raise ValueError("image_points_px and wall_points_mm must have the same length")
    if len(image_points) < 4:
        raise ValueError("at least four calibration point pairs are required")

    matrix: list[list[float]] = []
    vector: list[float] = []
    for index, (image_point, wall_point) in enumerate(zip(image_points, wall_points, strict=True)):
        if len(image_point) != 2 or len(wall_point) != 2:
            raise ValueError(f"calibration point {index} must contain exactly two coordinates")
        x_px, y_px = (_float(image_point[0], field="image x"), _float(image_point[1], field="image y"))
        x_mm, y_mm = (_float(wall_point[0], field="wall x"), _float(wall_point[1], field="wall y"))
        matrix.append([x_px, y_px, 1.0, 0.0, 0.0, 0.0, -x_mm * x_px, -x_mm * y_px])
        vector.append(x_mm)
        matrix.append([0.0, 0.0, 0.0, x_px, y_px, 1.0, -y_mm * x_px, -y_mm * y_px])
        vector.append(y_mm)

    if len(matrix) > 8:
        normal_matrix = [[0.0 for _ in range(8)] for _ in range(8)]
        normal_vector = [0.0 for _ in range(8)]
        for row, target in zip(matrix, vector, strict=True):
            for left_index, left_value in enumerate(row):
                normal_vector[left_index] += left_value * target
                for right_index, right_value in enumerate(row):
                    normal_matrix[left_index][right_index] += left_value * right_value
        matrix = normal_matrix
        vector = normal_vector

    return Homography(tuple(_solve_linear_system(matrix, vector)))  # type: ignore[arg-type]


def _read_calibration(path: Path) -> tuple[Homography, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("wall_plane_model"), dict):
        coefficients = payload["wall_plane_model"].get("coefficients") or {}
        homography = Homography(
            (
                _float(coefficients.get("uA"), field="wall_plane_model.coefficients.uA"),
                _float(coefficients.get("uB"), field="wall_plane_model.coefficients.uB"),
                _float(coefficients.get("uC"), field="wall_plane_model.coefficients.uC"),
                _float(coefficients.get("a"), field="wall_plane_model.coefficients.a"),
                _float(coefficients.get("b"), field="wall_plane_model.coefficients.b"),
                _float(coefficients.get("c"), field="wall_plane_model.coefficients.c"),
                _float(coefficients.get("g"), field="wall_plane_model.coefficients.g"),
                _float(coefficients.get("h"), field="wall_plane_model.coefficients.h"),
            )
        )
        return homography, _float(payload.get("floor_y_mm", 0.0), field="floor_y_mm")

    calibration = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else payload
    homography = compute_homography(
        calibration.get("image_points_px") or calibration.get("image_points"),
        calibration.get("wall_points_mm") or calibration.get("wall_points"),
    )
    floor_y_mm = _float(calibration.get("floor_y_mm", 0.0), field="floor_y_mm")
    return homography, floor_y_mm


def _parse_bbox_xywh(raw: str) -> tuple[float, float] | None:
    raw = raw.strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox_xywh must contain four comma-separated numbers")
    x_px, y_px, width_px, _height_px = (_float(part, field="bbox_xywh") for part in parts)
    return x_px + width_px / 2.0, y_px


def _bool_from_value(value: object) -> bool | None:
    if value in (None, ""):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "include", "included"}:
        return True
    if normalized in {"0", "false", "no", "n", "exclude", "excluded"}:
        return False
    raise ValueError("include_in_max must be true/false when provided")


def _default_exclusion_reason(row: dict[str, Any], measurement_role: str) -> str:
    explicit = str(row.get("exclusion_reason") or "").strip()
    if explicit:
        return explicit
    if measurement_role in {"ambiguous_blur", "motion_blur", "shadow", "reflection", "discarded"}:
        return f"measurement_role={measurement_role}"
    haystack = " ".join(
        str(row.get(key) or "") for key in ("confidence", "notes", "label_cat_playing", "label_cat_present")
    ).lower()
    if any(token in haystack for token in ("ambiguous", "discard", "empty", "not visibly", "not_visible")):
        return "annotation says point is not visibly on the cat"
    if "blur" in haystack and "visible" not in haystack:
        return "annotation is motion blur without a visible cat point"
    if "rough" in haystack:
        return "rough annotation excluded from max by default"
    return ""


def _observation_from_row(row: dict[str, Any], *, row_index: int) -> Observation | None:
    frame_id = str(row.get("frame_id") or row.get("image_relpath") or row.get("source_frame") or "").strip()
    if not frame_id:
        frame_id = f"row_{row_index:05d}"

    if str(row.get("label_cat_present") or "").strip().lower() in {"no", "false", "0"}:
        return None

    if row.get("top_x_px") not in (None, "") and row.get("top_y_px") not in (None, ""):
        x_px = _float(row.get("top_x_px"), field="top_x_px")
        y_px = _float(row.get("top_y_px"), field="top_y_px")
    else:
        candidate_bbox = ""
        if str(row.get("label_candidate_is_cat") or "").strip().lower() in {"yes", "true", "1"}:
            candidate_bbox = str(row.get("candidate_bbox_xywh") or "")
        parsed_bbox = _parse_bbox_xywh(str(row.get("bbox_xywh") or candidate_bbox or ""))
        if parsed_bbox is None:
            return None
        x_px, y_px = parsed_bbox

    measurement_role = str(row.get("measurement_role") or row.get("top_kind") or "visible_top").strip().lower()
    explicit_include = _bool_from_value(row.get("include_in_max"))
    exclusion_reason = _default_exclusion_reason(row, measurement_role)
    include_in_max = explicit_include if explicit_include is not None else not exclusion_reason
    if not include_in_max and not exclusion_reason:
        exclusion_reason = "include_in_max=false"

    return Observation(
        frame_id=frame_id,
        x_px=x_px,
        y_px=y_px,
        confidence=str(row.get("confidence") or "").strip(),
        source=str(row.get("source_chunk") or row.get("source") or "").strip(),
        measurement_role=measurement_role,
        include_in_max=include_in_max,
        exclusion_reason=exclusion_reason,
    )


def read_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    if path.suffix.lower() == ".jsonl":
        for row_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            observation = _observation_from_row(json.loads(line), row_index=row_index)
            if observation is not None:
                observations.append(observation)
        return observations

    with path.open(encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            observation = _observation_from_row(dict(row), row_index=row_index)
            if observation is not None:
                observations.append(observation)
    return observations


def measure_observations(
    observations: Iterable[Observation],
    *,
    homography: Homography,
    floor_y_mm: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for observation in observations:
        x_mm, y_mm = homography.project(observation.x_px, observation.y_px)
        height_mm = max(0.0, y_mm - floor_y_mm)
        records.append(
            {
                "frame_id": observation.frame_id,
                "cat_top_x_px": round(observation.x_px, 3),
                "cat_top_y_px": round(observation.y_px, 3),
                "wall_x_mm": round(x_mm, 1),
                "cat_top_height_mm": round(height_mm, 1),
                "cat_top_height_cm": round(height_mm / 10.0, 1),
                "confidence": observation.confidence,
                "measurement_role": observation.measurement_role,
                "include_in_max": observation.include_in_max,
                "exclusion_reason": observation.exclusion_reason,
                "source": observation.source,
            }
        )
    return records


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("no measured jump-height records to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, calibration_path: Path, observation_path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("no measured jump-height records to write")
    included_records = [record for record in records if record.get("include_in_max") is True]
    if not included_records:
        raise ValueError("no included jump-height records to write; mark one visible cat point include_in_max=true")
    maximum = max(included_records, key=lambda record: float(record["cat_top_height_mm"]))
    payload = {
        "kind": "cat_projector_jump_height_measurements_v1",
        "calibration_path": str(calibration_path),
        "observation_path": str(observation_path),
        "record_count": len(records),
        "included_record_count": len(included_records),
        "excluded_record_count": len(records) - len(included_records),
        "max_jump_height_mm": maximum["cat_top_height_mm"],
        "max_jump_height_cm": maximum["cat_top_height_cm"],
        "max_frame_id": maximum["frame_id"],
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="JSON with image_points_px and wall_points_mm.")
    parser.add_argument(
        "--observations", type=Path, required=True, help="CSV/JSONL rows with top_x_px/top_y_px or bbox_xywh."
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    homography, floor_y_mm = _read_calibration(args.calibration)
    records = measure_observations(read_observations(args.observations), homography=homography, floor_y_mm=floor_y_mm)
    _write_json(args.output_json, args.calibration, args.observations, records)
    if args.output_csv:
        _write_csv(args.output_csv, records)
    included = [record for record in records if record.get("include_in_max") is True]
    print(
        f"measured {len(records)} cat jump-height records; "
        f"included={len(included)}; max={max(record['cat_top_height_cm'] for record in included)} cm"
    )


if __name__ == "__main__":
    main()
