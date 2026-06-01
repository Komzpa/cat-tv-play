"""Wall-plane cat tracking for jump-height review and live summaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WallDetection:
    """A cat candidate already transformed into wall centimeters."""

    t: float
    x_cm: float
    y_cm: float
    confidence: float = 0.5
    area_px: int = 0
    source: str = "detector"


@dataclass(frozen=True)
class TrackOutput:
    """One tracker step after prediction, gating, and optional update."""

    t: float
    x_cm: float
    y_cm: float
    vx_cm_s: float
    vy_cm_s: float
    accepted: WallDetection | None
    accepted_raw_y_cm: float | None
    reason: str
    confirmed: bool


@dataclass(frozen=True)
class AcceptedJumpPoint:
    """Accepted raw and filtered wall height for peak extraction."""

    t: float
    raw_y_cm: float
    filtered_y_cm: float
    confidence: float


def wall_detection_from_measurement(
    measurement: object,
    *,
    t: float,
    area_px: int = 0,
    source: str = "measurement",
) -> WallDetection:
    """Adapt a wall-centimeter measurement point into tracker input."""

    wall_x_cm = getattr(measurement, "wall_x_cm", None)
    wall_y_cm = getattr(measurement, "wall_y_cm", None)
    if wall_x_cm is None or wall_y_cm is None:
        raise ValueError("measurement must have wall_x_cm and wall_y_cm before tracking")
    return WallDetection(
        t=t,
        x_cm=float(wall_x_cm),
        y_cm=float(wall_y_cm),
        confidence=float(getattr(measurement, "confidence", 0.0) or 0.0),
        area_px=area_px,
        source=str(getattr(measurement, "source", None) or source),
    )


class CatWallKalmanTracker:
    """2D wall-plane tracker with outlier gates before Kalman updates.

    False detections are rejected before they can mutate the state. Peak height
    should be read from accepted raw detections, because Kalman smoothing can
    suppress the actual apex.
    """

    def __init__(
        self,
        *,
        min_y_cm: float = 0.0,
        max_y_cm: float = 260.0,
        max_horizontal_speed_cm_s: float = 500.0,
        max_vertical_speed_cm_s: float = 850.0,
        gate_d2: float = 13.8,
        accel_noise_cm_s2: float = 2500.0,
        base_sigma_x_cm: float = 8.0,
        base_sigma_y_cm: float = 10.0,
        low_conf_extra_sigma_cm: float = 35.0,
        min_init_confidence: float = 0.35,
        confirm_hits: int = 2,
        max_misses_before_reset: int = 8,
    ) -> None:
        self.min_y_cm = min_y_cm
        self.max_y_cm = max_y_cm
        self.max_horizontal_speed_cm_s = max_horizontal_speed_cm_s
        self.max_vertical_speed_cm_s = max_vertical_speed_cm_s
        self.gate_d2 = gate_d2
        self.accel_noise_cm_s2 = accel_noise_cm_s2
        self.base_sigma_x_cm = base_sigma_x_cm
        self.base_sigma_y_cm = base_sigma_y_cm
        self.low_conf_extra_sigma_cm = low_conf_extra_sigma_cm
        self.min_init_confidence = min_init_confidence
        self.confirm_hits = confirm_hits
        self.max_misses_before_reset = max_misses_before_reset

        self.state: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self.last_t: float | None = None
        self.hits = 0
        self.misses = 0

    def reset(self) -> None:
        """Forget the current track."""

        self.state = None
        self.cov = None
        self.last_t = None
        self.hits = 0
        self.misses = 0

    def step(self, t: float, detections: Iterable[WallDetection]) -> TrackOutput | None:
        """Advance the tracker by one timestamp."""

        detections = list(detections)
        if self.state is None:
            return self._initialize(t, detections)

        assert self.cov is not None
        assert self.last_t is not None

        dt = max(1e-3, min(0.5, t - self.last_t))
        pred_state, pred_cov = self._predict(dt)

        best_detection: WallDetection | None = None
        best_score: float | None = None
        best_residual: np.ndarray | None = None
        best_s: np.ndarray | None = None
        best_r: np.ndarray | None = None

        for detection in detections:
            if self._physical_gate(pred_state, detection, dt) is not None:
                continue

            residual, s, r = self._innovation(pred_state, pred_cov, detection)
            d2 = float(residual.T @ np.linalg.inv(s) @ residual)
            if d2 > self.gate_d2:
                continue

            confidence_bonus = 2.0 * max(0.0, min(1.0, detection.confidence))
            area_bonus = min(1.0, detection.area_px / 2000.0)
            score = d2 - confidence_bonus - area_bonus

            if best_score is None or score < best_score:
                best_score = score
                best_detection = detection
                best_residual = residual
                best_s = s
                best_r = r

        if best_detection is None:
            self.state = pred_state
            self.cov = pred_cov
            self.last_t = t
            self.misses += 1

            if self.misses > self.max_misses_before_reset:
                self.reset()
                return None

            return TrackOutput(
                t=t,
                x_cm=float(pred_state[0]),
                y_cm=float(pred_state[1]),
                vx_cm_s=float(pred_state[2]),
                vy_cm_s=float(pred_state[3]),
                accepted=None,
                accepted_raw_y_cm=None,
                reason="predicted_no_accepted_detection",
                confirmed=self.hits >= self.confirm_hits,
            )

        assert best_residual is not None
        assert best_s is not None
        assert best_r is not None

        h = np.array(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        k = pred_cov @ h.T @ np.linalg.inv(best_s)
        updated_state = pred_state + k @ best_residual

        identity = np.eye(4)
        ikh = identity - k @ h
        updated_cov = ikh @ pred_cov @ ikh.T + k @ best_r @ k.T

        self.state = updated_state
        self.cov = updated_cov
        self.last_t = t
        self.hits += 1
        self.misses = 0

        return TrackOutput(
            t=t,
            x_cm=float(updated_state[0]),
            y_cm=float(updated_state[1]),
            vx_cm_s=float(updated_state[2]),
            vy_cm_s=float(updated_state[3]),
            accepted=best_detection,
            accepted_raw_y_cm=float(best_detection.y_cm),
            reason="accepted_detection",
            confirmed=self.hits >= self.confirm_hits,
        )

    def _initialize(self, t: float, detections: list[WallDetection]) -> TrackOutput | None:
        plausible = [
            detection
            for detection in detections
            if self.min_y_cm <= detection.y_cm <= self.max_y_cm and detection.confidence >= self.min_init_confidence
        ]
        if not plausible:
            return None

        best = max(
            plausible,
            key=lambda detection: (
                detection.confidence,
                min(detection.area_px, 5000),
                -abs(detection.y_cm),
            ),
        )
        self.state = np.array([best.x_cm, best.y_cm, 0.0, 0.0], dtype=float)
        self.cov = np.diag([20.0**2, 20.0**2, 400.0**2, 700.0**2])
        self.last_t = t
        self.hits = 1
        self.misses = 0

        return TrackOutput(
            t=t,
            x_cm=float(best.x_cm),
            y_cm=float(best.y_cm),
            vx_cm_s=0.0,
            vy_cm_s=0.0,
            accepted=best,
            accepted_raw_y_cm=float(best.y_cm),
            reason="initialized",
            confirmed=False,
        )

    def _predict(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        assert self.state is not None
        assert self.cov is not None

        f = np.array(
            (
                (1.0, 0.0, dt, 0.0),
                (0.0, 1.0, 0.0, dt),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=float,
        )
        g = np.array(
            (
                (0.5 * dt * dt, 0.0),
                (0.0, 0.5 * dt * dt),
                (dt, 0.0),
                (0.0, dt),
            ),
            dtype=float,
        )
        q = (self.accel_noise_cm_s2**2) * (g @ g.T)
        pred_state = f @ self.state
        pred_cov = f @ self.cov @ f.T + q
        return pred_state, pred_cov

    def _measurement_noise(self, detection: WallDetection) -> np.ndarray:
        confidence = max(0.0, min(1.0, detection.confidence))
        extra = (1.0 - confidence) * self.low_conf_extra_sigma_cm
        area_penalty = 20.0 if detection.area_px and detection.area_px < 200 else 0.0
        sigma_x = self.base_sigma_x_cm + extra + area_penalty
        sigma_y = self.base_sigma_y_cm + extra + area_penalty
        return np.diag([sigma_x**2, sigma_y**2])

    def _innovation(
        self,
        pred_state: np.ndarray,
        pred_cov: np.ndarray,
        detection: WallDetection,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = np.array(
            (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
            ),
            dtype=float,
        )
        z = np.array([detection.x_cm, detection.y_cm], dtype=float)
        r = self._measurement_noise(detection)
        residual = z - h @ pred_state
        s = h @ pred_cov @ h.T + r
        return residual, s, r

    def _physical_gate(self, pred_state: np.ndarray, detection: WallDetection, dt: float) -> str | None:
        if detection.y_cm < self.min_y_cm:
            return "below_floor"
        if detection.y_cm > self.max_y_cm:
            return "above_wall_limit"

        dx = abs(detection.x_cm - float(pred_state[0]))
        dy = abs(detection.y_cm - float(pred_state[1]))
        max_dx = self.max_horizontal_speed_cm_s * dt + 35.0
        max_dy = self.max_vertical_speed_cm_s * dt + 45.0
        if dx > max_dx:
            return "horizontal_teleport"
        if dy > max_dy:
            return "vertical_teleport"
        return None


def confirmed_peak_height_cm(
    points: Iterable[AcceptedJumpPoint],
    *,
    min_confidence: float = 0.35,
) -> float | None:
    """Return the highest accepted raw wall height."""

    accepted = [point for point in points if point.confidence >= min_confidence]
    if not accepted:
        return None
    return max(point.raw_y_cm for point in accepted)
