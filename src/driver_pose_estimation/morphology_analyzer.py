from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

from .models import FramePoseResult, Keypoint

ViewName = Literal["frontal", "lateral"]

FACE_KEYPOINTS = (0, 1, 2, 3, 4)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass(frozen=True)
class ViewMorphology:
    view: ViewName
    face_to_ankle_span_px: float
    shoulder_width_over_span: float | None
    hip_width_over_span: float | None
    torso_length_over_span: float | None
    upper_leg_length_over_span: float | None
    lower_leg_length_over_span: float | None
    keypoint_confidence: float

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class MorphologyProfile:
    frontal: ViewMorphology
    lateral: ViewMorphology
    physical_height_cm: None = None
    seat_position_recommendation: None = None
    limitations: tuple[str, ...] = (
        "Measurements are 2D image-space proportions, not physical dimensions.",
        (
            "A face landmark is used as the upper-body anchor; the model does not "
            "detect the top of the head."
        ),
        "Perspective and pose can change apparent segment lengths.",
        "Seat-position recommendations require vehicle and seat calibration.",
    )

    def to_dict(self) -> dict:
        return {
            "frontal": self.frontal.to_dict(),
            "lateral": self.lateral.to_dict(),
            "physical_height_cm": self.physical_height_cm,
            "seat_position_recommendation": (self.seat_position_recommendation),
            "limitations": list(self.limitations),
        }


class MorphologyAnalyzer:
    def __init__(self, keypoint_threshold: float) -> None:
        if not 0.0 <= keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be between 0 and 1")
        self.keypoint_threshold = keypoint_threshold

    def analyze_view(
        self,
        result: FramePoseResult,
        view: ViewName,
    ) -> ViewMorphology:
        if view not in ("frontal", "lateral"):
            raise ValueError(f"Unsupported morphology view: {view}")
        if result.driver is None:
            raise ValueError(f"No driver pose detected in the {view} view")

        points = {
            point.index: point
            for point in result.driver.keypoints
            if point.confidence >= self.keypoint_threshold
        }
        face = self._center(points, FACE_KEYPOINTS)
        if face is None:
            raise ValueError(f"Face landmarks are not visible in the {view} view")

        ankles = self._center(
            points,
            (LEFT_ANKLE, RIGHT_ANKLE),
        )
        if ankles is None:
            raise ValueError(f"At least one ankle must be visible in the {view} view")

        if view == "frontal":
            required = (
                LEFT_SHOULDER,
                RIGHT_SHOULDER,
                LEFT_HIP,
                RIGHT_HIP,
                LEFT_KNEE,
                RIGHT_KNEE,
                LEFT_ANKLE,
                RIGHT_ANKLE,
            )
            if any(index not in points for index in required):
                raise ValueError(
                    "The frontal view requires both shoulders, hips, knees, "
                    "and ankles to be visible"
                )
        elif not any(
            all(self._has_segment(points, pair) for pair in side_chain)
            for side_chain in (
                (
                    (LEFT_SHOULDER, LEFT_HIP),
                    (LEFT_HIP, LEFT_KNEE),
                    (LEFT_KNEE, LEFT_ANKLE),
                ),
                (
                    (RIGHT_SHOULDER, RIGHT_HIP),
                    (RIGHT_HIP, RIGHT_KNEE),
                    (RIGHT_KNEE, RIGHT_ANKLE),
                ),
            )
        ):
            raise ValueError(
                "The lateral view requires a visible shoulder-to-hip-to-knee-"
                "to-ankle body chain on at least one side"
            )

        span = self._distance(face, ankles)
        if span < 1.0:
            raise ValueError(f"The face-to-ankle span is too small in the {view} view")

        left_shoulder = points.get(LEFT_SHOULDER)
        right_shoulder = points.get(RIGHT_SHOULDER)
        left_hip = points.get(LEFT_HIP)
        right_hip = points.get(RIGHT_HIP)

        shoulder_center = self._center(
            points,
            (LEFT_SHOULDER, RIGHT_SHOULDER),
        )
        hip_center = self._center(
            points,
            (LEFT_HIP, RIGHT_HIP),
        )

        upper_leg_lengths = self._segment_lengths(
            points,
            (
                (LEFT_HIP, LEFT_KNEE),
                (RIGHT_HIP, RIGHT_KNEE),
            ),
        )
        lower_leg_lengths = self._segment_lengths(
            points,
            (
                (LEFT_KNEE, LEFT_ANKLE),
                (RIGHT_KNEE, RIGHT_ANKLE),
            ),
        )

        confidence_indices = set(FACE_KEYPOINTS)
        confidence_indices.update(
            (
                LEFT_SHOULDER,
                RIGHT_SHOULDER,
                LEFT_HIP,
                RIGHT_HIP,
                LEFT_KNEE,
                RIGHT_KNEE,
                LEFT_ANKLE,
                RIGHT_ANKLE,
            )
        )
        confidences = [
            points[index].confidence for index in confidence_indices if index in points
        ]

        return ViewMorphology(
            view=view,
            face_to_ankle_span_px=round(span, 2),
            shoulder_width_over_span=self._ratio(
                self._distance(left_shoulder, right_shoulder)
                if left_shoulder is not None and right_shoulder is not None
                else None,
                span,
            ),
            hip_width_over_span=self._ratio(
                self._distance(left_hip, right_hip)
                if left_hip is not None and right_hip is not None
                else None,
                span,
            ),
            torso_length_over_span=self._ratio(
                self._distance(shoulder_center, hip_center)
                if shoulder_center is not None and hip_center is not None
                else None,
                span,
            ),
            upper_leg_length_over_span=self._ratio(
                self._mean(upper_leg_lengths),
                span,
            ),
            lower_leg_length_over_span=self._ratio(
                self._mean(lower_leg_lengths),
                span,
            ),
            keypoint_confidence=round(
                sum(confidences) / len(confidences),
                3,
            ),
        )

    @staticmethod
    def build_profile(
        frontal: ViewMorphology,
        lateral: ViewMorphology,
    ) -> MorphologyProfile:
        if frontal.view != "frontal" or lateral.view != "lateral":
            raise ValueError(
                "A profile requires one frontal and one lateral measurement"
            )
        return MorphologyProfile(
            frontal=frontal,
            lateral=lateral,
        )

    def _has_segment(
        self,
        points: dict[int, Keypoint],
        pair: tuple[int, int],
    ) -> bool:
        return pair[0] in points and pair[1] in points

    def _segment_lengths(
        self,
        points: dict[int, Keypoint],
        pairs: tuple[tuple[int, int], ...],
    ) -> list[float]:
        return [
            self._distance(points[first], points[second])
            for first, second in pairs
            if self._has_segment(points, (first, second))
        ]

    @staticmethod
    def _center(
        points: dict[int, Keypoint],
        indices: tuple[int, ...],
    ) -> Keypoint | None:
        visible = [points[index] for index in indices if index in points]
        if not visible:
            return None
        return Keypoint(
            index=-1,
            name="center",
            x=sum(point.x for point in visible) / len(visible),
            y=sum(point.y for point in visible) / len(visible),
            confidence=sum(point.confidence for point in visible) / len(visible),
        )

    @staticmethod
    def _distance(
        first: Keypoint,
        second: Keypoint,
    ) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _ratio(value: float | None, span: float) -> float | None:
        return round(value / span, 4) if value is not None else None
