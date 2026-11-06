from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

from .models import PostureMetrics


@dataclass
class ComfortAssessment:
    overall_score: Optional[float]
    comfort_level: str
    upper_body_score: Optional[float]
    lower_body_score: Optional[float]
    pedal_reach_score: Optional[float]
    thigh_support_score: Optional[float]
    lumbar_support_score: Optional[float]
    steering_reach_score: Optional[float]
    fatigue_risk_score: Optional[float]
    confidence_score: float
    upper_body_available: bool
    lower_body_available: bool
    knee_angles_available: bool
    steering_reach_available: bool
    pose_visibility: float
    explanations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ComfortEvaluator:
    def __init__(self, history_size: int = 30) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")

        self._overall_score_history: deque[float] = deque(maxlen=history_size)
        self._forward_lean_history: deque[bool] = deque(maxlen=history_size)
        self._lateral_lean_history: deque[bool] = deque(maxlen=history_size)
        self._shoulder_imbalance_history: deque[bool] = deque(maxlen=history_size)
        self._leg_extension_history: deque[bool] = deque(maxlen=history_size)
        self._leg_flexion_history: deque[bool] = deque(maxlen=history_size)
        self._knee_asymmetry_history: deque[bool] = deque(maxlen=history_size)
        self._thigh_asymmetry_history: deque[bool] = deque(maxlen=history_size)

    def evaluate(self, posture: PostureMetrics) -> ComfortAssessment:
        explanations: list[str] = []
        recommendations: list[str] = []
        warnings: list[str] = []

        self._update_temporal_history(posture)
        confidence_score = self._confidence_score(posture)

        upper_body_score = self._evaluate_upper_body(
            posture, explanations, recommendations
        )
        lower_body_score = self._evaluate_lower_body(
            posture, explanations, recommendations
        )
        pedal_reach_score = self._evaluate_pedal_reach(
            posture, explanations, recommendations
        )
        thigh_support_score = self._evaluate_thigh_support(
            posture, explanations, recommendations
        )
        lumbar_support_score = self._evaluate_lumbar_support(
            posture, explanations, recommendations
        )
        steering_reach_score = self._evaluate_steering_reach(explanations)
        fatigue_risk_score = self._evaluate_postural_fatigue(
            posture, explanations, recommendations
        )

        raw_overall_score = self._weighted_average(
            (
                (upper_body_score, 0.25),
                (lower_body_score, 0.25),
                (pedal_reach_score, 0.20),
                (thigh_support_score, 0.10),
                (lumbar_support_score, 0.15),
                (steering_reach_score, 0.05),
            )
        )

        if raw_overall_score is None:
            overall_score = None
            warnings.append(
                "There is not enough visible pose information to calculate "
                "an overall comfort score."
            )
        else:
            self._overall_score_history.append(raw_overall_score)
            overall_score = self._round_score(
                self._average_float(self._overall_score_history)
            )

        if confidence_score < 50.0:
            warnings.append(
                "Pose confidence is low. Treat the assessment as provisional."
            )
        if not posture.upper_body_visible:
            warnings.append("Upper-body comfort could not be evaluated.")
        if not posture.lower_body_visible:
            warnings.append(
                "The hips and both knees must be visible for lower-body evaluation."
            )
        if not posture.ankles_visible:
            warnings.append(
                "Both ankles must be visible to calculate knee angles and pedal reach."
            )

        return ComfortAssessment(
            overall_score=overall_score,
            comfort_level=self.comfort_level(overall_score),
            upper_body_score=self._optional_round(upper_body_score),
            lower_body_score=self._optional_round(lower_body_score),
            pedal_reach_score=self._optional_round(pedal_reach_score),
            thigh_support_score=self._optional_round(thigh_support_score),
            lumbar_support_score=self._optional_round(lumbar_support_score),
            steering_reach_score=self._optional_round(steering_reach_score),
            fatigue_risk_score=self._optional_round(fatigue_risk_score),
            confidence_score=self._round_score(confidence_score),
            upper_body_available=posture.upper_body_visible,
            lower_body_available=posture.lower_body_visible,
            knee_angles_available=posture.average_knee_angle_deg is not None,
            steering_reach_available=False,
            pose_visibility=round(self._clamp(posture.visibility, 0.0, 1.0), 3),
            explanations=self._deduplicate_text(explanations),
            recommendations=self._deduplicate_text(recommendations),
            warnings=self._deduplicate_text(warnings),
        )

    def _evaluate_upper_body(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        if not posture.upper_body_visible:
            return None

        score = 100.0
        if posture.leaning_forward:
            score -= 22.0
            explanations.append("A forward-leaning posture was detected.")
            recommendations.append("Review backrest inclination and lumbar support.")
        if posture.leaning_left:
            score -= 12.0
            explanations.append("The upper body appears displaced toward the left.")
        if posture.leaning_right:
            score -= 12.0
            explanations.append("The upper body appears displaced toward the right.")
        if posture.shoulders_unbalanced:
            score -= 16.0
            explanations.append("The shoulder line appears unbalanced.")
            recommendations.append("Check lateral posture and back support.")
        if (
            posture.shoulder_tilt_deg is not None
            and abs(posture.shoulder_tilt_deg) >= 12.0
        ):
            score -= 6.0

        return self._clamp_score(score)

    def _evaluate_lower_body(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        if not posture.lower_body_visible:
            return None

        score = 100.0
        if posture.legs_too_extended:
            score -= 25.0
            explanations.append("The legs appear highly extended.")
            recommendations.append("Consider moving the seat slightly forward.")
        if posture.legs_too_flexed:
            score -= 25.0
            explanations.append("The legs appear strongly flexed.")
            recommendations.append("Consider moving the seat slightly backward.")
        if posture.knees_asymmetric:
            score -= 18.0
            explanations.append("The left and right knee angles appear asymmetric.")
            recommendations.append("Check pedal access and support for both legs.")
        if posture.thighs_asymmetric:
            score -= 15.0
            explanations.append("The thigh positions appear asymmetric.")
            recommendations.append("Review seat-cushion support and lateral posture.")
        if (
            posture.knee_height_difference_ratio is not None
            and posture.knee_height_difference_ratio > 0.25
        ):
            score -= 10.0
            explanations.append("A noticeable difference between knee heights was detected.")
        if (
            posture.ankle_height_difference_ratio is not None
            and posture.ankle_height_difference_ratio > 0.30
        ):
            score -= 8.0
            explanations.append("The ankle positions appear vertically asymmetric.")

        return self._clamp_score(score)

    def _evaluate_pedal_reach(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        angle = posture.average_knee_angle_deg
        if angle is None:
            return None
        if angle < 70.0:
            explanations.append("The knees appear extremely flexed.")
            recommendations.append("Move the seat backward and reassess pedal reach.")
            return 25.0
        if angle < 85.0:
            explanations.append("The knees appear strongly flexed.")
            recommendations.append("Consider moving the seat slightly backward.")
            return 50.0
        if angle < 95.0:
            explanations.append("The knees appear moderately flexed.")
            return 75.0
        if angle <= 125.0:
            explanations.append("The observed knee angle is in the prototype comfort range.")
            return 95.0
        if angle <= 145.0:
            explanations.append("The legs appear moderately extended.")
            return 75.0
        if angle <= 160.0:
            explanations.append("The legs appear highly extended.")
            recommendations.append("Consider moving the seat slightly forward.")
            return 50.0

        explanations.append("The knees appear almost straight.")
        recommendations.append("Move the seat forward and reassess pedal reach.")
        return 25.0

    def _evaluate_thigh_support(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        if not posture.lower_body_visible:
            return None

        score = 92.0
        if posture.thighs_asymmetric:
            score -= 28.0
            explanations.append(
                "Asymmetric thigh positions may indicate uneven cushion support."
            )
            recommendations.append("Review the simulated seat-cushion tilt.")
        if (
            posture.knee_height_difference_ratio is not None
            and posture.knee_height_difference_ratio > 0.20
        ):
            score -= 15.0

        left_angle = posture.left_thigh_angle_deg
        right_angle = posture.right_thigh_angle_deg
        if left_angle is not None and right_angle is not None:
            if self._angular_difference(left_angle, right_angle) >= 25.0:
                score -= 12.0

        return self._clamp_score(score)

    def _evaluate_lumbar_support(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        if not posture.upper_body_visible:
            return None

        score = 95.0
        if posture.leaning_forward:
            score -= 30.0
            explanations.append(
                "The forward posture may indicate insufficient back or lumbar support."
            )
            recommendations.append(
                "Increase simulated lumbar support or review the backrest position."
            )
        if posture.shoulders_unbalanced:
            score -= 15.0
        if posture.leaning_left or posture.leaning_right:
            score -= 10.0

        return self._clamp_score(score)

    @staticmethod
    def _evaluate_steering_reach(
        explanations: list[str],
    ) -> Optional[float]:
        explanations.append(
            "Steering reach is unavailable because elbow-angle metrics are not "
            "yet included in PostureMetrics."
        )
        return None

    def _evaluate_postural_fatigue(
        self,
        posture: PostureMetrics,
        explanations: list[str],
        recommendations: list[str],
    ) -> Optional[float]:
        if not posture.valid:
            return None

        histories = (
            self._forward_lean_history,
            self._lateral_lean_history,
            self._shoulder_imbalance_history,
            self._leg_extension_history,
            self._leg_flexion_history,
            self._knee_asymmetry_history,
            self._thigh_asymmetry_history,
        )
        if not any(histories):
            return None

        risk = (
            self._true_ratio(self._forward_lean_history) * 28.0
            + self._true_ratio(self._lateral_lean_history) * 14.0
            + self._true_ratio(self._shoulder_imbalance_history) * 14.0
            + self._true_ratio(self._leg_extension_history) * 16.0
            + self._true_ratio(self._leg_flexion_history) * 16.0
            + self._true_ratio(self._knee_asymmetry_history) * 6.0
            + self._true_ratio(self._thigh_asymmetry_history) * 6.0
        )
        risk = self._clamp_score(risk)

        if risk >= 60.0:
            explanations.append(
                "Recent pose history shows persistent discomfort indicators."
            )
            recommendations.append(
                "Review the seating position when the vehicle is safely parked."
            )
        elif risk >= 35.0:
            explanations.append(
                "Some posture deviations persisted across recent observations."
            )

        return risk

    def _confidence_score(self, posture: PostureMetrics) -> float:
        visibility_score = self._clamp(posture.visibility, 0.0, 1.0) * 100.0
        coverage_score = 0.0
        if posture.upper_body_visible:
            coverage_score += 35.0
        if posture.hips_visible:
            coverage_score += 20.0
        if posture.knees_visible:
            coverage_score += 25.0
        if posture.ankles_visible:
            coverage_score += 20.0

        return self._clamp_score(
            visibility_score * 0.65 + coverage_score * 0.35
        )

    def _update_temporal_history(self, posture: PostureMetrics) -> None:
        if not posture.valid:
            return
        if posture.upper_body_visible:
            self._forward_lean_history.append(posture.leaning_forward)
            self._lateral_lean_history.append(
                posture.leaning_left or posture.leaning_right
            )
            self._shoulder_imbalance_history.append(
                posture.shoulders_unbalanced
            )
        if posture.lower_body_visible:
            self._leg_extension_history.append(posture.legs_too_extended)
            self._leg_flexion_history.append(posture.legs_too_flexed)
            self._knee_asymmetry_history.append(posture.knees_asymmetric)
            self._thigh_asymmetry_history.append(posture.thighs_asymmetric)

    @staticmethod
    def _weighted_average(
        components: tuple[tuple[Optional[float], float], ...],
    ) -> Optional[float]:
        weighted_sum = 0.0
        available_weight = 0.0

        for score, weight in components:
            if score is None:
                continue
            weighted_sum += score * weight
            available_weight += weight

        if available_weight <= 0.0:
            return None
        return weighted_sum / available_weight

    @staticmethod
    def comfort_level(score: Optional[float]) -> str:
        if score is None:
            return "Unavailable"
        if score >= 90.0:
            return "Excellent"
        if score >= 75.0:
            return "Good"
        if score >= 60.0:
            return "Acceptable"
        if score >= 40.0:
            return "Poor"
        return "Critical"

    @staticmethod
    def _true_ratio(values: deque[bool]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _average_float(values: deque[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _optional_round(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(value, 1)

    @staticmethod
    def _round_score(value: float) -> float:
        return round(ComfortEvaluator._clamp_score(value), 1)

    @staticmethod
    def _clamp_score(value: float) -> float:
        return ComfortEvaluator._clamp(value, 0.0, 100.0)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    @staticmethod
    def _angular_difference(first: float, second: float) -> float:
        difference = abs(first - second)
        return min(difference, 360.0 - difference)

    @staticmethod
    def _deduplicate_text(values: list[str]) -> list[str]:
        unique_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_values.append(normalized)

        return unique_values
