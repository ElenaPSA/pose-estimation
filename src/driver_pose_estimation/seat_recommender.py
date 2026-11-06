from __future__ import annotations

from .models import (
    PostureMetrics,
    SeatAdjustment,
    SeatAxis,
)


class SeatRecommender:
    def recommend(
        self,
        posture: PostureMetrics,
        complaint: str,
    ) -> list[SeatAdjustment]:
        normalized: str = complaint.strip().lower()
        recommendations: list[SeatAdjustment] = []

        self._add_complaint_recommendations(
            normalized,
            recommendations,
        )

        if posture.valid:
            self._add_pose_recommendations(
                posture,
                recommendations,
            )

        return self._deduplicate(
            recommendations
        )

    @staticmethod
    def _add_complaint_recommendations(
        complaint: str,
        recommendations: list[SeatAdjustment],
    ) -> None:
        if any(
            phrase in complaint
            for phrase in (
                "too far",
                "cannot reach",
                "can't reach",
                "pedals are far",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.FORE_AFT,
                    delta=1.0,
                    reason=(
                        "The driver reported being "
                        "too far from the controls."
                    ),
                )
            )

        if any(
            phrase in complaint
            for phrase in (
                "too close",
                "too near",
                "steering wheel",
                "knees too bent",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.FORE_AFT,
                    delta=-1.0,
                    reason=(
                        "The driver reported being "
                        "too close to the controls."
                    ),
                )
            )

        if any(
            phrase in complaint
            for phrase in (
                "lower back",
                "back pain",
                "lumbar",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.LUMBAR,
                    delta=1.0,
                    reason=(
                        "The driver reported "
                        "lower-back discomfort."
                    ),
                )
            )

        if any(
            phrase in complaint
            for phrase in (
                "too low",
                "cannot see",
                "visibility",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.HEIGHT,
                    delta=1.0,
                    reason=(
                        "The driver reported that "
                        "the seat is too low."
                    ),
                )
            )

        if any(
            phrase in complaint
            for phrase in (
                "too high",
                "headroom",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.HEIGHT,
                    delta=-1.0,
                    reason=(
                        "The driver reported that "
                        "the seat is too high."
                    ),
                )
            )

        if any(
            phrase in complaint
            for phrase in (
                "thigh",
                "under the legs",
                "leg support",
            )
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.CUSHION_TILT,
                    delta=0.5,
                    reason=(
                        "The driver reported insufficient "
                        "thigh support."
                    ),
                )
            )

    @staticmethod
    def _add_pose_recommendations(
        posture: PostureMetrics,
        recommendations: list[SeatAdjustment],
    ) -> None:
        if posture.leaning_forward:
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.BACKREST,
                    delta=-0.5,
                    reason=(
                        "The detected torso geometry "
                        "suggests a forward posture."
                    ),
                )
            )

        if (
            posture.leaning_left
            or posture.leaning_right
            or posture.shoulders_unbalanced
        ):
            recommendations.append(
                SeatAdjustment(
                    axis=SeatAxis.LUMBAR,
                    delta=0.5,
                    reason=(
                        "The upper-body pose suggests "
                        "asymmetric support."
                    ),
                )
            )

        if posture.lower_body_visible:
            if posture.legs_too_extended:
                recommendations.append(
                    SeatAdjustment(
                        axis=SeatAxis.FORE_AFT,
                        delta=1.0,
                        reason=(
                            "The detected knee angles suggest "
                            "that the legs are highly extended."
                        ),
                    )
                )

            if posture.legs_too_flexed:
                recommendations.append(
                    SeatAdjustment(
                        axis=SeatAxis.FORE_AFT,
                        delta=-1.0,
                        reason=(
                            "The detected knee angles suggest "
                            "that the legs are highly flexed."
                        ),
                    )
                )

            if posture.thighs_asymmetric:
                recommendations.append(
                    SeatAdjustment(
                        axis=SeatAxis.CUSHION_TILT,
                        delta=0.5,
                        reason=(
                            "The detected thigh positions "
                            "appear asymmetric."
                        ),
                    )
                )

            if posture.knees_asymmetric:
                recommendations.append(
                    SeatAdjustment(
                        axis=SeatAxis.HEIGHT,
                        delta=0.5,
                        reason=(
                            "The detected knee angles "
                            "appear asymmetric."
                        ),
                    )
                )

    @staticmethod
    def _deduplicate(
        recommendations: list[SeatAdjustment],
    ) -> list[SeatAdjustment]:
        selected: dict[
            SeatAxis,
            SeatAdjustment,
        ] = {}

        for recommendation in recommendations:
            if recommendation.axis not in selected:
                selected[
                    recommendation.axis
                ] = recommendation

        return list(selected.values())[:2]