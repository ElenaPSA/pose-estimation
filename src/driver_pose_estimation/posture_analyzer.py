from __future__ import annotations

import math
from collections import deque

from .config import Settings
from .models import (
    FramePoseResult,
    Keypoint,
    PostureMetrics,
)


NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class PostureAnalyzer:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.confidence_threshold = (
            settings.keypoint_threshold
        )

        self.forward_threshold = (
            settings.posture_forward_threshold
        )

        self.lateral_threshold = (
            settings.posture_lateral_threshold
        )

        self.shoulder_tilt_threshold_degrees = (
            settings.shoulder_tilt_threshold_degrees
        )

        # Prototype thresholds in image-space pose angles.
        self.extended_leg_threshold_degrees = 155.0
        self.flexed_leg_threshold_degrees = 85.0
        self.knee_asymmetry_threshold_degrees = 18.0
        self.thigh_asymmetry_threshold_degrees = 15.0

        # Temporal smoothing prevents one noisy frame from
        # immediately changing the comfort assessment.
        self.smoothing_window_size = 7

        self._left_knee_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._right_knee_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._left_thigh_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._right_thigh_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._shoulder_tilt_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._head_offset_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._torso_ratio_history: deque[float] = deque(
            maxlen=self.smoothing_window_size
        )

        self._reference_torso_ratio: float | None = None

    def analyze(
        self,
        result: FramePoseResult,
    ) -> PostureMetrics:
        if result.driver is None:
            return PostureMetrics(
                valid=False,
            )

        points = {
            point.index: point
            for point in result.driver.keypoints
        }

        nose = self._visible(
            points.get(NOSE)
        )

        left_shoulder = self._visible(
            points.get(LEFT_SHOULDER)
        )

        right_shoulder = self._visible(
            points.get(RIGHT_SHOULDER)
        )

        left_hip = self._visible(
            points.get(LEFT_HIP)
        )

        right_hip = self._visible(
            points.get(RIGHT_HIP)
        )

        left_knee = self._visible(
            points.get(LEFT_KNEE)
        )

        right_knee = self._visible(
            points.get(RIGHT_KNEE)
        )

        left_ankle = self._visible(
            points.get(LEFT_ANKLE)
        )

        right_ankle = self._visible(
            points.get(RIGHT_ANKLE)
        )

        upper_body_visible = (
            left_shoulder is not None
            and right_shoulder is not None
        )

        hips_visible = (
            left_hip is not None
            and right_hip is not None
        )

        knees_visible = (
            left_knee is not None
            and right_knee is not None
        )

        ankles_visible = (
            left_ankle is not None
            and right_ankle is not None
        )

        lower_body_visible = (
            hips_visible
            and knees_visible
        )

        full_body_visible = (
            upper_body_visible
            and lower_body_visible
            and ankles_visible
        )

        if (
            not upper_body_visible
            and not lower_body_visible
        ):
            return PostureMetrics(
                upper_body_visible=False,
                hips_visible=hips_visible,
                knees_visible=knees_visible,
                ankles_visible=ankles_visible,
                lower_body_visible=False,
                full_body_visible=False,
                valid=False,
            )

        metrics = PostureMetrics(
            upper_body_visible=upper_body_visible,
            hips_visible=hips_visible,
            knees_visible=knees_visible,
            ankles_visible=ankles_visible,
            lower_body_visible=lower_body_visible,
            full_body_visible=full_body_visible,
            valid=True,
        )

        visible_points = [
            point
            for point in (
                nose,
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip,
                left_knee,
                right_knee,
                left_ankle,
                right_ankle,
            )
            if point is not None
        ]

        if visible_points:
            metrics.visibility = sum(
                point.confidence
                for point in visible_points
            ) / len(visible_points)

        if (
            upper_body_visible
            and left_shoulder is not None
            and right_shoulder is not None
        ):
            self._analyze_upper_body(
                metrics=metrics,
                nose=nose,
                left_shoulder=left_shoulder,
                right_shoulder=right_shoulder,
                left_hip=left_hip,
                right_hip=right_hip,
            )

        if (
            lower_body_visible
            and left_hip is not None
            and right_hip is not None
            and left_knee is not None
            and right_knee is not None
        ):
            self._analyze_lower_body(
                metrics=metrics,
                left_hip=left_hip,
                right_hip=right_hip,
                left_knee=left_knee,
                right_knee=right_knee,
                left_ankle=left_ankle,
                right_ankle=right_ankle,
            )

        return metrics

    def _analyze_upper_body(
        self,
        metrics: PostureMetrics,
        nose: Keypoint | None,
        left_shoulder: Keypoint,
        right_shoulder: Keypoint,
        left_hip: Keypoint | None,
        right_hip: Keypoint | None,
    ) -> None:
        shoulder_center_x = (
            left_shoulder.x
            + right_shoulder.x
        ) / 2.0

        shoulder_center_y = (
            left_shoulder.y
            + right_shoulder.y
        ) / 2.0

        shoulder_dx = (
            right_shoulder.x
            - left_shoulder.x
        )

        shoulder_dy = (
            right_shoulder.y
            - left_shoulder.y
        )

        shoulder_width = math.hypot(
            shoulder_dx,
            shoulder_dy,
        )

        if shoulder_width < 1.0:
            return

        raw_shoulder_tilt = math.degrees(
            math.atan2(
                shoulder_dy,
                max(
                    abs(shoulder_dx),
                    1e-6,
                ),
            )
        )

        self._shoulder_tilt_history.append(
            raw_shoulder_tilt
        )

        metrics.shoulder_tilt_deg = self._average(
            self._shoulder_tilt_history
        )

        metrics.shoulders_unbalanced = (
            abs(metrics.shoulder_tilt_deg)
            >= self.shoulder_tilt_threshold_degrees
        )

        if nose is not None:
            raw_head_offset = (
                nose.x - shoulder_center_x
            ) / shoulder_width

            self._head_offset_history.append(
                raw_head_offset
            )

            metrics.head_offset_ratio = self._average(
                self._head_offset_history
            )

            metrics.leaning_left = (
                metrics.head_offset_ratio
                < -self.lateral_threshold
            )

            metrics.leaning_right = (
                metrics.head_offset_ratio
                > self.lateral_threshold
            )

        if (
            left_hip is None
            or right_hip is None
        ):
            return

        hip_center_x = (
            left_hip.x
            + right_hip.x
        ) / 2.0

        hip_center_y = (
            left_hip.y
            + right_hip.y
        ) / 2.0

        torso_dx = (
            shoulder_center_x
            - hip_center_x
        )

        torso_dy = (
            hip_center_y
            - shoulder_center_y
        )

        torso_length = math.hypot(
            torso_dx,
            torso_dy,
        )

        if torso_length < 1.0:
            return

        metrics.torso_length_px = torso_length

        metrics.torso_angle_deg = math.degrees(
            math.atan2(
                torso_dx,
                max(
                    abs(torso_dy),
                    1e-6,
                ),
            )
        )

        current_torso_ratio = (
            torso_length
            / shoulder_width
        )

        self._torso_ratio_history.append(
            current_torso_ratio
        )

        smoothed_torso_ratio = self._average(
            self._torso_ratio_history
        )

        if self._reference_torso_ratio is None:
            self._reference_torso_ratio = (
                smoothed_torso_ratio
            )

        reference = max(
            self._reference_torso_ratio,
            1e-6,
        )

        metrics.torso_compression_ratio = (
            smoothed_torso_ratio
            / reference
        )

        metrics.leaning_forward = (
            metrics.torso_compression_ratio
            < 1.0 - self.forward_threshold
        )

    def _analyze_lower_body(
        self,
        metrics: PostureMetrics,
        left_hip: Keypoint,
        right_hip: Keypoint,
        left_knee: Keypoint,
        right_knee: Keypoint,
        left_ankle: Keypoint | None,
        right_ankle: Keypoint | None,
    ) -> None:
        hip_width = math.hypot(
            right_hip.x - left_hip.x,
            right_hip.y - left_hip.y,
        )

        hip_width = max(
            hip_width,
            1.0,
        )

        raw_left_thigh_angle = (
            self._segment_angle_from_horizontal(
                left_hip,
                left_knee,
            )
        )

        raw_right_thigh_angle = (
            self._segment_angle_from_horizontal(
                right_hip,
                right_knee,
            )
        )

        self._left_thigh_history.append(
            raw_left_thigh_angle
        )

        self._right_thigh_history.append(
            raw_right_thigh_angle
        )

        metrics.left_thigh_angle_deg = self._average(
            self._left_thigh_history
        )

        metrics.right_thigh_angle_deg = self._average(
            self._right_thigh_history
        )

        thigh_difference = self._angular_difference(
            metrics.left_thigh_angle_deg,
            metrics.right_thigh_angle_deg,
        )

        metrics.thighs_asymmetric = (
            thigh_difference
            >= self.thigh_asymmetry_threshold_degrees
        )

        metrics.knee_height_difference_ratio = (
            abs(
                left_knee.y
                - right_knee.y
            )
            / hip_width
        )

        if (
            left_ankle is None
            or right_ankle is None
        ):
            return

        raw_left_knee_angle = self._joint_angle(
            left_hip,
            left_knee,
            left_ankle,
        )

        raw_right_knee_angle = self._joint_angle(
            right_hip,
            right_knee,
            right_ankle,
        )

        if raw_left_knee_angle > 0.0:
            self._left_knee_history.append(
                raw_left_knee_angle
            )

        if raw_right_knee_angle > 0.0:
            self._right_knee_history.append(
                raw_right_knee_angle
            )

        if not self._left_knee_history:
            return

        if not self._right_knee_history:
            return

        metrics.left_knee_angle_deg = self._average(
            self._left_knee_history
        )

        metrics.right_knee_angle_deg = self._average(
            self._right_knee_history
        )

        metrics.average_knee_angle_deg = (
            metrics.left_knee_angle_deg
            + metrics.right_knee_angle_deg
        ) / 2.0

        metrics.legs_too_extended = (
            metrics.average_knee_angle_deg
            >= self.extended_leg_threshold_degrees
        )

        metrics.legs_too_flexed = (
            metrics.average_knee_angle_deg
            <= self.flexed_leg_threshold_degrees
        )

        metrics.knees_asymmetric = (
            abs(
                metrics.left_knee_angle_deg
                - metrics.right_knee_angle_deg
            )
            >= self.knee_asymmetry_threshold_degrees
        )

        metrics.ankle_height_difference_ratio = (
            abs(
                left_ankle.y
                - right_ankle.y
            )
            / hip_width
        )

    def _visible(
        self,
        point: Keypoint | None,
    ) -> Keypoint | None:
        if point is None:
            return None

        if (
            point.confidence
            < self.confidence_threshold
        ):
            return None

        return point

    @staticmethod
    def _segment_angle_from_horizontal(
        start: Keypoint,
        end: Keypoint,
    ) -> float:
        angle = math.degrees(
            math.atan2(
                end.y - start.y,
                end.x - start.x,
            )
        )

        return PostureAnalyzer._normalize_angle(
            angle
        )

    @staticmethod
    def _joint_angle(
        first: Keypoint,
        center: Keypoint,
        third: Keypoint,
    ) -> float:
        first_vector = (
            first.x - center.x,
            first.y - center.y,
        )

        second_vector = (
            third.x - center.x,
            third.y - center.y,
        )

        first_length = math.hypot(
            first_vector[0],
            first_vector[1],
        )

        second_length = math.hypot(
            second_vector[0],
            second_vector[1],
        )

        if (
            first_length < 1e-6
            or second_length < 1e-6
        ):
            return 0.0

        cosine = (
            first_vector[0]
            * second_vector[0]
            + first_vector[1]
            * second_vector[1]
        ) / (
            first_length
            * second_length
        )

        cosine = max(
            -1.0,
            min(
                1.0,
                cosine,
            ),
        )

        return math.degrees(
            math.acos(cosine)
        )

    @staticmethod
    def _average(
        values: deque[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _normalize_angle(
        angle: float,
    ) -> float:
        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    @staticmethod
    def _angular_difference(
        first: float,
        second: float,
    ) -> float:
        difference = abs(
            first - second
        )

        return min(
            difference,
            360.0 - difference,
        )