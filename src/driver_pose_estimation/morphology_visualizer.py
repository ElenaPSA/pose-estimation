from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import FramePoseResult, Keypoint

SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class MorphologyVisualizer:
    PANEL_WIDTH = 640
    PANEL_HEIGHT = 480
    HEADER_HEIGHT = 52
    BACKGROUND = (29, 31, 37)
    PANEL_BACKGROUND = (18, 20, 25)
    SKELETON_COLOR = (0, 255, 0)
    TEXT_COLOR = (245, 246, 248)
    MUTED_COLOR = (173, 179, 190)

    def __init__(self, keypoint_threshold: float) -> None:
        if not 0.0 <= keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be between 0 and 1")
        self.keypoint_threshold = keypoint_threshold

    def create_comparison(
        self,
        frontal_frame: Any,
        frontal_result: FramePoseResult,
        lateral_frame: Any,
        lateral_result: FramePoseResult,
    ) -> np.ndarray:
        frontal = self._annotate_view(
            frontal_frame,
            frontal_result,
            "A) FRONTAL",
        )
        lateral = self._annotate_view(
            lateral_frame,
            lateral_result,
            "B) LATERAL",
        )

        comparison = np.full(
            (
                self.HEADER_HEIGHT + self.PANEL_HEIGHT,
                self.PANEL_WIDTH * 2,
                3,
            ),
            self.BACKGROUND,
            dtype=np.uint8,
        )
        comparison[
            self.HEADER_HEIGHT :,
            : self.PANEL_WIDTH,
        ] = frontal
        comparison[
            self.HEADER_HEIGHT :,
            self.PANEL_WIDTH :,
        ] = lateral
        cv2.putText(
            comparison,
            "DRIVER MORPHOLOGY | NORMALIZED 2D POSE",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        return comparison

    def save_comparison(
        self,
        path: Path,
        frontal_frame: Any,
        frontal_result: FramePoseResult,
        lateral_frame: Any,
        lateral_result: FramePoseResult,
    ) -> np.ndarray:
        comparison = self.create_comparison(
            frontal_frame,
            frontal_result,
            lateral_frame,
            lateral_result,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), comparison):
            raise OSError(f"Could not save morphology visualization to {path}")
        return comparison

    def _annotate_view(
        self,
        frame: Any,
        result: FramePoseResult,
        label: str,
    ) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError(f"The {label} image is empty")
        if result.driver is None:
            raise ValueError(f"No driver pose is available for {label}")

        source_height, source_width = frame.shape[:2]
        scale = min(
            self.PANEL_WIDTH / source_width,
            self.PANEL_HEIGHT / source_height,
        )
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        panel = np.full(
            (self.PANEL_HEIGHT, self.PANEL_WIDTH, 3),
            self.PANEL_BACKGROUND,
            dtype=np.uint8,
        )
        left = (self.PANEL_WIDTH - resized_width) // 2
        top = (self.PANEL_HEIGHT - resized_height) // 2
        panel[top : top + resized_height, left : left + resized_width] = resized

        points = {
            point.index: point
            for point in result.driver.keypoints
            if point.confidence >= self.keypoint_threshold
        }

        def display_point(point: Keypoint) -> tuple[int, int]:
            return (
                round(point.x * scale + left),
                round(point.y * scale + top),
            )

        for first, second in SKELETON_EDGES:
            start = points.get(first)
            end = points.get(second)
            if start is None or end is None:
                continue
            cv2.line(
                panel,
                display_point(start),
                display_point(end),
                self.SKELETON_COLOR,
                3,
                cv2.LINE_AA,
            )

        for point in points.values():
            cv2.circle(
                panel,
                display_point(point),
                4,
                self.SKELETON_COLOR,
                -1,
                cv2.LINE_AA,
            )

        cv2.putText(
            panel,
            label,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Face landmarks shown; head outline is not estimated",
            (16, self.PANEL_HEIGHT - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            self.MUTED_COLOR,
            1,
            cv2.LINE_AA,
        )
        return panel
