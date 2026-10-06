from __future__ import annotations

from typing import Any

import cv2

from .config import Settings
from .models import (
    FramePoseResult,
    PoseDetection,
)


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


class PoseRenderer:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

    def render(
        self,
        frame: Any,
        result: FramePoseResult,
    ) -> Any:
        output = frame.copy()

        for pose in result.poses:
            is_driver = pose is result.driver

            self._draw_pose(
                output,
                pose,
                is_driver=is_driver,
            )

        status = (
            "Driver detected"
            if result.driver is not None
            else "No driver detected"
        )

        status_color = (
            (0, 255, 0)
            if result.driver is not None
            else (0, 0, 255)
        )

        cv2.putText(
            output,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            (
                f"Frame: {result.frame_number} | "
                f"People: {result.detected_people} | "
                f"Inference: "
                f"{result.processing_time_ms:.1f} ms"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            "Press Q or ESC to stop",
            (20, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output

    def _draw_pose(
        self,
        image: Any,
        pose: PoseDetection,
        is_driver: bool,
    ) -> None:
        box_color = (
            (0, 255, 0)
            if is_driver
            else (0, 165, 255)
        )

        point_color = (
            (0, 0, 255)
            if is_driver
            else (255, 0, 255)
        )

        skeleton_color = (
            (255, 0, 0)
            if is_driver
            else (255, 255, 0)
        )

        if self.settings.draw_bounding_box:
            box = pose.bounding_box

            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                box_color,
                3 if is_driver else 2,
            )

            label = (
                "Driver"
                if is_driver
                else "Person"
            )

            cv2.putText(
                image,
                (
                    f"{label} "
                    f"{box.confidence:.2f}"
                ),
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2,
                cv2.LINE_AA,
            )

        visible_points = {
            point.index: point
            for point in pose.keypoints
            if (
                point.confidence
                >= self.settings.keypoint_threshold
            )
        }

        if self.settings.draw_skeleton:
            for start, end in SKELETON_EDGES:
                start_point = visible_points.get(
                    start
                )

                end_point = visible_points.get(
                    end
                )

                if (
                    start_point is None
                    or end_point is None
                ):
                    continue

                cv2.line(
                    image,
                    (
                        int(start_point.x),
                        int(start_point.y),
                    ),
                    (
                        int(end_point.x),
                        int(end_point.y),
                    ),
                    skeleton_color,
                    2,
                    cv2.LINE_AA,
                )

        for point in visible_points.values():
            cv2.circle(
                image,
                (
                    int(point.x),
                    int(point.y),
                ),
                4,
                point_color,
                -1,
                cv2.LINE_AA,
            )