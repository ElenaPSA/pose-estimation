from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from ultralytics import YOLO

from .config import Settings
from .models import (
    BoundingBox,
    FramePoseResult,
    Keypoint,
    PoseDetection,
)


logger = logging.getLogger(__name__)


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class DriverPoseEstimator:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        logger.info(
            "Loading pose model: %s",
            settings.model_path,
        )

        self.model = YOLO(
            settings.model_path
        )

    def estimate(
        self,
        frame: Any,
        frame_number: int,
        timestamp_seconds: float,
    ) -> FramePoseResult:
        inference_started = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            classes=[0],
            device=self.settings.device,
            imgsz=self.settings.image_size,
            verbose=False,
        )

        poses: list[PoseDetection] = []

        for result in results:
            if (
                result.boxes is None
                or result.keypoints is None
            ):
                continue

            boxes = result.boxes.xyxy.cpu().numpy()

            box_confidences = (
                result.boxes.conf.cpu().numpy()
                if result.boxes.conf is not None
                else np.ones(len(boxes))
            )

            keypoint_xy = (
                result.keypoints.xy.cpu().numpy()
            )

            keypoint_confidence = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(
                    (
                        len(keypoint_xy),
                        len(COCO_KEYPOINT_NAMES),
                    )
                )
            )

            detection_count = min(
                len(boxes),
                len(keypoint_xy),
            )

            for person_index in range(
                detection_count
            ):
                x1, y1, x2, y2 = boxes[
                    person_index
                ]

                bounding_box = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(
                        box_confidences[
                            person_index
                        ]
                    ),
                )

                keypoints: list[Keypoint] = []

                person_points = keypoint_xy[
                    person_index
                ]

                person_confidences = (
                    keypoint_confidence[
                        person_index
                    ]
                )

                point_count = min(
                    len(person_points),
                    len(COCO_KEYPOINT_NAMES),
                )

                for index in range(
                    point_count
                ):
                    x, y = person_points[index]

                    keypoints.append(
                        Keypoint(
                            index=index,
                            name=(
                                COCO_KEYPOINT_NAMES[
                                    index
                                ]
                            ),
                            x=float(x),
                            y=float(y),
                            confidence=float(
                                person_confidences[
                                    index
                                ]
                            ),
                        )
                    )

                poses.append(
                    PoseDetection(
                        bounding_box=bounding_box,
                        keypoints=keypoints,
                    )
                )

        driver = self._select_driver(
            poses
        )

        processing_time_ms = (
            time.perf_counter()
            - inference_started
        ) * 1000.0

        return FramePoseResult(
            frame_number=frame_number,
            timestamp_seconds=timestamp_seconds,
            processing_time_ms=processing_time_ms,
            detected_people=len(poses),
            driver=driver,
            poses=poses,
        )

    @staticmethod
    def _select_driver(
        poses: list[PoseDetection],
    ) -> PoseDetection | None:
        """
        Select the largest detected person as the driver.

        For a fixed driver-facing camera, the driver is normally
        the largest visible person. This strategy can later be
        replaced by a configured driver region of interest.
        """
        if not poses:
            return None

        return max(
            poses,
            key=lambda pose: (
                pose.bounding_box.area
            ),
        )
