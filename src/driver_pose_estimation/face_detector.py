from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .models import PoseDetection

DETECTOR_INPUT_SIZE = (320, 320)
FACE_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class FaceBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float


class _FaceDetectorModel(Protocol):
    def setInputSize(self, input_size: tuple[int, int]) -> None: ...

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]: ...


class FaceHeightDetector:
    def __init__(
        self,
        model_path: Path | None = None,
        detector: _FaceDetectorModel | None = None,
    ) -> None:
        self.model_path = model_path or (
            Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
        )
        if detector is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"YuNet face detector model is missing: {self.model_path}"
                )
            detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=DETECTOR_INPUT_SIZE,
                score_threshold=FACE_CONFIDENCE_THRESHOLD,
                nms_threshold=0.3,
                top_k=5000,
            )
        self._detector = detector
        self._detector.setInputSize(DETECTOR_INPUT_SIZE)

    def detect(self, image: np.ndarray, pose: PoseDetection) -> FaceBox | None:
        image_height, image_width = image.shape[:2]
        box = pose.bounding_box
        x1 = max(0, math.floor(box.x1))
        y1 = max(0, math.floor(box.y1))
        x2 = min(image_width, math.ceil(box.x2))
        y2 = min(image_height, math.ceil(box.y2))
        person_width = x2 - x1
        person_height = y2 - y1
        if person_width < 1 or person_height < 1:
            return None

        head_region_height = min(
            person_height,
            max(1, math.ceil(person_height * 0.4)),
        )
        region = image[y1 : y1 + head_region_height, x1:x2]
        region_height, region_width = region.shape[:2]
        scale = min(
            DETECTOR_INPUT_SIZE[0] / region_width,
            DETECTOR_INPUT_SIZE[1] / region_height,
        )
        resized_width = max(1, round(region_width * scale))
        resized_height = max(1, round(region_height * scale))
        resized = cv2.resize(
            region,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (DETECTOR_INPUT_SIZE[0] - resized_width) // 2
        pad_y = (DETECTOR_INPUT_SIZE[1] - resized_height) // 2
        canvas = np.zeros(
            (DETECTOR_INPUT_SIZE[1], DETECTOR_INPUT_SIZE[0], 3),
            dtype=image.dtype,
        )
        canvas[
            pad_y : pad_y + resized_height,
            pad_x : pad_x + resized_width,
        ] = resized

        _, detections = self._detector.detect(canvas)
        if detections is None or len(detections) == 0:
            return None
        detection = max(detections, key=lambda row: float(row[14]))
        confidence = float(detection[14])
        face_x = (float(detection[0]) - pad_x) / scale + x1
        face_y = (float(detection[1]) - pad_y) / scale + y1
        face_width = float(detection[2]) / scale
        face_height = float(detection[3]) / scale
        left = max(0.0, face_x)
        top = max(0.0, face_y)
        right = min(float(image_width), face_x + face_width)
        bottom = min(float(y1 + head_region_height), face_y + face_height)
        if right <= left or bottom <= top:
            return None
        return FaceBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
            confidence=confidence,
        )
