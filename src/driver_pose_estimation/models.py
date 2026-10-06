from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Keypoint:
    index: int
    name: str
    x: float
    y: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }


@dataclass(slots=True)
class PoseDetection:
    bounding_box: BoundingBox
    keypoints: list[Keypoint] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounding_box": (
                self.bounding_box.to_dict()
            ),
            "keypoints": [
                point.to_dict()
                for point in self.keypoints
            ],
        }


@dataclass(slots=True)
class FramePoseResult:
    frame_number: int
    timestamp_seconds: float
    processing_time_ms: float
    detected_people: int
    driver: PoseDetection | None
    poses: list[PoseDetection] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame_number,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "detected_people": (
                self.detected_people
            ),
            "driver": (
                self.driver.to_dict()
                if self.driver is not None
                else None
            ),
            "poses": [
                pose.to_dict()
                for pose in self.poses
            ],
        }