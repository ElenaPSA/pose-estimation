from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# POSE DETECTION MODELS
# ============================================================


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Keypoint:
    index: int
    name: str
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    @property
    def width(self) -> float:
        return max(
            0.0,
            self.x2 - self.x1,
        )

    @property
    def height(self) -> float:
        return max(
            0.0,
            self.y2 - self.y1,
        )

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


@dataclass
class PoseDetection:
    bounding_box: BoundingBox
    keypoints: list[Keypoint] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FramePoseResult:
    frame_number: int
    timestamp_seconds: float
    processing_time_ms: float
    detected_people: int
    driver: Optional[PoseDetection]
    poses: list[PoseDetection] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# POSTURE AND COMFORT MODELS
# ============================================================


@dataclass
class PostureMetrics:
    # ---------------------------------------------------------
    # Upper-body geometry
    # ---------------------------------------------------------

    torso_angle_deg: Optional[float] = None
    shoulder_tilt_deg: Optional[float] = None
    head_offset_ratio: Optional[float] = None
    torso_compression_ratio: Optional[float] = None
    torso_length_px: Optional[float] = None

    leaning_forward: bool = False
    leaning_left: bool = False
    leaning_right: bool = False
    shoulders_unbalanced: bool = False

    # ---------------------------------------------------------
    # Lower-body geometry
    # ---------------------------------------------------------

    left_hip_angle_deg: Optional[float] = None
    right_hip_angle_deg: Optional[float] = None

    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None
    average_knee_angle_deg: Optional[float] = None

    left_thigh_angle_deg: Optional[float] = None
    right_thigh_angle_deg: Optional[float] = None

    knee_height_difference_ratio: Optional[float] = None
    ankle_height_difference_ratio: Optional[float] = None

    legs_too_extended: bool = False
    legs_too_flexed: bool = False
    knees_asymmetric: bool = False
    thighs_asymmetric: bool = False

    # ---------------------------------------------------------
    # Visibility
    # ---------------------------------------------------------

    upper_body_visible: bool = False
    hips_visible: bool = False
    knees_visible: bool = False
    ankles_visible: bool = False
    lower_body_visible: bool = False
    full_body_visible: bool = False

    visibility: float = 0.0
    valid: bool = False


# ============================================================
# SEAT MODELS
# ============================================================


class SeatAxis(str, Enum):
    FORE_AFT = "fore_aft"
    HEIGHT = "height"
    BACKREST = "backrest"
    LUMBAR = "lumbar"
    CUSHION_TILT = "cushion_tilt"


@dataclass(frozen=True)
class SeatAdjustment:
    axis: SeatAxis
    delta: float
    reason: str


@dataclass
class SeatState:
    fore_aft: float = 0.0
    height: float = 0.0
    backrest: float = 0.0
    lumbar: float = 0.0
    cushion_tilt: float = 0.0


# ============================================================
# VEHICLE AND AGENT MODELS
# ============================================================


@dataclass
class VehicleState:
    speed_kph: float = 0.0
    parking_brake: bool = True
    ignition_on: bool = True


@dataclass
class AgentDecision:
    message: str
    adjustments: list[SeatAdjustment] = field(
        default_factory=list
    )
    requires_confirmation: bool = False
    queued_until_stopped: bool = False