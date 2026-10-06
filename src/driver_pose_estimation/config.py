from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _as_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True, slots=True)
class Settings:
    # ---------------------------------------------------------
    # Camera
    # ---------------------------------------------------------
    camera_index: int = field(
        default_factory=lambda: int(
            os.getenv("CAMERA_INDEX", "0")
        )
    )

    camera_width: int = field(
        default_factory=lambda: int(
            os.getenv("CAMERA_WIDTH", "640")
        )
    )

    camera_height: int = field(
        default_factory=lambda: int(
            os.getenv("CAMERA_HEIGHT", "480")
        )
    )

    camera_fps: float = field(
        default_factory=lambda: float(
            os.getenv("CAMERA_FPS", "15")
        )
    )

    # ---------------------------------------------------------
    # Pose model
    # ---------------------------------------------------------
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "POSE_MODEL",
            "yolo11n-pose.pt",
        )
    )

    device: str = field(
        default_factory=lambda: os.getenv(
            "POSE_DEVICE",
            "cpu",
        )
    )

    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("POSE_CONFIDENCE", "0.25")
        )
    )

    iou_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("POSE_IOU", "0.45")
        )
    )

    keypoint_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("KEYPOINT_CONFIDENCE", "0.50")
        )
    )

    image_size: int = field(
        default_factory=lambda: int(
            os.getenv("POSE_IMAGE_SIZE", "640")
        )
    )

    process_every_n_frames: int = field(
        default_factory=lambda: int(
            os.getenv("PROCESS_EVERY_N_FRAMES", "1")
        )
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------
    display_video: bool = field(
        default_factory=lambda: _as_bool(
            "DISPLAY_VIDEO",
            True,
        )
    )

    mirror_camera: bool = field(
        default_factory=lambda: _as_bool(
            "MIRROR_CAMERA",
            True,
        )
    )

    draw_bounding_box: bool = field(
        default_factory=lambda: _as_bool(
            "DRAW_BOUNDING_BOX",
            True,
        )
    )

    draw_skeleton: bool = field(
        default_factory=lambda: _as_bool(
            "DRAW_SKELETON",
            True,
        )
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("OUTPUT_DIR", "outputs")
        )
    )

    save_history: bool = field(
        default_factory=lambda: _as_bool(
            "SAVE_HISTORY",
            True,
        )
    )

    log_every_n_frames: int = field(
        default_factory=lambda: int(
            os.getenv("LOG_EVERY_N_FRAMES", "10")
        )
    )

    @property
    def pose_history_path(self) -> Path:
        return self.output_dir / "pose_history.json"

    @property
    def pose_summary_path(self) -> Path:
        return self.output_dir / "pose_summary.json"

    def validate(self) -> None:
        if self.camera_index < 0:
            raise ValueError(
                "CAMERA_INDEX must be greater than or equal to zero"
            )

        if self.camera_width < 1:
            raise ValueError(
                "CAMERA_WIDTH must be positive"
            )

        if self.camera_height < 1:
            raise ValueError(
                "CAMERA_HEIGHT must be positive"
            )

        if self.camera_fps <= 0:
            raise ValueError(
                "CAMERA_FPS must be positive"
            )

        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "POSE_CONFIDENCE must be between 0 and 1"
            )

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(
                "POSE_IOU must be between 0 and 1"
            )

        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError(
                "KEYPOINT_CONFIDENCE must be between 0 and 1"
            )

        if self.image_size < 32:
            raise ValueError(
                "POSE_IMAGE_SIZE must be at least 32"
            )

        if self.process_every_n_frames < 1:
            raise ValueError(
                "PROCESS_EVERY_N_FRAMES must be at least 1"
            )

        if self.log_every_n_frames < 1:
            raise ValueError(
                "LOG_EVERY_N_FRAMES must be at least 1"
            )
