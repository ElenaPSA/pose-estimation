import cv2
import numpy as np
import pytest

from driver_pose_estimation.models import (
    BoundingBox,
    FramePoseResult,
    Keypoint,
    PoseDetection,
)
from driver_pose_estimation.morphology_visualizer import MorphologyVisualizer


def _result() -> FramePoseResult:
    keypoints = [
        Keypoint(
            index=index,
            name=f"point_{index}",
            x=100.0,
            y=100.0 + index * 5,
            confidence=0.9,
        )
        for index in range(17)
    ]
    driver = PoseDetection(
        bounding_box=BoundingBox(50.0, 20.0, 150.0, 220.0),
        keypoints=keypoints,
    )
    return FramePoseResult(
        frame_number=1,
        timestamp_seconds=0.1,
        processing_time_ms=10.0,
        detected_people=1,
        driver=driver,
        poses=[driver],
    )


def test_comparison_has_two_labeled_pose_panels() -> None:
    visualizer = MorphologyVisualizer(keypoint_threshold=0.5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    comparison = visualizer.create_comparison(
        frame,
        _result(),
        frame,
        _result(),
    )

    assert comparison.shape == (532, 1280, 3)
    assert np.count_nonzero(comparison[:, :, 1]) > 0


def test_save_comparison_creates_png(tmp_path) -> None:
    visualizer = MorphologyVisualizer(keypoint_threshold=0.5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    output = tmp_path / "nested" / "morphology.png"

    visualizer.save_comparison(
        output,
        frame,
        _result(),
        frame,
        _result(),
    )

    saved = cv2.imread(str(output))
    assert saved is not None
    assert saved.shape == (532, 1280, 3)


def test_comparison_rejects_missing_driver_pose() -> None:
    visualizer = MorphologyVisualizer(keypoint_threshold=0.5)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    missing_pose = FramePoseResult(
        frame_number=1,
        timestamp_seconds=0.1,
        processing_time_ms=10.0,
        detected_people=0,
        driver=None,
    )

    with pytest.raises(ValueError, match="No driver pose"):
        visualizer.create_comparison(
            frame,
            _result(),
            frame,
            missing_pose,
        )
