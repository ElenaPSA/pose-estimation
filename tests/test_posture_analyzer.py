from driver_pose_estimation.config import Settings
from driver_pose_estimation.models import (
    BoundingBox,
    FramePoseResult,
    Keypoint,
    PoseDetection,
)
from driver_pose_estimation.posture_analyzer import (
    PostureAnalyzer,
)


def test_valid_posture_result() -> None:
    points = [
        Keypoint(
            index=index,
            name=f"point_{index}",
            x=100.0,
            y=100.0,
            confidence=0.9,
        )
        for index in range(17)
    ]

    points[0] = Keypoint(
        0,
        "nose",
        100.0,
        80.0,
        0.9,
    )
    points[5] = Keypoint(
        5,
        "left_shoulder",
        80.0,
        120.0,
        0.9,
    )
    points[6] = Keypoint(
        6,
        "right_shoulder",
        120.0,
        120.0,
        0.9,
    )
    points[11] = Keypoint(
        11,
        "left_hip",
        85.0,
        200.0,
        0.9,
    )
    points[12] = Keypoint(
        12,
        "right_hip",
        115.0,
        200.0,
        0.9,
    )

    driver = PoseDetection(
        bounding_box=BoundingBox(
            50.0,
            50.0,
            150.0,
            250.0,
            0.9,
        ),
        keypoints=points,
    )

    frame = FramePoseResult(
        frame_number=1,
        timestamp_seconds=0.1,
        processing_time_ms=10.0,
        detected_people=1,
        driver=driver,
        poses=[driver],
    )

    analyzer = PostureAnalyzer(
        Settings()
    )

    result = analyzer.analyze(frame)

    assert result.valid is True