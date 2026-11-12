from driver_pose_estimation.models import BoundingBox, Keypoint, PoseDetection
from driver_pose_estimation.person_height_video import (
    height_estimation_score,
    select_tracked_person,
)


def make_pose(
    x1: float,
    y1: float,
    width: float,
    height: float,
    box_confidence: float,
    keypoint_confidence: float,
) -> PoseDetection:
    keypoints = [
        Keypoint(
            index,
            str(index),
            x1 + width / 2,
            y1 + height - 4,
            keypoint_confidence,
        )
        for index in (0, 5, 6, 11, 12, 13, 14, 15, 16)
    ]
    return PoseDetection(
        bounding_box=BoundingBox(
            x1,
            y1,
            x1 + width,
            y1 + height,
            box_confidence,
        ),
        keypoints=keypoints,
    )


def test_height_estimation_score_prefers_larger_clear_observation() -> None:
    farther = make_pose(100, 100, 80, 180, 0.95, 0.95)
    closer = make_pose(100, 30, 120, 350, 0.85, 0.85)

    assert height_estimation_score(closer, 480) > height_estimation_score(
        farther, 480
    )


def test_tracking_keeps_person_with_overlapping_box() -> None:
    previous = make_pose(100, 30, 80, 400, 0.8, 0.8)
    same_person = make_pose(110, 30, 80, 400, 0.8, 0.8)
    larger_other_person = make_pose(400, 20, 180, 420, 0.99, 0.99)

    selected = select_tracked_person(
        [larger_other_person, same_person],
        previous.bounding_box,
        640,
        480,
    )

    assert selected is same_person