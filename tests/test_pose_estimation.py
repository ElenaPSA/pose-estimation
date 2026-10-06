from driver_pose_estimation.models import (
    BoundingBox,
    PoseDetection,
)
from driver_pose_estimation.pose_estimator import (
    DriverPoseEstimator,
)


def test_select_driver_uses_largest_box() -> None:
    small = PoseDetection(
        bounding_box=BoundingBox(
            0,
            0,
            100,
            100,
            0.9,
        )
    )

    large = PoseDetection(
        bounding_box=BoundingBox(
            0,
            0,
            200,
            200,
            0.8,
        )
    )

    selected = DriverPoseEstimator._select_driver(
        [small, large]
    )

    assert selected is large