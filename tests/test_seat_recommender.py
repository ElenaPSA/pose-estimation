from driver_pose_estimation.models import (
    PostureMetrics,
    SeatAxis,
)
from driver_pose_estimation.seat_recommender import (
    SeatRecommender,
)


def test_lower_back_complaint() -> None:
    recommender = SeatRecommender()

    recommendations = recommender.recommend(
        posture=PostureMetrics(
            visibility=0.9,
            valid=True,
        ),
        complaint="I have lower back discomfort",
    )

    assert any(
        item.axis == SeatAxis.LUMBAR
        for item in recommendations
    )