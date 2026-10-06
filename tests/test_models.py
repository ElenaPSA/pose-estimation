from driver_pose_estimation.models import (
    BoundingBox,
)


def test_bounding_box_area() -> None:
    box = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=220,
        confidence=0.9,
    )

    assert box.width == 100
    assert box.height == 200
    assert box.area == 20_000