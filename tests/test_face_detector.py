import numpy as np
import pytest

from driver_pose_estimation.face_detector import FaceHeightDetector
from driver_pose_estimation.models import BoundingBox, PoseDetection


class _StubYuNet:
    def __init__(self, detections: np.ndarray | None) -> None:
        self.detections = detections
        self.input_size = None

    def setInputSize(self, input_size: tuple[int, int]) -> None:
        self.input_size = input_size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        assert image.shape[:2] == (320, 320)
        return 1, self.detections


def _pose() -> PoseDetection:
    return PoseDetection(
        bounding_box=BoundingBox(50, 20, 150, 240),
    )


def test_yunet_face_box_is_mapped_back_to_original_image_coordinates() -> None:
    detection = np.zeros((1, 15), dtype=np.float32)
    detection[0, :4] = (100, 25, 50, 60)
    detection[0, 14] = 0.92
    model = _StubYuNet(detection)
    detector = FaceHeightDetector(detector=model)

    result = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8), _pose())

    assert model.input_size == (320, 320)
    assert result is not None
    assert result.x == pytest.approx(81.25)
    assert result.y == pytest.approx(21.875)
    assert result.width == pytest.approx(15.625)
    assert result.height == pytest.approx(18.75)
    assert result.confidence == pytest.approx(0.92)


def test_yunet_returns_none_when_no_face_is_detected() -> None:
    detector = FaceHeightDetector(detector=_StubYuNet(None))

    result = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8), _pose())

    assert result is None


def test_yunet_returns_none_for_pose_outside_image() -> None:
    detector = FaceHeightDetector(detector=_StubYuNet(None))
    pose = PoseDetection(bounding_box=BoundingBox(500, 500, 600, 700))

    result = detector.detect(np.zeros((300, 400, 3), dtype=np.uint8), pose)

    assert result is None
