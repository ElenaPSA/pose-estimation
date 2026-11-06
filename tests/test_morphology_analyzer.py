import pytest

from driver_pose_estimation.models import (
    BoundingBox,
    FramePoseResult,
    Keypoint,
    PoseDetection,
)
from driver_pose_estimation.morphology_analyzer import MorphologyAnalyzer


def _result(
    missing: set[int] | None = None,
) -> FramePoseResult:
    coordinates = {
        0: (100.0, 20.0),
        1: (95.0, 18.0),
        2: (105.0, 18.0),
        3: (90.0, 22.0),
        4: (110.0, 22.0),
        5: (80.0, 60.0),
        6: (120.0, 60.0),
        11: (85.0, 110.0),
        12: (115.0, 110.0),
        13: (85.0, 170.0),
        14: (115.0, 170.0),
        15: (85.0, 230.0),
        16: (115.0, 230.0),
    }
    keypoints = [
        Keypoint(
            index=index,
            name=f"point_{index}",
            x=coordinates.get(index, (0.0, 0.0))[0],
            y=coordinates.get(index, (0.0, 0.0))[1],
            confidence=0.0 if missing and index in missing else 0.9,
        )
        for index in range(17)
    ]
    driver = PoseDetection(
        bounding_box=BoundingBox(50.0, 10.0, 150.0, 240.0),
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


def test_frontal_profile_uses_normalized_image_measurements() -> None:
    analyzer = MorphologyAnalyzer(keypoint_threshold=0.5)

    result = analyzer.analyze_view(_result(), "frontal")

    assert result.face_to_ankle_span_px == pytest.approx(210.0, abs=1.0)
    assert result.shoulder_width_over_span == pytest.approx(40.0 / 210.0, abs=0.001)
    assert result.hip_width_over_span == pytest.approx(30.0 / 210.0, abs=0.001)
    assert result.upper_leg_length_over_span == pytest.approx(60.0 / 210.0, abs=0.001)
    assert result.keypoint_confidence == 0.9


def test_frontal_view_rejects_missing_full_body_landmarks() -> None:
    analyzer = MorphologyAnalyzer(keypoint_threshold=0.5)

    with pytest.raises(ValueError, match="both shoulders"):
        analyzer.analyze_view(_result({6}), "frontal")


def test_lateral_view_allows_one_visible_body_chain() -> None:
    analyzer = MorphologyAnalyzer(keypoint_threshold=0.5)
    result = _result(
        {
            1,
            2,
            3,
            4,
            6,
            12,
            14,
            16,
        }
    )

    measurements = analyzer.analyze_view(result, "lateral")

    assert measurements.view == "lateral"
    assert measurements.shoulder_width_over_span is None
    assert measurements.upper_leg_length_over_span is not None


def test_profile_requires_the_correct_view_order() -> None:
    analyzer = MorphologyAnalyzer(keypoint_threshold=0.5)
    lateral = analyzer.analyze_view(_result(), "lateral")

    with pytest.raises(ValueError, match="one frontal"):
        analyzer.build_profile(frontal=lateral, lateral=lateral)


def test_profile_does_not_claim_physical_height_or_seat_settings() -> None:
    analyzer = MorphologyAnalyzer(keypoint_threshold=0.5)
    frontal = analyzer.analyze_view(_result(), "frontal")
    lateral = analyzer.analyze_view(_result(), "lateral")

    profile = analyzer.build_profile(frontal, lateral).to_dict()

    assert profile["physical_height_cm"] is None
    assert profile["seat_position_recommendation"] is None
    assert profile["frontal"]["torso_length_over_span"] is not None
    assert profile["lateral"]["lower_leg_length_over_span"] is not None
