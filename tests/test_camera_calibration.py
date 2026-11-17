import json

import cv2
import numpy as np
import pytest

from driver_pose_estimation.camera_calibration import (
    CameraCalibration,
    CameraCalibrationError,
    IMPLIED_DIAGONAL_FOV_DEG,
    calibrate_chessboard,
)


def calibration_data(camera_to_vehicle=None) -> dict:
    return {
        "schema_version": 1,
        "intrinsics": {
            "image_size_px": [640, 480],
            "camera_matrix": [
                [500, 0, 320],
                [0, 500, 240],
                [0, 0, 1],
            ],
            "distortion_coefficients": [0, 0, 0, 0, 0],
            "reprojection_rms_px": 0.2,
        },
        "camera_to_vehicle": camera_to_vehicle,
    }


def test_calibrated_ground_geometry_recovers_known_height() -> None:
    calibration = CameraCalibration.from_dict(
        calibration_data(
            {
                "position_m": [0, 0, 1.2],
                "rotation_matrix": [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
            }
        )
    )
    head_y = 240 - 500 * (1.75 - 1.2) / 3
    foot_y = 240 + 500 * 1.2 / 3

    estimate = calibration.estimate_height_cm(
        (320, head_y),
        (320, foot_y),
        (640, 480),
    )

    assert estimate.height_cm == pytest.approx(175, abs=0.1)
    assert estimate.vertical_alignment_residual_m < 1e-6


def test_metric_height_requires_vehicle_mount_pose() -> None:
    calibration = CameraCalibration.from_dict(calibration_data())

    with pytest.raises(CameraCalibrationError, match="requires camera_to_vehicle"):
        calibration.estimate_height_cm((320, 150), (320, 440), (640, 480))


def test_reported_mirror_camera_fov_builds_axis_specific_intrinsics() -> None:
    calibration = CameraCalibration.simulated_for_image((2048, 1280))
    intrinsics = calibration.to_dict()["intrinsics"]

    assert calibration.camera_matrix[0, 0] == pytest.approx(886.7038, abs=0.01)
    assert calibration.camera_matrix[1, 1] == pytest.approx(1361.9197, abs=0.01)
    assert intrinsics["image_size_px"] == [2048, 1280]
    assert intrinsics["horizontal_fov_deg"] == 98.22
    assert intrinsics["vertical_fov_deg"] == 50.34
    assert intrinsics["reported_diagonal_fov_deg"] == 113.83
    assert IMPLIED_DIAGONAL_FOV_DEG == pytest.approx(102.54, abs=0.01)


def test_incompatible_aspect_ratio_uses_generic_simulation() -> None:
    calibration = CameraCalibration.simulated_for_image((1080, 1920))

    assert calibration.horizontal_fov_deg == 70.0
    assert calibration.reported_diagonal_fov_deg is None
    assert calibration.camera_matrix[0, 0] == pytest.approx(calibration.camera_matrix[1, 1])


def test_mount_rotation_must_be_a_proper_rotation() -> None:
    data = calibration_data(
        {
            "position_m": [0, 0, 1.2],
            "rotation_matrix": np.zeros((3, 3)).tolist(),
        }
    )

    with pytest.raises(CameraCalibrationError, match="orthonormal"):
        CameraCalibration.from_dict(data)


def test_chessboard_calibration_requires_minimum_views(tmp_path) -> None:
    blank_path = tmp_path / "blank.png"
    assert cv2.imwrite(str(blank_path), np.zeros((480, 640, 3), dtype=np.uint8))

    with pytest.raises(CameraCalibrationError, match="at least 8"):
        calibrate_chessboard(
            [blank_path] * 8,
            board_size=(9, 6),
            square_size_m=0.025,
        )