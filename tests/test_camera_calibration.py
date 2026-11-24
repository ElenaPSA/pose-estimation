from dataclasses import replace
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
    calibration = CameraCalibration.simulated_for_image((1928, 1088))
    intrinsics = calibration.to_dict()["intrinsics"]

    assert calibration.camera_matrix[0, 0] == pytest.approx(830.0)
    assert calibration.camera_matrix[1, 1] == pytest.approx(1078.0)
    assert calibration.camera_matrix[0, 2] == pytest.approx(964.0)
    assert calibration.camera_matrix[1, 2] == pytest.approx(544.0)
    assert intrinsics["image_size_px"] == [1928, 1088]
    assert intrinsics["horizontal_fov_deg"] == pytest.approx(98.5434, abs=0.001)
    assert intrinsics["vertical_fov_deg"] == pytest.approx(53.5545, abs=0.001)
    assert intrinsics["reported_diagonal_fov_deg"] == 113.83
    assert IMPLIED_DIAGONAL_FOV_DEG == pytest.approx(103.4053, abs=0.001)


def test_1920_by_1080_stream_scales_native_intrinsics() -> None:
    calibration = CameraCalibration.simulated_for_image((1920, 1080))

    assert calibration.horizontal_fov_deg == pytest.approx(98.5434, abs=0.001)
    assert calibration.vertical_fov_deg == pytest.approx(53.5545, abs=0.001)
    assert calibration.reported_diagonal_fov_deg == 113.83
    assert calibration.camera_matrix[0, 0] == pytest.approx(830.0 * 1920 / 1928)
    assert calibration.camera_matrix[1, 1] == pytest.approx(1078.0 * 1080 / 1088)
    assert calibration.camera_matrix[0, 2] == pytest.approx(
        (964.0 + 0.5) * 1920 / 1928 - 0.5
    )
    assert calibration.camera_matrix[1, 2] == pytest.approx(
        (544.0 + 0.5) * 1080 / 1088 - 0.5
    )


def test_anticlockwise_rotated_video_uses_native_intrinsics() -> None:
    native = CameraCalibration.simulated_for_image((1928, 1088))
    calibration = CameraCalibration.simulated_for_image((1080, 1920))

    assert calibration.horizontal_fov_deg == pytest.approx(98.5434, abs=0.001)
    assert calibration.vertical_fov_deg == pytest.approx(53.5545, abs=0.001)
    assert calibration.reported_diagonal_fov_deg == 113.83
    assert calibration.camera_matrix[0, 0] == pytest.approx(
        native.camera_matrix[1, 1] * 1080 / 1088
    )
    assert calibration.camera_matrix[1, 1] == pytest.approx(
        native.camera_matrix[0, 0] * 1920 / 1928
    )
    assert calibration.camera_matrix[0, 2] == pytest.approx(
        (544.0 + 0.5) * 1080 / 1088 - 0.5
    )
    assert calibration.camera_matrix[1, 2] == pytest.approx(
        (1928 - 0.5 - 964.0) * 1920 / 1928 - 0.5
    )


def test_anticlockwise_rotation_preserves_native_pixel_rays() -> None:
    native = CameraCalibration.simulated_for_image((1928, 1088))
    rotated = CameraCalibration.simulated_for_image((1080, 1920))
    native_ray = np.asarray([0.25, -0.10, 1.0])
    native_pixel = native.camera_matrix @ native_ray
    rotated_pixel = np.asarray([
        (native_pixel[1] + 0.5) * 1080 / 1088 - 0.5,
        (1928 - 1 - native_pixel[0] + 0.5) * 1920 / 1928 - 0.5,
        1.0,
    ])

    rotated_ray = np.linalg.solve(rotated.camera_matrix, rotated_pixel)

    assert rotated_ray == pytest.approx([native_ray[1], -native_ray[0], 1.0])


@pytest.mark.parametrize(("pitch_deg", "roll_deg"), [(-10, 0), (0, 0), (-15, 3), (-5, -3)])
def test_rotated_intrinsics_recover_synthetic_height(pitch_deg, roll_deg) -> None:
    calibration = CameraCalibration.simulated_for_image(
        (1080, 1920), pitch_deg=pitch_deg, roll_deg=roll_deg, yaw_deg=70
    )
    camera_position = np.asarray(calibration.camera_position_vehicle_m)
    rotation = calibration.camera_to_vehicle_rotation
    foot_position = camera_position + np.asarray(
        [-3 * np.sin(np.radians(70)), 3 * np.cos(np.radians(70)), -1.15]
    )
    head_position = foot_position + np.asarray([0, 0, 1.75])

    def project(position):
        ray = rotation.T @ (position - camera_position)
        pixel = calibration.camera_matrix @ ray
        return tuple(pixel[:2] / pixel[2])

    estimate = calibration.estimate_height_cm(
        project(head_position), project(foot_position), (1080, 1920)
    )

    assert estimate.height_cm == pytest.approx(175, abs=0.1)


def test_starting_pitch_tilts_optical_axis_downward() -> None:
    calibration = CameraCalibration.simulated_for_image((1080, 1920))
    optical_axis = calibration.camera_to_vehicle_rotation @ np.asarray([0, 0, 1])

    assert optical_axis == pytest.approx([
        -np.sin(np.radians(93)) * np.cos(np.radians(11.5)),
        np.cos(np.radians(93)) * np.cos(np.radians(11.5)),
        -np.sin(np.radians(11.5)),
    ])
    assert np.linalg.det(calibration.camera_to_vehicle_rotation) == pytest.approx(1.0)


def test_fitted_default_pose_estimates_reference_frame() -> None:
    calibration = CameraCalibration.simulated_for_image((720, 1280))
    estimate = calibration.estimate_height_cm(
        (318.6667175, 236.8782959), (243.6583252, 1108.1826172), (720, 1280)
    )

    assert estimate.height_cm == pytest.approx(156.64, abs=0.02)


def test_yaw_changes_ground_coordinates_but_not_height() -> None:
    forward = CameraCalibration.simulated_for_image((1080, 1920), yaw_deg=0)
    left = CameraCalibration.simulated_for_image((1080, 1920), yaw_deg=70)
    head, foot = (500, 300), (500, 1500)

    forward_estimate = forward.estimate_height_cm(head, foot, (1080, 1920))
    left_estimate = left.estimate_height_cm(head, foot, (1080, 1920))

    assert left_estimate.height_cm == pytest.approx(forward_estimate.height_cm)
    assert left_estimate.vertical_alignment_residual_m == pytest.approx(
        forward_estimate.vertical_alignment_residual_m
    )
    assert left_estimate.foot_position_vehicle_m != pytest.approx(
        forward_estimate.foot_position_vehicle_m
    )


@pytest.mark.parametrize("yaw", [float("nan"), float("inf"), -181, 181])
def test_invalid_simulated_yaw_is_rejected(yaw: float) -> None:
    with pytest.raises(CameraCalibrationError, match="Invalid simulated"):
        CameraCalibration.simulated_for_image((1080, 1920), yaw_deg=yaw)


def test_incompatible_landscape_aspect_ratio_uses_generic_simulation() -> None:
    calibration = CameraCalibration.simulated_for_image((640, 480))

    assert calibration.horizontal_fov_deg == 70.0
    assert calibration.reported_diagonal_fov_deg is None
    assert calibration.camera_matrix[0, 0] == pytest.approx(calibration.camera_matrix[1, 1])


def test_mirror_position_uses_supplied_axes_and_ground_height() -> None:
    calibration = CameraCalibration.simulated_for_image((200, 250))
    assert calibration.camera_position_vehicle_m == (-0.85, -0.74, 0.87)
    assert calibration.ground_plane_z_m == pytest.approx(-0.28)
    assert CameraCalibration.from_dict(calibration.to_dict()).ground_plane_z_m == pytest.approx(-0.28)

    estimate = calibration.estimate_height_cm((100, 10), (100, 235), (200, 250))
    old_height = CameraCalibration.simulated_for_image(
        (200, 250), camera_height_m=0.87
    ).estimate_height_cm((100, 10), (100, 235), (200, 250))
    shifted = replace(
        calibration, camera_position_vehicle_m=(0.0, 0.0, 0.87)
    ).estimate_height_cm((100, 10), (100, 235), (200, 250))

    assert estimate.height_cm == pytest.approx(old_height.height_cm * 1.15 / 0.87)
    assert estimate.height_cm == pytest.approx(shifted.height_cm)
    assert estimate.foot_position_vehicle_m[0] == pytest.approx(shifted.foot_position_vehicle_m[0] - 0.85)
    assert estimate.foot_position_vehicle_m[1] == pytest.approx(shifted.foot_position_vehicle_m[1] - 0.74)
    assert estimate.foot_position_vehicle_m[2] == pytest.approx(-0.28)


def test_nonzero_ground_plane_recovers_known_height() -> None:
    data = calibration_data({
        "position_m": [0, 0, 0.87],
        "rotation_matrix": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "ground_plane_z_m": -0.28,
    })
    calibration = CameraCalibration.from_dict(data)
    head_y = 240 - 500 * (1.75 - 1.15) / 3
    foot_y = 240 + 500 * 1.15 / 3

    estimate = calibration.estimate_height_cm((320, head_y), (320, foot_y), (640, 480))

    assert estimate.height_cm == pytest.approx(175, abs=0.1)
    assert estimate.foot_position_vehicle_m[2] == pytest.approx(-0.28)


def test_missing_ground_plane_defaults_to_zero() -> None:
    calibration = CameraCalibration.from_dict(calibration_data({
        "position_m": [0, 0, 1.2],
        "rotation_matrix": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    }))

    assert calibration.ground_plane_z_m == 0.0


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