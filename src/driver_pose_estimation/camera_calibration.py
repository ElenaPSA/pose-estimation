"""Chessboard intrinsics and vehicle-frame monocular height geometry."""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

CAMERA_IMAGE_SIZE_PX = (1928, 1088)
SIMULATED_FX_PX = 830.0
SIMULATED_FY_PX = 1078.0
SIMULATED_CX_PX = 964.0
SIMULATED_CY_PX = 544.0
SIMULATED_HORIZONTAL_FOV_DEG = math.degrees(
    2 * math.atan(CAMERA_IMAGE_SIZE_PX[0] / (2 * SIMULATED_FX_PX))
)
SIMULATED_VERTICAL_FOV_DEG = math.degrees(
    2 * math.atan(CAMERA_IMAGE_SIZE_PX[1] / (2 * SIMULATED_FY_PX))
)
REPORTED_DIAGONAL_FOV_DEG = 113.83
IMPLIED_DIAGONAL_FOV_DEG = math.degrees(
    2
    * math.atan(
        math.hypot(
            math.tan(math.radians(SIMULATED_HORIZONTAL_FOV_DEG) / 2),
            math.tan(math.radians(SIMULATED_VERTICAL_FOV_DEG) / 2),
        )
    )
)
MIRROR_CAMERA_POSITION_VEHICLE_M = (-0.85, -0.74, 0.87)
SIMULATED_CAMERA_HEIGHT_M = 1.15
SIMULATED_CAMERA_HEIGHT_RANGE_M = (1.05, 1.25)
SIMULATED_CAMERA_YAW_DEG = 93.0
SIMULATED_CAMERA_PITCH_DEG = -11.5
SIMULATED_CAMERA_ROLL_DEG = -0.1
CAMERA_PITCH_SEARCH_RANGE_DEG = (-15.0, -5.0)
CAMERA_ROLL_SEARCH_RANGE_DEG = (-3.0, 3.0)
REFERENCE_VIDEO_IMAGE_SIZE_PX = (1080, 1920)
REFERENCE_VIDEO_FX_PX = 1074.0009061782127
REFERENCE_VIDEO_FY_PX = 2051.1354466621287
REFERENCE_VIDEO_PITCH_DEG = -5.8572611145829745
REFERENCE_VIDEO_ROLL_DEG = -8.519068841528615
REFERENCE_VIDEO_HORIZONTAL_FOV_DEG = math.degrees(
    2 * math.atan(REFERENCE_VIDEO_IMAGE_SIZE_PX[0] / (2 * REFERENCE_VIDEO_FX_PX))
)
REFERENCE_VIDEO_VERTICAL_FOV_DEG = math.degrees(
    2 * math.atan(REFERENCE_VIDEO_IMAGE_SIZE_PX[1] / (2 * REFERENCE_VIDEO_FY_PX))
)


class CameraCalibrationError(ValueError):
    """Invalid calibration data or degenerate camera geometry."""


@dataclass(frozen=True)
class HeightEstimate:
    height_cm: float
    vertical_alignment_residual_m: float
    foot_position_vehicle_m: tuple[float, float, float]


@dataclass(frozen=True)
class CameraCalibration:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    reprojection_rms_px: float | None
    camera_position_vehicle_m: tuple[float, float, float] | None = None
    camera_to_vehicle_rotation: np.ndarray | None = None
    horizontal_fov_deg: float | None = None
    vertical_fov_deg: float | None = None
    reported_diagonal_fov_deg: float | None = None
    ground_plane_z_m: float = 0.0

    @classmethod
    def from_dict(cls, data: object) -> CameraCalibration:
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise CameraCalibrationError("Calibration schema_version must be 1.")
        intrinsics = data.get("intrinsics")
        if not isinstance(intrinsics, dict):
            raise CameraCalibrationError("Calibration is missing intrinsics.")

        image_size = intrinsics.get("image_size_px")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in image_size)
        ):
            raise CameraCalibrationError(
                "intrinsics.image_size_px must be [width, height] in pixels."
            )

        matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise CameraCalibrationError("camera_matrix must be a finite 3x3 matrix.")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise CameraCalibrationError("Camera focal lengths must be positive.")
        if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-6):
            raise CameraCalibrationError("camera_matrix must end with [0, 0, 1].")

        distortion = np.asarray(
            intrinsics.get("distortion_coefficients"), dtype=np.float64
        ).reshape(-1)
        if distortion.size not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise CameraCalibrationError(
                "distortion_coefficients must contain 4, 5, 8, 12, or 14 finite values."
            )

        rms_value = intrinsics.get("reprojection_rms_px")
        rms = None if rms_value is None else float(rms_value)
        if rms is not None and (not np.isfinite(rms) or rms < 0):
            raise CameraCalibrationError("reprojection_rms_px must be non-negative.")

        def read_fov(name: str) -> float | None:
            value = intrinsics.get(name)
            if value is None:
                return None
            try:
                fov = float(value)
            except (TypeError, ValueError) as error:
                raise CameraCalibrationError(f"{name} must be a finite FOV in degrees.") from error
            if not np.isfinite(fov) or not 0 < fov < 179:
                raise CameraCalibrationError(f"{name} must be a finite FOV in degrees.")
            return fov

        horizontal_fov = read_fov("horizontal_fov_deg")
        vertical_fov = read_fov("vertical_fov_deg")
        diagonal_fov = read_fov("reported_diagonal_fov_deg")

        position: tuple[float, float, float] | None = None
        rotation: np.ndarray | None = None
        ground_plane_z = 0.0
        mount = data.get("camera_to_vehicle")
        if mount is not None:
            if not isinstance(mount, dict):
                raise CameraCalibrationError("camera_to_vehicle must be an object.")
            position_values = np.asarray(mount.get("position_m"), dtype=np.float64)
            rotation = np.asarray(mount.get("rotation_matrix"), dtype=np.float64)
            if position_values.shape != (3,) or not np.isfinite(position_values).all():
                raise CameraCalibrationError(
                    "camera_to_vehicle.position_m must be [right, forward, up]."
                )
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise CameraCalibrationError(
                    "camera_to_vehicle.rotation_matrix must be a finite 3x3 matrix."
                )
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
                raise CameraCalibrationError("Mount rotation matrix must be orthonormal.")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
                raise CameraCalibrationError("Mount rotation matrix determinant must be +1.")
            position = tuple(float(value) for value in position_values)
            try:
                ground_plane_z = float(mount.get("ground_plane_z_m", 0.0))
            except (TypeError, ValueError) as error:
                raise CameraCalibrationError("ground_plane_z_m must be finite.") from error
            if not np.isfinite(ground_plane_z):
                raise CameraCalibrationError("ground_plane_z_m must be finite.")

        return cls(
            image_width=image_size[0],
            image_height=image_size[1],
            camera_matrix=matrix,
            distortion_coefficients=distortion,
            reprojection_rms_px=rms,
            camera_position_vehicle_m=position,
            camera_to_vehicle_rotation=rotation,
            horizontal_fov_deg=horizontal_fov,
            vertical_fov_deg=vertical_fov,
            reported_diagonal_fov_deg=diagonal_fov,
            ground_plane_z_m=ground_plane_z,
        )

    @classmethod
    def simulated_for_image(
        cls,
        image_size_px: tuple[int, int],
        camera_height_m: float = SIMULATED_CAMERA_HEIGHT_M,
        horizontal_fov_deg: float = SIMULATED_HORIZONTAL_FOV_DEG,
        vertical_fov_deg: float = SIMULATED_VERTICAL_FOV_DEG,
        pitch_deg: float = SIMULATED_CAMERA_PITCH_DEG,
        roll_deg: float = SIMULATED_CAMERA_ROLL_DEG,
        yaw_deg: float = SIMULATED_CAMERA_YAW_DEG,
    ) -> CameraCalibration:
        """Build estimated intrinsics; positive yaw turns forward toward vehicle-left."""
        width, height = image_size_px
        if width <= 0 or height <= 0:
            raise CameraCalibrationError("Image dimensions must be positive.")
        if (
            camera_height_m <= 0
            or not 1 < horizontal_fov_deg < 179
            or not 1 < vertical_fov_deg < 179
            or not np.isfinite(pitch_deg)
            or not np.isfinite(roll_deg)
            or not np.isfinite(yaw_deg)
            or not -90 < pitch_deg < 90
            or not -180 <= roll_deg <= 180
            or not -180 <= yaw_deg <= 180
        ):
            raise CameraCalibrationError("Invalid simulated camera assumptions.")
        rotated_image = height > width
        matches_camera_aspect = abs(
            width / height - CAMERA_IMAGE_SIZE_PX[0] / CAMERA_IMAGE_SIZE_PX[1]
        ) <= 1e-2
        if not rotated_image and not matches_camera_aspect:
            horizontal_fov_deg = 70.0
            vertical_fov_deg = math.degrees(
                2 * math.atan(height / width * math.tan(math.radians(horizontal_fov_deg) / 2))
            )
        if rotated_image:
            native_width, native_height = CAMERA_IMAGE_SIZE_PX
            native_focal_x = native_width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
            native_focal_y = native_height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
            scale_x = width / native_height
            scale_y = height / native_width
            focal_x = native_focal_y * scale_x
            focal_y = native_focal_x * scale_y
            center_x = (SIMULATED_CY_PX + 0.5) * scale_x - 0.5
            center_y = (native_width - 0.5 - SIMULATED_CX_PX) * scale_y - 0.5
        elif matches_camera_aspect:
            native_width, native_height = CAMERA_IMAGE_SIZE_PX
            scale_x = width / native_width
            scale_y = height / native_height
            focal_x = (
                native_width
                / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
                * scale_x
            )
            focal_y = (
                native_height
                / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
                * scale_y
            )
            center_x = (SIMULATED_CX_PX + 0.5) * scale_x - 0.5
            center_y = (SIMULATED_CY_PX + 0.5) * scale_y - 0.5
        else:
            focal_x = width / (2 * math.tan(math.radians(horizontal_fov_deg) / 2))
            focal_y = height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
            center_x = width / 2
            center_y = height / 2
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        yaw = math.radians(yaw_deg)
        yaw_rotation = np.asarray([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        pitch_rotation = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ])
        level_rotation = np.asarray([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ])
        roll_rotation = np.asarray([
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return cls(
            image_width=width,
            image_height=height,
            camera_matrix=np.asarray(
                [
                    [focal_x, 0, center_x],
                    [0, focal_y, center_y],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            ),
            distortion_coefficients=np.zeros(5, dtype=np.float64),
            reprojection_rms_px=None,
            camera_position_vehicle_m=(
                MIRROR_CAMERA_POSITION_VEHICLE_M[0],
                MIRROR_CAMERA_POSITION_VEHICLE_M[1],
                MIRROR_CAMERA_POSITION_VEHICLE_M[2],
            ),
            camera_to_vehicle_rotation=(
                yaw_rotation @ pitch_rotation @ level_rotation @ roll_rotation
            ),
            horizontal_fov_deg=horizontal_fov_deg,
            vertical_fov_deg=vertical_fov_deg,
            reported_diagonal_fov_deg=(
                REPORTED_DIAGONAL_FOV_DEG if rotated_image or matches_camera_aspect else None
            ),
            ground_plane_z_m=MIRROR_CAMERA_POSITION_VEHICLE_M[2] - camera_height_m,
        )

    @staticmethod
    def matches_reference_video_size(image_size_px: tuple[int, int]) -> bool:
        width, height = image_size_px
        return width > 0 and height > 0 and abs(
            width / height - REFERENCE_VIDEO_IMAGE_SIZE_PX[0] / REFERENCE_VIDEO_IMAGE_SIZE_PX[1]
        ) <= 1e-3

    @classmethod
    def simulated_video_for_image(
        cls, image_size_px: tuple[int, int]
    ) -> CameraCalibration:
        """Use the shared reference-height fit only for matching portrait videos."""
        if not cls.matches_reference_video_size(image_size_px):
            return cls.simulated_for_image(image_size_px)
        width, height = image_size_px
        base = cls.simulated_for_image(
            image_size_px,
            pitch_deg=REFERENCE_VIDEO_PITCH_DEG,
            roll_deg=REFERENCE_VIDEO_ROLL_DEG,
        )
        return replace(
            base,
            camera_matrix=np.asarray([
                [REFERENCE_VIDEO_FX_PX * width / REFERENCE_VIDEO_IMAGE_SIZE_PX[0], 0, width / 2],
                [0, REFERENCE_VIDEO_FY_PX * height / REFERENCE_VIDEO_IMAGE_SIZE_PX[1], height / 2],
                [0, 0, 1],
            ], dtype=np.float64),
            horizontal_fov_deg=REFERENCE_VIDEO_HORIZONTAL_FOV_DEG,
            vertical_fov_deg=REFERENCE_VIDEO_VERTICAL_FOV_DEG,
            reported_diagonal_fov_deg=None,
        )

    def to_dict(self) -> dict[str, Any]:
        mount = None
        if (
            self.camera_position_vehicle_m is not None
            and self.camera_to_vehicle_rotation is not None
        ):
            mount = {
                "position_m": list(self.camera_position_vehicle_m),
                "rotation_matrix": self.camera_to_vehicle_rotation.tolist(),
                "ground_plane_z_m": self.ground_plane_z_m,
            }
        return {
            "schema_version": 1,
            "coordinate_system": {
                "vehicle_axes": "x right, y forward, z up; ground plane at ground_plane_z_m",
                "rotation_definition": (
                    "camera_to_vehicle_rotation maps OpenCV camera vectors "
                    "(x right, y down, z forward) into vehicle coordinates"
                ),
            },
            "intrinsics": {
                "image_size_px": [self.image_width, self.image_height],
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.distortion_coefficients.tolist(),
                "reprojection_rms_px": self.reprojection_rms_px,
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "vertical_fov_deg": self.vertical_fov_deg,
                "reported_diagonal_fov_deg": self.reported_diagonal_fov_deg,
            },
            "camera_to_vehicle": mount,
        }

    def _scaled_camera_matrix(self, image_width: int, image_height: int) -> np.ndarray:
        if image_width <= 0 or image_height <= 0:
            raise CameraCalibrationError("Image dimensions must be positive.")
        source_aspect = self.image_width / self.image_height
        image_aspect = image_width / image_height
        if abs(source_aspect - image_aspect) > 1e-3:
            raise CameraCalibrationError(
                "Calibration and image aspect ratios differ; resize/crop geometry is unknown."
            )
        scale_x = image_width / self.image_width
        scale_y = image_height / self.image_height
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= scale_x
        matrix[1, :] *= scale_y
        matrix[2, 2] = 1.0
        return matrix

    def estimate_height_cm(
        self,
        head_pixel: tuple[float, float],
        foot_pixel: tuple[float, float],
        image_size_px: tuple[int, int],
        max_vertical_alignment_residual_m: float = 0.25,
    ) -> HeightEstimate:
        """Estimate vertical height from a head pixel and ground-contact pixel.

        `image_size_px` is `(width, height)`. The person is assumed upright and
        the foot and head points are assumed to share the same horizontal
        vehicle coordinates. Extrinsics must map OpenCV camera axes to the
        documented vehicle axes and locate the camera relative to the ground plane.
        """
        if (
            self.camera_position_vehicle_m is None
            or self.camera_to_vehicle_rotation is None
        ):
            raise CameraCalibrationError(
                "Metric height requires camera_to_vehicle position and rotation."
            )
        image_width, image_height = image_size_px
        matrix = self._scaled_camera_matrix(image_width, image_height)
        pixels = np.asarray([head_pixel, foot_pixel], dtype=np.float64).reshape(-1, 1, 2)
        if not np.isfinite(pixels).all():
            raise CameraCalibrationError("Head and foot pixels must be finite.")
        normalized = cv2.undistortPoints(
            pixels,
            matrix,
            self.distortion_coefficients,
        ).reshape(2, 2)

        camera_position = np.asarray(self.camera_position_vehicle_m, dtype=np.float64)
        rotation = self.camera_to_vehicle_rotation

        def vehicle_ray(point: np.ndarray) -> np.ndarray:
            camera_ray = np.asarray((point[0], point[1], 1.0), dtype=np.float64)
            ray = rotation @ camera_ray
            length = float(np.linalg.norm(ray))
            if length <= 1e-9:
                raise CameraCalibrationError("Pixel ray is degenerate.")
            return ray / length

        head_ray = vehicle_ray(normalized[0])
        foot_ray = vehicle_ray(normalized[1])
        camera_height_m = camera_position[2] - self.ground_plane_z_m
        if not np.isfinite(camera_height_m) or camera_height_m <= 0 or foot_ray[2] >= -1e-6:
            raise CameraCalibrationError(
                "Foot ray does not intersect the vehicle ground plane in front of camera."
            )

        foot_distance = -camera_height_m / foot_ray[2]
        if foot_distance <= 0:
            raise CameraCalibrationError("Ground intersection is behind the camera.")
        foot_position = camera_position + foot_distance * foot_ray

        horizontal_ray = head_ray[:2]
        horizontal_norm_squared = float(horizontal_ray @ horizontal_ray)
        if horizontal_norm_squared <= 1e-8:
            raise CameraCalibrationError("Head ray is too close to vertical for height estimation.")
        head_distance = float(
            horizontal_ray @ (foot_position[:2] - camera_position[:2])
            / horizontal_norm_squared
        )
        if head_distance <= 0:
            raise CameraCalibrationError("Head point lies behind the camera.")
        projected_head = camera_position + head_distance * head_ray
        residual = float(np.linalg.norm(projected_head[:2] - foot_position[:2]))
        if residual > max_vertical_alignment_residual_m:
            raise CameraCalibrationError(
                "Head and foot rays do not support an upright vertical body "
                f"(horizontal residual {residual:.3f} m)."
            )

        height_m = float(projected_head[2] - self.ground_plane_z_m)
        if not 0.8 <= height_m <= 2.6:
            raise CameraCalibrationError(
                f"Geometry produced implausible height {height_m:.2f} m."
            )
        return HeightEstimate(
            height_cm=height_m * 100.0,
            vertical_alignment_residual_m=residual,
            foot_position_vehicle_m=tuple(float(value) for value in foot_position),
        )


def calibrate_chessboard(
    image_paths: Sequence[str | Path],
    board_size: tuple[int, int],
    square_size_m: float,
    minimum_views: int = 8,
) -> dict[str, Any]:
    """Calibrate intrinsics from chessboard images with fixed inner-corner size."""
    columns, rows = board_size
    if columns < 2 or rows < 2:
        raise CameraCalibrationError("Chessboard needs at least 2 inner corners per axis.")
    if not np.isfinite(square_size_m) or square_size_m <= 0:
        raise CameraCalibrationError("Chessboard square size must be positive meters.")
    if minimum_views < 3:
        raise CameraCalibrationError("At least 3 calibration views are required.")

    object_template = np.zeros((rows * columns, 3), dtype=np.float32)
    object_template[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    )
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used_images: list[str] = []
    rejected_images: list[str] = []

    for image_path in image_paths:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            rejected_images.append(str(path))
            continue
        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            raise CameraCalibrationError(
                "All chessboard images must have identical resolution."
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found = False
        corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                gray,
                board_size,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
        if not found:
            found, corners = cv2.findChessboardCorners(
                gray,
                board_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found and corners is not None:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        1e-3,
                    ),
                )

        if not found or corners is None:
            rejected_images.append(str(path))
            continue
        object_points.append(object_template.copy())
        image_points.append(np.asarray(corners, dtype=np.float32))
        used_images.append(str(path))

    if image_size is None or len(object_points) < minimum_views:
        raise CameraCalibrationError(
            f"Found {len(object_points)} usable chessboard views; "
            f"at least {minimum_views} are required."
        )

    rms, camera_matrix, distortion, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    )
    per_view_errors = []
    for objects, observed, rotation_vector, translation_vector in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
    ):
        projected, _ = cv2.projectPoints(
            objects,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )
        observed_points = np.asarray(observed, dtype=np.float64).reshape(-1, 2)
        projected_points = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        squared_errors = np.sum((observed_points - projected_points) ** 2, axis=1)
        per_view_errors.append(
            float(np.sqrt(np.mean(squared_errors)))
        )

    return {
        "schema_version": 1,
        "coordinate_system": {
            "vehicle_axes": "x right, y forward, z up; ground plane z=0",
            "rotation_definition": (
                "camera_to_vehicle_rotation maps OpenCV camera vectors "
                "(x right, y down, z forward) into vehicle coordinates"
            ),
        },
        "intrinsics": {
            "image_size_px": [image_size[0], image_size[1]],
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": distortion.reshape(-1).tolist(),
            "reprojection_rms_px": float(rms),
            "mean_per_view_error_px": float(np.mean(per_view_errors)),
            "board_inner_corners": [columns, rows],
            "square_size_m": square_size_m,
            "views_used": used_images,
            "views_rejected": rejected_images,
        },
        "camera_to_vehicle": None,
    }


def _expand_image_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in (matches or [pattern]))
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics from chessboard images."
    )
    parser.add_argument("--images", nargs="+", required=True, help="Image paths or globs")
    parser.add_argument("--board-columns", type=int, required=True, help="Inner corners")
    parser.add_argument("--board-rows", type=int, required=True, help="Inner corners")
    parser.add_argument(
        "--square-size-mm",
        type=float,
        required=True,
        help="Measured chessboard square size in millimeters",
    )
    parser.add_argument(
        "--camera-pose-json",
        type=Path,
        help="JSON with camera_to_vehicle.position_m and rotation_matrix",
    )
    parser.add_argument("--output", type=Path, default=Path("camera_calibration.json"))
    args = parser.parse_args()

    try:
        calibration = calibrate_chessboard(
            _expand_image_paths(args.images),
            (args.board_columns, args.board_rows),
            args.square_size_mm / 1000.0,
        )
        if args.camera_pose_json is not None:
            pose_data = json.loads(args.camera_pose_json.read_text(encoding="utf-8"))
            if not isinstance(pose_data, dict):
                raise CameraCalibrationError("Camera pose JSON must be an object.")
            calibration["camera_to_vehicle"] = pose_data.get(
                "camera_to_vehicle", pose_data
            )
            CameraCalibration.from_dict(calibration)
        args.output.write_text(
            json.dumps(calibration, indent=2) + "\n",
            encoding="utf-8",
        )
    except (CameraCalibrationError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))

    rms = calibration["intrinsics"]["reprojection_rms_px"]
    print(f"Calibration written to {args.output.resolve()}")
    print(
        f"Used {len(calibration['intrinsics']['views_used'])} views; "
        f"RMS reprojection error {rms:.3f} px."
    )
    if calibration["camera_to_vehicle"] is None:
        print("Camera pose is unset; metric height requires vehicle-mount extrinsics.")


if __name__ == "__main__":
    main()