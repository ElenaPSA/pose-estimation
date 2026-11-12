"""Detect a fully visible person in a video and overlay a height estimate.

Examples:
    # Webcam; pixel height only until calibrated
    python -m driver_pose_estimation.person_height_video

    # Webcam with a scale measured in the same plane as the person
    python -m driver_pose_estimation.person_height_video \\
        --reference-height-cm 100 --reference-height-px 420

    # Process a video file instead of the webcam
    python -m driver_pose_estimation.person_height_video \\
        --source input.mp4 --output annotated.mp4 \\
        --reference-height-cm 100 --reference-height-px 420

The reference object must be vertical, have a known physical height, and be
at approximately the same camera distance/depth as the person. A single
pixel-to-cm scale is not valid for people at substantially different depths.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .pose_estimator import DriverPoseEstimator

# COCO pose indices: nose=0, shoulders=5/6, hips=11/12,
# knees=13/14, ankles=15/16.
REQUIRED_KEYPOINTS = (0, 5, 6, 11, 12, 13, 14, 15, 16)


def parse_source(value: str) -> int | str:
    """Interpret a numeric source as a camera index, otherwise as a file path."""
    return int(value) if value.isdigit() else value


def full_body_is_visible(
    pose: Any,
    frame_width: int,
    frame_height: int,
    keypoint_threshold: float,
) -> bool:
    """Reject poses with missing landmarks or a bounding box touching frame edges."""
    box = pose.bounding_box
    edge_margin = 3

    if (
        box.x1 <= edge_margin
        or box.y1 <= edge_margin
        or box.x2 >= frame_width - edge_margin
        or box.y2 >= frame_height - edge_margin
    ):
        return False

    points = {point.index: point for point in pose.keypoints}
    if any(
        index not in points
        or points[index].confidence < keypoint_threshold
        for index in REQUIRED_KEYPOINTS
    ):
        return False

    # Ankles must be near the bottom of the detected person box; this helps
    # reject cases where a lower leg/foot has been cut off by the detection.
    ankle_y = max(points[15].y, points[16].y)
    if box.y2 - ankle_y > 0.18 * (box.y2 - box.y1):
        return False

    return True


def estimate_height_pixels(pose: Any) -> float:
    """Approximate head-to-floor image height from box top to lowest ankle."""
    ankles = [point.y for point in pose.keypoints if point.index in (15, 16)]
    return max(ankles) - pose.bounding_box.y1


def draw_person(
    frame: np.ndarray,
    pose: Any,
    height_px: float,
    height_cm: float | None,
) -> None:
    box = pose.bounding_box
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    color = (60, 220, 90)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw the COCO skeleton links for the major body chain.
    links = ((5, 6), (5, 11), (6, 12), (11, 12),
             (5, 7), (7, 9), (6, 8), (8, 10),
             (11, 13), (13, 15), (12, 14), (14, 16))
    points = {point.index: point for point in pose.keypoints}
    for first, second in links:
        if first in points and second in points:
            p1, p2 = points[first], points[second]
            cv2.line(frame, (int(p1.x), int(p1.y)),
                     (int(p2.x), int(p2.y)), (255, 180, 0), 2)
    for point in points.values():
        if point.confidence >= 0.35:
            cv2.circle(frame, (int(point.x), int(point.y)), 4, (0, 180, 255), -1)

    label = (
        f"Estimated height: {height_cm:.1f} cm"
        if height_cm is not None
        else f"Body height: {height_px:.0f} px | calibrate for cm"
    )
    cv2.rectangle(frame, (x1, max(0, y1 - 34)),
                  (min(frame.shape[1] - 1, x1 + 440), y1), color, -1)
    cv2.putText(frame, label, (x1 + 7, max(22, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (10, 10, 10), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0",
                        help="Camera index (default 0) or video file path")
    parser.add_argument("--output", default="person_height_output.mp4",
                        help="Annotated output video path")
    parser.add_argument("--reference-height-cm", type=float,
                        help="Known physical height of a vertical reference object, cm")
    parser.add_argument("--reference-height-px", type=float,
                        help="Pixel height of that reference object in this camera view")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open a preview window; still write the output video")
    args = parser.parse_args()

    if (args.reference_height_cm is None) != (args.reference_height_px is None):
        parser.error("Supply both --reference-height-cm and --reference-height-px")
    if args.reference_height_cm is not None and (
        args.reference_height_cm <= 0 or args.reference_height_px <= 0
    ):
        parser.error("Reference dimensions must be positive")

    settings = Settings()
    settings.validate()
    estimator = DriverPoseEstimator(settings)
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or settings.camera_width
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or settings.camera_height
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps < 1:
        fps = settings.camera_fps

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    cm_per_pixel = (
        args.reference_height_cm / args.reference_height_px
        if args.reference_height_cm is not None else None
    )
    recent_heights: deque[float] = deque(maxlen=9)
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if settings.mirror_camera and isinstance(source, int):
                frame = cv2.flip(frame, 1)

            result = estimator.estimate(frame, frame_number, frame_number / fps)
            candidates = [
                pose for pose in result.poses
                if full_body_is_visible(
                    pose, frame.shape[1], frame.shape[0], settings.keypoint_threshold
                )
            ]

            if candidates:
                # Select the largest fully visible person in the frame.
                person = max(candidates, key=lambda item: item.bounding_box.area)
                height_px = estimate_height_pixels(person)
                recent_heights.append(height_px)
                stable_height_px = float(np.median(recent_heights))
                height_cm = (
                    stable_height_px * cm_per_pixel
                    if cm_per_pixel is not None else None
                )
                draw_person(frame, person, stable_height_px, height_cm)
                status = "FULL BODY DETECTED"
                status_color = (60, 220, 90)
            else:
                recent_heights.clear()
                status = "WAITING: show one complete body (head to feet)"
                status_color = (0, 165, 255)

            cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2, cv2.LINE_AA)
            writer.write(frame)
            if not args.no_display:
                cv2.imshow("Full-body detection and height estimation", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Annotated video saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
