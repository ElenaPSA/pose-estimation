from __future__ import annotations

import logging
import os
import statistics
import time

import cv2

from .config import Settings
from .models import FramePoseResult
from .pose_estimator import DriverPoseEstimator
from .pose_renderer import PoseRenderer
from .serialization import save_json


logger = logging.getLogger(__name__)

WINDOW_NAME = "Driver pose estimation"
MAXIMUM_CONSECUTIVE_READ_FAILURES = 20


def _open_camera(
    settings: Settings,
) -> cv2.VideoCapture:
    if os.name == "nt":
        capture = cv2.VideoCapture(
            settings.camera_index,
            cv2.CAP_DSHOW,
        )
    else:
        capture = cv2.VideoCapture(
            settings.camera_index,
            cv2.CAP_ANY,
        )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            "Unable to open camera index "
            f"{settings.camera_index}"
        )

    capture.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        settings.camera_width,
    )

    capture.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        settings.camera_height,
    )

    capture.set(
        cv2.CAP_PROP_FPS,
        settings.camera_fps,
    )

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    logger.info(
        "Camera opened: index=%d resolution=%dx%d "
        "reported_fps=%.2f",
        settings.camera_index,
        int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ),
        int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ),
        capture.get(
            cv2.CAP_PROP_FPS
        ),
    )

    return capture


def _build_summary(
    results: list[FramePoseResult],
    frames_read: int,
    frames_skipped: int,
    elapsed_seconds: float,
) -> dict:
    inference_times = [
        result.processing_time_ms
        for result in results
    ]

    frames_with_driver = sum(
        result.driver is not None
        for result in results
    )

    return {
        "frames_read": frames_read,
        "frames_processed": len(results),
        "frames_skipped": frames_skipped,
        "frames_with_driver": frames_with_driver,
        "frames_without_driver": (
            len(results) - frames_with_driver
        ),
        "elapsed_seconds": elapsed_seconds,
        "average_inference_time_ms": (
            statistics.fmean(inference_times)
            if inference_times
            else None
        ),
    }


def process_video(
    settings: Settings,
) -> None:
    settings.validate()

    capture: cv2.VideoCapture | None = None
    history: list[FramePoseResult] = []

    frames_read = 0
    frames_skipped = 0
    read_failures = 0

    started = time.perf_counter()

    try:
        capture = _open_camera(settings)

        estimator = DriverPoseEstimator(
            settings
        )

        renderer = PoseRenderer(
            settings
        )

        logger.info(
            "Pose estimation started: "
            "model=%s device=%s",
            settings.model_path,
            settings.device,
        )

        while True:
            success, frame = capture.read()

            if (
                not success
                or frame is None
                or frame.size == 0
            ):
                read_failures += 1

                logger.warning(
                    "Invalid camera frame: %d/%d",
                    read_failures,
                    MAXIMUM_CONSECUTIVE_READ_FAILURES,
                )

                if (
                    read_failures
                    >= MAXIMUM_CONSECUTIVE_READ_FAILURES
                ):
                    raise RuntimeError(
                        "The camera stopped supplying "
                        "valid frames"
                    )

                time.sleep(0.05)
                continue

            read_failures = 0
            frames_read += 1

            if settings.mirror_camera:
                frame = cv2.flip(
                    frame,
                    1,
                )

            should_process = (
                (frames_read - 1)
                % settings.process_every_n_frames
                == 0
            )

            if not should_process:
                frames_skipped += 1

                if settings.display_video:
                    cv2.imshow(
                        WINDOW_NAME,
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key in (
                        ord("q"),
                        27,
                    ):
                        break

                continue

            timestamp_seconds = (
                time.perf_counter()
                - started
            )

            result = estimator.estimate(
                frame=frame,
                frame_number=frames_read,
                timestamp_seconds=(
                    timestamp_seconds
                ),
            )

            if settings.save_history:
                history.append(result)

            if (
                len(history)
                % settings.log_every_n_frames
                == 0
            ):
                logger.info(
                    "Frame=%d people=%d driver=%s "
                    "inference=%.1fms",
                    result.frame_number,
                    result.detected_people,
                    result.driver is not None,
                    result.processing_time_ms,
                )

            if settings.display_video:
                rendered_frame = renderer.render(
                    frame,
                    result,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    rendered_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ord("q"),
                    27,
                ):
                    logger.info(
                        "Processing interrupted by user"
                    )
                    break

    finally:
        if capture is not None:
            capture.release()

        cv2.destroyAllWindows()

        elapsed_seconds = (
            time.perf_counter()
            - started
        )

        if settings.save_history:
            save_json(
                settings.pose_history_path,
                [
                    result.to_dict()
                    for result in history
                ],
            )

            save_json(
                settings.pose_summary_path,
                _build_summary(
                    results=history,
                    frames_read=frames_read,
                    frames_skipped=frames_skipped,
                    elapsed_seconds=elapsed_seconds,
                ),
            )

        logger.info(
            "Finished: read=%d processed=%d "
            "skipped=%d elapsed=%.2fs",
            frames_read,
            len(history),
            frames_skipped,
            elapsed_seconds,
        )