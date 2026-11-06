from __future__ import annotations

import logging
import os
import statistics
import time

import cv2

from .comfort_agent import ComfortAgent
from .config import Settings
from .models import (
    FramePoseResult,
    PostureMetrics,
)
from .pose_estimator import DriverPoseEstimator
from .pose_renderer import (
    AgentOverlay,
    PoseRenderer,
)
from .posture_analyzer import PostureAnalyzer
from .comfort_evaluator import (
    ComfortEvaluator,
)
from .safety_gate import SeatSafetyGate
from .seat_controller import SimulatedSeatController
from .seat_recommender import SeatRecommender
from .serialization import save_json
from .vehicle_state import VehicleStateProvider


logger = logging.getLogger(__name__)

WINDOW_NAME = "Driver comfort assistant"
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

    actual_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    logger.info(
        "Camera opened: index=%d resolution=%dx%d "
        "reported_fps=%.2f",
        settings.camera_index,
        actual_width,
        actual_height,
        actual_fps,
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


def _execute_action(
    action: str,
    agent: ComfortAgent,
    vehicle_state: VehicleStateProvider,
) -> None:
    if action == "check_comfort":
        decision = agent.ask_about_comfort()

    elif action == "comfortable":
        decision = agent.answer_comfortable()

    elif action == "uncomfortable":
        decision = agent.answer_uncomfortable()

    elif action == "too_far":
        decision = agent.submit_complaint(
            "The seat is too far from the controls"
        )

    elif action == "too_close":
        decision = agent.submit_complaint(
            "The seat is too close to the steering wheel"
        )

    elif action == "lower_back":
        decision = agent.submit_complaint(
            "I have lower back discomfort"
        )

    elif action == "too_high":
        decision = agent.submit_complaint(
            "The seat is too high"
        )

    elif action == "too_low":
        decision = agent.submit_complaint(
            "The seat is too low"
        )

    elif action == "apply":
        decision = agent.confirm_adjustment()

    elif action == "cancel":
        decision = agent.cancel_adjustment()

    elif action == "feels_better":
        decision = agent.confirm_improvement()

    elif action == "adjust_again":
        decision = agent.request_another_adjustment()

    elif action == "moving":
        vehicle_state.set_moving(
            speed_kph=30.0
        )

        agent.message = (
            "The demo vehicle is now moving. "
            "Seat adjustment is locked for safety."
        )

        logger.info(
            "Vehicle simulation changed to moving"
        )

        print(
            "VEHICLE: Moving at 30 km/h. "
            "Seat adjustment is locked."
        )

        return

    elif action == "parked":
        vehicle_state.set_parked()

        if agent.pending_adjustments:
            agent.message = (
                "The vehicle is now parked. "
                "You may apply the pending recommendation."
            )
        else:
            agent.message = (
                "The demo vehicle is parked. "
                "Seat adjustment is available."
            )

        logger.info(
            "Vehicle simulation changed to parked"
        )

        print(
            "VEHICLE: Parked with parking brake engaged."
        )

        return

    else:
        logger.warning(
            "Unknown interface action: %s",
            action,
        )
        return

    print(f"AGENT: {decision.message}")


def _handle_keyboard(
    key: int,
    agent: ComfortAgent,
    vehicle_state: VehicleStateProvider,
) -> bool:
    if key in (
        ord("q"),
        27,
    ):
        logger.info(
            "Processing interrupted by user"
        )
        return False

    keyboard_actions = {
        ord("c"): "check_comfort",
        ord("1"): "comfortable",
        ord("2"): "uncomfortable",
        ord("f"): "too_far",
        ord("r"): "too_close",
        ord("b"): "lower_back",
        ord("h"): "too_high",
        ord("l"): "too_low",
        ord("y"): "apply",
        ord("n"): "cancel",
        ord("m"): "moving",
        ord("p"): "parked",
    }

    action = keyboard_actions.get(key)

    if action is not None:
        _execute_action(
            action=action,
            agent=agent,
            vehicle_state=vehicle_state,
        )

    return True


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
    last_question_time = started

    latest_posture = PostureMetrics(
    valid=False
    )

    estimator = DriverPoseEstimator(
        settings
    )

    renderer = PoseRenderer(
        settings
    )

    posture_analyzer = PostureAnalyzer(
        settings
    )

    comfort_evaluator = ComfortEvaluator()

    comfort_assessment = (
        comfort_evaluator.evaluate(
            latest_posture
        )
    )


    latest_result: FramePoseResult | None = None
    latest_rendered_frame = None

    estimator = DriverPoseEstimator(
        settings
    )

    renderer = PoseRenderer(
        settings
    )

    posture_analyzer = PostureAnalyzer(
        settings
    )

    comfort_evaluator = ComfortEvaluator()

    vehicle_state = VehicleStateProvider(
        initial_speed_kph=(
            settings.simulated_vehicle_speed_kph
        ),
        initial_parking_brake=(
            settings.simulated_parking_brake
        ),
    )

    seat_controller = (
        SimulatedSeatController()
    )

    agent = ComfortAgent(
        recommender=SeatRecommender(),
        safety_gate=SeatSafetyGate(),
        seat_controller=seat_controller,
        vehicle_state=vehicle_state,
    )

    display_state = {
        "camera_width": settings.camera_width,
    }

    try:
        capture = _open_camera(
            settings
        )

        if settings.display_video:
            cv2.namedWindow(
                WINDOW_NAME,
                cv2.WINDOW_AUTOSIZE,
            )

            def on_mouse(
                event: int,
                x: int,
                y: int,
                flags: int,
                parameter: object,
            ) -> None:
                del flags
                del parameter

                if event != cv2.EVENT_LBUTTONUP:
                    return

                camera_width = int(
                    display_state[
                        "camera_width"
                    ]
                )

                panel_x = x - camera_width

                if panel_x < 0:
                    return

                action = renderer.get_action_at(
                    panel_x,
                    y,
                )

                if action is None:
                    return

                _execute_action(
                    action=action,
                    agent=agent,
                    vehicle_state=vehicle_state,
                )

            cv2.setMouseCallback(
                WINDOW_NAME,
                on_mouse,
            )

        logger.info(
            "Pose and comfort-agent pipeline started: "
            "model=%s device=%s",
            settings.model_path,
            settings.device,
        )

        running = True

        while running:
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

            display_state[
                "camera_width"
            ] = frame.shape[1]

            should_process = (
                (frames_read - 1)
                % settings.process_every_n_frames
                == 0
            )

            if not should_process:
                frames_skipped += 1

                if (
                    settings.display_video
                    and latest_rendered_frame
                    is not None
                ):
                    cv2.imshow(
                        WINDOW_NAME,
                        latest_rendered_frame,
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    running = _handle_keyboard(
                        key=key,
                        agent=agent,
                        vehicle_state=vehicle_state,
                    )

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

            latest_result = result

            if (
                frames_read
                % settings.posture_analysis_interval_frames
                == 0
            ):
                latest_posture = (
                    posture_analyzer.analyze(
                        result
                    )
                )

                comfort_assessment = (
                    comfort_evaluator.evaluate(
                        latest_posture
                    )
                )

                agent.update_posture(
                    latest_posture
                )

            if settings.save_history:
                history.append(
                    result
                )

            if (
                result.frame_number
                % settings.log_every_n_frames
                == 0
            ):
                logger.info(
                    "Frame=%d people=%d driver=%s "
                    "posture_valid=%s inference=%.1fms",
                    result.frame_number,
                    result.detected_people,
                    result.driver is not None,
                    latest_posture.valid,
                    result.processing_time_ms,
                )

            now = time.perf_counter()

            if (
                settings.automatic_comfort_questions
                and agent.state.value
                == "MONITORING"
                and (
                    now - last_question_time
                    >= settings
                    .comfort_question_interval_seconds
                )
            ):
                decision = (
                    agent.ask_about_comfort()
                )

                print(
                    f"AGENT: {decision.message}"
                )

                last_question_time = now


            print("\n====================")
            print("AGENT STATE :", agent.state.value)
            print("MESSAGE     :", agent.message)

            if comfort_assessment.overall_score is not None:
                print(
                    "COMFORT SCORE :",
                    comfort_assessment.overall_score
                )

            print(
                "RECOMMENDATIONS :",
                comfort_assessment.recommendations
            )
            print("====================")

            overlay = AgentOverlay(
                agent_state=agent.state.value,
                agent_message=agent.message,
                posture=latest_posture,
                comfort=comfort_assessment,
                vehicle=vehicle_state.get_state(),
                seat=seat_controller.state,
            )

            latest_rendered_frame = (
                renderer.render(
                    frame=frame,
                    result=result,
                    overlay=overlay,
                )
            )

            if settings.display_video:
                cv2.imshow(
                    WINDOW_NAME,
                    latest_rendered_frame,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                running = _handle_keyboard(
                    key=key,
                    agent=agent,
                    vehicle_state=vehicle_state,
                )

        if latest_result is None:
            logger.warning(
                "No video frame was processed"
            )

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