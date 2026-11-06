from __future__ import annotations

import logging
import time

import cv2

from .config import Settings
from .morphology_analyzer import MorphologyAnalyzer, ViewMorphology
from .morphology_visualizer import MorphologyVisualizer
from .pose_estimator import DriverPoseEstimator
from .pose_renderer import PoseRenderer
from .serialization import save_json
from .video_processor import _open_camera

logger = logging.getLogger(__name__)

WINDOW_NAME = "Pre-entry driver morphology capture"


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    settings = Settings()
    settings.validate()
    estimator = DriverPoseEstimator(settings)
    analyzer = MorphologyAnalyzer(settings.keypoint_threshold)
    visualizer = MorphologyVisualizer(settings.keypoint_threshold)
    renderer = PoseRenderer(settings)
    capture = _open_camera(settings)
    captured: dict[str, ViewMorphology] = {}
    captured_frames = {}
    captured_results = {}
    frame_number = 0
    started = time.perf_counter()

    print(
        "Stand in the marked capture area before entering the vehicle. "
        "Keep your full body visible."
    )
    print("Press F to capture the frontal pose; press Q to quit.")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while True:
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError("The camera stopped supplying valid frames")

            frame_number += 1
            if settings.mirror_camera:
                frame = cv2.flip(frame, 1)

            result = estimator.estimate(
                frame=frame,
                frame_number=frame_number,
                timestamp_seconds=time.perf_counter() - started,
            )
            displayed = renderer.render(frame, result)

            if "frontal" not in captured:
                instruction = "FRONTAL VIEW: stand upright, then press F"
            elif "lateral" not in captured:
                instruction = "TURN SIDEWAYS: stand upright, then press L"
            else:
                instruction = "Two views captured. Press Q to finish."

            cv2.rectangle(
                displayed,
                (0, 0),
                (displayed.shape[1], 38),
                (20, 22, 27),
                -1,
            )
            cv2.putText(
                displayed,
                instruction,
                (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 246, 248),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, displayed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                if len(captured) < 2:
                    print("Capture cancelled; no profile was saved.")
                break

            view = None
            if key == ord("f") and "frontal" not in captured:
                view = "frontal"
            elif (
                key == ord("l") and "frontal" in captured and "lateral" not in captured
            ):
                view = "lateral"

            if view is None:
                continue

            try:
                measurement = analyzer.analyze_view(result, view)
            except ValueError as error:
                print(f"Cannot capture {view} view: {error}")
                continue

            captured[view] = measurement
            captured_frames[view] = frame.copy()
            captured_results[view] = result
            print(f"Captured {view} view.")
            if view == "frontal":
                print("Turn sideways, then press L to capture the lateral view.")
                continue

            profile = analyzer.build_profile(
                frontal=captured["frontal"],
                lateral=captured["lateral"],
            )
            visualizer.save_comparison(
                path=settings.morphology_visualization_path,
                frontal_frame=captured_frames["frontal"],
                frontal_result=captured_results["frontal"],
                lateral_frame=captured_frames["lateral"],
                lateral_result=captured_results["lateral"],
            )
            save_json(settings.morphology_profile_path, profile.to_dict())
            print(
                "Normalized morphology profile saved to "
                f"{settings.morphology_profile_path}"
            )
            print(
                "Annotated frontal/lateral visualization saved to "
                f"{settings.morphology_visualization_path}"
            )
            print(
                "No physical height or seat position is inferred; "
                "vehicle calibration is not configured."
            )
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
