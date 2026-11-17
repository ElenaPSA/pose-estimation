import base64
from io import BytesIO
import json
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import pytest

from driver_pose_estimation.camera_calibration import CameraCalibration
from driver_pose_estimation.face_detector import FaceBox
from driver_pose_estimation.models import (
    BoundingBox,
    FramePoseResult,
    Keypoint,
    PoseDetection,
)
from driver_pose_estimation.seat_hmi import HTML, MorphologyAnalyzer, _HmiHandler


def test_hmi_contains_driver_profiles_and_demo_seat_parameters() -> None:
    assert 'data-size="small"' in HTML
    assert 'data-size="medium"' in HTML
    assert 'data-size="large"' in HTML
    assert "BUILD CATEGORY" in HTML
    assert 'id="seatStudio"' in HTML
    assert 'src="/assets/seat-studio.js"' in HTML
    assert "SeatStudio.mount();" in HTML
    assert "SeatStudio.update(h);" in HTML
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in HTML
    assert "NO SEAT ACTUATION" in HTML
    assert 'src="/assets/seat-reference-side.png"' not in HTML
    assert 'src="/assets/seat-reference-definition.png"' not in HTML
    assert 'src="/assets/seat-reference-ranges.png"' not in HTML
    assert 'src="/assets/automotive-seat-reference-shadow.png"' not in HTML
    assert 'id="seatAnimation"' not in HTML
    assert 'id="driverMovement"' not in HTML
    assert 'src="/assets/driver-seat-profile.png"' not in HTML
    assert "Seat-position visualization · visual only" not in HTML
    assert 'class="configuration-layout"' not in HTML
    assert 'id="videoFile"' in HTML
    assert 'id="videoFileStatus"' in HTML
    assert 'id="cameraCalibrationFile"' not in HTML
    assert 'id="sourceVideo" hidden muted playsinline controls' in HTML
    assert 'accept="video/*"' in HTML
    assert 'id="sourceVideo"' in HTML
    assert 'id="decodedVideoPreview"' in HTML
    assert 'id="cancelVideo"' not in HTML
    assert "SIMULATED CAMERA ESTIMATE · NOT CALIBRATED" in HTML
    assert "0.85m mount height" in HTML
    assert 'src="/assets/video-capture.js"' in HTML
    assert "VideoCapture.mount({" in HTML
    assert 'id="lateralFile"' not in HTML
    assert 'id="referenceLength"' not in HTML
    assert 'id="analyzeImages"' not in HTML
    assert "LOAD VIDEO / AUTOMATIC FULL-BODY CAPTURE</h2>" in HTML
    assert 'id="detectionViewport"' in HTML
    assert 'id="detectionStage"' in HTML
    assert 'id="detectionImage"' in HTML
    assert "Detecting Person</div>" in HTML
    assert 'id="detectionStatus"' in HTML
    assert "function startDetectionAnimation(signal)" in HTML
    assert "detectionStatusTimers.push(window.setTimeout(() => {" in HTML
    assert "      }, duration));" in HTML
    assert HTML.index("startDetectionAnimation(signal);") < HTML.index('fetch("/analyze"')
    assert "function showDetectionBox(box)" in HTML
    assert ".detection-viewport.scanning #detectionImage { animation: grayscale-reveal 5s ease-out both; }" in HTML
    assert "@keyframes camera-highlight" in HTML
    assert "@keyframes person-scan" in HTML
    assert ".detection-viewport.scanning .detection-sweep { animation: person-scan 5s linear both; }" in HTML
    assert "100% { transform: translateY(var(--scan-distance)); opacity: .75; }" in HTML
    assert 'id="detectionColorImage"' in HTML
    assert "detectionColorImage.src = dataUrl;" in HTML
    assert "detectionColorImage.style.clipPath =" in HTML
    assert "await scanAnimation;" in HTML
    assert "cameraCalibrationFile" not in video_source
    assert "cameraCalibration" not in video_source
    assert HTML.count('<line id="cameraTo') == 4
    assert '<rect id="detectedPersonBox" class="person-box"/>' in HTML
    assert "<polygon" not in HTML
    assert ".detection-viewport.detected .detection-overlay { animation: camera-highlight" in HTML
    assert 'mode: "video_demo"' in HTML
    assert "simulated camera model" in HTML
    assert "SIMULATED CAMERA ESTIMATE · NOT CALIBRATED" in HTML
    assert "MODEL PERFORMANCE" in HTML
    assert 'id="morphologyConfidence"' in HTML
    assert 'id="faceConfidence"' in HTML
    assert "not physical measurements" in HTML
    assert 'id="measureHeight"' in HTML
    assert 'id="measureShoulder"' in HTML
    assert 'id="measureHip"' in HTML
    assert 'id="measureFace"' in HTML
    assert 'id="measureTorso"' in HTML
    assert 'id="measureLegs"' in HTML
    assert 'id="measureUpperBody"' not in HTML
    assert 'id="measureLowerBody"' not in HTML
    assert 'id="measureUpperLeg"' in HTML
    assert 'id="measureLowerLeg"' in HTML
    assert "Shoulder width</span>" in HTML
    assert "Hip width</span>" in HTML
    assert "Torso length</span>" in HTML
    assert "Upper leg length</span>" in HTML
    assert "Lower leg length</span>" in HTML
    assert "cm · simulated" in HTML
    assert 'id="torso"' not in HTML
    assert "proportionWord" not in HTML
    assert HTML.index("LOAD VIDEO / AUTOMATIC FULL-BODY CAPTURE") < HTML.index("DRIVER DIMENSIONS")
    assert HTML.index("SIMULATED CAMERA ESTIMATE · NOT CALIBRATED") < HTML.index('id="seatStudio"')


def test_hmi_serves_dashboard_locally() -> None:
    server = HTTPServer(("127.0.0.1", 0), _HmiHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request("GET", "/")
        response = connection.getresponse()
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert "DRIVER DIMENSIONS" in page

        connection.request("GET", "/assets/seat-studio.js")
        studio_response = connection.getresponse()
        studio_source = studio_response.read().decode("utf-8")
        assert studio_response.status == 200
        assert studio_response.getheader("Content-Type") == (
            "text/javascript; charset=utf-8"
        )
        assert "x: [0, 255]" in studio_source
        assert "z: [0, 80]" in studio_source
        assert "recline: [15, 55]" in studio_source
        assert "cancelAnimationFrame(frame)" in studio_source
        assert "reducedMotion()" in studio_source
        connection.request("GET", "/assets/video-capture.js")
        video_response = connection.getresponse()
        assert video_response.status == 200
        assert video_response.getheader("Content-Type") == "text/javascript; charset=utf-8"
        video_source = video_response.read().decode()
        assert 'fetch("/probe-frame"' in video_source
        assert 'fetch("/stream-video"' in video_source
        assert 'fetch("/transcode-video"' in video_source
        assert 'Playing converted local video' in video_source
        assert "Selected video: ${file.name}" in video_source
        assert "await video.play();" in video_source
        assert "confirmations >= 2 && observation.nearCamera" in video_source
        assert "confirmations >= 3" in video_source
        assert "while (!video.ended)" in video_source
        assert "result.confident && sameTrack" in video_source
        assert "observation.nearCamera" in video_source
        assert "Video ended. Scanning the closest complete frame" in video_source
        assert 'document.getElementById("cancelVideo")' not in video_source
        assert video_source.index("await video.play();") < video_source.index('fetch("/probe-frame"')
        for name in (
            "seat-reference-side.png",
            "seat-reference-definition.png",
            "seat-reference-ranges.png",
            "seat-side-motion.svg",
        ):
            connection.request("GET", f"/assets/{name}")
            asset_response = connection.getresponse()
            body = asset_response.read()
            assert asset_response.status == 200
            expected_type = "image/svg+xml" if name.endswith(".svg") else "image/png"
            assert asset_response.getheader("Content-Type") == expected_type
            assert body

        connection.request("GET", "/assets/driver-seat-profile.png")
        image_response = connection.getresponse()
        image_bytes = image_response.read()
        assert image_response.status == 200
        assert image_response.getheader("Content-Type") == "image/png"
        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        asset_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "driver_pose_estimation"
            / "assets"
            / "driver-seat-profile.png"
        )
        assert image_bytes == asset_path.read_bytes()

        connection.request("GET", "/assets/automotive-seat-adjustment-scheme.png")
        scheme_response = connection.getresponse()
        scheme_bytes = scheme_response.read()
        assert scheme_response.status == 200
        assert scheme_response.getheader("Content-Type") == "image/png"
        assert scheme_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        scheme_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "driver_pose_estimation"
            / "assets"
            / "automotive-seat-adjustment-scheme.png"
        )
        assert scheme_bytes == scheme_path.read_bytes()
        scheme_image = cv2.imdecode(
            np.frombuffer(scheme_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert scheme_image is not None
        fixed_seat_pixel = scheme_image[450, 200]
        assert max(fixed_seat_pixel) - min(fixed_seat_pixel) < 5
        assert 55 <= int(fixed_seat_pixel.mean()) <= 100
        inset_seat_pixel = scheme_image[180, 850]
        assert int(inset_seat_pixel.mean()) < 110
        assert scheme_image[300, 502, 0] > 200
        assert scheme_image[543, 300, 2] > 200

        connection.request("GET", "/assets/automotive-seat-proposed-overlay.png")
        overlay_response = connection.getresponse()
        overlay_bytes = overlay_response.read()
        assert overlay_response.status == 200
        assert overlay_response.getheader("Content-Type") == "image/png"
        overlay_image = cv2.imdecode(
            np.frombuffer(overlay_bytes, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        assert overlay_image is not None
        assert overlay_image.shape == (768, 1152, 4)
        assert np.any(overlay_image[:, :, 3] == 0)
        assert np.any(overlay_image[:, :, 3] > 0)
        assert np.all(overlay_image[:, 780:, 3] == 0)
        orange_pixel = overlay_image[450, 200, :3]
        assert int(orange_pixel[2]) > int(orange_pixel[1]) > int(orange_pixel[0])

        connection.request("GET", "/missing")
        missing_response = connection.getresponse()
        assert missing_response.status == 404
        missing_response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_hmi_analyzes_single_frontal_image_and_calibration(monkeypatch) -> None:
    class StubEstimator:
        calls = 0

        def estimate(self, frame, frame_number, timestamp_seconds):
            self.calls += 1
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
                    confidence=0.9,
                )
                for index in range(17)
            ]
            driver = PoseDetection(
                bounding_box=BoundingBox(50.0, 10.0, 150.0, 240.0),
                keypoints=keypoints,
            )
            return FramePoseResult(
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                processing_time_ms=1.0,
                detected_people=1,
                driver=driver,
                poses=[driver],
            )

    class StubFaceDetector:
        def detect(self, image, pose):
            return FaceBox(x=90, y=15, width=20, height=15, confidence=0.9)

    estimator = StubEstimator()
    monkeypatch.setattr(_HmiHandler, "pose_estimator", estimator)
    monkeypatch.setattr(_HmiHandler, "face_detector", StubFaceDetector())
    ok, encoded_image = cv2.imencode(".png", np.zeros((250, 200, 3), dtype=np.uint8))
    assert ok
    data_url = "data:image/png;base64," + base64.b64encode(encoded_image).decode()
    view = {
        "image": data_url,
        "marker_points": [{"x": 10, "y": 10}, {"x": 10, "y": 110}],
    }
    payload = json.dumps(
        {
            "reference_length_cm": 100,
            "frontal": view,
        }
    )

    server = HTTPServer(("127.0.0.1", 0), _HmiHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(
            "POST",
            "/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())

        assert response.status == 200
        assert estimator.calls == 1
        assert result["height_cm"] == 230.0
        assert result["segment_height_cm"] is None
        assert result["size_class"] == "large"
        assert result["build_category"] == "robust"
        assert result["dimensions_cm"]["face_height"] == 15.0
        assert result["dimensions_cm"]["torso"] is None
        assert result["dimensions_cm"]["lower_body"] is None
        assert result["dimensions_cm"]["upper_leg"] is None
        assert result["dimensions_cm"]["shoulder_width"] is not None
        assert result["detection_bbox"] == {
            "x": 25.0,
            "y": 4.0,
            "width": 50.0,
            "height": 92.0,
        }
        assert result["confidence"] == 0.9
        assert result["face_detection_confidence"] == 0.9
        assert "Build class is a rough" in " ".join(result["warnings"])
        assert "unavailable because this workflow uses one frontal image" in " ".join(
            result["warnings"]
        )

        with monkeypatch.context() as context:
            def missing_frontal_landmarks(self, frame_result, view_name):
                raise ValueError("The frontal view requires both shoulders and knees")

            context.setattr(
                MorphologyAnalyzer,
                "analyze_view",
                missing_frontal_landmarks,
            )
            connection.request(
                "POST",
                "/analyze",
                body=payload,
                headers={"Content-Type": "application/json"},
            )
            partial_response = connection.getresponse()
            partial_result = json.loads(partial_response.read())
            assert partial_response.status == 200
            assert partial_result["detection_bbox"] == result["detection_bbox"]
            assert partial_result["dimensions_cm"]["shoulder_width"] is None
            assert partial_result["dimensions_cm"]["hip_width"] is None
            assert "person bounding box was detected" in " ".join(
                partial_result["warnings"]
            )

        class NoFaceDetector:
            def detect(self, image, pose):
                return None

        monkeypatch.setattr(_HmiHandler, "face_detector", NoFaceDetector())
        connection.request(
            "POST",
            "/analyze",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        no_face_response = connection.getresponse()
        no_face_result = json.loads(no_face_response.read())
        assert no_face_response.status == 200
        assert estimator.calls == 3
        assert no_face_result["height_cm"] == 230.0
        assert no_face_result["face_detection_confidence"] is None
        assert no_face_result["dimensions_cm"]["face_height"] is None
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_hmi_rejects_bad_image_upload_without_initializing_model(monkeypatch) -> None:
    monkeypatch.setattr(_HmiHandler, "pose_estimator", None)
    server = HTTPServer(("127.0.0.1", 0), _HmiHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    try:
        connection.request(
            "POST",
            "/analyze",
            body=json.dumps({"reference_length_cm": 100}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        result = json.loads(response.read())

        assert response.status == 400
        assert "frontal" in result["error"]
        assert _HmiHandler.pose_estimator is None
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_video_transcode_endpoint_returns_playable_h264_mp4(tmp_path) -> None:
    source_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120)
    )
    if not writer.isOpened():
        pytest.skip("MJPG test video writer is unavailable")
    try:
        for frame_index in range(10):
            writer.write(
                np.full((120, 160, 3), frame_index * 20, dtype=np.uint8)
            )
    finally:
        writer.release()

    video_bytes = source_path.read_bytes()
    server = HTTPServer(("127.0.0.1", 0), _HmiHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(
            "POST",
            "/transcode-video",
            video_bytes,
            {"Content-Type": "application/octet-stream"},
        )
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            pytest.skip(f"Bundled FFmpeg H.264 encoder unavailable: {body[:200]!r}")
        assert response.getheader("Content-Type") == "video/mp4"

        preview_path = tmp_path / "preview.mp4"
        preview_path.write_bytes(body)
        preview = cv2.VideoCapture(str(preview_path))
        try:
            success, frame = preview.read()
            assert preview.isOpened() and success
            assert frame.shape[:2] == (120, 160)
        finally:
            preview.release()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def video_pose() -> FramePoseResult:
    coords = {
        0: (100, 25), 5: (80, 60), 6: (120, 60),
        11: (85, 105), 12: (115, 105), 13: (85, 150),
        14: (115, 150), 15: (85, 205), 16: (115, 205),
    }
    pose = PoseDetection(
        BoundingBox(50, 10, 150, 215),
        [Keypoint(i, str(i), *coords.get(i, (100, 30)), .9) for i in range(17)],
    )
    return FramePoseResult(0, 0, 1, 1, pose, [pose])


def test_video_accepts_complete_body_pose() -> None:
    assert _HmiHandler._full_body_reason(
        video_pose(), np.zeros((250, 200, 3))
    ) is None


@pytest.mark.parametrize(
    "problem",
    [
        "none",
        "crowd",
        "cropped",
        "ankle",
        "seated",
        "ankles_cropped",
        "ankles_not_at_box_end",
    ],
)
def test_video_full_body_gate(problem) -> None:
    result = video_pose()
    if problem == "none":
        result.driver = None
        result.detected_people = 0
    elif problem == "crowd":
        result.detected_people = 2
    elif problem == "cropped":
        result.driver.bounding_box = BoundingBox(50, 0, 150, 250)
    elif problem == "ankle":
        result.driver.keypoints[15] = Keypoint(15, "ankle", 85, 230, .1)
    elif problem == "seated":
        result.driver.keypoints[11] = Keypoint(11, "hip", 85, 175, .9)
    elif problem == "ankles_cropped":
        result.driver.keypoints[15] = Keypoint(15, "ankle", 85, 225, .9)
        result.driver.keypoints[16] = Keypoint(16, "ankle", 115, 225, .9)
    elif problem == "ankles_not_at_box_end":
        result.driver.keypoints[15] = Keypoint(15, "ankle", 85, 160, .9)
        result.driver.keypoints[16] = Keypoint(16, "ankle", 115, 160, .9)
    assert _HmiHandler._full_body_reason(result, np.zeros((250, 200, 3))) is not None


def test_video_tracker_ignores_initial_ineligible_detection(tmp_path) -> None:
    video_path = tmp_path / "tracking-start.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 250)
    )
    assert writer.isOpened()
    try:
        for index in range(40):
            writer.write(np.full((250, 200, 3), index * 4, dtype=np.uint8))
    finally:
        writer.release()

    handler = object.__new__(_HmiHandler)
    video_bytes = video_path.read_bytes()
    handler.rfile = BytesIO(video_bytes)
    probes = 0
    valid_track = {"x1": 0.25, "y1": 0.2, "x2": 0.75, "y2": 0.8}

    def probe(_image):
        nonlocal probes
        sample = probes
        probes += 1
        if sample == 0:
            return {
                "eligible": False,
                "confident": False,
                "track_box": {"x1": 0.0, "y1": 0.0, "x2": 0.7, "y2": 0.15},
                "reason": "Keep the whole head and feet away from the image edges.",
            }
        return {
            "eligible": True,
            "confident": True,
            "confidence_score": 0.9,
            "quality_score": 0.7 + sample * 0.01,
            "relative_height": 0.6,
            "near_camera": True,
            "track_box": valid_track,
            "reason": None,
        }

    handler._probe_image = probe
    captured = handler._capture_video(len(video_bytes))

    assert probes == 3
    assert captured["timestamp"] == 1.2


def test_video_probe_and_uncalibrated_analysis(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    class Estimator:
        detected_people = 1

        def estimate(self, image, *args):
            pose_result = video_pose()
            pose_result.detected_people = self.detected_people
            pose_result.driver.bounding_box = BoundingBox(
                50,
                10,
                150,
                215,
                min(1.0, 0.1 + float(image[0, 0, 0]) / 255),
            )
            return pose_result

    class FaceDetector:
        height = 26

        def detect(self, image, pose):
            return FaceBox(90, 15, 20, self.height, .9) if self.height else None

    face = FaceDetector()
    monkeypatch.setattr(_HmiHandler, "pose_estimator", Estimator())
    monkeypatch.setattr(_HmiHandler, "face_detector", face)
    ok, encoded = cv2.imencode(".png", np.zeros((250, 200, 3), dtype=np.uint8))
    assert ok
    payload = {
        "mode": "video_demo",
        "frontal": {"image": "data:image/png;base64," + base64.b64encode(encoded).decode()},
    }
    server = HTTPServer(("127.0.0.1", 0), _HmiHandler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)

    def post(path):
        connection.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    try:
        status, probe = post("/probe-frame")
        assert status == 200 and probe["eligible"]
        assert probe["track_box"] == {
            "x1": 0.25,
            "y1": 0.04,
            "x2": 0.75,
            "y2": 0.86,
        }
        assert probe["quality_score"] > 0
        assert "confidence_score" in probe
        assert "confident" in probe
        status, analysis = post("/analyze")
        assert status == 200
        simulated_model = CameraCalibration.simulated_for_image((200, 250))
        expected_height = simulated_model.estimate_height_cm(
            (100, 10), (100, 215), (200, 250)
        ).height_cm
        expected_span_cm = 176 * expected_height / 205
        assert analysis["height_cm"] == round(expected_height, 1)
        assert analysis["estimation_mode"] == "simulated_camera_geometry"
        assert analysis["calibration_mode"] == "simulated_vehicle_camera"
        assert analysis["face_detection_confidence"] == pytest.approx(.9)
        assert analysis["morphology_reference_span_cm"] == pytest.approx(
            expected_span_cm, abs=.1
        )
        centimeter_morphology = analysis["morphology_dimensions_cm"]
        assert centimeter_morphology["shoulder_width"] == pytest.approx(
            .2273 * expected_span_cm, abs=.1
        )
        assert centimeter_morphology["hip_width"] == pytest.approx(
            .1705 * expected_span_cm, abs=.1
        )
        assert centimeter_morphology["torso_length"] == pytest.approx(
            .2557 * expected_span_cm, abs=.1
        )
        assert analysis["height_sensitivity_range_cm"] is not None
        assert any("SIMULATION ONLY" in warning for warning in analysis["warnings"])
        assert analysis["face_detection_confidence"] == pytest.approx(.9)
        morphology = analysis["morphology_proportions"]
        assert morphology["reference_span_px"] == pytest.approx(176.0)
        assert morphology["shoulder_width_over_span"] == pytest.approx(0.2273)
        assert morphology["hip_width_over_span"] == pytest.approx(0.1705)
        assert morphology["torso_length_over_span"] == pytest.approx(0.2557)
        assert morphology["upper_leg_length_over_span"] == pytest.approx(0.2557)
        assert morphology["lower_leg_length_over_span"] == pytest.approx(0.3125)
        assert morphology["total_leg_length_over_span"] == pytest.approx(0.5682)
        morphology_cm = analysis["morphology_dimensions_cm"]
        reference_span_cm = analysis["morphology_reference_span_cm"]
        assert morphology_cm["shoulder_width"] == pytest.approx(
            morphology["shoulder_width_over_span"] * reference_span_cm
        )
        assert morphology_cm["hip_width"] == pytest.approx(
            morphology["hip_width_over_span"] * reference_span_cm
        )
        assert morphology_cm["total_leg_length"] == pytest.approx(
            morphology["total_leg_length_over_span"] * reference_span_cm
        )
        calibration = {
            "schema_version": 1,
            "intrinsics": {
                "image_size_px": [200, 250],
                "camera_matrix": [
                    [175, 0, 100],
                    [0, 175, 75],
                    [0, 0, 1],
                ],
                "distortion_coefficients": [0, 0, 0, 0, 0],
                "reprojection_rms_px": 0.2,
            },
            "camera_to_vehicle": {
                "position_m": [0, 0, 1.2],
                "rotation_matrix": [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
            },
        }
        payload["camera_calibration"] = calibration
        status, calibrated = post("/analyze")
        assert status == 200
        assert calibrated["estimation_mode"] == "calibrated_camera_geometry"
        assert calibrated["calibration_mode"] == "vehicle_camera"
        assert calibrated["height_cm"] == pytest.approx(175.7, abs=0.2)
        assert calibrated["height_geometry_residual_m"] < 0.001
        assert calibrated["dimensions_cm"]["shoulder_width"] is None
        assert not any("assumed to be 20 cm" in warning for warning in calibrated["warnings"])
        payload.pop("camera_calibration")
        video_path = tmp_path / "input.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 250))
        assert writer.isOpened()
        try:
            for frame_index in range(40):
                value = 20 if frame_index < 32 else min(255, 220 + (frame_index - 32) * 7)
                writer.write(np.full((250, 200, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        video_bytes = video_path.read_bytes()
        connection.request("POST", "/capture-video", video_bytes, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        captured = json.loads(response.read())
        assert response.status == 200
        assert 3.5 <= captured["timestamp"] < 4
        assert captured["image"].startswith("data:image/jpeg;base64,")
        assert captured["quality_score"] > 0
        connection.request("POST", "/stream-video", video_bytes, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "application/x-ndjson"
        events = [json.loads(line) for line in response.read().splitlines()]
        assert [event["type"] for event in events] == ["frame"] * 8 + ["captured"]
        assert events[-1]["image"] == events[-2]["image"]
        assert events[-1]["timestamp"] == events[-2]["timestamp"]
        assert events[0]["timestamp"] < events[1]["timestamp"]
        connection.request("POST", "/capture-video", b"invalid video", {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        assert response.status == 400
        assert "decode" in json.loads(response.read())["error"]
        for bad_face_height in (0, 10, 50):
            face.height = bad_face_height
            status, probe = post("/probe-frame")
            assert status == 200 and probe["eligible"]
            status, analysis = post("/analyze")
            assert status == 200
            assert analysis["estimation_mode"] == "simulated_camera_geometry"
        face.height = 26
        estimator.detected_people = 2
        assert post("/probe-frame")[1]["eligible"] is False
        assert post("/analyze")[0] == 400
        connection.request("POST", "/capture-video", video_bytes, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        assert response.status == 400
        assert "No confirmed full-body" in json.loads(response.read())["error"]
        connection.request("POST", "/stream-video", video_bytes, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        events = [json.loads(line) for line in response.read().splitlines()]
        assert events[0]["type"] == "frame"
        assert "No confirmed full-body" in events[-1]["error"]
        assert list(tmp_path.iterdir()) == [video_path]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
