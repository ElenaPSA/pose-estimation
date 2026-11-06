import base64
import json
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

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
    assert 'aria-label="Backrest Inclination"' in HTML
    assert 'aria-label="Seat Height"' in HTML
    assert 'aria-label="Seat Depth"' in HTML
    assert HTML.count('class="movement-label"') == 3
    assert ".movement-label {" in HTML
    assert "color: #ff861c; font-size: 10px; font-weight: 700" in HTML
    assert "Backrest Inclination</span>" in HTML
    assert "Seat Height</span>" in HTML
    assert "Seat Depth</span>" in HTML
    assert 'id="movementBackrestValue"' in HTML
    assert 'id="movementVerticalValue"' in HTML
    assert 'id="movementHorizontalValue"' in HTML
    assert 'aria-label="Seated driver illustration showing backrest recline movement"' in HTML
    assert 'aria-label="Seated driver illustration showing vertical seat movement"' in HTML
    assert 'aria-label="Seated driver illustration showing fore and aft seat movement"' in HTML
    assert HTML.count('class="movement-row"') == 3
    assert HTML.count('href="/assets/driver-seat-profile.png"') == 3
    assert HTML.count('class="movement-arrow"') == 3
    assert 'id="backrestMovementArrow"' in HTML
    assert 'id="verticalMovementArrow"' in HTML
    assert 'id="horizontalMovementArrow"' in HTML
    assert 'M250 269 A28 28 0 0 1 278 241 M271 242 L278 241 L276 248' in HTML
    assert 'M278 241 A28 28 0 0 0 250 269 M256 269 L250 269 L252 262' in HTML
    assert 'M88 278 H184 M176 272 L184 278 L176 284' in HTML
    assert 'M184 278 H88 M96 272 L88 278 L96 284' in HTML
    assert 'M265 106 H286' not in HTML
    assert "seatHeight < 0" in HTML
    assert "seatDepth < 0" in HTML
    assert "backrest < 100" in HTML
    assert 'stroke: #f27b16' in HTML
    assert "#horizontalMovementArrow { stroke-width: 2.5; }" in HTML
    assert "font-size: 15px; font-weight: 800" in HTML
    assert ".movement-copy .movement-number { color: #ff861c; font-size: 13px; }" in HTML
    assert "position: absolute; right: 9px; bottom: 5px" in HTML
    assert "grid-template-columns: repeat(3, minmax(0, 1fr))" in HTML
    assert "width: min(100%, 620px)" in HTML
    assert 'class="movement-unit">mm</span>' in HTML
    assert 'class="movement-unit">°</span>' in HTML
    assert "font-size: 20px; font-weight: 800" not in HTML
    assert "Lumbar support" not in HTML
    assert "Headrest" not in HTML
    assert "Pedal reach" not in HTML
    assert "NO SEAT ACTUATION" in HTML
    assert 'src="/assets/automotive-seat-adjustment-scheme.png"' in HTML
    assert 'src="/assets/automotive-seat-proposed-overlay.png"' in HTML
    assert 'src="/assets/automotive-seat-reference-shadow.png"' not in HTML
    assert 'id="seatProposal"' in HTML
    assert 'id="seatBackrestProposal"' in HTML
    assert 'id="preview"' in HTML
    assert "PREVIEW SIMULATED ADJUSTMENT" in HTML
    assert "seatProposal.getAnimations().forEach(animation => animation.cancel())" in HTML
    assert "const recline = clamp(backrestAngle - 100, -8, 10);" in HTML
    assert "rotate(${recline}deg)" in HTML
    assert "Showing orange simulated seat position" in HTML
    assert "const motionScale = 1.5 * 1.5 * 2;" in HTML
    assert "clamp(-depthOffset * 0.3, -18, 18) * motionScale * scale" in HTML
    assert "clamp(-heightOffset * 0.35, -18, 18) * motionScale * scale" in HTML
    assert 'style="background:#424d5d"></i>Fixed original position' in HTML
    assert "orange overlay previews" in HTML
    assert "against the fixed reference position" in HTML
    assert 'id="seatAnimation"' not in HTML
    assert 'id="driverMovement"' not in HTML
    assert 'src="/assets/driver-seat-profile.png"' not in HTML
    assert "Seat-position visualization · visual only" not in HTML
    assert 'class="configuration-layout"' not in HTML
    assert HTML.count('class="movement-row"') == 3
    assert 'id="frontalFile"' in HTML
    assert 'id="lateralFile"' not in HTML
    assert 'id="referenceLength"' in HTML
    assert 'id="analyzeImages"' in HTML
    assert "LOAD, SCAN AND ANALYZE IMAGE</h2>" in HTML
    assert 'id="detectionViewport"' in HTML
    assert 'id="detectionStage"' in HTML
    assert 'id="detectionImage"' in HTML
    assert "Detecting Person</div>" in HTML
    assert 'id="detectionStatus"' in HTML
    assert "function startDetectionAnimation()" in HTML
    assert "detectionStatusTimers.push(window.setTimeout(() => {" in HTML
    assert "      }, duration));" in HTML
    assert HTML.index("startDetectionAnimation();") < HTML.index('fetch("/analyze"')
    assert "function showDetectionBox(box)" in HTML
    assert ".detection-viewport.scanning #detectionImage { animation: grayscale-reveal 5s ease-out both; }" in HTML
    assert "@keyframes camera-highlight" in HTML
    assert "@keyframes person-scan" in HTML
    assert ".detection-viewport.scanning .detection-sweep { animation: person-scan 5s linear both; }" in HTML
    assert "100% { transform: translateY(var(--scan-distance)); opacity: .75; }" in HTML
    assert 'id="detectionColorImage"' in HTML
    assert "detectionColorImage.src = view.dataUrl;" in HTML
    assert "detectionColorImage.style.clipPath =" in HTML
    assert "await scanAnimation;" in HTML
    assert HTML.count('<line id="cameraTo') == 4
    assert '<rect id="detectedPersonBox" class="person-box"/>' in HTML
    assert "<polygon" not in HTML
    assert ".detection-viewport.detected .detection-overlay { animation: camera-highlight" in HTML
    assert "Image ready. Press ANALYZE IMAGE to detect the person." in HTML
    assert "ANALYZE IMAGE</button>" in HTML
    assert "Torso and leg measurements require a side-view image" in HTML
    assert "APPROXIMATE DRIVER MEASUREMENTS" in HTML
    assert "Height uses the frontal full-body detection box" in HTML
    assert "MODEL PERFORMANCE" in HTML
    assert 'id="morphologyConfidence"' in HTML
    assert 'id="faceConfidence"' in HTML
    assert "calibrated full-body detection box" in HTML
    assert 'id="measureHeight"' in HTML
    assert 'id="measureShoulder"' in HTML
    assert 'id="measureHip"' in HTML
    assert 'id="measureFace"' in HTML
    assert 'id="measureTorso"' in HTML
    assert 'id="measureLegs"' in HTML
    assert 'id="measureUpperBody"' not in HTML
    assert 'id="measureLowerBody"' not in HTML
    assert 'id="measureUpperLeg"' not in HTML
    assert 'id="measureLowerLeg"' not in HTML
    assert '<span class="number">' not in HTML
    assert "SEGMENT PROPORTIONS" not in HTML
    assert 'id="legs"' not in HTML
    assert 'id="torso"' not in HTML
    assert "proportionWord" not in HTML
    assert HTML.index("LOAD, SCAN AND ANALYZE IMAGE") < HTML.index("DRIVER DIMENSIONS")
    assert HTML.index('id="analyzeImages"') < HTML.index("DRIVER DIMENSIONS")
    assert HTML.index("APPROXIMATE DRIVER MEASUREMENTS") < HTML.index(
        "CONFIGURATION VECTOR · SIMULATION"
    )


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
