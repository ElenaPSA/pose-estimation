# Driver Pose Estimation

Real-Time Driver Pose Estimation using Webcam, OpenCV and MediaPipe.

---

# 1. Project Overview

Driver Pose Estimation is a computer vision application that captures live video from a webcam and estimates the driver's body pose in real time.

The application uses:

- OpenCV for video acquisition and visualization
- MediaPipe Pose for landmark detection
- NumPy for numerical processing
- A modular Python architecture
- Modern Python packaging using `pyproject.toml`

The project is intentionally simple, maintainable and extensible so that future features can be added easily:

- Head pose estimation
- Driver attention estimation
- Eye gaze tracking
- Drowsiness detection
- Driver monitoring systems
- RTMaps integration
- ROS integration
- MQTT publishing
- Video recording

---

# 2. High-Level Architecture

```text
┌───────────────────┐
│     Webcam        │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Video Capture    │
│ video_capture.py  │
└─────────┬─────────┘
          │ frame
          ▼
┌───────────────────┐
│   Pose Detector   │
│ pose_detector.py  │
│   MediaPipe Pose  │
└─────────┬─────────┘
          │ landmarks
          ▼
┌───────────────────┐
│ Pose Visualizer   │
│pose_visualizer.py │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ OpenCV Display    │
└───────────────────┘
```

---

# 3. Complete Processing Flow

The application continuously executes the following loop:

```text
Capture Frame
      ↓
Pose Detection
      ↓
Landmark Extraction
      ↓
Skeleton Drawing
      ↓
Display
      ↓
Next Frame
```

---

# 4. Project Structure

```text
driver-pose-estimation/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── driver_pose_estimation/
│       │
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── video_capture.py
│       ├── pose_detector.py
│       └── pose_visualizer.py
│
└── tests/
```

---

# 5. Description of Every File

---

## pyproject.toml

Modern Python packaging configuration.

Responsibilities:

- package definition
- dependencies
- build configuration
- installation entry point

Example:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "driver-pose-estimation"
version = "0.1.0"

dependencies = [
    "opencv-python",
    "mediapipe",
    "numpy"
]
```

---

## __init__.py

Marks the folder as a Python package.

Even if empty, this file is mandatory.

```python
# empty file
```

---

## config.py

Contains all configurable parameters.

Example:

```python
from dataclasses import dataclass

@dataclass
class Settings:
    CAMERA_ID = 0
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    WINDOW_NAME = "Driver Pose Estimation"
```

---

## video_capture.py

Handles webcam acquisition.

Responsibilities:

- open camera
- capture frames
- release resources

Input:

```text
Physical webcam
```

Output:

```text
OpenCV frame
```

---

## pose_detector.py

Wraps MediaPipe Pose.

Responsibilities:

- initialize MediaPipe
- process frames
- estimate body landmarks

Input:

```text
Frame
```

Output:

```text
Pose landmarks
```

Examples of detected landmarks:

```text
Nose
Eyes
Ears
Shoulders
Elbows
Wrists
Hips
Knees
Ankles
```

---

## pose_visualizer.py

Draws landmarks and skeleton connections.

Responsibilities:

- draw joints
- draw skeleton
- visualize results

Input:

```text
Frame + landmarks
```

Output:

```text
Annotated frame
```

---

## main.py

Main application entry point.

Responsibilities:

- initialize components
- execute processing loop
- display frames
- handle application shutdown

---

# 6. Installation

---

## Step 1 - Clone the Repository

```powershell
git clone <repository-url>
```

Example:

```powershell
git clone https://github.com/your-company/driver-pose-estimation.git
```

Enter the project:

```powershell
cd driver-pose-estimation
```

---

## Step 2 - Open VS Code

```powershell
code .
```

Or:

```text
File
 → Open Folder
 → driver-pose-estimation
```

---

## Step 3 - Open PowerShell Terminal

In VS Code:

```text
Terminal
    New Terminal
```

You should see:

```powershell
PS C:\Users\U604618\Stellantis\driver-pose-estimation>
```

---

## Step 4 - Verify Python Installation

```powershell
python --version
```

or:

```powershell
py --version
```

Expected:

```text
Python 3.10+
```

---

## Step 5 - Create Virtual Environment

```powershell
py -m venv .venv
```

This creates:

```text
.venv/
```

---

## Step 6 - Activate Virtual Environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Expected:

```text
(.venv) PS C:\Users\U604618\Stellantis\driver-pose-estimation>
```

---

## Step 7 - Upgrade Pip

```powershell
python -m pip install --upgrade pip
```

---

## Step 8 - Install the Project

Install in editable mode:

```powershell
pip install -e .
```

This command:

- installs dependencies
- installs the package locally
- enables live development

---

## Step 9 - Verify Installation

```powershell
pip list
```

Expected packages:

```text
mediapipe
opencv-python
numpy
```

---

# 7. Running the Application

---

## IMPORTANT

Do NOT execute:

```powershell
python src/driver_pose_estimation/main.py
```

because your project uses relative imports:

```python
from .config import Settings
```

This will produce:

```text
ImportError:
attempted relative import with no known parent package
```

---

## Correct Execution

From the project root:

```powershell
cd C:\Users\U604618\Stellantis\driver-pose-estimation
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run:

```powershell
python -m driver_pose_estimation.main
```

---

# 8. Runtime Architecture

When the application starts:

```text
main.py
   │
   ▼
VideoCapture
   │
   ▼
Webcam
   │
   ▼
Frame
   │
   ▼
PoseDetector
   │
   ▼
Body Landmarks
   │
   ▼
PoseVisualizer
   │
   ▼
Annotated Frame
   │
   ▼
OpenCV Window
```

The sequence is repeated for every frame.

---

# 9. Expected Result

After startup:

```text
+------------------------+
| Webcam Image           |
|                        |
|       Skeleton         |
|                        |
+------------------------+
```

You should see:

- live webcam stream
- body landmarks
- skeleton connections
- smooth real-time tracking

---

# 10. Stopping the Application

Close the window or press:

```text
q
```

The application will:

```text
Release webcam
Destroy OpenCV windows
Exit cleanly
```

---

# 11. Troubleshooting

---

## Error

```text
attempted relative import with no known parent package
```

### Cause

Wrong command:

```powershell
python src/driver_pose_estimation/main.py
```

### Solution

```powershell
pip install -e .
python -m driver_pose_estimation.main
```

---

## Error

```text
ModuleNotFoundError: mediapipe
```

### Solution

```powershell
pip install mediapipe
```

---

## Error

```text
ModuleNotFoundError: cv2
```

### Solution

```powershell
pip install opencv-python
```

---

## Error

```text
Cannot open camera
```

### Check webcam

```powershell
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

Expected:

```text
True
```

---

# 12. Future Extensions

## Pre-entry morphology profile (prototype)

Before the driver enters the vehicle, run the separate two-view capture command:

```powershell
driver-pose-capture
```

Stand upright with the full body visible in the camera. Press `F` for the
frontal view, turn sideways, then press `L` for the lateral view. Press `Q` to
exit. The frontal capture requires both sides of the shoulders, hips, knees,
and ankles; the lateral capture requires a visible face landmark and a
shoulder-to-ankle body chain on at least one side. If landmarks are missing,
the capture is rejected so the driver can reposition and try again.

The command saves normalized, two-dimensional image-space measurements to
`outputs/morphology_profile.json`. It does not save the captured images. The
profile includes shoulder and hip widths and torso and leg segment lengths
relative to the face-to-ankle image span, where visible.

It also saves a side-by-side frontal/lateral visualization with the detected
pose skeleton overlaid in green to `outputs/morphology_visualization.png`.
This visualization contains the two captured camera frames; treat it as
personal image data. The frames are held in memory during the capture session
and are written only as part of this annotated comparison.

This is not a physical height estimate: the pose model has no top-of-head
landmark, and image pixels have no physical scale without camera/scene
calibration. Perspective and body orientation also affect 2D measurements.
The profile therefore does not generate a seat-position recommendation.
Vehicle geometry and seat calibration are needed before a recommendation can
be derived; this prototype does not actuate a real seat.

After adding the current virtual environment's scripts directory to `PATH`, or
while the environment is activated, the command can be run from the project
root. Alternatively:

```powershell
python -m driver_pose_estimation.morphology_capture
```

## Seat settings HMI prototype

Launch the local browser HMI with:

```powershell
driver-seat-hmi
```

Or run it as a module with `python -m driver_pose_estimation.seat_hmi`.
It opens on `http://127.0.0.1:8765`; use `--port` to select another port.
Upload one frontal image. After you press **Analyze image**, a detection panel
beside the preview runs a five-second grayscale scan while local person
detection runs. When the scan finishes, the detected person's actual bounding
box is outlined in orange and connected at its four corners to an illustrated
camera at the lower left. The original image color is restored only inside the
detected box; the rest of the image remains grayscale. The scan effect is
simulated; the box comes from the pose detector's result. Enter the physical
length of a vertical reference marker visible beside the person, then click the
marker's two endpoints in the preview before selecting **Analyze image**. YOLO
pose inference and morphology analysis run locally; uploaded image bytes are not
saved. The image must be JPEG, PNG, or WebP and smaller than 8 MB. Use an
upright, full-body view with the marker and person at the same distance from the
camera and in the same plane.

The HMI uses the frontal YOLO person-detection box, scaled by the reference
marker, as its primary height estimate. The box must include the entire person
from the top of the head to the soles. YOLO pose landmarks provide approximate
shoulder/hip widths, and YuNet face detection provides approximate face height.
Torso and leg measurements and the segment-based height cross-check are
unavailable because they require a side-view image. The image upload/analysis
workflow is separate from the manual segment proportion controls. There are no
manual segment-proportion controls in the HMI. YuNet face detection is used
only for the approximate face-height
measurement and model-confidence display; a missed face no longer prevents the
full-body height estimate. Keep the person and marker upright, coplanar, and
at a similar camera distance. The estimator remains an approximate 2D
measurement, not a learned physical-height predictor. YuNet model weights and
license are in
`src/driver_pose_estimation/assets/`; the model comes from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Build category remains a rough person-box-width heuristic. Perspective,
clothing, pose, person-box quality, landmark quality, and marker placement
affect estimates, so treat them as approximate prototype values. If a person
box is detected but required pose landmarks are incomplete, the HMI still shows
the detected box and reports unavailable shoulder/hip measurements. If no
person is detected or calibration is invalid, the HMI reports an error instead
of filling the profile. The older normalized morphology JSON capture remains
available via `driver-pose-capture`.

The seat parameters are illustrative demo values for the HMI only. The
“Preview simulated adjustment” button plays an on-screen animation of the
illustrative seat-height and cushion-depth movements using the currently
displayed demo values. The configuration vector includes three movement
illustrations based on the seated-driver image. Orange arrows and values are
overlaid on the images; vertical and horizontal arrow directions reverse for
negative offsets, and the backrest arc reverses for recline below its
reference angle.
The supplied seat-adjustment scheme remains fixed as the
reference in its original colors; an orange-tinted cutout of the large seat
moves over it when previewed. The three smaller diagrams remain unchanged.
Clicking the button again replays the animation. The displayed fore/aft and
vertical movement is amplified by 4.5x relative to the original preview for
visibility and is not to scale; the animation does not actuate a seat or send
vehicle commands.
The profile height bands, parameter offsets, and recommendation rules require
validation and vehicle-specific calibration before they could be considered
for real use.

---

The architecture can easily be extended to:

```text
Driver Pose Estimation
        │
        ├── Head Pose Estimation
        ├── Eye Gaze Tracking
        ├── Driver Monitoring
        ├── Drowsiness Detection
        ├── Multi-Camera Support
        ├── Video Recording
        ├── MQTT Publishing
        ├── RTMaps Integration
        └── ROS Integration
```

---

# 13. Summary

The complete data flow is:

```text
Webcam
   ↓
Video Capture
   ↓
MediaPipe Pose
   ↓
Landmark Detection
   ↓
Skeleton Visualization
   ↓
OpenCV Display
```

The project is organized as a clean Python package and should always be launched with:

```powershell
python -m driver_pose_estimation.main
```

after:

```powershell
pip install -e .
```

from the project root directory.