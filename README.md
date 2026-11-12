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
Select a browser-playable video (MP4/H.264 or WebM, under 500 MB).
The browser plays it visibly on the left, while frames from the live playback
are checked sequentially (at least 0.5 seconds between checks, capped at 240).
Inference latency can reduce the sampling rate; brief full-body appearances
between checks can be missed. The local
`/probe-frame` endpoint checks each frame sequentially using YOLO pose detection.
Selection requires exactly one person, an uncropped box with a 1% edge margin,
confident nose, shoulder, hip, knee and ankle landmarks, an upright pose, and a
clear YuNet face. These are visibility heuristics, not proof of full-body quality.
Two consecutive eligible frames are required, not just a single detection.
Playback then pauses and seeks back to the exact second captured frame before
the five-second grayscale scan and `/analyze` request start on the right.
The source remains visible and frozen on the left throughout analysis.
These checks do not prove identity continuity or measurement accuracy.
If browser decoding fails, `/stream-video` automatically uses the local OpenCV
decoder, streaming a visible sampled JPEG preview to the left panel, with the
same two-frame pose/face confirmation. It samples about twice per second across
the duration, capped at 240 evenly spaced samples; the browser displays these
in sequence, at most two per second, not full-frame-rate playback. This
fallback temporarily stores the uploaded video in an isolated system temporary
directory, deleted when capture succeeds or fails. The final preview frame
stays frozen on the left while the detection panel scans on the right.
Cancel aborts the browser request but an ongoing
server-side search can finish before its temporary file is removed.
No reference-length entry,
marker clicks or Analyze Image button are required. Cancel stops the browser
workflow; a model inference already running on the server may finish.
Invalid/unsupported videos, missing full-body frames and inference errors are
shown explicitly. Choose the video again to retry.

When analysis and the scan finish, the actual detected box is outlined in orange
and connected at its four corners to an illustrated camera. Original color is
restored only within the box; the rest remains grayscale. Reduced-motion mode
skips the animation delay. Normally only downscaled JPEG frames (longest side
1280 px) are sent to the local server. The fallback uploads the video locally;
neither mode saves permanent media or sends it externally.

**Uncalibrated demo estimates, not physical measurements:** an ordinary video
does not identify absolute centimetres. The selected demo approach assumes
the detected face-box height is **20 cm**, then computes
`height = person-box height in pixels / face-box height in pixels * 20 cm`.
Shoulder/hip widths inherit this same assumed scale. The displayed face height
is therefore the assumption, not an independently measured dimension.
Frames producing heights outside 120-230 cm are rejected, not clamped.
This heuristic can be substantially wrong due to face shape, perspective,
camera angle and detector box quality. It is not a learned height predictor.
Real measurements still require calibrated camera/scene geometry or a known
reference. The legacy calibrated `/analyze` payload remains supported for API
callers, but is no longer exposed in this automatic video UI.
Torso/leg measurements are unavailable. Build class remains a rough
person-box-width heuristic. Size and seat targets use the accepted demo height.
YuNet model weights and
license are in
`src/driver_pose_estimation/assets/`; the model comes from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

Perspective, clothing and landmark quality also affect the estimates.
The older normalized morphology JSON capture remains available via
`driver-pose-capture`.

The seat parameters are illustrative demo values for the HMI only. The
full-width Kinematic Studio uses one detailed right-facing SVG seat model for
the three static movement cards and the combined animation. The supplied raster
sheet/crops are no longer displayed: their left-facing seat and old origin labels
conflict with the confirmed convention. No front or top views are shown.
The model, constants, UI and animation share `assets/seat-studio.js`.

- Fixed origin O is the location of the marked cushion-bottom point with the
  seat **fully rearward and lowest**. This is not a measured automotive SRP.
- X positive is forward: **0 mm fully rearward, 255 mm fully forward**.
- Z positive is upward; travel above the lowest setting is **0-80 mm**.
  Floor-to-marker height is **70 + Z mm**, within **70-150 mm**.
- Backrest recline is **15-55 degrees from vertical**, increasing rearward.
- Y positive is vehicle-left; there is no simulated lateral movement.

The wheel is always on the right, seat cushion points right and backrest reclines
left. A reference panel before the movement cards specifies the agreed
**nominal comfort reference: X = 127.5 mm, Z = 40 mm (110 mm from floor),
recline = 25 degrees**. This is a demonstration baseline, not a universal or
validated comfortable position or the car's unknown current position.
A dedicated right-facing gray-seat schematic above the three movement cards
illustrates the fore/aft travel ruler, vertical travel ruler and recline sector
from vertical. Reference markers locate X 127.5 mm, Z 40 mm and recline 25 degrees
inside their respective limits. Floor heights and absolute reference values are
also labelled. On narrow screens the detailed schematic scrolls horizontally
within its panel to preserve label readability. The three movement cards and
combined animation are unchanged.
Static cards show one axis changing, keeping the other axes at this reference.
All four figures show a filled gray reference seat behind a filled orange
simulated seat. The slightly translucent orange seat allows overlap comparison;
at zero movement the two poses coincide. Legends identify both colors.
Dashed lines, separate orange accent lines and movement arrows are omitted.
The main card values and live
animation readouts are signed movements: `target - nominal comfort reference`.
Secondary card captions retain absolute reference and target coordinates and
physical travel ranges. Positive X is forward, positive Z upward, and positive
recline more rearward; zero means no movement.

The hero plays a cancellable 4.2-second sequence: fore/aft, elevation, then
recline, with live numeric readouts, stage highlighting and a progress bar.
Replay resets to the nominal comfort reference (zero signed movements), with
its gray seat fixed throughout. Profile changes cancel any in-progress animation
and show the new target immediately; stale animation callbacks cannot overwrite it.
Reduced-motion preference shows the final pose immediately. Drawing displacement
is 0.75 SVG units per mm on both axes with a shared hinge for the whole backrest
and headrest. Rail/cushion shapes and hinge are illustrative, not vehicle CAD.
There are no vehicle commands.

Automatic height-band demo rules remain, not a calibrated morphology-to-comfort
model: the former distances 0/15/28 mm rearward from the forward stop are rebased
as `X = 255 - distance`, giving absolute **255/240/227 mm**.
This changes coordinates, not physical target positions;
vertical `round(base + 0.6 * (height - reference height))` is limited to 0-80 mm;
backrest demo targets are 25/25/27 degrees from vertical. The reference heights
are 158/178/195 cm and vertical bases 0/20/38 mm. Build category does not change
these targets. Physical travel bounds are enforced, but do not validate comfort
or mechanical reachability of combinations. The diagram does not create an
ergonomic recommendation. Use stationary secured-vehicle evaluation and the
manufacturer's seating/restraint guidance, not these demo targets as instructions.
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