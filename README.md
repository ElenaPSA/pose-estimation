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