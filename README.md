# Driver Pose Estimation

A Python project for detecting a driver's body pose, summarizing image-space morphology, and previewing illustrative seat adjustments. The primary browser workflow accepts a **local video file**; it does not use a webcam.

> **Height is not calibrated by default.** The HMI uses an explicitly simulated camera model (70-degree horizontal field of view, 0.85 m mount height, level and forward-facing, zero lens distortion). Those values are assumptions, not measurements. The displayed height and sensitivity interval are not physical measurements.

## What the Project Does

- Runs Ultralytics YOLO pose detection and returns person boxes plus 17 COCO body keypoints.
- Uploads a local video to the Seat HMI, checks frames for a complete upright person, and analyzes a selected frame.
- Computes image-space frontal/lateral morphology ratios in a separate pre-entry capture workflow.
- Offers a chessboard intrinsic-calibration utility and vehicle-pose geometry API. The HMI does not currently have a calibration-file upload control.
- Shows illustrative seat targets in a browser simulation. It sends no vehicle commands and does not actuate a real seat.

The HMI runs YuNet face detection on the selected frame and displays its detection confidence. Face size is not used as the height scale.

## Requirements

- Windows, macOS, or Linux.
- Python 3.10 through 3.14 (`requires-python = ">=3.10,<3.15"`).
- A browser for the HMI.
- A webcam is needed only for the separate live-camera commands, not for the HMI video workflow.

The project dependencies are NumPy, OpenCV, `imageio-ffmpeg` (bundled FFmpeg for local browser-preview transcoding), and Ultralytics. The configured pose model defaults to `yolo11n-pose.pt` in the project root; set `POSE_MODEL` if the weights are stored elsewhere.

## Install on Windows

Open PowerShell in the project directory:

```powershell
cd C:\Users\U604618\Stellantis\pose-estimation
py --list
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Use an installed Python version from 3.10 to 3.14 in place of `3.13` if necessary. If PowerShell blocks activation, allow it for this terminal only, then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Activation is optional. Commands can also use the environment's interpreter directly, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Editable installation makes the console commands below available. Run them from the project root so the default relative model path resolves correctly.

## Launch the HMI

The main workflow is to upload a video file in the browser:

```powershell
driver-seat-hmi
```

Equivalent module command:

```powershell
python -m driver_pose_estimation.seat_hmi
```

Choose a browser-playable video in the **Driver video** file input. The limit is 500 MB. The HMI listens on `http://127.0.0.1:8765/` and attempts to open the browser automatically. Choose a different port with either command:

```powershell
driver-seat-hmi --port 8766
python -m driver_pose_estimation.seat_hmi --port 8766
```

The browser plays the uploaded video while checking frames. If Chromium cannot decode the source codec, the HMI uses bundled FFmpeg to make a local H.264/MP4 preview (downscaled to a maximum 640px side and 12fps); browser playback and frame checks use that preview. If transcoding is unavailable, the local OpenCV fallback processes sampled frames from the original and displays them instead. Frame checks are sequential, capped at 240 samples; inference time can cause brief appearances between checks to be missed. Selection tracks the same person, requires consecutive complete-body detections, and prefers a near-camera frame that still passes the full-body gate. If no such near-camera pair occurs, the best stable complete frame is used at the end. The detection scan and analysis then run on the right.

The HMI temporarily stores the upload and any transcoded preview in a system temporary directory and removes them when processing finishes. It does not save permanent copies or send video externally. A browser decoding error can produce an aborted source-blob request in the developer console; the local transcode/fallback handles that case.

## Height Estimate and Camera Assumptions

With no measured intrinsics or mount pose, the HMI creates a simulated camera model:

| Parameter | Default assumption |
|---|---:|
| Horizontal field of view | 70 degrees |
| Camera height above flat ground | 0.85 m |
| Mount orientation | Level, facing forward |
| Lens distortion | Zero |
| Ground plane | Vehicle `z = 0` |
| Sensitivity interval | 0.75-0.95 m camera height |

For height, the code uses the person-box top center as the head image point and the box bottom plus ankle-centered horizontal position as the ground-contact image point. It undistorts those pixels, constructs rays, intersects the foot ray with the ground plane, and projects the head ray onto the same vertical line. The reported sensitivity interval recomputes height at the two mount-height endpoints; it does not cover uncertainty in FOV, pitch, roll, lens distortion, vehicle pitch, body lean, or landmark placement.

This is a geometry simulation, **not camera calibration**. A plausible-looking number does not make the assumed camera parameters true. For example, on the checked `rotated_2.mp4` clip the simulated path selected approximately 9.08 seconds and returned about 176.5 cm, with a mount-height sensitivity interval of about 155.7-197.3 cm. Treat this as a demonstration only.

The HMI displays frontal shoulder width, hip width, torso length, and upper,
lower, and total leg length in **simulated centimeters**. The analyzer first
computes 2D ratios to the face-center-to-ankle-center pixel span, then scales
those ratios using the simulated person-box height. These values are not
calibrated physical dimensions and are not corrected for perspective or body
orientation. The reported mount-height sensitivity range applies to height;
morphology values inherit the same simulated scale. Seat targets derived from
the simulated height are illustrative only.

## Optional Camera Calibration Tools

The calibration utility estimates **intrinsics** from chessboard images. It does not infer a vehicle camera mount pose from the chessboard. You must separately measure the camera position and orientation relative to the vehicle/floor frame if using calibrated geometry.

Example with a 9-by-6 inner-corner board and measured 25 mm squares:

```powershell
driver-camera-calibrate --images "calibration/*.jpg" --board-columns 9 --board-rows 6 --square-size-mm 25 --output camera_calibration.json
```

Use at least eight sharp views at varied image positions and board orientations. Images must share resolution and use the same camera focus, zoom, crop, and resolution as the video. `board-columns` and `board-rows` count **inner corners**, not squares. `square-size-mm` must be physically measured.

To include measured vehicle pose, provide `--camera-pose-json`:

```powershell
driver-camera-calibrate --images "calibration/*.jpg" --board-columns 9 --board-rows 6 --square-size-mm 25 --camera-pose-json camera_pose.json --output camera_calibration.json
```

The pose file contains numeric values in this shape:

```json
{
  "position_m": [0.0, 0.0, 0.85],
  "rotation_matrix": [
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0]
  ]
}
```

**The values above are placeholders, not a calibration for this vehicle.** `position_m` is `[right, forward, up]` in the vehicle frame. The rotation maps OpenCV camera axes (`x right, y down, z forward`) into vehicle axes (`x right, y forward, z up`). The vehicle floor is `z=0`. The JSON includes chessboard reprojection RMS; a low RMS alone does not validate the measured camera pose or height estimate.

The current HMI always uses its built-in simulated profile and has no calibration-file input. The `/analyze` API accepts a `camera_calibration` object for integrations that supply measured calibration. Do not copy the example pose values into a real measurement workflow.

## Other Commands

### `driver-pose`: live webcam pose/comfort demo

```powershell
driver-pose
```

Equivalent:

```powershell
python -m driver_pose_estimation.main
```

This is the separate **webcam** workflow. It runs pose/posture analysis and a simulated comfort agent; it is not the uploaded-video HMI. The simulated vehicle and seat do not move a real vehicle. Common keyboard actions are `C` (comfort check), `1`/`2` (comfortable/uncomfortable), `F` (too far), `R` (too close), `B` (lower-back discomfort), `H`/`L` (too high/low), `Y` (apply simulated adjustment), `N` (cancel), `M` (simulate moving), `P` (simulate parked), and `Q`/`Esc` (quit).

### `driver-pose-capture`: two-view morphology capture

```powershell
driver-pose-capture
```

Equivalent:

```powershell
python -m driver_pose_estimation.morphology_capture
```

This separate webcam workflow asks for an upright frontal view (`F`) and then a lateral view (`L`); press `Q` to finish. It saves `outputs/morphology_profile.json` and `outputs/morphology_visualization.png`. These are normalized image-space proportions, not physical measurements, and do not produce a seat recommendation.

### `person_height_video`: annotated video height experiment

This module accepts a webcam index by default or a video file with `--source`:

```powershell
python -m driver_pose_estimation.person_height_video --source .\rotated_2.mp4 --output .\outputs\person_height_output.mp4 --capture-output .\outputs\person_height_capture.jpg
```

Useful arguments:

| Argument | Meaning |
|---|---|
| `--source` | Camera index such as `0` (default), or a video path |
| `--output` | Annotated video output; default `person_height_output.mp4` |
| `--capture-output` | Best tracked-person still; default `person_height_capture.jpg` |
| `--reference-height-cm` | Known height of a vertical reference object |
| `--reference-height-px` | Its image-space height in the same view |
| `--no-display` | Write output without opening the preview window |

For centimeter output, supply both reference arguments, and place the measured vertical reference at approximately the same camera depth as the person. Without it, the video annotation reports image-space pixels, not physical centimeters. The script tracks one person, tolerates up to five missed frames, and saves the clearest/largest fully visible observation. A single pixel-to-centimeter scale is invalid across substantially different camera depths.

### `driver-camera-calibrate`: chessboard intrinsic calibration

```powershell
driver-camera-calibrate --help
```

Equivalent:

```powershell
python -m driver_pose_estimation.camera_calibration --help
```

This writes a calibration JSON; it does not change the HMI's simulated defaults. The pose input is user-measured, not estimated by the chessboard routine.

## Configuration

`driver-pose` and the webcam morphology capture use environment variables. Defaults are shown below; set them in the PowerShell session before running a command.

| Variable | Default | Purpose |
|---|---:|---|
| `CAMERA_INDEX` | `0` | Webcam index for live-camera commands |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Requested webcam frame size |
| `CAMERA_FPS` | `15` | Requested webcam frame rate |
| `POSE_MODEL` | `yolo11n-pose.pt` | YOLO pose weights |
| `POSE_DEVICE` | `cpu` | Inference device |
| `POSE_CONFIDENCE` | `0.25` | Person detection threshold |
| `POSE_IOU` | `0.45` | Detection NMS IoU threshold |
| `KEYPOINT_CONFIDENCE` | `0.50` | Pose landmark threshold |
| `POSE_IMAGE_SIZE` | `640` | YOLO inference image size |
| `PROCESS_EVERY_N_FRAMES` | `1` | Live camera frame sampling interval |
| `MIRROR_CAMERA` | `true` | Mirror live webcam display |
| `DISPLAY_VIDEO` | `true` | Show the live webcam window |
| `OUTPUT_DIR` | `outputs` | JSON/profile output directory |
| `SAVE_HISTORY` | `true` | Save live pose history |
| `LOG_EVERY_N_FRAMES` | `10` | Live processing log interval |

Example:

```powershell
$env:CAMERA_INDEX = "1"
$env:POSE_DEVICE = "cpu"
driver-pose
```

These webcam settings do not select the HMI video. The HMI receives its input from the browser's video-file picker.

## Outputs

- `driver-pose`: `outputs/pose_history.json` and `outputs/pose_summary.json` when history saving is enabled.
- `driver-pose-capture`: `outputs/morphology_profile.json` and `outputs/morphology_visualization.png`.
- `person_height_video`: annotated video and selected still, at the paths given by `--output` and `--capture-output`.
- `driver-camera-calibrate`: `camera_calibration.json` by default, or the path passed to `--output`.
- HMI: no permanent media output; fallback decoding uses a temporary local video file that is removed after processing.

## Tests and Lint

Install development dependencies with `python -m pip install -e ".[dev]"`, then run:

```powershell
python -m pytest
ruff check src tests
```

## Troubleshooting

- **`driver-seat-hmi` is not recognized:** activate `.venv` and install the project with `python -m pip install -e .`, or run `python -m driver_pose_estimation.seat_hmi`.
- **Port 8765 is already in use:** start with `driver-seat-hmi --port 8766` and open the printed URL.
- **HMI cannot decode the video in-browser:** it first attempts a local H.264/MP4 transcode with bundled FFmpeg; if that is unavailable, it displays sampled frames. Use a valid video file below 500 MB.
- **`driver-pose` cannot open a camera:** check `CAMERA_INDEX` and confirm Windows camera access is allowed. This does not affect HMI video-file uploads.
- **Model weights cannot be found:** run from the project root or set `POSE_MODEL` to the weights path.
- **A calibrated height is needed:** collect same-camera chessboard views and measure the camera mount pose relative to the vehicle floor. The simulated HMI result is not a substitute.
