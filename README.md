# Driver Pose Estimation

A Python project for detecting a driver's body pose, summarizing image-space morphology, and previewing illustrative seat adjustments. The primary browser workflow accepts a **local video file**; it does not use a webcam.

> **Height is not fully calibrated.** The HMI uses an internal height estimate for size classification and morphology scaling; it does not display the height itself. The camera orientation and lens distortion remain unmeasured, so the estimate and derived measurements are not validated physical measurements.

## What the Project Does

- Runs Ultralytics YOLO pose detection and returns person boxes plus 17 COCO body keypoints.
- Uploads a local video to the Seat HMI, checks frames for a complete upright person, and analyzes a selected frame.
- Computes image-space frontal/lateral morphology ratios in a separate pre-entry capture workflow.
- Offers a chessboard intrinsic-calibration utility and vehicle-pose geometry API. The HMI does not currently have a calibration-file upload control.
- Classifies estimated height provisionally as small (below 160 cm), medium (160–180 cm inclusive), or large (above 180 cm). The HMI sends no vehicle commands.
- Shows a seat-movement simulation using the internal estimated height. A manually selected size uses a representative height; the simulation is illustrative, not a safety or comfort recommendation.

The seat simulation fits two reported trials at 163 and 185 cm. Relative to the nominal reference, `deltaX_mm = -4.545*(height_cm-174)` and `deltaZ_mm = 0.909*(height_cm-174)` approximately, clipped to seat travel limits. Results outside the trial range are unvalidated extrapolations.
The 153 and 178 cm videos supply reference heights, not measured seat targets; they do not add seat-fitting trials.

The backrest is a separate **illustrative, unvalidated hypothesis**, not a fitted comfort or driving recommendation. It assumes a nominal 50 cm torso at the 25-degree reference and retains that torso's horizontal shoulder offset: `angle = asin(50 * sin(25 degrees) / estimated_torso_cm)`, bounded to 20–30 degrees. This assumed reference and the ±5-degree cap are not measured; the detected torso is a projected, uncalibrated image-space estimate. A torso estimate outside 15–50% of estimated height, or one that is missing, leaves the backrest estimate unavailable and shows the nominal angle only as a visual placeholder. Manual size selection has no torso estimate. Control reach, comfort and the measured initial position are not captured. The preview does not actuate the seat.

The detecting-person panel shows the selected frame with available 2D pose segments and illustrative estimated lengths, rather than a bounding box. The HMI runs YuNet face detection on the selected frame; face size is not used as the height scale.

Use the view buttons to switch between video and profile, seat adjustments, animation, and reference details. Reference details include the assumed nominal-seat schematic. Camera assumptions are expandable on the video view. The layout is compact on desktop screens and allows scrolling on smaller screens or when content is expanded.

All seat views share a fixed, shaded steering-wheel schematic with rim, spokes, hub and column, positioned closer to the seat. The backrest card shows its signed movement and absolute target; the drawings have no angle labels or arcs. Without detected torso data, the drawing uses the nominal 25-degree angle and the card labels the movement unavailable. Its geometry is illustrative, not a measured control-reach model.

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

Each running server keeps its HTML and JavaScript version together. After changing code, stop the old server with `Ctrl+C` and launch it again. The HMI refuses to start if the requested port is already in use; stop the existing server or select a different port with `--port`.

The browser plays the uploaded video while the right-hand figure mirrors its playback. A separate video element scans up to four frames per second, capped at 240 samples. Once analysis completes, the right-hand figure freezes on the selected frame and reveals available pose segments; the left video continues playing. The selected frame's timestamp may differ from the current playback time. If Chromium cannot decode the source codec, the HMI uses bundled FFmpeg to make a local H.264/MP4 preview (downscaled to a maximum 640px side and 12fps); browser playback and frame checks use that preview. If transcoding is unavailable, the local OpenCV fallback scans up to four original frames per second, capped at 240 and at the source frame count, and shows sampled frames instead of synchronized full video playback. Brief appearances between checks may still be missed. Selection tracks the same person, requires consecutive complete-body detections, and prefers a near-camera frame that still passes the full-body gate. If no such near-camera pair occurs, the best stable complete frame is used at the end.

The HMI temporarily stores the upload and any transcoded preview in a system temporary directory and removes them when processing finishes. It does not save permanent copies or send video externally. A browser decoding error can produce an aborted source-blob request in the developer console; the local transcode/fallback handles that case.

## Height Estimate and Camera Assumptions

For matching landscape frames and portrait frames assumed to be rotated from this camera, the HMI uses the supplied estimated intrinsics to form a pinhole camera matrix. These are not a full calibration:

| Parameter | Default assumption |
|---|---:|
| Native active resolution | 1928 × 1088 px |
| Horizontal field of view (derived from `fx`) | 98.54 degrees |
| Vertical field of view (derived from `fy`) | 53.55 degrees |
| Reported diagonal field of view | 113.83 degrees |
| Estimated intrinsics | `fx = 830 px`, `fy = 1078 px`, `cx = 964 px`, `cy = 544 px` |
| Lens distortion | Assumed zero, not measured |
| Reported mirror camera position | X = 0.74 m aft, Y = -0.85 m, Z = 0.87 m in the supplied vehicle frame |
| Camera pose model values | yaw +93 degrees from forward toward vehicle-left (direction unverified); effective pitch -11.5 degrees downward; roll -0.1 degrees |
| Pitch / roll fitting ranges | -15 to -5 degrees / -3 to +3 degrees; 0.1-degree grid |
| Camera height above ground | 1.15 m |
| Ground plane | Vehicle `z = -0.28 m` |
| Illustrative camera-height sensitivity | 1.05-1.25 m; not a measured uncertainty interval |

The supplied intrinsic values are estimates, not a chessboard calibration. Their derived horizontal/vertical FOVs imply a diagonal FOV of about 103.41 degrees, not the reported 113.83 degrees. The HMI retains the diagonal value as reported metadata and warns about the discrepancy; lens distortion remains assumed zero.

Frames within 1% of the 1928 × 1088 active-array aspect ratio use the estimated intrinsics scaled to their dimensions. This includes the 1920 × 1080 horizontal stream. Portrait frames use the estimated native intrinsics rotated 90 degrees anticlockwise, including the principal point, and scaled to the video frame; cropping is not modeled. Use the rotated portrait videos for this workflow. Unmatched landscape aspect ratios still use the generic 70-degree simulation. The pose model uses yaw +93 degrees from vehicle-forward toward vehicle-left, effective pitch -11.5 degrees, and roll -0.1 degrees. The yaw sign and camera extrinsics are not measured.

For height, the code uses the person-box top center as the head image point and the box bottom plus ankle-centered horizontal position as the ground-contact image point. It undistorts those pixels, constructs rays, intersects the foot ray with the ground plane, and projects the head ray onto the same vertical line. The reported sensitivity interval recomputes height at the two mount-height endpoints; it does not cover uncertainty in FOV, pitch, roll, lens distortion, vehicle pitch, body lean, or landmark placement.

The provided diagram has X positive aft, Y positive right, and Z positive up. The HMI maps the reported (0.74, -0.85, 0.87) m to its existing [right, forward, up] convention as (-0.85, -0.74, 0.87) m, assuming the origins coincide. With the separately measured camera height of 1.15 m above ground, the ground plane is at vehicle Z = 0.87 - 1.15 = -0.28 m. Geometry uses the camera's vehicle-frame position and this ground plane; height is measured relative to the ground, not the vehicle Z origin.

This remains a geometry simulation, **not a complete camera calibration**. Pitch is positive upward, and roll is a rotation about the displayed camera optical axis. The effective pitch and roll were fitted using one selected frame from each of seven labeled videos, with equal weight per person in squared error. The -11.5 degree pitch differs from the user-supplied -15 degree physical-pitch assumption; neither fitted angle should be interpreted as a measured physical angle. The best grid fit (-11.5, 0.0 degrees) failed the video selection gate on one video, so the adjacent (-11.5, -0.1 degrees) fit was used and all seven videos were rechecked. Person-weighted root mean square error on those fitting frames decreased from 8.12 cm to 7.70 cm versus the previous -12/0 model, while per-video mean absolute error increased from 7.43 cm to 7.72 cm. These are training results, not independent validation; one person's reference height was measured without footwear. Yaw cannot be recovered from height in this model and remains the supplied +93 degree assumption. Measure camera orientation relative to the vehicle and calibrate lens distortion before using metric height. Changing horizontal camera position changes reconstructed vehicle-frame ground coordinates but, for the same rays and flat ground, does not itself change the person's height. Treat HMI estimates as demonstrations only.

A subsequent OpenCV-selection check added a fourth person, reported as 153 cm. Its selected frame estimated 161.5 cm, while the existing seven frames estimated 156.6–195.3 cm for people reported as 163, 178, and 185 cm. The eight-frame, person-weighted RMSE was 7.91 cm; those seven frames were used in the existing fit, so this is not independent validation. A shared linear correction reduced fitting RMSE to 7.42 cm but increased leave-one-person-out RMSE to 10.59 cm and estimated the 153 cm person at 162.7 cm when fitted to all four people. It was not adopted. Browser transcoding can select different frames. More measured camera geometry, distortion, and independent labeled people and frames are needed to demonstrate a reliable improvement.

An exploratory multi-frame check used eligible OpenCV-decoded frames sampled at up to four frames per second, with the current camera model and person tracking. The median of estimates within 0.55 seconds of the selected frame reduced person-weighted RMSE across these eight videos from 8.01 to 6.09 cm, but worsened the 153 cm example from 161.8 to 162.4 cm and the 178 cm example from 180.7 to 173.4 cm. These are uncompressed probe-frame results, not the browser analysis pipeline or independent validation. Full-video and near-camera medians performed worse overall. No multi-frame estimator was deployed.

For a level camera with a centered principal point and no distortion, the focal length cancels from the estimated height for the same head and foot pixels. Consequently, supplying a different FOV alone may not change the result or improve accuracy; measured camera height, pitch and roll, lens distortion, and video rotation/crop are needed to establish the actual geometry.

The HMI displays frontal shoulder width, hip width, torso length, and upper,
lower, and total leg length in **simulated centimeters**. The analyzer first
computes 2D ratios to the face-center-to-ankle-center pixel span, then scales
those ratios using the simulated person-box height. These values are not
calibrated physical dimensions and are not corrected for perspective or body
orientation. The reported mount-height sensitivity range applies to height;
morphology values inherit the same simulated scale. The illustrative seat
simulation uses the same unvalidated height estimate.

## Optional Camera Calibration Tools

The calibration utility estimates **intrinsics and lens distortion** from chessboard images. It does not infer a vehicle camera mount pose from the chessboard. The reported camera position and height above ground are included in the simulated HMI profile, but its orientation must be measured for calibrated geometry. This is important for a side-mounted camera, whose orientation is not represented by the HMI's placeholder forward-facing pose.

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

**The values above are placeholders, not a calibration for this vehicle.** `position_m` is `[right, forward, up]` in the vehicle frame. The rotation maps OpenCV camera axes (`x right, y down, z forward`) into vehicle axes (`x right, y forward, z up`). Optionally include `ground_plane_z_m` in the pose object; it defaults to 0 for existing calibration files. For the reported mirror measurements it is -0.28 m. The JSON includes chessboard reprojection RMS; a low RMS alone does not validate the measured camera pose or height estimate.

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
