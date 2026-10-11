# Project Argos


Project Argos is a local, consent-based stereo computer-vision prototype. It captures one or two webcams, optionally rectifies stereo images, detects faces with OpenCV YuNet, stabilizes detections with persistent track IDs, and shows image coordinates in a minimal C-scope display. Optional local ArcFace recognition uses explicitly enrolled templates.

Unmatched people stay `Unknown`; the app never enrolls faces on its own. Click **IDENTITIES** in the C-scope or press **c** to open the graphical catalogue. Its text uses the bundled Fira Code fonts in `static/`. It lists each permanent ID (`FACE-0001`, `FACE-0002`, …), display name, Friendly/Unknown status, and template count. Select a row to rename it or change its status. The catalogue lists active track IDs. Select the exact `TRK-###` row, then choose **ENROLL TRACK** to create a new face ID from that track or select a saved `FACE-####` and choose **ASSOCIATE** to add that track's face template to it. The app waits for the selected track's local face embedding if it is still being computed. **Delete** removes a selected saved identity and its local embeddings after confirmation; the ID is never reused. Every 60 seconds, a background check looks for identity pairs above `identity.duplicate_similarity_threshold` (default 0.92). It asks before deleting anything and keeps the older ID if the user confirms a duplicate. Catalogue metadata and ArcFace templates stay in local `data/face_templates.json`. Track N remains separate from the saved face ID. Corrupt catalogue files are preserved and ignored with a warning.

## Modules

```text
camera/         dual-webcam capture and stereo calibration/rectification loader
detection/      FaceDetector API and YuNet (confidence filtering and NMS)
tracking/       nearest-centroid association and EMA-smoothed fused image-space state
identity/       optional local ArcFace ONNX embeddings and explicit enrollment
geometry/       pixel and normalized image coordinate helpers
ranging/        RangeProvider interface (stereo disparity is a future provider)
fusion/         future multi-provider fusion extension point
visualization/  camera overlay and 2D C-scope foundation
config/         camera indices, baseline, image size, and pipeline settings
tests/          coordinate and target-state logic tests
```

## Setup

Use Python 3.10 or newer. From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Edit `config/default.yaml` to set `cameras.mode` to `single` or `stereo`, camera indices/resolution/FPS, and measured physical baseline. YuNet downloads its configured ONNX file into `models/` on first start if missing and network access works. Tune `detection.score_threshold`, `nms_threshold`, and `input_size` (`frame` or a square size such as 320). `performance.detection_cadence` runs YuNet once every N frames while tracking bridges skipped frames. Face embeddings run on a background worker with a bounded queue. Unmatched live tracks are rechecked against saved templates every `identity.unknown_recheck_seconds` (default 1.5 seconds); recognized tracks stop rechecking. Duplicate checks also run in the background using `identity.duplicate_check_interval_seconds` (default 60 seconds). The HUD shows measured FPS. Optional per-stage profiling can be enabled with `performance.profile`. The `FaceDetectorYN` implementation applies its built-in NMS.

Run with both cameras attached:

```powershell
argos
```

To start with one webcam (configured by `left_index`) and skip opening the second camera:

```powershell
argos --mode single
```

You can also double-click `Launch Argos.cmd` in File Explorer, choose Single camera or Stereo cameras, and click **Start Argos**. This opens a small Windows desktop launcher; it does not require typing a command.

Single-camera mode supports YuNet detection, tracking, optional recognition, pixel coordinates, and the C-scope. Stereo rectification and range are unavailable in this mode. Track boxes/centers are smoothed with configurable EMA alpha (`tracking.ema_alpha`, default 0.3); each target also retains `raw_box`. Stereo coordinates are currently not independently smoothed. Camera reads remain sequential and the stereo cameras are not hardware synchronized.

Press `q` or Escape to close. If no calibration exists, the app still displays detections from the left camera and clearly reports that images are unrectified and range is unavailable. Capture synchronization is sequential in this prototype; webcams and drivers may not provide hardware synchronization.

## Calibration file

When available, `calibration/stereo.yml` is expected to contain OpenCV matrices named `M1`, `D1`, `M2`, `D2`, `R`, and `T`. `M1`/`M2` are camera matrices, `D1`/`D2` distortion coefficients, and `R`/`T` are stereo extrinsics. The loader generates rectification maps. This milestone does not yet provide a calibration capture wizard or compute disparity/depth; those belong in follow-up work. Keep calibration and any future biometric material local and out of version control.

## Tests

```powershell
python -m pytest
```

Tests cover box centers, pixel/normalized coordinate conversion, target defaults, tracker ID continuity across skipped detections, catalogue ID persistence/rename/category/association/deletion, corrupt catalogue recovery, duplicate-similarity checks and confirmation UI, detection cadence, and unknown-track recheck timing. They do not require cameras.

## ArcFace weights and license

Recognition uses ONNX Runtime directly with the `w600k_r50.onnx` ArcFace recognition model from InsightFace's `buffalo_l` pack (release v0.7). The ArcFace file expects the five YuNet landmarks and a 112×112 aligned face. The model is optional: place `w600k_r50.onnx` in `models/` or set `identity.model_path`; if absent, detection/tracking still run and identity stays `Unknown`.

Both configured weights are now present in the local, git-ignored `models/` folder. For the official InsightFace [v0.7 buffalo_l release](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip), the archive SHA-256 was `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f`; only `w600k_r50.onnx` was extracted. **License:** InsightFace's Python library source is MIT, but its pretrained model weights, including manually downloaded and auto-downloaded packs, are provided for non-commercial research purposes only. The weights are not unrestricted; get separate permission from InsightFace for commercial use. YuNet weights are the OpenCV Zoo `face_detection_yunet_2023mar.onnx` model (SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`; MIT per its model directory license).

The default cosine similarity threshold (0.45) is only a starting point. Similarity varies with camera, lighting, pose, and the people enrolled; it is not a calibrated guarantee. Tune against consenting users and representative captures. The JSON file is local plaintext embedding data, so protect or delete it as appropriate.

There is no friend/foe classification, human-targeting logic, or weapon guidance.
