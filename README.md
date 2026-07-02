# Carton Damage Detection

Real-time detection and unique counting of damaged vs normal cartons using a YOLO object detection model with object tracking. Supports webcam input, video files, and single images.

## Overview

This project detects cartons on a conveyor or in a camera view and classifies each one as `damaged` or `normal`. It uses object tracking (ByteTrack) combined with a frame-consistency check to count each physical carton only once, even as it moves across multiple frames. A position-based deduplication safeguard also prevents double-counting when the tracker assigns a new ID to an object it briefly lost track of.

## Features

- Detection using a custom-trained YOLO model
- Unique object counting based on tracking ID consistency across consecutive frames
- Deduplication safeguard against tracker ID switches
- Works with webcam, video files, or a single image
- Optional annotated video output
- Optional automatic saving of frames when a new damaged carton is counted
- Headless mode for running without a display window

## Requirements

- Python 3.9 or newer
- ultralytics
- opencv-python

Install dependencies:

```
pip install ultralytics opencv-python
```

## Model

The detection model (`best.pt`) is trained separately (e.g. in Google Colab) using the Ultralytics YOLO training pipeline. The model must have two classes:

- `damaged`
- `normal`

### Exporting the model for faster CPU inference

Running the raw `.pt` model on CPU is significantly slower than using an exported format. Use `export_model.py` to convert the trained model to OpenVINO (recommended for Intel CPUs) or ONNX:

```
python export_model.py --model best.pt --format openvino --imgsz 640
```

This produces a folder named `best_openvino_model`. Use this folder as the `--model` argument when running detection.

Alternative, ONNX format:

```
python export_model.py --model best.pt --format onnx --imgsz 640
```

## Usage

Basic usage with a video file:

```
python damage_box_detection.py --model best_openvino_model --source video.mp4
```

Using a webcam (index 0):

```
python damage_box_detection.py --model best_openvino_model --source 0
```

Save annotated output to a video file:

```
python damage_box_detection.py --model best_openvino_model --source video.mp4 --output result.mp4
```

Run without a display window (headless):

```
python damage_box_detection.py --model best_openvino_model --source video.mp4 --output result.mp4 --no-display
```
The default confidence and box-size thresholds are tuned for isolated or few objects per frame. For scenes with multiple cartons close together, lower these thresholds:

```
python damage_box_detection.py --model best_openvino_model --source video.mp4 --conf 0.5 --min-box-ratio 0.005
```

If small or distant cartons are still missed, increase the inference resolution:

```
python damage_box_detection.py --model best_openvino_model --source video.mp4 --conf 0.5 --min-box-ratio 0.005 --imgsz 1280
```

Note that increasing `--imgsz` will reduce processing speed.

## Programmatic usage / backend integration

`run_inference.py` provides a headless function for calling the detector from other code, such as a web backend. It reuses the same detection, tracking, and counting logic as the main script, with the tuned default settings already applied.

Python usage:

```python
from run_inference import run_detection

result = run_detection("uploads/video.mp4")
print(result)
# {
#   "success": True,
#   "output_video": "uploads/video_detected.mp4",
#   "total_damaged": 3,
#   "total_ok": 12,
#   "frames_processed": 455
# }
```

Command-line usage (for non-Python backends, called via subprocess):

```
python run_inference.py --source uploads/video.mp4 --output results/output.mp4
```

This prints a single line of JSON to stdout with the same fields shown above.

## Project files

| File | Purpose |
|---|---|
| `damage_box_detection.py` | Main detection script, run manually from the command line |
| `export_model.py` | Exports a trained `.pt` model to OpenVINO or ONNX format |
| `run_inference.py` | Headless wrapper for programmatic/backend integration |
| `best.pt` | Trained YOLO model weights |
| `best_openvino_model/` | Exported OpenVINO version of the model, used for faster CPU inference |

## Notes

- The `damaged` and `normal` class names must match exactly (case-insensitive) what the model was trained with. A warning is printed at startup if either class is missing from the loaded model.
- Counting is based on tracking ID consistency, not on crossing a line. An object is counted once its track ID has been detected with the same label for a set number of consecutive frames.
- If a video source ends and `--loop` is not set, the script exits automatically after processing all frames.
