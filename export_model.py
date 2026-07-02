"""
Small helper script to export a trained YOLO model (best.pt) to a faster
format for CPU inference. Run this ONCE after training, then point
detect_damaged_boxes.py to the exported model instead of the .pt file.

Usage:
    python export_model.py
    python export_model.py --model best.pt --format openvino
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to a faster CPU format")
    parser.add_argument("--model", type=str, default="best.pt",
                         help="Path to the trained .pt model (default: best.pt)")
    parser.add_argument("--format", type=str, default="openvino",
                         choices=["openvino", "onnx"],
                         help="Export format. 'openvino' is fastest on Intel CPUs. "
                              "'onnx' is a good universal alternative. Default: openvino")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Input size to bake into the exported model (default: 640). "
                              "Should match the --imgsz you plan to use for inference.")
    args = parser.parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Exporting to format='{args.format}' with imgsz={args.imgsz} ... "
          f"(this can take a minute or two)")
    exported_path = model.export(format=args.format, imgsz=args.imgsz)

    print(f"[DONE] Exported model ready at: {exported_path}")
    print(f"[NEXT STEP] Run detection with:")
    print(f"    python detect_damaged_boxes.py --model {exported_path} --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
