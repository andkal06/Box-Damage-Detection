import argparse
import os
import sys
import time
from datetime import datetime

import cv2
if sys.platform.startswith("win"):
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] The 'ultralytics' library is not installed.")
    print("        Run: pip install ultralytics")
    sys.exit(1)
    
LABEL_RUSAK = "damaged"
LABEL_OK = "normal"

OUTPUT_DIR = "damage_detection_results"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_FRAME_HILANG = 90

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect + count damaged/OK boxes UNIQUELY (using tracking ID + "
                     "consistency of detection across several consecutive frames) from a webcam OR "
                     "a video/image file"
    )
    parser.add_argument("--model", type=str, default="best.pt",
                         help="Path to the YOLO model file (.pt). Default: best.pt")
    parser.add_argument("--source", type=str, default="0",
                         help="Input source: webcam index (0,1,2,...) OR path to a video file "
                              "(mp4/avi/mov/mkv) OR path to an image file (jpg/png). Default: 0")
    parser.add_argument("--output", type=str, default=None,
                         help="Output video file path to save the annotated result (example: result.mp4)")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="Minimum confidence threshold (default: 0.8)")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Image resolution size used during inference (default: 640). "
                              "Increase (e.g. 1280) if many small/far boxes are not "
                              "detected, especially in images with many objects at once.")
    parser.add_argument("--min-box-ratio", type=float, default=0.015,
                         help="Minimum box area relative to frame area, 0-1 (default: 0.015)")
    parser.add_argument("--width", type=int, default=1280,
                         help="Camera resolution width, only used in webcam mode (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                         help="Camera resolution height, only used in webcam mode (default: 720)")
    parser.add_argument("--consistent-frames", type=int, default=5,
                         help="Number of CONSECUTIVE frames a track ID must be consistently "
                              "detected (with the same label) before it is counted as "
                              "a unique object. Increase if false positives still "
                              "occur momentarily, decrease if objects pass too "
                              "quickly through the camera so they don't reach this "
                              "threshold in time. Default: 5")
    parser.add_argument("--dedup-radius-ratio", type=float, default=0.12,
                         help="ANTI DOUBLE-COUNT SAFEGUARD: when a new track ID is about to be "
                              "counted, its position is checked against objects (with the same "
                              "label) that were JUST counted. If the distance is closer than "
                              "this radius (as a ratio of the frame diagonal, 0-1), it is "
                              "considered an ID-switch of the same physical object -> NOT "
                              "counted again. Default: 0.12")
    parser.add_argument("--dedup-window-frames", type=int, default=45,
                         help="How many frames back an object that was just counted is "
                              "still considered 'just counted' for the dedup check "
                              "above. Default: 45")
    parser.add_argument("--disable-dedup", action="store_true",
                         help="Disable the dedup safeguard above (revert to pure per-ID counting)")
    parser.add_argument("--save-defect", action="store_true",
                         help="Automatically save an image every time a NEW damaged box is counted")
    parser.add_argument("--no-display", action="store_true",
                         help="Run without a display window (headless/server mode)")
    parser.add_argument("--loop", action="store_true",
                         help="If the source is a video file, restart from the beginning after the video ends")
    parser.add_argument("--debug", action="store_true",
                         help="Print all detections + track ID + streak to the terminal")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="Ultralytics tracker config to use (default: bytetrack.yaml)")
    return parser.parse_args()


def load_model(model_path: str) -> YOLO:
    # OpenVINO exports are a FOLDER (e.g. best_openvino_model), not a single
    # file, so we accept either an existing file OR an existing directory here.
    if not os.path.isfile(model_path) and not os.path.isdir(model_path):
        print(f"[ERROR] Model file/folder not found: {model_path}")
        sys.exit(1)
    try:
        model = YOLO(model_path)
        print(f"[INFO] Model successfully loaded: {model_path}")
        print(f"[INFO] Model class list: {model.names}")

        nama_class_lower = [str(v).lower() for v in model.names.values()]
        if LABEL_RUSAK not in nama_class_lower:
            print(f"[WARNING] Class '{LABEL_RUSAK}' not found in the model! "
                  f"Existing classes: {list(model.names.values())}")
        if LABEL_OK not in nama_class_lower:
            print(f"[WARNING] Class '{LABEL_OK}' not found in the model! "
                  f"Existing classes: {list(model.names.values())}")

        print_device_info(model_path)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)


def print_device_info(model_path: str):
    """
    Print which device inference will run on (CPU/GPU) and, if running on
    CPU with a plain .pt model, print a suggestion to export the model to a
    faster format for noticeably higher FPS. Informational only — does not
    change any detection/tracking/counting logic.
    """
    try:
        import torch
        cuda_ada = torch.cuda.is_available()
    except Exception:
        cuda_ada = False

    ext = os.path.splitext(model_path)[1].lower()

    if cuda_ada:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown GPU"
        print(f"[INFO] Inference device: GPU (CUDA) - {gpu_name}")
        return

    print("[INFO] Inference device: CPU (no CUDA GPU detected)")

    if ext == ".pt":
        print("[TIP] Running on CPU with a raw .pt model is the slowest combination "
              "and is usually why FPS looks low. To speed this up without changing "
              "any detection logic, export the model once to a faster CPU format:")
        print("        from ultralytics import YOLO")
        print(f"        YOLO('{model_path}').export(format='openvino')   "
              "# best if you have an Intel CPU")
        print(f"        YOLO('{model_path}').export(format='onnx')       "
              "# good universal alternative")
        print("      Then run this script again with the exported model, e.g.:")
        base = os.path.splitext(model_path)[0]
        print(f"        --model {base}_openvino_model   (or)   --model {base}.onnx")
        print("      This alone commonly gives a 2-3x FPS improvement on CPU. "
              "Lowering --imgsz (e.g. 480 or 416) also helps, at the cost of "
              "missing very small/far objects.")
    elif ext == ".onnx":
        print("[INFO] Using an ONNX model on CPU - good choice, this is faster than a raw .pt file.")
    else:
        print("[INFO] Using an exported/optimized model format on CPU.")


def is_webcam_source(source: str) -> bool:
    return source.isdigit()


def is_image_source(source: str) -> bool:
    return os.path.splitext(source)[1].lower() in IMAGE_EXTS


def open_source(source: str, width: int, height: int):
    if is_webcam_source(source):
        cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera index {cam_index}.")
            sys.exit(1)
        print(f"[INFO] Webcam index {cam_index} successfully opened ({width}x{height}).")
        return cap, True, False, 30.0

    if not os.path.isfile(source):
        print(f"[ERROR] Source file not found: {source}")
        sys.exit(1)

    if is_image_source(source):
        print(f"[INFO] Source is a single image: {source}")
        return None, False, True, 0.0

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {source}")
        sys.exit(1)

    fps_asal = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video file successfully opened: {source} "
          f"(original fps: {fps_asal:.1f}, total frames: {total_frame})")
    return cap, False, False, fps_asal


def draw_status_bar(frame, fps, rusak_unik, ok_unik, rusak_frame, ok_frame):
    h, w = frame.shape[:2]

    font_scale = max(0.35, min(0.6, w / 1000.0 * 0.6))
    tebal = 1 if font_scale < 0.45 else 2
    tinggi_bar = int(70 * (font_scale / 0.6))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, tinggi_bar), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    baris1 = f"FPS:{fps:.1f} Frame = Damaged : {rusak_frame} OK : {ok_frame}"
    baris2 = f"Overall total = Damaged : {rusak_unik} OK : {ok_unik}"

    y1 = int(tinggi_bar * 0.4)
    y2 = int(tinggi_bar * 0.85)

    cv2.putText(frame, baris1, (8, y1), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 255, 255), tebal, cv2.LINE_AA)
    cv2.putText(frame, baris2, (8, y2), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 200, 255), tebal, cv2.LINE_AA)
    return frame


def get_screen_size():
    """Detect screen resolution (Windows only). Falls back to 1280x800 on failure."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            user32 = ctypes.windll.user32
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return 1280, 800


def letterbox_ke_target(frame, target_w, target_h):
    """
    Scale the frame so it fits within target_w x target_h without distortion,
    then place it centered on a black canvas of size EXACTLY target_w x target_h.
    The result is that the window will always be exactly target_w x target_h,
    with no leftover empty area outside the window.
    """
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    skala = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * skala)), max(1, int(h * skala))
    interp = cv2.INTER_AREA if skala < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    kanvas = (30, 30, 30)  # dark gray, not pure black, so it isn't too high-contrast
    hasil = cv2.copyMakeBorder(
        resized,
        top=(target_h - new_h) // 2,
        bottom=target_h - new_h - (target_h - new_h) // 2,
        left=(target_w - new_w) // 2,
        right=target_w - new_w - (target_w - new_w) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=kanvas,
    )
    return hasil


def proses_frame_dengan_tracking(model, frame, args, track_state, recently_counted, frame_counter):
    """
    Run tracking on a single frame, draw the boxes, and count UNIQUE boxes
    based on ID CONSISTENCY: as soon as a track ID is detected with the
    same label for `args.consistent_frames` CONSECUTIVE frames,
    it is immediately counted once at that moment (no need to cross a line).

    If a track ID is temporarily lost (occluded) for a frame or its label
    changes, its streak is reset from the start -> this is what makes it
    resistant to momentary noise/false positives.

    ANTI DOUBLE-COUNT SAFEGUARD: the tracker (ByteTrack) sometimes loses
    an ID and then assigns a NEW ID to the same physical object once it
    reappears (e.g. it was briefly occluded by another box). So that the
    same object is not counted twice, every time an ID is about to be
    counted, its position is first checked against the `recently_counted`
    list (objects with the same label that were just counted within the
    last few frames). If the distance is close -> it is considered an
    ID-switch and NOT added to the total.

    Returns: annotated_frame, rusak_frame_ini, ok_frame_ini, rusak_baru_unik, ok_baru_unik
    """
    h, w = frame.shape[:2]
    luas_frame = w * h

    try:
        results = model.track(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            persist=True,
            tracker=args.tracker,
            verbose=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to run tracking: {e}")
        return frame, 0, 0, 0, 0

    result = results[0]
    annotated_frame = frame.copy()

    rusak_frame_ini = 0
    ok_frame_ini = 0
    rusak_baru_unik = 0
    ok_baru_unik = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            try:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = str(model.names.get(cls_id, str(cls_id)))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if box.id is not None else None
            except Exception as e:
                if args.debug:
                    print(f"[DEBUG] Failed to parse box: {e}")
                continue

            luas_box = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            rasio_luas = luas_box / luas_frame if luas_frame > 0 else 0

            if rasio_luas < args.min_box_ratio:
                continue

            label_lower = label.lower()
            is_rusak = (label_lower == LABEL_RUSAK)
            is_ok = (label_lower == LABEL_OK)

            if is_rusak:
                rusak_frame_ini += 1
            elif is_ok:
                ok_frame_ini += 1

            # -------- unique counting logic based on ID consistency --------
            streak = 0
            sudah_dihitung = False
            is_duplikat = False
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if track_id is not None:
                state = track_state.get(track_id)

                if state is None:
                    state = {
                        "label": label_lower,
                        "streak": 1,
                        "counted": False,
                        "is_duplikat": False,
                        "last_seen": frame_counter,
                    }
                    track_state[track_id] = state
                else:
                    lanjut_langsung = (frame_counter - state["last_seen"]) == 1
                    if lanjut_langsung and state["label"] == label_lower:
                        # still on the next frame & same label -> streak keeps going
                        state["streak"] += 1
                    else:
                        # there was a gap (temporarily lost for a few frames) OR the label changed
                        # -> start the streak over from the beginning
                        state["label"] = label_lower
                        state["streak"] = 1
                    state["last_seen"] = frame_counter

                if not state["counted"] and state["streak"] >= args.consistent_frames:
                    state["counted"] = True

                    # ---- SAFEGUARD: check position dedup before adding to the total ----
                    duplikat_dari = None
                    if not args.disable_dedup:
                        diagonal = (w ** 2 + h ** 2) ** 0.5
                        radius_dedup = args.dedup_radius_ratio * diagonal
                        for entry in recently_counted:
                            if entry["label"] != state["label"]:
                                continue
                            if (frame_counter - entry["frame"]) > args.dedup_window_frames:
                                continue
                            jarak = ((cx - entry["cx"]) ** 2 + (cy - entry["cy"]) ** 2) ** 0.5
                            if jarak <= radius_dedup:
                                duplikat_dari = entry
                                break

                    if duplikat_dari is not None:
                        # considered an ID-switch of the same physical object -> DO NOT add to the total
                        state["is_duplikat"] = True
                        print(f"[DEDUP] Track #{track_id} ({state['label']}) is at a position close to "
                              f"an object that was just counted (old track from frame "
                              f"{duplikat_dari['frame']}) -> treated as an ID-switch, NOT "
                              f"added to the total.")
                    else:
                        if state["label"] == LABEL_RUSAK:
                            rusak_baru_unik += 1
                        elif state["label"] == LABEL_OK:
                            ok_baru_unik += 1
                        recently_counted.append({
                            "cx": cx, "cy": cy,
                            "label": state["label"],
                            "frame": frame_counter,
                        })

                streak = state["streak"]
                sudah_dihitung = state["counted"]
                is_duplikat = state.get("is_duplikat", False)

            if args.debug:
                print(f"[DEBUG] id={track_id} label={label} conf={conf_score:.2f} "
                      f"area_ratio={rasio_luas:.4f} streak={streak}/{args.consistent_frames} "
                      f"counted={sudah_dihitung} duplicate={is_duplikat}")

            # -------- draw box + text --------
            warna = (0, 0, 255) if is_rusak else (0, 255, 0)
            tebal_box = 3 if sudah_dihitung else 2
            cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), warna, tebal_box)

            id_text = f"#{track_id} " if track_id is not None else ""
            if track_id is None:
                progres_text = ""
            elif is_duplikat:
                progres_text = " [ALREADY COUNTED]"
            elif sudah_dihitung:
                progres_text = " [COUNTED]"
            else:
                progres_text = f" [{streak}/{args.consistent_frames}]"

            warna_teks = (0, 165, 255) if is_duplikat else warna
            teks_box = f"{id_text}{label} {conf_score:.2f}{progres_text}"
            cv2.putText(annotated_frame, teks_box, (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_teks, 2, cv2.LINE_AA)

    # clean up old tracks that are no longer visible (save memory)
    id_hilang = [tid for tid, st in track_state.items()
                 if frame_counter - st["last_seen"] > MAX_FRAME_HILANG]
    for tid in id_hilang:
        del track_state[tid]

    # clean up recently_counted entries that are past the dedup window
    recently_counted[:] = [
        entry for entry in recently_counted
        if frame_counter - entry["frame"] <= args.dedup_window_frames
    ]

    return annotated_frame, rusak_frame_ini, ok_frame_ini, rusak_baru_unik, ok_baru_unik


def main():
    print("========== SCRIPT VERSION: v6-id-consistent-dedup-v1 ==========")
    args = parse_args()

    if args.save_defect and not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"[INFO] Capture results folder created: {OUTPUT_DIR}")

    model = load_model(args.model)
    cap, is_webcam, is_image, fps_asal = open_source(args.source, args.width, args.height)

    total_rusak_unik = 0
    total_ok_unik = 0

    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            print(f"[ERROR] Failed to read image: {args.source}")
            sys.exit(1)

        results = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        result = results[0]
        annotated_frame = frame.copy()
        rusak_frame_ini = 0
        ok_frame_ini = 0

        h, w = frame.shape[:2]
        luas_frame = w * h

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                label = str(model.names.get(cls_id, str(cls_id)))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                luas_box = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if luas_frame > 0 and (luas_box / luas_frame) < args.min_box_ratio:
                    continue
                label_lower = label.lower()
                is_rusak = (label_lower == LABEL_RUSAK)
                warna = (0, 0, 255) if is_rusak else (0, 255, 0)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), warna, 2)
                cv2.putText(annotated_frame, f"{label} {conf_score:.2f}",
                            (int(x1), max(0, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, warna, 2, cv2.LINE_AA)
                if is_rusak:
                    rusak_frame_ini += 1
                elif label_lower == LABEL_OK:
                    ok_frame_ini += 1

        total_rusak_unik = rusak_frame_ini
        total_ok_unik = ok_frame_ini
        annotated_frame = draw_status_bar(annotated_frame, 0.0, total_rusak_unik,
                                           total_ok_unik, rusak_frame_ini, ok_frame_ini)

        if args.output:
            cv2.imwrite(args.output, annotated_frame)
            print(f"[INFO] Detection result saved: {args.output}")

        if not args.no_display:
            win_name = "QC Damaged Box Detection - Image"
            screen_w, screen_h = get_screen_size()
            TARGET_W = int(screen_w * 0.9)
            TARGET_H = int(screen_h * 0.85)
            tampil_frame = letterbox_ke_target(annotated_frame, TARGET_W, TARGET_H)
            cv2.imshow(win_name, tampil_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"[SUMMARY] Total damaged detections: {total_rusak_unik} | Total OK detections: {total_ok_unik}")
        return

    video_writer = None
    prev_time = time.time()
    fps = 0.0
    frame_counter = 0
    track_state = {}
    recently_counted = []  # dedup safeguard: [{"cx","cy","label","frame"}, ...]

    print("[INFO] Starting detection + tracking. Press 'q' to quit, 's' for a manual screenshot.")
    print(f"[INFO] Source: {'webcam index ' + args.source if is_webcam else args.source}")
    print(f"[INFO] ID consistency threshold: {args.consistent_frames} consecutive frames "
          f"before an object is counted as unique")
    if args.disable_dedup:
        print("[INFO] Position dedup safeguard: DISABLED (--disable-dedup)")
    else:
        print(f"[INFO] Position dedup safeguard: radius={args.dedup_radius_ratio} (ratio of frame diagonal), "
              f"window={args.dedup_window_frames} frames")
    print(f"[INFO] Confidence threshold: {args.conf} | Min box ratio: {args.min_box_ratio} | imgsz: {args.imgsz}")

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                if is_webcam:
                    print("[WARNING] Failed to read frame from camera. Retrying...")
                    time.sleep(0.5)
                    continue
                else:
                    if args.loop:
                        print("[INFO] Video ended, restarting from the beginning (--loop enabled)...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        track_state.clear()  # reset tracking so IDs don't carry over from the previous loop
                        recently_counted.clear()
                        continue
                    print("[INFO] Video finished processing (all frames have been read).")
                    break

            frame_counter += 1

            annotated_frame, rusak_frame_ini, ok_frame_ini, rusak_baru, ok_baru = \
                proses_frame_dengan_tracking(model, frame, args, track_state, recently_counted, frame_counter)

            total_rusak_unik += rusak_baru
            total_ok_unik += ok_baru

            if rusak_baru > 0 and args.save_defect:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filepath = os.path.join(OUTPUT_DIR, f"damaged_{timestamp}.jpg")
                cv2.imwrite(filepath, annotated_frame)
                print(f"[ALERT] New damaged box counted! Saved: {filepath}")

            curr_time = time.time()
            elapsed = curr_time - prev_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = curr_time

            annotated_frame = draw_status_bar(
                annotated_frame, fps, total_rusak_unik, total_ok_unik,
                rusak_frame_ini, ok_frame_ini
            )

            if args.output and video_writer is None:
                out_h, out_w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                target_fps = fps_asal if fps_asal > 0 else 25.0
                video_writer = cv2.VideoWriter(args.output, fourcc, target_fps, (out_w, out_h))
                print(f"[INFO] Saving result to video: {args.output} "
                      f"({out_w}x{out_h} @ {target_fps:.1f}fps)")

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if not args.no_display:
                win_name = "QC Damaged Box Detection"
                screen_w, screen_h = get_screen_size()
                TARGET_W = int(screen_w * 0.9)
                TARGET_H = int(screen_h * 0.85)
                tampil_frame = letterbox_ke_target(annotated_frame, TARGET_W, TARGET_H)
                cv2.imshow(win_name, tampil_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] Stopping program ('q' pressed)...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    manual_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(manual_path, annotated_frame)
                    print(f"[INFO] Manual screenshot saved: {manual_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Program stopped manually (Ctrl+C).")

    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("[INFO] Source released & windows closed. Done.")
        print(f"[SUMMARY] Total unique count (ID-consistent) -> Damaged: {total_rusak_unik} | OK: {total_ok_unik}")


if __name__ == "__main__":
    main()