"""Run a trained YOLOX ONNX detector on a live webcam feed."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .cli import config as load_config
from .onnx_inference import postprocess, preprocess


def main():
    parser = argparse.ArgumentParser(description="Live YOLOX webcam detection")
    parser.add_argument("--model", required=True, type=Path, help="Exported YOLOX .onnx model")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror the webcam preview")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"Model file not found: {args.model}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("Install ONNX Runtime first: pip install onnxruntime") from exc

    cfg = load_config(args.config)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model.resolve()), providers=providers)
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    input_height = int(shape[2]) if isinstance(shape[2], int) else int(cfg["image_size"])
    input_width = int(shape[3]) if isinstance(shape[3], int) else int(cfg["image_size"])

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try another --camera index.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("Live detection started. Press Q or Esc in the video window to quit.")
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor, ratio = preprocess(Image.fromarray(rgb), (input_height, input_width))
            output = session.run(None, {input_meta.name: tensor})[0]
            detections = postprocess(output, ratio, frame.shape, args.threshold, num_classes=len(cfg["classes"]))

            for class_id, score, box in detections:
                x1, y1, x2, y2 = (int(round(value)) for value in box)
                color = (40, 220, 40)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cfg['classes'][class_id]} {score:.2f}"
                text_y = max(22, y1 - 8)
                cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-9)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            cv2.putText(frame, f"FPS: {smoothed_fps:.1f} | Q: sair", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("YOLOX - webcam em tempo real", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
