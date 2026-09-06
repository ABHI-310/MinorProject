"""
edge_inference.py
Edge inference pipeline for Raspberry Pi 4 (4GB RAM) with Pi Camera Module 2.
Executes quantized ONNX YOLOv8 model via ONNX Runtime, calculates equipment health decay,
and reports inspection telemetry to the central FastAPI backend.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np
import onnxruntime as rt
import requests

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
]


class EdgeInference:
    """Handles ONNX Runtime session, letterboxing, and telemetry reporting"""

    def __init__(self, model_path: str, backend_url: str, equipment_id: str = "PI_EDGE_001"):
        self.model_path = model_path
        self.backend_url = backend_url
        self.equipment_id = equipment_id

        print(f"Loading ONNX Model: {model_path}")
        options = rt.SessionOptions()
        options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4  # Utilize all 4 Cortex-A72 cores

        self.session = rt.InferenceSession(model_path, sess_options=options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"✓ ONNX Model Loaded Successfully.")
        print(f"  Input Tensor: {self.input_name} {self.input_shape}")
        print(f"  Outputs: {self.output_names}")

    def preprocess(self, image: np.ndarray, imgsz: int = 640):
        """Letterbox resize image preserving aspect ratio with padding"""
        h, w = image.shape[:2]
        scale = min(imgsz / h, imgsz / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to 640x640 canvas
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        x_offset = (imgsz - new_w) // 2
        y_offset = (imgsz - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Normalize [0, 255] -> [0.0, 1.0], HWC -> CHW, add batch dimension
        img_float = canvas.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
        return img_batch, scale, (x_offset, y_offset)

    def postprocess(self, outputs, scale, pad, conf_threshold=0.35):
        """
        Parses YOLOv8 ONNX raw output [1, 4 + num_classes, 8400]
        """
        raw = outputs[0]
        if raw.ndim == 3 and raw.shape[1] == 10:  # 4 bbox coords + 6 classes
            predictions = np.transpose(raw[0], (1, 0))  # [8400, 10]
        elif raw.ndim == 3 and raw.shape[2] == 10:
            predictions = raw[0]
        else:
            predictions = raw.reshape(-1, raw.shape[-1])

        x_offset, y_offset = pad
        detections = []

        for row in predictions:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= conf_threshold:
                # YOLO format: x_center, y_center, width, height
                cx, cy, bw, bh = row[0:4]
                # Adjust for padding and scale back to original image
                orig_x1 = (cx - bw / 2 - x_offset) / scale
                orig_y1 = (cy - bh / 2 - y_offset) / scale
                orig_x2 = (cx + bw / 2 - x_offset) / scale
                orig_y2 = (cy + bh / 2 - y_offset) / scale

                cls_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"defect_{class_id}"
                detections.append({
                    "class_id": class_id,
                    "class_name": cls_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(float(c), 1) for c in [orig_x1, orig_y1, orig_x2, orig_y2]],
                })

        return detections

    def run_inference(self, image: np.ndarray):
        """Full inference pipeline on frame"""
        input_batch, scale, pad = self.preprocess(image)
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_batch})
        detections = self.postprocess(raw_outputs, scale, pad)
        return detections

    def send_to_backend(self, detections: list, health_score: float, rul_cycles: int):
        """Sends inspection telemetry payload to FastAPI backend"""
        payload = {
            "equipment_id": self.equipment_id,
            "detections": detections,
            "health_score": round(health_score, 4),
            "rul_cycles": rul_cycles,
            "inference_source": "pi_edge",
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            url = f"{self.backend_url}/detect"
            res = requests.post(url, json=payload, timeout=3.0)
            if res.status_code == 200:
                return True
            else:
                print(f"Notice: Backend responded with status {res.status_code}")
                return False
        except Exception as e:
            # Telemetry transmission failure does not halt edge inspection
            return False


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Edge Defect Detection Loop")
    parser.add_argument("--model", type=str, default="weights/adapted_model.onnx")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000")
    parser.add_argument("--equipment-id", type=str, default="PI_EDGE_001")
    parser.add_argument("--max-frames", type=int, default=0, help="0 for infinite execution")
    args = parser.parse_args()

    # Check model path or create mock ONNX if needed
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Warning: {args.model} not found. Searching in local directory...")
        fallback_onnx = list(Path("weights").glob("*.onnx"))
        if fallback_onnx:
            model_path = fallback_onnx[0]
        else:
            print("No ONNX model found. Please run: python edge/quantize_model.py")
            return

    engine = EdgeInference(str(model_path), args.backend_url, args.equipment_id)

    # Initialize Camera
    picam2 = None
    cap = None
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        picam2.preview_configuration.main.size = (640, 480)
        picam2.preview_configuration.main.format = "RGB888"
        picam2.configure("preview")
        picam2.start()
        print("✓ Initialized native CSI camera (Pi Camera Module 2)")
    except Exception:
        print("Notice: Native libcamera/Picamera2 unavailable. Attempting OpenCV VideoCapture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Notice: No live webcam connected. Operating in synthetic inspection test loop.")
            cap = None

    frame_count = 0
    fps_start_time = time.time()

    print("\nStarting Edge Monitoring Loop (Press Ctrl+C to stop)...")
    try:
        while True:
            t0 = time.time()

            # Capture frame
            if picam2:
                frame = picam2.capture_array()
            elif cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # Synthetic simulated PCB frame for offline / demo testing
                frame = np.full((480, 640, 3), (30, 90, 30), dtype=np.uint8)
                cv2.line(frame, (100, 100), (500, 400), (0, 215, 255), 2)
                time.sleep(0.15)  # Simulate frame timing

            # Run inference
            detections = engine.run_inference(frame)
            latency_ms = (time.time() - t0) * 1000

            # Calculate Equipment Health & RUL
            defect_count = len(detections)
            health_score = max(0.20, 1.0 - (defect_count * 0.04))
            rul_cycles = int(health_score * 500)

            frame_count += 1

            # Dispatch telemetry to backend every 5 frames
            if frame_count % 5 == 0:
                engine.send_to_backend(detections, health_score, rul_cycles)

            # Benchmark print every 15 frames
            if frame_count % 15 == 0:
                elapsed = time.time() - fps_start_time
                fps = 15.0 / elapsed
                fps_start_time = time.time()
                print(
                    f"Frame {frame_count:04d} | Defects: {defect_count} | "
                    f"Health: {health_score*100:.1f}% | RUL: {rul_cycles}c | "
                    f"Latency: {latency_ms:.1f}ms | Speed: {fps:.1f} FPS"
                )

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames threshold ({args.max_frames}). Halting loop.")
                break

    except KeyboardInterrupt:
        print("\nEdge inference loop terminated by user.")
    finally:
        if picam2:
            picam2.close()
        if cap:
            cap.release()
        print("✓ Camera resources released.")


if __name__ == "__main__":
    main()

