"""
quantize_model.py
Exports trained YOLOv8 / adapted model to ONNX format with graph optimization
and INT8 dynamic quantization for efficient edge deployment on Raspberry Pi 4 (ARM Cortex-A72).
Output target size: <50 MB for edge execution @ 2-5 FPS.
"""

import argparse
import os
from pathlib import Path
import torch
from ultralytics import YOLO


def quantize_yolov8_to_onnx(
    model_path: str = "weights/best.pt",
    output_path: str = "weights/adapted_model.onnx",
    imgsz: int = 640,
    dynamic_quant: bool = True,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Check model path
    input_file = Path(model_path)
    if not input_file.exists():
        print(f"Notice: {model_path} not found. Utilizing yolov8n.pt as baseline for quantization demonstration...")
        model_path = "yolov8n.pt"

    print(f"=== Exporting & Quantizing Model for Raspberry Pi Edge ===")
    print(f"Input model: {model_path}")
    print(f"Target ONNX destination: {output_path}")

    model = YOLO(model_path)

    # 1. Export to base ONNX format
    exported_onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,  # FP32 base export for CPU/ARM execution
        opset=14,
        simplify=True,
    )
    print(f"✓ Base ONNX exported to: {exported_onnx_path}")

    # 2. Dynamic INT8 Quantization using ONNX Runtime
    if dynamic_quant:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_output = str(output_file)
            print("Applying ONNX Runtime dynamic INT8 quantization...")
            quantize_dynamic(
                model_input=exported_onnx_path,
                model_output=quant_output,
                weight_type=QuantType.QUInt8,
            )
            print(f"✓ Quantized INT8 model generated at: {quant_output}")
            target_path = quant_output
        except Exception as e:
            print(f"Notice during quantization ({e}). Preserving simplified ONNX export...")
            import shutil
            shutil.copy2(exported_onnx_path, output_file)
            target_path = str(output_file)
    else:
        import shutil
        shutil.copy2(exported_onnx_path, output_file)
        target_path = str(output_file)

    # Report file size
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"✓ Final Edge Model Footprint: {size_mb:.2f} MB (Constraint <= 50 MB: {'PASS' if size_mb <= 50 else 'EXCEEDED'})")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Export & quantize model for Raspberry Pi 4 edge deployment")
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--output", type=str, default="weights/adapted_model.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    quantize_yolov8_to_onnx(args.model, args.output, args.imgsz)


if __name__ == "__main__":
    main()

