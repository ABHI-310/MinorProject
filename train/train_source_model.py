"""
train_source_model.py
Trains YOLOv8 on PKU-Market-PCB source dataset to achieve >=95% mAP.
Saves best weights to weights/best.pt and logs training results.
"""

import argparse
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


def train_source_yolov8(
    data_yaml: str = "data/pku_market_processed/data.yaml",
    model_name: str = "yolov8m.pt",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
    device: str = None,
    project: str = "runs/source_training",
    name: str = "yolov8m_pku",
):
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"=== Starting Source Domain YOLOv8 Training ===")
    print(f"Model Architecture: {model_name}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {imgsz}")

    # Fallback to unprocessed data.yaml if processed does not exist yet
    data_path = Path(data_yaml)
    if not data_path.exists():
        fallback_path = Path("data/pku_market/data.yaml")
        if fallback_path.exists():
            print(f"Notice: {data_path} not found. Using fallback {fallback_path}")
            data_yaml = str(fallback_path)
        else:
            raise FileNotFoundError(
                f"Data configuration '{data_yaml}' not found. Please run preprocessing first."
            )

    model = YOLO(model_name)

    # Launch Ultralytics training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        patience=patience,
        device=device,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        augment=True,
        mosaic=1.0,
        mixup=0.0,
        save_period=10,
        project=project,
        name=name,
        exist_ok=True,
    )

    # Destination directory for weights
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    run_weights = Path(project) / name / "weights"
    if (run_weights / "best.pt").exists():
        shutil.copy2(run_weights / "best.pt", weights_dir / "best.pt")
        print(f"✓ Copied best weights to: {weights_dir / 'best.pt'}")
    if (run_weights / "last.pt").exists():
        shutil.copy2(run_weights / "last.pt", weights_dir / "last.pt")

    print(f"✓ Source model training completed. Outputs saved to: {Path(project) / name}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on PKU Source Domain")
    parser.add_argument("--data", type=str, default="data/pku_market_processed/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Batch auto-tune for CPU
    batch = args.batch
    if not torch.cuda.is_available() and batch > 8:
        print("CUDA not detected. Automatically using batch size 4 for CPU safety.")
        batch = 4

    train_source_yolov8(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
    )


if __name__ == "__main__":
    main()

