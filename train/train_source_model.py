"""
train_source_model.py
=====================
Source-Domain YOLOv8n Model Training for PCB Defect Detection.

Loads configuration from configs/train_config.yaml and data.yaml.
Trains YOLOv8n on the processed PKU-Market-PCB dataset with:
  - Pretrained COCO weights (yolov8n.pt)
  - Early stopping (patience)
  - On-the-fly Ultralytics data augmentation
  - Checkpoint saving (best.pt, last.pt)
  - Hardware auto-detection (CPU or GPU)

Outputs:
  - runs/source_model/weights/best.pt
  - runs/source_model/weights/last.pt
  - runs/source_model/results.csv
  - reports/training_summary.json

Usage:
    python train/train_source_model.py
    python train/train_source_model.py --epochs 30 --batch-size 4 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    get_configs_dir,
    get_project_root,
    get_reports_dir,
    get_runs_dir,
    load_train_config,
    save_json_report,
    setup_logging,
)

logger = setup_logging("train_source")


def resolve_device(configured_device: str) -> str:
    """Determine compute device (cuda if available, else cpu)."""
    if configured_device:
        device = configured_device.lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return device

    if torch.cuda.is_available():
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        return "0"
    else:
        logger.info("No CUDA GPU detected. Using CPU for training.")
        return "cpu"


def train(
    data_yaml: Path,
    config_override: dict = None,
) -> Path:
    """
    Train YOLOv8n source domain model.

    Returns:
        Path to best.pt checkpoint.
    """
    cfg = load_train_config()
    if config_override:
        cfg.update({k: v for k, v in config_override.items() if v is not None})

    device = resolve_device(cfg.get("device", ""))
    epochs = int(cfg.get("epochs", 50))
    batch_size = int(cfg.get("batch_size", 4))
    img_size = int(cfg.get("image_size", 640))
    patience = int(cfg.get("patience", 15))
    seed = int(cfg.get("seed", 42))
    workers = int(cfg.get("workers", 4))
    model_name = cfg.get("model", "yolov8n.pt")

    aug = cfg.get("augmentation", {})
    project_dir = get_project_root() / cfg.get("project", "runs")
    run_name = cfg.get("name", "source_model")

    logger.info("=" * 65)
    logger.info("STARTING PCB YOLOV8 SOURCE MODEL TRAINING")
    logger.info("=" * 65)
    logger.info(f"  Model:        {model_name}")
    logger.info(f"  Data Config:  {data_yaml}")
    logger.info(f"  Device:       {device.upper()}")
    logger.info(f"  Epochs:       {epochs} (patience: {patience})")
    logger.info(f"  Batch Size:   {batch_size}")
    logger.info(f"  Image Size:   {img_size}x{img_size}")
    logger.info(f"  Output Dir:   {project_dir / run_name}")
    logger.info("=" * 65)

    start_time = time.time()

    # Load YOLO model
    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        device=device,
        workers=workers,
        seed=seed,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        lrf=cfg.get("lrf", 0.01),
        momentum=cfg.get("momentum", 0.937),
        weight_decay=cfg.get("weight_decay", 0.0005),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        degrees=aug.get("degrees", 10.0),
        scale=aug.get("scale", 0.3),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.5),
        hsv_v=aug.get("hsv_v", 0.3),
        mosaic=aug.get("mosaic", 1.0),
        close_mosaic=aug.get("close_mosaic", 10),
        verbose=True,
    )

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

    weights_dir = project_dir / run_name / "weights"
    best_weights = weights_dir / "best.pt"
    last_weights = weights_dir / "last.pt"

    logger.info("=" * 65)
    logger.info(f"[OK] Training completed in {time_str}")
    logger.info(f"  Best checkpoint: {best_weights} ({'FOUND' if best_weights.exists() else 'NOT FOUND'})")
    logger.info(f"  Last checkpoint: {last_weights} ({'FOUND' if last_weights.exists() else 'NOT FOUND'})")
    logger.info("=" * 65)

    # Record summary in reports
    summary = {
        "status": "COMPLETED",
        "model_name": model_name,
        "device": device,
        "epochs_requested": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "image_size": img_size,
        "elapsed_seconds": elapsed_time,
        "elapsed_formatted": time_str,
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "results_dir": str(project_dir / run_name),
    }

    reports_dir = get_reports_dir()
    save_json_report(summary, reports_dir / "training_summary.json")
    logger.info(f"Training summary saved to {reports_dir / 'training_summary.json'}")

    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on PKU-Market-PCB dataset.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', '0', etc.)")
    parser.add_argument("--model", type=str, default=None, help="YOLO model checkpoint/yaml")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    args = parser.parse_args()

    data_yaml = Path(args.data) if args.data else (get_configs_dir() / "data.yaml")
    if not data_yaml.exists():
        logger.error(f"Data config not found: {data_yaml}. Run scripts/03_train_test_split.py first.")
        sys.exit(1)

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "model": args.model,
    }

    train(data_yaml, overrides)


if __name__ == "__main__":
    main()
