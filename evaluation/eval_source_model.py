"""
eval_source_model.py
====================
Held-out Test Split Evaluation for PCB Source Domain YOLOv8n Model.

Loads trained model checkpoint (best.pt) and evaluates strictly on the
test split. Computes:
  - Overall mAP@0.5 and mAP@0.5:0.95
  - Per-class Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95
  - Confusion Matrix
  - Inference time per sample
  - Visual prediction samples overlaid on test images

Outputs:
  - reports/source_model_metrics.json
  - reports/per_class_metrics.csv
  - reports/confusion_matrix_source.png
  - reports/prediction_samples/*.jpg

Usage:
    python evaluation/eval_source_model.py
    python evaluation/eval_source_model.py --weights runs/source_model/weights/best.pt
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_configs_dir,
    get_project_root,
    get_reports_dir,
    get_runs_dir,
    save_json_report,
    setup_logging,
)

logger = setup_logging("eval_source")


def evaluate_test_set(
    weights_path: Path,
    data_yaml: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.60,
    img_size: int = 640,
    device: str = "cpu",
) -> dict:
    """
    Evaluate trained model weights strictly on the test set split.
    """
    if not weights_path.exists():
        logger.error(f"Checkpoint not found at: {weights_path}")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("EVALUATING MODEL ON HELD-OUT TEST SET")
    logger.info("=" * 65)
    logger.info(f"  Weights:    {weights_path}")
    logger.info(f"  Data YAML:  {data_yaml}")
    logger.info(f"  Confidence: {conf_threshold}")
    logger.info(f"  IoU Thresh: {iou_threshold}")
    logger.info(f"  Device:     {device}")
    logger.info("=" * 65)

    model = YOLO(str(weights_path))

    start_eval = time.time()
    val_results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=img_size,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        plots=True,
        save_json=True,
        verbose=True,
    )
    total_eval_time = time.time() - start_eval

    # Extract metrics safely from Ultralytics result object
    metrics_box = val_results.box

    mp = float(metrics_box.mp) if hasattr(metrics_box, "mp") else 0.0
    mr = float(metrics_box.mr) if hasattr(metrics_box, "mr") else 0.0
    map50 = float(metrics_box.map50) if hasattr(metrics_box, "map50") else 0.0
    map_all = float(metrics_box.map) if hasattr(metrics_box, "map") else 0.0
    f1_mean = (2 * mp * mr) / max(mp + mr, 1e-6)

    # Per-class metrics
    per_class = {}
    p_curve = metrics_box.p if hasattr(metrics_box, "p") else []
    r_curve = metrics_box.r if hasattr(metrics_box, "r") else []
    ap50_curve = metrics_box.ap50 if hasattr(metrics_box, "ap50") else []
    ap_curve = metrics_box.ap if hasattr(metrics_box, "ap") else []

    # Map indices to names
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        p = float(p_curve[cls_id]) if len(p_curve) > cls_id else 0.0
        r = float(r_curve[cls_id]) if len(r_curve) > cls_id else 0.0
        ap50 = float(ap50_curve[cls_id]) if len(ap50_curve) > cls_id else 0.0
        ap = float(ap_curve[cls_id]) if len(ap_curve) > cls_id else 0.0
        f1 = (2 * p * r) / max(p + r, 1e-6)

        per_class[cls_name] = {
            "class_id": cls_id,
            "precision": p,
            "recall": r,
            "f1_score": f1,
            "mAP50": ap50,
            "mAP50_95": ap,
        }

    speed_info = val_results.speed if hasattr(val_results, "speed") else {}

    results_data = {
        "evaluation_split": "test",
        "checkpoint": str(weights_path),
        "overall_metrics": {
            "mAP50": map50,
            "mAP50_95": map_all,
            "mean_precision": mp,
            "mean_recall": mr,
            "mean_f1": f1_mean,
        },
        "per_class_metrics": per_class,
        "inference_speed_ms": speed_info,
        "total_eval_time_sec": total_eval_time,
    }

    # Save reports
    reports_dir = get_reports_dir()
    save_json_report(results_data, reports_dir / "source_model_metrics.json")
    logger.info(f"Saved test metrics to {reports_dir / 'source_model_metrics.json'}")

    # Save per-class CSV
    csv_path = reports_dir / "per_class_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Class Name", "Precision", "Recall", "F1 Score", "mAP@0.5", "mAP@0.5:0.95"])
        for cls_name, m in per_class.items():
            writer.writerow([
                m["class_id"],
                cls_name,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['mAP50']:.4f}",
                f"{m['mAP50_95']:.4f}",
            ])
    logger.info(f"Saved per-class CSV to {csv_path}")

    # Copy confusion matrix plot if generated by YOLO
    val_save_dir = Path(val_results.save_dir) if hasattr(val_results, "save_dir") else None
    if val_save_dir and (val_save_dir / "confusion_matrix.png").exists():
        cm_target = reports_dir / "confusion_matrix_source.png"
        shutil.copy2(val_save_dir / "confusion_matrix.png", cm_target)
        logger.info(f"Copied confusion matrix to: {cm_target}")

    # Generate visual prediction samples on test images
    generate_prediction_samples(model, data_yaml, reports_dir / "prediction_samples", conf_threshold)

    # Print summary table
    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 75)
    print(f"Overall mAP@0.5:      {map50:.4f}")
    print(f"Overall mAP@0.5:0.95: {map_all:.4f}")
    print(f"Mean Precision:       {mp:.4f}")
    print(f"Mean Recall:          {mr:.4f}")
    print(f"Mean F1 Score:        {f1_mean:.4f}")
    print("-" * 75)
    print(f"{'Class Name':<18s} | {'Prec':<7s} | {'Recall':<7s} | {'F1':<7s} | {'mAP@50':<7s} | {'mAP@50-95':<7s}")
    print("-" * 75)
    for name, m in per_class.items():
        print(f"{name:<18s} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['f1_score']:<7.4f} | {m['mAP50']:<7.4f} | {m['mAP50_95']:<7.4f}")
    print("=" * 75)

    return results_data


def generate_prediction_samples(
    model: YOLO,
    data_yaml: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> None:
    """Run inference on test images and save prediction visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(data_yaml, "r", encoding="utf-8") as f:
        import yaml
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", ""))
    test_rel = data_cfg.get("test", "images/test")
    test_img_dir = (base_path / test_rel) if not Path(test_rel).is_absolute() else Path(test_rel)

    if not test_img_dir.exists():
        logger.warning(f"Test images directory not found at: {test_img_dir}")
        return

    test_images = sorted([f for f in test_img_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".png"}])
    logger.info(f"Generating prediction visualizations for {len(test_images)} test images...")

    for img_path in test_images:
        preds = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        for pred in preds:
            res_plotted = pred.plot(labels=True, conf=True)
            out_file = output_dir / f"pred_{img_path.name}"
            cv2.imwrite(str(out_file), res_plotted)

    logger.info(f"Saved visual prediction samples to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n on test split.")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file (default: best.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0)")
    args = parser.parse_args()

    # Find weights
    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        weights_path = get_project_root() / "runs" / "source_model" / "weights" / "best.pt"
        if not weights_path.exists():
            # Check last.pt fallback
            fallback = get_project_root() / "runs" / "source_model" / "weights" / "last.pt"
            if fallback.exists():
                logger.warning(f"best.pt not found, using last.pt at {fallback}")
                weights_path = fallback
            else:
                logger.error(f"Checkpoint not found at {weights_path}. Train the model first.")
                sys.exit(1)

    data_yaml = Path(args.data) if args.data else (get_configs_dir() / "data.yaml")
    evaluate_test_set(weights_path, data_yaml, conf_threshold=args.conf, iou_threshold=args.iou, device=args.device)


if __name__ == "__main__":
    main()
