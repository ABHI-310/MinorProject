"""
02_dataset_stats.py
===================
Dataset Analysis and Visualization for PKU-Market-PCB.

Generates comprehensive dataset statistics:
  - Class distribution (instance count & image count)
  - Bounding box sizes, areas, and aspect ratios
  - Number of defects per image distribution
  - Visualization plots (bar charts, distributions)
  - Sample annotated images saved to reports/sample_images/

Outputs:
  - reports/dataset_analysis.json
  - reports/class_distribution.png
  - reports/bbox_analysis.png
  - reports/sample_images/*.jpg

Usage:
    python scripts/02_dataset_stats.py
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_raw_dataset_dir,
    get_reports_dir,
    parse_yolo_label,
    save_json_report,
    setup_logging,
    validate_annotation,
)

logger = setup_logging("02_stats")

# Distinct color palette for 6 defect classes (BGR for OpenCV, RGB for Matplotlib)
CLASS_COLORS_RGB = [
    (220, 53, 69),    # 0: missing_hole - Red
    (253, 126, 20),   # 1: mouse_bite - Orange
    (255, 193, 7),    # 2: open - Yellow/Amber
    (40, 167, 69),    # 3: short - Green
    (23, 162, 184),   # 4: spur - Cyan/Teal
    (111, 66, 193),   # 5: spurious_copper - Purple
]
CLASS_COLORS_BGR = [(c[2], c[1], c[0]) for c in CLASS_COLORS_RGB]


import importlib
pku_download_module = importlib.import_module("scripts.01_download_pku_dataset")
find_all_images = pku_download_module.find_all_images
find_label_for_image = pku_download_module.find_label_for_image


def analyze_dataset(dataset_dir: Path) -> dict:
    """Analyze images and annotations in the dataset directory."""
    image_files = find_all_images(dataset_dir)

    if not image_files:
        logger.error(f"No images found in {dataset_dir}")
        sys.exit(1)

    logger.info(f"Analyzing {len(image_files)} images and annotations...")

    class_counts = {name: 0 for name in CLASS_NAMES}
    image_has_class = {name: 0 for name in CLASS_NAMES}
    annotations_per_image = []
    
    widths = []
    heights = []
    areas = []
    aspect_ratios = []

    per_class_box_areas = {name: [] for name in CLASS_NAMES}
    image_resolutions = []

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            image_resolutions.append((w, h))
        else:
            w, h = 640, 640

        lbl_path = find_label_for_image(img_path, dataset_dir)
        annotations = parse_yolo_label(lbl_path) if (lbl_path and lbl_path.exists()) else []

        valid_annotations = [ann for ann in annotations if not validate_annotation(ann)]
        annotations_per_image.append(len(valid_annotations))

        classes_in_this_img = set()

        for ann in valid_annotations:
            cls_id = ann["class_id"]
            cls_name = CLASS_NAMES[cls_id]
            class_counts[cls_name] += 1
            classes_in_this_img.add(cls_name)

            bw = ann["width"]
            bh = ann["height"]
            area = bw * bh
            ar = bw / max(bh, 1e-6)

            widths.append(bw)
            heights.append(bh)
            areas.append(area)
            aspect_ratios.append(ar)
            per_class_box_areas[cls_name].append(area)

        for cls_name in classes_in_this_img:
            image_has_class[cls_name] += 1

    total_annotations = sum(class_counts.values())

    stats = {
        "dataset_summary": {
            "total_images": len(image_files),
            "total_annotations": total_annotations,
            "avg_annotations_per_image": float(np.mean(annotations_per_image)) if annotations_per_image else 0.0,
            "min_annotations_per_image": int(np.min(annotations_per_image)) if annotations_per_image else 0,
            "max_annotations_per_image": int(np.max(annotations_per_image)) if annotations_per_image else 0,
            "common_resolution": f"{image_resolutions[0][0]}x{image_resolutions[0][1]}" if image_resolutions else "unknown",
        },
        "class_distribution": {
            cls_name: {
                "instances": class_counts[cls_name],
                "percentage_of_total": float(class_counts[cls_name] / max(total_annotations, 1) * 100),
                "images_containing": image_has_class[cls_name],
                "image_presence_pct": float(image_has_class[cls_name] / max(len(image_files), 1) * 100),
                "avg_normalized_area": float(np.mean(per_class_box_areas[cls_name])) if per_class_box_areas[cls_name] else 0.0,
            }
            for cls_name in CLASS_NAMES
        },
        "bounding_box_metrics": {
            "normalized_width": {
                "mean": float(np.mean(widths)) if widths else 0.0,
                "std": float(np.std(widths)) if widths else 0.0,
                "min": float(np.min(widths)) if widths else 0.0,
                "max": float(np.max(widths)) if widths else 0.0,
            },
            "normalized_height": {
                "mean": float(np.mean(heights)) if heights else 0.0,
                "std": float(np.std(heights)) if heights else 0.0,
                "min": float(np.min(heights)) if heights else 0.0,
                "max": float(np.max(heights)) if heights else 0.0,
            },
            "normalized_area": {
                "mean": float(np.mean(areas)) if areas else 0.0,
                "min": float(np.min(areas)) if areas else 0.0,
                "max": float(np.max(areas)) if areas else 0.0,
            },
            "aspect_ratio": {
                "mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
                "min": float(np.min(aspect_ratios)) if aspect_ratios else 0.0,
                "max": float(np.max(aspect_ratios)) if aspect_ratios else 0.0,
            },
        },
    }

    return stats, {
        "widths": widths,
        "heights": heights,
        "areas": areas,
        "aspect_ratios": aspect_ratios,
        "annotations_per_image": annotations_per_image,
        "class_counts": class_counts,
        "image_has_class": image_has_class,
    }


def generate_plots(raw_data: dict, output_dir: Path) -> None:
    """Generate and save visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class Distribution Bar Chart
    plt.figure(figsize=(10, 5))
    names = CLASS_NAMES
    instances = [raw_data["class_counts"][name] for name in names]
    images_pres = [raw_data["image_has_class"][name] for name in names]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width / 2, instances, width, label="Instances", color="#2b5c8f")
    plt.bar(x + width / 2, images_pres, width, label="Images with Class", color="#e27c3e")

    plt.title("Defect Class Distribution (PKU-Market-PCB)", fontsize=13, fontweight="bold")
    plt.xlabel("Defect Class", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    dist_plot_path = output_dir / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved class distribution plot to: {dist_plot_path}")

    # 2. Bounding Box Geometry Analysis (4-panel subplot)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Width distribution
    axes[0, 0].hist(raw_data["widths"], bins=20, color="#17a2b8", edgecolor="black", alpha=0.8)
    axes[0, 0].set_title("Normalized Box Width")
    axes[0, 0].set_xlabel("Width (relative)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(linestyle="--", alpha=0.5)

    # Height distribution
    axes[0, 1].hist(raw_data["heights"], bins=20, color="#28a745", edgecolor="black", alpha=0.8)
    axes[0, 1].set_title("Normalized Box Height")
    axes[0, 1].set_xlabel("Height (relative)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(linestyle="--", alpha=0.5)

    # Aspect ratio distribution
    axes[1, 0].hist(raw_data["aspect_ratios"], bins=20, color="#ffc107", edgecolor="black", alpha=0.8)
    axes[1, 0].set_title("Aspect Ratio (Width / Height)")
    axes[1, 0].set_xlabel("Ratio")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(linestyle="--", alpha=0.5)

    # Defects per image
    axes[1, 1].hist(raw_data["annotations_per_image"], bins=range(0, max(raw_data["annotations_per_image"]) + 2),
                    color="#dc3545", edgecolor="black", alpha=0.8, align="left")
    axes[1, 1].set_title("Defects per Image Distribution")
    axes[1, 1].set_xlabel("Number of Defects")
    axes[1, 1].set_ylabel("Number of Images")
    axes[1, 1].grid(linestyle="--", alpha=0.5)

    plt.suptitle("PKU-Market-PCB Defect Geometry Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    bbox_plot_path = output_dir / "bbox_analysis.png"
    plt.savefig(bbox_plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved bbox analysis plot to: {bbox_plot_path}")


def draw_annotated_samples(dataset_dir: Path, output_dir: Path, num_samples: int = 6) -> None:
    """Draw bounding boxes and class labels on sample images and save to disk."""
    sample_dir = output_dir / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_all_images(dataset_dir)
    selected = image_files[:num_samples]
    logger.info(f"Rendering {len(selected)} sample annotated images...")

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = find_label_for_image(img_path, dataset_dir)
        annotations = parse_yolo_label(lbl_path) if (lbl_path and lbl_path.exists()) else []

        for ann in annotations:
            if validate_annotation(ann):
                continue

            cls_id = ann["class_id"]
            color = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]
            cls_name = CLASS_NAMES[cls_id]

            # Convert YOLO normalized coords to pixel bounds
            xc, yc = ann["x_center"] * w, ann["y_center"] * h
            bw, bh = ann["width"] * w, ann["height"] * h
            x1 = int(round(xc - bw / 2))
            y1 = int(round(yc - bh / 2))
            x2 = int(round(xc + bw / 2))
            y2 = int(round(yc + bh / 2))

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw label banner
            label_text = f"{cls_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            text_y1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img, (x1, text_y1), (x1 + tw + 6, text_y1 + th + baseline + 4), color, -1)
            cv2.putText(img, label_text, (x1 + 3, text_y1 + th + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        out_path = sample_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_path), img)

    logger.info(f"Saved sample annotated images to: {sample_dir}")


def main():
    dataset_dir = get_raw_dataset_dir()
    if not dataset_dir.exists():
        logger.error(f"Dataset not found at {dataset_dir}. Run scripts/01_download_pku_dataset.py first.")
        sys.exit(1)

    reports_dir = get_reports_dir()

    stats, raw_data = analyze_dataset(dataset_dir)

    # Save JSON report
    report_path = reports_dir / "dataset_analysis.json"
    save_json_report(stats, report_path)
    logger.info(f"Saved statistical report to: {report_path}")

    # Generate Plots
    generate_plots(raw_data, reports_dir)

    # Render Sample Images
    draw_annotated_samples(dataset_dir, reports_dir)

    # Print summary
    print("\n" + "=" * 65)
    print("PKU-MARKET-PCB DATASET STATISTICS SUMMARY")
    print("=" * 65)
    summ = stats["dataset_summary"]
    print(f"Total Images:               {summ['total_images']}")
    print(f"Total Defect Annotations:   {summ['total_annotations']}")
    print(f"Avg Defects per Image:      {summ['avg_annotations_per_image']:.2f}")
    print(f"Range per Image:            {summ['min_annotations_per_image']} to {summ['max_annotations_per_image']}")
    print(f"Native Resolution:          {summ['common_resolution']}")
    print("\nPer-Class Distribution:")
    for cls_name, info in stats["class_distribution"].items():
        print(f"  {cls_name:<18s}: {info['instances']:3d} boxes ({info['percentage_of_total']:5.1f}%) across {info['images_containing']:2d} images ({info['image_presence_pct']:4.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
