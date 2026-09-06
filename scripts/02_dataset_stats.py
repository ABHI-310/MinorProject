"""
02_dataset_stats.py
Analyzes the PKU-Market-PCB dataset:
- Class frequencies and distribution
- Bounding box dimension statistics (aspect ratios, relative sizes)
- Exports reports/dataset_analysis.json and reports/dataset_distribution.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PKU_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
]


def analyze_dataset(data_dir: Path, reports_dir: Path):
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = data_dir / "labels"
    images_dir = data_dir / "images"

    if not labels_dir.exists():
        # Fallback to searching recursively for .txt files
        label_files = list(data_dir.rglob("*.txt"))
    else:
        label_files = list(labels_dir.glob("*.txt"))

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) if images_dir.exists() else []

    class_counts = {c: 0 for c in PKU_CLASSES}
    box_widths = []
    box_heights = []
    boxes_per_image = []

    for lbl_file in label_files:
        count_in_img = 0
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(PKU_CLASSES):
                        cls_name = PKU_CLASSES[cls_id]
                        class_counts[cls_name] += 1
                        box_widths.append(float(parts[3]))
                        box_heights.append(float(parts[4]))
                        count_in_img += 1
        boxes_per_image.append(count_in_img)

    total_annotations = sum(class_counts.values())
    total_images = len(image_files) if image_files else len(label_files)

    stats = {
        "dataset_name": "PKU-Market-PCB",
        "total_images": total_images,
        "total_annotations": total_annotations,
        "classes": PKU_CLASSES,
        "class_distribution": class_counts,
        "class_percentages": {
            k: (v / total_annotations * 100) if total_annotations > 0 else 0
            for k, v in class_counts.items()
        },
        "bbox_stats": {
            "mean_width": float(np.mean(box_widths)) if box_widths else 0.0,
            "mean_height": float(np.mean(box_heights)) if box_heights else 0.0,
            "mean_aspect_ratio": float(np.mean(np.array(box_widths) / (np.array(box_heights) + 1e-6)))
            if box_widths
            else 0.0,
            "avg_defects_per_image": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
        },
    }

    # Save JSON report
    json_path = reports_dir / "dataset_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved dataset analysis to {json_path}")

    # Generate Visualization Chart
    plt.figure(figsize=(10, 5))
    classes = list(class_counts.keys())
    counts = [class_counts[c] for c in classes]

    plt.bar(classes, counts, color="#2b5c8f", edgecolor="black", alpha=0.85)
    plt.title("Defect Class Distribution in PKU-Market-PCB", fontsize=14, fontweight="bold")
    plt.xlabel("Defect Class", fontsize=12)
    plt.ylabel("Annotation Count", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_path = reports_dir / "dataset_distribution.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"✓ Saved distribution chart to {chart_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument("--data-dir", type=str, default="data/pku_market", help="Data directory")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Reports directory")
    args = parser.parse_args()

    analyze_dataset(Path(args.data_dir), Path(args.reports_dir))


if __name__ == "__main__":
    main()

