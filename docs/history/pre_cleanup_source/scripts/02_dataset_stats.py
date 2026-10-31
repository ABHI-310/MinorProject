"""Dataset statistics and sample visualizations for PKU-Market-PCB."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CLASS_NAMES, PKU_ALL_IMAGES, PKU_ALL_LABELS, REPORTS_DIR, ensure_project_dirs  # noqa: E402
from preprocessing.yolo_io import iter_images, load_yolo_labels, yolo_to_xyxy  # noqa: E402


def collect_stats(images_dir: Path, labels_dir: Path) -> dict:
    images = iter_images(images_dir)
    box_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    empty = 0
    samples = []

    for image_path in tqdm(images, desc="scan dataset"):
        labels = load_yolo_labels(labels_dir / f"{image_path.stem}.txt")
        if not labels:
            empty += 1
            continue
        present = set()
        for class_id, *_ in labels:
            name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"class_{class_id}"
            box_counts[name] += 1
            present.add(name)
        for name in present:
            image_counts[name] += 1
        if len(samples) < 12:
            samples.append(str(image_path))

    return {
        "num_images": len(images),
        "empty_label_images": empty,
        "boxes_per_class": dict(box_counts),
        "images_per_class": dict(image_counts),
        "classes": CLASS_NAMES,
        "sample_images": samples,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
    }


def plot_class_distribution(stats: dict, output_path: Path) -> None:
    names = CLASS_NAMES
    boxes = [stats["boxes_per_class"].get(name, 0) for name in names]
    images = [stats["images_per_class"].get(name, 0) for name in names]
    x = range(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, boxes, color="#3b82f6")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_title("Boxes per class")
    axes[1].bar(x, images, color="#10b981")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_title("Images containing class")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_annotation_samples(image_paths: list[str], labels_dir: Path, output_dir: Path, limit: int = 8) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for image_path in image_paths[:limit]:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels = load_yolo_labels(labels_dir / f"{path.stem}.txt")
        for label in labels:
            class_id = label[0]
            x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)
            color = colors[class_id % len(colors)]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id)
            cv2.putText(
                image,
                name,
                (int(x1), max(20, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        out_path = output_dir / f"sample_{path.stem}.jpg"
        cv2.imwrite(str(out_path), image)
        written.append(str(out_path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PKU-Market-PCB dataset statistics.")
    parser.add_argument("--images", type=Path, default=PKU_ALL_IMAGES)
    parser.add_argument("--labels", type=Path, default=PKU_ALL_LABELS)
    args = parser.parse_args()
    ensure_project_dirs()
    stats = collect_stats(args.images, args.labels)
    plot_path = REPORTS_DIR / "pku_class_distribution.png"
    plot_class_distribution(stats, plot_path)
    sample_paths = save_annotation_samples(
        stats["sample_images"],
        args.labels,
        REPORTS_DIR / "pku_annotation_samples",
    )
    stats["visualizations"] = {
        "class_distribution": str(plot_path),
        "annotation_samples": sample_paths,
    }
    report_path = REPORTS_DIR / "dataset_analysis.json"
    report_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("num_images", "empty_label_images", "boxes_per_class")}, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
