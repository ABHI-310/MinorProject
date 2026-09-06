"""
06_sample_target_images.py
Stratified few-shot sampling of 50-100 target images from DsPCBSD+.
Converts annotations to the aligned 6-class space, moving the selected samples
to data/dspcbsd/labelled_few_shot/ and remaining images to data/dspcbsd/unlabelled/.
Outputs reports/target_sampling_stats.json.
"""

import argparse
import json
from pathlib import Path
import random
import shutil

PKU_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
]


def sample_target_images(
    dspcbsd_dir: Path,
    mapping_file: Path,
    reports_dir: Path,
    num_samples: int = 75,
    seed: int = 42,
):
    random.seed(seed)
    dspcbsd_dir = Path(dspcbsd_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    images_dir = dspcbsd_dir / "images"
    labels_dir = dspcbsd_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")

    # Load mapping
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    class_id_lookup = {int(k): int(v) for k, v in mapping_data["class_id_lookup"].items()}

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    # Index by mapped class
    samples_by_class = {c: [] for c in range(len(PKU_CLASSES))}
    other_samples = []

    for img in image_files:
        lbl = labels_dir / f"{img.stem}.txt"
        mapped_class = None
        if lbl.exists():
            with open(lbl, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    raw_cls = int(first_line.split()[0])
                    mapped_class = class_id_lookup.get(raw_cls, 0)

        if mapped_class is not None:
            samples_by_class[mapped_class].append(img)
        else:
            other_samples.append(img)

    per_class_target = max(1, num_samples // len(PKU_CLASSES))
    selected_few_shot = []

    for cls_id, files in samples_by_class.items():
        random.shuffle(files)
        selected_few_shot.extend(files[:per_class_target])

    # Fill remaining up to num_samples if needed
    if len(selected_few_shot) < num_samples:
        remaining_pool = [img for img in image_files if img not in selected_few_shot]
        random.shuffle(remaining_pool)
        needed = num_samples - len(selected_few_shot)
        selected_few_shot.extend(remaining_pool[:needed])

    selected_set = set(selected_few_shot)
    unlabelled_pool = [img for img in image_files if img not in selected_set]

    # Destination directories
    few_shot_dir = dspcbsd_dir / "labelled_few_shot"
    few_shot_img = few_shot_dir / "images"
    few_shot_lbl = few_shot_dir / "labels"
    few_shot_img.mkdir(parents=True, exist_ok=True)
    few_shot_lbl.mkdir(parents=True, exist_ok=True)

    unlabelled_dir = dspcbsd_dir / "unlabelled" / "images"
    unlabelled_dir.mkdir(parents=True, exist_ok=True)

    # Process few-shot samples (with mapped labels)
    class_distribution = {c: 0 for c in PKU_CLASSES}
    for img in selected_few_shot:
        shutil.copy2(img, few_shot_img / img.name)
        src_lbl = labels_dir / f"{img.stem}.txt"
        dst_lbl = few_shot_lbl / f"{img.stem}.txt"

        if src_lbl.exists():
            converted_lines = []
            with open(src_lbl, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        raw_cls = int(parts[0])
                        mapped_cls = class_id_lookup.get(raw_cls, 0)
                        converted_lines.append(f"{mapped_cls} {' '.join(parts[1:5])}")
                        class_distribution[PKU_CLASSES[mapped_cls]] += 1
            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(converted_lines) + "\n")

    # Process unlabelled images
    for img in unlabelled_pool:
        shutil.copy2(img, unlabelled_dir / img.name)

    stats = {
        "total_target_images": len(image_files),
        "few_shot_count": len(selected_few_shot),
        "unlabelled_pool_count": len(unlabelled_pool),
        "few_shot_class_distribution": class_distribution,
    }

    stats_file = reports_dir / "target_sampling_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Stratified few-shot sampling complete:")
    print(f"  Sampled {len(selected_few_shot)} images to {few_shot_dir}")
    print(f"  Stored {len(unlabelled_pool)} unlabelled images in {unlabelled_dir.parent}")
    print(f"  Saved statistics to {stats_file}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sample few-shot target domain images")
    parser.add_argument("--dspcbsd-dir", type=str, default="data/dspcbsd")
    parser.add_argument("--mapping", type=str, default="data/dspcbsd/class_mapping.json")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--shots", type=int, default=75, help="Number of labelled few-shot images (50-100)")
    args = parser.parse_args()

    sample_target_images(
        Path(args.dspcbsd_dir),
        Path(args.mapping),
        Path(args.reports_dir),
        num_samples=args.shots,
    )


if __name__ == "__main__":
    main()

