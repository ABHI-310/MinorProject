"""
03_train_test_split.py
Performs stratified train/val/test split (70% train / 15% val / 15% test) on PKU-Market-PCB dataset.
Organizes files into train, val, and test partitions with images/ and labels/ subdirectories,
and generates data.yaml.
"""

import argparse
from pathlib import Path
import random
import shutil
import yaml

PKU_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
]


def split_dataset(
    data_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Missing images or labels directory in {data_dir}")

    image_files = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.png"))
        + list(images_dir.glob("*.jpeg"))
    )

    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    # Build primary class for each image for stratified splitting
    items_by_class = {c: [] for c in range(len(PKU_CLASSES))}
    uncategorized = []

    for img_path in image_files:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        primary_cls = None
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    primary_cls = int(first_line.split()[0])

        if primary_cls is not None and 0 <= primary_cls < len(PKU_CLASSES):
            items_by_class[primary_cls].append(img_path)
        else:
            uncategorized.append(img_path)

    splits = {"train": [], "val": [], "test": []}

    for cls_id, paths in items_by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits["train"].extend(paths[:n_train])
        splits["val"].extend(paths[n_train : n_train + n_val])
        splits["test"].extend(paths[n_train + n_val :])

    # Add uncategorized
    random.shuffle(uncategorized)
    n = len(uncategorized)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits["train"].extend(uncategorized[:n_train])
    splits["val"].extend(uncategorized[n_train : n_train + n_val])
    splits["test"].extend(uncategorized[n_train + n_val :])

    print("Splitting results:")
    for split_name, paths in splits.items():
        print(f"  {split_name}: {len(paths)} images")

    # Create destination directories
    for split_name, paths in splits.items():
        target_img_dir = data_dir / "images" / split_name
        target_lbl_dir = data_dir / "labels" / split_name
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in paths:
            # Move or copy to split folder
            dst_img = target_img_dir / img_path.name
            src_lbl = labels_dir / f"{img_path.stem}.txt"
            dst_lbl = target_lbl_dir / f"{img_path.stem}.txt"

            shutil.copy2(img_path, dst_img)
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    # Generate data.yaml
    data_yaml = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(PKU_CLASSES),
        "names": PKU_CLASSES,
    }

    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"✓ Generated YOLOv8 dataset configuration: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Stratified Train/Val/Test Split")
    parser.add_argument("--data-dir", type=str, default="data/pku_market", help="Dataset directory")
    parser.add_argument("--train", type=float, default=0.70, help="Train fraction")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction")
    args = parser.parse_args()

    split_dataset(Path(args.data_dir), args.train, args.val, args.test)


if __name__ == "__main__":
    main()

