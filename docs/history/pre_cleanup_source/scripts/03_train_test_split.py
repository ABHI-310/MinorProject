"""Stratified 70/15/15 train/val/test split for the PKU YOLO pool."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    PKU_ALL_IMAGES,
    PKU_ALL_LABELS,
    PKU_DIR,
    PKU_SPLIT_IMAGES,
    PKU_SPLIT_LABELS,
    PKU_YAML,
    REPORTS_DIR,
    ensure_project_dirs,
)
from preprocessing.yolo_io import iter_images, load_yolo_labels, primary_class_id, write_yolo_yaml  # noqa: E402


def _copy_pair(image_path: Path, labels_dir: Path, split: str) -> None:
    dest_image = PKU_SPLIT_IMAGES / split / image_path.name
    dest_label = PKU_SPLIT_LABELS / split / f"{image_path.stem}.txt"
    dest_image.parent.mkdir(parents=True, exist_ok=True)
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_image)
    src_label = labels_dir / f"{image_path.stem}.txt"
    if src_label.exists():
        shutil.copy2(src_label, dest_label)
    else:
        dest_label.write_text("")


def stratified_split(
    images_dir: Path = PKU_ALL_IMAGES,
    labels_dir: Path = PKU_ALL_LABELS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    ensure_project_dirs()
    images = iter_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images in {images_dir}. Run scripts/01_download_pku_dataset.py first.")

    y = []
    for image_path in images:
        labels = load_yolo_labels(labels_dir / f"{image_path.stem}.txt")
        y.append(primary_class_id(labels))

    test_ratio = 1.0 - train_ratio - val_ratio
    try:
        train_imgs, hold_imgs, train_y, hold_y = train_test_split(
            images,
            y,
            test_size=(val_ratio + test_ratio),
            random_state=seed,
            stratify=y,
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs, _, _ = train_test_split(
            hold_imgs,
            hold_y,
            test_size=(1.0 - relative_val),
            random_state=seed,
            stratify=hold_y,
        )
        stratified = True
    except ValueError:
        train_imgs, hold_imgs = train_test_split(
            images, test_size=(val_ratio + test_ratio), random_state=seed
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            hold_imgs, test_size=(1.0 - relative_val), random_state=seed
        )
        stratified = False

    for split in ("train", "val", "test"):
        split_img = PKU_SPLIT_IMAGES / split
        split_lbl = PKU_SPLIT_LABELS / split
        if split_img.exists():
            shutil.rmtree(split_img)
        if split_lbl.exists():
            shutil.rmtree(split_lbl)

    for image_path in train_imgs:
        _copy_pair(image_path, labels_dir, "train")
    for image_path in val_imgs:
        _copy_pair(image_path, labels_dir, "val")
    for image_path in test_imgs:
        _copy_pair(image_path, labels_dir, "test")

    write_yolo_yaml(PKU_YAML, PKU_DIR)
    summary = {
        "stratified": stratified,
        "train": len(train_imgs),
        "val": len(val_imgs),
        "test": len(test_imgs),
        "yaml": str(PKU_YAML),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "pku_split_stats.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified PKU train/val/test splits.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = stratified_split(train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
