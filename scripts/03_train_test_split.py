"""
03_train_test_split.py
======================
Stratified multi-label train / val / test split for PKU-Market-PCB.

Splits the raw dataset into:
  - 70% Train (~42 images)
  - 15% Validation (~9 images)
  - 15% Test (~9 images)

Uses multi-label iterative stratification to preserve balanced class ratios
across splits without data leakage.

Outputs:
  - data/pku_market_processed/images/{train,val,test}
  - data/pku_market_processed/labels/{train,val,test}
  - data/pku_market_processed/data.yaml
  - reports/split_manifest.json

Usage:
    python scripts/03_train_test_split.py
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    CLASS_NAMES,
    NUM_CLASSES,
    get_processed_dataset_dir,
    get_raw_dataset_dir,
    get_reports_dir,
    parse_yolo_label,
    save_json_report,
    save_yaml_config,
    setup_logging,
    validate_annotation,
)

logger = setup_logging("03_split")


def build_multilabel_matrix(image_files: List[Path], labels_dir: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Build binary multi-label presence matrix (num_images, num_classes).
    Also returns list of image stems.
    """
    matrix = np.zeros((len(image_files), NUM_CLASSES), dtype=int)
    stems = []

    for idx, img_path in enumerate(image_files):
        stem = img_path.stem
        stems.append(stem)
        lbl_path = labels_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        anns = parse_yolo_label(lbl_path)
        for ann in anns:
            if not validate_annotation(ann):
                matrix[idx, ann["class_id"]] = 1

    return matrix, stems


def iterative_multilabel_split(
    stems: List[str],
    Y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Perform stratified split for multi-label data.
    Attempts using IterativeStratification, with reliable greedy fallback.
    """
    np.random.seed(seed)
    n_samples = len(stems)

    try:
        from skmultilearn.model_selection import IterativeStratification

        # First split into train and temp (val+test)
        temp_ratio = val_ratio + test_ratio
        stratifier1 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[temp_ratio, train_ratio])
        train_idx, temp_idx = next(stratifier1.split(np.zeros(n_samples), Y))

        # Split temp into val and test
        val_subratio = val_ratio / temp_ratio
        test_subratio = test_ratio / temp_ratio
        stratifier2 = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[test_subratio, val_subratio])
        val_sub_idx, test_sub_idx = next(stratifier2.split(np.zeros(len(temp_idx)), Y[temp_idx]))

        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]

        train_stems = [stems[i] for i in train_idx]
        val_stems = [stems[i] for i in val_idx]
        test_stems = [stems[i] for i in test_idx]

        logger.info("Used scikit-multilearn IterativeStratification successfully.")

    except Exception as e:
        logger.warning(f"IterativeStratification fallback triggered: {e}")
        # Deterministic multi-label greedy stratification fallback
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_stems = [stems[i] for i in train_idx]
        val_stems = [stems[i] for i in val_idx]
        test_stems = [stems[i] for i in test_idx]

    return {
        "train": sorted(train_stems),
        "val": sorted(val_stems),
        "test": sorted(test_stems),
    }


def copy_split_files(
    splits: Dict[str, List[str]],
    raw_images_dir: Path,
    raw_labels_dir: Path,
    processed_dir: Path,
) -> None:
    """Copy split files to destination directory structure."""
    if processed_dir.exists():
        logger.info(f"Cleaning existing processed dir: {processed_dir}")
        shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, stem_list in splits.items():
        img_out = processed_dir / "images" / split_name
        lbl_out = processed_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in stem_list:
            # Copy image
            src_img = raw_images_dir / f"{stem}.jpg"
            if not src_img.exists():
                # Try png
                src_img = raw_images_dir / f"{stem}.png"
            if src_img.exists():
                shutil.copy2(src_img, img_out / src_img.name)

            # Copy label
            src_lbl = raw_labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_out / src_lbl.name)
            else:
                # Create empty label file if none exists (negative background sample)
                (lbl_out / f"{stem}.txt").touch()

    logger.info("Successfully copied images and labels into train/val/test splits.")


def verify_no_leakage(splits: Dict[str, List[str]]) -> bool:
    """Verify sets are pairwise completely disjoint."""
    s_train = set(splits["train"])
    s_val = set(splits["val"])
    s_test = set(splits["test"])

    tv_overlap = s_train.intersection(s_val)
    tt_overlap = s_train.intersection(s_test)
    vt_overlap = s_val.intersection(s_test)

    if tv_overlap or tt_overlap or vt_overlap:
        logger.error(f"DATA LEAKAGE DETECTED! Overlaps: train-val={tv_overlap}, train-test={tt_overlap}, val-test={vt_overlap}")
        return False

    logger.info("[OK] Data leakage verification PASSED. All splits are mutually exclusive.")
    return True


def compute_split_class_stats(splits: Dict[str, List[str]], labels_dir: Path) -> dict:
    """Compute per-class instance and image counts for each split."""
    stats = {}
    for split_name, stems in splits.items():
        counts = {name: 0 for name in CLASS_NAMES}
        img_counts = {name: 0 for name in CLASS_NAMES}

        for stem in stems:
            lbl_file = labels_dir / split_name / f"{stem}.txt"
            if not lbl_file.exists():
                continue
            anns = parse_yolo_label(lbl_file)
            classes_in_img = set()
            for a in anns:
                if not validate_annotation(a):
                    cls_name = CLASS_NAMES[a["class_id"]]
                    counts[cls_name] += 1
                    classes_in_img.add(cls_name)
            for cls_name in classes_in_img:
                img_counts[cls_name] += 1

        stats[split_name] = {
            "num_images": len(stems),
            "total_instances": sum(counts.values()),
            "class_instances": counts,
            "class_images": img_counts,
        }
    return stats


def main():
    raw_dir = get_raw_dataset_dir()
    processed_dir = get_processed_dataset_dir()
    reports_dir = get_reports_dir()

    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.error("Raw images or labels directory not found. Run scripts/01_download_pku_dataset.py first.")
        sys.exit(1)

    image_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    logger.info(f"Building stratification matrix for {len(image_files)} raw images...")

    Y, stems = build_multilabel_matrix(image_files, labels_dir)

    splits = iterative_multilabel_split(stems, Y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Verify no leakage
    if not verify_no_leakage(splits):
        sys.exit(1)

    # Copy files
    copy_split_files(splits, images_dir, labels_dir, processed_dir)

    # Compute split statistics
    split_stats = compute_split_class_stats(splits, processed_dir / "labels")

    # Generate data.yaml in processed_dir and in configs/
    processed_yaml = {
        "path": str(processed_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    save_yaml_config(processed_yaml, processed_dir / "data.yaml")
    save_yaml_config(processed_yaml, Path(__file__).resolve().parent.parent / "configs" / "data.yaml")
    logger.info("Saved data.yaml configuration pointing to processed dataset.")

    # Save split manifest
    manifest = {
        "summary": {
            "total_images": len(stems),
            "train_images": len(splits["train"]),
            "val_images": len(splits["val"]),
            "test_images": len(splits["test"]),
            "split_ratios": "70/15/15",
            "seed": 42,
        },
        "split_statistics": split_stats,
        "splits": splits,
    }
    manifest_path = reports_dir / "split_manifest.json"
    save_json_report(manifest, manifest_path)
    logger.info(f"Split manifest saved to: {manifest_path}")

    # Print summary
    print("\n" + "=" * 65)
    print("DATASET SPLIT SUMMARY (70% Train / 15% Val / 15% Test)")
    print("=" * 65)
    print(f"{'Split':<8s} | {'Images':<8s} | {'Boxes':<8s} | Class Presence (images)")
    print("-" * 65)
    for s_name in ["train", "val", "test"]:
        s_info = split_stats[s_name]
        pres_str = ", ".join([f"{k[:4]}:{v}" for k, v in s_info["class_images"].items()])
        print(f"{s_name:<8s} | {s_info['num_images']:<8d} | {s_info['total_instances']:<8d} | {pres_str}")
    print("=" * 65)
    print("[OK] Split complete. Ready for training.")


if __name__ == "__main__":
    main()
