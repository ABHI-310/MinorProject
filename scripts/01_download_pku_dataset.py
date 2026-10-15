"""
01_download_pku_dataset.py
===========================
Dataset Acquisition & Verification for PKU-Market-PCB.

Downloads the dataset from Roboflow in YOLOv8 format, or uses an existing
local copy. Performs comprehensive integrity checks on the downloaded data.

Usage:
    python scripts/01_download_pku_dataset.py
    python scripts/01_download_pku_dataset.py --local-path "D:/path/to/existing/dataset"

Environment:
    ROBOFLOW_API_KEY - Required for Roboflow download (free account).
"""

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import cv2

# Add project root to path
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

logger = setup_logging("01_download")


# =============================================================================
# Dataset Download
# =============================================================================

def download_from_roboflow(target_dir: Path) -> Path:
    """
    Download the PKU-Market-PCB dataset from Roboflow Universe.

    Requires ROBOFLOW_API_KEY environment variable.

    Returns:
        Path to the downloaded dataset directory.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        logger.error(
            "ROBOFLOW_API_KEY environment variable not set.\n"
            "  1. Create a free account at https://roboflow.com\n"
            "  2. Go to Settings > API Keys\n"
            "  3. Set the environment variable:\n"
            "     Windows:  set ROBOFLOW_API_KEY=your_key_here\n"
            "     Linux:    export ROBOFLOW_API_KEY=your_key_here\n"
            "  4. Re-run this script."
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    logger.info("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("jr-mqqnk").project("pcb-defects-detection-anddl")
    version = project.version(1)

    logger.info(f"Downloading dataset to: {target_dir}")
    dataset = version.download("yolov8", location=str(target_dir))

    logger.info(f"Download complete. Location: {dataset.location}")
    return Path(dataset.location)


def copy_local_dataset(local_path: Path, target_dir: Path) -> Path:
    """
    Copy a local dataset to the project's data directory.

    Only copies raw image and label files (not processed/augmented data).
    Handles both flat structure and split (train/val/test) structure.

    Returns:
        Path to the copied dataset directory.
    """
    if not local_path.exists():
        logger.error(f"Local dataset path does not exist: {local_path}")
        sys.exit(1)

    images_dir = local_path / "images"
    labels_dir = local_path / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        logger.error(
            f"Expected images/ and labels/ directories in: {local_path}\n"
            f"  Found: {[p.name for p in local_path.iterdir() if p.is_dir()]}"
        )
        sys.exit(1)

    # If target already exists with data, check if it's already valid
    if target_dir.exists() and (target_dir / "images").exists():
        img_count = len(list((target_dir / "images").glob("*.jpg")))
        if img_count > 0:
            logger.info(
                f"Target directory already contains {img_count} images. "
                f"Skipping copy. Delete {target_dir} to force re-copy."
            )
            return target_dir

    logger.info(f"Copying raw dataset from: {local_path}")
    logger.info(f"  -> Target: {target_dir}")

    # Create target dirs
    target_images = target_dir / "images"
    target_labels = target_dir / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    # Copy flat images (skip subdirectories which are split copies)
    for img_file in sorted(images_dir.glob("*")):
        if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            shutil.copy2(img_file, target_images / img_file.name)
            copied_images += 1

    # Copy flat labels (skip subdirectories)
    for lbl_file in sorted(labels_dir.glob("*")):
        if lbl_file.is_file() and lbl_file.suffix.lower() == ".txt":
            shutil.copy2(lbl_file, target_labels / lbl_file.name)
            copied_labels += 1

    # Also copy data.yaml if it exists
    data_yaml = local_path / "data.yaml"
    if data_yaml.exists():
        shutil.copy2(data_yaml, target_dir / "data.yaml")
        logger.info("  Copied data.yaml")

    logger.info(f"  Copied {copied_images} images and {copied_labels} labels.")

    if copied_images == 0:
        # Maybe the dataset only has split structure (train/val/test), no flat files
        logger.warning("No flat image files found. Checking for split structure...")
        for split_name in ["train", "valid", "val", "test"]:
            split_img_dir = images_dir / split_name
            split_lbl_dir = labels_dir / split_name
            if split_img_dir.exists():
                target_split_img = target_images / split_name
                target_split_lbl = target_labels / split_name
                target_split_img.mkdir(parents=True, exist_ok=True)
                target_split_lbl.mkdir(parents=True, exist_ok=True)

                for img_file in sorted(split_img_dir.glob("*")):
                    if img_file.is_file() and img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        shutil.copy2(img_file, target_split_img / img_file.name)
                        copied_images += 1

                if split_lbl_dir.exists():
                    for lbl_file in sorted(split_lbl_dir.glob("*")):
                        if lbl_file.is_file() and lbl_file.suffix.lower() == ".txt":
                            shutil.copy2(lbl_file, target_split_lbl / lbl_file.name)
                            copied_labels += 1

                logger.info(f"  Split '{split_name}': {len(list(target_split_img.glob('*')))} images")

        logger.info(f"  Total copied: {copied_images} images, {copied_labels} labels")

    return target_dir


# =============================================================================
# Dataset Verification
# =============================================================================

def find_all_images(dataset_dir: Path) -> list[Path]:
    """Find all image files in the dataset, supporting flat, images/<split>, or <split>/images."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []

    # Case 1: Check <split>/images structure (standard Roboflow)
    for split_dir in ["train", "valid", "val", "test"]:
        s_img = dataset_dir / split_dir / "images"
        if s_img.exists() and s_img.is_dir():
            for f in sorted(s_img.iterdir()):
                if f.is_file() and f.suffix.lower() in image_extensions:
                    images.append(f)

    if images:
        return images

    # Case 2: Check dataset_dir/images
    images_dir = dataset_dir / "images"
    if images_dir.exists():
        for f in sorted(images_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in image_extensions:
                images.append(f)
            elif f.is_dir():
                for sf in sorted(f.iterdir()):
                    if sf.is_file() and sf.suffix.lower() in image_extensions:
                        images.append(sf)

    return images


def find_label_for_image(image_path: Path, dataset_dir: Path) -> Path | None:
    """Find the corresponding label file for an image across structures."""
    label_name = image_path.stem + ".txt"

    # Case 1: image is in .../images/... -> look in sibling labels/...
    if image_path.parent.name == "images":
        labels_dir = image_path.parent.parent / "labels"
        target = labels_dir / label_name
        if target.exists():
            return target

    # Case 2: relative to dataset_dir/images
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    try:
        relative = image_path.relative_to(images_dir)
        target = labels_dir / relative.parent / label_name
        if target.exists():
            return target
    except ValueError:
        pass

    # Case 3: flat search
    for candidate in dataset_dir.rglob(label_name):
        if candidate.is_file():
            return candidate

    return None


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset(dataset_dir: Path) -> dict:
    """
    Perform comprehensive verification of the dataset.

    Returns:
        Dict with verification results and statistics.
    """
    report = {
        "dataset_path": str(dataset_dir),
        "structure_check": {},
        "image_count": 0,
        "label_count": 0,
        "total_annotations": 0,
        "class_distribution": {name: 0 for name in CLASS_NAMES},
        "images_per_class": {name: 0 for name in CLASS_NAMES},
        "issues": {
            "missing_labels": [],
            "orphan_labels": [],
            "unreadable_images": [],
            "empty_labels": [],
            "invalid_annotations": [],
            "invalid_class_ids": [],
            "out_of_bounds_coords": [],
        },
        "duplicates": [],
        "overall_valid": True,
    }

    # --- Structure check ---
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    has_flat = images_dir.exists() and labels_dir.exists()
    has_splits = (dataset_dir / "train" / "images").exists() and (dataset_dir / "train" / "labels").exists()

    report["structure_check"]["has_flat_structure"] = has_flat
    report["structure_check"]["has_split_structure"] = has_splits

    if not has_flat and not has_splits:
        report["overall_valid"] = False
        logger.error(f"Missing valid dataset directory structure in: {dataset_dir}")
        return report

    # Check for split subdirs
    for split in ["train", "val", "valid", "test"]:
        report["structure_check"][f"{split}/images"] = (dataset_dir / split / "images").exists() or (images_dir / split).exists()
        report["structure_check"][f"{split}/labels"] = (dataset_dir / split / "labels").exists() or (labels_dir / split).exists()

    # --- Find all images ---
    all_images = find_all_images(dataset_dir)
    report["image_count"] = len(all_images)
    logger.info(f"Found {len(all_images)} images")

    # --- Image readability check ---
    logger.info("Checking image readability...")
    readable_count = 0
    for img_path in all_images:
        img = cv2.imread(str(img_path))
        if img is None:
            report["issues"]["unreadable_images"].append(str(img_path))
            logger.warning(f"  UNREADABLE: {img_path.name}")
        else:
            readable_count += 1
    logger.info(f"  {readable_count}/{len(all_images)} images readable")

    # --- Label pairing check ---
    logger.info("Checking image-label pairing...")
    label_count = 0
    all_annotations = 0
    images_with_class = {name: set() for name in CLASS_NAMES}

    for img_path in all_images:
        label_path = find_label_for_image(img_path, dataset_dir)

        if label_path is None:
            report["issues"]["missing_labels"].append(str(img_path))
            continue

        label_count += 1
        annotations = parse_yolo_label(label_path)

        if len(annotations) == 0:
            report["issues"]["empty_labels"].append(str(label_path))
            continue

        # Track classes found in this image
        classes_in_image = set()

        for ann in annotations:
            # Validate each annotation
            issues = validate_annotation(ann)
            if issues:
                for issue in issues:
                    report["issues"]["invalid_annotations"].append(
                        f"{label_path.name}: {issue}"
                    )
                    if "class_id" in issue:
                        report["issues"]["invalid_class_ids"].append(
                            f"{label_path.name}: {issue}"
                        )
                    if "out of [0, 1]" in issue or "extends beyond" in issue:
                        report["issues"]["out_of_bounds_coords"].append(
                            f"{label_path.name}: {issue}"
                        )
                continue

            # Count valid annotations
            all_annotations += 1
            class_id = ann["class_id"]
            class_name = CLASS_NAMES[class_id]
            report["class_distribution"][class_name] += 1
            classes_in_image.add(class_name)

        # Track images containing each class
        for cls_name in classes_in_image:
            images_with_class[cls_name].add(img_path.stem)

    report["label_count"] = label_count
    report["total_annotations"] = all_annotations
    report["images_per_class"] = {k: len(v) for k, v in images_with_class.items()}

    logger.info(f"  {label_count}/{len(all_images)} images have labels")
    logger.info(f"  {all_annotations} total valid annotations")

    # Check for orphan labels (labels without images)
    all_image_stems = {img.stem for img in all_images}
    labels_dir = dataset_dir / "labels"
    for lbl_file in labels_dir.rglob("*.txt"):
        if lbl_file.stem not in all_image_stems:
            report["issues"]["orphan_labels"].append(str(lbl_file))

    # --- Duplicate detection ---
    logger.info("Checking for duplicate images (MD5 hash)...")
    hash_to_files: dict[str, list[str]] = {}
    for img_path in all_images:
        file_hash = compute_file_hash(img_path)
        if file_hash not in hash_to_files:
            hash_to_files[file_hash] = []
        hash_to_files[file_hash].append(str(img_path))

    for file_hash, files in hash_to_files.items():
        if len(files) > 1:
            report["duplicates"].append({
                "hash": file_hash,
                "files": files,
            })

    if report["duplicates"]:
        logger.warning(f"  Found {len(report['duplicates'])} sets of duplicate images")
    else:
        logger.info("  No duplicates found")

    # --- Summary ---
    total_issues = sum(len(v) for v in report["issues"].values())
    if total_issues > 0:
        report["overall_valid"] = False
        logger.warning(f"Found {total_issues} total issues (see report for details)")
    else:
        logger.info("Dataset verification PASSED — no issues found")

    return report


def print_verification_summary(report: dict) -> None:
    """Print a human-readable summary of the verification report."""
    print("\n" + "=" * 70)
    print("DATASET VERIFICATION REPORT")
    print("=" * 70)
    print(f"  Path:              {report['dataset_path']}")
    print(f"  Total images:      {report['image_count']}")
    print(f"  Total labels:      {report['label_count']}")
    print(f"  Total annotations: {report['total_annotations']}")
    print(f"  Overall valid:     {'YES' if report['overall_valid'] else 'NO — see issues'}")
    print()

    print("  Class distribution (annotation instances):")
    for cls_name, count in report["class_distribution"].items():
        img_count = report["images_per_class"].get(cls_name, 0)
        print(f"    {cls_name:<20s}: {count:4d} instances in {img_count:3d} images")

    # Issues summary
    print()
    has_issues = False
    for issue_type, items in report["issues"].items():
        if items:
            has_issues = True
            print(f"  [!] {issue_type}: {len(items)}")
            for item in items[:5]:  # Show first 5
                print(f"      - {item}")
            if len(items) > 5:
                print(f"      ... and {len(items) - 5} more")

    if not has_issues:
        print("  [OK] No issues found")

    if report["duplicates"]:
        print(f"\n  [!] Duplicate image sets: {len(report['duplicates'])}")
        for dup in report["duplicates"][:3]:
            print(f"      Hash {dup['hash'][:12]}...: {len(dup['files'])} copies")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download/copy and verify the PKU-Market-PCB dataset."
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to an existing local copy of the dataset. "
             "If not provided, downloads from Roboflow.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download/copy, only run verification on existing data.",
    )
    args = parser.parse_args()

    target_dir = get_raw_dataset_dir()

    if args.skip_download:
        if not target_dir.exists():
            logger.error(f"--skip-download specified but no dataset at: {target_dir}")
            sys.exit(1)
        logger.info(f"Skipping download, verifying existing dataset at: {target_dir}")
    elif args.local_path:
        # Copy from local path
        local_path = Path(args.local_path).resolve()
        target_dir = copy_local_dataset(local_path, target_dir)
    else:
        # Download from Roboflow
        if target_dir.exists() and (target_dir / "images").exists():
            img_count = len(find_all_images(target_dir))
            if img_count > 0:
                logger.info(
                    f"Dataset already exists at {target_dir} with {img_count} images. "
                    f"Skipping download. Delete the directory to re-download."
                )
            else:
                target_dir = download_from_roboflow(target_dir)
        else:
            target_dir = download_from_roboflow(target_dir)

    # Verify
    logger.info("Starting dataset verification...")
    report = verify_dataset(target_dir)
    print_verification_summary(report)

    # Save report
    report_path = get_reports_dir() / "dataset_validation_report.json"
    save_json_report(report, report_path)
    logger.info(f"Verification report saved to: {report_path}")

    # Also save as human-readable text
    text_report_path = get_reports_dir() / "dataset_validation_report.txt"
    with open(text_report_path, "w", encoding="utf-8") as f:
        f.write("DATASET VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Path: {report['dataset_path']}\n")
        f.write(f"Total images: {report['image_count']}\n")
        f.write(f"Total labels: {report['label_count']}\n")
        f.write(f"Total annotations: {report['total_annotations']}\n")
        f.write(f"Overall valid: {'YES' if report['overall_valid'] else 'NO'}\n\n")

        f.write("Class distribution:\n")
        for cls_name, count in report["class_distribution"].items():
            img_count = report["images_per_class"].get(cls_name, 0)
            f.write(f"  {cls_name:<20s}: {count:4d} instances in {img_count:3d} images\n")

        f.write("\nIssues:\n")
        for issue_type, items in report["issues"].items():
            if items:
                f.write(f"  {issue_type}: {len(items)}\n")
                for item in items:
                    f.write(f"    - {item}\n")

        if report["duplicates"]:
            f.write(f"\nDuplicates: {len(report['duplicates'])} sets\n")
            for dup in report["duplicates"]:
                f.write(f"  Hash {dup['hash']}: {dup['files']}\n")

    logger.info(f"Text report saved to: {text_report_path}")

    if not report["overall_valid"]:
        logger.warning(
            "Dataset has issues. Review the report before proceeding. "
            "Minor issues (like duplicates in split subdirs) may be acceptable."
        )
    else:
        logger.info("Dataset is ready for analysis (02_dataset_stats.py)")


if __name__ == "__main__":
    main()
