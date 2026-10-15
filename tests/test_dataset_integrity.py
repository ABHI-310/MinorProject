"""
Unit tests for raw dataset integrity checks.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import get_raw_dataset_dir, parse_yolo_label, validate_annotation
import importlib
verify_module = importlib.import_module("scripts.01_download_pku_dataset")
verify_dataset = verify_module.verify_dataset


def test_raw_dataset_exists():
    raw_dir = get_raw_dataset_dir()
    has_images = (raw_dir / "images").exists() or (raw_dir / "train" / "images").exists()
    has_labels = (raw_dir / "labels").exists() or (raw_dir / "train" / "labels").exists()
    assert has_images, "Images missing from raw dataset"
    assert has_labels, "Labels missing from raw dataset"


def test_raw_dataset_verification():
    raw_dir = get_raw_dataset_dir()
    report = verify_dataset(raw_dir)
    assert report["overall_valid"] is True, f"Verification failed with issues: {report['issues']}"
    assert report["image_count"] > 0, "No images found in dataset"
    assert report["label_count"] == report["image_count"], "Image and label count mismatch"
