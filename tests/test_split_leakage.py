"""
Unit tests for split leakage detection and manifest verification.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import get_processed_dataset_dir, get_reports_dir
import importlib
split_module = importlib.import_module("scripts.03_train_test_split")
verify_no_leakage = split_module.verify_no_leakage


def test_processed_split_no_leakage():
    processed_dir = get_processed_dataset_dir()
    assert processed_dir.exists(), "Processed dataset directory does not exist"

    def get_split_stems(split_name, alt_name=None):
        p1 = processed_dir / split_name / "images"
        p2 = processed_dir / "images" / split_name
        if alt_name:
            p3 = processed_dir / alt_name / "images"
            p4 = processed_dir / "images" / alt_name
            for p in [p1, p2, p3, p4]:
                if p.exists():
                    return {f.stem for f in p.glob("*") if f.is_file()}
        for p in [p1, p2]:
            if p.exists():
                return {f.stem for f in p.glob("*") if f.is_file()}
        return set()

    train_imgs = get_split_stems("train")
    val_imgs = get_split_stems("val", alt_name="valid")
    test_imgs = get_split_stems("test")

    splits = {
        "train": list(train_imgs),
        "val": list(val_imgs),
        "test": list(test_imgs),
    }

    assert len(train_imgs) > 0, "Train split is empty"
    assert len(val_imgs) > 0, "Val split is empty"
    assert len(test_imgs) > 0, "Test split is empty"

    assert verify_no_leakage(splits) is True


def test_split_manifest_exists():
    manifest_file = get_reports_dir() / "split_manifest.json"
    assert manifest_file.exists(), f"Split manifest not found at {manifest_file}"
