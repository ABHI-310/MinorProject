"""
Unit tests for YOLO label parsing and validation.
"""

import sys
from pathlib import Path
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    CLASS_NAMES,
    NUM_CLASSES,
    parse_yolo_label,
    validate_annotation,
)


def test_parse_valid_label():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        tmp.write("0 0.500000 0.500000 0.200000 0.300000\n")
        tmp.write("3 0.250000 0.750000 0.100000 0.150000\n")
        tmp_path = Path(tmp.name)

    try:
        anns = parse_yolo_label(tmp_path)
        assert len(anns) == 2
        assert anns[0]["class_id"] == 0
        assert anns[0]["x_center"] == 0.5
        assert anns[0]["width"] == 0.2

        assert anns[1]["class_id"] == 3
        assert anns[1]["y_center"] == 0.75

        for a in anns:
            issues = validate_annotation(a)
            assert len(issues) == 0, f"Expected valid annotation, got: {issues}"
    finally:
        tmp_path.unlink()


def test_empty_label_file():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tmp:
        tmp_path = Path(tmp.name)

    try:
        anns = parse_yolo_label(tmp_path)
        assert len(anns) == 0
    finally:
        tmp_path.unlink()


def test_invalid_class_id():
    ann = {
        "class_id": 99,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
        "line_num": 1,
    }
    issues = validate_annotation(ann)
    assert any("out of range" in issue for issue in issues)


def test_out_of_bounds_coords():
    ann = {
        "class_id": 0,
        "x_center": 1.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
        "line_num": 1,
    }
    issues = validate_annotation(ann)
    assert any("out of [0, 1]" in issue for issue in issues)
