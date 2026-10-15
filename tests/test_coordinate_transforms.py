"""
Unit tests for letterboxing and bounding box coordinate transforms.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.preprocessing import (
    letterbox_image,
    transform_yolo_bbox_to_letterbox,
    validate_image_tensor,
)


def test_letterbox_dimensions():
    # Square aspect ratio test
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, r, (dw, dh) = letterbox_image(img, target_size=(640, 640))

    assert padded.shape == (640, 640, 3)
    assert r == 1.0  # 640/640
    assert dw == 0.0
    assert dh == 80.0  # (640 - 480) / 2


def test_bbox_transform_symmetry():
    # Centered box in a 400x400 image resized to 640x640
    ann = {
        "class_id": 1,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
    }
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    padded, r, (dw, dh) = letterbox_image(img, target_size=(640, 640))
    transformed = transform_yolo_bbox_to_letterbox(ann, (400, 400), (640, 640), r, (dw, dh))

    assert abs(transformed["x_center"] - 0.5) < 1e-4
    assert abs(transformed["y_center"] - 0.5) < 1e-4
    assert 0.0 < transformed["width"] <= 1.0
    assert 0.0 < transformed["height"] <= 1.0


def test_validate_image_tensor():
    valid_img = np.zeros((640, 640, 3), dtype=np.uint8)
    assert len(validate_image_tensor(valid_img)) == 0

    invalid_channels = np.zeros((640, 640), dtype=np.uint8)
    assert len(validate_image_tensor(invalid_channels)) > 0
