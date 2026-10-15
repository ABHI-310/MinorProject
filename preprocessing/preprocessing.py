"""
preprocessing.py
================
Image Preprocessing and Coordinate Transformation Utilities.

Provides:
  - Letterbox resizing with aspect-ratio preservation and padding.
  - Coordinate adjustment for YOLO bounding boxes under letterboxing.
  - Image format and color space verification.
  - Batch preprocessing and validation.

Ultralytics performs online augmentation (mosaic, flips, HSV) during training;
this module ensures raw images conform strictly to input geometries and verifies
geometric transformations.

Usage:
    python preprocessing/preprocessing.py --test-demo
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.common import (
    CLASS_NAMES,
    parse_yolo_label,
    setup_logging,
    validate_annotation,
)

logger = setup_logging("preprocessing")


def letterbox_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image to a 32-pixel-multiple rectangle with letterbox padding.

    Args:
        image: Source image (H, W, C).
        target_size: Target (width, height).
        color: Padding fill color (default gray 114).
        auto: Minimum rectangle padding.
        scale_fill: Stretch to fit target_size without padding.
        scale_up: Scale up if smaller than target_size.

    Returns:
        (padded_image, scale_ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (target_size[1], target_size[0])  # [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scale_fill:  # stretch
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return padded_img, r, (dw, dh)


def transform_yolo_bbox_to_letterbox(
    ann: dict,
    orig_shape: Tuple[int, int],
    target_size: Tuple[int, int],
    scale_ratio: float,
    pad: Tuple[float, float],
) -> dict:
    """
    Transform normalized YOLO bounding box coordinates from original image
    to letterboxed image.

    Args:
        ann: Annotation dict with keys x_center, y_center, width, height (normalized 0..1).
        orig_shape: (orig_h, orig_w).
        target_size: (target_w, target_h).
        scale_ratio: Resizing scale factor applied.
        pad: (pad_w, pad_h) in pixels.

    Returns:
        New dict with adjusted normalized coordinates.
    """
    orig_h, orig_w = orig_shape
    target_w, target_h = target_size
    pad_w, pad_h = pad

    # Un-normalize to original pixel coordinates
    orig_xc = ann["x_center"] * orig_w
    orig_yc = ann["y_center"] * orig_h
    orig_bw = ann["width"] * orig_w
    orig_bh = ann["height"] * orig_h

    # Scale and apply padding
    new_xc = orig_xc * scale_ratio + pad_w
    new_yc = orig_yc * scale_ratio + pad_h
    new_bw = orig_bw * scale_ratio
    new_bh = orig_bh * scale_ratio

    # Re-normalize to target dimensions
    norm_xc = np.clip(new_xc / target_w, 0.0, 1.0)
    norm_yc = np.clip(new_yc / target_h, 0.0, 1.0)
    norm_bw = np.clip(new_bw / target_w, 0.0, 1.0)
    norm_bh = np.clip(new_bh / target_h, 0.0, 1.0)

    return {
        "class_id": ann["class_id"],
        "x_center": float(norm_xc),
        "y_center": float(norm_yc),
        "width": float(norm_bw),
        "height": float(norm_bh),
    }


def validate_image_tensor(image: np.ndarray) -> List[str]:
    """Check image channels, depth, and values."""
    issues = []
    if image is None:
        return ["Image is None"]
    if len(image.shape) != 3 or image.shape[2] != 3:
        issues.append(f"Image does not have 3 channels (shape: {image.shape})")
    if image.dtype != np.uint8:
        issues.append(f"Expected uint8 dtype, got {image.dtype}")
    return issues


def run_demo() -> None:
    """Demonstrate letterboxing and coordinate transforms with assertions."""
    print("Running preprocessing verification demo...")
    # Synthetic test image (300 x 400)
    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
    test_img[:] = (200, 100, 50)

    # Box centered at 0.5, 0.5 with width 0.2, height 0.2
    dummy_ann = {
        "class_id": 0,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.2,
        "height": 0.2,
    }

    padded, r, (dw, dh) = letterbox_image(test_img, target_size=(640, 640))
    assert padded.shape == (640, 640, 3), f"Expected (640, 640, 3), got {padded.shape}"

    transformed = transform_yolo_bbox_to_letterbox(
        dummy_ann, (300, 400), (640, 640), r, (dw, dh)
    )

    # Center must remain exactly 0.5 because padding is symmetric
    assert abs(transformed["x_center"] - 0.5) < 1e-3, f"Expected 0.5, got {transformed['x_center']}"
    assert abs(transformed["y_center"] - 0.5) < 1e-3, f"Expected 0.5, got {transformed['y_center']}"
    assert transformed["width"] > 0 and transformed["height"] > 0

    print("[OK] Preprocessing letterbox and coordinate transform assertions passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-demo", action="store_true", help="Run self-test demo")
    args = parser.parse_args()

    if args.test_demo:
        run_demo()
    else:
        run_demo()
