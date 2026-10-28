from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.yolo_io import letterbox, remap_labels_letterbox, yolo_to_xyxy


def test_letterbox_box_stays_inside_canvas():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    labels = [(0, 0.5, 0.5, 0.25, 0.4)]
    canvas, scale, dx, dy = letterbox(image, 640)
    assert canvas.shape == (640, 640, 3)
    remapped = remap_labels_letterbox(labels, 200, 100, scale, dx, dy, 640)
    assert len(remapped) == 1
    x1, y1, x2, y2 = yolo_to_xyxy(remapped[0], 640, 640)
    assert 0 <= x1 < x2 <= 640
    assert 0 <= y1 < y2 <= 640
    xc, yc = remapped[0][1] * 640, remapped[0][2] * 640
    assert abs(xc - 320) < 5
    assert abs(yc - 320) < 5


if __name__ == "__main__":
    test_letterbox_box_stays_inside_canvas()
    print("ok")
