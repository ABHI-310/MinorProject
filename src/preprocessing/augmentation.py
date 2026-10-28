"""BBox-aware augmentations for YOLO labels (uint8 images)."""

from __future__ import annotations

import random

import cv2
import numpy as np

from preprocessing.yolo_io import Label, xyxy_to_yolo, yolo_to_xyxy


def _clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    return (
        float(np.clip(x1, 0, w)),
        float(np.clip(y1, 0, h)),
        float(np.clip(x2, 0, w)),
        float(np.clip(y2, 0, h)),
    )


def random_horizontal_flip(
    image: np.ndarray,
    labels: list[Label],
    p: float = 0.5,
) -> tuple[np.ndarray, list[Label]]:
    if random.random() >= p:
        return image, labels
    flipped = cv2.flip(image, 1)
    new_labels = [(c, 1.0 - x, y, w, h) for c, x, y, w, h in labels]
    return flipped, new_labels


def random_brightness(image: np.ndarray, delta: float = 0.2) -> np.ndarray:
    factor = 1.0 + random.uniform(-delta, delta)
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255)
    return adjusted.astype(np.uint8)


def gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_rotation(
    image: np.ndarray,
    labels: list[Label],
    max_degrees: float = 15.0,
) -> tuple[np.ndarray, list[Label]]:
    h, w = image.shape[:2]
    angle = random.uniform(-max_degrees, max_degrees)
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderValue=(114, 114, 114))

    new_labels: list[Label] = []
    for label in labels:
        class_id = label[0]
        x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        ones = np.ones((4, 1), dtype=np.float32)
        pts = np.hstack([corners, ones])
        transformed = matrix.dot(pts.T).T
        nx1, ny1 = transformed[:, 0].min(), transformed[:, 1].min()
        nx2, ny2 = transformed[:, 0].max(), transformed[:, 1].max()
        nx1, ny1, nx2, ny2 = _clip_box(nx1, ny1, nx2, ny2, w, h)
        converted = xyxy_to_yolo(class_id, nx1, ny1, nx2, ny2, w, h)
        if converted is not None:
            new_labels.append(converted)
    return rotated, new_labels


def random_zoom(
    image: np.ndarray,
    labels: list[Label],
    min_zoom: float = 0.8,
    max_zoom: float = 1.2,
) -> tuple[np.ndarray, list[Label]]:
    h, w = image.shape[:2]
    zoom = random.uniform(min_zoom, max_zoom)
    new_w, new_h = max(1, int(w * zoom)), max(1, int(h * zoom))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    if zoom >= 1.0:
        x0 = max(0, (new_w - w) // 2)
        y0 = max(0, (new_h - h) // 2)
        crop = resized[y0 : y0 + h, x0 : x0 + w]
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        ch, cw = crop.shape[:2]
        canvas[:ch, :cw] = crop
        ox, oy = -x0, -y0
        scale_x = new_w / w
        scale_y = new_h / h
    else:
        x0 = (w - new_w) // 2
        y0 = (h - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        ox, oy = x0, y0
        scale_x = new_w / w
        scale_y = new_h / h

    new_labels: list[Label] = []
    for label in labels:
        class_id = label[0]
        x1, y1, x2, y2 = yolo_to_xyxy(label, w, h)
        x1 = x1 * scale_x + ox
        x2 = x2 * scale_x + ox
        y1 = y1 * scale_y + oy
        y2 = y2 * scale_y + oy
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
        converted = xyxy_to_yolo(class_id, x1, y1, x2, y2, w, h)
        if converted is not None:
            new_labels.append(converted)
    return canvas, new_labels


def augment_sample(
    image: np.ndarray,
    labels: list[Label],
    *,
    rotation_deg: float = 15.0,
    flip_p: float = 0.5,
    brightness: float = 0.2,
    noise_sigma: float = 5.0,
    zoom_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[np.ndarray, list[Label]]:
    image, labels = random_rotation(image, labels, max_degrees=rotation_deg)
    image, labels = random_horizontal_flip(image, labels, p=flip_p)
    image, labels = random_zoom(image, labels, min_zoom=zoom_range[0], max_zoom=zoom_range[1])
    image = random_brightness(image, delta=brightness)
    image = gaussian_noise(image, sigma=noise_sigma)
    return image, labels
