"""YOLO label I/O, VOC conversion, and letterbox geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

from config import (
    CLASS_NAMES,
    CLASS_TO_ID,
    IMAGE_EXTENSIONS,
    NUM_CLASSES,
    YOLO_IMGSZ,
    normalize_class_name,
)

Label = tuple[int, float, float, float, float]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if is_image_file(p))


def load_yolo_labels(label_path: Path) -> list[Label]:
    if not label_path.exists():
        return []
    labels: list[Label] = []
    for raw in label_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO line in {label_path}: {line!r}")
        class_id, x, y, w, h = int(float(parts[0])), *map(float, parts[1:])
        labels.append((class_id, x, y, w, h))
    return labels


def save_yolo_labels(label_path: Path, labels: list[Label]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def yolo_to_xyxy(label: Label, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    _, xc, yc, w, h = label
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def xyxy_to_yolo(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> Label | None:
    x1 = float(np.clip(x1, 0, img_w))
    y1 = float(np.clip(y1, 0, img_h))
    x2 = float(np.clip(x2, 0, img_w))
    y2 = float(np.clip(y2, 0, img_h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    return (class_id, xc, yc, bw / img_w, bh / img_h)


def letterbox(
    image: np.ndarray,
    new_size: int = YOLO_IMGSZ,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    dx = (new_size - new_w) // 2
    dy = (new_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas, scale, dx, dy


def remap_labels_letterbox(
    labels: list[Label],
    img_w: int,
    img_h: int,
    scale: float,
    dx: int,
    dy: int,
    new_size: int,
) -> list[Label]:
    remapped: list[Label] = []
    for label in labels:
        class_id = label[0]
        x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)
        x1 = x1 * scale + dx
        y1 = y1 * scale + dy
        x2 = x2 * scale + dx
        y2 = y2 * scale + dy
        converted = xyxy_to_yolo(class_id, x1, y1, x2, y2, new_size, new_size)
        if converted is not None:
            remapped.append(converted)
    return remapped


def voc_xml_to_yolo_labels(xml_path: Path, img_w: int, img_h: int) -> list[Label]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None:
        w_node, h_node = size.findtext("width"), size.findtext("height")
        if w_node and h_node:
            img_w, img_h = int(float(w_node)), int(float(h_node))

    labels: list[Label] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            continue
        class_name = normalize_class_name(name)
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        converted = xyxy_to_yolo(CLASS_TO_ID[class_name], xmin, ymin, xmax, ymax, img_w, img_h)
        if converted is not None:
            labels.append(converted)
    return labels


def write_yolo_yaml(yaml_path: Path, dataset_root: Path) -> None:
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def primary_class_id(labels: list[Label]) -> int:
    if not labels:
        return 0
    counts = {}
    for class_id, *_ in labels:
        counts[class_id] = counts.get(class_id, 0) + 1
    return max(counts, key=counts.get)
