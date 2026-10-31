"""Convert mapped DsPCBSD+ labels and create labelled few-shot/test/unlabelled pools."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DSPCBSD_FEW_SHOT_DIR, DSPCBSD_MAPPING_PATH, DSPCBSD_RAW_DIR, DSPCBSD_TEST_DIR, DSPCBSD_UNLABELLED_DIR, REPORTS_DIR  # noqa: E402
from preprocessing.yolo_io import iter_images, load_yolo_labels, save_yolo_labels  # noqa: E402


def _locate_yolo_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image in iter_images(raw_dir):
        label = image.with_suffix(".txt")
        if not label.exists() and "images" in image.parts:
            idx = image.parts.index("images")
            label = Path(*image.parts[:idx], "labels", *image.parts[idx + 1 :]).with_suffix(".txt")
        if label.exists():
            pairs.append((image, label))
    return pairs


def _mapped_labels(label_path: Path, index_mapping: dict[int, int]) -> list[tuple[int, float, float, float, float]]:
    return [(index_mapping[class_id], x, y, w, h) for class_id, x, y, w, h in load_yolo_labels(label_path) if class_id in index_mapping]


def _copy_pair(image: Path, labels, destination: Path, ordinal: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{ordinal:06d}_{image.stem}"
    shutil.copy2(image, destination / f"{stem}{image.suffix.lower()}")
    save_yolo_labels(destination / f"{stem}.txt", labels)


def sample_target_images(raw_dir: Path, mapping_path: Path, few_shot: int = 90, test_size: int = 300, seed: int = 42) -> dict:
    mapping = json.loads(mapping_path.read_text())
    index_mapping = {entry["target_id"]: entry["source_id"] for entry in mapping["mapping"] if entry["source_id"] is not None}
    samples = []
    for image, label_path in _locate_yolo_pairs(raw_dir):
        labels = _mapped_labels(label_path, index_mapping)
        if labels:
            primary = Counter(label[0] for label in labels).most_common(1)[0][0]
            samples.append((image, labels, primary))
    if len(samples) < few_shot + test_size:
        raise ValueError(f"Only {len(samples)} mapped labelled images; need {few_shot + test_size}.")
    random.Random(seed).shuffle(samples)
    by_class = defaultdict(list)
    for item in samples:
        by_class[item[2]].append(item)
    chosen = []
    while len(chosen) < few_shot and any(by_class.values()):
        for class_id in sorted(by_class):
            if by_class[class_id] and len(chosen) < few_shot:
                chosen.append(by_class[class_id].pop())
    remainder = [item for items in by_class.values() for item in items]
    random.Random(seed).shuffle(remainder)
    test, unlabelled = remainder[:test_size], remainder[test_size:]
    for folder in (DSPCBSD_FEW_SHOT_DIR, DSPCBSD_TEST_DIR, DSPCBSD_UNLABELLED_DIR):
        if folder.exists():
            shutil.rmtree(folder)
    for idx, (image, labels, _) in enumerate(chosen): _copy_pair(image, labels, DSPCBSD_FEW_SHOT_DIR, idx)
    for idx, (image, labels, _) in enumerate(test): _copy_pair(image, labels, DSPCBSD_TEST_DIR, idx)
    for idx, (image, labels, _) in enumerate(unlabelled): _copy_pair(image, labels, DSPCBSD_UNLABELLED_DIR, idx)
    stats = {"few_shot_images": len(chosen), "test_images": len(test), "unlabelled_images": len(unlabelled), "few_shot_class_counts": dict(Counter(x[2] for x in chosen))}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "target_sampling_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DSPCBSD_RAW_DIR)
    parser.add_argument("--mapping", type=Path, default=DSPCBSD_MAPPING_PATH)
    parser.add_argument("--few-shot", type=int, default=90)
    parser.add_argument("--test-size", type=int, default=300)
    args = parser.parse_args()
    print(sample_target_images(args.raw_dir, args.mapping, args.few_shot, args.test_size))
