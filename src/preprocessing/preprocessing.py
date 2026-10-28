"""OpenCV preprocessing for PKU-Market-PCB (letterbox to 640, train-time extra augs)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    PKU_PROCESSED_DIR,
    PKU_PROCESSED_YAML,
    PKU_SPLIT_IMAGES,
    PKU_SPLIT_LABELS,
    YOLO_IMGSZ,
    ensure_project_dirs,
)
from preprocessing.augmentation import augment_sample  # noqa: E402
from preprocessing.yolo_io import (  # noqa: E402
    iter_images,
    letterbox,
    load_yolo_labels,
    remap_labels_letterbox,
    save_yolo_labels,
    write_yolo_yaml,
)


def _label_path_for(image_path: Path, labels_root: Path, split: str) -> Path:
    return labels_root / split / f"{image_path.stem}.txt"


def _process_image(
    image_path: Path,
    label_path: Path,
    out_image: Path,
    out_label: Path,
    augment: bool,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    labels = load_yolo_labels(label_path)

    canvas, scale, dx, dy = letterbox(image, new_size=YOLO_IMGSZ)
    labels = remap_labels_letterbox(labels, w, h, scale, dx, dy, YOLO_IMGSZ)

    if augment:
        canvas, labels = augment_sample(canvas, labels)

    out_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_image), out_bgr)
    save_yolo_labels(out_label, labels)


def preprocess_split_dataset(
    images_root: Path = PKU_SPLIT_IMAGES,
    labels_root: Path = PKU_SPLIT_LABELS,
    output_dir: Path = PKU_PROCESSED_DIR,
    extra_train_augs: int = 2,
) -> Path:
    """Letterbox all splits to 640×640. Extra augmented copies for train only.

    Images are stored as uint8 JPEG. Pixel [0,1] normalization is left to
    Ultralytics so YOLO loaders receive valid image files.
    """
    ensure_project_dirs()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        split_images = images_root / split
        paths = iter_images(split_images)
        if not paths:
            raise FileNotFoundError(
                f"No images in {split_images}. Run scripts/03_train_test_split.py first."
            )
        for image_path in tqdm(paths, desc=f"preprocess {split}"):
            src_label = _label_path_for(image_path, labels_root, split)
            dst_image = output_dir / "images" / split / image_path.name
            dst_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
            _process_image(
                image_path,
                src_label,
                dst_image,
                dst_label,
                augment=False,
            )
            if split == "train":
                for i in range(extra_train_augs):
                    aug_name = f"{image_path.stem}_aug{i}{image_path.suffix}"
                    _process_image(
                        image_path,
                        src_label,
                        output_dir / "images" / split / aug_name,
                        output_dir / "labels" / split / f"{image_path.stem}_aug{i}.txt",
                        augment=True,
                    )

    write_yolo_yaml(PKU_PROCESSED_YAML, output_dir)
    return PKU_PROCESSED_YAML


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PKU splits for YOLOv8.")
    parser.add_argument("--extra-train-augs", type=int, default=2)
    args = parser.parse_args()
    yaml_path = preprocess_split_dataset(extra_train_augs=args.extra_train_augs)
    print(f"Wrote processed dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
