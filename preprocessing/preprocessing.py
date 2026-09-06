"""
OpenCV Preprocessing Pipeline for PCB Defect Detection
Includes aspect-ratio-preserving letterbox resize (640x640), normalization,
bounding-box coordinate remapping, and optional augmentation.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
try:
    from .augmentation import PCBAugmentor
except (ImportError, ValueError):
    try:
        from preprocessing.augmentation import PCBAugmentor
    except (ImportError, ValueError):
        from augmentation import PCBAugmentor


class PCBPreprocessor:
    def __init__(self, target_size=(640, 640), fill_value=(114, 114, 114)):
        self.target_size = target_size
        self.fill_value = fill_value
        self.augmentor = PCBAugmentor()

    @staticmethod
    def read_yolo_labels(label_path: Path):
        """Read YOLO formatted label file [cls, x_c, y_c, w, h]"""
        if not label_path.exists():
            return []
        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:5]]
                    boxes.append([cls_id] + coords)
        return boxes

    @staticmethod
    def write_yolo_labels(label_path: Path, bboxes: list):
        """Write YOLO formatted label file"""
        label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as f:
            for box in bboxes:
                cls_id = int(box[0])
                coords = " ".join(f"{coord:.6f}" for coord in box[1:])
                f.write(f"{cls_id} {coords}\n")

    def letterbox(self, image: np.ndarray, bboxes: list):
        """
        Resize image to target_size (640, 640) preserving aspect ratio with padding.
        Corrects normalized YOLO bounding box coordinates to the new canvas.
        """
        target_w, target_h = self.target_size
        orig_h, orig_w = image.shape[:2]

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill_value
        )

        adjusted_bboxes = []
        for box in bboxes:
            cls_id, x_c, y_c, w, h = box
            # Original pixel coordinates
            abs_x = x_c * orig_w
            abs_y = y_c * orig_h
            abs_w = w * orig_w
            abs_h = h * orig_h

            # Scaled and translated
            scaled_x = abs_x * scale + left
            scaled_y = abs_y * scale + top
            scaled_w = abs_w * scale
            scaled_h = abs_h * scale

            # Renormalize to 640x640 canvas
            new_norm_x = scaled_x / target_w
            new_norm_y = scaled_y / target_h
            new_norm_w = scaled_w / target_w
            new_norm_h = scaled_h / target_h

            # Ensure valid bounds
            new_norm_x = max(0.001, min(0.999, new_norm_x))
            new_norm_y = max(0.001, min(0.999, new_norm_y))
            new_norm_w = max(0.001, min(0.999, new_norm_w))
            new_norm_h = max(0.001, min(0.999, new_norm_h))

            adjusted_bboxes.append([cls_id, new_norm_x, new_norm_y, new_norm_w, new_norm_h])

        return canvas, adjusted_bboxes

    def process_single(self, image_path: Path, label_path: Path, augment: bool = False):
        """
        Process single image: verify format, resize with letterbox, and optionally augment.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        bboxes = self.read_yolo_labels(label_path)
        canvas, new_bboxes = self.letterbox(image, bboxes)

        if augment:
            canvas, new_bboxes = self.augmentor.augment(canvas, new_bboxes)

        return canvas, new_bboxes

    def process_dataset(
        self,
        src_dir: Path,
        dst_dir: Path,
        splits=("train", "val", "test"),
        augment_train: bool = True,
    ):
        """
        Process an entire split dataset directory structure.
        """
        src_dir = Path(src_dir)
        dst_dir = Path(dst_dir)

        total_processed = 0

        for split in splits:
            img_src_dir = src_dir / "images" / split
            lbl_src_dir = src_dir / "labels" / split

            # Also check alternative folder layout: src_dir / split / images
            if not img_src_dir.exists():
                img_src_dir = src_dir / split / "images"
                lbl_src_dir = src_dir / split / "labels"

            if not img_src_dir.exists():
                print(f"Split {split} images directory not found at {img_src_dir}, skipping.")
                continue

            img_dst_dir = dst_dir / "images" / split
            lbl_dst_dir = dst_dir / "labels" / split
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            lbl_dst_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(
                list(img_src_dir.glob("*.jpg"))
                + list(img_src_dir.glob("*.png"))
                + list(img_src_dir.glob("*.jpeg"))
            )

            is_train = split == "train" and augment_train
            print(f"Processing {split} split: {len(image_files)} images (Augmentation={is_train})...")

            for img_path in image_files:
                lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
                canvas, bboxes = self.process_single(img_path, lbl_path, augment=False)

                # Save base processed image & labels
                dst_img_path = img_dst_dir / f"{img_path.stem}.jpg"
                dst_lbl_path = lbl_dst_dir / f"{img_path.stem}.txt"
                cv2.imwrite(str(dst_img_path), canvas)
                self.write_yolo_labels(dst_lbl_path, bboxes)
                total_processed += 1

                # If training split, also save an augmented version
                if is_train:
                    aug_canvas, aug_bboxes = self.augmentor.augment(canvas.copy(), bboxes)
                    aug_img_path = img_dst_dir / f"{img_path.stem}_aug.jpg"
                    aug_lbl_path = lbl_dst_dir / f"{img_path.stem}_aug.txt"
                    cv2.imwrite(str(aug_img_path), aug_canvas)
                    self.write_yolo_labels(aug_lbl_path, aug_bboxes)
                    total_processed += 1

        # Generate data.yaml in dst_dir
        class_names = [
            "missing_hole",
            "mouse_bite",
            "open",
            "short",
            "spur",
            "spurious_copper",
        ]
        data_yaml = {
            "path": str(dst_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(class_names),
            "names": class_names,
        }
        with open(dst_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"[OK] Processed {total_processed} images saved to {dst_dir}")
        print(f"[OK] Created configuration at {dst_dir / 'data.yaml'}")
        return total_processed


def main():
    parser = argparse.ArgumentParser(description="OpenCV Preprocessing & Augmentation Pipeline")
    parser.add_argument("--src", type=str, default="data/pku_market", help="Source directory")
    parser.add_argument(
        "--dst", type=str, default="data/pku_market_processed", help="Destination directory"
    )
    parser.add_argument("--test", action="store_true", help="Run self-test with synthetic image")
    args = parser.parse_args()

    preprocessor = PCBPreprocessor()

    if args.test:
        print("Running PCBPreprocessor self-test...")
        dummy_img = np.full((720, 1280, 3), 180, dtype=np.uint8)
        cv2.rectangle(dummy_img, (200, 200), (300, 300), (0, 0, 255), -1)
        # Sample box: class 0, center (250/1280, 250/720), width (100/1280), height (100/720)
        dummy_box = [[0, 250 / 1280, 250 / 720, 100 / 1280, 100 / 720]]

        canvas, adjusted = preprocessor.letterbox(dummy_img, dummy_box)
        assert canvas.shape == (640, 640, 3), f"Expected (640, 640, 3), got {canvas.shape}"
        assert len(adjusted) == 1, "Box lost during letterbox transform"

        aug_canvas, aug_boxes = preprocessor.augmentor.augment(canvas, adjusted)
        assert aug_canvas.shape == (640, 640, 3)
        print("[OK] PCBPreprocessor self-test passed successfully!")
        return

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"Warning: Source directory '{src}' does not exist yet. Run scripts/01_download_pku_dataset.py first.")
        return

    preprocessor.process_dataset(src, dst)


if __name__ == "__main__":
    main()
