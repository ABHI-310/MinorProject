"""Download PKU-Market-PCB and convert it to a flat YOLO image/label pool."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PKU_ALL_IMAGES, PKU_ALL_LABELS, PKU_DIR, PKU_RAW_DIR, ensure_project_dirs  # noqa: E402
from preprocessing.yolo_io import (  # noqa: E402
    iter_images,
    load_yolo_labels,
    save_yolo_labels,
    voc_xml_to_yolo_labels,
)

KAGGLE_DATASET = "akhatova/pcb-defects"
ROBOFLOW_WORKSPACE = "jr-mqqnk"
ROBOFLOW_PROJECT = "pcb-defects-detection-anddl"
ROBOFLOW_VERSION = 1


def _copy_unique(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and dest.resolve() != src.resolve():
        dest = dest_dir / f"{src.stem}_{src.parent.name}{src.suffix}"
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    return dest


def _xml_index(raw_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for xml_path in raw_root.rglob("*.xml"):
        index[xml_path.stem] = xml_path
    return index


def _yolo_label_nearby(image_path: Path) -> Path | None:
    stem = image_path.stem
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent / f"{stem}.txt",
        image_path.parent.parent / "labels" / f"{stem}.txt",
        image_path.parent.parent / "labels" / image_path.parent.name / f"{stem}.txt",
    ]
    if "images" in image_path.parts:
        idx = image_path.parts.index("images")
        prefix_parts = image_path.parts[:idx]
        suffix_parts = image_path.parts[idx + 1 : -1]
        mapped = Path(*prefix_parts) if prefix_parts else Path(".")
        mapped = mapped / "labels"
        if suffix_parts:
            mapped = mapped.joinpath(*suffix_parts)
        candidates.append(mapped / f"{stem}.txt")
    if image_path.parent.name == "images":
        candidates.append(image_path.parent.parent / "labels" / f"{stem}.txt")
    for path in candidates:
        if path.exists():
            return path
    return None


def convert_to_yolo_pool(raw_root: Path, images_out: Path = PKU_ALL_IMAGES, labels_out: Path = PKU_ALL_LABELS) -> int:
    if images_out.exists():
        shutil.rmtree(images_out)
    if labels_out.exists():
        shutil.rmtree(labels_out)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    xml_by_stem = _xml_index(raw_root)
    images = iter_images(raw_root)
    converted = 0
    missing_labels = 0

    for image_path in tqdm(images, desc="convert to YOLO"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        labels = []
        xml_path = xml_by_stem.get(image_path.stem)
        yolo_path = _yolo_label_nearby(image_path)
        if xml_path is not None:
            labels = voc_xml_to_yolo_labels(xml_path, w, h)
        elif yolo_path is not None:
            labels = load_yolo_labels(yolo_path)

        dest_image = _copy_unique(image_path, images_out)
        dest_label = labels_out / f"{dest_image.stem}.txt"
        save_yolo_labels(dest_label, labels)
        converted += 1
        if not labels:
            missing_labels += 1

    print(f"Converted {converted} images into {images_out}")
    if missing_labels:
        print(f"Warning: {missing_labels} images had empty labels")
    if converted == 0:
        raise RuntimeError(f"No images found under {raw_root}")
    return converted


def _unzip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def download_kaggle(dest: Path, dataset: str = KAGGLE_DATASET) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_roboflow(dest: Path, api_key: str) -> None:
    from roboflow import Roboflow

    dest.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    project.version(ROBOFLOW_VERSION).download("yolov8", location=str(dest))


def download_url(url: str, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", url, "-o", str(dest_zip)], check=True)


def ingest(
    *,
    zip_path: Path | None,
    url: str | None,
    source: str,
    roboflow_api_key: str | None,
    kaggle_dataset: str,
    skip_download: bool,
) -> None:
    ensure_project_dirs()
    raw = PKU_RAW_DIR
    if skip_download:
        if not raw.exists():
            raise SystemExit(f"Missing {raw}. Download first or pass --zip / --source.")
    else:
        if raw.exists():
            shutil.rmtree(raw)
        raw.mkdir(parents=True, exist_ok=True)

        if zip_path:
            _unzip(zip_path, raw)
        elif url:
            dest_zip = PKU_DIR / "download.zip"
            download_url(url, dest_zip)
            _unzip(dest_zip, raw)
        elif source in {"auto", "roboflow"} and (roboflow_api_key or os.getenv("ROBOFLOW_API_KEY")):
            key = roboflow_api_key or os.environ["ROBOFLOW_API_KEY"]
            download_roboflow(raw, key)
        elif source in {"auto", "kaggle"}:
            try:
                download_kaggle(raw, kaggle_dataset)
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                if source == "kaggle":
                    raise
                raise SystemExit(
                    "Automatic download failed. Provide one of:\n"
                    "  --zip /path/to/pku.zip\n"
                    "  --url https://...\n"
                    "  --source kaggle  (needs ~/.kaggle/kaggle.json)\n"
                    "  --source roboflow --roboflow-api-key KEY\n"
                    f"Original error: {exc}"
                ) from exc
        else:
            raise SystemExit("No download source selected.")

    convert_to_yolo_pool(raw)
    print(f"YOLO pool ready: {PKU_ALL_IMAGES} and {PKU_ALL_LABELS}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize PKU-Market-PCB.")
    parser.add_argument("--source", choices=["auto", "kaggle", "roboflow"], default="auto")
    parser.add_argument("--zip", type=Path, dest="zip_path")
    parser.add_argument("--url")
    parser.add_argument("--roboflow-api-key")
    parser.add_argument("--kaggle-dataset", default=KAGGLE_DATASET)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Convert existing files in data/pku_market/raw without downloading.",
    )
    args = parser.parse_args()
    ingest(
        zip_path=args.zip_path,
        url=args.url,
        source=args.source,
        roboflow_api_key=args.roboflow_api_key,
        kaggle_dataset=args.kaggle_dataset,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
