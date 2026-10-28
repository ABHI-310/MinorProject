"""Project paths and PKU-Market-PCB class definitions."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
PKU_DIR = DATA_DIR / "pku_market"
PKU_RAW_DIR = PKU_DIR / "raw"
PKU_ALL_IMAGES = PKU_DIR / "all" / "images"
PKU_ALL_LABELS = PKU_DIR / "all" / "labels"
PKU_SPLIT_IMAGES = PKU_DIR / "images"
PKU_SPLIT_LABELS = PKU_DIR / "labels"
PKU_YAML = PKU_DIR / "data.yaml"

PKU_PROCESSED_DIR = DATA_DIR / "pku_market_processed"
PKU_PROCESSED_YAML = PKU_PROCESSED_DIR / "data.yaml"

WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"

DSPCBSD_DIR = DATA_DIR / "dspcbsd"
DSPCBSD_RAW_DIR = DSPCBSD_DIR / "raw"
DSPCBSD_IMAGES = DSPCBSD_DIR / "images"
DSPCBSD_LABELS = DSPCBSD_DIR / "labels"
DSPCBSD_FEW_SHOT_DIR = DSPCBSD_DIR / "labelled_few_shot"
DSPCBSD_UNLABELLED_DIR = DSPCBSD_DIR / "unlabelled"
DSPCBSD_TEST_DIR = DSPCBSD_DIR / "test"
DSPCBSD_MAPPING_PATH = DSPCBSD_DIR / "class_mapping.json"
RUL_DIR = DATA_DIR / "phase3_rul"

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open",
    "short",
    "spur",
    "spurious_copper",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# PKU VOC / folder names → canonical YOLO class names
CLASS_ALIASES = {
    "missing_hole": "missing_hole",
    "missing hole": "missing_hole",
    "missinghole": "missing_hole",
    "mouse_bite": "mouse_bite",
    "mouse bite": "mouse_bite",
    "mousebite": "mouse_bite",
    "open": "open",
    "open_circuit": "open",
    "open circuit": "open",
    "opencircuit": "open",
    "short": "short",
    "spur": "spur",
    "spurious_copper": "spurious_copper",
    "spurious copper": "spurious_copper",
    "spuriouscopper": "spurious_copper",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

YOLO_IMGSZ = 640
NUM_CLASSES = len(CLASS_NAMES)


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        PKU_DIR,
        PKU_RAW_DIR,
        PKU_ALL_IMAGES,
        PKU_ALL_LABELS,
        DSPCBSD_DIR,
        DSPCBSD_RAW_DIR,
        RUL_DIR,
        WEIGHTS_DIR,
        RUNS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def normalize_class_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = " ".join(key.split())
    collapsed = key.replace(" ", "_")
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if collapsed in CLASS_ALIASES:
        return CLASS_ALIASES[collapsed]
    raise ValueError(f"Unknown PKU class name: {name!r}")
