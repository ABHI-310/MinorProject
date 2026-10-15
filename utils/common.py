"""
PCB Smart Maintenance - Shared Utilities
=========================================
Common helper functions used across all Phase 1 scripts.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Optional

import yaml


# =============================================================================
# Path Helpers
# =============================================================================

def get_project_root() -> Path:
    """Return the project root directory (pcb-smart-maintenance/)."""
    # Works whether called from scripts/, train/, evaluation/, etc.
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the data/ directory path."""
    return get_project_root() / "data"


def get_raw_dataset_dir() -> Path:
    """Return the raw dataset directory path."""
    return get_data_dir() / "pku_market"


def get_processed_dataset_dir() -> Path:
    """Return the processed dataset directory path."""
    return get_data_dir() / "pku_market_processed"


def get_reports_dir() -> Path:
    """Return the reports/ directory path."""
    d = get_project_root() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_configs_dir() -> Path:
    """Return the configs/ directory path."""
    return get_project_root() / "configs"


def get_runs_dir() -> Path:
    """Return the runs/ directory path."""
    d = get_project_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Configuration Loading
# =============================================================================

def load_yaml_config(config_path: Path) -> dict:
    """Load a YAML configuration file and return as dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    return config


def load_train_config() -> dict:
    """Load the training configuration."""
    return load_yaml_config(get_configs_dir() / "train_config.yaml")


def load_data_config() -> dict:
    """Load the YOLO data configuration."""
    return load_yaml_config(get_configs_dir() / "data.yaml")


def save_yaml_config(config: dict, config_path: Path) -> None:
    """Save a dict to a YAML file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Set up a logger with console output and optional file output.

    Args:
        name: Logger name (typically the script name).
        level: Logging level.
        log_file: Optional path to write log output to a file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# YOLO Label Parsing and Validation
# =============================================================================

# Class names matching data.yaml order
CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

NUM_CLASSES = len(CLASS_NAMES)


def parse_yolo_label(label_path: Path) -> list[dict]:
    """
    Parse a YOLO-format label file.

    Each line should contain: class_id x_center y_center width height
    All coordinates are normalized to [0, 1].

    Args:
        label_path: Path to the .txt label file.

    Returns:
        List of dicts, each with keys: class_id, x_center, y_center, width, height, raw_line, line_num.
        If the file is empty, returns an empty list.
    """
    annotations = []
    if not label_path.exists():
        return annotations

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            annotations.append({
                "raw_line": line,
                "line_num": line_num,
            })

            parts = line.split()
            if len(parts) != 5:
                annotations[-1]["error"] = f"Expected 5 fields, got {len(parts)}"
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError as e:
                annotations[-1]["error"] = f"Numeric parse error: {e}"
                continue

            annotations[-1].update({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            })

    return annotations


def validate_annotation(ann: dict) -> list[str]:
    """
    Validate a single parsed YOLO annotation dict.

    Args:
        ann: Dict from parse_yolo_label().

    Returns:
        List of error/warning strings. Empty list means valid.
    """
    issues = []

    # Check for parse errors
    if "error" in ann:
        issues.append(f"Line {ann['line_num']}: {ann['error']}")
        return issues

    # Check required fields exist
    if "class_id" not in ann:
        issues.append(f"Line {ann['line_num']}: Missing parsed fields")
        return issues

    # Class ID range
    if ann["class_id"] < 0 or ann["class_id"] >= NUM_CLASSES:
        issues.append(
            f"Line {ann['line_num']}: class_id {ann['class_id']} "
            f"out of range [0, {NUM_CLASSES - 1}]"
        )

    # Coordinate bounds [0, 1]
    for coord_name in ["x_center", "y_center", "width", "height"]:
        val = ann[coord_name]
        if val < 0.0 or val > 1.0:
            issues.append(
                f"Line {ann['line_num']}: {coord_name}={val:.6f} out of [0, 1]"
            )

    # Width/height > 0
    if ann["width"] <= 0:
        issues.append(f"Line {ann['line_num']}: width={ann['width']} must be > 0")
    if ann["height"] <= 0:
        issues.append(f"Line {ann['line_num']}: height={ann['height']} must be > 0")

    # Bbox extends beyond image boundary
    x_min = ann["x_center"] - ann["width"] / 2
    y_min = ann["y_center"] - ann["height"] / 2
    x_max = ann["x_center"] + ann["width"] / 2
    y_max = ann["y_center"] + ann["height"] / 2

    tolerance = 0.001  # Small tolerance for floating point
    if x_min < -tolerance or y_min < -tolerance:
        issues.append(
            f"Line {ann['line_num']}: bbox extends beyond top-left "
            f"(x_min={x_min:.4f}, y_min={y_min:.4f})"
        )
    if x_max > 1.0 + tolerance or y_max > 1.0 + tolerance:
        issues.append(
            f"Line {ann['line_num']}: bbox extends beyond bottom-right "
            f"(x_max={x_max:.4f}, y_max={y_max:.4f})"
        )

    return issues


def save_json_report(data: dict, path: Path) -> None:
    """Save a dict as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
