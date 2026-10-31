"""Download DsPCBSD+ to the project data directory using the active venv."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DSPCBSD_RAW_DIR, ensure_project_dirs  # noqa: E402

KAGGLE_DATASET = "enisteper1/dataset-of-pcb-surface-defects-dspcbsd"


def download(destination: Path = DSPCBSD_RAW_DIR, zip_path: Path | None = None) -> Path:
    ensure_project_dirs()
    destination.mkdir(parents=True, exist_ok=True)
    if zip_path:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(destination)
        return destination
    if shutil.which("kaggle") is None and not (Path(sys.executable).parent / "kaggle").exists():
        raise RuntimeError("Kaggle CLI is unavailable. Install requirements in .venv and configure ~/.kaggle/kaggle.json.")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(destination), "--unzip"],
        check=True,
    )
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, help="Existing downloaded archive; avoids Kaggle access.")
    parser.add_argument("--destination", type=Path, default=DSPCBSD_RAW_DIR)
    args = parser.parse_args()
    print(download(args.destination, args.zip))
