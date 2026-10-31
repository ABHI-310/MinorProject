"""Shared import and subprocess paths for repository entry points."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
for path in reversed(PATHS):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join(PATHS + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))

