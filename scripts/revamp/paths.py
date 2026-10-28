"""Resolve pre-cleanup evidence without editing its immutable contents."""
import json
from pathlib import Path

def historical_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if path.exists():
        return path
    mapping_path = root / "configs/legacy_paths.json"
    if not mapping_path.exists():
        return path
    mapping = json.loads(mapping_path.read_text())
    relative = Path(relative).as_posix()
    for old, new in sorted(mapping.items(),key=lambda item:len(item[0]),reverse=True):
        if relative == old or relative.startswith(old + "/"):
            candidate = root / new / relative[len(old):].lstrip("/")
            if not candidate.resolve().is_relative_to(root):
                raise ValueError("Historical path escapes repository")
            return candidate
    return path
