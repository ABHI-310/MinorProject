"""Map DsPCBSD+ source labels into the six PKU defect classes.

The source class list is read from a YOLO data.yaml when available.  Unknown
classes are deliberately recorded as ``null`` instead of silently mislabelled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import CLASS_NAMES, DSPCBSD_MAPPING_PATH, DSPCBSD_RAW_DIR  # noqa: E402

# Semantic, reviewable mapping.  Update this table after inspecting the dataset's yaml.
ALIASES = {
    "missing_hole": "missing_hole", "pinhole": "missing_hole", "hole": "missing_hole",
    "mouse_bite": "mouse_bite", "mousebite": "mouse_bite", "bite": "mouse_bite",
    "open": "open", "open_circuit": "open", "broken_trace": "open", "trace_damage": "open",
    "short": "short", "short_circuit": "short", "bridge": "short", "solder_bridge": "short",
    "spur": "spur", "burr": "spur",
    "spurious_copper": "spurious_copper", "copper": "spurious_copper", "extra_copper": "spurious_copper",
    "missing_component": "missing_hole", "misalignment": "spur", "offset": "spur", "scratch": "open",
}


def _normalise(name: str) -> str:
    return "_".join(name.lower().strip().replace("-", " ").split())


def discover_names(raw_dir: Path) -> list[str]:
    yamls = sorted(raw_dir.rglob("*.yaml")) + sorted(raw_dir.rglob("*.yml"))
    for path in yamls:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            names = data.get("names")
            if isinstance(names, dict):
                return [str(names[i]) for i in sorted(names, key=lambda value: int(value))]
            if isinstance(names, list):
                return [str(name) for name in names]
        except (OSError, yaml.YAMLError):
            continue
    return []


def build_mapping(names: list[str]) -> dict:
    entries = []
    for index, name in enumerate(names):
        normalised = _normalise(name)
        mapped = ALIASES.get(normalised)
        entries.append({"target_id": index, "target_class": name, "source_class": mapped, "source_id": CLASS_NAMES.index(mapped) if mapped in CLASS_NAMES else None})
    return {"source_classes": CLASS_NAMES, "target_classes": names, "mapping": entries, "review_required": [entry["target_class"] for entry in entries if entry["source_class"] is None]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DSPCBSD_RAW_DIR)
    parser.add_argument("--names", nargs="*", help="Override class names from dataset YAML.")
    parser.add_argument("--output", type=Path, default=DSPCBSD_MAPPING_PATH)
    args = parser.parse_args()
    names = args.names or discover_names(args.raw_dir)
    if not names:
        raise SystemExit("Could not find a target data.yaml. Pass --names after inspecting the downloaded dataset.")
    mapping = build_mapping(names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Wrote {args.output}; unmapped: {mapping['review_required']}")
