"""
05_class_mapping.py
Defines and executes the semantic alignment between the 9 defect classes of the target domain
(DsPCBSD+) and the 6 defect classes of the source domain (PKU-Market-PCB).
Generates data/dspcbsd/class_mapping.json.
"""

import argparse
import json
from pathlib import Path

PKU_CLASSES = [
    "missing_hole",     # 0
    "mouse_bite",       # 1
    "open",             # 2
    "short",            # 3
    "spur",             # 4
    "spurious_copper",  # 5
]

DSPCBSD_CLASSES = [
    "missing_hole",         # 0
    "mouse_bite",           # 1
    "open_circuit",         # 2
    "short",                # 3
    "spur",                 # 4
    "spurious_copper",      # 5
    "pin_hole",             # 6
    "scratch",              # 7
    "solder_joint_defect",  # 8
]

# Mapping rationale based on defect morphology and IPC-A-610 standards:
# - missing_hole (0) & pin_hole (6) -> Class 0 (hole/void defects)
# - mouse_bite (1) -> Class 1 (edge breakdown/nibbling)
# - open_circuit (2) & scratch (7) -> Class 2 (trace disconnection/interruption)
# - short (3) & solder_joint_defect (8) -> Class 3 (bridging / solder defect)
# - spur (4) -> Class 4 (copper protrusion)
# - spurious_copper (5) -> Class 5 (isolated excess copper)
TARGET_TO_SOURCE_MAPPING = {
    0: {"name": "missing_hole", "mapped_to_id": 0, "mapped_to_name": "missing_hole", "rationale": "Direct 1-to-1 match"},
    1: {"name": "mouse_bite", "mapped_to_id": 1, "mapped_to_name": "mouse_bite", "rationale": "Direct 1-to-1 match"},
    2: {"name": "open_circuit", "mapped_to_id": 2, "mapped_to_name": "open", "rationale": "Direct equivalent nomenclature"},
    3: {"name": "short", "mapped_to_id": 3, "mapped_to_name": "short", "rationale": "Direct 1-to-1 match"},
    4: {"name": "spur", "mapped_to_id": 4, "mapped_to_name": "spur", "rationale": "Direct 1-to-1 match"},
    5: {"name": "spurious_copper", "mapped_to_id": 5, "mapped_to_name": "spurious_copper", "rationale": "Direct 1-to-1 match"},
    6: {"name": "pin_hole", "mapped_to_id": 0, "mapped_to_name": "missing_hole", "rationale": "Through-hole drilling & void defect group"},
    7: {"name": "scratch", "mapped_to_id": 2, "mapped_to_name": "open", "rationale": "Surface trace severance & open-circuit precursor"},
    8: {"name": "solder_joint_defect", "mapped_to_id": 3, "mapped_to_name": "short", "rationale": "Solder bridging and excess conductive deposit group"},
}


def create_class_mapping(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_payload = {
        "source_domain": {
            "name": "PKU-Market-PCB",
            "num_classes": len(PKU_CLASSES),
            "classes": PKU_CLASSES,
        },
        "target_domain": {
            "name": "DsPCBSD+",
            "num_classes": len(DSPCBSD_CLASSES),
            "classes": DSPCBSD_CLASSES,
        },
        "mapping": TARGET_TO_SOURCE_MAPPING,
        "class_id_lookup": {
            str(k): v["mapped_to_id"] for k, v in TARGET_TO_SOURCE_MAPPING.items()
        },
    }

    out_file = output_dir / "class_mapping.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(mapping_payload, f, indent=2)

    print(f"✓ Saved class mapping schema to {out_file}")
    print("\nClass Mapping Summary:")
    for target_id, info in TARGET_TO_SOURCE_MAPPING.items():
        print(f"  Target [{target_id}] '{info['name']}' -> Source [{info['mapped_to_id']}] '{info['mapped_to_name']}' ({info['rationale']})")

    return mapping_payload


def main():
    parser = argparse.ArgumentParser(description="Generate DsPCBSD+ to PKU class mapping")
    parser.add_argument("--output-dir", type=str, default="data/dspcbsd")
    args = parser.parse_args()

    create_class_mapping(Path(args.output_dir))


if __name__ == "__main__":
    main()

