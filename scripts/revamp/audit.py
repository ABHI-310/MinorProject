"""Audit existing PCB inputs without modifying data, models, splits or the app.

Run with ``python -m revamp.audit --output reports/revamp/phase1_v1``.
No Ultralytics import, model deserialization, training or network access is needed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/datasets.json"
SPLITS = ("train", "val", "test")
ROLES = {"train": "training", "val": "development_validation", "test": "development_test"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_labels(path: Path, class_ids: set[int]) -> list[list[float]]:
    """Missing != empty. Reject malformed, fractional, nonfinite and off-image boxes."""
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise ValueError(f"line {number}: nonnumeric label") from error
        if len(values) != 5 or not all(math.isfinite(x) for x in values):
            raise ValueError(f"line {number}: expected five finite numbers")
        cls, x, y, w, h = values
        if not cls.is_integer() or int(cls) not in class_ids:
            raise ValueError(f"line {number}: unsupported class {cls}")
        if w <= 0 or h <= 0 or min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
            raise ValueError(f"line {number}: nonpositive or out-of-bounds box")
        rows.append([int(cls), x, y, w, h])
    return rows


def normalized_box(cls: int, box, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    if not all(math.isfinite(v) for v in box) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"invalid original box: {box}")
    return [cls, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]


def same_labels(actual, expected, width: int, height: int) -> bool:
    """Compare labels irrespective of annotation order, at subpixel tolerance."""
    if len(actual) != len(expected):
        return False
    if not actual:
        return True
    a, b = np.asarray(sorted(actual)), np.asarray(sorted(expected))
    return bool(np.array_equal(a[:, 0], b[:, 0]) and np.max(np.abs(a[:, 1:] - b[:, 1:]) * [width, height, width, height]) <= 0.05)


def perceptual_hash(image: Image.Image) -> int:
    # Match the earlier target pHash convention; no claim of rotation invariance.
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    values = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].ravel()
    bits = values > np.median(values[1:])
    bits[0] = False
    return int("".join("1" if bit else "0" for bit in bits), 2)


class Originals:
    """Index local original annotations, preserving COCO IDs within each file."""

    def __init__(self, root: Path, spec: dict, names: list[str]):
        self.folder = root / spec["original"]
        self.spec, self.names = spec, names
        self.files = []
        self.lookup = {}
        self.provenance = {}
        if not self.folder.is_dir():
            raise ValueError(f"original dataset directory missing: {self.folder}")
        if spec["format"] == "voc":
            for path in sorted((self.folder / "images").rglob("*")):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    if path.name in self.lookup:
                        raise ValueError(f"ambiguous original image filename: {path.name}")
                    self.lookup[path.name] = path
            if not self.lookup:
                raise ValueError("no original VOC images")
        else:
            sampling = root / spec["prepared"] / "sampling.json"
            self.files.append(sampling)
            self.provenance = json.loads(sampling.read_text())["provenance"]
            for split in ("train", "val"):
                path = self.folder / "annotations" / f"instances_{split}2017.json"
                self.files.append(path)
                doc = json.loads(path.read_text())
                categories = {str(c["id"]): c["name"] for c in doc["categories"]}
                if len(categories) != len(doc["categories"]) or categories != spec["category_names"]:
                    raise ValueError(f"unexpected COCO category mapping: {path}")
                images = {im["id"]: im for im in doc["images"]}
                if len(images) != len(doc["images"]):
                    raise ValueError(f"duplicate image ID within {path}")
                annotations = defaultdict(list)
                seen = set()
                for annotation in doc["annotations"]:
                    if annotation["id"] in seen or annotation["image_id"] not in images or str(annotation["category_id"]) not in categories:
                        raise ValueError(f"invalid or duplicate COCO annotation reference in {path}")
                    seen.add(annotation["id"])
                    annotations[annotation["image_id"]].append(annotation)
                for ident, im in images.items():
                    name = Path(im["file_name"]).name
                    key = (split, name)
                    if key in self.lookup:
                        raise ValueError(f"ambiguous original image filename: {key}")
                    self.lookup[key] = (im, annotations[ident], path)

    def get(self, split: str, name: str, width: int, height: int):
        if self.spec["format"] == "voc":
            image = self.lookup[name]
            annotation = self.folder / "Annotations" / image.parent.name / (image.stem + ".xml")
            doc = ET.parse(annotation).getroot()
            if (int(doc.findtext("size/width")), int(doc.findtext("size/height"))) != (width, height):
                raise ValueError("original XML dimensions disagree with prepared image")
            labels = []
            for obj in doc.findall("object"):
                label = obj.findtext("name").strip().lower().replace(" ", "_")
                label = {"open": "open_circuit"}.get(label, label)
                box = obj.find("bndbox")
                labels.append(normalized_box(self.names.index(label), [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")], width, height))
            return image, annotation, labels, 0, "original"
        original_split = self.provenance[f"{split}/{name}"]["original_split"]
        expected_split = "val" if split == "test" else "train"
        if original_split != expected_split:
            raise ValueError("prepared split violates the existing official train/validation boundary")
        im, annotations, annotation_file = self.lookup[(original_split, name)]
        if (im["width"], im["height"]) != (width, height):
            raise ValueError("COCO dimensions disagree with prepared image")
        labels, clipped = [], 0
        for annotation in annotations:
            category = str(annotation["category_id"])
            if category not in self.spec["category_mapping"]:
                raise ValueError(f"unsupported original defect category {category}; do not hide its annotation")
            x, y, w, h = annotation["bbox"]
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError("invalid COCO box")
            raw = [x, y, x + w, y + h]
            box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
            correction = max(abs(a - b) for a, b in zip(raw, box))
            if correction > 0.02:
                raise ValueError("COCO edge correction exceeds documented 0.02 pixel rounding tolerance")
            clipped += int(correction > 1e-6)
            labels.append(normalized_box(self.spec["category_mapping"][category], box, width, height))
        return self.folder / f"{original_split}2017" / name, annotation_file, labels, clipped, original_split


def duplicate_findings(rows: list[dict]):
    exact, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get("image_sha256"):
            exact[row["image_sha256"]].append(row)
        if row.get("group_id"):
            groups[(row["dataset"], row["group_id"])].append(row)
    leaks = [items for items in exact.values() if len({r["split"] for r in items}) > 1]
    group_leaks = [items for items in groups.values() if len({r["split"] for r in items}) > 1]
    # All images, including empty labels and malformed-label cases, participate.
    near = []
    for dataset in sorted({r["dataset"] for r in rows}):
        pool = [r for r in rows if r["dataset"] == dataset and r.get("phash")]
        for i, first in enumerate(SPLITS):
            left = [(r, int(r["phash"], 16)) for r in pool if r["split"] == first]
            for second in SPLITS[i + 1:]:
                right = [(r, int(r["phash"], 16)) for r in pool if r["split"] == second]
                for a, ha in left:
                    for b, hb in right:
                        distance = (ha ^ hb).bit_count()
                        if distance <= 4 and a["image_sha256"] != b["image_sha256"]:
                            near.append({"dataset": dataset, "first": a["image"], "second": b["image"], "distance": distance})
    return leaks, group_leaks, near


def audit_dataset(root: Path, spec: dict, names: list[str]):
    folder = root / spec["prepared"]
    rows, issues, inputs = [], [], []
    annotation_hashes = {}

    def issue(code, detail, path=None, severity="error"):
        issues.append({"dataset": spec["id"], "severity": severity, "code": code, "detail": detail, "path": str(path.relative_to(root)) if path else None})

    yaml_path = folder / "data.yaml"
    originals = None
    try:
        inputs.append(yaml_path)
        data = yaml.safe_load(yaml_path.read_text())
        actual_names = data["names"]
        if isinstance(actual_names, dict):
            actual_names = [actual_names[i] for i in range(len(actual_names))]
        if actual_names != names or any(data.get(split) != f"images/{split}" for split in SPLITS):
            raise ValueError("unexpected class order or split layout in data.yaml")
        originals = Originals(root, spec, names)
        inputs.extend(originals.files)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        issue("original_or_config_invalid", str(error))

    for split in SPLITS:
        image_dir, label_dir = folder / "images" / split, folder / "labels" / split
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not image_dir.is_dir() or not label_dir.is_dir():
            issue("missing_split", f"missing image or label directory for {split}")
        if len(images) != spec["expected_images"][split]:
            issue("inventory_changed", f"{split}: expected {spec['expected_images'][split]} images, found {len(images)}; review dataset version")
        expected_labels = set()
        for number, path in enumerate(images, 1):
            label = label_dir / path.relative_to(image_dir).with_suffix(".txt")
            if label in expected_labels:
                issue("ambiguous_label", "multiple images share one label filename", label)
            expected_labels.add(label)
            group = path.stem.split("_")[0] if spec["grouping"] == "board_template_prefix" else None
            row = {"dataset": spec["id"], "split": split, "evaluation_role": ROLES[split],
                   "image": path.relative_to(root).as_posix(), "label": label.relative_to(root).as_posix(),
                   "parent_id": f"{spec['id']}/{path.name}", "group_id": group, "labels": [],
                   "original_verified": False, "negative_status": "not_assessed"}
            rows.append(row)
            try:
                row["image_sha256"] = sha256(path)
                with Image.open(path) as image:
                    image.load()  # Decode pixels; header validation alone misses corrupt images.
                    width, height = image.size
                    row.update(width=width, height=height, phash=f"{perceptual_hash(image):016x}")
                    if image.getexif().get(274, 1) != 1:
                        issue("exif_orientation", "non-identity EXIF orientation requires explicit geometry handling", path)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                issue("unreadable_image", str(error), path)
                continue
            try:
                row["label_sha256"] = sha256(label)
                row["labels"] = read_labels(label, set(spec["expected_class_ids"]))
                row["negative_status"] = "unverified_empty_annotation" if not row["labels"] else "contains_annotated_defects"
            except (OSError, ValueError) as error:
                issue("invalid_label", str(error), label)
                continue
            if originals is not None:
                try:
                    original, annotation, expected, clipped, original_split = originals.get(split, path.name, width, height)
                    if annotation not in annotation_hashes:
                        annotation_hashes[annotation] = sha256(annotation)
                    row.update(original_image=original.relative_to(root).as_posix(),
                               original_annotation=annotation.relative_to(root).as_posix(),
                               original_annotation_sha256=annotation_hashes[annotation], original_split=original_split,
                               original_image_sha256=sha256(original), edge_rounding_boxes=clipped)
                    if row["original_image_sha256"] != row["image_sha256"]:
                        raise ValueError("prepared image differs from original bytes; no image transformation is expected")
                    if not same_labels(row["labels"], expected, width, height):
                        raise ValueError("prepared boxes/classes differ from original annotations")
                    row["original_verified"] = True
                except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError) as error:
                    issue("original_mismatch", str(error), path)
            if number % 500 == 0:
                print(f"{spec['id']}/{split}: {number}/{len(images)} inspected", flush=True)
        for orphan in sorted(set(label_dir.rglob("*.txt")) - expected_labels):
            issue("orphan_label", "label has no matching image", orphan)
        covered = {int(box[0]) for r in rows if r["split"] == split for box in r["labels"]}
        if covered != set(spec["expected_class_ids"]):
            issue("class_coverage", f"{split}: expected {spec['expected_class_ids']}, found {sorted(covered)}")
    if spec["grouping"] == "unknown":
        issue("unknown_board_groups", "Board-disjointness cannot be established for target images.", severity="warning")
    issue("development_test", "Legacy test has already influenced development; it is not an untouched final test.", severity="warning")
    empty = sum(r["negative_status"] == "unverified_empty_annotation" for r in rows)
    issue("negative_coverage", f"{empty} empty annotations; none is certified healthy by this audit. Normal-board specificity is unestablished.", severity="warning")
    return rows, issues, [p for p in inputs if p.is_file()]


def baseline_inventory(root: Path, specs: list[dict], names: list[str]):
    records, issues, inputs = [], [], []
    for spec in specs:
        folder = root / spec["model_folder"]
        record = {"domain": spec["id"], "new_inference_run": False, "checkpoint_verified": False}
        try:
            info_path, weights, metrics_path = folder / "model_info.json", folder / "model.pt", root / spec["metrics"]
            inputs.extend([info_path, weights, metrics_path])
            info, saved = json.loads(info_path.read_text()), json.loads(metrics_path.read_text())
            digest = sha256(weights)
            metric = saved.get("development_test", saved)
            if digest != info["checkpoint_sha256"] or digest != metric["checkpoint_sha256"]:
                raise ValueError("checkpoint/metadata/metric SHA256 mismatch")
            if info["classes"] != names or info.get("evaluated_class_ids", list(range(6))) != spec["expected_class_ids"]:
                raise ValueError("model metadata class order or evaluated coverage mismatch")
            record.update(checkpoint_verified=True, checkpoint=weights.relative_to(root).as_posix(),
                          checkpoint_sha256=digest, metadata=info, saved_metrics=metric,
                          evaluation_status="historical development benchmark; not reproduced by this audit",
                          checkpoint_contents_loaded=False)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append({"severity": "error", "code": "baseline_invalid", "dataset": spec["id"], "detail": str(error)})
        records.append(record)
    reported_path = root / "reports/accuracy/six_class_kaggle_reported_results.json"
    if reported_path.is_file():
        inputs.append(reported_path)
        records.append({"domain": "mixed", "checkpoint_verified": False, "new_inference_run": False,
                        "evidence_status": "reported metrics only; candidate checkpoint not verified",
                        "reported_results": json.loads(reported_path.read_text())})
    return records, issues, [p for p in inputs if p.is_file()]


def render_samples(root: Path, output: Path, rows: list[dict], names: list[str]):
    """One seeded example per available class and train/val split; never test."""
    directory = output / "samples"
    directory.mkdir()
    review = []
    rng = random.Random(42)
    for dataset in sorted({r["dataset"] for r in rows}):
        for split in ("train", "val"):
            panels = []
            for cls, name in enumerate(names):
                pool = [r for r in rows if r["dataset"] == dataset and r["split"] == split and r["original_verified"] and any(b[0] == cls for b in r["labels"])]
                if not pool:
                    continue
                row = rng.choice(pool)
                with Image.open(root / row["image"]) as raw:
                    image = raw.convert("RGB")
                w, h = image.size
                boxes = [[int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h] for c, x, y, bw, bh in row["labels"]]
                selected = next(b for b in boxes if b[0] == cls)
                draw = ImageDraw.Draw(image)
                for c, x1, y1, x2, y2 in boxes:
                    draw.rectangle((x1, y1, x2, y2), outline="#ff3030", width=max(1, round(w / 900)))
                _, x1, y1, x2, y2 = selected
                size = min(max(128, round(max(x2 - x1, y2 - y1) * 4)), w, h)
                left = max(0, min(w - size, round((x1 + x2 - size) / 2)))
                top = max(0, min(h - size, round((y1 + y2 - size) / 2)))
                detail = image.crop((left, top, left + size, top + size))
                panel = Image.new("RGB", (960, 320), "#f4f4f4")
                panel.paste(ImageOps.contain(image, (460, 270)), (10, 40))
                panel.paste(ImageOps.contain(detail, (460, 270)), (490, 40))
                ImageDraw.Draw(panel).text((10, 8), f"{dataset}/{split} | {name} | {Path(row['image']).name} | overview + detail", fill="black")
                panels.append(panel)
                review.append({"dataset": dataset, "split": split, "class_name": name, "image": row["image"],
                               "image_sha256": row["image_sha256"], "status": "pending_visual_review",
                               "contact_sheet": f"samples/{dataset}_{split}.jpg"})
            if panels:
                sheet = Image.new("RGB", (960, 320 * len(panels)), "white")
                for i, panel in enumerate(panels):
                    sheet.paste(panel, (0, i * 320))
                sheet.save(directory / f"{dataset}_{split}.jpg", quality=92)
    return review


def run(root: Path, output: Path, config_path: Path = CONFIG):
    root, output = root.resolve(), output.resolve()
    config = json.loads(config_path.read_text())
    protected = [root / "data"] + [root / spec["model_folder"] for spec in config["datasets"]]
    if any(output.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("output must not be inside a dataset or deployed model directory")
    output.mkdir(parents=True, exist_ok=False)
    rows, issues, inputs = [], [], [config_path, Path(__file__)]
    for spec in config["datasets"]:
        print(f"Auditing {spec['id']} against local originals...", flush=True)
        data_rows, data_issues, data_inputs = audit_dataset(root, spec, config["classes"])
        rows.extend(data_rows)
        issues.extend(data_issues)
        inputs.extend(data_inputs)
    leaks, group_leaks, near = duplicate_findings(rows)
    for code, findings in (("exact_split_leak", leaks), ("group_split_leak", group_leaks)):
        for group in findings:
            issues.append({"severity": "error", "code": code, "images": [r["image"] for r in group]})
    if near:
        issues.append({"severity": "warning", "code": "near_duplicate_review", "detail": f"{len(near)} cross-split pHash candidates require adjudication; no automatic removal."})
    baselines, baseline_issues, baseline_inputs = baseline_inventory(root, config["datasets"], config["classes"])
    issues.extend(baseline_issues)
    inputs.extend(baseline_inputs)
    reviews = render_samples(root, output, rows, config["classes"])
    datasets = {}
    for spec in config["datasets"]:
        splits = {}
        for split in SPLITS:
            selected = [r for r in rows if r["dataset"] == spec["id"] and r["split"] == split]
            counts = Counter(int(box[0]) for r in selected for box in r["labels"])
            sides = [min(b[3] * r["width"], b[4] * r["height"]) for r in selected for b in r["labels"]]
            splits[split] = {"role": ROLES[split], "images": len(selected), "instances": {name: counts[i] for i, name in enumerate(config["classes"])},
                             "original_verified_images": sum(r["original_verified"] for r in selected),
                             "groups": sorted({r["group_id"] for r in selected if r["group_id"]}),
                             "empty_annotations": sum(r["negative_status"] == "unverified_empty_annotation" for r in selected),
                             "short_box_side_native_p10_p50_p90": np.percentile(sides, [10, 50, 90]).tolist() if sides else []}
        datasets[spec["id"]] = {"splits": splits, "source_url": spec["source_url"], "usage": spec["usage"], "grouping": spec["grouping"]}
    errors = sum(i["severity"] == "error" for i in issues)
    summary = {"schema_version": 1, "phase": 1, "structural_checks_passed": errors == 0,
               "training_authorized_by_audit": False, "status": "blocked" if errors else "structural_checks_passed_review_required",
               "errors": errors, "warnings": sum(i["severity"] == "warning" for i in issues),
               "datasets": datasets, "cross_split_exact_duplicate_sets": len(leaks), "cross_split_group_leaks": len(group_leaks),
               "cross_split_phash_candidates": len(near), "visual_review_samples": len(reviews),
               "new_training_or_inference": False,
               "limitations": ["Original label agreement is not independent verification of physical truth.",
                               "Target board identities unknown; pHash cannot prove independence.",
                               "Historical test sets are development benchmarks.",
                               "Target missing-hole coverage is absent.",
                               "Normal full-board specificity and camera generalization are unestablished.",
                               "External candidates are not downloaded or validated in Phase 1.",
                               "Sample review is pending and cannot establish whole-dataset annotation quality."]}
    with (output / "manifest.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    for name, value in (("summary", summary), ("issues", issues), ("baselines", baselines), ("near_duplicates", near), ("review_samples", reviews)):
        write_json(output / f"{name}.json", value)
    lines = ["# Phase 1 audit", "", f"Status: **{summary['status']}**. Errors: {errors}; warnings: {summary['warnings']}.",
             "", "This run inspected existing inputs. It did not train, run inference, repair labels or alter splits.", ""]
    for domain, dataset in datasets.items():
        lines.append(f"## {domain.capitalize()}")
        for split, stat in dataset["splits"].items():
            lines.append(f"- {split} ({stat['role']}): {stat['images']} images; {sum(stat['instances'].values())} boxes; {stat['original_verified_images']} original matches.")
        lines.append("")
    lines.extend(["## Findings", ""] + [f"- {issue['severity'].upper()}: {issue['code']} — {issue.get('detail', issue.get('images'))}" for issue in issues])
    lines.extend(["", "## Required next steps", "", "Review the train/validation contact sheets and any near-duplicate candidates. Resolve blockers before Phase 2. Keep unsupported classes and evaluated test status explicit. No automatic promotion or training follows this audit.", ""])
    (output / "SUMMARY.md").write_text("\n".join(lines))
    source_hashes = {}
    for path in sorted(set(inputs)):
        key = path.resolve().relative_to(root).as_posix() if path.resolve().is_relative_to(root) else str(path.resolve())
        source_hashes[key] = sha256(path)
    artifact_hashes = {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "artifact_manifest.json", {"sha256": artifact_hashes, "audit_inputs_sha256": source_hashes,
                                                  "python": sys.version, "pillow": Image.__version__, "numpy": np.__version__, "opencv": cv2.__version__})
    print(f"Phase 1: {errors} errors, {summary['warnings']} warnings. Report: {output / 'SUMMARY.md'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True, help="New report directory (existing directories are refused)")
    args = parser.parse_args()
    try:
        summary = run(args.root, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Audit could not complete: {error}\n")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
