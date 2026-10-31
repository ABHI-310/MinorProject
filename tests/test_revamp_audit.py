"""Adversarial audit checks: catch bad data before costly training is allowed."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from revamp.audit import (
    CONFIG, Originals, audit_dataset, baseline_inventory, duplicate_findings,
    read_labels, run, same_labels, sha256,
)

NAMES = json.loads(CONFIG.read_text())["classes"]


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "repo"
    spec = dict(id="source", prepared="data/prepared", original="data/original", format="voc",
                expected_class_ids=[0], expected_images={s: 1 for s in ("train", "val", "test")},
                grouping="board_template_prefix", source_url="https://example.org/source",
                usage="test fixture", model_folder="models/source", metrics="models/source/metrics.json")
    folder = root / spec["prepared"]
    rng = np.random.default_rng(4)
    for index, split in enumerate(("train", "val", "test"), 1):
        name = f"{index:02d}_missing_hole_01"
        image = root / spec["original"] / "images/Missing_hole" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(image)
        xml = root / spec["original"] / "Annotations/Missing_hole" / f"{name}.xml"
        xml.parent.mkdir(parents=True, exist_ok=True)
        xml.write_text("<annotation><size><width>64</width><height>64</height></size>"
                       "<object><name>missing_hole</name><bndbox><xmin>16</xmin><ymin>16</ymin>"
                       "<xmax>48</xmax><ymax>48</ymax></bndbox></object></annotation>")
        prepared = folder / "images" / split / image.name
        prepared.parent.mkdir(parents=True)
        prepared.write_bytes(image.read_bytes())
        label = folder / "labels" / split / f"{name}.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 .5 .5 .5 .5\n")
    (folder / "data.yaml").write_text(yaml.safe_dump(dict(names=NAMES, train="images/train", val="images/val", test="images/test")))
    model = root / spec["model_folder"]
    model.mkdir(parents=True)
    (model / "model.pt").write_bytes(b"opaque fixture; deliberately never unpickled")
    digest = sha256(model / "model.pt")
    (model / "model_info.json").write_text(json.dumps(dict(classes=NAMES, evaluated_class_ids=[0], checkpoint_sha256=digest)))
    (model / "metrics.json").write_text(json.dumps(dict(checkpoint_sha256=digest, mAP50=.5)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(classes=NAMES, datasets=[spec])))
    return root, spec, config


@pytest.mark.parametrize("line", ["0.5 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .5 .5 inf .1",
                                  "0 .5 .5 -.1 .1", "0 .99 .5 .1 .1", "6 .5 .5 .1 .1",
                                  "0 .5 .5 .1", "bad label"])
def test_invalid_labels_rejected(tmp_path, line):
    path = tmp_path / "label.txt"
    path.write_text(line)
    with pytest.raises(ValueError):
        read_labels(path, set(range(6)))


def test_empty_is_distinct_from_missing(tmp_path):
    path = tmp_path / "empty.txt"
    with pytest.raises(FileNotFoundError):
        read_labels(path, {0})
    path.write_text("\n")
    assert read_labels(path, {0}) == []


def test_box_comparison_checks_geometry_classes_and_order():
    boxes = [[0, .5, .5, .2, .2], [1, .2, .3, .1, .1]]
    assert same_labels(boxes, boxes[::-1], 3000, 2000)
    assert not same_labels(boxes, [[0, .51, .5, .2, .2], boxes[1]], 3000, 2000)
    assert not same_labels(boxes, [[2, .5, .5, .2, .2], boxes[1]], 3000, 2000)


def test_duplicate_check_includes_empty_and_invalid_labels():
    rows = [dict(dataset="source", split=s, image=s, image_sha256="same", group_id="01", labels=[])
            for s in ("train", "val")]
    exact, groups, near = duplicate_findings(rows)
    assert len(exact) == len(groups) == 1
    assert near == []


def test_perceptual_candidate_is_not_an_exact_duplicate():
    rows = [dict(dataset="target", split=s, image=s, image_sha256=s, group_id=None, phash=f"{h:016x}")
            for s, h in (("train", 0), ("val", 7), ("test", 65535))]
    exact, groups, near = duplicate_findings(rows)
    assert not exact and not groups
    assert len(near) == 1 and near[0]["distance"] == 3


def test_complete_run_is_non_destructive_reproducible_and_never_samples_test(source):
    root, spec, config = source
    before = {p: sha256(p) for p in root.rglob("*") if p.is_file()}
    report = root.parent / "report"
    result = run(root, report, config)
    assert result["errors"] == 0
    assert not result["training_authorized_by_audit"]
    assert all(sha256(p) == digest for p, digest in before.items())
    assert {p for p in root.rglob("*") if p.is_file()} == set(before)
    reviews = json.loads((report / "review_samples.json").read_text())
    assert len(reviews) == 2 and {r["split"] for r in reviews} == {"train", "val"}
    records = [json.loads(line) for line in (report / "manifest.jsonl").read_text().splitlines()]
    assert all(r["original_verified"] for r in records)
    assert records[-1]["evaluation_role"] == "development_test"
    artifact = json.loads((report / "artifact_manifest.json").read_text())
    assert all(sha256(report / name) == digest for name, digest in artifact["sha256"].items())
    with pytest.raises(FileExistsError):
        run(root, report, config)
    other = root.parent / "repeat"
    run(root, other, config)
    assert (other / "manifest.jsonl").read_bytes() == (report / "manifest.jsonl").read_bytes()
    assert (other / "review_samples.json").read_bytes() == (report / "review_samples.json").read_bytes()


def test_output_inside_input_tree_refused(source):
    root, spec, config = source
    for path in (root / "data/new_report", root / spec["model_folder"] / "new_report"):
        with pytest.raises(ValueError, match="output"):
            run(root, path, config)
        assert not path.exists()


def test_shifted_annotation_blocks_audit(source):
    root, spec, _ = source
    label = next((root / spec["prepared"] / "labels/train").glob("*.txt"))
    label.write_text("0 .6 .5 .5 .5\n")  # Valid YOLO geometry, incorrect relative to original.
    rows, issues, _ = audit_dataset(root, spec, NAMES)
    assert any(i["code"] == "original_mismatch" for i in issues)
    assert not rows[0]["original_verified"]


def test_corrupt_image_missing_label_and_orphan_are_reported(source):
    root, spec, _ = source
    folder = root / spec["prepared"]
    next((folder / "images/train").glob("*.png")).write_bytes(b"not an image")
    next((folder / "labels/val").glob("*.txt")).unlink()
    (folder / "labels/test/orphan.txt").write_text("0 .5 .5 .1 .1")
    _, issues, _ = audit_dataset(root, spec, NAMES)
    codes = {i["code"] for i in issues}
    assert {"unreadable_image", "invalid_label", "orphan_label"} <= codes


def test_checkpoint_and_metric_tampering_are_rejected(source):
    root, spec, _ = source
    records, issues, _ = baseline_inventory(root, [spec], NAMES)
    assert not issues and records[0]["checkpoint_verified"]
    (root / spec["metrics"]).write_text('{"checkpoint_sha256":"wrong"}')
    records, issues, _ = baseline_inventory(root, [spec], NAMES)
    assert not records[0]["checkpoint_verified"] and issues[0]["code"] == "baseline_invalid"


def test_coco_ids_scoped_per_file_and_unsupported_classes_not_dropped(tmp_path):
    spec = dict(original="raw", prepared="prepared", format="coco",
                category_names={"1": "SH", "6": "HB"}, category_mapping={"1": 3})
    (tmp_path / "raw/annotations").mkdir(parents=True)
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared/sampling.json").write_text(json.dumps({"provenance": {
        "train/a.jpg": {"original_split": "train"}, "test/a.jpg": {"original_split": "val"}}}))
    doc = dict(categories=[dict(id=1, name="SH"), dict(id=6, name="HB")],
               images=[dict(id=0, file_name="a.jpg", width=100, height=100)],
               annotations=[dict(id=0, image_id=0, category_id=1, bbox=[-.01, 2, 10, 10])])
    for split in ("train", "val"):
        (tmp_path / f"raw/annotations/instances_{split}2017.json").write_text(json.dumps(doc))
    originals = Originals(tmp_path, spec, NAMES)
    assert originals.get("train", "a.jpg", 100, 100)[3] == 1
    assert originals.get("test", "a.jpg", 100, 100)[4] == "val"
    assert originals.get("train", "a.jpg", 100, 100)[2][0][0] == 3
    doc["annotations"][0]["category_id"] = 6
    (tmp_path / "raw/annotations/instances_train2017.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unsupported original"):
        Originals(tmp_path, spec, NAMES).get("train", "a.jpg", 100, 100)
