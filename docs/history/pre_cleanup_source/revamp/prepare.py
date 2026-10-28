"""Build new, parent-traceable train/validation views from the Phase 1 audit.

No test pixels are read or copied. Background candidates are review material,
not training negatives. All outputs require a previously nonexistent directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageOps

from revamp.audit import ROOT, normalized_box, sha256, write_json


def pixel_boxes(row):
    w, h = row["width"], row["height"]
    return [(int(c), ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h))
            for c, x, y, bw, bh in row["labels"]]


def crop_labels(row, window, margin=8):
    """Reject truncated or near-edge defects instead of silently dropping boxes."""
    x, y, right, bottom = window
    labels = []
    for cls, (a, b, c, d) in pixel_boxes(row):
        if c <= x or a >= right or d <= y or b >= bottom:
            continue
        if a < x+margin or b < y+margin or c > right-margin or d > bottom-margin:
            return None
        labels.append(normalized_box(cls, [a-x, b-y, c-x, d-y], right-x, bottom-y))
    return labels


def positive_window(row, size=640):
    size = min(size, row["width"], row["height"])
    boxes = pixel_boxes(row)
    offset = int(row["image_sha256"][:8], 16) % len(boxes)
    for _, (a, b, c, d) in boxes[offset:] + boxes[:offset]:
        x = max(0, min(row["width"]-size, round((a+c-size)/2)))
        y = max(0, min(row["height"]-size, round((b+d-size)/2)))
        window = (x, y, x+size, y+size)
        labels = crop_labels(row, window)
        if labels:
            return window, labels
    return None


def negative_window(row, size=512, margin=32):
    if min(row["width"], row["height"]) < size:
        return None
    for y in range(0, row["height"]-size+1, size//2):
        for x in range(0, row["width"]-size+1, size//2):
            if all(c <= x-margin or a >= x+size+margin or d <= y-margin or b >= y+size+margin
                   for _, (a, b, c, d) in pixel_boxes(row)):
                return (x, y, x+size, y+size)
    return None


def save_view(root, output, row, kind, window=None, labels=None):
    name = f'{row["dataset"]}__{Path(row["image"]).stem}__{kind}'
    image_rel = f'images/{row["split"]}/{name}.jpg'
    label_rel = f'labels/{row["split"]}/{name}.txt'
    image, label = output/image_rel, output/label_rel
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    if window:
        with Image.open(root/row["image"]) as original:
            original.convert("RGB").crop(window).save(image, quality=95, subsampling=0)
        width, height = window[2]-window[0], window[3]-window[1]
    else:
        shutil.copyfile(root/row["image"], image)
        width, height = row["width"], row["height"]
    labels = row["labels"] if labels is None else labels
    label.write_text("".join(f'{int(c)} {x:.10f} {y:.10f} {w:.10f} {h:.10f}\n' for c,x,y,w,h in labels))
    return {"dataset":row["dataset"], "split":row["split"], "parent_id":row["parent_id"],
            "group_id":row["group_id"], "kind":kind, "window":window,
            "image":image_rel, "label":label_rel, "width":width, "height":height,
            "image_sha256":sha256(image), "label_sha256":sha256(label),
            "parent_image_sha256":row["image_sha256"], "parent_label_sha256":row["label_sha256"],
            "labels":labels}


def prepare(root, audit, output):
    root, audit, output = Path(root).resolve(), Path(audit).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new version directory.")
    seal = json.loads((audit/"artifact_manifest.json").read_text())
    for base, hashes in [(audit, seal["sha256"]), (root, seal["audit_inputs_sha256"])]:
        for relative, digest in hashes.items():
            if sha256(base/relative) != digest:
                raise ValueError(f"Phase 1 evidence changed: {relative}")
    issues = json.loads((audit/"issues.json").read_text())
    if any(issue["severity"] == "error" for issue in issues):
        raise ValueError("Phase 1 has blocking findings.")
    all_rows = [json.loads(line) for line in (audit/"manifest.jsonl").read_text().splitlines()]
    rows = [r for r in all_rows if r["split"] in ("train", "val")]
    for row in rows:
        if not row["original_verified"] or not row["labels"]:
            raise ValueError("Preparation requires verified positive parents.")
        for key in ("image", "label"):
            if sha256(root/row[key]) != row[key+"_sha256"]:
                raise ValueError(f'Input changed since audit: {row[key]}')
    output.mkdir(parents=True)
    views, candidates, counts = [], [], Counter()
    review = output/"review"; review.mkdir()
    for row in rows:
        views.append(save_view(root, output, row, "full"))
        if row["dataset"] != "source" or row["split"] != "train":
            continue
        positive = positive_window(row)
        if positive:
            window, labels = positive
            views.append(save_view(root, output, row, "crop", window, labels))
        # A bounded review pilot: four candidates per known source training group.
        if counts[row["group_id"]] < 4:
            window = negative_window(row)
            if window:
                candidate = save_view(root, review, row, "negative_candidate", window, [])
                candidate["status"] = "unreviewed_not_for_training"
                candidates.append(candidate); counts[row["group_id"]] += 1
    names = json.loads((root/"revamp/datasets.json").read_text())["classes"]
    manifest = {"schema":1, "names":names, "views":views,
                "audit_manifest_sha256":sha256(audit/"manifest.jsonl"),
                "preparation_code_sha256":sha256(Path(__file__)),
                "excluded_development_test_parents":sum(r["split"]=="test" for r in all_rows),
                "negative_policy":"Only separately reviewed training crops may be added when packaging.",
                "limitations":["Target board independence unknown; target class 0 unsupported.",
                               "No normal full-board benchmark; cropped background is not a normal-board benchmark.",
                               "Historical test partitions remain development tests, excluded from this package."]}
    write_json(output/"manifest.json", manifest)
    write_json(review/"candidates.json", {"candidates":candidates})
    for start in range(0, len(candidates), 6):
        page = Image.new("RGB", (3*512, 2*544), "white")
        draw = ImageDraw.Draw(page)
        for index, candidate in enumerate(candidates[start:start+6]):
            x, y = (index%3)*512, (index//3)*544
            with Image.open(review/candidate["image"]) as im:
                page.paste(ImageOps.contain(im, (512,512)), (x,y+32))
            draw.text((x+4,y+6), f'{start+index}: {candidate["parent_id"]}', fill="black")
        page.save(review/f"candidates_{start//6+1}.jpg", quality=95)
    write_json(output/"summary.json", {"counts":dict(Counter(f'{r["dataset"]}/{r["split"]}/{r["kind"]}' for r in views)),
                                       "negative_candidates_excluded":len(candidates)})
    print((output/"summary.json").read_text())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path, default=ROOT/"reports/revamp/phase1_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.audit, args.output)
