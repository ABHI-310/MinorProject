"""Portable, offline package verification and writable dataset configuration."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

NAMES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"Unsafe path: {relative}")
    result = (Path(root)/relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return result


def verify_views(root, manifest):
    """Validate files, decoded geometry, parent splits, group splits and taxonomy."""
    from PIL import Image
    if manifest["names"] != NAMES:
        raise ValueError("Class order changed.")
    seen, parents, groups, hashes, classes = set(), defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    for row in manifest["views"]:
        domain, split, kind = row["dataset"], row["split"], row["kind"]
        if domain not in ("source", "target") or split not in ("train", "val"):
            raise ValueError("Unknown domain or development-test data in package.")
        if kind not in ("full", "crop", "reviewed_background") or (kind != "full" and (domain != "source" or split != "train")):
            raise ValueError("Derived views must be source training images.")
        if not row["parent_id"].startswith(domain+"/"):
            raise ValueError("Parent belongs to another domain.")
        if kind == "reviewed_background" and not row.get("review"):
            raise ValueError("Background has no review record.")
        for key in ("image", "label"):
            if row[key] in seen:
                raise ValueError("Duplicate file reference.")
            seen.add(row[key])
            path = safe_path(root, row[key])
            if digest(path) != row[key+"_sha256"]:
                raise ValueError(f'Changed {key}: {row[key]}')
        with Image.open(safe_path(root, row["image"])) as im:
            im.load()
            if im.size != (row["width"], row["height"]):
                raise ValueError("Wrong image dimensions.")
        parsed = [[float(v) for v in line.split()] for line in safe_path(root,row["label"]).read_text().splitlines() if line.strip()]
        if len(parsed) != len(row["labels"]):
            raise ValueError("Manifest/label count mismatch.")
        if (not parsed) != (kind == "reviewed_background"):
            raise ValueError("Unexpected empty or nonempty annotations.")
        for values, expected in zip(parsed, row["labels"]):
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid label row.")
            c, x, y, w, h = values
            allowed = range(6) if domain == "source" else range(1,6)
            if c != int(c) or int(c) not in allowed or min(w,h) <= 0 or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
                raise ValueError("Invalid class or box geometry.")
            if len(expected) != 5 or not all(math.isfinite(v) for v in expected) or any(abs(a-b)>1e-8 for a,b in zip(values,expected)):
                raise ValueError("Manifest/label geometry mismatch.")
            classes[(domain,split)].add(int(c))
        parents[row["parent_id"]].add(split)
        if row["group_id"] is not None:
            groups[(domain,row["group_id"])].add(split)
        hashes[row["image_sha256"]].add(split)
    if any(len(v)>1 for pool in (parents,groups,hashes) for v in pool.values()):
        raise ValueError("Parent, group or exact-image leakage across splits.")
    for domain in ("source", "target"):
        for split in ("train", "val"):
            if classes[(domain,split)] != set(range(6) if domain == "source" else range(1,6)):
                raise ValueError(f"Missing required classes: {domain}/{split}")
    return dict(Counter(f'{r["dataset"]}/{r["split"]}/{r["kind"]}' for r in manifest["views"]))


def verify_package(root):
    root = Path(root)
    record = json.loads((root/"package_hashes.json").read_text())
    for name, checksum in record["sha256"].items():
        if digest(safe_path(root,name)) != checksum:
            raise ValueError(f"Package checksum mismatch: {name}")
    manifest = json.loads((root/"manifest.json").read_text())
    required = {r[k] for r in manifest["views"] for k in ("image","label")}
    required |= {"manifest.json", "weights/yolov8n.pt", "weights/yolov8s.pt", "weights/source_baseline.pt", "weights/target_baseline.pt"}
    if not required <= set(record["sha256"]):
        raise ValueError("Incomplete package hash coverage.")
    referenced = {r[k] for r in manifest["views"] for k in ("image", "label")}
    actual = {p.relative_to(root).as_posix() for folder in ("images","labels")
              for p in (root/folder).rglob("*") if p.is_file() and p.suffix != ".cache"}
    if actual != referenced:
        raise ValueError("Unlisted image or label in package.")
    return verify_views(root,manifest)


def extract_package(archive, destination):
    """Extract into a new folder only; reject duplicate, traversal and symlink entries."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new extraction directory; existing data will not be overwritten.")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries.")
        for info in z.infolist():
            safe_path(destination, info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in ZIP.")
        if z.testzip() is not None:
            raise ValueError("Damaged ZIP.")
        destination.mkdir(parents=True)
        z.extractall(destination)
    verify_package(destination)
    return destination


def configure(root, output):
    """Separate source/target recipes; val always uses unchanged original views."""
    import yaml
    root, output = Path(root).resolve(), Path(output).resolve()
    verify_package(root)
    manifest = json.loads((root/"manifest.json").read_text())
    configs, contents = {}, {}
    for recipe, domain, full_only in [("source_full","source",True), ("source_views","source",False), ("target","target",True)]:
        for split in ("train", "val"):
            images = [str(root/r["image"]) for r in manifest["views"] if r["dataset"]==domain and r["split"]==split and (not full_only or r["kind"]=="full")]
            contents[output/f"{recipe}_{split}.txt"] = "\n".join(images)+"\n"
        path = output/f"{recipe}.yaml"
        contents[path] = yaml.safe_dump({"path":str(root), "train":str(output/f"{recipe}_train.txt"),
                                       "val":str(output/f"{recipe}_val.txt"), "names":NAMES})
        configs[recipe] = str(path)
    for path, content in contents.items():
        if path.exists() and path.read_text() != content:
            raise ValueError("Existing configuration differs: " + str(path))
    output.mkdir(parents=True,exist_ok=True)
    for path, content in contents.items():
        if not path.exists():
            path.write_text(content)
    return configs
