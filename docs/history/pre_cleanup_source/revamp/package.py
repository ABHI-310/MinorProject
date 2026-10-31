"""Package reviewed Phase 2 assets and a verification-only Kaggle notebook."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import zipfile

from revamp.audit import ROOT, sha256, write_json
from revamp.runtime import safe_path, verify_package, verify_views


def approved_background(prepared, review_path):
    candidates_path = prepared/"review/candidates.json"
    candidates = json.loads(candidates_path.read_text())["candidates"]
    review = json.loads(review_path.read_text())
    if review["candidates_sha256"] != sha256(candidates_path) or not review.get("reviewer"):
        raise ValueError("Review does not match candidates or identify reviewer.")
    decisions = review["decisions"]
    by_parent = {c["parent_id"]:c for c in candidates}
    if len(decisions) != len(by_parent) or {d["parent_id"] for d in decisions} != set(by_parent):
        raise ValueError("Every candidate must have exactly one review decision.")
    accepted = []
    for decision in decisions:
        if decision["image_sha256"] != by_parent[decision["parent_id"]]["image_sha256"]:
            raise ValueError("Review image hash mismatch.")
        if decision["decision"] not in ("accept", "exclude") or not decision.get("reason"):
            raise ValueError("Review decision/reason missing.")
        if decision["decision"] == "accept":
            row = copy.deepcopy(by_parent[decision["parent_id"]])
            if row["split"] != "train" or row["dataset"] != "source" or row["labels"]:
                raise ValueError("Background must be an empty source training crop.")
            row["kind"] = "reviewed_background"
            row["review"] = {"reviewer":review["reviewer"],"reason":decision["reason"],"record_sha256":sha256(review_path)}
            row.pop("status", None)
            accepted.append(row)
    if len({r["image_sha256"] for r in accepted}) != len(accepted):
        raise ValueError("Duplicate accepted background images.")
    return accepted


def notebook(package_hash):
    cells = []
    def md(text):
        cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(True)})
    def code(text):
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(True)})
    md("# PCB revamp — Phase 2 verification\n\nAttach the private Phase 2 dataset. This notebook verifies inputs and runs a **six-image learning/resume check only**. Full training is Phase 3. All learning-check scores are on memorized training images, not validation accuracy.\n\nUse a GPU if available; CPU also works. Internet is needed only to install dependencies. Keep Kaggle's installed CUDA PyTorch. Save/download the generated smoke report and logs before ending the session.")
    code("import sys, subprocess, importlib.metadata as meta\nfrom pathlib import Path\nfrom packaging.requirements import Requirement\n# Preserve Kaggle's CUDA stack and NumPy ABI while resolving other dependencies.\nconstraints = Path('/kaggle/working/pcb_runtime_constraints.txt')\nconstraints.write_text(''.join(f'{name}=={meta.version(name)}\\n' for name in ['torch', 'torchvision', 'numpy']))\ndef satisfied():\n    try:\n        if meta.version('ultralytics') != '8.4.142': return False\n        for text in meta.requires('ultralytics'):\n            req = Requirement(text)\n            if req.marker and not req.marker.evaluate({'extra': ''}): continue\n            if meta.version(req.name) not in req.specifier: return False\n        return True\n    except meta.PackageNotFoundError:\n        return False\nif not satisfied():\n    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.4.142', '--constraint', str(constraints)], check=True)\nassert satisfied(), 'Ultralytics dependencies are incomplete'\nimport torch, torchvision\n")
    code("from pathlib import Path, PurePosixPath\nimport hashlib, json, shutil, zipfile, os\nWORK = Path('/kaggle/working/pcb_phase2')\nif WORK.exists():\n    raise RuntimeError('Folder already exists. Use a new WORK path, or restart a fresh session. No files were overwritten.')\nINPUT = Path('/kaggle/input')\nrecords = list(INPUT.rglob('package_hashes.json'))\narchives = list(INPUT.rglob('pcb_revamp_phase2.zip'))\nif len(records) == 1:\n    shutil.copytree(records[0].parent, WORK)\nelif not records and len(archives) == 1:\n    with zipfile.ZipFile(archives[0]) as z:\n        names = z.namelist()\n        if len(names) != len(set(names)): raise ValueError('Duplicate archive entries')\n        for info in z.infolist():\n            p = PurePosixPath(info.filename)\n            if p.is_absolute() or '..' in p.parts or '\\\\' in info.filename or ((info.external_attr >> 16) & 0o170000) == 0o120000:\n                raise ValueError('Unsafe archive path')\n        if z.testzip() is not None: raise ValueError('Damaged archive')\n        z.extractall(WORK)\nelse:\n    raise RuntimeError('Attach exactly one Phase 2 package (expanded dataset or ZIP).')\n\ndef file_hash(path):\n    h = hashlib.sha256()\n    with path.open('rb') as f:\n        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)\n    return h.hexdigest()\n" + f"EXPECTED_MANIFEST = {package_hash!r}\n" + "assert file_hash(WORK/'package_hashes.json') == EXPECTED_MANIFEST, 'Wrong package version'\nrecord = json.loads((WORK/'package_hashes.json').read_text())\nfor name, expected in record['sha256'].items():\n    path = (WORK/name).resolve()\n    assert path.is_relative_to(WORK.resolve()), 'Unsafe package path'\n    assert file_hash(path) == expected, 'Changed asset: '+name\nos.environ['YOLO_CONFIG_DIR'] = '/kaggle/working/pcb_yolo_settings'\nos.environ['WANDB_MODE'] = 'disabled'\nsys.path.insert(0, str(WORK))\nfrom revamp.runtime import verify_package, configure\nprint(verify_package(WORK))\nCONFIGS = configure(WORK, Path('/kaggle/working/pcb_phase2_configs'))\nprint(CONFIGS)\n")
    code("import platform, ultralytics\n# Import failures here should be resolved before training; never silently change torch.\nfrom ultralytics import YOLO\nenvironment = {'python':platform.python_version(), 'torch':torch.__version__, 'torchvision':torchvision.__version__, 'ultralytics':ultralytics.__version__, 'cuda_available':torch.cuda.is_available(), 'cuda':torch.version.cuda}\nPath('/kaggle/working/pcb_environment.json').write_text(json.dumps(environment, indent=2))\nprint(environment)\nfor name in ['yolov8n', 'yolov8s']:\n    assert len(YOLO(str(WORK/f'weights/{name}.pt')).names) == 80\nfor name in ['source', 'target']:\n    assert list(YOLO(str(WORK/f'weights/{name}_baseline.pt')).names.values()) == json.loads((WORK/'manifest.json').read_text())['names']\n")
    code("# This launches two fresh Python processes and checks saved optimizer continuity.\n# It trains only SIX training crops for 80 tiny epochs. It does not train a candidate.\ncommand = [sys.executable, '-m', 'revamp.smoke', '--package', str(WORK), '--output', '/kaggle/working/pcb_phase2_smoke', '--device', '0' if torch.cuda.is_available() else 'cpu']\nsubprocess.run(command, cwd=WORK, check=True)\nprint(Path('/kaggle/working/pcb_phase2_smoke/report.json').read_text())\n")
    md("## Stop here\n\nA passing check means data loading, learning and checkpoint recovery work in this environment. It does **not** establish PCB accuracy. Download `pcb_environment.json`, `pcb_phase2_smoke/report.json`, `pause.log` and `resume.log`. Phase 3 will add the timed GPU pilot, baseline reproduction and bounded specialist training. Source and target validation must stay separate. Target missing-hole performance is unsupported.")
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"cells":[dict(cell,id=f"phase2-{i}") for i,cell in enumerate(cells)]}


def build(root, prepared, output, review_path=None):
    root, prepared, output = Path(root).resolve(), Path(prepared).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Package output exists; use a new version directory.")
    manifest = json.loads((prepared/"manifest.json").read_text())
    verify_views(prepared,manifest)
    background = approved_background(prepared,Path(review_path)) if review_path else []
    output.mkdir(parents=True)
    package = output/"package"; package.mkdir()
    for base, rows in [(prepared,manifest["views"]),(prepared/"review",background)]:
        for row in rows:
            for key in ("image","label"):
                source, dest = safe_path(base,row[key]), safe_path(package,row[key])
                if sha256(source) != row[key+"_sha256"]:
                    raise ValueError("Changed preparation asset.")
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,dest)
    manifest["views"].extend(background)
    write_json(package/"manifest.json",manifest)
    (package/"revamp").mkdir(); (package/"weights").mkdir()
    for name in ("__init__.py", "runtime.py", "smoke.py", "checkpoints.py"):
        shutil.copyfile(root/"revamp"/name,package/"revamp"/name)
    weight_paths = {"yolov8n.pt":"weights/yolov8n.pt","yolov8s.pt":"weights/yolov8s.pt",
                    "source_baseline.pt":"phase1_final/model.pt","target_baseline.pt":"phase2/current_target/model.pt"}
    for name, source in weight_paths.items():
        shutil.copyfile(root/source,package/"weights"/name)
    shutil.copyfile(root/"revamp/datasets.json",package/"dataset_sources.json")
    if review_path:
        shutil.copyfile(review_path,package/"background_review.json")
    (package/"USAGE.md").write_text("""# Academic PCB training inputs

For this university project. Keep uploads private. Preserve original attribution
and source terms in dataset_sources.json; do not relabel data as public domain.
PKU: https://robotics.pkusz.edu.cn/resources/dataset/ (research use).
DsPCBSD+: https://doi.org/10.6084/m9.figshare.24970329 (CC BY 4.0).
Changes: six-class conversion; unsupported target images excluded; inherited
splits; one bounded native-resolution crop per eligible source training image;
background pilot only where explicitly reviewed. Test data excluded.
Ultralytics YOLO: https://github.com/ultralytics/ultralytics (AGPL-3.0).
Local COCO initialization files are included with recorded hashes; baseline
weights are previous project outputs, not new revamp results.

Use weights/yolov8s.pt for fresh COCO initialization. Baseline weights are for
comparison. Never initialize a source-only/few-shot claim from target_baseline.pt.
Target has no missing_hole examples. Background crops are not normal full boards.
All score comparisons to old test splits are development comparisons.

Phase 2 notebook stops after preflight. Full training is Phase 3.
""")
    write_json(package/"package_hashes.json", {"schema":1,"sha256":{p.relative_to(package).as_posix():sha256(p) for p in sorted(package.rglob("*")) if p.is_file()}})
    counts = verify_package(package)
    archive = output/"pcb_revamp_phase2.zip"
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                z.write(path,path.relative_to(package))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Package ZIP failed CRC.")
    nb = output/"PCB_Revamp_Phase2_Kaggle.ipynb"
    from revamp.build_resume_notebook import build as resume_notebook
    resume_notebook(package,nb)
    write_json(output/"build_report.json", {"status":"packaged", "counts":counts, "archive_bytes":archive.stat().st_size,
        "archive_sha256":sha256(archive), "notebook_sha256":sha256(nb), "package_hashes_sha256":sha256(package/"package_hashes.json"),
        "builder_sha256":sha256(Path(__file__)), "test_images_included":0,
        "local_pretrained_files":{"origin":"Existing project weights, 80-class metadata checked locally; no upstream checksum attestation.",
                                 "files":{name:sha256(package/"weights"/name) for name in weight_paths}}})
    print((output/"build_report.json").read_text())
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--background-review",type=Path)
    args = parser.parse_args()
    build(args.root,args.prepared,args.output,args.background_review)
