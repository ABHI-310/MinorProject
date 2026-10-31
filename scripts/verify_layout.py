"""Verify relocated audit inputs and the unchanged Kaggle payload without training."""
import json
from pathlib import Path
from _bootstrap import ROOT
from revamp.audit import sha256
from revamp.paths import historical_path
from revamp.runtime import verify_package

def main():
    audit = ROOT / "reports/revamp/phase1_v1"
    record = json.loads((audit/"artifact_manifest.json").read_text())
    checked = 0
    for base, hashes in [(audit,record["sha256"]),(ROOT,record["audit_inputs_sha256"])]:
        for relative, expected in hashes.items():
            path = historical_path(base,relative)
            if sha256(path) != expected:
                raise ValueError("Historical input changed: " + relative)
            checked += 1
    for row in map(json.loads,(audit/"manifest.jsonl").read_text().splitlines()):
        for key in ("image","label"):
            if sha256(historical_path(ROOT,row[key])) != row[key+"_sha256"]:
                raise ValueError("Audited data changed: " + row[key])
            checked += 1
    package = ROOT / "artifacts/kaggle/phase2_v2"
    build = json.loads((package/"build_report.json").read_text())
    if sha256(package/"pcb_revamp_phase2.zip") != build["archive_sha256"]:
        raise ValueError("Kaggle ZIP changed")
    counts = verify_package(package/"package")
    result = {"status":"passed","historical_hashes_checked":checked,
              "kaggle_zip_unchanged":True,"package_counts":counts,"training_started":False}
    (ROOT/"reports/cleanup/integrity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
