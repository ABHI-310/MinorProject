import json
from pathlib import Path

from config import ROOT
from revamp.paths import historical_path


def test_historical_paths_use_longest_matching_prefix(tmp_path):
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs/legacy_paths.json").write_text(json.dumps({"old":"archive/old","old/special":"data/new"}))
    assert historical_path(tmp_path,"old/special/image.jpg") == tmp_path/"data/new/image.jpg"
    assert historical_path(tmp_path,"old/other") == tmp_path/"archive/old/other"
    assert historical_path(tmp_path,"older/file") == tmp_path/"older/file"


def test_maintained_entrypoints_and_artifacts_have_canonical_locations():
    for relative in ["scripts/manage.py","src/backend/main.py","configs/datasets.json",
                     "notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb",
                     "artifacts/models/source/model.pt","artifacts/models/target/model.pt",
                     "data/application/maintenance.db","data/prepared/source/data.yaml",
                     "data/prepared/target/data.yaml"]:
        assert (ROOT/relative).is_file(),relative
    assert not (ROOT/"phase1_fast").exists()
    assert not (ROOT/"phase2").exists()


def test_dataset_catalog_only_references_existing_inputs():
    document=json.loads((ROOT/"configs/datasets.json").read_text())
    for dataset in document["datasets"]:
        for key in ("prepared","original","model_folder","metrics"):
            assert (ROOT/dataset[key]).exists(),dataset[key]
