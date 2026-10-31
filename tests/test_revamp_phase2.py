import copy
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from revamp.prepare import crop_labels, positive_window, negative_window
from revamp.runtime import NAMES, digest, extract_package, safe_path, verify_views, verify_package
from revamp.package import notebook, approved_background
from revamp import checkpoints


def parent(labels):
    return {"width":1000,"height":1000,"labels":labels,"image_sha256":"1"*64}


def test_crop_keeps_all_inside_boxes_and_rejects_partial_defect():
    row = parent([[0,.3,.3,.1,.1],[1,.55,.3,.1,.1]])
    assert len(crop_labels(row,(200,200,700,700))) == 2
    assert crop_labels(row,(200,200,550,700)) is None
    assert crop_labels(row,(0,0,100,100)) == []


def test_crop_geometry_and_margin():
    row = parent([[2,.5,.5,.1,.1]])
    assert crop_labels(row,(400,400,600,600)) == [[2,.5,.5,.5,.5]]
    assert crop_labels(row,(450,450,650,650)) is None
    window, labels = positive_window(row,640)
    assert window == (180,180,820,820)
    assert labels[0][0] == 2


def test_negative_window_never_intersects_expanded_box():
    assert negative_window(parent([[0,.5,.5,1,1]])) is None
    row = parent([[0,.8,.8,.1,.1]])
    assert negative_window(row) == (0,0,512,512)


@pytest.fixture
def dataset(tmp_path):
    rows=[]
    for domain in ("source","target"):
        for split in ("train","val"):
            for cls in (range(6) if domain=="source" else range(1,6)):
                name=f"{domain}_{split}_{cls}"
                image=tmp_path/f"images/{split}/{name}.jpg"
                label=tmp_path/f"labels/{split}/{name}.txt"
                image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(32,32),(len(rows)*9,0,0)).save(image)
                label.write_text(f"{cls} .5 .5 .25 .25\n")
                rows.append({"dataset":domain,"split":split,"kind":"full","parent_id":domain+"/"+name,
                    "group_id":split if domain=="source" else None,"image":str(image.relative_to(tmp_path)),
                    "label":str(label.relative_to(tmp_path)),"image_sha256":digest(image),"label_sha256":digest(label),
                    "width":32,"height":32,"labels":[[cls,.5,.5,.25,.25]]})
    return tmp_path,{"names":NAMES,"views":rows}


def test_valid_package_views(dataset):
    root,manifest=dataset
    assert verify_views(root,manifest)["source/train/full"] == 6


def test_package_rejects_extra_test_image_and_bad_hash_coverage(dataset):
    root,manifest=dataset
    (root/"manifest.json").write_text(json.dumps(manifest))
    (root/"weights").mkdir()
    for name in ("yolov8n.pt","yolov8s.pt","source_baseline.pt","target_baseline.pt"):
        (root/"weights"/name).write_bytes(b"fixture only")
    hashes={p.relative_to(root).as_posix():digest(p) for p in root.rglob("*") if p.is_file()}
    record=root/"package_hashes.json";record.write_text(json.dumps({"sha256":hashes}))
    verify_package(root)
    extra=root/"images/test/extra.jpg";extra.parent.mkdir();extra.write_bytes(b"excluded test")
    with pytest.raises(ValueError,match="Unlisted"):verify_package(root)
    del hashes[manifest["views"][0]["label"]]
    record.write_text(json.dumps({"sha256":hashes}))
    with pytest.raises(ValueError,match="coverage"):verify_package(root)


@pytest.mark.parametrize("mode",["parent","group","pixels","test","derived_val","geometry","hash","missing_class","empty"])
def test_reject_bad_views(dataset,mode):
    root,manifest=dataset
    rows=manifest["views"];train=rows[0];val=rows[6]
    if mode=="parent": val["parent_id"]=train["parent_id"]
    elif mode=="group": val["group_id"]=train["group_id"]
    elif mode=="pixels":
        (root/val["image"]).write_bytes((root/train["image"]).read_bytes());val["image_sha256"]=train["image_sha256"]
    elif mode=="test": val["split"]="test"
    elif mode=="derived_val":val["kind"]="crop"
    elif mode=="geometry":val["labels"][0][1]=.7
    elif mode=="hash":(root/train["label"]).write_text("changed")
    elif mode=="missing_class":rows.pop(0)
    elif mode=="empty":
        (root/train["label"]).write_text("");train["label_sha256"]=digest(root/train["label"]);train["labels"]=[]
    with pytest.raises(ValueError): verify_views(root,manifest)


@pytest.mark.parametrize("name",["../escape","/absolute","nested/../../escape","a\\b"])
def test_unsafe_path(tmp_path,name):
    with pytest.raises(ValueError):safe_path(tmp_path,name)


def test_bad_zip_and_existing_output(tmp_path):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:z.writestr("../outside","no")
    with pytest.raises(ValueError):extract_package(archive,tmp_path/"new")
    assert not (tmp_path/"new").exists()
    with pytest.raises(ValueError):extract_package(archive,tmp_path)


def test_notebook_cells_compile_and_never_include_full_training():
    nb=notebook("a"*64)
    for i,cell in enumerate(nb["cells"]):
        if cell["cell_type"]=="code":compile("".join(cell["source"]),f"cell{i}","exec")
    assert "model.train(" not in json.dumps(nb)
    assert "--constraint" in json.dumps(nb)


def test_review_hash_and_missing_decisions(tmp_path):
    folder=tmp_path/"review";folder.mkdir()
    row={"parent_id":"source/one","image_sha256":"a","split":"train","dataset":"source","labels":[]}
    candidates=folder/"candidates.json";candidates.write_text(json.dumps({"candidates":[row]}))
    review=tmp_path/"review.json"
    review.write_text(json.dumps({"reviewer":"test","candidates_sha256":digest(candidates),"decisions":[]}))
    with pytest.raises(ValueError):approved_background(tmp_path,review)
    decision={"parent_id":"source/one","image_sha256":"wrong","decision":"accept","reason":"reviewed"}
    record={"reviewer":"test","candidates_sha256":digest(candidates),"decisions":[decision]}
    review.write_text(json.dumps(record))
    with pytest.raises(ValueError):approved_background(tmp_path,review)


def test_checkpoint_names_do_not_collide(tmp_path):
    assert checkpoints.checkpoint_path(tmp_path/"source") != checkpoints.checkpoint_path(tmp_path/"target")


def test_checkpoint_roundtrip_and_signature(tmp_path):
    from types import SimpleNamespace
    import torch
    run=tmp_path/"source";training=run/"training";weights=training/"weights";weights.mkdir(parents=True)
    for name in ("last.pt","best.pt"):
        torch.save({"epoch":0,"optimizer":{"state":{1:{"step":1}}}},weights/name)
    (training/"args.yaml").write_text("epochs: 2\n")
    (training/"results.csv").write_text("epoch,loss\n1,1.0\n")
    (run/"signature.json").write_text(json.dumps({"sha256":"expected"}))
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters())
    trainer=SimpleNamespace(epoch=0,stop=False,last=weights/"last.pt",best=weights/"best.pt",
        stopper=SimpleNamespace(best_fitness=.1,best_epoch=1,possible_stop=False,patience=10),
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1))
    archive=checkpoints.save_checkpoint(trainer,run)
    fresh=tmp_path/"fresh"
    with pytest.raises(RuntimeError):checkpoints.restore_checkpoint(archive,fresh,"wrong")
    assert not fresh.exists()
    checkpoints.restore_checkpoint(archive,fresh,"expected")
    assert (fresh/"training/trainer_state.pt").exists()
    assert digest(fresh/"training/weights/last.pt")==digest(weights/"last.pt")
    for missing in ("checkpoint_manifest.json","training/trainer_state.pt"):
        bad=tmp_path/(Path(missing).stem+"_missing.zip")
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(bad,"w") as target:
            for name in source.namelist():
                if name != missing:target.writestr(name,source.read(name))
        with pytest.raises(RuntimeError,match="Incomplete"):checkpoints.restore_checkpoint(bad,tmp_path/"refused","expected")
        assert not (tmp_path/"refused").exists()
