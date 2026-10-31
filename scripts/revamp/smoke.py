"""Tiny training-set memorization and fresh-process resume check, NOT evaluation.

Uses six training crops, one per class. No validation or test images are used.
The deliberately repeated train/eval sample tests whether the pipeline can learn.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from revamp.runtime import digest, verify_package

RECIPE = dict(epochs=80, patience=100, imgsz=320, batch=6, workers=0,
              optimizer="AdamW", lr0=.003, lrf=.1, nbs=6, cos_lr=True,
              warmup_epochs=0, seed=42, deterministic=True, amp=False,
              mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0, scale=0,
              fliplr=0, flipud=0, hsv_h=0, hsv_s=0, hsv_v=0,
              close_mosaic=0, save=True, save_period=-1, plots=False)


def signature(package):
    import torch
    import ultralytics
    return hashlib.sha256((digest(Path(package)/"package_hashes.json") + json.dumps(RECIPE, sort_keys=True)
                           + torch.__version__ + ultralytics.__version__).encode()).hexdigest()


def worker(package, data, run, resume, device):
    import torch
    from ultralytics import YOLO
    from revamp import checkpoints as ck
    torch.set_num_threads(4)
    run = Path(run)
    expected = signature(package)
    if resume:
        if json.loads((run/"signature.json").read_text())["sha256"] != expected:
            raise ValueError("Checkpoint signature differs from dataset/recipe/environment.")
        raw = torch.load(run/"training/weights/last.pt", map_location="cpu", weights_only=False)
        completed = int(raw["epoch"]) + 1
        if completed < 1 or raw["optimizer"] is None:
            raise ValueError("Expected a completed epoch with resumable optimizer state.")
        del raw
        ck.trim_results(run, completed)
        if completed >= RECIPE["epochs"]:
            print("All epochs already saved; rebuilding the report without retraining.", flush=True)
            return
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run/"signature.json").write_text(json.dumps({"sha256":expected, "recipe":RECIPE}))
    model = YOLO(str(run/"training/weights/last.pt" if resume else Path(package)/"weights/yolov8n.pt"))
    def save(trainer):
        ck.save_checkpoint(trainer, run)
    def pause(trainer):
        if not resume and trainer.epoch == 0:
            raise ck.TrainingPaused()
    def restored(trainer):
        ck.restore_trainer_state(trainer,run)
        if resume:
            if trainer.start_epoch != completed or not trainer.optimizer.state:
                raise RuntimeError("Optimizer/epoch was not restored.")
            (run/"resume_verified.json").write_text(json.dumps({"start_epoch":trainer.start_epoch,
                "optimizer_state_entries":len(trainer.optimizer.state), "pid":os.getpid()}))
    model.add_callback("on_model_save",save)
    model.add_callback("on_fit_epoch_end",pause)
    model.add_callback("on_train_start",restored)
    try:
        if resume:
            model.train(resume=True, device=device, data=str(data), save_dir=str(run/"training"))
        else:
            model.train(data=str(data), device=device, project=str(run), name="training", exist_ok=True, **RECIPE)
    except ck.TrainingPaused:
        return
    finally:
        trainer = getattr(model,"trainer",None)
        for loader in [getattr(trainer,"train_loader",None),getattr(trainer,"test_loader",None)]:
            if callable(getattr(loader,"close",None)):
                loader.close()


def check(package, output, device="cpu", checkpoint=None, resume_only=False):
    """Restart-safe coordinator. The last atomically published ZIP is authoritative."""
    import fcntl
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/"run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This run is already active; do not launch a second worker.") from error
        return _check(package, output, device, checkpoint, resume_only)


def checkpoint_archive(path, output):
    """Accept a downloaded ZIP or a Kaggle-expanded checkpoint directory."""
    import zipfile
    from revamp.runtime import safe_path
    path = Path(path).resolve()
    if path.is_file() and path.suffix == ".zip":
        return path
    if path.name == "checkpoint_manifest.json":
        path = path.parent
    record = json.loads((path/"checkpoint_manifest.json").read_text())
    for name, expected in record["sha256"].items():
        if digest(safe_path(path,name)) != expected:
            raise ValueError("Changed expanded checkpoint: " + name)
    target = Path(output)/"imported_checkpoint.zip"
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in [*record["sha256"],"checkpoint_manifest.json"]:
            z.write(safe_path(path,name),name)
    temporary.replace(target)
    return target


def _check(package, output, device, checkpoint, resume_only):
    import yaml
    from revamp import checkpoints as ck
    package, output = Path(package).resolve(), Path(output).resolve()
    verify_package(package)
    expected = signature(package)
    marker = output/"run_signature.json"
    if marker.exists() and json.loads(marker.read_text())["sha256"] != expected:
        raise ValueError("Existing run uses another package, recipe or environment.")
    if (output/"report.json").exists():
        if not marker.exists():
            raise ValueError("Completed run lacks a matching resume marker; use a new output directory.")
        report = json.loads((output/"report.json").read_text())
        if report["package_hashes_sha256"] != digest(package/"package_hashes.json"):
            raise ValueError("Report/package mismatch.")
        if report["status"] != "passed":
            raise RuntimeError("Completed learning check failed; inspect its report rather than silently retraining.")
        print("Already complete; reusing report without training.", flush=True)
        return report
    first, fresh = output/"first_run", output/"fresh_run"
    local = ck.checkpoint_path(fresh) if ck.checkpoint_path(fresh).exists() else ck.checkpoint_path(first)
    if checkpoint and local.exists():
        raise ValueError("Local progress exists. Clear RESTORE_FROM to resume it; use a new run folder for an imported checkpoint.")
    if resume_only and not checkpoint and not local.exists():
        raise RuntimeError("Resume requested but no checkpoint exists. No training was started.")
    if checkpoint:
        imported = checkpoint_archive(checkpoint, output)
        ck.restore_checkpoint(imported, fresh, expected)
        # Publish the verified imported archive locally before launching any work.
        shutil.copyfile(imported, ck.checkpoint_path(fresh))
        local = ck.checkpoint_path(fresh)
    marker.write_text(json.dumps({"sha256":expected,"recipe":RECIPE}))
    manifest = json.loads((package/"manifest.json").read_text())
    selected = {}
    for row in manifest["views"]:
        if row["dataset"]=="source" and row["split"]=="train" and row["kind"]=="crop":
            classes = {int(box[0]) for box in row["labels"]}
            if len(classes)==1:
                selected.setdefault(next(iter(classes)),row)
    if set(selected) != set(range(6)):
        raise ValueError("Cannot form a six-class learning check from training crops.")
    data = output/"tiny"; (data/"images").mkdir(parents=True,exist_ok=True); (data/"labels").mkdir(exist_ok=True)
    for cls, row in sorted(selected.items()):
        for key, destination in [("image",data/f"images/{cls}.jpg"),("label",data/f"labels/{cls}.txt")]:
            if destination.exists():
                if digest(destination) != row[key+"_sha256"]:
                    raise ValueError("Changed tiny learning input: " + str(destination))
            else:
                shutil.copyfile(package/row[key],destination)
    cfg = data/"data.yaml"
    cfg.write_text(yaml.safe_dump({"path":str(data), "train":"images", "val":"images", "names":manifest["names"]}))
    env = dict(os.environ, YOLO_CONFIG_DIR=str(output/"yolo_settings"), WANDB_MODE="disabled", COMET_MODE="DISABLED")
    command = [sys.executable,"-m","revamp.smoke","--package",str(package),"--worker","--data",str(cfg),"--device",str(device)]
    start = time.monotonic()
    if not local.exists():
        if first.exists():
            # An interruption before the first checkpoint has no epoch to resume.
            # Preserve that partial directory instead of overwriting its files.
            first.rename(output/f"incomplete_first_run_{time.time_ns()}")
        print("Starting tiny check; the first checkpoint will be saved after epoch 1.", flush=True)
        with (output/"pause.log").open("a") as stream:
            subprocess.run(command+["--output",str(first)], env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        local = ck.checkpoint_path(first)
    ck.restore_checkpoint(local,fresh,expected)
    if local != ck.checkpoint_path(fresh):
        shutil.copyfile(local,ck.checkpoint_path(fresh))
    if first.exists():
        first.rename(output/f"preserved_first_run_{time.time_ns()}")
    print("Continuing from the last complete checkpoint. Progress: " + str(output/"resume.log"),flush=True)
    with (output/"resume.log").open("a") as stream:
        subprocess.run(command+["--output",str(fresh),"--resume"],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    with (fresh/"training/results.csv").open() as stream:
        rows = list(csv.DictReader(stream,skipinitialspace=True))
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(1,RECIPE["epochs"]+1)):
        raise RuntimeError("Resumed CSV has missing or repeated epochs.")
    first_loss = sum(float(r["train/box_loss"]) for r in rows[:5])/5
    last_loss = sum(float(r["train/box_loss"]) for r in rows[-5:])/5
    ap50 = max(float(r["metrics/mAP50(B)"]) for r in rows)
    learned = last_loss < .85*first_loss and ap50 >= .5
    report = {"status":"passed" if learned else "failed_learning_gate", "seconds":round(time.monotonic()-start,1),
              "scope":"Six TRAINING crops evaluated on themselves. Functional evidence ONLY; not model accuracy.",
              "recipe":RECIPE, "package_hashes_sha256":digest(package/"package_hashes.json"),
              "selected_parent_ids":[r["parent_id"] for r in selected.values()],
              "checks":["Real training paused after one completed epoch", "Optimizer retained in checksummed ZIP",
                        "Resume in separate Python process and fresh folder with original run moved away",
                        "Optimizer restored at saved epoch (or final checkpoint reused)", "CSV has each planned epoch exactly once"],
              "learning_gate":{"first_5_box_loss":first_loss,"last_5_box_loss":last_loss,"best_train_sample_ap50":ap50,
                               "required":"Last-five mean box loss <85% of first-five; memorized-sample AP50 >=0.50"}}
    report_tmp = output/"report.tmp"
    report_tmp.write_text(json.dumps(report,indent=2)+"\n")
    report_tmp.replace(output/"report.json")
    print(json.dumps(report,indent=2))
    if not learned:
        raise RuntimeError("Tiny learning check failed. Inspect logs; do not start full training.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--data",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--resume-only",action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.package,args.data,args.output,args.resume,args.device)
    else:
        check(args.package,args.output,args.device,args.checkpoint,args.resume_only)
