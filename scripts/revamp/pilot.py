"""Bounded specialist pilot: held-out validation, baseline comparison, epoch recovery."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time

from revamp import checkpoints as ck
from revamp.runtime import NAMES, configure, digest, verify_package
from revamp.smoke import checkpoint_archive


def recipe(domain):
    # Full precision avoids introducing an AMP scaler that our checkpoint helper
    # does not yet persist. No stochastic transforms during the timing pilot.
    return dict(epochs=3, patience=10, imgsz=960 if domain == 'source' else 640,
                batch=4 if domain == 'source' else 8, workers=0, optimizer='AdamW',
                lr0=.0005, lrf=.1, nbs=4 if domain == 'source' else 8,
                cos_lr=True, warmup_epochs=0, seed=42, deterministic=True,
                amp=False, mosaic=0., mixup=0., copy_paste=0., cutmix=0.,
                degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                fliplr=0., flipud=0., hsv_h=0., hsv_s=0., hsv_v=0., bgr=0.,
                augmentations=[], close_mosaic=0, save=True, save_period=-1,
                plots=False, cache=False)


def environment():
    import platform
    import torch
    import torchvision
    import ultralytics
    import numpy
    return dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, ultralytics=ultralytics.__version__,
                numpy=numpy.__version__, cuda=torch.version.cuda)


def identity(package, domain):
    record = dict(package_sha256=digest(Path(package)/'package_hashes.json'),
                  domain=domain, recipe=recipe(domain), initialization='weights/yolov8s.pt',
                  environment=environment(), code={name:digest(Path(__file__).with_name(name))
                    for name in ['pilot.py','checkpoints.py','runtime.py','smoke.py']})
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(), record


def atomic_json(path, record):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def metrics_record(result):
    box = result.box
    return dict(map50=float(box.map50), map50_95=float(box.map),
                precision=float(box.mp), recall=float(box.mr),
                per_class={NAMES[int(c)]:dict(ap50=float(box.ap50[i]),
                    ap50_95=float(box.ap[i]), precision=float(box.p[i]), recall=float(box.r[i]))
                    for i,c in enumerate(box.ap_class_index)},
                speed_ms_per_image={k:float(v) for k,v in result.speed.items()})


def evaluate(weights, data, run, name, settings, device, expected):
    from ultralytics import YOLO
    path = run/(name+'.json')
    key = dict(run_signature=expected, weights_sha256=digest(weights), imgsz=settings['imgsz'],
               batch=settings['batch'], conf=.001, iou=.7, max_det=300, split='val')
    if path.exists():
        record = json.loads(path.read_text())
        if record['identity'] != key:
            raise ValueError('Saved evaluation differs: '+str(path))
        return record
    start=time.monotonic()
    model=YOLO(str(weights))
    if list(model.names.values()) != NAMES:
        raise ValueError('Baseline/candidate class order differs.')
    result=model.val(data=str(data),split='val',imgsz=settings['imgsz'],batch=settings['batch'],
                     device=device,workers=0,half=False,conf=.001,iou=.7,max_det=300,
                     augment=False,plots=False,save_json=False,project=str(run/'evaluation'),
                     name=name,exist_ok=True,verbose=True)
    record=dict(identity=key,metrics=metrics_record(result),seconds=time.monotonic()-start)
    atomic_json(path,record)
    return record


def assert_no_albumentations(dataset):
    """Fail if the installed trainer silently enables its optional default transforms."""
    def visit(transform):
        if type(transform).__name__ == 'Albumentations':
            applied=getattr(transform,'transform',None)
            if applied is not None and getattr(applied,'transforms',[]):
                raise RuntimeError('Unexpected Albumentations transforms enabled.')
        for child in getattr(transform,'transforms',[]):
            visit(child)
    visit(dataset.transforms)


def train(package, data, run, domain, device, expected, resume):
    import torch
    from ultralytics import YOLO
    settings=recipe(domain)
    completed=0
    if resume:
        raw=torch.load(run/'training/weights/last.pt',map_location='cpu',weights_only=False)
        completed=int(raw['epoch'])+1
        if completed < 1 or raw['optimizer'] is None:
            raise ValueError('Checkpoint has no completed epoch/optimizer.')
        del raw
        ck.trim_results(run,completed)
        if completed >= settings['epochs']:
            print('All pilot epochs saved; evaluation/report recovery only.',flush=True)
            return
    model=YOLO(str(run/'training/weights/last.pt' if resume else package/'weights/yolov8s.pt'))
    started=[None]
    def on_start(trainer):
        assert_no_albumentations(trainer.train_loader.dataset)
        ck.restore_trainer_state(trainer,run)
        if resume and (trainer.start_epoch != completed or not trainer.optimizer.state):
            raise RuntimeError('Optimizer/epoch recovery failed.')
        print(f'PILOT READY: epoch {trainer.start_epoch+1}; optional augmentations disabled',flush=True)
    def epoch_start(trainer):
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        started[0]=time.monotonic()
    def save(trainer):
        record=json.loads((run/'signature.json').read_text())
        record.setdefault('epoch_measurements',[]).append(dict(epoch=trainer.epoch+1,
            train_and_val_seconds=time.monotonic()-started[0],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else None))
        atomic_json(run/'signature.json',record)
        ck.save_checkpoint(trainer,run)
    model.add_callback('on_train_start',on_start)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_model_save',save)
    try:
        if resume:
            model.train(resume=True,device=device,data=str(data),save_dir=str(run/'training'),augmentations=[])
        else:
            model.train(data=str(data),device=device,project=str(run),name='training',exist_ok=True,**settings)
    finally:
        trainer=getattr(model,'trainer',None)
        for loader in [getattr(trainer,'train_loader',None),getattr(trainer,'test_loader',None)]:
            if callable(getattr(loader,'close',None)):loader.close()


def run(package, output, domain='source', device='0', checkpoint=None, resume_only=False):
    import fcntl
    package,output=Path(package).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Pilot already active.') from error
        return _run(package,output,domain,device,checkpoint,resume_only)


def _run(package,output,domain,device,checkpoint,resume_only):
    import torch
    import ultralytics
    if ultralytics.__version__ != '8.4.142':raise ValueError('Use pinned Ultralytics 8.4.142.')
    if device != 'cpu' and not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU before the pilot.')
    torch.set_num_threads(4)
    verify_package(package)
    expected,details=identity(package,domain)
    run=output/'pilot_run';archive=ck.checkpoint_path(run)
    if checkpoint and archive.exists():raise ValueError('Local progress exists; clear RESTORE_FROM.')
    if resume_only and not checkpoint and not archive.exists():raise RuntimeError('Resume requested but no checkpoint exists.')
    marker=run/'signature.json'
    if marker.exists() and json.loads(marker.read_text())['sha256'] != expected:
        raise ValueError('Different package, recipe, code or environment; use a new output folder.')
    if checkpoint:
        imported=checkpoint_archive(checkpoint,output)
        ck.restore_checkpoint(imported,run,expected)
        shutil.copyfile(imported,archive)
    if archive.exists():ck.restore_checkpoint(archive,run,expected)
    if not marker.exists():
        run.mkdir(parents=True,exist_ok=True)
        atomic_json(marker,dict(sha256=expected,identity=details,epoch_measurements=[]))
    report_path=output/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report['signature'] != expected or not archive.exists() or report['checkpoint_sha256'] != digest(archive):
            raise ValueError('Completed report/checkpoint mismatch.')
        print('Pilot already complete; reusing report.',flush=True)
        return report
    configs=configure(package,output/'configs')
    data=configs['source_views' if domain=='source' else 'target']
    settings=recipe(domain)
    baseline=evaluate(package/f'weights/{domain}_baseline.pt',data,run,
                      f'baseline_{domain}_validation',settings,device,expected)
    # Interrupted epoch files cannot override the authoritative atomic ZIP.
    if not archive.exists() and (run/'training').exists():
        (run/'training').rename(run/f'incomplete_training_{time.time_ns()}')
    train(package,data,run,domain,device,expected,archive.exists())
    ck.restore_checkpoint(archive,run,expected)
    with (run/'training/results.csv').open() as stream:
        rows=list(csv.DictReader(stream,skipinitialspace=True))
    if [int(float(r['epoch'])) for r in rows] != list(range(1,settings['epochs']+1)):
        raise RuntimeError('Pilot epochs incomplete or duplicated.')
    candidate=evaluate(run/'training/weights/best.pt',data,run,'candidate_validation',settings,device,expected)
    measurements=json.loads(marker.read_text())['epoch_measurements']
    if [r['epoch'] for r in measurements] != list(range(1,settings['epochs']+1)):
        raise RuntimeError('Pilot timing history incomplete or duplicated.')
    manifest=json.loads((package/'manifest.json').read_text())
    domain_rows=[r for r in manifest['views'] if r['dataset']==domain]
    steady=statistics.median(m['train_and_val_seconds'] for m in measurements[1:] or measurements)
    report=dict(status='pilot_completed',signature=expected,checkpoint_sha256=digest(archive),
                identity=details,device=torch.cuda.get_device_name(0) if device!='cpu' else 'cpu',
                counts={s:sum(r['split']==s for r in domain_rows) for s in ['train','val']},
                baseline=baseline,candidate=candidate,
                delta={k:candidate['metrics'][k]-baseline['metrics'][k] for k in ['map50','map50_95','precision','recall']},
                epoch_measurements=measurements,steady_epoch_seconds_excluding_checkpoint=steady,
                lower_bound_60_epoch_hours=steady*60/3600,
                limitations=['Three epochs are a pipeline/timing pilot, not a final accuracy result.',
                    'Estimate excludes checkpoint ZIP writing, setup, final evaluation and possible later recipe changes.',
                    'Validation is separate from training; no development-test images used.',
                    'Target missing-hole performance unsupported; target board independence unknown.',
                    'No normal full-board benchmark; no production model replaced.'])
    atomic_json(report_path,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--domain',choices=['source','target'],default='source')
    parser.add_argument('--device',default='0')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--resume-only',action='store_true')
    args=parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(args.output/'yolo_settings'))
    os.environ.setdefault('WANDB_MODE','disabled')
    os.environ.setdefault('COMET_MODE','DISABLED')
    run(args.package,args.output,args.domain,args.device,args.checkpoint,args.resume_only)

if __name__=='__main__':main()
