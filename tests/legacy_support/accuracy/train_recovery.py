"""Kaggle entry point: one resumable six-class recovery run; validation only."""
from pathlib import Path
import hashlib,json,os,shutil,time,zipfile,csv,io,tempfile
import yaml

RECIPE=dict(epochs=80,patience=15,imgsz=960,batch=8,workers=2,optimizer='AdamW',lr0=.0005,lrf=.01,
 cos_lr=True,warmup_epochs=3.,warmup_bias_lr=0.,nbs=32,freeze=0,seed=42,deterministic=True,amp=True,
 mosaic=0.,mixup=0.,degrees=5.,translate=.03,scale=.1,fliplr=.5,flipud=.5,
 hsv_h=0.,hsv_s=.1,hsv_v=.1,close_mosaic=0,save=True,save_period=10,plots=True)


class TrainingPaused(Exception):
 """Leave the training loop before final_eval strips the optimizer."""


def checkpoint_path(run):
 return Path(run).parent/'six_class_recovery_checkpoint.zip'


def restore_checkpoint(archive,run,signature=None):
 """Verify the entire backup before replacing any run files. Own checkpoints only."""
 archive=Path(archive);run=Path(run).resolve()
 with zipfile.ZipFile(archive) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise RuntimeError('Duplicate checkpoint archive entries.')
  for name in names:
   if not (run/name).resolve().is_relative_to(run):raise RuntimeError('Unsafe checkpoint path.')
  if z.testzip() is not None:raise RuntimeError('Checkpoint ZIP is damaged.')
  record=json.loads(z.read('signature.json'))
  expected=signature
  if (run/'signature.json').exists():
   existing=json.loads((run/'signature.json').read_text())['sha256']
   if expected is not None and existing!=expected:raise RuntimeError('Run signature mismatch.')
   expected=existing
  if expected is not None and record['sha256']!=expected:raise RuntimeError('Checkpoint dataset/recipe mismatch.')
  for required in ['training/weights/last.pt','training/weights/best.pt','training/results.csv','training/args.yaml']:
   if required not in names:raise RuntimeError('Incomplete checkpoint: '+required)
  if 'checkpoint_manifest.json' in names:
   manifest=json.loads(z.read('checkpoint_manifest.json'))
   actual=set(names)-{'checkpoint_manifest.json'}
   if actual!=set(manifest['sha256']):raise RuntimeError('Checkpoint file list mismatch.')
   for name,digest in manifest['sha256'].items():
    if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise RuntimeError('Checkpoint checksum mismatch: '+name)
  run.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=run.parent,prefix='checkpoint-restore-') as temporary:
   staging=Path(temporary)
   for name in names:
    if name.endswith('/'):continue
    target=staging/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
   for name in names:
    if name.endswith('/'):continue
    target=run/name;target.parent.mkdir(parents=True,exist_ok=True);os.replace(staging/name,target)
 print('Verified checkpoint restored:',archive,flush=True)


def save_checkpoint(trainer,run):
 """Publish a complete, checksummed backup atomically after every saved epoch."""
 import torch,random,numpy as np
 run=Path(run);epoch=int(trainer.epoch)
 state={'epoch':epoch,'training_finished':bool(trainer.stop),
  'stopper':{key:getattr(trainer.stopper,key) for key in ['best_fitness','best_epoch','possible_stop','patience']},
  'scheduler':trainer.scheduler.state_dict(),'python_rng':random.getstate(),
  'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
  'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
 state_file=run/'training/trainer_state.pt';state_file.parent.mkdir(parents=True,exist_ok=True)
 state_tmp=state_file.with_suffix('.tmp');torch.save(state,state_tmp);os.replace(state_tmp,state_file)
 files=[Path(trainer.last),Path(trainer.best),run/'training/args.yaml',run/'training/results.csv',run/'signature.json',state_file]
 files.extend(p for p in [run/'baseline_source_validation.json',run/'baseline_target_validation.json'] if p.is_file())
 if any(not p.is_file() for p in files):raise RuntimeError('Cannot publish an incomplete checkpoint.')
 archive=checkpoint_path(run);temporary=archive.with_suffix('.tmp')
 try:
  hashes={}
  with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for p in files:
    payload=p.read_bytes();name=p.relative_to(run).as_posix();hashes[name]=hashlib.sha256(payload).hexdigest();z.writestr(name,payload)
   z.writestr('checkpoint_manifest.json',json.dumps({'format':'pcb_pause_resume_v1','completed_epochs':epoch+1,'training_finished':state['training_finished'],'sha256':hashes}))
  with zipfile.ZipFile(temporary) as z:
   if z.testzip() is not None:raise RuntimeError('Checkpoint verification failed.')
  os.replace(temporary,archive)
 finally:temporary.unlink(missing_ok=True)
 print(f'CHECKPOINT SAVED: epoch {epoch+1} — {archive}',flush=True)
 return archive


def restore_trainer_state(trainer,run):
 import torch,random,numpy as np
 path=Path(run)/'training/trainer_state.pt'
 if not path.exists() or not trainer.resume:return
 state=torch.load(path,map_location='cpu',weights_only=False)
 if state['epoch']+1!=trainer.start_epoch:raise RuntimeError('Checkpoint and trainer state refer to different epochs.')
 for key,value in state['stopper'].items():setattr(trainer.stopper,key,value)
 trainer.scheduler.load_state_dict(state['scheduler'])
 random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
 if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])


def trim_results(run,completed_epochs):
 """Discard CSV rows written by an epoch whose weights were not committed."""
 path=Path(run)/'training/results.csv'
 if not path.exists():return
 with path.open(newline='') as stream:
  reader=csv.DictReader(stream,skipinitialspace=True);fields=reader.fieldnames
  rows=[row for row in reader if int(float(row['epoch']))<=completed_epochs]
 output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 path.write_text(output.getvalue())


def configure(data):
 manifest=json.loads((data/'six_class_manifest.json').read_text())
 for domain in ['source','target']:
  info=json.loads((data/f'{domain}_info.json').read_text())
  assert hashlib.sha256((data/f'{domain}_best.pt').read_bytes()).hexdigest()==info['checkpoint_sha256']
 configs={}
 for name,prefix in [('mixed',None),('source','pku_'),('target','target_')]:
  val=[str(data/p) for p in manifest['images']['val'] if prefix is None or Path(p).name.startswith(prefix)]
  text=data/f'val_{name}.txt';text.write_text('\n'.join(val)+'\n')
  cfg=data/f'data_{name}.yaml';cfg.write_text(yaml.safe_dump({'path':str(data),'train':'images/train','val':str(text),'names':dict(enumerate(manifest['names']))}))
  configs[name]=cfg
 return manifest,configs


def main(data,run=Path('/kaggle/working/six_class_recovery_run'),device=0,control=None,resume_only=False,pause_after_epochs=None):
 import torch
 from ultralytics import YOLO
 data=Path(data).resolve();run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True)
 manifest,configs=configure(data);names=manifest['names']
 signature=hashlib.sha256((data/'six_class_manifest.json').read_bytes()+json.dumps(RECIPE,sort_keys=True).encode()).hexdigest()
 marker=run/'signature.json';model=None
 # Resume must never silently fall back to the source model.
 if resume_only:
  if checkpoint_path(run).is_file() and not (run/'training_complete.json').exists():restore_checkpoint(checkpoint_path(run),run,signature)
  if not (run/'training/weights/last.pt').is_file():raise RuntimeError('Resume refused: checkpoint missing. Attach your downloaded recovery ZIP and run setup. No new training started.')
 if marker.exists():assert json.loads(marker.read_text())['sha256']==signature,'Different dataset or recipe; do not resume this folder.'
 marker.write_text(json.dumps({'sha256':signature,'recipe':RECIPE},indent=2))
 def export(status):
  (run/'status.json').write_text(json.dumps({'status':status,'test_used':False,'recipe':RECIPE},indent=2))
  destination=run.parent/'six_class_recovery_results.zip';temporary=destination.with_suffix('.tmp')
  with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for p in sorted(run.rglob('*')):
    if p.is_file() and not p.name.startswith('epoch'):z.write(p,p.relative_to(run.parent))
  temporary.replace(destination);print('Saved results:',destination,flush=True)
 def recovery(trainer):
  archive=save_checkpoint(trainer,run)
  if control is not None:control.checkpoint_saved(int(trainer.epoch)+1,archive)
 def pause_at_boundary(trainer):
  requested=(control is not None and control.pause_requested())
  requested|=pause_after_epochs is not None and int(trainer.epoch)-int(trainer.start_epoch)+1>=pause_after_epochs
  if requested and not trainer.stop:raise TrainingPaused()
 def estimate(trainer):
  if int(trainer.epoch)+1 in [3,10,20,40]:print('Approximate remaining training minutes:',round(trainer.epoch_time*(trainer.epochs-trainer.epoch-1)/60,1),'plus evaluation',flush=True)
 def evaluate(weights,domain,size,label):
  path=run/(label+'.json');weight_hash=hashlib.sha256(Path(weights).read_bytes()).hexdigest()
  if path.exists():
   saved=json.loads(path.read_text());assert saved['checkpoint_sha256']==weight_hash;return saved
  model=YOLO(str(weights));assert [model.names[i] for i in range(6)]==names
  classes=list(range(6)) if domain=='source' else [1,2,3,4,5]
  metrics=model.val(data=str(configs[domain]),split='val',imgsz=size,batch=8,device=device,workers=2,
     classes=classes,conf=.001,plots=False,project=str(run),name=label,exist_ok=True)
  result={'checkpoint_sha256':weight_hash,'domain':domain,'split':'validation','image_size':size,
    'mAP50':float(metrics.box.map50),'mAP50_95':float(metrics.box.map),'precision':float(metrics.box.mp),'recall':float(metrics.box.mr),
    'per_class':{names[int(c)]:{'mAP50':float(metrics.box.ap50[i]),'mAP50_95':float(metrics.box.ap[i].mean())} for i,c in enumerate(metrics.box.ap_class_index)}}
  path.write_text(json.dumps(result,indent=2));return result
 try:
  if control is not None and control.pause_requested():raise TrainingPaused()
  baseline_source=evaluate(data/'source_best.pt','source',960,'baseline_source_validation')
  baseline_target=evaluate(data/'target_best.pt','target',640,'baseline_target_validation')
  training=run/'training';last=training/'weights/last.pt';best=training/'weights/best.pt';done=run/'training_complete.json'
  if not done.exists():
   interrupted=False
   finished=False
   if last.exists():
    saved=torch.load(last,map_location='cpu',weights_only=False);interrupted=saved.get('optimizer') is not None and saved.get('epoch',-1)>=0
    completed=int(saved.get('epoch',-1))+1
    if interrupted:trim_results(run,completed)
    state_file=training/'trainer_state.pt'
    if state_file.exists():
     state=torch.load(state_file,map_location='cpu',weights_only=False)
     finished=state.get('training_finished',False) and state['epoch']+1==completed
    finished|=completed>=RECIPE['epochs'];del saved
   elif (training/'results.csv').exists():raise RuntimeError('Training history exists but last.pt is missing. Restore your checkpoint; refusing to restart.')
   if control is not None and control.pause_requested():raise TrainingPaused()
   model=YOLO(str(last if interrupted else data/'source_best.pt'))
   model.add_callback('on_model_save',recovery);model.add_callback('on_fit_epoch_end',estimate)
   model.add_callback('on_fit_epoch_end',pause_at_boundary)
   model.add_callback('on_train_start',lambda trainer:restore_trainer_state(trainer,run))
   if interrupted and not finished:model.train(resume=True,device=device,save_dir=str(training),data=str(configs['mixed']))
   elif not last.exists():model.train(data=str(configs['mixed']),project=str(run),name='training',exist_ok=True,device=device,**RECIPE)
   assert best.is_file();done.write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest()}))
  assert hashlib.sha256(best.read_bytes()).hexdigest()==json.loads(done.read_text())['checkpoint_sha256']
  candidate_source=evaluate(best,'source',960,'candidate_source_validation')
  candidate_target=evaluate(best,'target',960,'candidate_target_validation')
  passed=(candidate_source['mAP50']>=baseline_source['mAP50']-.02 and candidate_target['mAP50']>=baseline_target['mAP50']-.02
    and candidate_source['mAP50_95']>=baseline_source['mAP50_95']-.02 and candidate_target['mAP50_95']>=baseline_target['mAP50_95']-.02
    and candidate_source['per_class']['missing_hole']['mAP50']>=baseline_source['per_class']['missing_hole']['mAP50']-.02)
  summary={'candidate_ready_for_review':passed,'rule':'Source and target validation mAP50 and mAP50:95 each within 2 percentage points of specialist baseline; missing-hole AP50 also protected. Passing does not prove field accuracy.',
   'baseline_source':baseline_source,'baseline_target':baseline_target,'candidate_source':candidate_source,'candidate_target':candidate_target,
   'no_automatic_production_replacement':True,'test_used':False,'limitations':json.loads((data/'sampling.json').read_text())['limitations']}
  (run/'summary.json').write_text(json.dumps(summary,indent=2));shutil.copy2(best,run/'candidate_six_class.pt')
  (run/'candidate_model_info.json').write_text(json.dumps({'model':'YOLOv8s','image_size':960,'classes':names,'evaluated_class_ids':list(range(6)),
    'checkpoint_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),'status':'validation candidate, not production model','method':'six-class mixed-domain supervised recovery'},indent=2))
  export('complete');print('Candidate passed review gates:',passed,flush=True)
  return {'status':'complete','results':str(run.parent/'six_class_recovery_results.zip')}
 except (TrainingPaused,KeyboardInterrupt):
  # Keep the last *complete* ZIP; a hard interrupt could have cut a raw .pt write short.
  archive=checkpoint_path(run)
  trainer=getattr(model,'trainer',None)
  if not archive.is_file() and trainer is not None and Path(trainer.last).is_file():
   # Interrupt during publication of the first ZIP: retry only a valid, completed checkpoint.
   raw=torch.load(trainer.last,map_location='cpu',weights_only=False)
   valid=raw.get('optimizer') is not None and raw.get('epoch',-1)==trainer.epoch
   del raw
   if valid:save_checkpoint(trainer,run)
  if archive.is_file() and not (run/'training_complete.json').exists():restore_checkpoint(archive,run,signature)
  export('paused')
  if archive.is_file():print('PAUSED. Download',archive,'before ending the Kaggle session. Resume will continue from its last completed epoch.',flush=True)
  else:print('PAUSED before the first completed epoch. No training checkpoint exists yet; Start training will reuse cached baseline checks.',flush=True)
  return {'status':'paused','checkpoint':str(archive) if archive.is_file() else None}
 except BaseException:
  try:export('interrupted_or_failed')
  except Exception as error:print('Results export failed:',error,'Checkpoints remain in',run)
  raise
 finally:
  trainer=getattr(model,'trainer',None)
  for loader in [getattr(trainer,'train_loader',None),getattr(trainer,'test_loader',None)]:
   if callable(getattr(loader,'close',None)):loader.close()
