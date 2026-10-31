"""Frozen checkpoint helpers adapted from accuracy/train_recovery.py.

Per-run archive names avoid collisions between specialist experiments.
Use only trusted, locally produced PyTorch checkpoints.
"""
from pathlib import Path
import hashlib,json,os,zipfile,csv,io,tempfile

class TrainingPaused(Exception):
 """Leave the training loop before final_eval strips the optimizer."""


def checkpoint_path(run):
 return Path(run).with_name(Path(run).name + '_checkpoint.zip')


def restore_checkpoint(archive,run,signature=None):
 """Verify the entire backup before replacing any run files. Own checkpoints only."""
 archive=Path(archive);run=Path(run).resolve()
 with zipfile.ZipFile(archive) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise RuntimeError('Duplicate checkpoint archive entries.')
  for name in names:
   if Path(name).is_absolute() or '\\' in name or '..' in Path(name).parts or not (run/name).resolve().is_relative_to(run):raise RuntimeError('Unsafe checkpoint path.')
  for info in z.infolist():
   if ((info.external_attr >> 16) & 0o170000) == 0o120000:raise RuntimeError('Symlink in checkpoint.')
  if z.testzip() is not None:raise RuntimeError('Checkpoint ZIP is damaged.')
  record=json.loads(z.read('signature.json'))
  expected=signature
  if (run/'signature.json').exists():
   existing=json.loads((run/'signature.json').read_text())['sha256']
   if expected is not None and existing!=expected:raise RuntimeError('Run signature mismatch.')
   expected=existing
  if expected is not None and record['sha256']!=expected:raise RuntimeError('Checkpoint dataset/recipe mismatch.')
  for required in ['training/weights/last.pt','training/weights/best.pt','training/results.csv','training/args.yaml','training/trainer_state.pt','checkpoint_manifest.json']:
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

