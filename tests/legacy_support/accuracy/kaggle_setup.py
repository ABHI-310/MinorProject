"""Kaggle setup: accept split NumPy uploads, original ZIP, or extracted dataset."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,shutil,zipfile,stat,sys
import numpy as np


def digest_file(path):
 value=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
 return value.hexdigest()


def discover(inputs):
 queue=[(Path(inputs),0)];folders=[];archives=[];recoveries=[];uploads=[]
 while queue:
  folder,depth=queue.pop()
  for p in folder.iterdir():
   if p.is_file():
    if p.name=='six_class_manifest.json':folders.append(p.parent)
    elif p.name=='pcb_six_class_recovery.zip':archives.append(p)
    elif p.name=='six_class_recovery_checkpoint.zip':recoveries.append(p)
    elif p.name=='signature.json' and (p.parent/'training/weights/last.pt').is_file():recoveries.append(p.parent)
    elif p.name=='pcb_upload_manifest.json':uploads.append(p)
   elif p.is_dir() and p.name not in ['images','labels','node_modules'] and depth<5:queue.append((p,depth+1))
 return folders,archives,recoveries,uploads


def reconstruct(manifest_path,destination):
 manifest_path=Path(manifest_path);destination=Path(destination)
 doc=json.loads(manifest_path.read_text());assert doc['format']=='pcb_numpy_upload_v1','Wrong upload manifest'
 assert doc['parts'] and len({p['file'] for p in doc['parts']})==len(doc['parts'])
 if destination.exists() and destination.stat().st_size==doc['archive_bytes'] and digest_file(destination)==doc['archive_sha256']:
  print('Reusing verified local ZIP.',flush=True);return destination
 assert shutil.disk_usage(destination.parent).free>doc['archive_bytes']+1024**3,'Not enough free space to restore ZIP.'
 temporary=destination.with_suffix('.tmp');overall=hashlib.sha256();total=0
 try:
  with temporary.open('wb') as output:
   for i,part in enumerate(doc['parts'],1):
    source=manifest_path.parent/part['file']
    assert Path(part['file']).name==part['file'],'Invalid upload filename'
    assert source.is_file(),f'Missing upload part: {part["file"]}. Upload all parts and the JSON in the same Kaggle dataset.'
    array=np.load(source,mmap_mode='r',allow_pickle=False)
    assert array.dtype==np.uint8 and array.ndim==1 and array.size==part['payload_bytes'],f'Invalid upload part: {part["file"]}'
    check=hashlib.sha256()
    for offset in range(0,array.size,8*1024*1024):
     block=array[offset:offset+8*1024*1024].tobytes();output.write(block);check.update(block);overall.update(block);total+=len(block)
     if offset%(64*1024*1024)==0:print(f'Restoring part {i}/{len(doc["parts"])}: {min(offset+len(block),array.size)//1024**2}/{array.size//1024**2} MiB',flush=True)
    del array
    assert check.hexdigest()==part['payload_sha256'],f'Upload part failed integrity check: {part["file"]}'
   assert total==doc['archive_bytes'] and overall.hexdigest()==doc['archive_sha256'],'Restored ZIP checksum mismatch'
  temporary.replace(destination)
 except BaseException:
  temporary.unlink(missing_ok=True);raise
 print('Original training ZIP restored and verified.',flush=True);return destination


def safe_extract(archive,destination):
 destination=Path(destination)
 with zipfile.ZipFile(archive) as z:
  for member in z.infolist():
   assert (destination/member.filename).resolve().is_relative_to(destination.resolve())
   assert not stat.S_ISLNK(member.external_attr>>16)
  z.extractall(destination)


def setup(inputs=Path('/kaggle/input'),working=Path('/kaggle/working')):
 inputs=Path(inputs);working=Path(working);working.mkdir(parents=True,exist_ok=True)
 data=working/'pcb_six_class_recovery';run=working/'six_class_recovery_run'
 folders,archives,recoveries,uploads=discover(inputs)
 if uploads:
  assert len(uploads)==1,'Attach exactly one split-upload dataset.'
  archive=reconstruct(uploads[0],working/'pcb_six_class_recovery.zip')
 elif len(archives)==1:archive=archives[0]
 else:archive=None
 data.mkdir(exist_ok=True)
 if archive:
  # A fresh extraction also repairs a partially completed earlier setup.
  print('Extracting training data locally…',flush=True);safe_extract(archive,data)
 elif len(folders)==1:
  source=folders[0];manifest=json.loads((source/'six_class_manifest.json').read_text())
  def copy_one(name):
   target=data/name
   assert target.resolve().is_relative_to(data.resolve())
   if not target.is_file():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
  with ThreadPoolExecutor(max_workers=8) as pool:
   for i,_ in enumerate(pool.map(copy_one,[*manifest['sha256'],'six_class_manifest.json']),1):
    if i%500==0:print('Copied/checked',i,'files',flush=True)
 else:raise RuntimeError('Attach one split-upload dataset, original ZIP, or extracted dataset.')
 manifest=json.loads((data/'six_class_manifest.json').read_text());assert manifest['package_id']=='pcb_six_class_recovery_v1'
 for i,(name,digest) in enumerate(manifest['sha256'].items(),1):
  assert (data/name).resolve().is_relative_to(data.resolve())
  assert digest_file(data/name)==digest,name
  if i%500==0:print('Verified',i,'files locally',flush=True)
 print('Dataset:',manifest['splits'],flush=True)
 assert shutil.disk_usage(working).free>5*1024**3
 if not (run/'signature.json').exists() and recoveries:
  assert len(recoveries)==1,'Attach exactly one recovery checkpoint version.'
  recovery=recoveries[0];run.mkdir(exist_ok=True)
  if recovery.is_file():safe_extract(recovery,run)
  else:
   # Kaggle often expands ZIPs uploaded as datasets. Accept that form too.
   selected=['signature.json','checkpoint_manifest.json','baseline_source_validation.json','baseline_target_validation.json',
    'training/args.yaml','training/results.csv','training/trainer_state.pt','training/weights/last.pt','training/weights/best.pt']
   for name in selected:
    source=recovery/name
    if source.is_file():
     target=run/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  checkpoint_manifest=run/'checkpoint_manifest.json'
  if checkpoint_manifest.is_file():
   for name,digest in json.loads(checkpoint_manifest.read_text())['sha256'].items():
    assert (run/name).resolve().is_relative_to(run.resolve())
    assert digest_file(run/name)==digest,'Damaged recovery checkpoint: '+name
  print('Restored interrupted run:',recovery)
 sys.path.insert(0,str(data));print('Ready. No training has started yet.')
 return data,run
