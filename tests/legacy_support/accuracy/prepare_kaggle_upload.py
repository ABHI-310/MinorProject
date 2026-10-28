"""Split a verified training ZIP into opaque NumPy upload parts without changing it."""
from pathlib import Path
import hashlib,json,os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CHUNK_BYTES=256*1024*1024

def prepare(archive,output,chunk_bytes=CHUNK_BYTES):
 archive=Path(archive);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 total=archive.stat().st_size;parts=[];overall=hashlib.sha256()
 with archive.open('rb') as source:
  number=0
  while source.tell()<total:
   number+=1;size=min(chunk_bytes,total-source.tell());name=f'pcb_six_class_part_{number:02d}.npy'
   path=output/name;temporary=output/(name+'.tmp')
   array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(size,));digest=hashlib.sha256();offset=0
   while offset<size:
    block=source.read(min(8*1024*1024,size-offset))
    if not block:raise RuntimeError('Training ZIP changed during preparation.')
    array[offset:offset+len(block)]=np.frombuffer(block,dtype=np.uint8)
    digest.update(block);overall.update(block);offset+=len(block)
   array.flush();del array;os.replace(temporary,path)
   parts.append({'file':name,'payload_bytes':size,'payload_sha256':digest.hexdigest()})
   print(name,round(path.stat().st_size/1024**2,1),'MiB',flush=True)
 manifest={'format':'pcb_numpy_upload_v1','archive_name':'pcb_six_class_recovery.zip','archive_bytes':total,
   'archive_sha256':overall.hexdigest(),'parts':parts,
   'note':'Upload all four .npy parts and this JSON together. The split-upload notebook restores the original ZIP locally.'}
 (output/'pcb_upload_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return manifest

if __name__=='__main__':
 prepare(ROOT/'accuracy/pcb_six_class_recovery.zip',ROOT/'accuracy/kaggle_upload')
