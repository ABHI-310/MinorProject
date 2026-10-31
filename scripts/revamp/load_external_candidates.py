import importlib,torch,numpy as np
class TrainingMetadata:
 def __init__(self,*a,**kw): pass
 def __setstate__(self,state): pass
 def __getattr__(self,name): return TrainingMetadata

def load(path):
 allowed=[]
 for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
  if name.startswith(('albumentations.','numpy.random.')) or name=='random.Random':
   allowed.append((TrainingMetadata,name))
  elif name.startswith(('torch.nn.modules.','ultralytics.nn.')) or name in ['builtins.getattr','numpy._core.multiarray._reconstruct','numpy.ndarray','numpy.dtype']:
   module,attr=name.rsplit('.',1); allowed.append((getattr(importlib.import_module(module),attr),name))
  else: raise RuntimeError(f'Unreviewed global: {name}')
 allowed.extend([type(np.dtype('uint32')),type(np.dtype('float32')),type(np.dtype('float64')),type(np.dtype('int64'))])
 with torch.serialization.safe_globals(allowed):
  return torch.load(path,map_location='cpu',weights_only=True)
if __name__=='__main__':
 for name in ['damage','bare']:
  ck=load(f'artifacts/external_models/pcb_candidates/{name}.pt')
  ck['model'].args = {'task':'detect', 'imgsz':ck.get('train_args',{}).get('imgsz')}
  print(name,ck['model'].names,ck.get('train_args',{}).get('imgsz'))
  torch.save({'model':ck['model'],'train_args':{'imgsz':ck.get('train_args',{}).get('imgsz'),'task':'detect'},'date':ck.get('date'),'version':ck.get('version')},f'artifacts/external_models/pcb_candidates/{name}_inference.pt')
