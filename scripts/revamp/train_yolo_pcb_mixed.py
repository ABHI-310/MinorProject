"""Bounded YOLO-PCB mixed-domain development training with epoch-level recovery.
Not an independent final evaluation; PKU ancestry and target grouping are limitations.
"""
import argparse,hashlib,importlib,json,math,os,random,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
import cv2
import yaml
NAMES=['missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,d):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
def atomic_torch(p,d):
 tmp=Path(str(p)+'.tmp');torch.save(d,tmp);os.replace(tmp,p)
def box_iou(a,b):
 inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
 return inter/max(1e-12,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter)
def ap101(tp,confidence,total):
 if not total or not len(tp):return 0.
 order=np.argsort(-np.asarray(confidence),kind='stable');tp=np.asarray(tp,dtype=float)[order]
 recall=np.cumsum(tp)/total;precision=np.cumsum(tp)/np.arange(1,len(tp)+1)
 precision=np.maximum.accumulate(precision[::-1])[::-1]
 return float(np.mean([precision[recall>=r].max(initial=0.) for r in np.linspace(0,1,101)]))
def evaluate_predictions(items,classes):
 per_class={};levels=np.linspace(.5,.95,10)
 for c in classes:
  total=sum(sum(int(g[0])==c for g in gt) for gt,pred in items);aps=[];fixed={str(t):dict(tp=0,fp=0,fn=0) for t in [.25,.5,.75,.9]}
  for threshold in levels:
   flags=[];scores=[]
   for gt,pred in items:
    gt=[g[1:] for g in gt if int(g[0])==c];used=set()
    for p in sorted([p for p in pred if int(p[5])==c],key=lambda x:-x[4]):
     candidates=[(box_iou(np.asarray(p[:4]),np.asarray(g)),i) for i,g in enumerate(gt) if i not in used]
     overlap,i=max(candidates,default=(0.,-1));hit=overlap>=threshold
     if hit:used.add(i)
     flags.append(hit);scores.append(p[4])
   aps.append(ap101(flags,scores,total))
   if threshold==levels[0]:
    for t in fixed:
     mask=np.asarray(scores)>=float(t);hits=int(np.asarray(flags,dtype=bool)[mask].sum());fixed[t]=dict(tp=hits,fp=int(mask.sum())-hits,fn=total-hits)
  per_class[NAMES[c]]=dict(labels=total,ap50=aps[0],ap50_95=float(np.mean(aps)),operating_points=fixed)
 return dict(map50=float(np.mean([v['ap50'] for v in per_class.values()])),map50_95=float(np.mean([v['ap50_95'] for v in per_class.values()])),per_class=per_class,metric='101-point precision-envelope AP; confidence floor 0.001, class-wise IoU matching; separate domains, development only')

def select_target_budget(rows,budget,seed=42):
 # Image-level budget, not per-board K. Build nested 50/100 subsets from TRAIN only.
 unique={}
 for r in rows:
  if r['split']!='train' or r['dataset']!='target':raise ValueError('Target selection must use training only')
  unique.setdefault(r['image_sha256'],r)
 pool=list(unique.values());rng=np.random.default_rng(seed);pool=[pool[i] for i in rng.permutation(len(pool))]
 if not budget:return pool
 chosen=[];covered=set()
 for c in range(1,6):
  if c in covered:continue
  r=next((r for r in pool if any(int(a[0])==c for a in r['labels']) and r not in chosen),None)
  if r is None:raise ValueError('Target class absent: '+str(c))
  chosen.append(r);covered.update(int(a[0]) for a in r['labels'])
 chosen += [r for r in pool if r not in chosen]
 if len(chosen)<budget:raise ValueError('Insufficient unique target images')
 return chosen[:budget]

def checked_step(model,optimizer,loss,context):
 if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite FP32 loss; '+context)
 loss.backward()
 bad=[n for n,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
 if bad:
  optimizer.zero_grad(set_to_none=True)
  raise RuntimeError('Nonfinite FP32 gradients before optimizer step: '+str(bad[:8])+'; '+context)
 torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
 optimizer.step()

def trusted_model(path):
 allowed=[]
 for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
  if not name.startswith(('models.common.','models.yolo.','torch.nn.modules.','numpy.')):raise ValueError('Unreviewed checkpoint global: '+name)
  mod,attr=name.rsplit('.',1);allowed.append((getattr(importlib.import_module(mod),attr),name))
 allowed.extend([type(np.dtype('float32')),type(np.dtype('float64'))])
 with torch.serialization.safe_globals(allowed):model=torch.load(path,map_location='cpu',weights_only=True)['model'].float()
 assert list(model.names)==NAMES
 return model

def publish(run):
 members=['last.pt','best.pt','history.json','signature.json','environment.json','target_selection.json']
 hashes={n:digest(run/n) for n in members};tmp=run/'resume.zip.tmp'
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n in members:z.write(run/n,n)
  z.writestr('checksums.json',json.dumps(hashes))
 os.replace(tmp,run/'resume.zip')

def restore(archive,run):
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();expected={'last.pt','best.pt','history.json','signature.json','environment.json','target_selection.json','checksums.json'}
  if len(names)!=len(set(names)) or set(names)!=expected:raise ValueError('Unexpected backup members')
  hashes=json.loads(z.read('checksums.json'))
  if set(hashes)!=expected-{'checksums.json'}:raise ValueError('Incomplete hashes')
  for n,h in hashes.items():
   if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Corrupt '+n)
  # Refuse to overwrite an active local run. Fresh directories only.
  if (run/'last.pt').exists():raise ValueError('Restore requires a fresh run directory')
  for n in hashes:
   tmp=run/(n+'.tmp');tmp.write_bytes(z.read(n));os.replace(tmp,run/n)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--package',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
 p.add_argument('--stop-after-epoch',type=int,default=5);p.add_argument('--resume-archive',type=Path)
 p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=4);p.add_argument('--imgsz',type=int,default=608)
 p.add_argument('--target-budget',type=int,choices=[0,50,100],default=50,help='Unique labeled target TRAIN images total; 0 is fully supervised comparison only')
 p.add_argument('--patience',type=int,default=8);p.add_argument('--device',default='cuda');p.add_argument('--probe',action='store_true')
 a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
 os.environ.setdefault('MPLCONFIGDIR',str(a.run/'matplotlib'));torch.set_num_threads(4)
 if a.batch<2 or a.batch%2:raise ValueError('Batch must be positive and even; equal domain contribution')
 if a.imgsz%32:raise ValueError('Input size must be divisible by 32')
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Enable a Kaggle GPU accelerator first')
 sys.path.insert(0,str(a.bundle/'legacy'))
 from utils.datasets import letterbox
 from utils.general import non_max_suppression,scale_coords
 import types
 source=(a.bundle/'legacy/utils/loss.py').read_text()
 patched=source.replace('gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)','gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)')
 if source==patched:raise ValueError('Expected reviewed legacy loss version')
 lossmod=types.ModuleType('reviewed_loss_compat');exec(compile(patched,'reviewed_loss_compat','exec'),lossmod.__dict__)
 manifest=json.loads((a.package/'manifest.json').read_text())
 rows={d:{s:[r for r in manifest['views'] if r['dataset']==d and r['split']==s and r['kind']=='full'] for s in ['train','val']} for d in ['source','target']}
 if any(not rows[d][s] for d in rows for s in rows[d]):raise ValueError('Missing dataset partition')
 rows['target']['train']=select_target_budget(rows['target']['train'],a.target_budget)
 if a.probe:
  for d in rows:
   for s in rows[d]:rows[d][s]=rows[d][s][:2]
 for d in rows:
  for s in rows[d]:
   for r in rows[d][s]:
    if digest(a.package/r['image'])!=r['image_sha256'] or digest(a.package/r['label'])!=r['label_sha256']:raise ValueError('Package hash mismatch')
 torch.manual_seed(42);np.random.seed(42);random.seed(42)
 device=torch.device(a.device);model=trusted_model(a.bundle/'initial.pt').requires_grad_(True).to(device)
 model.nc=6;model.gr=1.;model.hyp=yaml.safe_load((a.bundle/'legacy/data/hyp.scratch.yaml').read_text())
 # Match legacy training gain scaling to current number of classes/layers/size.
 nl=model.model[-1].nl;model.hyp['box']*=3/nl;model.hyp['cls']*=6/80*3/nl;model.hyp['obj']*=(a.imgsz/640)**2*3/nl
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5)
 # FP32 is deliberate: the legacy loss has not passed a GPU AMP stability check.
 scaler=torch.amp.GradScaler('cuda',enabled=False)
 env={'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__,'python':sys.version,'device':str(device)}
 selection=[dict(parent_id=r['parent_id'],image_sha256=r['image_sha256']) for r in rows['target']['train']]
 recipe=dict(precision='fp32',target_budget=a.target_budget,target_selection_sha256=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest(),epochs=a.epochs,batch=a.batch,imgsz=a.imgsz,patience=a.patience,probe=a.probe,manifest_sha256=digest(a.package/'manifest.json'),initial_sha256=digest(a.bundle/'initial.pt'),runner_sha256=digest(__file__),loss_sha256=hashlib.sha256(patched.encode()).hexdigest(),torch=str(torch.__version__),numpy=np.__version__,optimizer='AdamW',lr=1e-4,seed=42,augmentation='none in first controlled pilot',domain_sampling='equal source and target per batch')
 signature=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
 if a.resume_archive:restore(a.resume_archive,a.run)
 if (a.run/'signature.json').exists() and json.loads((a.run/'signature.json').read_text())['signature']!=signature:raise ValueError('Data, code, environment, or recipe differs from checkpoint')
 atomic_json(a.run/'target_selection.json',dict(scope='Unique target training images total, not per-board K; target board independence unknown',budget=a.target_budget,selected=selection))
 atomic_json(a.run/'signature.json',dict(signature=signature,recipe=recipe));atomic_json(a.run/'environment.json',env)
 start_epoch=0;best=-1.;bad=0;history=[]
 if (a.run/'last.pt').exists():
  ck=torch.load(a.run/'last.pt',map_location='cpu',weights_only=True)
  if ck['signature']!=signature:raise ValueError('State signature differs')
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
  start_epoch=ck['completed_epochs'];best=ck['best'];bad=ck['bad'];history=ck['history'];torch.set_rng_state(ck['torch_rng'])
  if device.type=='cuda' and ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
  print('Resumed completed epoch',start_epoch,flush=True)
  if ck.get('finished'):print('Run already finished. Reusing checkpoint.',flush=True);return
 def batch_input(batch):
  images=[];targets=[]
  for index,r in enumerate(batch):
   im=cv2.imread(str(a.package/r['image']));im1,ratio,pad=letterbox(im,new_shape=a.imgsz,auto=False)
   images.append(np.ascontiguousarray(im1[:,:,::-1].transpose(2,0,1)))
   for c,x,y,w,h in r['labels']:
    targets.append([index,c,(x*im.shape[1]*ratio[0]+pad[0])/a.imgsz,(y*im.shape[0]*ratio[1]+pad[1])/a.imgsz,w*im.shape[1]*ratio[0]/a.imgsz,h*im.shape[0]*ratio[1]/a.imgsz])
  return torch.from_numpy(np.stack(images)).to(device).float()/255,torch.tensor(targets,dtype=torch.float32,device=device).reshape(-1,6)
 def validation(domain):
  model.eval();items=[]
  for r in rows[domain]['val']:
   x,_=batch_input([r])
   with torch.inference_mode():pred=non_max_suppression(model(x)[0],.001,.7)[0]
   pred=pred.clone()
   if len(pred):pred[:,:4]=scale_coords(x.shape[2:],pred[:,:4],(r['height'],r['width']))
   gt=[[c,(x-w/2)*r['width'],(y-h/2)*r['height'],(x+w/2)*r['width'],(y+h/2)*r['height']] for c,x,y,w,h in r['labels']]
   items.append((gt,pred.detach().cpu().tolist()))
  return evaluate_predictions(items,range(6) if domain=='source' else range(1,6))
 if not start_epoch:
  atomic_json(a.run/'initial_validation.json',{d:validation(d) for d in rows})
 for epoch in range(start_epoch,min(a.epochs,a.stop_after_epoch)):
  start=time.time();model.train();rng=np.random.default_rng(42+epoch);half=a.batch//2
  steps=math.ceil(max(len(rows[d]['train']) for d in rows)/half)
  orders={d:np.concatenate([rng.permutation(len(rows[d]['train'])) for _ in range(math.ceil(steps*half/len(rows[d]['train'])))])[:steps*half] for d in rows}
  losses=[]
  for step in range(steps):
   batch=[rows[d]['train'][i] for d in rows for i in orders[d][step*half:(step+1)*half]]
   x,t=batch_input(batch);opt.zero_grad(set_to_none=True)
   # Disable autocast for both model and legacy geometric loss.
   with torch.autocast(device_type=device.type,enabled=False):loss,items=lossmod.compute_loss(model(x.float()),t.float(),model)
   context=json.dumps(dict(epoch=epoch+1,step=step+1,parents=[r['parent_id'] for r in batch],loss_items=items.tolist()))
   try:checked_step(model,opt,loss,context)
   except RuntimeError as error:
    atomic_json(a.run/'failure.json',dict(error=str(error),context=context,environment=env));raise
   losses.append(float(loss.detach()))
   if (step+1)%100==0:print(f'epoch {epoch+1} step {step+1}/{steps} loss {np.mean(losses):.5f}',flush=True)
  scheduler.step();metrics={d:validation(d) for d in rows}
  # Harmonic mean stops one high-scoring domain from hiding total failure in the other.
  x,y=metrics['source']['map50'],metrics['target']['map50'];score=2*x*y/(x+y+1e-12)
  improved=score>best
  if improved:best=score;bad=0
  else:bad+=1
  history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.time()-start,score=score,metrics=metrics))
  state=dict(signature=signature,completed_epochs=epoch+1,best=best,bad=bad,finished=epoch+1==a.epochs or bad>=a.patience,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
  atomic_torch(a.run/'last.pt',state)
  if improved:atomic_torch(a.run/'best.pt',dict(model=model.state_dict(),completed_epochs=epoch+1,signature=signature,metrics=metrics))
  atomic_json(a.run/'history.json',history);publish(a.run)
  print(json.dumps(history[-1]),flush=True);print('DOWNLOAD BACKUP:',a.run/'resume.zip',flush=True)
  if state['finished']:break
 print('Stopped at an epoch boundary. Rerun with a higher --stop-after-epoch to continue the same schedule.',flush=True)
if __name__=='__main__':main()
