"""Mixed-domain gradient/optimizer/resume probe. Functional evidence, not accuracy."""
import sys,json,time,hashlib,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from revamp.validate_yolo_pcb import load_legacy,BASE
import torch,numpy as np,cv2,yaml
from utils.datasets import letterbox

def main():
 torch.set_num_threads(4);torch.manual_seed(42);start=time.time()
 out=ROOT/'artifacts/verification/yolo_pcb_finetuning_probe';out.mkdir(parents=True,exist_ok=True)
 package=ROOT/'artifacts/kaggle/phase2_v2/package';manifest=json.loads((package/'manifest.json').read_text())
 selected=[]
 for domain,classes in [('source',range(6)),('target',range(1,6))]:
  for cls in classes:
   selected.append(next(r for r in manifest['views'] if r['dataset']==domain and r['split']=='train' and r['kind']=='full' and any(int(x[0])==cls for x in r['labels'])))
 model=load_legacy(BASE/'JiaLim98/weights/baseline_fpn_loss.pt');model.requires_grad_(True);model.train()
 model.hyp=yaml.safe_load((BASE/'JiaLim98/data/hyp.scratch.yaml').read_text());model.gr=1.;model.nc=6
 # Original code uses float clamp bounds for integer tensor indices. Preserve
 # matching/loss semantics, cast only the bounds for current PyTorch compatibility.
 source=(BASE/'JiaLim98/utils/loss.py').read_text()
 patched=source.replace('gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)','gj.clamp_(0, int(gain[3]) - 1), gi.clamp_(0, int(gain[2]) - 1)')
 assert patched!=source
 loss_module=types.ModuleType('loss_compat_probe');exec(compile(patched,'YOLO-PCB-loss-compat','exec'),loss_module.__dict__)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=5e-4)
 original=model.model[0].conv.conv.weight.detach().clone();records=[]
 # Every source and supported target class appears. Source and target are paired
 # in updates; these tiny repeated samples are never used to claim model quality.
 for step in range(6):
  batch=[selected[step],selected[6+(step%5)]];inputs=[];labels=[]
  for i,r in enumerate(batch):
   assert r['split']=='train'
   path=package/r['image'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['image_sha256']
   assert hashlib.sha256((package/r['label']).read_bytes()).hexdigest()==r['label_sha256']
   im=cv2.imread(str(path));img,ratio,pad=letterbox(im,new_shape=608,auto=False)
   inputs.append(np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1)))
   for cls,x,y,w,h in r['labels']:
    labels.append([i,cls,(x*im.shape[1]*ratio[0]+pad[0])/608,(y*im.shape[0]*ratio[1]+pad[1])/608,w*im.shape[1]*ratio[0]/608,h*im.shape[0]*ratio[1]/608])
  x=torch.from_numpy(np.stack(inputs)).float()/255;t=torch.tensor(labels,dtype=torch.float32)
  optimizer.zero_grad();pred=model(x);loss,items=loss_module.compute_loss(pred,t,model)
  assert torch.isfinite(loss).all();loss.backward()
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  optimizer.step();records.append(dict(step=step+1,loss=float(loss.detach()),loss_items=items.tolist(),parents=[r['parent_id'] for r in batch]))
  print(records[-1],flush=True)
 changed=not torch.equal(original,model.model[0].conv.conv.weight.detach());assert changed
 ck={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'completed_steps':len(records)}
 torch.save(ck,out/'training_probe_checkpoint.pt');restored=torch.load(out/'training_probe_checkpoint.pt',weights_only=True,map_location='cpu')
 optimizer2=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer2.load_state_dict(restored['optimizer_state_dict']);assert len(optimizer2.state)==len(optimizer.state)
 assert restored['completed_steps']==6
 report=dict(status='passed',scope='Only six mixed-domain optimizer updates on training images. Not a trained replacement, accuracy result, or learning/convergence guarantee.',seconds=time.time()-start,checkpoint_source_sha256=hashlib.sha256((BASE/'JiaLim98/weights/baseline_fpn_loss.pt').read_bytes()).hexdigest(),compatibility_patch='Integer cast for grid-index clamp bounds; explicitly re-enable gradients on exported checkpoint parameters',original_loss_sha256=hashlib.sha256(source.encode()).hexdigest(),patched_loss_sha256=hashlib.sha256(patched.encode()).hexdigest(),all_losses_and_gradients_finite=True,backbone_weights_changed=changed,optimizer_state_restored=True,records=records)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(report['status'],flush=True)
if __name__=='__main__':main()
