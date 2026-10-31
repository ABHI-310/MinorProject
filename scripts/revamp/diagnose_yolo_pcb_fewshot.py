"""Read-only training-fit diagnostic for the direct-cell few-shot run."""
import argparse,importlib.util,sys,json,hashlib,os
from pathlib import Path
import torch,cv2,numpy as np

def run(run_dir,bundle,package,device='cuda'):
 run_dir=Path(run_dir);bundle=Path(bundle);package=Path(package)
 runner=run_dir.parent/'train_yolo_pcb_fewshot_fp32.py'
 # Local functional probes can provide their recorded runner path explicitly.
 if not runner.exists():raise ValueError('Expected the direct-cell runner beside the run directory')
 os.environ.setdefault('MPLCONFIGDIR',str(run_dir.parent/'matplotlib'))
 signature=json.loads((run_dir/'signature.json').read_text());recipe=signature['recipe']
 assert recipe['precision']=='fp32' and recipe['target_budget'] in [50,100]
 ids=recipe['target_image_ids'];assert len(set(ids))==len(ids)
 assert len(ids)==recipe['target_budget'] or recipe.get('probe'), 'Target count differs from recorded budget'
 assert hashlib.sha256(runner.read_bytes()).hexdigest()==recipe['runner_sha256']
 assert hashlib.sha256((bundle/'initial.pt').read_bytes()).hexdigest()==recipe['initial_sha256']
 assert hashlib.sha256((package/'manifest.json').read_bytes()).hexdigest()==recipe['manifest_sha256']
 sys.path.insert(0,str(bundle/'legacy'))
 spec=importlib.util.spec_from_file_location('verified_fewshot_runner',runner);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 from utils.datasets import letterbox
 from utils.general import non_max_suppression,scale_coords
 ck=torch.load(run_dir/'best.pt',weights_only=True,map_location='cpu');assert ck['signature']==signature['signature']
 torch.set_num_threads(4);model=module.trusted_model(bundle/'initial.pt');model.load_state_dict(ck['model']);model=model.to(device).eval()
 manifest=json.loads((package/'manifest.json').read_text());lookup={r['parent_id']:r for r in manifest['views'] if r['dataset']=='target' and r['split']=='train' and r['kind']=='full'}
 rows=[lookup[i] for i in ids];size=recipe['imgsz'];items=[];image_counts={n:0 for n in module.NAMES[1:]};box_counts={n:0 for n in module.NAMES[1:]}
 for index,r in enumerate(rows,1):
  assert hashlib.sha256((package/r['image']).read_bytes()).hexdigest()==r['image_sha256']
  assert hashlib.sha256((package/r['label']).read_bytes()).hexdigest()==r['label_sha256']
  im=cv2.imread(str(package/r['image']));scaled=letterbox(im,new_shape=size,auto=False)[0]
  x=torch.from_numpy(np.ascontiguousarray(scaled[:,:,::-1].transpose(2,0,1))).to(device).float().unsqueeze(0)/255
  with torch.inference_mode():pred=non_max_suppression(model(x)[0],.001,.7)[0]
  pred=pred.clone()
  if len(pred):pred[:,:4]=scale_coords(x.shape[2:],pred[:,:4],im.shape)
  gt=[[c,(x-w/2)*r['width'],(y-h/2)*r['height'],(x+w/2)*r['width'],(y+h/2)*r['height']] for c,x,y,w,h in r['labels']]
  items.append((gt,pred.cpu().tolist()))
  for c in range(1,6):
   count=sum(int(a[0])==c for a in r['labels']);box_counts[module.NAMES[c]]+=count;image_counts[module.NAMES[c]]+=bool(count)
  if index%10==0:print('Checked training images',index,'/',len(rows),flush=True)
 fit=module.evaluate_predictions(items,range(1,6));val=ck['metrics']['target']
 report=dict(scope='TRAINING-fit diagnostic ONLY; training AP is not held-out accuracy.',checkpoint_epoch=ck['completed_epochs'],training_images=len(rows),class_image_counts=image_counts,class_box_counts=box_counts,training_fit=fit,target_validation=val)
 (run_dir/'training_fit_diagnostic.json').write_text(json.dumps(report,indent=2))
 print('\nBest checkpoint epoch:',ck['completed_epochs'])
 print('TARGET TRAIN AP50:',round(100*fit['map50'],2),'% (seen images; not accuracy evidence)')
 print('TARGET VALIDATION AP50:',round(100*val['map50'],2),'%')
 for n in module.NAMES[1:]:print(n,': train images=',image_counts[n],', train boxes=',box_counts[n],', train AP50=',round(100*fit['per_class'][n]['ap50'],2),'%, val AP50=',round(100*val['per_class'][n]['ap50'],2),'%')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(a.run,a.bundle,a.package,a.device)
