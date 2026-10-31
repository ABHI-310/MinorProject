"""Bounded adapter plumbing pilot. No full experiment or accuracy claim."""
from _bootstrap import ROOT
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ap_per_class
from models.adaptation_training import AdaptationSystem, save_state, restore_state


def image_batch(package,rows,size,device):
    images=[];boxes=[];classes=[];indices=[]
    for index,row in enumerate(rows):
        path=package/row['image']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['image_sha256']
        assert hashlib.sha256((package/row['label']).read_bytes()).hexdigest()==row['label_sha256']
        image=cv2.imread(str(path));h,w=image.shape[:2]
        ratio=min(size/w,size/h);nw,nh=round(w*ratio),round(h*ratio)
        left,top=(size-nw)//2,(size-nh)//2
        canvas=np.full((size,size,3),114,dtype=np.uint8)
        canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        images.append(torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1))
        for cid,x,y,bw,bh in row['labels']:
            boxes.append([(x*nw+left)/size,(y*nh+top)/size,bw*nw/size,bh*nh/size])
            classes.append([cid]);indices.append(index)
    return dict(img=torch.stack(images).to(device).float()/255,
                bboxes=torch.tensor(boxes,device=device,dtype=torch.float32).reshape(-1,4),
                cls=torch.tensor(classes,device=device,dtype=torch.float32).reshape(-1,1),
                batch_idx=torch.tensor(indices,device=device,dtype=torch.long))


def validate(system,package,rows,size,device):
    system.eval();correct=[];scores=[];predicted=[];targets=[]
    thresholds=np.arange(.5,.951,.05)
    with torch.inference_mode():
        for row in rows:
            batch=image_batch(package,[row],size,device)
            result=non_max_suppression(system(batch['img']),conf_thres=.001,iou_thres=.7,classes=[1,2,3,4,5],nc=6,max_det=300)[0].cpu().numpy()
            b=batch['bboxes'].cpu().numpy()*size
            gt=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),axis=1)
            labels=batch['cls'].cpu().numpy().reshape(-1).astype(int)
            mask=(labels>=1)&(labels<=5);gt=gt[mask];labels=labels[mask]
            tp=np.zeros((len(result),10),dtype=bool)
            for column,threshold in enumerate(thresholds):
                used=set()
                for i in np.argsort(-result[:,4],kind='stable'):
                    options=[j for j,c in enumerate(labels) if j not in used and c==int(result[i,5])]
                    if not options:continue
                    a=result[i,:4];g=gt[options]
                    inter=np.maximum(np.minimum(a[2:],g[:,2:])-np.maximum(a[:2],g[:,:2]),0).prod(1)
                    union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(g[:,2:]-g[:,:2],0).prod(1)-inter
                    iou=inter/np.maximum(union,1e-12);j=int(np.argmax(iou))
                    if iou[j]>=threshold:tp[i,column]=True;used.add(options[j])
            correct.append(tp);scores.extend(result[:,4]);predicted.extend(result[:,5]);targets.extend(labels)
    system.train()
    ap=ap_per_class(np.concatenate(correct),np.array(scores),np.array(predicted),np.array(targets),plot=False)[5]
    return dict(images=len(rows),mAP50=float(ap[:,0].mean()) if len(ap) else 0.,mAP50_95=float(ap.mean()) if len(ap) else 0.,
                protocol='Confidence-ordered same-class greedy matching, IoU .50:.95; AP via Ultralytics; may differ from default validator matching')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=ROOT/'artifacts/kaggle/phase2_v2/package')
    p.add_argument('--selection',type=Path,default=ROOT/'artifacts/models/k500/selection.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['full','head','adapter','mmd','weighted'],default='weighted')
    p.add_argument('--steps',type=int,default=5);p.add_argument('--size',type=int,default=640)
    p.add_argument('--device',default='cpu');p.add_argument('--val-limit',type=int,default=403)
    p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args();assert args.steps>0 and args.size>=64 and args.size%32==0
    assert 1<=args.val_limit<=403
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    assert hashlib.sha256((args.package/'weights/source_baseline.pt').read_bytes()).hexdigest()=='79170e0bf457afc6a8ab40bb05e2539b5724d0e7ac6d074ad78a252a32a69c00'
    doc=json.loads((args.package/'manifest.json').read_text());selection=json.loads(args.selection.read_text())
    membership=set(selection['retained_k100_image_sha256']);assert len(membership)==100
    target=sorted([r for r in doc['views'] if r['dataset']=='target' and r['split']=='train' and r['kind']=='full' and r['image_sha256'] in membership],key=lambda r:r['image'])
    assert len(target)==100
    sources=sorted([r for r in doc['views'] if r['dataset']=='source' and r['split']=='train' and r['kind']=='full'],key=lambda r:r['image'])
    rng=np.random.default_rng(42);sources=[sources[int(i)] for i in rng.permutation(len(sources))]
    replay=[]
    for cid in range(6):
        if not any(any(int(b[0])==cid for b in r['labels']) for r in replay):
            replay.append(next(r for r in sources if any(int(b[0])==cid for b in r['labels'])))
    replay=(replay+[r for r in sources if r not in replay])[:100]
    target=[target[int(i)] for i in rng.permutation(100)]
    pairs=[]
    for step in range(args.steps):
        t=target[step%100];cls={int(b[0]) for b in t['labels']}&set(range(1,6))
        options=[r for r in replay if cls&{int(b[0]) for b in r['labels']}]
        assert options,'No source replay image sharing a defect class'
        pairs.append([options[int(rng.integers(len(options)))],t])
    counts={c:sum(int(b[0])==c for r in target for b in r['labels']) for c in range(1,6)}
    weights={c:1/math.sqrt(counts[c]) for c in counts};mean=sum(weights.values())/5
    weights={c:w/mean for c,w in weights.items()}
    experiment=dict(mode=args.mode,size=args.size,steps=args.steps,seed=42,lr=.0001,
                    code_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in {
                        'runner':Path(__file__),'integration':ROOT/'src/models/adaptation_training.py',
                        'regions':ROOT/'src/models/defect_regions.py','mmd':ROOT/'src/models/mmd_loss.py'}.items()},
                    replay=[r['image_sha256'] for r in replay],target=sorted(membership),
                    pairs=[[r['image_sha256'] for r in pair] for pair in pairs],weights=weights)
    system=AdaptationSystem(args.package/'weights/source_baseline.pt',mode=args.mode,class_weights=weights).to(args.device)
    optimizer=torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],lr=.0001)
    state=args.output/'last.pt';start=0
    if args.resume:
        assert state.is_file(),'No checkpoint to resume'
        start=restore_state(state,system,optimizer,experiment)
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'experiment.json').write_text(json.dumps(experiment,indent=2))
    for step in range(start,args.steps):
        # Position-derived schedule requires no mutable scheduler state.
        lr=.0001*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        for group in optimizer.param_groups:group['lr']=lr
        loss,log=system.training_loss(image_batch(args.package,pairs[step],args.size,args.device),1)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in system.parameters() if p.requires_grad],10,error_if_nonfinite=True)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        if not all(torch.isfinite(p).all() for p in system.parameters()):raise RuntimeError('Nonfinite weights')
        save_state(state,system,optimizer,step+1,experiment)
        print(json.dumps(dict(step=step+1,lr=lr,**log)),flush=True)
        if args.stop_after and step+1>=args.stop_after:print('PAUSED',flush=True);return
    rows=sorted([r for r in doc['views'] if r['dataset']=='target' and r['split']=='val' and r['kind']=='full'],key=lambda r:r['image'])[:args.val_limit]
    metrics=validate(system,args.package,rows,args.size,args.device)
    (args.output/'pilot_report.json').write_text(json.dumps(dict(scope='Bounded plumbing pilot, not a full scored research run',**metrics),indent=2))
    print('PILOT COMPLETE',metrics,flush=True)

if __name__=='__main__':main()
