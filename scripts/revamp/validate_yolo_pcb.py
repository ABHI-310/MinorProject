"""Development comparison; shared dataset ancestry means independence is unverified."""
import os,sys,json,hashlib,importlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/external_models/yolo_pcb_review'
os.environ.setdefault('MPLCONFIGDIR','/tmp/pcb-mpl')
os.environ.setdefault('YOLO_CONFIG_DIR',str(BASE/'ultralytics_config'))
sys.path[:0]=[str(BASE/'deps'),str(BASE/'JiaLim98'),str(ROOT/'scripts')]
import numpy as np
import torch
import cv2
from revamp.diagnose import analyze
from revamp.runtime import NAMES

def load_legacy(path):
    allowed=[]
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        if not name.startswith(('models.common.','models.yolo.','torch.nn.modules.','numpy.')):
            raise ValueError(f'Unreviewed global: {name}')
        mod,attr=name.rsplit('.',1)
        allowed.append((getattr(importlib.import_module(mod),attr),name))
    allowed.extend([type(np.dtype('float64')),type(np.dtype('float32'))])
    with torch.serialization.safe_globals(allowed):
        ck=torch.load(path,map_location='cpu',weights_only=True)
    model=ck['model'].float().eval()
    assert list(model.names)==NAMES,model.names
    return model

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["source","target"],default="source");args=parser.parse_args()
    torch.set_num_threads(4)
    package=ROOT/'artifacts/kaggle/phase2_v2/package'
    out=ROOT/('artifacts/verification/yolo_pcb_comparison' if args.dataset=='source' else 'artifacts/verification/yolo_pcb_target');out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((package/'manifest.json').read_text())
    rows=[r for r in manifest['views'] if r['dataset']==args.dataset and r['split']=='val' and r['kind']=='full']
    summary={'dataset':args.dataset,'missing_hole_supported':args.dataset=='source','scope':'Development comparison only. Shared PKU/TDD-Net dataset ancestry; external training overlap unverified. No independent accuracy claim. No clean full-board benchmark available.', 'images':len(rows),'confidence':.25,'nms_iou':.7,'matching_iou':.5,'metric':'Greedy one-to-one same-class box matching; not mAP','models':{}}
    from utils.datasets import letterbox
    from utils.general import non_max_suppression,scale_coords
    from ultralytics import YOLO
    configs=[('yolo_pcb_608',BASE/'JiaLim98/weights/baseline_fpn_loss.pt',608),('yolo_pcb_960',BASE/'JiaLim98/weights/baseline_fpn_loss.pt',960),('app_baseline_960',ROOT/'artifacts/models/source/model.pt',960)]
    if args.dataset=='target':
        configs=[c for c in configs if c[0]!='yolo_pcb_608']
    for name,weights,size in configs:
        cached=out/(name+'.json')
        if cached.exists():
            summary['models'][name]=json.loads(cached.read_text())['summary'];continue
        legacy=name.startswith('yolo_pcb');model=load_legacy(weights) if legacy else YOLO(str(weights))
        details=[];counts={n:dict(tp=0,fp=0,fn=0) for n in NAMES};start=time.time()
        for i,row in enumerate(rows):
            path=package/row['image']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['image_sha256']
            assert hashlib.sha256((package/row['label']).read_bytes()).hexdigest()==row['label_sha256']
            if legacy:
                im=cv2.imread(str(path));im1=letterbox(im,new_shape=size,auto=False)[0]
                x=torch.from_numpy(np.ascontiguousarray(im1[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
                with torch.inference_mode():pred=non_max_suppression(model(x)[0],.25,.7)[0]
                pred=pred.clone()
                if len(pred):pred[:,:4]=scale_coords(x.shape[2:],pred[:,:4],im.shape)
                predictions=[{'class':int(c),'box':[float(v) for v in box],'confidence':float(conf)} for *box,conf,c in pred.tolist()]
            else:
                r=model.predict(str(path),imgsz=size,conf=.25,iou=.7,max_det=300,device='cpu',half=False,rect=False,verbose=False)[0]
                predictions=[{'class':int(c),'box':box,'confidence':float(conf)} for box,conf,c in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist())]
            d=analyze(row,predictions);d['predictions']=predictions;details.append(d)
            for cls in NAMES:
                for k in ['tp','fp','fn']:counts[cls][k]+=d['counts'][cls][k]
            if (i+1)%20==0:print(name,i+1,len(rows),flush=True)
        totals={k:sum(c[k] for c in counts.values()) for k in ['tp','fp','fn']}
        result={'sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'imgsz':size,'counts':counts,'totals':totals,'precision':totals['tp']/max(1,totals['tp']+totals['fp']),'recall':totals['tp']/max(1,totals['tp']+totals['fn']),'seconds':time.time()-start}
        cached.write_text(json.dumps({'summary':result,'details':details},indent=2))
        summary['models'][name]=result
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        del model
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
