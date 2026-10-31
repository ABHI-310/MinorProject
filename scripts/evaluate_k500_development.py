"""Evaluate fixed K500 settings on the previously used 992-image development set."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'data/runtime/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'data/runtime/matplotlib'))
import numpy as np
import torch
import yaml
from ultralytics import YOLO


def match(predictions, ground_truth):
    counts = {c: dict(tp=0, fp=0, fn=0) for c in range(1, 6)}
    used = set()
    for p in sorted(predictions, key=lambda p: -p['confidence']):
        candidates = [i for i,g in enumerate(ground_truth) if i not in used and g['class']==p['class']]
        best, score = None, -1
        a = np.array(p['box'])
        for i in candidates:
            b = np.array(ground_truth[i]['box'])
            inter = np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
            union = np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
            iou = inter/max(union,1e-12)
            if iou > score: best,score=i,iou
        if score >= .5:
            counts[p['class']]['tp']+=1;used.add(best)
        else: counts[p['class']]['fp']+=1
    for i,g in enumerate(ground_truth):
        if i not in used: counts[g['class']]['fn']+=1
    return counts


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT/'artifacts/models/k500/model.pt')
    parser.add_argument('--sha256', default='c05a9c35608076e0b0af158ac4ffcabd954f3adfa6928552387e35e875615f6d')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/revamp/k500_development_evaluation')
    parser.add_argument('--split', choices=['val','test'], default='test')
    args=parser.parse_args()
    split=args.split
    expected_images,expected_boxes=(992,1849) if split=='test' else (403,746)
    torch.set_num_threads(4)
    data=ROOT/'data/prepared/target'
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    weights=args.weights
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest==args.sha256
    manifest=json.loads((data/'manifest.json').read_text())['sha256']
    selection=json.loads((ROOT/'artifacts/models/k500/selection.json').read_text())
    prohibited=set(selection['training_image_sha256'])
    if split=='test':prohibited |= set(selection['validation_image_sha256'])
    images=sorted((data/'images'/split).glob('*.jpg'));assert len(images)==expected_images
    for p in images:
        for f in [p,data/'labels'/split/p.with_suffix('.txt').name]:
            h=hashlib.sha256(f.read_bytes()).hexdigest()
            assert h==manifest[f.relative_to(data).as_posix()]
            if f==p: assert h not in prohibited
    model=YOLO(str(weights));names=model.names
    config=out/'evaluation_data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(data),train='images/train',val='images/val',test='images/test',names=names)))
    start=time.time();totals={c:dict(tp=0,fp=0,fn=0) for c in range(1,6)};details=[]
    # A directory source respects batch size; a Python list would be one large batch.
    for r in model.predict(source=str(data/'images'/split),stream=True,batch=1,imgsz=640,
                           conf=.25,iou=.4,rect=False,max_det=300,classes=list(range(1,6)),
                           agnostic_nms=False,augment=False,device='cpu',verbose=False,save=False):
        p=Path(r.path);height,width=r.orig_shape
        gt=[]
        for line in (data/'labels'/split/p.with_suffix('.txt').name).read_text().splitlines():
            if not line.strip():continue
            c,x,y,w,h=map(float,line.split())
            gt.append({'class':int(c),'box':[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]})
        predictions=[{'class':int(c),'confidence':float(s),'box':box} for box,s,c in zip(r.boxes.xyxy.tolist(),r.boxes.conf.tolist(),r.boxes.cls.tolist())]
        counts=match(predictions,gt)
        for c in totals:
            for k in totals[c]:totals[c][k]+=counts[c][k]
        details.append(dict(image=p.name,ground_truth=gt,predictions=predictions,counts=counts))
        if len(details)%100==0:print('Fixed-threshold evaluation:',len(details),'/',expected_images,flush=True)
    assert len(details)==expected_images
    (out/'predictions.json').write_text(json.dumps(details))
    overall={k:sum(v[k] for v in totals.values()) for k in ['tp','fp','fn']}
    assert overall['tp']+overall['fn']==expected_boxes
    for v in [overall,*totals.values()]:
        v['precision']=v['tp']/max(v['tp']+v['fp'],1)
        v['recall']=v['tp']/max(v['tp']+v['fn'],1)
        v['f1']=2*v['tp']/max(2*v['tp']+v['fp']+v['fn'],1)
    report=dict(role=('Previously used development test; not an untouched final test' if split=='test' else 'Development validation'),images=expected_images,
                labelled_boxes=expected_boxes,checkpoint_sha256=digest,imgsz=640,confidence=.25,nms_iou=.4,
                matching_iou=.5,matching='Confidence-ordered class-aware one-to-one',overall=overall,
                per_class={names[c]:v for c,v in totals.items()},
                limitations=['Board independence UNKNOWN','No defect-free images','Settings fixed from earlier validation'],
                ultralytics=__import__('ultralytics').__version__,torch=str(torch.__version__))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Fixed-threshold results:',json.dumps(overall),flush=True)
    print('Computing AP separately at conf=0.001; no threshold tuning.',flush=True)
    metrics=YOLO(str(weights)).val(data=str(config),split=split,imgsz=640,batch=1,device='cpu',workers=0,
        rect=False,conf=.001,iou=.4,max_det=300,classes=list(range(1,6)),agnostic_nms=False,
        augment=False,plots=False,save_json=False,project=str(out),name='ap',exist_ok=True,verbose=False)
    report['mAP50']=float(metrics.box.map50);report['mAP50_95']=float(metrics.box.map)
    for c,ap in zip(metrics.box.ap_class_index,metrics.box.ap50):report['per_class'][names[int(c)]]['AP50']=float(ap)
    report['seconds']=time.time()-start
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
