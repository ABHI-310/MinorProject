"""Validation-only error counts at a fixed threshold; no training or test access."""
import argparse
import json
import os
from pathlib import Path
import zipfile
from revamp.runtime import digest, verify_package, NAMES


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def match(gt,pred,threshold=.5):
    candidates=sorted([(iou(g['box'],p['box']),i,j) for i,g in enumerate(gt)
                       for j,p in enumerate(pred) if g['class']==p['class']],reverse=True)
    used_g,used_p=set(),set()
    for overlap,i,j in candidates:
        if overlap>=threshold and i not in used_g and j not in used_p:
            used_g.add(i);used_p.add(j)
    return used_g,used_p


def analyze(row,pred):
    gt=[]
    for cls,x,y,w,h in row['labels']:
        gt.append(dict(class_id=int(cls),box=[(x-w/2)*row['width'],(y-h/2)*row['height'],
                    (x+w/2)*row['width'],(y+h/2)*row['height']]))
    gt=[{'class':g['class_id'],'box':g['box']} for g in gt]
    used_g,used_p=match(gt,pred)
    counts={n:dict(tp=0,fp=0,fn=0) for n in NAMES}
    misses=[]
    for i,g in enumerate(gt):
        counts[NAMES[g['class']]]['tp' if i in used_g else 'fn']+=1
        if i not in used_g:
            same=max([iou(g['box'],p['box']) for p in pred if p['class']==g['class']],default=0)
            other=max([iou(g['box'],p['box']) for p in pred if p['class']!=g['class']],default=0)
            misses.append(dict(class_name=NAMES[g['class']],box=g['box'],best_same_class_iou=same,
                               best_other_class_iou=other,short_edge_at_960=min(g['box'][2]-g['box'][0],g['box'][3]-g['box'][1])*960/max(row['width'],row['height'])))
    for i,p in enumerate(pred):
        if i not in used_p:counts[NAMES[p['class']]]['fp']+=1
    return dict(parent_id=row['parent_id'],image=row['image'],counts=counts,
                misses=misses,unmatched_predictions=[p for i,p in enumerate(pred) if i not in used_p])


def checkpoint_weights(archive,output):
    with zipfile.ZipFile(archive) as z:
        record=json.loads(z.read('checkpoint_manifest.json'))
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(record['sha256'])|{'checkpoint_manifest.json'}:
            raise ValueError('Invalid checkpoint member list')
        for name,expected in record['sha256'].items():
            if __import__('hashlib').sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Checkpoint checksum mismatch')
        payload=z.read('training/weights/best.pt')
        path=output/'candidate.pt';path.write_bytes(payload)
        return path,json.loads(z.read('signature.json'))


def run(package,checkpoint,output,device='cpu'):
    import torch
    from ultralytics import YOLO
    package,output=Path(package).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Use a new diagnostics output folder; existing results preserved.')
    verify_package(package);output.mkdir(parents=True)
    candidate,signature=checkpoint_weights(checkpoint,output)
    if signature.get('identity',{}).get('package_sha256')!=digest(package/'package_hashes.json'):
        raise ValueError('Checkpoint/package mismatch')
    torch.set_num_threads(4)
    manifest=json.loads((package/'manifest.json').read_text())
    rows=[r for r in manifest['views'] if r['dataset']=='source' and r['split']=='val' and r['kind']=='full']
    summary={'scope':'Source validation only. Fixed conf=0.25, IoU=0.5, greedy one-to-one same-class matching. These counts are not mAP or reported optimal-threshold precision/recall.',
             'images':len(rows),'checkpoint_sha256':digest(checkpoint),'models':{}}
    for name,weights in [('baseline',package/'weights/source_baseline.pt'),('candidate',candidate)]:
        model=YOLO(str(weights))
        if list(model.names.values())!=NAMES:raise ValueError('Class order differs')
        details=[];counts={n:dict(tp=0,fp=0,fn=0) for n in NAMES}
        for index,row in enumerate(rows,1):
            result=model.predict(str(package/row['image']),imgsz=960,conf=.25,iou=.7,max_det=300,
                                 device=device,half=False,rect=False,verbose=False)[0]
            pred=[{'class':int(c),'box':box,'confidence':float(conf)} for c,box,conf in
                  zip(result.boxes.cls.cpu().tolist(),result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
            detail=analyze(row,pred);details.append(detail)
            for n in NAMES:
                for k in ['tp','fp','fn']:counts[n][k]+=detail['counts'][n][k]
            if index%20==0:print(name,index,'/',len(rows),flush=True)
        (output/f'{name}_errors.json').write_text(json.dumps(details,indent=2))
        summary['models'][name]={'weights_sha256':digest(weights),'counts':counts}
        del model
    (output/'diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive=output/'diagnosis.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['diagnosis.json','baseline_errors.json','candidate_errors.json']:z.write(output/name,name)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(args.output.parent/'diagnosis_yolo'))
    run(args.package,args.checkpoint,args.output,args.device)
