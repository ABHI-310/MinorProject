"""Reproduce inference on six user photos; this is not a labelled accuracy benchmark.

Run from repository root with its .venv after downloading the pinned files in manifest.json.
Original external checkpoints are restricted-loaded by load_external_candidates.py;
training-only augmentation/RNG objects are discarded without running their code.
"""
import json, time, hashlib, platform
from pathlib import Path
import torch, ultralytics
from ultralytics import YOLO
from PIL import Image, ImageOps
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'artifacts/external_models/pcb_candidates'
OUT = ROOT / 'artifacts/verification/external_candidates'
INPUT = ROOT / 'artifacts/verification/ui_failure_cases'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((WEIGHTS/'thresholds.json').read_text())['per_class']
    report = {'scope':'Six unlabelled user photos; qualitative inspection only, not accuracy. Training overlap unknown.',
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__},
              'manifest':json.loads((WEIGHTS/'manifest.json').read_text()), 'models':{}}
    for name, size in [('damage',800),('bare',640)]:
        model = YOLO(str(WEIGHTS/f'{name}_inference.pt'))
        record = {'imgsz':size, 'device':'cpu','iou':0.5,'rect':True, 'names':model.names,'images':[]}
        for i in range(1,7):
            path=INPUT/f'input_{i}.jpeg'
            with Image.open(path) as im:
                bgr=np.array(ImageOps.exif_transpose(im).convert('RGB'))[:,:,::-1].copy()
            start=time.monotonic()
            r=model.predict(bgr,imgsz=size,conf=.05,iou=.5,device='cpu',verbose=False,half=False)[0]
            detections=[{'class_id':int(b.cls.item()),'class':model.names[int(b.cls.item())],
                         'confidence':float(b.conf.item()),'xyxy':b.xyxy[0].tolist()} for b in r.boxes]
            item={'image':i,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'seconds':time.monotonic()-start,'at_005':detections}
            for mode in ['common_025','author_default']:
                keep=[j for j,d in enumerate(detections) if d['confidence'] >= (thresholds[d['class']] if mode=='author_default' and name=='damage' else .25)]
                item[mode]=[detections[j] for j in keep]
                vis=r.new();vis.boxes=r.boxes[keep]
                vis.save(filename=str(OUT/f'{name}_{i}_{mode}.jpg'),line_width=2,font_size=12)
            record['images'].append(item)
            print(name,i,[(d['class'],round(d['confidence'],3)) for d in item['author_default']],flush=True)
        report['models'][name]=record
        (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    # Contact sheet is a visual inspection aid; originals retain full-resolution labels.
    from PIL import ImageDraw
    sheet=Image.new('RGB',(1200,6*480),'white');draw=ImageDraw.Draw(sheet)
    for row in range(6):
        for col,name in enumerate(['damage','bare']):
            p=OUT/f'{name}_{row+1}_author_default.jpg'
            with Image.open(p) as im:
                im.thumbnail((590,445))
                sheet.paste(im,(col*600+(600-im.width)//2,row*480+30))
            draw.text((col*600+10,row*480+8),f'Photo {row+1}: {name} / author default',fill='black')
    sheet.save(OUT/'comparison.jpg')

if __name__=='__main__': main()
