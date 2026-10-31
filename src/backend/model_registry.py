"""Load only the verified source, supervised target, and K500 candidate checkpoints."""
from pathlib import Path
import base64
import hashlib
import json
import os
from threading import RLock
import time

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/runtime/matplotlib'))


class ModelRegistry:
    def __init__(self):
        self.folders={'source':ROOT/'artifacts/models/source','target':ROOT/'artifacts/models/target','k500':ROOT/'artifacts/models/k500','k500_coco':ROOT/'artifacts/models/k500_coco'}
        self.loaded={};self.lock=RLock()

    @staticmethod
    def canonical(name):
        name='target' if name=='adapted' else name
        if name not in ('source','target','k500','k500_coco'):raise ValueError('model_type must be source, target, k500, or k500_coco (adapted is a legacy alias)')
        return name

    def list_models(self):
        records=[]
        for name,folder in self.folders.items():
            path=folder/'model_info.json';info=json.loads(path.read_text()) if path.is_file() else {}
            is_target=name=='target'
            records.append({'model_type':name,'model_name':'YOLOv8s supervised target' if is_target else 'YOLOv8s source',
                'available':bool(info) and (folder/'model.pt').is_file(),
                'checkpoint_sha256':info.get('checkpoint_sha256'),
                'classes':info.get('classes',[]),'evaluated_class_ids':info.get('evaluated_class_ids',list(range(6))),
                'map_score':info.get('development_test_mAP50') if is_target else info.get('baseline_test_mAP50'),
                'map50_95':info.get('development_test_mAP50_95') if is_target else info.get('baseline_test_mAP50_95'),
                'precision':info.get('development_test_precision') if is_target else None,
                'recall':info.get('development_test_recall') if is_target else None,
                'evaluation':'DsPCBSD+ development set, 992 images' if is_target else 'PKU source holdout, 93 images',
                'method':'larger-label supervised fine-tuning' if is_target else 'source-domain supervised training',
                'image_size':info.get('image_size'), 'loaded':name in self.loaded})
            if name in ('k500','k500_coco'):
                records[-1].update(model_name='YOLOv8s COCO → K500' if name=='k500_coco' else 'YOLOv8s K500 adapted candidate',
                    map_score=info.get('validation_mAP50'), map50_95=info.get('validation_mAP50_95'),
                    precision=info.get('operating_point',{}).get('precision'),
                    recall=info.get('operating_point',{}).get('recall'),
                    evaluation='Target validation, 403 images; settings selected on this set, not a final test',
                    method=('COCO-pretrained detector fine-tuned on 500 labelled target images' if name=='k500_coco' else 'PCB-source adaptation using 500 labelled target images'),
                    operating_point=info.get('operating_point'), limitations=info.get('limitations',[]),
                    nms_iou=info.get('inference',{}).get('iou'))
        return records

    def infer(self,image,model_type,confidence,annotate):
        import cv2
        import torch
        from ultralytics import YOLO
        name=self.canonical(model_type)
        with self.lock:
            if name not in self.loaded:
                folder=self.folders[name];info=json.loads((folder/'model_info.json').read_text())
                weights=folder/'model.pt'
                if hashlib.sha256(weights.read_bytes()).hexdigest()!=info['checkpoint_sha256']:
                    raise ValueError('Checkpoint checksum mismatch')
                model=YOLO(str(weights))
                if [model.names[i] for i in range(len(model.names))]!=info['classes']:
                    raise ValueError('Checkpoint class order mismatch')
                self.loaded[name]=(model,info)
            model,info=self.loaded[name];torch.set_num_threads(4)
            start=time.perf_counter()
            inference=info.get('inference',{})
            options={key:inference[key] for key in ('iou','rect','max_det','agnostic_nms','augment') if key in inference}
            result=model.predict(source=image,imgsz=info['image_size'],conf=confidence,
                classes=info.get('evaluated_class_ids'),device=os.getenv('PCB_DEVICE','cpu'),verbose=False,**options)[0]
            elapsed=(time.perf_counter()-start)*1000
            detections=[{'class_id':int(b.cls.item()),'class_name':model.names[int(b.cls.item())],
                'confidence':float(b.conf.item()),'bbox':[float(v) for v in b.xyxy[0].cpu().tolist()]} for b in result.boxes]
            annotated=None
            if annotate:
                ok,encoded=cv2.imencode('.jpg',result.plot())
                if not ok:raise ValueError('Could not encode annotated image')
                annotated='data:image/jpeg;base64,'+base64.b64encode(encoded).decode()
            return {'detections':detections,'inference_time_ms':round(elapsed,2),
                    'checkpoint_sha256':info['checkpoint_sha256'],'image_width':image.shape[1],
                    'image_height':image.shape[0],'confidence_cutoff':confidence,'annotated_image':annotated}
