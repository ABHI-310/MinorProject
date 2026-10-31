"""Exercise a candidate through an isolated app registry; never changes active models."""
import os,sys,time,json,hashlib,base64,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from revamp.validate_yolo_pcb import load_legacy,BASE
import torch,numpy as np,cv2
from PIL import Image,ImageOps,ImageEnhance,ImageFilter
from utils.datasets import letterbox
from utils.general import non_max_suppression,scale_coords
WEIGHTS=BASE/'JiaLim98/weights/baseline_fpn_loss.pt'
SHA='616b6054e463042ca4d4e056c677a02f3e59948465b68e6a2cf4b01ac88a84ea'
assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==SHA
torch.set_num_threads(4);start=time.perf_counter();candidate=load_legacy(WEIGHTS);load_ms=(time.perf_counter()-start)*1000
sys.path.insert(0,str(ROOT/'src'))
from backend.model_registry import ModelRegistry
from backend.main import create_app
from fastapi.testclient import TestClient
from revamp.diagnose import analyze

class CandidateRegistry(ModelRegistry):
    def list_models(self):
        rows=super().list_models()
        for r in rows:
            if r['model_type']=='source':
                r.update(model_name='YOLO-PCB replacement validation ONLY',checkpoint_sha256=SHA,image_size=960,map_score=None,map50_95=None,precision=None,recall=None,evaluation='Replacement validation pending',method='External pretrained checkpoint')
        return rows
    def infer(self,image,model_type,confidence,annotate):
        if self.canonical(model_type)!='source':return super().infer(image,model_type,confidence,annotate)
        with self.lock:
            start=time.perf_counter();scaled=letterbox(image,new_shape=960,auto=False)[0]
            tensor=torch.from_numpy(np.ascontiguousarray(scaled[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
            with torch.inference_mode():p=non_max_suppression(candidate(tensor)[0],confidence,.7)[0]
            p=p.clone()
            if len(p):p[:,:4]=scale_coords(tensor.shape[2:],p[:,:4],image.shape)
            detections=[];canvas=image.copy()
            for *box,score,c in p.tolist():
                x1,y1,x2,y2=box
                assert np.isfinite(box).all() and 0<=x1<x2<=image.shape[1] and 0<=y1<y2<=image.shape[0]
                detections.append(dict(class_id=int(c),class_name=candidate.names[int(c)],confidence=score,bbox=box))
                if annotate:cv2.rectangle(canvas,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            encoded=None
            if annotate:
                ok,buf=cv2.imencode('.jpg',canvas);assert ok
                encoded='data:image/jpeg;base64,'+base64.b64encode(buf).decode()
            return dict(detections=detections,inference_time_ms=(time.perf_counter()-start)*1000,checkpoint_sha256=SHA,image_width=image.shape[1],image_height=image.shape[0],confidence_cutoff=confidence,annotated_image=encoded)

def main():
    out=ROOT/'artifacts/verification/yolo_pcb_replacement';out.mkdir(parents=True,exist_ok=True)
    package=ROOT/'artifacts/kaggle/phase2_v2/package'
    rows=[r for r in json.loads((package/'manifest.json').read_text())['views'] if r['dataset']=='source' and r['split']=='val' and r['kind']=='full']
    registry=CandidateRegistry();app=create_app('sqlite:///'+str(out/'isolated_api.db'),registry,out/'api_images')
    api_checks=[];times=[];stress=[]
    with TestClient(app) as client:
        for cls in range(6):
            row=next(r for r in rows if any(int(a[0])==cls for a in r['labels']))
            raw=(package/row['image']).read_bytes();im=np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))[:,:,::-1].copy()
            direct=registry.infer(im,'source',.9,True)
            r=client.post('/api/detect',files={'file':('pcb.jpg',raw,'image/jpeg')},data={'model_type':'source','confidence':'.9','annotate':'true','equipment_id':'REPLACEMENT_VALIDATION'})
            assert r.status_code==200,r.text
            result=r.json();assert result['detections']==direct['detections']
            assert result['checkpoint_sha256']==SHA
            assert result['annotated_image'].startswith('data:image/jpeg;base64,')
            api_checks.append(dict(class_id=cls,parent_id=row['parent_id'],api_matches_direct=True,detections=len(result['detections']),image_width=result['image_width'],image_height=result['image_height']))
            for _ in range(3):times.append(registry.infer(im,'source',.9,False)['inference_time_ms'])
        for row in rows[::10]:
            im=Image.open(package/row['image']).convert('RGB')
            for name,variant in [('original',im),('darker',ImageEnhance.Brightness(im).enhance(.7)),('brighter',ImageEnhance.Brightness(im).enhance(1.3)),('blur_1px',im.filter(ImageFilter.GaussianBlur(1)))]:
                pred=registry.infer(np.asarray(variant)[:,:,::-1].copy(),'source',.9,False)
                d=analyze(row,[{'class':x['class_id'],'box':x['bbox'],'confidence':x['confidence']} for x in pred['detections']])
                stress.append(dict(parent_id=row['parent_id'],variant=name,counts={k:sum(c[k] for c in d['counts'].values()) for k in ['tp','fp','fn']}))
        blanks=[]
        for value in [0,127,255]:
            r=registry.infer(np.full((960,960,3),value,dtype=np.uint8),'source',.9,False)
            blanks.append(dict(pixel_value=value,detections=len(r['detections'])))
    summary=dict(checkpoint_sha256=SHA,load_ms=load_ms,api_checks=api_checks,latency_ms=dict(n=len(times),median=float(np.median(times)),p95=float(np.percentile(times,95)),scope='Local CPU, 4 torch threads; preprocessing+inference+NMS, annotation excluded. No hardware deployment claim.'),stress_scope='12 source validation images, shared ancestry, exploratory fixed .90 threshold; artificial perturbations are not camera validation.',stress=stress,synthetic_blank_checks=blanks,limitations=['No independent six-class holdout','No clean full-board false-positive benchmark','Legacy models/utils module namespace requires isolation before production integration'])
    (out/'integration_and_stress.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
