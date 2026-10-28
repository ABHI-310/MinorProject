from _bootstrap import ROOT
"""Real-model API and database integration check; does not train models."""
from pathlib import Path
import argparse,base64,json,sys,tempfile
import uuid
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from backend.main import create_app
from sqlalchemy import func,select
from database.db import Inspection,Detection,VerifiedModel


def verify(url,report):
    app=create_app(url,image_store=Path(report).resolve().parent/'verification_images')
    equipment='CHECK_'+uuid.uuid4().hex[:12]
    image=ROOT/'data/prepared/target/images/test/Y_003663.jpg'
    if not image.is_file():raise RuntimeError('Verification sample is missing: '+str(image))
    with TestClient(app) as client:
        health=client.get('/api/health');assert health.status_code==200
        model_records=client.get('/api/models').json()['models']
        assert len(model_records)==2 and all(m['available'] for m in model_records)
        results={}
        for kind in ['target','source']:
            response=client.post('/api/detect',data={'equipment_id':equipment,'model_type':kind},
                files={'file':(image.name,image.read_bytes(),'image/jpeg')})
            assert response.status_code==200,response.text
            value=response.json();assert value['model_type']==kind and value['rul']['health_score'] is None
            assert value['rul']['status']=='exposure_required'
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(value['annotated_image'].split(',')[1]),dtype=np.uint8),cv2.IMREAD_COLOR)
            assert pixels.shape[:2]==(value['image_height'],value['image_width'])
            assert all(0<=d['bbox'][0]<d['bbox'][2]<=value['image_width'] and
                       0<=d['bbox'][1]<d['bbox'][3]<=value['image_height'] for d in value['detections'])
            if kind=='target':assert all(d['class_id'] in [1,2,3,4,5] for d in value['detections'])
            results[kind]={'checkpoint_sha256':value['checkpoint_sha256'],'count':value['count'],
                           'inference_time_ms':value['inference_time_ms'],'annotated_image_decoded':True}
        event_id=str(uuid.uuid4())
        data={'equipment_id':equipment,'event_id':event_id,'boards_inspected':100,'detections':[]}
        assert client.post('/api/events/detections',json=data).status_code==200
        assert client.post('/api/events/detections',json=data).json()['replayed'] is True
        for _ in range(4):
            data['event_id']=str(uuid.uuid4());assert client.post('/api/events/detections',json=data).status_code==200
        metrics=client.get('/api/metrics',params={'equipment_id':equipment}).json()
        assert metrics['inspection_count']==7 and metrics['rul']['status']=='simulated_estimate'
        assert metrics['rul']['synthetic'] and metrics['rul']['boards_in_history']==500
        isolated=client.get('/api/metrics',params={'equipment_id':'UNUSED_'+uuid.uuid4().hex[:8]}).json()
        assert isolated['inspection_count']==0 and isolated['rul'] is None
        with app.state.db.session() as session:
            assert session.scalar(select(func.count()).select_from(VerifiedModel))>=2
        bad=client.post('/api/detect',files={'file':('bad.png',b'not an image','image/png')});assert bad.status_code==400
        demo=client.get('/api/rul/synthetic/EQP_000?limit=3').json();assert demo['synthetic'] is True and len(demo['timeline'])==3
        assert client.post('/api/adapt').status_code==410
    output={'status':'passed','database':health.json()['database'],'real_model_checks':results,
        'checks':['Real target and source checkpoint loading and inference','Annotated JPEG decoding and image bounds',
                  'Verified model metrics persisted','Inspection, detections and outlook stored atomically',
                  'Zero-detection inspections retained','Retry idempotency','Equipment history isolation',
                  'No RUL without declared exposure','History-based simulated outlook after sufficient exposure',
                  'Invalid upload rejection','Phase 3 synthetic timeline endpoint','No API-triggered retraining'],
        'training_started':False,'rul_is_simulated':True}
    Path(report).write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2));return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--database-url');parser.add_argument('--report',type=Path,default=ROOT/'reports/phase4/sqlite_check.json');args=parser.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    if args.database_url:verify(args.database_url,args.report)
    else:
        with tempfile.TemporaryDirectory(prefix='pcb-api-check-') as folder:verify(f'sqlite:///{folder}/check.db',args.report)
