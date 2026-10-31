"""Run the fixed demo through the real API using isolated temporary storage."""
import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from _bootstrap import ROOT
from fastapi.testclient import TestClient
from backend.main import create_app
from evaluate_k500_development import match


def main():
    folder=ROOT/'examples/k500_demo'
    manifest=json.loads((folder/'manifest.json').read_text())
    output=ROOT/'reports/revamp/semester_demo_verification.json'
    expected='b79737ec6ad99626f4c21be904e91a5495fd986723c22da65ab3dbeacca46722'
    records=[];totals=dict(tp=0,fp=0,fn=0)
    with tempfile.TemporaryDirectory(prefix='pcb-demo-') as temporary:
        storage=Path(temporary)
        with TestClient(create_app('sqlite:///'+str(storage/'demo.db'),image_store=storage/'images')) as client:
            for row in manifest['images']:
                image=folder/row['image']
                assert hashlib.sha256(image.read_bytes()).hexdigest()==row['image_sha256']
                event=str(uuid4())
                payload=dict(equipment_id='PRESENTATION_CHECK',model_type='k500_coco',confidence='.25',boards_inspected='1',event_id=event)
                with image.open('rb') as stream:
                    response=client.post('/api/detect',data=payload,files={'file':(image.name,stream,'image/jpeg')})
                assert response.status_code==200,response.text
                data=response.json();assert data['checkpoint_sha256']==expected
                assert data['model_type']=='k500_coco'
                for url in [data['image_url'],data['annotated_image_url']]:assert client.get(url).status_code==200
                width,height=row['width'],row['height'];gt=[]
                for cid,x,y,w,h in row['labels']:
                    gt.append({'class':int(cid),'box':[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]})
                predictions=[{'class':d['class_id'],'box':d['bbox'],'confidence':d['confidence']} for d in data['detections']]
                counts=match(predictions,gt)
                counts={key:sum(c[key] for c in counts.values()) for key in totals}
                for key in totals:totals[key]+=counts[key]
                records.append(dict(image=row['image'],expected=row['expected_counts'],predictions=data['detections'],counts=counts))
                print(row['image'],counts,flush=True)
            dashboard=client.get('/api/dashboard?equipment_id=PRESENTATION_CHECK').json()
            assert dashboard['inspection_count']==10
            assert dashboard['rul']['status']=='insufficient_history'
            assert dashboard['maintenance_trend']['status'] in ('review_required','no_rule_trigger')
            demo=client.get('/api/maintenance/demo').json()
            assert demo['synthetic'] and demo['trend']['alert'] and not demo['stored']
            assert client.get('/api/dashboard?equipment_id=PRESENTATION_CHECK').json()['inspection_count']==10
            assert client.get('/api/dashboard?equipment_id=OTHER').json()['inspection_count']==0
    result=dict(status='PASS',role='Functional demonstration on familiar validation images; not independent accuracy',
                checkpoint_sha256=expected,real_history_modified=False,settings=dict(confidence=.25,imgsz=640,nms_iou=.4,matching_iou=.5),
                checks=['10 real CPU model uploads','original/annotated image retrieval','saved inspection history','equipment isolation','no simulated lifetime from insufficient exposure','synthetic review alert without history writes'],
                demo_detection_counts=totals,images=records)
    output.write_text(json.dumps(result,indent=2)+'\n');print('PASS:',output,totals,flush=True)

if __name__=='__main__':main()
