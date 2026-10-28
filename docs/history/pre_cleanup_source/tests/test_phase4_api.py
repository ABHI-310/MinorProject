from uuid import uuid4
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func,select
from sqlalchemy.exc import IntegrityError
import pytest

from backend.main import create_app
from database.db import Detection,Inspection


class StubRegistry:
    @staticmethod
    def canonical(name):
        if name=='adapted':return 'target'
        if name not in ['source','target']:raise ValueError('Invalid model type')
        return name
    def list_models(self):return []
    def infer(self,image,model_type,confidence,annotate):
        return {'detections':[{'class_id':2,'class_name':'open_circuit','confidence':.8,'bbox':[1.,2.,5.,6.]}],
                'inference_time_ms':1.,'image_width':image.shape[1],'image_height':image.shape[0],
                'confidence_cutoff':confidence,'checkpoint_sha256':'a'*64,'annotated_image':None}


def client_for(tmp_path):return TestClient(create_app(f'sqlite:///{tmp_path}/api.db',StubRegistry(),tmp_path/'images'))

def test_zero_defect_history_is_persisted_and_equipment_isolated(tmp_path):
    with client_for(tmp_path) as c:
        for _ in range(5):
            response=c.post('/events/detections',json={'equipment_id':'LINE_A','boards_inspected':100,'detections':[]})
            assert response.status_code==200
        assert response.json()['rul']['status']=='simulated_estimate'
        assert response.json()['rul']['synthetic'] is True
        metrics=c.get('/metrics?equipment_id=LINE_A').json()
        assert metrics['inspection_count']==5 and metrics['detection_count']==0
        assert len(metrics['history'])==5
        other=c.get('/metrics?equipment_id=LINE_B').json()
        assert other['inspection_count']==0 and other['rul'] is None


def test_missing_exposure_and_short_history_do_not_invent_rul(tmp_path):
    with client_for(tmp_path) as c:
        payload={'detections':[]}
        assert c.post('/events/detections',json=payload).json()['rul']['status']=='exposure_required'
        response=c.post('/events/detections',json={**payload,'boards_inspected':100}).json()
        assert response['rul']['rul_cycles'] is None and response['rul']['status']=='insufficient_history'


def test_edge_retry_is_idempotent_and_conflicts_are_rejected(tmp_path):
    with client_for(tmp_path) as c:
        payload={'event_id':str(uuid4()),'detections':[{'class_name':'open','confidence':.9,'bbox':[0,0,10,10]}]}
        first=c.post('/events/detections',json=payload);second=c.post('/events/detections',json=payload)
        assert first.status_code==second.status_code==200 and second.json()['replayed'] is True
        assert second.json()['detections'][0]['class_name']=='open_circuit'
        bad=c.post('/events/detections',json={**payload,'detections':[]});assert bad.status_code==409
        with c.app.state.db.session() as s:
            assert s.scalar(select(func.count()).select_from(Inspection))==1
            assert s.scalar(select(func.count()).select_from(Detection))==1


def test_actual_upload_path_replay_and_bad_inputs(tmp_path):
    with client_for(tmp_path) as c:
        stream=BytesIO();Image.new('RGB',(10,12)).save(stream,format='PNG');raw=stream.getvalue()
        event=str(uuid4());data={'equipment_id':'LINE_A','model_type':'adapted','event_id':event}
        response=c.post('/api/detect',data=data,files={'file':('pcb.png',raw,'image/png')})
        assert response.status_code==200
        value=response.json();assert value['model_type']=='target' and value['image_width']==10
        assert value['detections'][0]['class_name']=='open_circuit' and value['rul']['rul_cycles'] is None
        assert c.post('/api/detect',data=data,files={'file':('pcb.png',raw,'image/png')}).json()['replayed'] is True
        assert c.post('/detect',files={'file':('bad.png',b'bad','image/png')}).status_code==400
        assert c.post('/detect',data={'confidence':'nan'},files={'file':('pcb.png',raw,'image/png')}).status_code==422
        assert c.post('/detect',data={'model_type':'bad'},files={'file':('pcb.png',raw,'image/png')}).status_code==422
        assert c.post('/events/detections',json={'detections':[{'class_name':'short','confidence':.9,'bbox':[4,0,1,2]}]}).status_code==422
        assert c.post('/adapt').status_code==410


def test_time_filter_and_synthetic_demo(tmp_path):
    with client_for(tmp_path) as c:
        assert c.get('/health').json()['database']=='sqlite'
        assert c.get('/metrics?hours=0').status_code==422
        assert c.get('/rul/synthetic/EQP_000?limit=3').json()['timeline'][0]['synthetic'] is True
        assert c.get('/rul/synthetic/MISSING').status_code==404


def test_partial_database_write_is_rolled_back(tmp_path):
    with client_for(tmp_path) as c:
        db=c.app.state.db
        with pytest.raises(IntegrityError):
            db.record_inspection('LINE_A',[{'class_name':None,'confidence':.8,'bbox':[0,0,1,1]}],
                'edge',100,'a'*64)
        with db.session() as session:
            assert session.scalar(select(func.count()).select_from(Inspection))==0
            assert session.scalar(select(func.count()).select_from(Detection))==0


def test_metrics_honour_time_window(tmp_path):
    from datetime import datetime,timedelta,timezone
    with client_for(tmp_path) as c:
        c.post('/events/detections',json={'detections':[]})
        with c.app.state.db.session() as session:
            event=session.scalar(select(Inspection))
            event.timestamp=datetime.now(timezone.utc)-timedelta(hours=48)
        assert c.get('/metrics?hours=24').json()['inspection_count']==0
        assert c.get('/metrics?hours=72').json()['inspection_count']==1


def test_dashboard_api_prefix_shares_the_legacy_pipeline(tmp_path):
    with client_for(tmp_path) as c:
        assert c.get('/api/health').status_code==200
        response=c.post('/api/events/detections',json={'equipment_id':'PREFIX_CHECK','detections':[]})
        assert response.status_code==200
        assert c.get('/metrics?equipment_id=PREFIX_CHECK').json()['inspection_count']==1
        assert c.get('/api/metrics?equipment_id=PREFIX_CHECK').json()['inspection_count']==1
        assert c.get('/api/models').status_code==200
        assert c.post('/api/adapt').status_code==410


def test_supplied_frontend_unknown_rul_contract(tmp_path):
    with client_for(tmp_path) as c:
        stream=BytesIO();Image.new('RGB',(10,12)).save(stream,format='PNG')
        response=c.post('/detect',data={'model_type':'adapted'},files={'file':('pcb.png',stream.getvalue(),'image/png')})
        assert response.status_code==200
        assert response.json()['model_type']=='target' and response.json()['rul'] is None
        assert response.json()['rul_status']['status']=='exposure_required'
        assert c.get('/metrics').json()['rul'] is None
        assert c.get('/api/metrics').json()['rul']['status']=='exposure_required'


def test_dashboard_counts_are_not_limited_to_recent_history(tmp_path):
    with client_for(tmp_path) as c:
        for _ in range(102):
            assert c.post('/api/events/detections',json={'equipment_id':'LINE_A','detections':[
                {'class_name':'short','confidence':.9,'bbox':[0,0,10,10]}]}).status_code==200
        c.post('/api/events/detections',json={'equipment_id':'LINE_A','detections':[]})
        value=c.get('/api/dashboard?equipment_id=LINE_A').json()
        assert value['inspection_count']==103 and value['defective_inspections']==102
        assert value['detection_count']==value['critical_detection_count']==102
        assert value['defect_counts']=={'short':102}
        assert len(value['history'])==100 and value['history_limit']==100
        assert sum(day.get('short',0) for day in value['daily_defects'].values())==102
        assert c.get('/api/dashboard?equipment_id=LINE_B').json()['inspection_count']==0
        assert c.get('/api/dashboard?days=31').status_code==422


def test_saved_images_survive_replay_and_app_restart(tmp_path):
    raw=BytesIO();Image.new('RGB',(10,12),'red').save(raw,format='PNG')
    event=str(uuid4());data={'event_id':event}
    with client_for(tmp_path) as c:
        response=c.post('/api/detect',data=data,files={'file':('pcb.png',raw.getvalue(),'image/png')})
        assert response.status_code==200
        url=response.json()['image_url']
        assert c.get(url).headers['content-type']=='image/jpeg'
        with Image.open(BytesIO(c.get(url).content)) as image:assert image.size==(10,12)
        replay=c.post('/api/detect',data=data,files={'file':('pcb.png',raw.getvalue(),'image/png')}).json()
        assert replay['replayed'] and replay['image_url']==url
        assert c.get('/api/inspection-images/invalid/original').status_code==404
        assert c.get('/api/inspection-images/'+'a'*64+'/invalid').status_code==404
    with client_for(tmp_path) as c:
        assert c.get(url).status_code==200
        assert c.get('/api/dashboard').json()['history'][0]['image_url']==url


def test_dashboard_excludes_older_window(tmp_path):
    from datetime import timedelta
    from database.db import utcnow
    with client_for(tmp_path) as c:
        c.post('/api/events/detections',json={'detections':[{'class_name':'open','confidence':.9,'bbox':[0,0,10,10]}]})
        with c.app.state.db.session() as s:
            for item in s.scalars(select(Inspection)):item.timestamp=utcnow()-timedelta(days=31)
            for item in s.scalars(select(Detection)):item.timestamp=utcnow()-timedelta(days=31)
        value=c.get('/api/dashboard').json()
        assert value['inspection_count']==value['detection_count']==value['critical_detection_count']==0
        assert value['history']==[] and value['daily_defects']=={}


def test_default_upload_uses_six_class_source_checkpoint(tmp_path):
    raw=BytesIO();Image.new('RGB',(10,12)).save(raw,format='PNG')
    with client_for(tmp_path) as c:
        result=c.post('/api/detect',files={'file':('pcb.png',raw.getvalue(),'image/png')})
        assert result.status_code==200 and result.json()['model_type']=='source'
