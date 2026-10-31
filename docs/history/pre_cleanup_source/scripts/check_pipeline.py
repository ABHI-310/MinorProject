"""Check a running local baseline using the bundled public PCB sample; no training."""
from pathlib import Path
import argparse,json,re,uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]


def check(base_url):
    equipment='CHECK_'+uuid.uuid4().hex[:10]
    with httpx.Client(base_url=base_url.rstrip('/'),timeout=60) as client:
        home=client.get('/');assert home.status_code==200 and 'PCB Smart Maintenance' in home.text
        provenance=client.get('/frontend_provenance.json').json()
        assert provenance['branch']=='Real-Minor' and provenance['source_unchanged'] is True
        assert provenance['dashboard_data']=='live API and persisted inspection records'
        for asset in re.findall(r'(?:src|href)="(/assets/[^\"]+)"',home.text):
            assert client.get(asset).status_code==200,asset
        health=client.get('/api/health');assert health.status_code==200
        assert client.get('/health').status_code==200
        sample=client.get('/demo/pcb.jpg');assert sample.status_code==200
        models=client.get('/api/models').json()['models']
        assert all(model['available'] for model in models) and len(models)==2
        fields={'equipment_id':equipment,'model_type':'target','event_id':str(uuid.uuid4()),'annotate':'false'}
        first=client.post('/api/detect',data=fields,files={'file':('sample.jpg',sample.content,'image/jpeg')})
        assert first.status_code==200,first.text
        assert first.json()['rul']['health_score'] is None
        assert client.get(first.json()['image_url']).status_code==200
        retry=client.post('/api/detect',data=fields,files={'file':('sample.jpg',sample.content,'image/jpeg')})
        assert retry.status_code==200 and retry.json()['replayed'] is True
        zero=client.post('/api/detect',data={**fields,'event_id':str(uuid.uuid4()),'confidence':'1'},
                         files={'file':('sample.jpg',sample.content,'image/jpeg')})
        assert zero.status_code==200 and zero.json()['count']==0
        metrics=client.get('/api/metrics',params={'equipment_id':equipment}).json()
        assert metrics['inspection_count']==2 and len(metrics['history'])==2
        assert metrics['detection_count']==first.json()['count']
        dashboard=client.get('/api/dashboard',params={'equipment_id':equipment}).json()
        assert dashboard['inspection_count']==2
        assert dashboard['detection_count']==metrics['detection_count']
        assert dashboard['health']['database']==health.json()['database']
        source=client.post('/api/detect',data={**fields,'equipment_id':equipment+'_SOURCE',
            'model_type':'source','event_id':str(uuid.uuid4()),'annotate':'true'},
            files={'file':('sample.jpg',sample.content,'image/jpeg')})
        assert source.status_code==200,source.text
        assert source.json()['model_type']=='source'
        assert source.json()['annotated_image_url']
        assert client.get(source.json()['annotated_image_url']).status_code==200
        other=client.get('/api/metrics',params={'equipment_id':equipment+'_OTHER'}).json()
        assert other['inspection_count']==0
        slow=client.get('/api/rul/synthetic/EQP_000').json()
        assert slow['synthetic'] and slow['failure_observed'] is False
        assert all(row['rul_cycles'] is None for row in slow['timeline'])
        normal=client.get('/api/rul/synthetic/EQP_001').json()
        assert normal['synthetic'] and normal['failure_observed'] is True
        report={'status':'passed','url':base_url,'test_equipment_id':equipment,
                'database':health.json()['database'],'training_started':False,
                'frontend':provenance,
                'checks':['Dashboard HTML and compiled assets','Bundled sample image',
                          'Canonical API and legacy compatibility','Both verified model records',
                          'Real source and target image inference','Saved images and annotations','Live dashboard aggregates',
                          'Idempotent retry','Zero-detection inspection saved',
                          'History and totals persisted','Equipment isolation',
                          'Synthetic observed-failure and censored lifecycles','Unknown live RUL without exposure']}
        (ROOT/'reports/pipeline').mkdir(exist_ok=True)
        (ROOT/'reports/pipeline/http_checks.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8000')
    check(parser.parse_args().url)
