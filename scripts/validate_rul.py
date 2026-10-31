from _bootstrap import ROOT
"""Validate Phase 3 generated labels and partition boundaries."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rul_synthesis.degradation_model import EquipmentDegradationModel,DEFECT_SEVERITY


def validate():
    folder=ROOT/'data/synthetic/phase3_rul'
    summary=json.loads((folder/'summary.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    payload=json.loads((folder/'scenarios.json').read_text());items=payload['scenarios']
    regenerated=EquipmentDegradationModel(seed=summary['seed']).simulate_batch(summary['equipment'],summary['observation_horizon_cycles'])
    assert items==[EquipmentDegradationModel.to_dict(x) for x in regenerated]
    assignments=json.loads((folder/'splits.json').read_text())['assignments']
    assert len(assignments)==len(items)==len({s['equipment_id'] for s in items})
    with (folder/'rul_data.csv').open() as h:rows=list(csv.DictReader(h))
    grouped={}
    for row in rows:grouped.setdefault(row['equipment_id'],[]).append(row)
    for item in items:
        points=grouped[item['equipment_id']]
        assert len(points)==len(item['health_curve'])
        assert [int(p['cycle']) for p in points]==list(range(len(points)))
        health=np.asarray(item['health_curve']);assert np.isfinite(health).all() and ((health>=0)&(health<=1)).all()
        assert (np.diff(health)<=0).all()
        failure=item['failure_cycle']
        if item['failure_observed']:
            assert failure==len(points)-1 and health[-1]<=item['failure_threshold']
            assert (health[:-1]>item['failure_threshold']).all()
        else:
            assert failure is None and (health>item['failure_threshold']).all()
        for cycle,p in enumerate(points):
            assert p['synthetic']=='True' and p['split']==assignments[item['equipment_id']]
            assert p['failure_observed']==str(item['failure_observed'])
            counts={k:int(p['defects_'+k]) for k in DEFECT_SEVERITY}
            assert all(n>=0 for n in counts.values())
            assert sum(counts.values())==int(p['defect_count'])==item['defect_progression'][cycle]
            assert abs(sum(DEFECT_SEVERITY[k]*v for k,v in counts.items())-float(p['severity_score']))<1e-9
            if failure is None:
                assert p['rul_cycles']=='' and p['failure_cycle']==''
                assert int(p['rul_lower_bound_cycles'])==len(points)-cycle
            else:
                assert int(p['rul_cycles'])==failure-cycle
                assert int(p['rul_lower_bound_cycles'])==failure-cycle
    splits={}
    for split in ['train','val','test']:
        with (folder/f'{split}.csv').open() as h:subset=list(csv.DictReader(h))
        assert subset==[r for r in rows if r['split']==split]
        splits[split]={r['equipment_id'] for r in subset}
    assert not (splits['train']&splits['val'] or splits['train']&splits['test'] or splits['val']&splits['test'])
    assert len(rows)==summary['rows'] and len(grouped)==summary['equipment']
    report={'status':'passed','synthetic':True,'equipment':len(items),'cycle_records':len(rows),
            'observed_failures':sum(s['failure_observed'] for s in items),
            'censored_equipment':sum(not s['failure_observed'] for s in items),
            'equipment_per_split':{s:len(v) for s,v in splits.items()},
            'checks':['All generated file hashes','Seed reproduces all scenarios','Every cycle/health/count/severity record',
                      'First threshold crossing and exact per-cycle RUL','Unknown exact labels and valid lower bounds for censoring',
                      'Disjoint equipment across CSV partitions','Partition CSV rows exactly match full dataset'],
            'real_world_rul_accuracy_claimed':False}
    output=ROOT/'reports/phase3/validation.json';output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':validate()
