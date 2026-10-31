from revamp.diagnose import iou,match,analyze
from revamp.build_diagnostic_notebook import build


def test_iou_and_one_to_one_matches():
    box=[0,0,10,10]
    assert iou(box,box)==1 and iou(box,[20,20,30,30])==0
    gt=[{'class':0,'box':box}]
    assert match(gt,[{'class':1,'box':box}])==(set(),set())
    a,b=match(gt,[{'class':0,'box':box},{'class':0,'box':box}])
    assert a=={0} and len(b)==1


def test_wrong_class_is_miss_and_false_positive():
    row={'labels':[[0,.5,.5,.2,.2]],'width':100,'height':100,'parent_id':'source/x','image':'x.jpg'}
    r=analyze(row,[{'class':1,'box':[40,40,60,60],'confidence':.9}])
    assert r['counts']['missing_hole']['fn']==1
    assert r['counts']['mouse_bite']['fp']==1
    assert r['misses'][0]['best_other_class_iou']==1


def test_diagnostic_notebook_has_no_training_command(tmp_path):
    p=tmp_path/'package';p.mkdir();(p/'package_hashes.json').write_text('{}')
    nb=build(p,tmp_path/'diag.ipynb')
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
    run=''.join(nb['cells'][4]['source'])
    assert 'revamp.diagnose' in run and 'revamp.train' not in run
    assert 'diagnosis.zip' in run and 'killpg' in run
