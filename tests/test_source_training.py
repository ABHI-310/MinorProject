import ast
import base64
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from revamp import train_source as training
from revamp.build_source_notebook import build


def test_early_stop_history_must_be_complete():
    rows=[{'epoch':'1'},{'epoch':'2'}]
    assert training.validate_history(rows,{'epoch':1,'training_finished':True},60)==2
    with pytest.raises(RuntimeError,match='intentional'):
        training.validate_history(rows,{'epoch':1,'training_finished':False},60)
    with pytest.raises(RuntimeError,match='duplicated'):
        training.validate_history([rows[0],rows[0]],{'epoch':1,'training_finished':True},60)


def test_regression_gate_never_promotes_automatically():
    def scores(ap=.9,localization=.47):
        return {'map50':ap,'map50_95':localization,'per_class':{n:{'ap50':ap} for n in training.NAMES}}
    baseline=scores();candidate=scores(.91,.49)
    assert training.review_candidate(baseline,candidate)['worth_further_review']
    assert not training.review_candidate(baseline,candidate)['automatic_promotion']
    candidate['per_class']['spur']['ap50']=.8
    assert not training.review_candidate(baseline,candidate)['worth_further_review']
    candidate=scores(.91,.49);candidate['per_class'].pop('short')
    assert not training.review_candidate(baseline,candidate)['worth_further_review']


def test_early_stopped_checkpoint_does_not_restart(tmp_path,monkeypatch):
    import torch,ultralytics
    run=tmp_path/'run';(run/'training/weights').mkdir(parents=True)
    torch.save({'epoch':1,'optimizer':{'state':{1:{}}}},run/'training/weights/last.pt')
    torch.save({'epoch':1,'training_finished':True},run/'training/trainer_state.pt')
    monkeypatch.setattr(ultralytics,'YOLO',lambda *a:pytest.fail('Finished model retrained'))
    training.train(tmp_path,tmp_path/'data.yaml',run,'source','cpu','test',True)


def test_source_notebook_has_own_identity_and_matching_code(tmp_path):
    package=tmp_path/'package';package.mkdir();(package/'package_hashes.json').write_text('{}')
    nb=build(package,tmp_path/'train.ipynb')
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
    setup=''.join(nb['cells'][2]['source'])
    node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EMBEDDED_TOOLS' for t in n.targets))
    payload=ast.literal_eval(node.value)
    assert 'pilot.py' not in payload
    assert base64.b64decode(payload['train_source.py']['base64'])==Path(training.__file__).read_bytes()
    run=''.join(nb['cells'][4]['source'])
    assert 'revamp.train_source' in run and 'source_run_checkpoint.zip' in run
    assert 'source_training_tools' in setup and 'killpg' in run
    assert training.recipe('source')['epochs']==60
    with pytest.raises(ValueError):training.recipe('target')
