import ast
import base64
import json
from pathlib import Path
import pytest
from revamp import train_source,train_full_boards
from revamp.build_full_boards_notebook import build


def test_only_training_views_change_recipe_stays_identical():
    assert train_full_boards.recipe('source')==train_source.recipe('source')


def test_full_board_config_is_used(tmp_path,monkeypatch):
    monkeypatch.setattr(train_full_boards,'verify_package',lambda _:None)
    monkeypatch.setattr(train_full_boards,'identity',lambda *a:('test',{}))
    monkeypatch.setattr(train_full_boards,'configure',lambda *a:{'source_full':'full.yaml','source_views':'mixed.yaml'})
    monkeypatch.setattr(train_full_boards,'evaluate',lambda *a:{})
    class StopCheck(Exception):pass
    def train(package,data,*args):
        assert data=='full.yaml'
        raise StopCheck
    monkeypatch.setattr(train_full_boards,'train',train)
    with pytest.raises(StopCheck):train_full_boards.run(tmp_path,tmp_path/'run',device='cpu')


def test_notebook_embeds_correct_runner_and_distinct_paths(tmp_path):
    package=tmp_path/'package';package.mkdir();(package/'package_hashes.json').write_text('{}')
    nb=build(package,tmp_path/'nb.ipynb')
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'cell','exec')
    setup=''.join(nb['cells'][2]['source'])
    node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EMBEDDED_TOOLS' for t in n.targets))
    payload=ast.literal_eval(node.value)
    assert 'train_source.py' not in payload
    assert base64.b64decode(payload['train_full_boards.py']['base64'])==Path(train_full_boards.__file__).read_bytes()
    run=''.join(nb['cells'][4]['source'])
    assert 'revamp.train_full_boards' in run and 'full_boards_run_checkpoint.zip' in run
    assert 'full_boards_v2' in run


def test_old_training_checkpoint_identity_cannot_match(tmp_path):
    (tmp_path/'package_hashes.json').write_text('{}')
    old,_=train_source.identity(tmp_path,'source')
    new,record=train_full_boards.identity(tmp_path,'source')
    assert old!=new and record['training_views']=='full_only'
