import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from revamp import smoke
from revamp.runtime import configure
from revamp.build_resume_notebook import build


def test_notebook_embeds_resume_and_compiles(tmp_path):
    package=tmp_path/"package";package.mkdir()
    (package/"package_hashes.json").write_text('{}')
    nb=build(package,tmp_path/"resume.ipynb")
    for i,cell in enumerate(nb['cells']):
        if cell['cell_type']=='code':compile(''.join(cell['source']),f'cell{i}','exec')
    setup=''.join(nb['cells'][2]['source']);run=''.join(nb['cells'][4]['source'])
    assert 'if not WORK.exists()' in setup
    assert 'EMBEDDED_TOOLS' in setup and "sys.modules" in setup
    assert '--checkpoint' in run and '--resume-only' in run
    assert 'killpg' in run and 'finally:' in run and 'FileLink' in run
    # Constraints must contain actual newlines, not literal backslash-n text.
    dependency=''.join(nb['cells'][1]['source'])
    assert "\\\\n' for name" not in dependency


@pytest.fixture
def coordinator(tmp_path,monkeypatch):
    package=tmp_path/"package";package.mkdir()
    (package/"package_hashes.json").write_text('{}')
    monkeypatch.setattr(smoke,'verify_package',lambda _:None)
    monkeypatch.setattr(smoke,'signature',lambda _:'same')
    return package,tmp_path/"run"


def test_resume_only_never_starts_without_checkpoint(coordinator,monkeypatch):
    package,run=coordinator
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:pytest.fail('Unexpected training'))
    with pytest.raises(RuntimeError,match='no checkpoint'):
        smoke.check(package,run,resume_only=True)


def test_completed_run_is_reused_and_mismatch_refused(coordinator,monkeypatch):
    package,run=coordinator;run.mkdir()
    (run/'run_signature.json').write_text(json.dumps({'sha256':'same'}))
    report={'status':'passed','package_hashes_sha256':smoke.digest(package/'package_hashes.json')}
    (run/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:pytest.fail('Unexpected retraining'))
    assert smoke.check(package,run)==report
    (run/'run_signature.json').write_text(json.dumps({'sha256':'different'}))
    with pytest.raises(ValueError,match='another package'):
        smoke.check(package,run)


def test_concurrent_worker_refused(coordinator):
    import fcntl
    package,run=coordinator;run.mkdir()
    with (run/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already active'):
            smoke.check(package,run)


def test_import_does_not_roll_back_local_progress(coordinator):
    package,run=coordinator;run.mkdir()
    (run/'fresh_run_checkpoint.zip').write_bytes(b'existing progress')
    with pytest.raises(ValueError,match='Local progress exists'):
        smoke.check(package,run,checkpoint=run/'old.zip')


def test_expanded_checkpoint_checks_hashes(tmp_path):
    folder=tmp_path/'expanded';folder.mkdir()
    (folder/'signature.json').write_text('{}')
    record={'sha256':{'signature.json':smoke.digest(folder/'signature.json')}}
    (folder/'checkpoint_manifest.json').write_text(json.dumps(record))
    archive=smoke.checkpoint_archive(folder,tmp_path)
    with zipfile.ZipFile(archive) as z:assert set(z.namelist())=={'signature.json','checkpoint_manifest.json'}
    (folder/'signature.json').write_text('tampered')
    with pytest.raises(ValueError,match='Changed expanded'):
        smoke.checkpoint_archive(folder,tmp_path)


def test_setup_config_reuses_matching_files_and_rejects_stale(tmp_path,monkeypatch):
    from revamp import runtime
    monkeypatch.setattr(runtime,'verify_package',lambda _:None)
    package=tmp_path/'package';package.mkdir()
    manifest={'names':runtime.NAMES,'views':[{'dataset':d,'split':s,'kind':'full','image':f'{d}/{s}.jpg'} for d in ['source','target'] for s in ['train','val']]}
    (package/'manifest.json').write_text(json.dumps(manifest))
    output=tmp_path/'configs'
    first=configure(package,output)
    times={p:p.stat().st_mtime_ns for p in output.iterdir()}
    assert configure(package,output)==first
    assert times=={p:p.stat().st_mtime_ns for p in output.iterdir()}
    (output/'target_train.txt').write_text('stale path')
    with pytest.raises(ValueError,match='configuration differs'):configure(package,output)


def test_worker_accepts_later_epoch_and_skips_final_checkpoint(tmp_path,monkeypatch):
    import torch
    import ultralytics
    from types import SimpleNamespace
    run=tmp_path/'run';weights=run/'training/weights';weights.mkdir(parents=True)
    (run/'signature.json').write_text(json.dumps({'sha256':'same'}))
    monkeypatch.setattr(smoke,'signature',lambda _:'same')
    calls=[]
    class FakeYOLO:
        def __init__(self,path):self.callbacks={}
        def add_callback(self,event,callback):self.callbacks[event]=callback
        def train(self,**kwargs):
            calls.append(kwargs)
            trainer=SimpleNamespace(start_epoch=5,optimizer=SimpleNamespace(state={1:{'step':5}}))
            self.callbacks['on_train_start'](trainer)
    monkeypatch.setattr(ultralytics,'YOLO',FakeYOLO)
    torch.save({'epoch':4,'optimizer':{'state':{1:{'step':5}}}},weights/'last.pt')
    smoke.worker(tmp_path,tmp_path/'data.yaml',run,True,'cpu')
    assert len(calls)==1 and calls[0]['resume'] is True
    assert json.loads((run/'resume_verified.json').read_text())['start_epoch']==5
    torch.save({'epoch':79,'optimizer':{'state':{1:{'step':80}}}},weights/'last.pt')
    smoke.worker(tmp_path,tmp_path/'data.yaml',run,True,'cpu')
    assert len(calls)==1
