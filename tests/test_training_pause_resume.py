import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from accuracy import train_recovery as training
from accuracy.notebook_controls import TrainingSession


def fake_trainer(run, epoch=0):
    folder = run/'training'
    (folder/'weights').mkdir(parents=True, exist_ok=True)
    (run/'signature.json').write_text(json.dumps({'sha256': 'same-recipe'}))
    (folder/'args.yaml').write_text('epochs: 80\n')
    (folder/'results.csv').write_text('epoch,fitness\n1,0.5\n')
    for name in ['last.pt', 'best.pt']:
        torch.save({'epoch': epoch, 'optimizer': {'state': {0: {'step': torch.tensor(1)}}}}, folder/'weights'/name)
    return SimpleNamespace(epoch=epoch, stop=False, last=folder/'weights/last.pt', best=folder/'weights/best.pt',
        stopper=SimpleNamespace(best_fitness=.5, best_epoch=epoch+1, possible_stop=False, patience=15),
        scheduler=SimpleNamespace(state_dict=lambda: {'last_epoch': epoch}))


def test_checkpoint_roundtrip_and_optimizer_survive(tmp_path):
    run=tmp_path/'old/run'
    archive=training.save_checkpoint(fake_trainer(run), run)
    restored=tmp_path/'new/run'
    training.restore_checkpoint(archive, restored, 'same-recipe')
    checkpoint=torch.load(restored/'training/weights/last.pt', weights_only=False)
    assert checkpoint['epoch']==0 and checkpoint['optimizer']['state']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('checkpoint_manifest.json'))['completed_epochs']==1
    state=torch.load(restored/'training/trainer_state.pt', weights_only=False)
    assert state['stopper']['best_epoch']==1 and state['scheduler']['last_epoch']==0


def test_failed_checkpoint_publication_keeps_previous_zip(tmp_path, monkeypatch):
    run=tmp_path/'run';trainer=fake_trainer(run)
    archive=training.save_checkpoint(trainer, run)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    original=zipfile.ZipFile.writestr
    def fail(self, name, *args, **kwargs):
        if name=='checkpoint_manifest.json':raise OSError('simulated disk failure')
        return original(self,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'writestr',fail)
    with pytest.raises(OSError,match='disk failure'):training.save_checkpoint(trainer,run)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest
    assert not archive.with_suffix('.tmp').exists()


def test_restore_rejects_wrong_recipe_or_tampering_before_writing(tmp_path):
    run=tmp_path/'run';archive=training.save_checkpoint(fake_trainer(run),run)
    destination=tmp_path/'restore'
    with pytest.raises(RuntimeError,match='mismatch'):training.restore_checkpoint(archive,destination,'different-recipe')
    assert not destination.exists()
    corrupt=tmp_path/'corrupt.zip'
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(corrupt,'w') as target:
        for name in source.namelist():target.writestr(name,b'changed' if name=='training/results.csv' else source.read(name))
    with pytest.raises(RuntimeError,match='checksum mismatch'):training.restore_checkpoint(corrupt,destination,'same-recipe')
    assert not destination.exists()


def test_csv_never_repeats_uncommitted_epoch(tmp_path):
    folder=tmp_path/'training';folder.mkdir()
    path=folder/'results.csv';path.write_text('epoch,fitness\n1,.5\n2,.6\n')
    training.trim_results(tmp_path,1)
    assert [r['epoch'] for r in csv.DictReader(path.open())]==['1']


def test_controller_pause_and_resume_without_duplicate_workers(tmp_path):
    import threading
    entered=threading.Event();release=threading.Event();calls=[]
    def runner(data,run,**kwargs):
        calls.append(kwargs['resume_only']);entered.set();release.wait(5)
        assert kwargs['control'].pause_requested()
        return {'status':'paused'}
    session=TrainingSession(tmp_path/'data',tmp_path/'run',runner=runner)
    session.start();assert entered.wait(5)
    with pytest.raises(RuntimeError,match='already running'):session.start()
    session.pause();release.set();session.thread.join(5)
    assert session.status=='paused' and not session.running and calls==[False]
    with pytest.raises(RuntimeError,match='no checkpoint'):session.start(resume=True)


def test_resume_without_checkpoint_never_starts_training(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    (data/'six_class_manifest.json').write_text('{}')
    monkeypatch.setattr(training,'configure',lambda _:({'names':list(range(6))},{}))
    with pytest.raises(RuntimeError,match='Resume refused'):
        training.main(data,tmp_path/'run',device='cpu',resume_only=True)
    assert not (tmp_path/'run/training').exists()


def test_notebook_uses_embedded_update_without_reuploading_data():
    from accuracy.build_notebook import build
    notebook=build()
    text='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'EMBEDDED_CODE' in text and 'controls = show_controls' in text
    assert 'KAGGLE_API_TOKEN' not in text
    for cell in notebook['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'cell','exec')


def test_widget_buttons_trigger_pause_and_resume(tmp_path,monkeypatch):
    import threading
    import IPython.display
    pytest.importorskip('ipywidgets')
    from accuracy.notebook_controls import show_controls
    displayed=[]
    monkeypatch.setattr(IPython.display,'display',displayed.append)
    session=show_controls(tmp_path/'data',tmp_path/'run',device='cpu')
    entered=threading.Event();release=threading.Event();calls=[]
    def runner(data,run,**kwargs):
        calls.append(kwargs['resume_only'])
        if kwargs['resume_only']:return {'status':'complete'}
        entered.set();release.wait(5)
        assert session.pause_requested()
        session.archive.write_bytes(b'downloadable test checkpoint')
        session.checkpoint_saved(1,session.archive)
        return {'status':'paused'}
    session.runner=runner
    start,pause,resume=displayed[0].children[0].children
    assert not start.disabled and pause.disabled and resume.disabled
    start.click();assert entered.wait(5)
    assert start.disabled and not pause.disabled and resume.disabled
    pause.click();assert session.pause_requested()
    release.set();session.thread.join(5)
    assert session.status=='paused' and not resume.disabled and pause.disabled
    assert 'six_class_recovery_checkpoint.zip' in displayed[0].children[2].value
    resume.click();session.thread.join(5)
    assert session.status=='complete' and calls==[False,True]
