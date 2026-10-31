import json
from types import SimpleNamespace
import pytest
from revamp import pilot
from revamp.build_pilot_notebook import build


def test_pilot_notebook_embeds_matching_runner_and_compiles(tmp_path):
    package=tmp_path/'package';package.mkdir()
    (package/'package_hashes.json').write_text('{}')
    nb=build(package,tmp_path/'pilot.ipynb')
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'pilot cell','exec')
    setup=''.join(nb['cells'][2]['source']);run=''.join(nb['cells'][4]['source'])
    assert "'pilot.py'" in setup and 'pcb_pilot_tools' in setup
    assert 'revamp.pilot' in run and 'killpg' in run and '--resume-only' in run
    assert "DOMAIN = 'source'" in run and 'pilot_run_checkpoint.zip' in run


def test_optional_augmentations_fail_closed():
    Albumentations=type('Albumentations',(),{})
    alb=Albumentations();alb.transform=SimpleNamespace(transforms=['unexpected blur'])
    dataset=SimpleNamespace(transforms=SimpleNamespace(transforms=[alb]))
    with pytest.raises(RuntimeError,match='Unexpected'):pilot.assert_no_albumentations(dataset)
    alb.transform.transforms=[]
    pilot.assert_no_albumentations(dataset)


def test_absent_target_class_is_not_reported_as_measured():
    box=SimpleNamespace(map50=.5,map=.2,mp=.4,mr=.6,ap_class_index=[1,5],
                        ap50=[.4,.6],ap=[.1,.3],p=[.3,.5],r=[.5,.7])
    record=pilot.metrics_record(SimpleNamespace(box=box,speed={'inference':2.}))
    assert set(record['per_class'])=={'mouse_bite','spurious_copper'}


@pytest.fixture
def coordinator(tmp_path,monkeypatch):
    import ultralytics
    monkeypatch.setattr(ultralytics,'__version__','8.4.142')
    monkeypatch.setattr(pilot,'verify_package',lambda _:None)
    monkeypatch.setattr(pilot,'identity',lambda *a:('same',{}))
    monkeypatch.setattr(pilot,'train',lambda *a:pytest.fail('Unexpected training'))
    return tmp_path/'package',tmp_path/'run'


def test_resume_only_and_import_conflict_fail_before_training(coordinator):
    package,output=coordinator
    with pytest.raises(RuntimeError,match='no checkpoint'):
        pilot.run(package,output,device='cpu',resume_only=True)
    (output/'pilot_run_checkpoint.zip').write_bytes(b'local')
    with pytest.raises(ValueError,match='Local progress'):
        pilot.run(package,output,device='cpu',checkpoint=output/'imported.zip')


def test_changed_recipe_or_environment_rejected(coordinator):
    package,output=coordinator
    (output/'pilot_run').mkdir(parents=True)
    (output/'pilot_run/signature.json').write_text(json.dumps({'sha256':'old'}))
    with pytest.raises(ValueError,match='Different package'):
        pilot.run(package,output,device='cpu')


def test_concurrent_pilot_refused(coordinator):
    import fcntl
    package,output=coordinator;output.mkdir()
    with (output/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='already active'):pilot.run(package,output,device='cpu')
