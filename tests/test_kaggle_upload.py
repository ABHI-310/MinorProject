import json
from pathlib import Path
import numpy as np
import pytest
from accuracy.prepare_kaggle_upload import prepare
from accuracy.kaggle_setup import reconstruct,discover,setup


def test_split_upload_roundtrip_and_completed_reuse(tmp_path):
    archive=tmp_path/'source.zip';archive.write_bytes(bytes(range(256))*7)
    folder=tmp_path/'upload';manifest=prepare(archive,folder,chunk_bytes=512)
    assert len(manifest['parts'])==4
    for part in manifest['parts']:
        a=np.load(folder/part['file'],allow_pickle=False)
        assert a.dtype==np.uint8 and a.ndim==1
    output=tmp_path/'restored.zip'
    reconstruct(folder/'pcb_upload_manifest.json',output)
    assert output.read_bytes()==archive.read_bytes()
    stamp=output.stat().st_mtime_ns
    reconstruct(folder/'pcb_upload_manifest.json',output)
    assert output.stat().st_mtime_ns==stamp


def test_split_upload_detects_corrupt_or_missing_part(tmp_path):
    archive=tmp_path/'source.zip';archive.write_bytes(bytes(range(256))*3)
    folder=tmp_path/'upload';manifest=prepare(archive,folder,chunk_bytes=256)
    part=folder/manifest['parts'][0]['file'];a=np.load(part,mmap_mode='r+');a[0]^=255;a.flush();del a
    with pytest.raises(AssertionError,match='integrity check'):
        reconstruct(folder/'pcb_upload_manifest.json',tmp_path/'restored.zip')
    assert not (tmp_path/'restored.tmp').exists()
    part.unlink()
    with pytest.raises(AssertionError,match='Missing upload part'):
        reconstruct(folder/'pcb_upload_manifest.json',tmp_path/'restored.zip')


def test_setup_restores_dataset_from_nested_kaggle_input(tmp_path):
    import zipfile,hashlib
    archive=tmp_path/'package.zip';payload=b'known dataset file'
    manifest={'package_id':'pcb_six_class_recovery_v1','sha256':{'example.txt':hashlib.sha256(payload).hexdigest()},'splits':{'train':{'images':1}}}
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('six_class_manifest.json',json.dumps(manifest));z.writestr('example.txt',payload)
    upload=tmp_path/'input/datasets/user/recovery'
    prepare(archive,upload,chunk_bytes=128)
    assert discover(tmp_path/'input')[3]==[upload/'pcb_upload_manifest.json']
    data,run=setup(tmp_path/'input',tmp_path/'working')
    assert (data/'example.txt').read_bytes()==payload
    assert run.name=='six_class_recovery_run'


def test_setup_restores_kaggle_expanded_checkpoint_dataset(tmp_path):
    import hashlib
    inputs=tmp_path/'input'
    data=inputs/'datasets/user/data_set';data.mkdir(parents=True)
    payload=b'original dataset';(data/'example.txt').write_bytes(payload)
    manifest={'package_id':'pcb_six_class_recovery_v1','sha256':{'example.txt':hashlib.sha256(payload).hexdigest()},'splits':{}}
    (data/'six_class_manifest.json').write_text(json.dumps(manifest))
    checkpoint=inputs/'datasets/user/checkpoint'
    files={'signature.json':b'{"sha256":"known"}', 'training/weights/last.pt':b'last',
           'training/weights/best.pt':b'best','training/args.yaml':b'epochs: 80','training/results.csv':b'epoch\n1\n'}
    for name,content in files.items():
        p=checkpoint/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
    (checkpoint/'checkpoint_manifest.json').write_text(json.dumps({'sha256':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}}))
    _,run=setup(inputs,tmp_path/'working')
    assert (run/'training/weights/last.pt').read_bytes()==b'last'
