import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from backend.model_registry import ModelRegistry


@pytest.mark.parametrize('candidate', ['k500','k500_coco'])
def test_candidate_metadata_does_not_claim_test_scores(candidate):
    registry = ModelRegistry()
    record = next(r for r in registry.list_models() if r['model_type'] == candidate)
    assert 'validation' in record['evaluation']
    assert 'not a final test' in record['evaluation']
    assert registry.canonical('adapted') == 'target'
    assert registry.canonical(candidate) == candidate


@pytest.mark.parametrize('candidate', ['k500','k500_coco'])
def test_candidate_inference_settings_and_hash_guard(tmp_path, monkeypatch, candidate):
    import ultralytics
    folder = tmp_path / 'k500'
    folder.mkdir()
    (folder / 'model.pt').write_bytes(b'fixture')
    info = dict(checkpoint_sha256=hashlib.sha256(b'fixture').hexdigest(),
                classes=['a'], image_size=640, evaluated_class_ids=[1,2,3,4,5],
                inference=dict(iou=.4, rect=False, max_det=300, agnostic_nms=False, augment=False))
    (folder / 'model_info.json').write_text(json.dumps(info))
    calls = []
    class FakeModel:
        names = {0:'a'}
        def __init__(self, path): pass
        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[])]
    monkeypatch.setattr(ultralytics, 'YOLO', FakeModel)
    registry = ModelRegistry()
    registry.folders[candidate] = folder
    registry.infer(np.zeros((32,48,3), dtype=np.uint8), candidate, .25, False)
    assert calls[0]['conf'] == .25 and calls[0]['imgsz'] == 640
    assert calls[0]['classes'] == [1,2,3,4,5]
    for key, value in info['inference'].items(): assert calls[0][key] == value
    (folder / 'model.pt').write_bytes(b'tampered')
    registry.loaded.clear()
    with pytest.raises(ValueError, match='checksum'):
        registry.infer(np.zeros((32,48,3), dtype=np.uint8), candidate, .25, False)
