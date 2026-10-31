from pathlib import Path
import os
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[1]/'data/runtime/ultralytics'))
import pytest
import torch
from models.adaptation_training import AdaptationSystem, save_state, restore_state

SOURCE=Path(__file__).resolve().parents[1]/'artifacts/kaggle/phase2_v2/package/weights/source_baseline.pt'


def batch():
    return dict(img=torch.rand(2,3,128,128),batch_idx=torch.tensor([0,1]),
                cls=torch.tensor([[1.],[1.]]),bboxes=torch.tensor([[.5,.5,.3,.3],[.45,.45,.25,.25]]))


@pytest.mark.skipif(not SOURCE.exists(),reason='Source weights not installed')
def test_detection_alignment_save_and_resume(tmp_path):
    torch.set_num_threads(4);torch.manual_seed(42)
    system=AdaptationSystem(SOURCE,mode='weighted')
    optimizer=torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],lr=.0001)
    data=batch()
    loss,log=system.training_loss(data,1)
    assert torch.isfinite(loss) and log['alignment']>0 and log['shared_classes']==[1]
    assert not system._features
    loss.backward()
    assert all(a.up.weight.grad.abs().sum()>0 for a in system.adapters.values())
    assert all(p.grad is None for n,p in system.detector.named_parameters() if not n.startswith('model.22.'))
    optimizer.step();optimizer.zero_grad()
    path=tmp_path/'resume.pt';save_state(path,system,optimizer,1,{'selection':'fixed'})
    restored=AdaptationSystem(SOURCE,mode='weighted')
    other=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.0001)
    assert restore_state(path,restored,other,{'selection':'fixed'})==1
    for a,b in zip(system.state_dict().values(),restored.state_dict().values()):assert torch.equal(a,b)
    # Same next optimization update after restoring optimizer and model state.
    for model,opt in [(system,optimizer),(restored,other)]:
        next_loss,_=model.training_loss(data,1);next_loss.backward();opt.step();opt.zero_grad()
    for a,b in zip(system.state_dict().values(),restored.state_dict().values()):assert torch.equal(a,b)
    with pytest.raises(ValueError,match='mismatch'):restore_state(path,restored,other,{'selection':'different'})


@pytest.mark.skipif(not SOURCE.exists(),reason='Source weights not installed')
@pytest.mark.parametrize('mode',['full','head','adapter','mmd'])
def test_control_losses_are_finite(mode):
    torch.set_num_threads(4)
    system=AdaptationSystem(SOURCE,mode=mode)
    loss,log=system.training_loss(batch(),1)
    assert torch.isfinite(loss)
    if mode in ('full','head','adapter'):assert log['alignment']==0
    loss.backward()
