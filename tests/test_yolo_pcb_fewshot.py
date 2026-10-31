"""Regression checks for numerical failure handling and true image budgets."""
import pytest
import torch
from scripts.revamp.train_yolo_pcb_mixed import checked_step, select_target_budget

def samples():
 return [dict(split='train',dataset='target',image_sha256=str(i),labels=[[(i%5)+1,.5,.5,.1,.1]]) for i in range(120)]

def test_nested_unique_training_image_budgets():
 rows=samples();rows.append(rows[0].copy())
 a=select_target_budget(rows,50);b=select_target_budget(rows,100)
 assert len(a)==len({r['image_sha256'] for r in a})==50
 assert len(b)==100 and a==b[:50]
 assert {int(x[0]) for r in a for x in r['labels']}=={1,2,3,4,5}

def test_validation_cannot_enter_target_selection():
 rows=samples();rows[0]['split']='val'
 with pytest.raises(ValueError,match='training only'):select_target_budget(rows,50)

def test_finite_loss_with_infinite_gradient_never_updates_optimizer():
 model=torch.nn.Linear(1,1,bias=False)
 with torch.no_grad():model.weight.zero_()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
 loss=model.weight.sqrt().sum()  # value=0, derivative=inf
 assert torch.isfinite(loss)
 with pytest.raises(RuntimeError,match='Nonfinite FP32 gradients'):checked_step(model,optimizer,loss,'test batch')
 assert model.weight.item()==0 and len(optimizer.state)==0

def test_finite_gradient_updates_weights():
 model=torch.nn.Linear(1,1,bias=False)
 with torch.no_grad():model.weight.fill_(2.)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
 checked_step(model,optimizer,model.weight.square().sum(),'finite batch')
 assert model.weight.item()<2 and torch.isfinite(model.weight).all()
