import pytest
import torch
from models.defect_regions import IdentityResidualAdapter, pool_defect_regions
from models.mmd_loss import DefectAwareMMDLoss


def test_identity_adapter_preserves_features_and_receives_gradient():
    x=torch.randn(2,8,6,6)
    adapter=IdentityResidualAdapter(8,4)
    y=adapter(x)
    assert torch.equal(x,y)
    y.square().mean().backward()
    assert adapter.up.weight.grad.abs().sum()>0


def test_region_pool_selects_correct_image_and_spatial_region():
    x=torch.zeros(2,2,8,8)
    x[0,0,:,:4]=1
    x[0,1,:,4:]=1
    x[1,1]=1
    boxes=torch.tensor([[0.,0.,.5,1.],[.5,0.,1.,1.],[0.,0.,1.,1.]])
    got=pool_defect_regions(x,boxes,torch.tensor([0,0,1]))
    assert torch.allclose(got,torch.tensor([[1.,0.],[0.,1.],[0.,1.]]),atol=1e-6)


def test_class_alignment_backpropagates_through_adapter():
    adapter=IdentityResidualAdapter(8,4)
    boxes=torch.tensor([[.1,.1,.9,.9],[.2,.2,.8,.8]])
    indices=torch.tensor([0,1])
    source=pool_defect_regions(adapter(torch.randn(2,8,8,8)),boxes,indices)
    target=pool_defect_regions(adapter(torch.randn(2,8,8,8)+1),boxes,indices)
    loss=DefectAwareMMDLoss(class_weights={1:1,2:2})(source,target,torch.tensor([1,2]),torch.tensor([1,2]))
    assert torch.isfinite(loss)
    loss.backward()
    assert adapter.up.weight.grad.abs().sum()>0


def test_empty_regions_and_invalid_geometry():
    x=torch.randn(1,4,5,5)
    assert pool_defect_regions(x,torch.empty(0,4),torch.empty(0,dtype=torch.long)).shape==(0,4)
    with pytest.raises(ValueError):pool_defect_regions(x,torch.tensor([[.9,0.,.1,1.]]),torch.tensor([0]))
    with pytest.raises(ValueError):pool_defect_regions(x,torch.tensor([[0.,0.,1.,1.]]),torch.tensor([1]))
