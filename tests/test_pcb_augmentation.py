import pytest,torch
from scripts.revamp.pcb_augmentation import transform_pcb,augment_pcb_batch

@pytest.mark.parametrize('k',range(4))
@pytest.mark.parametrize('lr',[False,True])
@pytest.mark.parametrize('ud',[False,True])
def test_boxes_match_transformed_pixels(k,lr,ud):
 image=torch.zeros(3,10,10);image[:,1:6,2:5]=1
 labels=torch.tensor([[0,3,.35,.35,.3,.5]])
 out,t=transform_pcb(image,labels,k,lr,ud)
 y,x=torch.where(out[0]>.5)
 expected=torch.tensor([((x.min()+x.max()+1)/2)/10,((y.min()+y.max()+1)/2)/10,(x.max()-x.min()+1)/10,(y.max()-y.min()+1)/10])
 assert torch.allclose(t[0,2:],expected,atol=1e-6)
 assert torch.equal(t[0,:2],labels[0,:2])

def test_batch_is_finite_and_does_not_modify_originals():
 x=torch.rand(4,3,32,32);t=torch.tensor([[i,1,.5,.5,.2,.1] for i in range(4)]);original_x=x.clone();original_t=t.clone()
 a,b=augment_pcb_batch(x,t)
 assert torch.equal(x,original_x) and torch.equal(t,original_t)
 assert torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
 assert torch.equal(b[:,:2],t[:,:2])
