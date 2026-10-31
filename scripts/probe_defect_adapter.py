"""CPU architecture/gradient probe; random inputs are never accuracy evidence."""
from _bootstrap import ROOT
import json
import os
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
import torch
from ultralytics import YOLO
from models.defect_regions import IdentityResidualAdapter, pool_defect_regions
from models.mmd_loss import DefectAwareMMDLoss


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    model=YOLO(str(ROOT/'artifacts/kaggle/phase2_v2/package/weights/source_baseline.pt')).model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    x=torch.randn(2,3,128,128)
    with torch.no_grad():baseline=model(x)[0].clone()
    adapters={};captured={};handles=[]
    def install(index):
        layer=model.model[index]
        def hook(module,args,output):
            if index not in adapters:adapters[index]=IdentityResidualAdapter(output.shape[1],32)
            result=adapters[index](output);captured[index]=result
            return result
        handles.append(layer.register_forward_hook(hook))
    for index in [15,18,21]:install(index)
    adapted=model(x)[0]
    assert torch.equal(adapted.detach(),baseline)
    losses=[]
    for feature in captured.values():
        vectors=pool_defect_regions(feature,torch.tensor([[.2,.2,.8,.8],[.1,.1,.9,.9]]),torch.tensor([0,1]))
        losses.append(DefectAwareMMDLoss()(vectors[:1],vectors[1:],torch.tensor([1]),torch.tensor([1])))
    loss=torch.stack(losses).mean();assert torch.isfinite(loss);loss.backward()
    gradients={str(i):float(adapter.up.weight.grad.abs().sum()) for i,adapter in adapters.items()}
    assert all(value>0 for value in gradients.values())
    assert all(parameter.grad is None for parameter in model.parameters())
    record={'scope':'Random-input CPU architecture and gradient check only; no training or accuracy claim',
            'source_predictions_preserved_at_initialization':True,'backbone_gradients_absent':True,
            'adapter_gradient_sums':gradients,'feature_shapes':{str(i):list(f.shape) for i,f in captured.items()},
            'adapter_parameters':sum(p.numel() for a in adapters.values() for p in a.parameters()),'alignment_loss':float(loss.detach())}
    out=ROOT/'reports/revamp/defect_adapter_probe.json';out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
    for handle in handles:handle.remove()

if __name__=='__main__':main()
