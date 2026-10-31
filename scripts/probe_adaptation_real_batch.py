"""One real-image CPU optimization step; a plumbing check, not accuracy evidence."""
from _bootstrap import ROOT
import os
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'data/runtime/ultralytics'))
import json
import torch
import cv2
from models.adaptation_training import AdaptationSystem


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    package=ROOT/'artifacts/kaggle/phase2_v2/package'
    manifest=json.loads((package/'manifest.json').read_text())
    membership=set(json.loads((ROOT/'artifacts/models/k500/selection.json').read_text())['retained_k100_image_sha256'])
    source=next(r for r in manifest['views'] if r['dataset']=='source' and r['split']=='train' and r['kind']=='full' and any(int(b[0])==1 for b in r['labels']))
    target=next(r for r in manifest['views'] if r['dataset']=='target' and r['split']=='train' and r['image_sha256'] in membership and any(int(b[0])==1 for b in r['labels']))
    images=[];indices=[];classes=[];boxes=[]
    for i,row in enumerate([source,target]):
        im=cv2.imread(str(package/row['image']));assert im is not None
        im=cv2.resize(im,(128,128))[:,:,::-1].copy()
        images.append(torch.from_numpy(im).permute(2,0,1).float()/255)
        for cid,x,y,w,h in row['labels']:
            indices.append(i);classes.append([cid]);boxes.append([x,y,w,h])
    batch=dict(img=torch.stack(images),batch_idx=torch.tensor(indices),cls=torch.tensor(classes,dtype=torch.float32),bboxes=torch.tensor(boxes,dtype=torch.float32))
    system=AdaptationSystem(package/'weights/source_baseline.pt',mode='weighted',class_weights={1:1,2:1,3:1,4:1,5:1})
    optimizer=torch.optim.AdamW([p for p in system.parameters() if p.requires_grad],lr=.0001)
    loss,details=system.training_loss(batch,1);assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    assert all(torch.isfinite(p).all() for p in system.parameters())
    report=dict(scope='One CPU real-image/label optimization step; not GPU training or accuracy evidence',
                geometry='128x128 direct square resize for plumbing only; normalized labels preserved',
                source=source['image'],target=target['image'],source_boxes=len(source['labels']),target_boxes=len(target['labels']),
                total_loss=float(loss.detach()),loss_details=details,
                trainable_parameters=sum(p.numel() for p in system.parameters() if p.requires_grad),
                adapter_parameters=sum(p.numel() for p in system.adapters.parameters()))
    path=ROOT/'reports/revamp/adaptation_real_batch_probe.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
