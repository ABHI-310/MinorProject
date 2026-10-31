"""Detection-plus-region-alignment integration, independent of Ultralytics Trainer.

Checkpoint state dictionaries require reconstruction from the original source
checkpoint and the same configuration. They are not ordinary YOLO .pt exports.
"""
from pathlib import Path
import hashlib
import os
import random
import numpy as np
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from models.defect_regions import IdentityResidualAdapter, pool_defect_regions
from models.mmd_loss import DefectAwareMMDLoss


class AdaptationSystem(nn.Module):
    def __init__(self, source, mode='weighted', coefficient=.1, class_weights=None):
        super().__init__()
        if mode not in ('full','head','adapter','mmd','weighted'):
            raise ValueError('Unknown experiment mode')
        if coefficient < 0 or not np.isfinite(coefficient):
            raise ValueError('Alignment coefficient must be finite and nonnegative')
        self.source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()
        self.config=dict(mode=mode,coefficient=coefficient,class_weights=dict(class_weights or {}))
        self.detector=YOLO(str(source)).model.float()
        self.detector.args=get_cfg()
        self.adapters=nn.ModuleDict()
        self._features={}
        self._capture=False
        self._handles=[]
        if mode in ('adapter','mmd','weighted'):
            for index,channels in [(15,128),(18,256),(21,512)]:
                key=str(index)
                self.adapters[key]=IdentityResidualAdapter(channels,32)
                self._handles.append(self.detector.model[index].register_forward_hook(self._hook(key)))
        self.alignment=DefectAwareMMDLoss(class_weights=class_weights)
        for name,parameter in self.detector.named_parameters():
            parameter.requires_grad_(mode=='full' or name.startswith('model.22.'))
        # Fixed DFL projection remains frozen, as in standard YOLO training.
        for name,parameter in self.detector.named_parameters():
            if '.dfl.' in name:parameter.requires_grad_(False)
        self.train()

    def _hook(self,key):
        def apply(module,args,output):
            adapted=self.adapters[key](output)
            if self._capture:self._features[key]=adapted
            return adapted
        return apply

    def train(self,mode=True):
        super().train(mode)
        if mode and self.config['mode']!='full':
            for layer in list(self.detector.model)[:-1]:layer.eval()
        return self

    def forward(self,images):
        return self.detector(images)

    def training_loss(self,batch,source_images):
        """Combined batch: source images first, target images second.

        bboxes are normalized xywh AFTER detector transforms; cls and batch_idx
        correspond to these boxes. All arms receive the same combined batch.
        """
        if not 0 < source_images < len(batch['img']):
            raise ValueError('Batch must contain both source and target images')
        self._capture=True;self._features={}
        try:
            detection,items=self.detector.loss(batch)
            # Normalize summed detection loss by combined batch size.
            detection=detection.sum()/len(batch['img'])
            alignment=detection*0
            common=[]
            if self.config['mode'] in ('mmd','weighted'):
                boxes=batch['bboxes'];center,size=boxes[:,:2],boxes[:,2:]
                xyxy=torch.cat((center-size/2,center+size/2),dim=1).clamp(0,1)
                indices=batch['batch_idx'].long().reshape(-1)
                labels=batch['cls'].long().reshape(-1)
                source=(indices<source_images)&(labels>=1)&(labels<=5)
                target=(indices>=source_images)&(labels>=1)&(labels<=5)
                common=sorted(set(labels[source].tolist())&set(labels[target].tolist()))
                losses=[]
                for feature in self._features.values():
                    vectors=pool_defect_regions(feature,xyxy,indices)
                    if self.config['mode']=='weighted':
                        loss=self.alignment(vectors[source],vectors[target],labels[source],labels[target])
                    else:
                        loss=self.alignment(vectors[source],vectors[target])
                    losses.append(loss)
                if losses:alignment=torch.stack(losses).mean()
            total=detection+self.config['coefficient']*alignment
            return total,dict(detection=float(detection.detach()),alignment=float(alignment.detach()),
                              shared_classes=common,alignment_active=bool(common))
        finally:
            self._capture=False;self._features={}


def save_state(path,system,optimizer,step,experiment):
    """Atomic, lossless local checkpoint for exact same-code reconstruction."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(config=system.config,source_sha256=system.source_sha256,
                 model=system.state_dict(),optimizer=optimizer.state_dict(),step=step,experiment=experiment,
                 python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)


def restore_state(path,system,optimizer,experiment):
    """Load only trusted checkpoints produced by this code; rejects changed setup."""
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['source_sha256']!=system.source_sha256 or state['config']!=system.config or state['experiment']!=experiment:
        raise ValueError('Checkpoint experiment/config/source mismatch')
    system.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['step']
