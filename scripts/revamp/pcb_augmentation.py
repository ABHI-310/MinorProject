"""Mild PCB augmentation with exact normalized YOLO-box transforms."""
import torch

def transform_pcb(image,labels,k=0,flip_lr=False,flip_ud=False,brightness=1.,contrast=1.):
 labels=labels.clone();image=torch.rot90(image,k%4,dims=(1,2))
 for _ in range(k%4):
  x,y,w,h=[labels[:,i].clone() for i in [2,3,4,5]]
  labels[:,2]=y;labels[:,3]=1-x;labels[:,4]=h;labels[:,5]=w
 if flip_lr:image=torch.flip(image,[2]);labels[:,2]=1-labels[:,2]
 if flip_ud:image=torch.flip(image,[1]);labels[:,3]=1-labels[:,3]
 image=((image-.5)*contrast+.5)*brightness
 return image.clamp(0,1).contiguous(),labels

def augment_pcb_batch(images,targets):
 images=images.clone();targets=targets.clone()
 for i in range(len(images)):
  mask=targets[:,0]==i
  image,labels=transform_pcb(images[i],targets[mask],k=int(torch.randint(4,()).item()),flip_lr=bool(torch.rand(()).item()<.5),flip_ud=bool(torch.rand(()).item()<.5),brightness=.8+.4*torch.rand(()).item(),contrast=.9+.2*torch.rand(()).item())
  images[i]=image;targets[mask]=labels
 return images,targets
