# New controlled pilot: SAME 100 target images, add training-only augmentation.
from pathlib import Path
import hashlib, json, subprocess, os, signal, sys, time
from IPython.display import display, FileLink
WORK = Path('/kaggle/working/pcb_yolo_fewshot_direct_v1')
BASE_RUN = WORK/'training_k100_fp32'
BASE_SCRIPT = WORK/'train_yolo_pcb_fewshot_k100_fp32.py'
BUNDLE, PACKAGE = Path(BUNDLE), Path(PACKAGE)
signature = json.loads((BASE_RUN/'signature.json').read_text())
assert signature['recipe']['target_budget'] == 100
raw = BASE_SCRIPT.read_bytes()
assert hashlib.sha256(raw).hexdigest() == signature['recipe']['runner_sha256']
source = raw.decode('utf-8')
augmentation_code = '''"""Mild PCB augmentation with exact normalized YOLO-box transforms."""
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
'''
changes = [
    ('def trusted_model(path):', augmentation_code+'\ndef trusted_model(path):'),
    ('x,t=batch_input(batch);opt.zero_grad(set_to_none=True)',
     'x,t=batch_input(batch);x,t=augment_pcb_batch(x,t);opt.zero_grad(set_to_none=True)'),
    ("augmentation='none in first controlled pilot'",
     "augmentation='right_angle_rotations_flips_brightness_0.8_1.2_contrast_0.9_1.1'"),
]
for old, new in changes:
    assert source.count(old) == 1, 'Unexpected runner contents; patch stopped.'
    source = source.replace(old, new, 1)
compile(source, 'k100_augmented_runner', 'exec')
runner = WORK/'train_yolo_pcb_k100_aug_fp32.py'
runner.write_text(source)
NEW_RUN = WORK/'training_k100_aug_fp32'
STOP_AFTER_EPOCH = 5
cmd_aug = [sys.executable, str(runner), '--package', str(PACKAGE),
           '--bundle', str(BUNDLE), '--run', str(NEW_RUN), '--device', 'cuda',
           '--epochs', '40', '--batch', '4', '--imgsz', '608', '--patience', '8',
           '--stop-after-epoch', str(STOP_AFTER_EPOCH)]
env_aug = os.environ.copy()
env_aug['MPLCONFIGDIR'] = str(WORK/'matplotlib')
log_aug = WORK/'training_k100_aug.log'
with log_aug.open('a', buffering=1) as stream:
    process = subprocess.Popen(cmd_aug, stdout=stream, stderr=subprocess.STDOUT,
                               env=env_aug, start_new_session=True)
    print('Same 100 target images, FP32, training-only augmentation.')
    print('New experiment uses original YOLO-PCB initialization; existing results are preserved.')
    previous = None
    try:
        while process.poll() is None:
            time.sleep(10)
            lines = log_aug.read_text(errors='replace').splitlines()
            latest = lines[-1] if lines else ''
            if latest and latest != previous:
                print(latest[:2000], flush=True)
                previous = latest
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        print('Interrupted. Rerun this cell to resume its last completed epoch.')
    else:
        if process.returncode:
            print(log_aug.read_text(errors='replace')[-12000:])
            raise RuntimeError('Augmented run failed; inspect log.')
if (NEW_RUN/'history.json').exists():
    for row in json.loads((NEW_RUN/'history.json').read_text()):
        print('Epoch', row['epoch'],
              'source AP50', round(100*row['metrics']['source']['map50'], 2),
              'target AP50', round(100*row['metrics']['target']['map50'], 2))
for name in ['resume.zip','history.json','initial_validation.json','signature.json']:
    if (NEW_RUN/name).exists(): display(FileLink(str(NEW_RUN/name)))
if log_aug.exists(): display(FileLink(str(log_aug)))
