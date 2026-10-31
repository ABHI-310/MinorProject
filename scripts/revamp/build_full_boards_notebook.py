"""Deliver the controlled full-board-only source experiment."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from revamp.build_source_notebook import build as source_notebook


def build(package,output):
    with tempfile.TemporaryDirectory() as temporary:
        nb=source_notebook(package,Path(temporary)/'source.ipynb')
    setup=''.join(nb['cells'][2]['source'])
    node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='EMBEDDED_TOOLS' for t in n.targets))
    payload=ast.literal_eval(node.value);payload.pop('train_source.py')
    raw=Path(__file__).with_name('train_full_boards.py').read_bytes()
    payload['train_full_boards.py']=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    start=setup.index('EMBEDDED_TOOLS = ');end=setup.index('\nimport base64, importlib',start)
    setup=setup[:start]+'EMBEDDED_TOOLS = '+repr(payload)+'\n'+setup[end:]
    nb['cells'][2]['source']=[setup.replace('pcb_source_training_tools','pcb_full_boards_tools')]
    run=''.join(nb['cells'][4]['source'])
    for old,new in [('revamp.train_source','revamp.train_full_boards'),
                    ('source_run_checkpoint.zip','full_boards_run_checkpoint.zip'),
                    ('pcb_phase3_{DOMAIN}_train_v1','pcb_phase3_{DOMAIN}_full_boards_v2')]:
        run=run.replace(old,new)
    nb['cells'][4]['source']=run.splitlines(True)
    nb['cells'][0]['source']=['''# PCB source experiment 2 — full boards only

Use the same version-2 dataset, T4 GPU, Internet and a fresh kernel. No new dataset upload or previous checkpoint is required.

**One experimental change:** use the 480 original training boards instead of 953 full/crop/background views. The same 120 original validation boards are retained. YOLOv8s COCO initialization, seed, 960px, batch 4, full precision, disabled augmentations, AdamW, learning rate, 60-epoch maximum, 3 warmup epochs and patience 15 stay unchanged. Fewer images mean fewer updates per epoch; this is a practical training-view comparison, not an equal-compute causal study.

Experiment 1 overfit after epoch 9. Its median defect short edge at 960px was 81px in crops versus 18px on full training boards and 24px on validation boards. Crop scale/context mismatch is a hypothesis, not a proven explanation. This run tests whether removing derived views improves full-board validation. No improvement is guaranteed.

Budget approximately **1.5–2 hours** if all 60 epochs run, estimated from half as many training batches; early stopping may finish sooner. Runtime must be measured. The existing baseline remains deployed. Checkpoints are saved every completed epoch; download before leaving Kaggle. Only this experiment's checkpoint can resume it.
''']
    nb['cells'][5]['source']=['''## Download immediately after completion

Download **full_boards_run_checkpoint.zip**, **report.json**, **training.log** and **pcb_environment.json** into `downloads/source_full_boards_v2/`. During interruption the checkpoint is the priority. Session-local checkpoints disappear if Kaggle removes session storage.

Same session: RESTORE_FROM=None and rerun. New session: attach the original dataset plus this experiment's trusted checkpoint and set RESTORE_FROM; clear it on later local reruns. Finished and early-stopped checkpoints rebuild missing reports without retraining. The output paths and signatures differ from experiment 1, so its checkpoint is rejected.

Compare against experiment 1 (82.18% mAP50 / 37.09% mAP50–95) and the re-evaluated retained baseline (previously 90.79% / 47.04%). The existing review gate remains unchanged. No model is promoted automatically. If this does not improve generalization, inspect validation predictions before adding augmentation or another model; do not launch an automatic sweep or use development-test results for tuning.
''']
    for i,c in enumerate(nb['cells']):
        c['id']=f'full-boards-{i}'
        if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(nb,stream,indent=2)
    return nb

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.package,args.output)
