"""Deliver the source-training experiment without changing the verified pilot."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from revamp.build_pilot_notebook import build as pilot_notebook


def build(package,output):
    with tempfile.TemporaryDirectory() as temporary:
        nb=pilot_notebook(package,Path(temporary)/'pilot.ipynb')
    nb['cells'][0]['source']=['''# PCB Phase 3 — full source training, experiment 1

Attach the existing version-2 dataset. Enable **T4 GPU + Internet** and use a fresh kernel. This notebook trains YOLOv8s from packaged COCO weights, not the completed pilot or old PCB baseline.

**Up to 60 epochs**, early stopping after 15 epochs without improvement, 3 warmup epochs, 960px, batch 4, full precision. Train: 953 source views; validation: 120 separate full boards. Augmentations stay explicitly disabled for this first controlled full run. The pilot measured about 2.7 minutes per epoch including intervening checkpoint overhead: budget roughly 3 hours if all epochs run, subject to Kaggle performance.

The best checkpoint is selected by validation mAP50–95. The old source baseline is evaluated under the same settings. A better result is not guaranteed and the app model is never replaced automatically.

Every completed epoch saves a complete recovery ZIP. Download it before ending a session. Resume requires the same package, code, recipe and key environment versions. Pilot/smoke checkpoints are incompatible. Finished and early-stopped runs recover evaluation without training again.
''']
    setup=''.join(nb['cells'][2]['source'])
    node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='EMBEDDED_TOOLS' for t in n.targets))
    payload=ast.literal_eval(node.value)
    payload.pop('pilot.py')
    raw=Path(__file__).with_name('train_source.py').read_bytes()
    payload['train_source.py']=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    start=setup.index('EMBEDDED_TOOLS = ');end=setup.index('\nimport base64, importlib',start)
    setup=setup[:start]+'EMBEDDED_TOOLS = '+repr(payload)+'\n'+setup[end:]
    setup=setup.replace('pcb_pilot_tools','pcb_source_training_tools')
    nb['cells'][2]['source']=setup.splitlines(True)
    nb['cells'][3]['source']=[''.join(nb['cells'][3]['source']).replace('this pilot','source training')]
    run=''.join(nb['cells'][4]['source'])
    for old,new in [('Run source first. Review its report before scheduling the target pilot.',
                     'Run the source experiment only; download results for review.'),
                    ('revamp.pilot','revamp.train_source'),('pilot_run_checkpoint.zip','source_run_checkpoint.zip'),
                    ('pcb_phase3_{DOMAIN}_pilot_v1','pcb_phase3_{DOMAIN}_train_v1'),
                    ('pilot.log','training.log'),('Pilot output','Training output'),('Pilot stopped','Training stopped')]:
        run=run.replace(old,new)
    nb['cells'][4]['source']=run.splitlines(True)
    nb['cells'][5]['source']=['''## Download and review

Download **source_run_checkpoint.zip**, **report.json**, **training.log** and **pcb_environment.json**. Keep them together in a new run folder, separate from the pilot downloads.

Same session: leave RESTORE_FROM=None and rerun the training cell. New session: attach the original dataset and this experiment's trusted checkpoint ZIP (or expanded folder), then set RESTORE_FROM. Clear it for subsequent local reruns. RESUME_ONLY=True refuses to start without a checkpoint. A session loss before a checkpoint is persisted outside that session cannot be recovered.

The report compares baseline and selected candidate, including all six classes. The predeclared review gate asks for at least +1 percentage point mAP50–95, no more than a 1-point mAP50 drop and no class AP50 drop over 5 points. This is a review filter, not proof of significance or a promise of improvement. Retain the old baseline if the candidate is worse. Do not tune on development-test images.

Stop for review before scheduling another model, target training or deployment. No automatic model replacement occurs.
''']
    for i,c in enumerate(nb['cells']):
        c['id']=f'source-training-{i}'
        if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(nb,stream,indent=2)
    return nb

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.package,args.output)
