"""Build the Phase 3 source pilot notebook using the unchanged Phase 2 package."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import tempfile
from revamp.build_resume_notebook import build as build_resume


def build(package, output):
    with tempfile.TemporaryDirectory() as temporary:
        nb=build_resume(package,Path(temporary)/'base.ipynb')
    nb['cells'][0]['source']=['''# PCB Phase 3 — source specialist pilot

Attach the same version-2 PCB dataset. Select **GPU T4** and enable Internet for dependency setup. No dataset re-upload is needed. Start in a fresh kernel when switching notebooks.

This run evaluates the existing source baseline, then trains YOLOv8s from COCO weights for **3 epochs on 953 training views**. Validation uses **120 separate full-board images**, at 960 pixels, batch 4. Augmentations are disabled for this controlled pilot. Full precision and zero loader workers simplify recovery. This is a runtime and learning pilot, not the final accuracy experiment.

The run saves complete checkpoint ZIPs after each epoch. Download the ZIP before ending the session. Rerunning resumes local progress; an attached ZIP or expanded checkpoint can restore another session. Resume requires matching package, code, recipe and dependency versions.
''']
    setup=''.join(nb['cells'][2]['source'])
    start=setup.index('EMBEDDED_TOOLS = ');end=setup.index('\nimport base64, importlib',start)
    payload={}
    for name in ['__init__.py','runtime.py','checkpoints.py','smoke.py','pilot.py']:
        raw=Path(__file__).with_name(name).read_bytes()
        payload[name]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    setup=setup[:start]+'EMBEDDED_TOOLS = '+repr(payload)+'\n'+setup[end:]
    setup=setup.replace('/kaggle/working/pcb_resume_tools','/kaggle/working/pcb_pilot_tools')
    nb['cells'][2]['source']=setup.splitlines(True)
    env=''.join(nb['cells'][3]['source'])
    env+="\nassert torch.cuda.is_available(), 'Enable GPU T4 before running this pilot.'\nenvironment['gpu'] = torch.cuda.get_device_name(0)\nPath('/kaggle/working/pcb_environment.json').write_text(json.dumps(environment, indent=2))\n"
    nb['cells'][3]['source']=env.splitlines(True)
    run='''# Run source first. Review its report before scheduling the target pilot.
DOMAIN = 'source'
RESTORE_FROM = None  # Attached pilot_run_checkpoint.zip or expanded checkpoint folder
RESUME_ONLY = False  # True refuses a fresh start without a checkpoint
RUN = Path(f'/kaggle/working/pcb_phase3_{DOMAIN}_pilot_v1')
RUN.mkdir(parents=True, exist_ok=True)
import signal
from IPython.display import display, FileLink
command = [sys.executable, '-m', 'revamp.pilot', '--package', str(WORK),
           '--output', str(RUN), '--domain', DOMAIN, '--device', '0']
if RESTORE_FROM: command += ['--checkpoint', str(RESTORE_FROM)]
if RESUME_ONLY: command += ['--resume-only']
env = dict(os.environ, PYTHONPATH=str(TOOLS), MPLCONFIGDIR='/kaggle/working/pcb_mpl',
           YOLO_CONFIG_DIR=str(RUN/'yolo_settings'), WANDB_MODE='disabled', COMET_MODE='DISABLED')
print('Pilot output:', RUN)
print('Baseline validation runs first. Checkpoints start after training epoch 1.')
print('Interrupt stops the worker. Rerun to resume the last saved epoch.')
log = RUN/'pilot.log'
with log.open('a') as stream:
    process = subprocess.Popen(command, cwd=TOOLS, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            try:
                returncode = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                lines = log.read_text(errors='replace').splitlines()
                if lines: print(lines[-1][-500:])
        if returncode:
            print('\\n'.join(log.read_text(errors='replace').splitlines()[-25:]))
            raise RuntimeError('Pilot stopped or failed. Inspect pilot.log; saved checkpoints remain available. For OOM, stop and review before changing the recipe.')
    except KeyboardInterrupt:
        try: os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError: pass
        try: process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
        print('Stopped. Rerun to recover the last complete epoch.')
    finally:
        for path in [RUN/'pilot_run_checkpoint.zip', RUN/'report.json', log,
                     Path('/kaggle/working/pcb_environment.json')]:
            if path.exists(): display(FileLink(os.path.relpath(path, Path.cwd())))
if (RUN/'report.json').exists(): print((RUN/'report.json').read_text())
'''
    nb['cells'][4]['source']=run.splitlines(True)
    nb['cells'][5]['source']=['''## Stop and review

Download **report.json**, **pilot.log**, **pcb_environment.json** and **pilot_run_checkpoint.zip**. Send the report and log for review before a longer training run. Low validation scores after three epochs do not mean the dataset or pipeline failed; do not select a model from this pilot alone.

Same session: leave RESTORE_FROM=None and rerun. New session: attach the original PCB dataset and your trusted pilot checkpoint; set RESTORE_FROM to the ZIP or expanded folder. Clear RESTORE_FROM on subsequent local reruns. A completed checkpoint rebuilds missing evaluation/report outputs without training again. Checkpoints from the six-image smoke test are incompatible and rejected.

Timing estimates exclude setup, checkpoint writing and final evaluation. Source and target are separate experiments. Target has five supported classes and cannot establish missing-hole performance. Do not change DOMAIN until the source pilot has been reviewed. No app model is replaced.
''']
    for i,cell in enumerate(nb['cells']):
        cell['id']=f'phase3-pilot-{i}'
        if cell['cell_type']=='code':compile(''.join(cell['source']),cell['id'],'exec')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(nb,stream,indent=2)
    return nb

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.package,args.output)
