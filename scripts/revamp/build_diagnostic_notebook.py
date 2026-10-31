"""Build an inference-only diagnostic notebook for the latest full-board run."""
import argparse,ast,base64,hashlib,json,tempfile
from pathlib import Path
from revamp.build_full_boards_notebook import build as training_notebook


def build(package,output):
    with tempfile.TemporaryDirectory() as temporary:
        nb=training_notebook(package,Path(temporary)/'base.ipynb')
    setup=''.join(nb['cells'][2]['source'])
    node=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EMBEDDED_TOOLS' for t in n.targets))
    payload=ast.literal_eval(node.value)
    raw=Path(__file__).with_name('diagnose.py').read_bytes()
    payload['diagnose.py']=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
    start=setup.index('EMBEDDED_TOOLS = ');end=setup.index('\nimport base64, importlib',start)
    setup=setup[:start]+'EMBEDDED_TOOLS = '+repr(payload)+'\n'+setup[end:]
    nb['cells'][2]['source']=[setup.replace('pcb_full_boards_tools','pcb_diagnosis_tools')]
    nb['cells'][0]['source']=['''# PCB validation error diagnosis — no training

Use the same version-2 dataset and the full-board experiment's checkpoint ZIP. If its Kaggle working folder still exists, the default path below works. Otherwise attach the downloaded ZIP as a private dataset and set CHECKPOINT to its path. Keep the original checkpoint downloaded before ending the session.

This notebook compares the old baseline and selected candidate on the 120 source validation boards. It measures missed defects and unmatched predictions at a fixed confidence of 0.25, with IoU 0.5 matching. It also records class/box details for diagnosing errors. No training or development-test evaluation occurs. These diagnostic counts are not mAP and use a different operating point from the training report.
''']
    nb['cells'][4]['source']=['''CHECKPOINT = Path('/kaggle/working/pcb_phase3_source_full_boards_v2/full_boards_run_checkpoint.zip')
if not CHECKPOINT.is_file():
    matches = list(Path('/kaggle/input').rglob('full_boards_run_checkpoint.zip'))
    if len(matches) != 1: raise RuntimeError('Set CHECKPOINT to the trusted downloaded full-board checkpoint ZIP.')
    CHECKPOINT = matches[0]
RUN = Path('/kaggle/working/pcb_source_diagnosis_v2')
if RUN.exists(): raise RuntimeError('Output exists; download its diagnosis.zip or choose a new RUN folder.')
import signal
from IPython.display import display, FileLink
command = [sys.executable, '-m', 'revamp.diagnose', '--package', str(WORK), '--checkpoint', str(CHECKPOINT), '--output', str(RUN), '--device', '0']
env = dict(os.environ, PYTHONPATH=str(TOOLS), MPLCONFIGDIR='/kaggle/working/pcb_mpl')
process = subprocess.Popen(command, cwd=TOOLS, env=env, start_new_session=True)
try:
    if process.wait(): raise RuntimeError('Diagnostic failed; see output above. No training was performed.')
except KeyboardInterrupt:
    try: os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError: pass
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
    raise
finally:
    if (RUN/'diagnosis.zip').exists(): display(FileLink(os.path.relpath(RUN/'diagnosis.zip', Path.cwd())))
''']
    nb['cells'][5]['source']=['Download **diagnosis.zip** and keep the four original training outputs. Save them in `downloads/source_full_boards_v2/`. The ZIP contains per-image and per-class error counts for both models. No model is promoted or retrained.']
    for i,c in enumerate(nb['cells']):
        c['id']=f'diagnosis-{i}'
        if c['cell_type']=='code':compile(''.join(c['source']),c['id'],'exec')
    with Path(output).open('x') as stream:json.dump(nb,stream,indent=2)
    return nb

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();build(args.package,args.output)
