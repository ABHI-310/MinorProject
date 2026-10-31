"""Package the reviewed legacy model and resumable mixed-domain Kaggle runner."""
import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'artifacts/kaggle/yolo_pcb_fewshot_v2';bundle=out/'bundle'
shutil.copyfile(ROOT/'scripts/revamp/train_yolo_pcb_mixed.py',bundle/'train_yolo_pcb_mixed.py')
(bundle/'ATTRIBUTION.txt').write_text('YOLO-PCB by JiaLim98, upstream tree d15641ec341a7403d85917114a0806afc0805ac9. See UPSTREAM_LICENSE and UPSTREAM_README.md. PKU/TDD-Net dataset usage is academic research only; DsPCBSD+ CC BY 4.0. Data and third-party rights remain separate from code licensing.\n')
hashes={str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='bundle_hashes.json'}
(bundle/'bundle_hashes.json').write_text(json.dumps(hashes,indent=2))
archive=out/'pcb_yolo_fewshot_training_v2.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(bundle.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(bundle)))
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(keepends=True)))
def code(s):cells.append(dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True)))
md('''# YOLO-PCB: 50-image target adaptation, FP32

Remove the old v1 training bundle input. Upload **pcb_yolo_fewshot_training_v2.zip** and the existing **pcb_revamp_phase2.zip** as Kaggle notebook inputs (new private dataset or reuse the existing phase-2 input). Enable a GPU. No paid services.

This is a new training experiment initialized from YOLO-PCB, not from our app model. It uses 480 source training boards and **50 unique labeled target TRAIN images total**, with equal domain contribution per batch. This is an image-level budget, not K shots per board type. Target board independence remains unknown. FP32 is used throughout; no AMP. Validation: 120 source images (six classes), 403 target images (five classes; no missing-hole claim). Historical development partitions are not an independent final test.

First run: stop at epoch 5 of the fixed 40-epoch schedule. Download the recovery ZIP. Inspect both domains before extending the same run to epoch 40. The pilot is not guaranteed to succeed; it measures whether this adaptation is worth continuing.
''')
code('''from pathlib import Path
import json, zipfile, hashlib, sys, os, torch
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/pcb_yolo_fewshot_v2')
WORK.mkdir(parents=True, exist_ok=True)
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'Enable a GPU in Kaggle notebook settings.'
assert hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'), 'This runner requires PyTorch 2.6+ checkpoint APIs.'

def extract_checked(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        assert len(names)==len(set(names)), 'Duplicate ZIP paths'
        for info in z.infolist():
            n=info.filename
            assert not Path(n).is_absolute() and '..' not in Path(n).parts and '\\\\' not in n
            assert ((info.external_attr>>16)&0o170000)!=0o120000, 'ZIP symlink rejected'
        assert z.testzip() is None, 'Damaged ZIP'
        z.extractall(dest)
    return dest

bundles=[p.parent for p in INPUT.rglob('train_yolo_pcb_mixed.py') if (p.parent/'initial.pt').is_file() and (p.parent/'legacy').is_dir()]
if bundles:
    assert len(bundles)==1, 'Attach only one training bundle version.'
    BUNDLE=bundles[0]
else:
    candidates=list(INPUT.rglob('pcb_yolo_fewshot_training_v2.zip'))
    assert len(candidates)==1, 'Attach pcb_yolo_fewshot_training_v2.zip.'
    BUNDLE=extract_checked(candidates[0], WORK/'bundle')
for name, expected in json.loads((BUNDLE/'bundle_hashes.json').read_text()).items():
    assert hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest()==expected, 'Bundle checksum mismatch: '+name

packages=[]
for p in INPUT.rglob('manifest.json'):
    if (p.parent/'images').is_dir() and (p.parent/'labels').is_dir():
        record=json.loads(p.read_text())
        if record.get('schema')==1 and 'views' in record: packages.append(p.parent)
if packages:
    assert len(packages)==1, 'Attach only the verified phase-2 package.'
    PACKAGE=packages[0]
else:
    candidates=list(INPUT.rglob('pcb_revamp_phase2.zip'))
    assert len(candidates)==1, 'Attach the existing pcb_revamp_phase2.zip.'
    PACKAGE=extract_checked(candidates[0], WORK/'package')
print('Bundle:', BUNDLE, '\\nData:', PACKAGE)
''')
md('''## Run the five-epoch pilot
The training schedule is fixed at 40 epochs from the beginning. `STOP_AFTER_EPOCH = 5` pauses at an epoch boundary; increasing it later continues the same schedule. Precision, target-image selection, budget, batch and image size are part of the checkpoint signature: do not change them while resuming. **Version-1 mixed/AMP checkpoints are incompatible: do not attach them as a resume input to this experiment.**

Every epoch saves `last.pt`, `best.pt`, metrics and a checksummed `resume.zip`. An interrupted incomplete epoch is rerun from the last completed checkpoint. A ZIP does not survive Kaggle deleting the session unless you download it or save notebook outputs.
''')
code('''import subprocess, signal
STOP_AFTER_EPOCH = 5  # After pilot review, set 40 and rerun to continue.
TARGET_BUDGET = 50  # 50 or 100 images TOTAL. Separate fresh run for each budget.
RUN = WORK/f'k{TARGET_BUDGET}_training'
# Fresh session: attach your downloaded resume.zip as a private Kaggle input.
# Leave empty to start/reuse the current run. To restore, set its actual path.
RESUME_ARCHIVE = ''
cmd=[sys.executable, str(BUNDLE/'train_yolo_pcb_mixed.py'),
     '--package',str(PACKAGE),'--bundle',str(BUNDLE),'--run',str(RUN),
     '--device','cuda','--epochs','40','--batch','4','--imgsz','608',
     '--patience','8','--target-budget',str(TARGET_BUDGET),'--stop-after-epoch',str(STOP_AFTER_EPOCH)]
if RESUME_ARCHIVE and not (RUN/'last.pt').exists(): cmd += ['--resume-archive',RESUME_ARCHIVE]
env=os.environ.copy();env['MPLCONFIGDIR']=str(WORK/'matplotlib')
log=WORK/'training.log'
with log.open('a', buffering=1) as stream:
    process=subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    print('Log:', log, '\\nCheckpoint:', RUN/'resume.zip', flush=True)
    try:
        while process.poll() is None:
            import time
            time.sleep(10)
            lines=log.read_text(errors='replace').splitlines()
            if lines: print(lines[-1][:1500], flush=True)
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL);process.wait()
        print('Interrupted. Rerun to restart from the last completed epoch.')
    else:
        if process.returncode:
            print(log.read_text(errors='replace')[-12000:])
            raise RuntimeError('Training failed. Inspect the log; saved completed epochs remain recoverable.')
''')
code('''from IPython.display import display, FileLink
if (RUN/'history.json').exists():
    history=json.loads((RUN/'history.json').read_text())
    for row in history:
        print('Epoch',row['epoch'],'seconds',round(row['seconds']),
              'source AP50',round(row['metrics']['source']['map50'],4),
              'target AP50',round(row['metrics']['target']['map50'],4))
    print('Download recovery ZIP and history before ending the session:')
    display(FileLink(str(RUN/'resume.zip')))
    display(FileLink(str(RUN/'history.json')))
    display(FileLink(str(RUN/'target_selection.json')))
    display(FileLink(str(WORK/'training.log')))
    if (RUN/'initial_validation.json').exists(): display(FileLink(str(RUN/'initial_validation.json')))
else:
    print('No completed training epoch yet. Inspect training.log.')
''')
md('''## What to do after epoch 5
1. Download **resume.zip**, **history.json**, **training.log**, and **initial_validation.json** when present. The recovery ZIP also contains **target_selection.json** documenting exactly which labeled target images were used.
2. Review both domains and the per-class metrics against initial validation. Do not extend a run whose losses are nonfinite or whose target metrics remain flat without investigating.
3. If the pilot supports continuing, set `STOP_AFTER_EPOCH = 40` and rerun the training cell. The optimizer and learning-rate schedule continue from the saved epoch; patience can stop the run earlier.
4. For a fresh Kaggle session, attach the same two inputs and **resume.zip**, set `RESUME_ARCHIVE` to its actual `/kaggle/input/.../resume.zip` path, and run the cells. Keep the same runtime/recipe; mismatches fail explicitly.

These outputs are development evidence. Clean-board false-alarm measurement and independently sourced missing-hole examples remain necessary before making broad six-class reliability claims.
''')
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},nbformat=4,nbformat_minor=5)
path=ROOT/'notebooks/YOLO_PCB_Few_Shot_Kaggle.ipynb';path.write_text(json.dumps(nb,indent=2))
print(path);print(archive,archive.stat().st_size)
