"""Embed current training/control code; keep the already uploaded dataset unchanged."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


def build(interactive=True):
    cells = []

    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))

    def code(text):
        compile(text, 'notebook cell', 'exec')
        cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

    md('''# PCB six-class training — Pause and Resume
Attach your existing `data_set`. No dataset re-upload or API token is needed. Select GPU T4 and Internet on. Set Session options → Persistence → Files as an extra precaution.

Run the setup cells, then click **Start training** in the controls. The kernel stays available for the **Pause** button. Pause finishes the current epoch, saves the full optimizer/model state, and returns. **Resume** continues from that saved epoch; it refuses to start from scratch if the checkpoint is missing. Only one training worker can run at a time.

The recovery ZIP is updated and verified after **every completed epoch**. **Download it before leaving or ending the Kaggle session.** A full session reset can delete local files; a paused run is not stored outside Kaggle until you download the ZIP. After a reset, attach that ZIP with the original dataset and run setup again, then click Resume. Both intact ZIPs and Kaggle-expanded checkpoint datasets are accepted. Attach only one recovery version.

This notebook embeds the updated training code, so the old script inside `data_set` cannot override it. The dataset, six labels, training recipe and production weights are unchanged. Validation is reported separately for source and target; no test images are used.
''')
    if not interactive:
        md('''## Background run
This edition runs synchronously for **Save Version → Save & Run All**. It saves the same checkpoint after each epoch. Use the Pause/Resume notebook for interactive buttons. A notebook output is not a confirmed off-session backup until saved/downloaded.
''')
    code('''assert not ('controls' in globals() and controls.running), 'Training is active. Use Pause; do not rerun setup.'
import os, subprocess, sys
os.environ.setdefault('YOLO_CONFIG_DIR', '/kaggle/working/ultralytics_config')
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics==8.4.142', 'ipywidgets>=8,<9'])
import torch
assert torch.cuda.is_available(), 'Select GPU T4 before continuing.'
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__)
print('Using device 0; keep the existing Kaggle Torch installation.')
''')
    embedded = {name: (ROOT/name).read_text() for name in ['train_recovery.py', 'notebook_controls.py']}
    code('from pathlib import Path\nimport sys, importlib, hashlib\n'
         "CODE = Path('/kaggle/working/pcb_training_code')\nCODE.mkdir(exist_ok=True)\n"
         f'EMBEDDED_CODE = {embedded!r}\n'
         'for name, source in EMBEDDED_CODE.items():\n'
         "    (CODE/name).write_text(source)\n"
         "print('Pause/resume code installed:', hashlib.sha256((CODE/'train_recovery.py').read_bytes()).hexdigest()[:12])\n")
    code((ROOT/'kaggle_setup.py').read_text() + '\nDATA, RUN = setup()\n'
         '# Prefer the embedded update over the script in the original upload.\n'
         'sys.path.insert(0, str(CODE))\nimportlib.invalidate_caches()\n'
         "for name in ['train_recovery', 'notebook_controls']:\n"
         '    if name in sys.modules:\n'
         '        importlib.reload(sys.modules[name])\n')
    if interactive:
        code('''from notebook_controls import show_controls
# Re-running this cell must not create a second worker or lose the Pause button.
if 'controls' in globals() and controls.running:
    print('Training already running. Use the existing Pause button above.')
else:
    controls = show_controls(DATA, RUN, device=0)
''')
    else:
        code('''from train_recovery import main
# Existing progress is resumed. Missing progress is never treated as a resume.
has_checkpoint = (RUN/'training/weights/last.pt').exists() or (RUN.parent/'six_class_recovery_checkpoint.zip').exists()
main(DATA, RUN, device=0, resume_only=has_checkpoint)
''')
    md('''## Pause, download, resume
1. Click **Pause** and wait for **paused** (the current epoch must finish).
2. Click the checkpoint ZIP link and confirm the download on your laptop.
3. In the same session, click **Resume** whenever ready.
4. In a new session, attach the downloaded checkpoint alongside `data_set`, run setup, then click **Resume**.

The final results link appears when training and both validation reports finish. Download `six_class_recovery_results.zip` for review. No production model is replaced automatically.

If widgets are unavailable, the synchronous training function also handles the notebook's normal Interrupt/Stop as a pause at the last committed epoch. A factory reset, forced process kill, or expired session cannot execute cleanup; recovery then requires a surviving/downloaded checkpoint. Do not use Factory reset to pause.
''')
    for i, cell in enumerate(cells):
        cell['id'] = f'pause-resume-{i}'
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'cells': cells}


if __name__ == '__main__':
    notebook = build()
    for name in ['PCB_Six_Class_Recovery_Kaggle.ipynb', 'PCB_Six_Class_SplitUpload_Kaggle.ipynb', 'PCB_Six_Class_Pause_Resume_Kaggle.ipynb']:
        (ROOT/name).write_text(json.dumps(notebook, indent=2))
    (ROOT/'PCB_Six_Class_Background_Kaggle.ipynb').write_text(json.dumps(build(interactive=False), indent=2))
    print('Notebooks generated; all Python cells compiled.')
