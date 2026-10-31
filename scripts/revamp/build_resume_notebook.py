"""Deliver a checkpoint-enabled notebook without re-uploading the 992 MB data ZIP."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from revamp.package import notebook


def build(package, output):
    package, output = Path(package), Path(output)
    package_hash = hashlib.sha256((package/"package_hashes.json").read_bytes()).hexdigest()
    nb = notebook(package_hash)
    # Fix the dependency cell's newline escaping while preserving the pinned stack.
    nb["cells"][1]["source"] = ["".join(nb["cells"][1]["source"]).replace("\\\\n", "\\n")]
    setup = "".join(nb["cells"][2]["source"])
    start = setup.index("if WORK.exists():")
    end = setup.index("\ndef file_hash(path):")
    extraction = setup[setup.index("INPUT = Path", start):end]
    setup = setup[:start]+"if not WORK.exists():\n"+"\n".join("    "+line for line in extraction.splitlines())+"\n"+setup[end:]
    # Verify the old immutable package first, then install a separate code overlay.
    split = setup.index("sys.path.insert(0, str(WORK))")
    payload = {}
    for name in ("__init__.py","runtime.py","checkpoints.py","smoke.py"):
        raw = Path(__file__).with_name(name).read_bytes()
        payload[name] = {"base64":base64.b64encode(raw).decode(),"sha256":hashlib.sha256(raw).hexdigest()}
    overlay = '''
import base64, importlib
TOOLS = Path('/kaggle/working/pcb_resume_tools')
(TOOLS/'revamp').mkdir(parents=True, exist_ok=True)
for name, record in EMBEDDED_TOOLS.items():
    raw = base64.b64decode(record['base64'])
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    destination = TOOLS/'revamp'/name
    if destination.exists() and destination.read_bytes() != raw:
        raise RuntimeError('Different resume code already installed; use a fresh kernel/session.')
    if not destination.exists(): destination.write_bytes(raw)
sys.path.insert(0, str(TOOLS))
# A previous notebook import can remain cached after rerunning a cell.
for name in list(sys.modules):
    if name == 'revamp' or name.startswith('revamp.'): del sys.modules[name]
importlib.invalidate_caches()
from revamp.runtime import verify_package, configure
print(verify_package(WORK))
CONFIGS = configure(WORK, Path('/kaggle/working/pcb_phase2_configs'))
print('Setup verified; existing matching files reused.', CONFIGS)
'''
    nb["cells"][2]["source"] = (setup[:split]+"EMBEDDED_TOOLS = "+repr(payload)+"\n"+overlay).splitlines(True)
    nb["cells"][0]["source"] = ["# PCB Phase 2 — checkpoints and resume\n\nUse your existing **version-2 dataset ZIP**. No data re-upload is needed. This notebook still runs only the six-image Phase 2 check; full candidate training remains Phase 3.\n\nEvery completed epoch saves weights, optimizer, scheduler, random states and history in an atomic checkpoint ZIP. Rerun cells after an interruption: matching setup is reused and training resumes from the last fully saved epoch. At most the unfinished epoch needs repeating. Completed runs reuse their report.\n\n**Before ending a session, download the latest checkpoint ZIP.** `/kaggle/working` is session storage, not a durable backup. After a session reset, attach the downloaded checkpoint as a private Kaggle dataset and set `RESTORE_FROM` below. A ZIP, an expanded checkpoint folder, or its `checkpoint_manifest.json` is accepted. Resume requires the same dataset, recipe, PyTorch and Ultralytics versions.\n"]
    run = '''# Rerun this cell to continue local progress; it will not restart completed epochs.
# New session: set RESTORE_FROM to your attached checkpoint ZIP or expanded folder.
RESTORE_FROM = None  # Example: '/kaggle/input/my-pcb-checkpoint/fresh_run_checkpoint.zip'
RESUME_ONLY = False  # True refuses a new start when no checkpoint is available.
RUN = Path('/kaggle/working/pcb_phase2_smoke_resume')
import signal
from IPython.display import display, FileLink

command = [sys.executable, '-m', 'revamp.smoke', '--package', str(WORK),
           '--output', str(RUN), '--device', '0' if torch.cuda.is_available() else 'cpu']
if RESTORE_FROM: command += ['--checkpoint', str(RESTORE_FROM)]
if RESUME_ONLY: command += ['--resume-only']
env = dict(os.environ, PYTHONPATH=str(TOOLS), MPLCONFIGDIR='/kaggle/working/pcb_mpl')
print('Checkpoints: ', RUN)
print('Use Interrupt to stop; rerun this cell to resume. Logs: pause.log / resume.log')
process = subprocess.Popen(command, cwd=TOOLS, env=env, start_new_session=True)
try:
    returncode = process.wait()
    if returncode: raise RuntimeError('Check stopped or failed; see the logs. Saved checkpoints remain available.')
except KeyboardInterrupt:
    # Stop the coordinator AND its training subprocess; avoid an orphan GPU worker.
    try: os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError: pass
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
    print('Stopped. Rerun this cell to restore the last complete epoch.')
finally:
    for name in ['fresh_run_checkpoint.zip', 'first_run_checkpoint.zip']:
        path = RUN/name
        if path.exists():
            print('DOWNLOAD checkpoint:', path)
            display(FileLink(os.path.relpath(path, Path.cwd())))
            break
    else:
        print('No completed epoch saved yet. A new start is needed only if epoch 1 never finished.')
if (RUN/'report.json').exists(): print((RUN/'report.json').read_text())
'''
    nb["cells"][4]["source"] = run.splitlines(True)
    nb["cells"][5]["source"] = ["## Resume steps\n\n**Same session:** leave `RESTORE_FROM = None` and rerun the training cell. Setup cells can also be rerun safely. `RESUME_ONLY = True` prevents an accidental fresh start.\n\n**New session:** attach the original version-2 dataset plus your downloaded checkpoint; run setup; set `RESTORE_FROM` to the attached ZIP or expanded directory; run the training cell. On subsequent reruns in that session clear `RESTORE_FROM` so local, newer progress is used. Never resume from a checkpoint you do not trust.\n\nThe latest complete ZIP is authoritative, even if the raw weights or CSV were interrupted during a write. A final-epoch checkpoint rebuilds the report without training again. If the session disappeared before a checkpoint was downloaded or otherwise persisted, the lost files cannot be recovered by this notebook.\n\nStop after this check. Its scores are memorization results, not model accuracy. Save the environment JSON and report for Phase 3.\n"]
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(nb,stream,indent=2)
    return nb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    build(args.package,args.output)
