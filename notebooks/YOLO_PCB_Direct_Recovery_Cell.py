# Replace only the training cell. Keep the original setup cell and inputs.
from pathlib import Path
import hashlib, subprocess, signal, os, sys, time
from IPython.display import display, FileLink

if 'BUNDLE' not in globals() or 'PACKAGE' not in globals():
    raise RuntimeError('Run the existing setup cell first; BUNDLE and PACKAGE are required.')
BUNDLE, PACKAGE = Path(BUNDLE), Path(PACKAGE)
source = (BUNDLE/'train_yolo_pcb_mixed.py').read_bytes()
assert hashlib.sha256(source).hexdigest() == '0d9ae5df93eb7b770da0102dcf38bc37c23674ec21510532c781ab3cee1fb78b', 'This patch is for the original v1 runner from your traceback. No files were changed.'
source = source.decode('utf-8')

def replace_once(old, new):
    global source
    assert source.count(old) == 1, 'Unexpected runner contents; patch stopped.'
    source = source.replace(old, new, 1)

replace_once("scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')",
             "scaler=torch.amp.GradScaler('cuda',enabled=False)")
replace_once("with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):",
             "with torch.autocast(device_type=device.type,enabled=False):")
replace_once(" if a.probe:\n", """ # Fixed image-level few-shot budget, selected only from target TRAIN rows.
 pool=[];seen=set()
 for row in rows['target']['train']:
  if row['image_sha256'] not in seen:
   pool.append(row);seen.add(row['image_sha256'])
 rng=np.random.default_rng(42);pool=[pool[i] for i in rng.permutation(len(pool))]
 chosen=[];covered=set()
 for cls in range(1,6):
  if cls in covered:continue
  row=next((r for r in pool if any(int(x[0])==cls for x in r['labels']) and r not in chosen),None)
  if row is None:raise ValueError('Target training class missing: '+str(cls))
  chosen.append(row);covered.update(int(x[0]) for x in row['labels'])
 chosen += [r for r in pool if r not in chosen]
 if len(chosen)<50:raise ValueError('Need at least 50 unique target training images')
 rows['target']['train']=chosen[:50]
 assert all(r['dataset']=='target' and r['split']=='train' for r in rows['target']['train'])
 print('Target labeled TRAIN images total:',len(rows['target']['train']),'; precision: FP32',flush=True)
 if a.probe:
""")
replace_once('recipe=dict(epochs=', "recipe=dict(precision='fp32',target_budget=50,target_image_ids=[r['parent_id'] for r in rows['target']['train']],epochs=")
compile(source, 'fewshot_fp32_runner', 'exec')

# New experiment folder: never load the old full-target AMP checkpoint.
WORK = Path('/kaggle/working/pcb_yolo_fewshot_direct_v1')
WORK.mkdir(parents=True, exist_ok=True)
RUN = WORK/'training_k50_fp32'
runner = WORK/'train_yolo_pcb_fewshot_fp32.py'
runner.write_text(source)
STOP_AFTER_EPOCH = 5  # Later set 40 to continue this SAME run after pilot review.
cmd = [sys.executable, str(runner), '--package', str(PACKAGE),
       '--bundle', str(BUNDLE), '--run', str(RUN), '--device', 'cuda',
       '--epochs', '40', '--batch', '4', '--imgsz', '608', '--patience', '8',
       '--stop-after-epoch', str(STOP_AFTER_EPOCH)]
env = os.environ.copy()
env['MPLCONFIGDIR'] = str(WORK/'matplotlib')
log = WORK/'training.log'
with log.open('a', buffering=1) as stream:
    process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    print('50 labeled target TRAIN images total; FP32; original YOLO-PCB initialization.')
    print('Log:', log, '\nBackup:', RUN/'resume.zip', flush=True)
    previous = None
    try:
        while process.poll() is None:
            time.sleep(10)
            lines = log.read_text(errors='replace').splitlines()
            latest = lines[-1] if lines else ''
            if latest and latest != previous:
                print(latest[:2000], flush=True)
                previous = latest
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        print('Stopped. Rerun this cell to restart from the last completed epoch.')
    else:
        if process.returncode:
            print(log.read_text(errors='replace')[-12000:])
            raise RuntimeError('Training failed. Finite-gradient protection is still enabled; inspect this new log.')

for name in ['resume.zip', 'history.json', 'initial_validation.json', 'signature.json']:
    if (RUN/name).exists(): display(FileLink(str(RUN/name)))
if log.exists(): display(FileLink(str(log)))
