"""Summarize fixed confidence settings from saved development predictions."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from revamp.diagnose import analyze
p=ROOT/'artifacts/kaggle/phase2_v2/package';rows={r['image']:r for r in json.loads((p/'manifest.json').read_text())['views']}
b=ROOT/'artifacts/verification/yolo_pcb_comparison';result={'scope':'Development validation only; external training overlap unverified. These are fixed-threshold box counts, not mAP. Threshold 0.90 comes from author inference example. No independent clean-board false-positive rate.','models':{}}
for f in sorted(b.glob('*.json')):
 d=json.loads(f.read_text())
 if 'details' not in d:continue
 result['models'][f.stem]={}
 for t in [.25,.5,.9]:
  totals={k:0 for k in ['tp','fp','fn']}
  for x in d['details']:
   row=rows[x['image']]
   assert hashlib.sha256((p/row['label']).read_bytes()).hexdigest()==row['label_sha256']
   y=analyze(row,[a for a in x['predictions'] if a['confidence']>=t])
   for c in y['counts'].values():
    for k in totals:totals[k]+=c[k]
  totals.update(precision=totals['tp']/max(1,totals['tp']+totals['fp']),recall=totals['tp']/max(1,totals['tp']+totals['fn']))
  result['models'][f.stem][str(t)]=totals
(b/'threshold_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
