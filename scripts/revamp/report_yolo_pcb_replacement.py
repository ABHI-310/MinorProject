"""Build the replacement decision from saved evidence without changing app models."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from revamp.diagnose import analyze
base=ROOT/'artifacts/verification';out=base/'yolo_pcb_replacement';out.mkdir(exist_ok=True)
package=ROOT/'artifacts/kaggle/phase2_v2/package';manifest=json.loads((package/'manifest.json').read_text());rows={r['image']:r for r in manifest['views']}
comparisons={}
for domain,folder in [('source','yolo_pcb_comparison'),('target','yolo_pcb_target')]:
 comparisons[domain]={}
 for file in sorted((base/folder).glob('*.json')):
  data=json.loads(file.read_text())
  if 'details' not in data:continue
  settings={}
  for threshold in [.25,.5,.9]:
   per_class={}
   for detail in data['details']:
    r=rows[detail['image']]
    assert hashlib.sha256((package/r['image']).read_bytes()).hexdigest()==r['image_sha256']
    assert hashlib.sha256((package/r['label']).read_bytes()).hexdigest()==r['label_sha256']
    d=analyze(r,[p for p in detail['predictions'] if p['confidence']>=threshold])
    for n,c in d['counts'].items():
     totals=per_class.setdefault(n,dict(tp=0,fp=0,fn=0))
     for k in totals:totals[k]+=c[k]
   totals={k:sum(c[k] for c in per_class.values()) for k in ['tp','fp','fn']}
   totals['precision']=totals['tp']/(totals['tp']+totals['fp']) if totals['tp']+totals['fp'] else None
   totals['recall']=totals['tp']/(totals['tp']+totals['fn']) if totals['tp']+totals['fn'] else None
   settings[str(threshold)]=dict(totals=totals,per_class=per_class)
  comparisons[domain][file.stem]=dict(images=len(data['details']),weights_sha256=data['summary']['sha256'],imgsz=data['summary']['imgsz'],thresholds=settings)
stress=json.loads((out/'integration_and_stress.json').read_text());stress_totals={}
for r in stress['stress']:
 d=stress_totals.setdefault(r['variant'],dict(tp=0,fp=0,fn=0))
 for k in d:d[k]+=r['counts'][k]
active={}
for name in ['source','target']:
 p=ROOT/f'artifacts/models/{name}/model.pt';info=json.loads(p.with_name('model_info.json').read_text());actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==info['checkpoint_sha256'];active[name]=actual
src_hash={r['image_sha256'] for r in rows.values() if r['dataset']=='source'};tgt_hash={r['image_sha256'] for r in rows.values() if r['dataset']=='target' and r['split']=='val'}
report=dict(date='2026-10-29',decision='NO-GO for unconditional replacement',reason='Good PKU-family development performance, but zero recall on DsPCBSD+ validation for five supported classes.',comparisons=comparisons,integration=stress,stress_totals=stress_totals,active_checkpoint_hashes_verified=active,exact_source_package_target_val_overlap=len(src_hash&tgt_hash),gates={'class_order':'passed: all six exact names','api_contract':'passed in isolated test registry for six labeled inputs','coordinate_bounds':'passed for tested API and stress predictions','runtime':'measured on local CPU only','PKU_external_training_overlap':'unresolved; source metrics are development only','other_domain_transfer':'failed at both tested input sizes','clean_board_false_positive_rate':'not established; blank-image sanity checks are not real clean boards','independent_all_six_class_accuracy':'not established','production_loader':'legacy loader required; custom models/utils modules need isolation','promotion':'not performed'},protocol={'confidence_settings':[.25,.5,.9],'threshold_090_origin':'author README inference example; not independent calibration','nms_iou':.7,'match_iou':.5,'matching':'greedy same-class one-to-one; count metrics, not mAP','geometry':'square letterbox; RGB float32 0..1; legacy scores objectness times class probability','target_scope':'five supported labels only; no missing-hole evaluation','baseline_scope':'controlled square inference; not claimed to reproduce default rectangular API preprocessing','test_partitions':'not used in this validation','latency':'not a controlled speed comparison between models'})
(out/'replacement_report.json').write_text(json.dumps(report,indent=2))
text='''# YOLO-PCB replacement validation — 2026-10-29

Decision: **do not replace the app's existing models with this checkpoint yet.**

YOLO-PCB is a promising PKU-family candidate. At 960 pixels and the author's example confidence of 0.90 it matched 531/562 source-validation defects, with one unmatched prediction and 31 misses (99.81% precision, 94.48% recall). At the UI's current 0.25 default it matched 561, with 190 unmatched predictions. Confidence scores are not calibrated between models. Source validation has shared PKU/TDD-Net ancestry with the external checkpoint; external training membership is unknown. These are development counts, not independent accuracy or mAP.

On all 403 DsPCBSD+ validation images (746 boxes, five supported classes), YOLO-PCB matched **zero defects at either 608 or 960 input size** at 0.25 confidence. At 608 there were three unmatched predictions; at 960 no predictions. Precision is undefined when there are no predictions; recall is zero. The app's source model also matched zero, while its target-trained model matched 591 with 366 unmatched predictions and 155 misses under controlled square preprocessing. This demonstrates failure to transfer across an in-scope imaging domain. Missing hole is untested in this domain. Exact byte overlap with source package images is zero, which does not prove independence from every external training source.

The candidate passed isolated API/direct prediction equality checks on one source image per class, finite in-bounds box checks, annotation encoding, database persistence through the existing API, and checkpoint identity. This uses a test registry injected into the app; active model loaders and checkpoints were not replaced. Legacy YOLOv5 classes require a compatible loader and module isolation; this is not a file-only swap.

Local CPU inference measurements and exploratory brightness/blur checks are recorded in replacement_report.json. The perturbation check used 12 familiar source images and 54 boxes, so it is not a new domain benchmark. Synthetic black/gray/white blanks produced no predictions at 0.90, but they do not establish false-positive performance on clean PCBs.

No independent six-class holdout or real clean-board false-positive benchmark is available. Training overlap, independent accuracy and clean-board behavior remain open. Do not use the strong source score in a paper as evidence of new-board generalization.

Next: retain current models, obtain a documented independent six-class set plus clean boards, and either restrict a future YOLO-PCB option to a demonstrated imaging domain or fine-tune it on the intended domain using group-disjoint splits. Select thresholds on validation only and keep the final test sealed. Do not infer machine remaining life from these results.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/revamp/validate_yolo_pcb.py --dataset source
.venv/bin/python scripts/revamp/validate_yolo_pcb.py --dataset target
.venv/bin/python scripts/revamp/check_yolo_pcb_replacement.py
.venv/bin/python scripts/revamp/report_yolo_pcb_replacement.py
```

The validation scripts reuse cached predictions. Evidence: artifacts/verification/yolo_pcb_comparison, artifacts/verification/yolo_pcb_target and artifacts/verification/yolo_pcb_replacement. Model/source downloads and isolated seaborn dependency are in artifacts/external_models/yolo_pcb_review. Nothing in the live model registry was changed.
'''
(ROOT/'reports/revamp/yolo_pcb_replacement_validation.md').write_text(text)
print(report['decision']);print(stress_totals);print(stress['latency_ms'])
