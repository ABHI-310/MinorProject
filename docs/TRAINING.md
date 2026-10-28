# PCB model revamp

This is the new, phased workflow. Historical scripts, datasets, checkpoints and the
working app remain intact. Phases 1 and 2 are implemented. Phase 2 includes a tiny
training-only learning/resume check; it does not train or promote an app model.
Full candidate training is Phase 3.

## Phases and completion gates

1. **Data and baseline audit (implemented).** Inventory prepared PKU and DsPCBSD+
   data; compare every prepared annotation and image with its local original;
   verify class coverage, splits and checkpoint hashes; produce review images.
   Gate: structural/provenance checks pass, sampled annotation review is recorded,
   and remaining data limitations are explicit. This does not certify every label.
2. **Training-data preparation and Kaggle packaging (implemented locally).** Create a new versioned
   dataset from accepted inputs, add training-only scale-preserving views and
   explicitly reviewed background candidates, and build one reproducible Kaggle notebook.
   Gate: no parent/split leakage, valid geometry, complete package hashes, tiny
   learning check and fresh-session resume check pass. Do not package development
   test images as training or validation.
3. **Source and target training.** Train compact pretrained YOLOv8s candidates,
   then a bounded mixed-domain candidate if warranted. Select using validation.
   Gate: compare against reproduced baselines on identical evaluation settings;
   retain specialists if a unified model sacrifices either domain.
4. **Adaptation comparison.** Compare source-only, plain fine-tuning and matched
   adapter controls at an explicitly counted label budget. MMD is an experiment,
   not a required positive result. Gate: source-only initialization, paired data,
   no target-pretrained weights in the few-shot claim, honest uncertainty.
5. **Frozen evaluation and model packaging.** Freeze model, resolution,
   preprocessing and thresholds; evaluate development and approved external data;
   package weights, metrics, predictions and provenance. Gate: per-class/domain
   reporting and correct evaluation status, with no automatic model promotion.
6. **App integration and submission.** Connect approved models to the existing
   upload workflow, demonstrate synthetic degradation trends, and document results.
   Gate: API parity, preserved rollback models and a reproducible demo.

Hardware/camera integration, on-device training and real-machine lifetime
validation are future scope. Synthetic RUL remains labelled synthetic. Arbitrary
uploaded images are not evidence of one machine's chronological degradation.

## Phase 1

Run from the repository root using the existing environment:

```sh
.venv/bin/python scripts/manage.py audit --output reports/revamp/phase1_new
.venv/bin/python -m pytest tests/test_revamp_audit.py -q
```

Every run requires a **new** output directory; existing reports are never replaced.
`--root` permits auditing a copy of this repository. Only the selected report
directory is written. An output directory inside data or either deployed model
folder is refused. Exit code 1 means a blocking audit finding; 2 means the audit
could not start. Warnings require interpretation and do not silently fix data.

Outputs:

- `summary.json` and `SUMMARY.md`: checks, counts, blockers and limitations.
- `manifest.jsonl`: one record per prepared image, including annotation hash,
  parent/group identity, original provenance and split role.
- `issues.json`: actionable failures and warnings with file references.
- `near_duplicates.json`: cross-split perceptual-hash candidates for review.
- `baselines.json`: hash-verified checkpoint metadata and previously saved metrics;
  **not newly reproduced inference results**.
- `review_samples.json` and `samples/*.jpg`: deterministic train/validation
  annotation overviews and detail crops. No test-image galleries are created.
- `artifact_manifest.json`: hashes of all generated artifacts plus audit inputs
  (including code/configuration) for reproducibility.

The legacy `test` partitions are explicitly `development_test`: they have already
been inspected. Auditing their structure does not restore test independence.
PKU groups are board-template prefixes, not proof of distinct manufactured boards.
DsPCBSD+ board identities are unavailable. pHash is only a screening heuristic;
zero candidates does not prove absence of visual leakage. Empty labels are
unverified negatives until human review, never automatically healthy images.

Raw-to-prepared agreement verifies faithful conversion, not the physical truth of
the original annotation. Class 0 (missing hole) is unsupported in target data;
hole breakout, pinhole and missing pad must never be silently mapped to it.

## Phase 2

The current data upload is **`artifacts/kaggle/phase2_v2/pcb_revamp_phase2.zip`**. Use the
updated **`notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb`** with that same ZIP; no data
re-upload is needed if it is already attached. The original version-2 notebook is
preserved, but the updated notebook adds usable restart/recovery controls.
Version 1 is retained as
an intermediate build; use version 2, which includes the corrected dependency
setup and stricter checkpoint/package checks. The ZIP is about 992 MB (946 MiB).
Keep it private and retain the source attribution bundled in `USAGE.md`.

Inputs:

- Source training: 480 original full boards, 470 native 640-pixel defect crops,
  and three reviewed background pilot crops. At most one positive crop per parent;
  ten parents had no safe crop. Full boards are about half of these views.
- Source validation: all 120 original full boards, without crops or resampling.
- Target training/validation: the existing 3,606/403 images, unchanged.
- All 1,085 development-test parents are excluded. No new external dataset is used.

Three configurations are generated: `source_full` for the full-board control,
`source_views` for the crop/background candidate, and `target` for the separate
target specialist. Do not combine their validation scores into one aggregate.
The existing specialist weights are included for later baseline reproduction;
`weights/yolov8s.pt` is the existing 80-class COCO initialization for fresh training.
The nano checkpoint is only for the cheap functional check. Local checkpoint
metadata and hashes were checked; no upstream checksum attestation is claimed.

Background review was bounded: 24 candidates contained only seven distinct image
hashes. Three clear representatives were accepted; ambiguous or repeated crops
were excluded. The recorded reviewer is Codex, not a human PCB expert. These three
background crops are a conservative pilot, **not** a healthy-board benchmark or
evidence that false alarms are solved. A larger expert-reviewed negative set is
an optional later improvement, not a prerequisite for the first training run.

Rebuild into **new** directories from the repository root:

```sh
.venv/bin/python scripts/manage.py prepare --output data/prepared/revamp_new
# Inspect review/candidates_*.jpg and record decisions against candidates.json.
# The existing review applies only to the exact candidates in phase2_v1.
.venv/bin/python scripts/manage.py package \
  --prepared data/prepared/revamp_v1 \
  --background-review reports/revamp/phase2_background_review.json \
  --output artifacts/kaggle/new
.venv/bin/python -m pytest tests/test_revamp_audit.py tests/test_revamp_phase2.py tests/test_training_pause_resume.py -q
```

Omit `--background-review` to package zero negatives. Unreviewed candidates are
never included automatically. Package and preparation outputs refuse existing
directories. All views record parent IDs, windows, labels, split and hashes.
Partially intersecting or near-edge annotations cause a crop to be rejected.

Kaggle handoff:

1. Upload the version-2 ZIP as a **private dataset** and import the updated resume notebook.
2. Attach that dataset. Both an expanded dataset and an attached ZIP are supported;
   attach only one version. The notebook copies/extracts to `/kaggle/working`.
3. Run setup with Internet available if dependencies need installation. Ultralytics
   is pinned to 8.4.142; constraints preserve Kaggle's installed torch, torchvision
   and NumPy. If incompatible, setup fails instead of replacing the CUDA stack.
4. Run the bounded check on CPU or one available GPU. Download the environment JSON,
   smoke report, `pause.log` and `resume.log`. Keep these for the Phase 3 handoff.
5. Stop there. This notebook has no full-candidate training command.

### Checkpoints and interruptions

Every completed epoch publishes an atomic ZIP containing last/best weights,
optimizer, scheduler, early-stopping state, random states, training arguments and
CSV history. A partial epoch or interrupted ZIP publication leaves the previous
complete backup available. At most the unfinished epoch needs repeating.

- **Same session:** rerun the training cell with `RESTORE_FROM = None`. Setup cells
  reuse matching files. Saved progress resumes automatically; completed reports
  are reused without retraining. Set `RESUME_ONLY = True` to prohibit a fresh start.
- **Intentional stop:** use notebook Interrupt. The cell stops the coordinator and
  its training subprocess, then shows the latest checkpoint download link.
- **Before ending/resetting a session:** download `fresh_run_checkpoint.zip`
  (or `first_run_checkpoint.zip` if only the initial epoch completed).
- **New session:** attach your downloaded checkpoint as a private dataset, keep
  the original data package attached, and set `RESTORE_FROM` to its ZIP, expanded
  checkpoint directory, or `checkpoint_manifest.json`. Run setup and training.
  Clear `RESTORE_FROM` on later reruns so local, newer progress is used.

The run signature rejects a different dataset, recipe, PyTorch or Ultralytics
version. Only load your own trusted checkpoints. `/kaggle/working` is session
storage: an unexpected session loss can lose files that were never downloaded or
otherwise persisted. Checkpoints do not provide automatic cloud backup.

The updated notebook embeds a separately hashed code overlay in
`/kaggle/working/pcb_resume_tools`; it does not change the original dataset package.
Its dependency constraints also use proper newline-separated package versions.
Rebuild the updated notebook into a new filename with:

```sh
.venv/bin/python scripts/manage.py notebook \
  --package artifacts/kaggle/phase2_v2/package \
  --output /tmp/PCB_Revamp_Phase2_Resume_Kaggle.ipynb
```

The learning check intentionally uses the **same six training crops** for training
and evaluation. It requires decreasing box loss and a minimum memorization score.
It pauses after epoch one, exports an optimizer-bearing ZIP, moves the old run away,
and resumes in another Python process/folder. CSV epoch continuity and restored
optimizer state are checked. This demonstrates practical recovery, not bitwise
equivalence across hardware/PyTorch versions. It never uses validation or test data.

To repeat locally against the actual packaged code (use an absolute output path):

```sh
cd artifacts/kaggle/phase2_v2/package
YOLO_CONFIG_DIR=/tmp/pcb-revamp-settings ../../../../.venv/bin/python -m revamp.smoke \
  --package . --output /tmp/pcb-phase2-smoke-new --device cpu
```

The Kaggle service and GPU environment must still pass their own notebook check.
Local CPU success does not establish Kaggle compatibility or a candidate's accuracy.

## Next phase

Phase 3 starts with a timed Kaggle GPU pilot and reproduces each existing specialist
on its matching validation split. Then run a bounded source experiment and target
experiment, with per-run checkpoint exports and fixed validation protocols. Compare
full-board source training with the prepared views before deciding on a mixed-domain
run. Freeze recipes and thresholds before development-test evaluation. No camera or
app changes are needed for this phase.

## Scope and working targets

Targets are planning goals, not guaranteed outcomes or reasons to tune on test:

- Source historical development: 80–85% AP50 and 40–45% AP50:95.
- Target historical development: 82–86% AP50 and 40–45% AP50:95.
- Improve target spur recall toward 65–70%, recording the precision trade-off.
- Seek approximately 80% precision and recall at a validation-selected threshold.
- Report external generalization separately; no external target is asserted before
  establishing the baseline and supported classes.

Budget envelope: 10–14 working days, 12–18 core GPU hours plus 4–6 hours reserve,
re-estimated after a timed Kaggle pilot. No broad model search. No paid services.

## Phase 3 — source pilot (current next step)

Import `notebooks/PCB_Revamp_Phase3_Pilot_Kaggle.ipynb` into Kaggle. Keep the same
version-2 dataset attached; enable GPU T4 and Internet for dependency setup. Use
a fresh kernel when switching from the smoke notebook. Run cells in order and
leave `DOMAIN = 'source'`, `RESTORE_FROM = None`, `RESUME_ONLY = False` initially.

The notebook first evaluates the existing source baseline on 120 held-out source
validation boards. It then trains YOLOv8s from the packaged COCO initialization
for three epochs on 953 source views: 480 full boards, 470 positive crops and
three reviewed backgrounds. Both baseline and candidate are evaluated at 960
pixels with the same validation settings. Batch size is four, workers zero,
full precision, with explicit augmentation settings including an empty
Albumentations list. No development-test data is used and no app weights change.

The report records per-class metrics, baseline differences, training/validation
epoch times and peak GPU memory. Three epochs do not establish final accuracy.
The 60-epoch time projection is only a lower bound: it excludes checkpoint ZIP
writes, setup and final evaluation, and must be remeasured if the recipe changes.

Download these files from the links beneath the run cell:

- `report.json`
- `pilot.log`
- `pcb_environment.json`
- `pilot_run_checkpoint.zip`

Same-session interruption: leave RESTORE_FROM unset and rerun; the atomic ZIP
restores the last completed epoch. New session: attach the original dataset and
trusted pilot ZIP (or expanded checkpoint); set RESTORE_FROM to that path.
Clear RESTORE_FROM on later local reruns. RESUME_ONLY=True refuses a fresh start.
Recipe, code and key environment versions must match. A completed checkpoint
can rebuild missing evaluation/report outputs without repeating training.
An interruption before epoch 1 finishes has no training checkpoint to resume.

Stop after the source pilot for review. The runner supports a separate target
pilot (3606 train / 403 validation, 640 pixels, batch eight), but the notebook
does not run it automatically. Target missing-hole performance remains unsupported.
Do not silently reduce resolution or batch size following an OOM; preserve the
log and choose a new versioned recipe/run directory after review.

## Full source training — experiment 1 (current next step)

The T4 source pilot is verified. Its held-out mAP50 rose from 0.303 to 0.532 to
0.710 over three epochs; the older source baseline remains stronger at 0.908.
The pilot demonstrates that the training setup works, not that the new recipe
will beat the baseline.

Import `notebooks/PCB_Revamp_Phase3_Source_Training_Kaggle.ipynb` into a fresh
Kaggle kernel with the same version-2 dataset, GPU T4 and Internet enabled.
Leave `RESTORE_FROM=None` and `RESUME_ONLY=False` for the initial run. No pilot
checkpoint is required. This experiment starts from COCO YOLOv8s weights with
a new 60-epoch cosine schedule, three warmup epochs and early-stopping patience
15. Warmup bias LR is zero. Resolution 960, batch four, full precision, zero
workers and explicit disabled augmentations retain the verified pilot setup.
Training uses 953 source views; validation uses 120 separate original boards.

Expect roughly three hours on a similar T4 if all 60 epochs execute. The pilot
CSV showed approximately 162 seconds between completed epochs including the
intervening checkpoint work. Speed can change; early stopping can shorten the run.

The pinned Ultralytics version selects best.pt using validation mAP50–95.
The report's predeclared review gate is +1 percentage point mAP50–95 over the
re-evaluated baseline, no more than a 1-point mAP50 drop and no per-class AP50
drop greater than 5 points. This is a review filter, not statistical evidence.
The old baseline stays deployed regardless of that flag until a later review.
No development-test data is used. If the new candidate is worse, retain the old
model and investigate validation errors before proposing another bounded run.

Download `source_run_checkpoint.zip`, `report.json`, `training.log`, and
`pcb_environment.json` into a new folder, separate from pilot downloads.
The checkpoint includes both last and selected-best weights, optimizer, scheduler,
random states, CSV, baseline results and epoch telemetry. Rerun the cell to resume
local progress. For a new session attach this experiment's checkpoint and set
RESTORE_FROM; clear it for later local reruns. Early-stopped and final checkpoints
rebuild evaluation/report outputs without restarting training. Smoke/pilot ZIPs
are rejected because they have a different recipe and code identity.

## Experiment 2 — full-board-only comparison (current next step)

Run `notebooks/PCB_Revamp_Phase3_Full_Boards_Kaggle.ipynb` with the same dataset,
T4 GPU, Internet and a fresh kernel. Do not attach experiment 1's checkpoint.
The sole experiment change is training-view selection: 480 original full boards
instead of 953 mixed full/crop/background views. The 120 validation boards,
COCO initialization, seed, resolution, optimizer, LR schedule, augmentation
settings, batch size, 60-epoch maximum and patience 15 stay unchanged. Because
there are fewer training batches, this is not an equal-compute causal study.

Evidence: experiment 1's best epoch was 9, after which train losses decreased
while validation performance deteriorated. At 960px letterbox scaling the median
defect short edge is 18.35px on full training boards, 81px in training crops and
23.98px on validation boards. Crop scale/context mismatch is a hypothesis; the
experiment tests it rather than assuming it caused the failure. Historical
baseline fine-tuning was already attempted and did not beat that baseline, so
we are not blindly repeating that recovery recipe.

Budget approximately 1.5–2 hours if all epochs run, extrapolated from fewer
training batches rather than measured for this run. Download four outputs into
`downloads/source_full_boards_v2/`: `full_boards_run_checkpoint.zip`, `report.json`,
`training.log`, `pcb_environment.json`. Every completed epoch produces a resumable
checkpoint; signatures prevent importing an incompatible run. Compare against
experiment 1 and the retained baseline. Keep the existing review gate and never
promote automatically. If results fail to improve, inspect validation predictions
before deciding on a separate augmentation experiment or model change. No test
images or external dataset changes are introduced here.
