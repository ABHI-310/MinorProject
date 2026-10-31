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
.venv/bin/python -m revamp.audit --output reports/revamp/phase1_v1
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

The current data upload is **`data/revamp_kaggle_v2/pcb_revamp_phase2.zip`**. Use the
updated **`revamp/PCB_Revamp_Phase2_Resume_Kaggle.ipynb`** with that same ZIP; no data
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
.venv/bin/python -m revamp.prepare --output data/revamp_phase2_new
# Inspect review/candidates_*.jpg and record decisions against candidates.json.
# The existing review applies only to the exact candidates in phase2_v1.
.venv/bin/python -m revamp.package \
  --prepared data/revamp_phase2_v1 \
  --background-review reports/revamp/phase2_background_review.json \
  --output data/revamp_kaggle_new
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
.venv/bin/python -m revamp.build_resume_notebook \
  --package data/revamp_kaggle_v2/package \
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
cd data/revamp_kaggle_v2/package
YOLO_CONFIG_DIR=/tmp/pcb-revamp-settings ../../../.venv/bin/python -m revamp.smoke \
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
