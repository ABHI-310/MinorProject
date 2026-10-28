# Implementation status

## Phase 1 — complete, with documented limitations

Completed on 2026-10-23. The new audit entry point is `revamp.audit`.

- Audited 5,694 prepared images and 12,267 boxes against their local originals.
- Source: 693 images, 2,953 boxes; 480/120/93 train/validation/development-test.
- Target: 5,001 images, 9,314 boxes; 3,606/403/992 train/validation/development-test.
- Every prepared image matched original bytes; all labels matched the original
  XML/COCO annotations under the documented geometry tolerance.
- Zero structural/provenance errors; five warnings retained in the generated report.
- Zero exact cross-split duplicate sets, known-group overlaps or within-domain
  cross-split pHash candidates at Hamming distance <=4.
- Both deployed checkpoints matched their metadata and saved evaluation hashes.
  No checkpoint was deserialized, and no new inference metric was measured.
- Four contact sheets / 22 train-validation sample images were visually inspected.
  This bounded assistant review is not expert adjudication of every annotation.
- 18 focused tests passed, including corrupt images, missing/invalid labels,
  displaced boxes, empty-label duplicate leakage, checkpoint mismatch, COCO ID
  scoping, protected output paths and deterministic non-destructive audit output.

Evidence: `reports/revamp/phase1_v1/`. The automated snapshot is immutable; the
subsequent visual review and phase decision are in `reports/revamp/phase1_review.md`.

## Phase 2 — complete; Kaggle functional pass reported

Implemented on 2026-10-25. New entry points: `revamp.prepare`, `revamp.package`,
`revamp.runtime` and the bounded `revamp.smoke` check. Existing app/data/models
remain unchanged.

- Prepared 480 source full-board training images plus 470 safe native defect crops;
  ten parents had no crop meeting the full-box/margin rule.
- Preserved 120 source validation, 3,606 target training and 403 target validation
  images. Excluded all 1,085 development-test parents.
- Reviewed 24 background candidates (seven distinct image hashes); accepted three
  conservative representatives, excluded ambiguous/repeated examples. Review was
  performed by Codex, not certified by a human PCB expert.
- Built separate source-full, source-views and target configurations. Source full
  boards remain approximately half of the source-views training examples.
- Current upload: `artifacts/kaggle/phase2_v2/pcb_revamp_phase2.zip`, about 992 MB, with
  updated `notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb`. Intermediate v1 and the
  original v2 notebook are preserved; the dataset ZIP does not need re-uploading.
- All assets hashed; new-directory-only output; archive path/CRC checks; checkpoint
  backups require their manifest and trainer state, with unique per-run filenames.
- 49 focused tests passed. Independently recomputed crop geometry for all derived
  positive views. Viewed a six-class positive-crop contact sheet.
- Executed the final notebook's ZIP setup, verification, configuration and four
  model metadata checks locally, loading code from the extracted package.
- Verified 11,413 Phase 1 artifact/input/image/label hashes remain unchanged.
- Real six-image learning/fresh-process resume test passed on CPU, including the
  final code extracted from the version-2 ZIP (130.5 seconds). All 80 tiny epochs
  appear exactly once; optimizer state was restored. Scores from this check are
  memorization, never model evaluation results.

Evidence: `reports/revamp/phase2_verification/`, `artifacts/verification/phase2_smoke_v2/report.json`,
`artifacts/verification/phase2_notebook_check/setup_check.json`, `phase2_background_review.json`, and
`artifacts/kaggle/phase2_v2/build_report.json`. See `reports/revamp/phase2_summary.md`.

Remaining constraints are accepted for dataset preparation, not erased:

- Target board-level independence is unknown.
- Missing-hole target examples and normal full-board benchmarks are unavailable.
- Existing test results are development comparisons.
- The Roboflow export remains excluded.
- External datasets remain candidates, not validated inputs.
- Only three background crops are accepted; normal-board performance is unmeasured.
- Kaggle functional success was reported by the user on 2026-10-26. The pasted
  output does not identify CPU/GPU or dependency versions; environment capture
  and full candidate training remain pending.

Phases 3–6 are pending. Next: capture the Kaggle environment, then add a timed
GPU pilot, reproduced validation baselines and bounded specialist training in
Phase 3. See README.md for scope, completion gates and budgets.

## Notebook checkpoint update

Use `notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb` with the unchanged version-2
dataset. It embeds the updated runner, reuses matching setup/configuration files,
saves a complete checkpoint after every epoch, resumes from any saved epoch,
imports ZIP or expanded-folder backups, and skips already completed training.
`RESUME_ONLY` prevents accidental fresh starts. Interrupt stops the process group
and exposes the latest checkpoint download link. Checkpoints must be downloaded
or otherwise persisted before session storage disappears.

57 focused tests pass. The real CPU test passed: interruption after epoch 5,
same-session completion, imported epoch-5 checkpoint advancing in a fresh process,
final-checkpoint report recovery, and completed-run reuse without retraining.
Evidence is recorded in `artifacts/verification/checkpoint_resume_check/report.json`.
The earlier Phase 2 artifacts remain historical snapshots and were not rewritten.

## Kaggle result reported on 2026-10-26

The user supplied a passing 80-epoch, six-training-image check in 234.8 seconds.
The package fingerprint matches the delivered v2 package. Mean box loss decreased
from 3.464408 (first five epochs) to 0.092846 (last five); training-sample AP50 was
0.995. This is memorization evidence, not detection accuracy on unseen PCBs.
The output reports successful checkpoint restoration and complete epoch history.
Checkpoint bytes, runtime versions and device have not been independently checked.
Recorded in `reports/revamp/phase2_kaggle_reported_2026-10-26.json`; no session URL
or access token was retained. Preserve `pcb_environment.json`, the smoke report,
logs and checkpoint before ending the Kaggle session. Do not use the six-image
smoke weights as the initialization for full training.

## Repository organization — 2026-10-26

The active workflow now uses `scripts/manage.py`, packages under `src/`, the
current notebook under `notebooks/`, categorized `data/`, and model/run outputs
under `artifacts/`. The superseded archive was subsequently removed with user
authorization; small audit snapshots and regression helpers were retained in
`docs/history/pre_cleanup_source/` and `tests/legacy_support/`. Downloads was untouched. See `docs/REPOSITORY_LAYOUT.md` and `reports/cleanup/SUMMARY.md`.

88 tests, dashboard build, real-model API checks and a fresh zero-error dataset
audit passed. The uploaded Kaggle ZIP and current notebook are unchanged; the
training phase has not advanced during cleanup.

## Phase 3 pilot implementation — 2026-10-26

The downloaded Kaggle smoke files were independently inspected: Tesla T4, Python
3.13.15, Torch 2.11.0+cu128 and Ultralytics 8.4.142. All checkpoint member hashes
matched, the CSV contained epochs 1–80 exactly once, and optimizer state was
retained in the ZIP. Optional Albumentations defaults appeared in that smoke log.
Those transforms are explicitly disabled and checked in the new pilot.

`revamp.pilot` and `notebooks/PCB_Revamp_Phase3_Pilot_Kaggle.ipynb` implement the
next bounded step: existing source baseline validation, three epochs from COCO
YOLOv8s on 953 training views, and candidate evaluation on the separate 120
validation boards. Report includes per-class metrics, memory and timing.
Atomic checkpoints preserve baseline evidence and timing history; incompatible
imports and concurrent workers are rejected. Full precision avoids depending on
unpersisted AMP scaler state. Existing Phase 2 notebook/package are unchanged.

The actual Kaggle source pilot and longer training are still pending. No pilot
accuracy or GPU runtime is claimed from local functional tests.

Local pilot verification passed: 94 automated tests; real CPU training on tiny
synthetic fixtures interrupted after epoch 1; a fresh process/folder restored the
ZIP and completed epochs 2–3; all CSV/telemetry epochs and checkpoint hashes
verified. Completed-checkpoint report recovery and completed-run reuse also
passed. Notebook embedded source hashes match the maintained runner. Evidence:
`reports/revamp/phase3_pilot_local_verification.json`. These fixture results do
not estimate PCB accuracy or GPU throughput.

## Verified T4 pilot and full source-training handoff

The four pilot files under `downloads/` were inspected. ZIP CRC, all member hashes,
run signature, package/code hashes, candidate hash, baseline record and telemetry
match. Epochs 1–3 occur exactly once. Optional Albumentations transforms are off.
Source candidate validation mAP50/mAP50–95: 0.71047/0.33073; retained baseline:
0.90789/0.47038. Peak allocated memory ~5.7 GiB; approximately 153 seconds per
training+validation epoch, ~162 seconds including intervening checkpoint work.

Full source experiment 1 is implemented in `revamp.train_source` and
`notebooks/PCB_Revamp_Phase3_Source_Training_Kaggle.ipynb`: fresh COCO initialization,
60-epoch maximum, patience 15, warmup 3, same data/resolution/batch/augmentation
settings as the pilot, validation mAP50–95 checkpoint selection and an explicit
review-only regression gate. The completed pilot code/notebook remain unchanged.
Actual full training on Kaggle is pending; no production model was replaced.

## Experiment 1 reviewed; experiment 2 prepared

Downloaded rerun in `downloads/new_report.json`, `source_run_checkpoint.zip`,
`training.log` and `new_pcb_environment.json` passed checkpoint/file hashes,
signature, epoch continuity and environment checks. Best epoch 9; stopped at 24.
Candidate mAP50/mAP50–95 0.82177/0.37091 versus baseline 0.90789/0.47038.
Do not promote. Training losses kept decreasing as validation deteriorated.

Experiment 2 removes derived training views while preserving experiment 1's
recipe. `revamp.train_full_boards` uses `source_full` (480 train, 120 validation)
and distinct run/checkpoint identities. Notebook:
`notebooks/PCB_Revamp_Phase3_Full_Boards_Kaggle.ipynb`. This tests the training-view
mismatch hypothesis; it does not claim crop mixing caused the failure.
GPU execution is pending. No baseline, old runner, notebook or dataset was changed.

## Full-board run result and diagnostics

The new downloads are `full_boards_run_checkpoint.zip`, `report_3.json`,
`training_3.log` and `pcb_environment_3.json`. Their hashes, signature and 25
epochs were verified. Best epoch 10; candidate mAP50/mAP50–95 0.80065/0.36491.
This is below both experiment 1 and the retained baseline. Removing crop views
did not fix aggregate performance in this single-seed comparison.

Added `revamp.diagnose` and an optional inference-only Kaggle notebook. Actual
CPU diagnostics ran on all 120 source validation boards for both candidates
and the retained baseline at fixed confidence .25 and IoU .5. They are error
counts, not mAP or the reports' optimized precision/recall. Results live under
`artifacts/verification/source_v1_diagnosis/` and `source_v2_diagnosis/`.
105 automated tests passed. No further training was started. Historical ZIP
inspection matched the baseline's exact weight hash to its original args: it
used augmentation and nbs=64, unlike the no-augmentation/nbs=4 new experiments.
Those differences need controlled investigation; a causal conclusion is not
established by the current runs.
