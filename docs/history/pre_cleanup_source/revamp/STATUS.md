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
- Current upload: `data/revamp_kaggle_v2/pcb_revamp_phase2.zip`, about 992 MB, with
  updated `revamp/PCB_Revamp_Phase2_Resume_Kaggle.ipynb`. Intermediate v1 and the
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

Evidence: `reports/revamp/phase2_verification/`, `phase2_smoke_v2/report.json`,
`phase2_notebook_check/setup_check.json`, `phase2_background_review.json`, and
`data/revamp_kaggle_v2/build_report.json`. See `reports/revamp/phase2_summary.md`.

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

Use `revamp/PCB_Revamp_Phase2_Resume_Kaggle.ipynb` with the unchanged version-2
dataset. It embeds the updated runner, reuses matching setup/configuration files,
saves a complete checkpoint after every epoch, resumes from any saved epoch,
imports ZIP or expanded-folder backups, and skips already completed training.
`RESUME_ONLY` prevents accidental fresh starts. Interrupt stops the process group
and exposes the latest checkpoint download link. Checkpoints must be downloaded
or otherwise persisted before session storage disappears.

57 focused tests pass. The real CPU test passed: interruption after epoch 5,
same-session completion, imported epoch-5 checkpoint advancing in a fresh process,
final-checkpoint report recovery, and completed-run reuse without retraining.
Evidence is recorded in `reports/revamp/checkpoint_resume_check/report.json`.
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
