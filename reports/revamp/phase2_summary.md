# Phase 2 verification — 2026-10-25

Phase 2 is complete locally. The current deliverable is the version-2 Kaggle
package and matching notebook. Full candidate training, Kaggle GPU timings and
new validation accuracy measurements have not been performed.

## Deliverables

- `data/revamp_phase2_v1/`: versioned, parent-traceable prepared views.
- `data/revamp_kaggle_v2/pcb_revamp_phase2.zip`: 992,459,844 bytes.
- `data/revamp_kaggle_v2/PCB_Revamp_Phase2_Kaggle.ipynb`: verification notebook.
- `revamp/prepare.py`, `package.py`, `runtime.py`, `checkpoints.py`, `smoke.py`:
  preparation, review gate, packaging, portable checks and tiny recovery exercise.
- `revamp/README.md`: reproducible commands and Kaggle handoff instructions.

ZIP SHA-256: `7a015b90a74d00d4ac422be404e997ca96ac39c22810423b706a1537aa34cf84`.
The intermediate v1 package remains preserved; use v2.

## Data decisions

Source: 480 full training images, 470 native-resolution positive crops and three
reviewed background crops; 120 unchanged validation images. Ten proposed positive
crops were rejected because they could not preserve complete boxes with an
eight-pixel margin. Source full boards are 50.4% of source-view training samples.

Target: 3,606 training and 403 validation images, with five supported classes and
the canonical six-class output ordering. Target class 0 remains unsupported.
All 1,085 development-test parents are excluded from the package. Validation has
no generated crops. Source and target configurations are separate.

Twenty-four negative candidates contained seven distinct image hashes. All were
visually inspected at native crop size in four contact sheets. Three representatives
were accepted; textured/ambiguous examples and duplicates were excluded. This was
an assistant review, not a human expert certification. Three cropped negatives
cannot establish normal-board performance. Six positive crop examples, covering
all classes, were also inspected; no obvious box displacement was seen.

## Verification

- 49 focused tests passed: audit regression, crop geometry, split/parent/group
  leakage, changed labels, class coverage, unsafe archives, review provenance,
  notebook syntax, checkpoint roundtrip and existing pause/resume behavior.
- Independently recomputed the geometry of every derived positive view against
  the Phase 1 parents; all 5,079 nonnegative views passed parent/split checks.
- Final ZIP passed CRC; all packaged assets passed their recorded SHA-256 hashes.
- Executed the final notebook's ZIP branch locally: discovery, extraction,
  hash checks, image/label verification, writable configurations, environment log
  and all four checkpoint metadata checks. Code loaded from the extracted ZIP.
- Final notebook passed nbformat schema validation. Dependency installation is
  designed to preserve installed torch/torchvision/NumPy via constraints; an actual
  Kaggle install and GPU session still need verification.
- 11,413 Phase 1 artifact/input/prepared-image/label checksums remained unchanged.
- Real tiny learning/resume check passed using the final extracted package:
  pause after epoch one, checksummed ZIP export, old run moved away, new Python
  process/folder, restored optimizer, and epochs 1–80 exactly once in results.csv.

The final tiny run took 130.5 seconds on local CPU (Apple M4, Python 3.11.14,
torch 2.7.1, torchvision 0.22.1, Ultralytics 8.4.142). Its first-five mean box loss
was 3.003762 and last-five mean was 0.051494. It passed the predefined memorization
gate. All evaluation in that check used the same six **training** crops; its scores
are not validation accuracy or generalization evidence. The sample includes only
one source board-template group and is intentionally unsuitable for benchmarking.

## Remaining limits and next phase

Target board independence is unknown. No normal full-board benchmark exists.
Old test partitions remain development comparisons. No external dataset was added.
No production model, frontend, API or camera code was changed. No files from the
existing pipeline were deleted.

Next: run the delivered verification notebook in Kaggle, retain its environment
and smoke outputs, then implement Phase 3's timed GPU pilot, reproduced specialist
validation baselines and bounded source/target training. A mixed-domain experiment
is conditional on specialist results. Camera integration and real-machine RUL
remain outside this training phase.
