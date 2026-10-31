# Phase 1 review and decision

Date: 2026-10-23. Reviewer: Codex assistant, visual inspection of generated images.
This record supplements the untouched automated `phase1_v1/` snapshot.

## Evidence inspected

- `phase1_v1/summary.json`, `issues.json`, `SUMMARY.md` and audit tool output.
- `phase1_v1/samples/source_train.jpg`: six classes, six image samples.
- `phase1_v1/samples/source_val.jpg`: six classes, six image samples.
- `phase1_v1/samples/target_train.jpg`: five shared classes, five image samples.
- `phase1_v1/samples/target_val.jpg`: five shared classes, five image samples.

Each row contains a full-image overview and a detail crop around an annotated
instance. All selected sample filenames and image hashes are preserved in
`phase1_v1/review_samples.json`. The automated file correctly records review as
pending at generation time; this supplemental record records the later inspection.
No development-test image gallery was used for this visual review.

## Observations

No obvious systematic box displacement is apparent in these selected samples.
Source crops show the expected hole, conductor-break, connection, notch,
protrusion and residual-copper patterns. Target samples contain much smaller,
sometimes ambiguous conductor-edge features. Agreement with original labels
does not resolve all semantic ambiguity or annotation completeness.

The strong full-board versus close-up appearance/scale difference remains visible.
Do not stretch source boards to square images, interpret enlarging target images
as recovered detail, or use these 22 samples as a whole-dataset quality estimate.
For multi-defect target images, the red boxes include other annotations in the
same image; a row's category names the selected detail instance, not every box.

## Decision

**Ready for Phase 2 training-data preparation with the recorded limitations.**
This is not approval of model accuracy, an independent test claim, or a reason to
start training automatically. No data was relabelled or removed during review.

Proceed with verified PKU source originals and the existing five-shared-class
DsPCBSD+ subset. Preserve missing-hole as source-supported only. Keep the original
splits intact and exclude the known-problematic Roboflow export. Review training
negative candidates before inclusion. Do not manufacture healthy-board or
camera-performance claims. Target board independence is still unknown.

## Validation

Executed `.venv/bin/python -m pytest tests/test_revamp_audit.py -q`: **18 passed**.
Executed `.venv/bin/python -m revamp.audit --output reports/revamp/phase1_v1`:
**exit 0; zero errors, five warnings**. Source and target hashes were verified;
previous metrics were inventoried, not reproduced by inference.

No GPU time, new dataset downloads, model training, app changes or deployment
were used for Phase 1.
