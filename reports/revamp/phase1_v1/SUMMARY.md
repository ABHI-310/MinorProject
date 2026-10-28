# Phase 1 audit

Status: **structural_checks_passed_review_required**. Errors: 0; warnings: 5.

This run inspected existing inputs. It did not train, run inference, repair labels or alter splits.

## Source
- train (training): 480 images; 1922 boxes; 480 original matches.
- val (development_validation): 120 images; 562 boxes; 120 original matches.
- test (development_test): 93 images; 469 boxes; 93 original matches.

## Target
- train (training): 3606 images; 6719 boxes; 3606 original matches.
- val (development_validation): 403 images; 746 boxes; 403 original matches.
- test (development_test): 992 images; 1849 boxes; 992 original matches.

## Findings

- WARNING: development_test — Legacy test has already influenced development; it is not an untouched final test.
- WARNING: negative_coverage — 0 empty annotations; none is certified healthy by this audit. Normal-board specificity is unestablished.
- WARNING: unknown_board_groups — Board-disjointness cannot be established for target images.
- WARNING: development_test — Legacy test has already influenced development; it is not an untouched final test.
- WARNING: negative_coverage — 0 empty annotations; none is certified healthy by this audit. Normal-board specificity is unestablished.

## Required next steps

Review the train/validation contact sheets and any near-duplicate candidates. Resolve blockers before Phase 2. Keep unsupported classes and evaluated test status explicit. No automatic promotion or training follows this audit.
