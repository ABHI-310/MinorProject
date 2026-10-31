# K500 app candidate

The detector dropdown includes `K500 adapted candidate · 5 labels` alongside the existing source and target models. Selecting K500 resets confidence to 0.25; users may change it for new uploads.

Inference uses 640 pixels, NMS IoU 0.4, rect=False, max_det=300, class-aware NMS, no inference augmentation, and class IDs 1–5. Missing-hole detection is excluded.

The checkpoint SHA-256 is `c05a9c35608076e0b0af158ac4ffcabd954f3adfa6928552387e35e875615f6d`.

Validation on 403 images: mAP50 58.52%, mAP50–95 21.79%. At confidence 0.25, pooled precision is 68.01%, recall 56.70%, and F1 61.84%. Settings were selected on this validation set; these are not final test results. Defect-free-board false alarms and board independence remain unverified.

## Local setup

The binary model is ignored by Git. On a new checkout, obtain the verified candidate ZIP and install it once:

```sh
.venv/bin/python scripts/install_k500_candidate.py downloads/intermediate/pcb_k500_candidate_v1.zip
.venv/bin/python scripts/manage.py app
```

The installer rejects an existing destination and does not replace source or target models. Installed model and evidence live in `artifacts/models/k500/`. The application validates checkpoint hashes before loading. The launcher additionally checks installed K500 weights.

## Verification

```sh
.venv/bin/python -m pytest tests/test_k500_integration.py tests/test_phase4_api.py -q
.venv/bin/python scripts/manage.py app --check
```

A real CPU inference API smoke check also passed using a temporary database and one packaged validation image. This checks integration, not general accuracy.
