# Model and run artifacts

- `models/source/`, `models/target/`: preserved baseline checkpoint packages.
- `pretrained/`: local COCO initialization files.
- `kaggle/phase2_v2/`: unchanged upload ZIP, expanded payload and build report.
- `verification/`: completed/interrupt-tested tiny runs, checkpoint ZIPs, logs and
  extracted notebook test payloads. These weights are not production candidates.

Use `notebooks/PCB_Revamp_Phase2_Resume_Kaggle.ipynb` with the existing upload ZIP.
New full training belongs under a new `training/<run-id>/` directory. Do not overwrite
baselines or existing packages. This folder is excluded from normal Git staging.
