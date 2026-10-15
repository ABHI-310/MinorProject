# PCB Smart Maintenance System — Phase 1: Source-Domain YOLOv8 Training

A reproducible, functional prototype pipeline for defect detection on printed circuit boards (PCBs) using YOLOv8n and the PKU-Market-PCB dataset.

---

## 1. Project Overview

Phase 1 establishes a reproducible baseline model trained on the source domain (PKU-Market-PCB dataset) across 6 defect classes:
1. `missing_hole`
2. `mouse_bite`
3. `open`
4. `short`
5. `spur`
6. `spurious_copper`

### Key Features
- **Hardware-friendly:** Configured CPU-first with automatic GPU fallback.
- **Data Integrity:** Strict YOLO annotation schema validation, coordinate bounds checking, and duplicate detection.
- **Multi-Label Stratification:** 70% Train / 15% Val / 15% Test split preserving defect class distributions without data leakage.
- **On-the-fly Augmentation:** Utilizes Ultralytics' internal augmentation pipeline (mosaic, flips, HSV) without disk bloat.
- **Transparent Evaluation:** Separate validation vs. held-out test split metrics, per-class PR curves, and visual error analysis.

---

## 2. Directory Structure

```text
pcb-smart-maintenance/
├── configs/
│   ├── data.yaml                      # YOLO dataset configuration
│   └── train_config.yaml              # Hyperparameters (epochs, batch, patience, etc.)
├── scripts/
│   ├── 01_download_pku_dataset.py     # Dataset download/copy and verification
│   ├── 02_dataset_stats.py            # Statistical analysis & visualization
│   └── 03_train_test_split.py         # Multi-label stratified 70/15/15 split
├── preprocessing/
│   └── preprocessing.py               # Letterboxing & bbox coordinate adjustments
├── train/
│   └── train_source_model.py          # YOLOv8n training pipeline with early stopping
├── evaluation/
│   └── eval_source_model.py           # Test set evaluation & prediction generation
├── utils/
│   └── common.py                      # Shared utilities, logging, label parsing
├── tests/
│   ├── test_label_parsing.py          # Unit tests for YOLO label parsing
│   ├── test_coordinate_transforms.py  # Unit tests for letterbox & bbox math
│   ├── test_dataset_integrity.py      # Unit tests for raw data integrity
│   └── test_split_leakage.py          # Unit tests for leakage & completeness
├── data/                              # Datasets (excluded from Git)
│   ├── pku_market/                    # Raw source dataset
│   └── pku_market_processed/          # Stratified train/val/test splits
├── reports/                           # Output reports, metrics & plots
│   ├── dataset_validation_report.json # Integrity validation results
│   ├── dataset_analysis.json          # Dataset statistics
│   ├── class_distribution.png         # Defect frequency plot
│   ├── bbox_analysis.png              # Box geometry & count distributions
│   ├── split_manifest.json            # Exact image list per split
│   └── sample_images/                 # Rendered ground truth samples
└── requirements.txt                   # Pinned Python dependencies
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.11 installed.

### Setup Virtual Environment
```bash
# Create virtual environment with Python 3.11
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Execution Workflow

Run each stage sequentially:

### Step 1: Acquire & Verify Dataset
To copy raw data from an existing directory:
```bash
python scripts/01_download_pku_dataset.py --local-path "D:/Programming/Minor Project/data/pku_market"
```
Or to download directly from Roboflow (requires `ROBOFLOW_API_KEY` environment variable):
```bash
python scripts/01_download_pku_dataset.py
```

### Step 2: Analyze Dataset Statistics
Generate distribution charts, geometry statistics, and sample ground-truth visualizations:
```bash
python scripts/02_dataset_stats.py
```
Outputs saved to `reports/`.

### Step 3: Stratified Train / Val / Test Split
Create the 70/15/15 multi-label stratified split:
```bash
python scripts/03_train_test_split.py
```
Generates split folders in `data/pku_market_processed/` and logs `reports/split_manifest.json`.

### Step 4: Run Unit Tests
Verify coordinate transformations, label parsing, and zero data leakage:
```bash
pytest tests/
```

### Step 5: Train YOLOv8n Source Model
Start training (reads parameters from `configs/train_config.yaml`):
```bash
python train/train_source_model.py
```
*Optional CLI overrides:*
```bash
python train/train_source_model.py --epochs 30 --batch-size 4 --device cpu
```

### Step 6: Evaluate on Held-Out Test Set
Evaluate the best checkpoint (`runs/source_model/weights/best.pt`) on the test set:
```bash
python evaluation/eval_source_model.py
```
Generates `reports/source_model_metrics.json`, `reports/per_class_metrics.csv`, confusion matrix, and visual detection samples in `reports/prediction_samples/`.

---

## 5. Configuration Reference

- `configs/train_config.yaml`:
  - `model`: YOLO architecture checkpoint (default `yolov8n.pt`)
  - `epochs`: Maximum training epochs (default `50`)
  - `batch_size`: Batch size (default `4` for CPU safety)
  - `patience`: Early stopping patience (default `15` epochs)
  - `image_size`: Input resolution (default `640`)
  - `device`: Compute device (`cpu` or `0` for GPU)
  - `augmentation`: Probabilities for horizontal flip, rotation, scale, HSV, and mosaic.
