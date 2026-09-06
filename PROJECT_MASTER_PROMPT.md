# AI Smart Maintenance System – Complete Coding Prompt
## No Cuts. Full Scope. All Objectives Intact.

**Read this prompt at the start of every Claude coding session for this project.**

---

## Context & Reference

This project implements an **AI-driven predictive maintenance system for PCB manufacturing** using domain-adaptive deep learning and edge AI. All architectural details, datasets, hardware specs, and methodology are documented in: **`PCB_Minor_Project_README.md`** (share this file across all sessions).

**Non-Negotiable Requirements**:
- ✅ Problem statement: UNCHANGED
- ✅ All 7 objectives: PRESERVED (including RUL estimation + Raspberry Pi deployment)
- ✅ Public datasets only: PKU-Market-PCB (source) + DsPCBSD+ (target)
- ✅ Hardware: Raspberry Pi 4 (4GB RAM), Pi Camera Module 2
- ✅ Tech stack: YOLOv8 + PyTorch + FastAPI + PostgreSQL + React + ONNX
- ✅ Timeline: 8 weeks, no extensions
- ✅ Deliverables: Code + trained models + API + dashboard + Pi deployment

---

## Phase-by-Phase Coding Requirements

### PHASE 1: Source-Domain YOLOv8 Training (W1–W2)

**Objective**: Train YOLOv8 detector on PKU-Market-PCB to achieve **≥95% mAP**.

#### 1.1 Dataset Preparation

**Input Dataset**: 
- PKU-Market-PCB: 1,386 original images (6 defect classes)
- Classes: `missing_hole`, `mouse_bite`, `open`, `short`, `spur`, `spurious_copper`
- Format: YOLO (txt annotations, image files)

**Tasks**:
1. Download from: https://universe.roboflow.com/jr-mqqnk/pcb-defects-detection-anddl/dataset/1
2. Verify dataset structure (images + labels folders)
3. Create `data.yaml` for YOLOv8:
   ```yaml
   path: /path/to/dataset
   train: images/train
   val: images/val
   test: images/test
   nc: 6
   names: ['missing_hole', 'mouse_bite', 'open', 'short', 'spur', 'spurious_copper']
   ```

**Code to Generate**:
- `scripts/01_download_pku_dataset.py`
  - Download PKU-Market-PCB (free, YOLO format)
  - Verify 1,386 images + 6 classes
  - Output: `data/pku_market/` directory structure
  
- `scripts/02_dataset_stats.py`
  - Count images per class
  - Compute class distribution
  - Visualize annotation samples
  - Output: `reports/dataset_analysis.json` + sample visualizations
  
- `scripts/03_train_test_split.py`
  - Stratified split: 70% train / 15% val / 15% test (by class)
  - Create train/val/test folders
  - Preserve YOLO label format
  - Output: `data/pku_market/train|val|test/` with images + labels

#### 1.2 OpenCV Preprocessing Pipeline

**Code to Generate**:
- `preprocessing/preprocessing.py`
  - Input: raw PKU images + YOLO labels
  - Operations (in sequence):
    1. Load image, verify format (RGB/BGR)
    2. Normalize pixel values: [0–255] → [0–1]
    3. Resize to 640×640 (YOLOv8 standard, preserve aspect ratio + pad)
    4. **Augmentation** (for training only):
       - Random rotation: ±15°
       - Random horizontal flip: 50% probability
       - Brightness adjustment: ±20%
       - Gaussian noise: σ=5
       - Random zoom: 0.8–1.2×
    5. Update YOLO labels (bbox coords) if image is resized
  - Output: augmented dataset + updated YAML config
  - Save to `data/pku_market_processed/`

#### 1.3 YOLOv8 Training

**Code to Generate**:
- `train/train_source_model.py`
  ```python
  from ultralytics import YOLO
  
  def train_source_yolov8():
      model = YOLO('yolov8m.pt')  # Medium model
      results = model.train(
          data='data/pku_market_processed/data.yaml',
          epochs=100,
          imgsz=640,
          batch=16,  # Adjust for GPU memory
          patience=20,  # Early stopping
          device=0,  # GPU (auto-select)
          optimizer='SGD',
          lr0=0.01,
          momentum=0.937,
          weight_decay=0.0005,
          augment=True,
          mosaic=1.0,
          mixup=0.0,
          save_period=10,  # Save checkpoint every 10 epochs
          project='runs/source_training',
          name='yolov8m_pku',
      )
      return results
  
  if __name__ == '__main__':
      results = train_source_yolov8()
  ```
  - **Success Criteria**:
    - mAP@0.5:0.95 ≥ 0.95 on validation set
    - Per-class precision ≥ 0.92
    - Per-class recall ≥ 0.92
    - Training loss converges
  - **Output**:
    - `runs/source_training/yolov8m_pku/weights/best.pt` (best model)
    - `runs/source_training/yolov8m_pku/weights/last.pt` (last checkpoint)
    - `runs/source_training/yolov8m_pku/results.csv` (training metrics)

#### 1.4 Source Model Evaluation

**Code to Generate**:
- `evaluation/eval_source_model.py`
  - Load `best.pt` from source training
  - Run inference on test set (50% holdout images)
  - Compute metrics:
    - mAP@0.5, mAP@0.75, mAP@0.5:0.95
    - Per-class precision, recall, F1
    - Confusion matrix
    - False positive rate, false negative rate
  - Output:
    - `reports/source_model_metrics.json`
    - `reports/confusion_matrix_source.png`
    - `reports/per_class_metrics.csv`

---

### PHASE 2: Domain Adaptation Layer (W3–W4)

**Objective**: Build lightweight adapter + MMD alignment to achieve **85–90% mAP on target domain** with only 50–100 labelled images.

#### 2.1 Target Domain Dataset Preparation

**Input Dataset**:
- DsPCBSD+: 10,259 images (9 defect classes from real manufacturing)
- Multi-layer PCBs (different from PKU's 2-layer synthetic boards)

**Tasks**:
1. Download from: https://www.kaggle.com/datasets/enisteper1/dataset-of-pcb-surface-defects-dspcbsd
   OR Figshare: https://doi.org/10.6084/m9.figshare.24970329
2. Verify 10,259 images + 20,276 annotations (YOLO or COCO format)
3. Map 9 DsPCBSD classes → 6 PKU classes (class alignment)
4. Stratified sample: **50–100 labelled images** (for few-shot fine-tuning)
5. Remaining unlabelled: ~9,100+ images (for semi-supervised alignment, optional)

**Code to Generate**:
- `scripts/04_download_dspcbsd_dataset.py`
  - Download DsPCBSD+ (Kaggle or Figshare)
  - Verify structure (images + annotations)
  - Output: `data/dspcbsd/` directory

- `scripts/05_class_mapping.py`
  - Map 9 DsPCBSD classes → 6 PKU classes (document mapping logic)
  - Create mapping JSON: `data/dspcbsd/class_mapping.json`
  - Output statistics: which DsPCBSD classes map to which PKU classes

- `scripts/06_sample_target_images.py`
  - Stratified sampling: 50–100 labelled images (balanced per class)
  - Remaining images → unlabelled pool
  - Output:
    - `data/dspcbsd/labelled_few_shot/` (50–100 images + labels)
    - `data/dspcbsd/unlabelled/` (remaining images, no labels)
    - `reports/target_sampling_stats.json`

#### 2.2 Adapter Architecture Implementation

**Architecture**:
```
YOLOv8 backbone (Darknet encoder) — FROZEN
    ↓ (2048-D features)
Lightweight Adapter (2 FC layers):
    - FC1: 2048 → 512 (ReLU)
    - BatchNorm
    - Dropout(0.2)
    - FC2: 512 → 128 (ReLU)
    ↓
MMD Alignment Layer (kernel-based)
    ↓ (aligned features)
YOLOv8 Detection Head — FINE-TUNED
```

**Code to Generate**:
- `models/adapter.py`
- `models/mmd_loss.py`

#### 2.3 Domain Adaptation Training

**Code to Generate**:
- `train/train_adapter.py`

#### 2.4 Few-Shot Evaluation

**Code to Generate**:
- `evaluation/eval_adapted_model.py`

---

### PHASE 3: Synthetic RUL Data Generation (W4–W5, Parallel)

**Objective**: Generate temporal equipment degradation sequences linking defect progression → RUL estimation.

#### 3.1 Degradation Model Design

**Code to Generate**:
- `rul_synthesis/degradation_model.py`

#### 3.2 RUL Dataset Export

**Code to Generate**:
- `rul_synthesis/export_rul_data.py`

---

### PHASE 4: Backend API & PostgreSQL (W5–W6)

**Objective**: Build FastAPI inference endpoint + PostgreSQL logging for real-time detection.

#### 4.1 PostgreSQL Database Setup

**Code to Generate**:
- `database/schema.sql`
- `database/db.py`

#### 4.2 FastAPI Inference Service

**Code to Generate**:
- `backend/main.py`

#### 4.3 Docker Configuration

**Code to Generate**:
- `backend/requirements.txt`
- `backend/Dockerfile`

---

### PHASE 5: React Dashboard (W6–W7)

**Objective**: Build real-time dashboard for production teams.

#### 5.1 Dashboard Setup

**Code to Generate**:
- `frontend/package.json`

#### 5.2 Dashboard Components

**Code to Generate**:
- `frontend/src/App.js`
- `frontend/src/App.css`
- `frontend/src/components/LiveFeed.js`
- `frontend/src/components/TrendChart.js`
- `frontend/src/components/AlertPanel.js`
- `frontend/src/components/MetricsPanel.js`
- `frontend/public/index.html`

---

### PHASE 6: Raspberry Pi Edge Deployment (W7–W8)

**Objective**: Deploy quantized model + inference on Pi 4.

#### 6.1 Model Quantization

**Code to Generate**:
- `edge/quantize_model.py`

#### 6.2 Raspberry Pi Setup Script

**Code to Generate**:
- `edge/setup_pi.sh`

#### 6.3 Edge Inference Script

**Code to Generate**:
- `edge/edge_inference.py`
- `edge/requirements_pi.txt`

---

### PHASE 7: Comparative Analysis & Documentation (W7–W8)

**Objective**: Evaluate baseline vs. adapted model, generate final report.

#### 7.1 Comparative Metrics

**Code to Generate**:
- `evaluation/comparative_analysis.py`

#### 7.2 Final Report Template

**Code to Generate**:
- `FINAL_REPORT.md`

---

## Execution Workflow

```
WEEK 1-2: SOURCE TRAINING
├── 01_download_pku_dataset.py ✓
├── 02_dataset_stats.py ✓
├── 03_train_test_split.py ✓
├── preprocessing.py ✓
├── train_source_model.py ✓
└── eval_source_model.py ✓

WEEK 3-4: DOMAIN ADAPTATION
├── 04_download_dspcbsd_dataset.py ✓
├── 05_class_mapping.py ✓
├── 06_sample_target_images.py ✓
├── models/adapter.py ✓
├── models/mmd_loss.py ✓
├── train_adapter.py ✓
└── eval_adapted_model.py ✓

WEEK 4-5: RUL SYNTHESIS (PARALLEL)
├── rul_synthesis/degradation_model.py ✓
└── rul_synthesis/export_rul_data.py ✓

WEEK 5-6: BACKEND & DATABASE
├── database/schema.sql ✓
├── database/db.py ✓
├── backend/main.py ✓
├── backend/requirements.txt ✓
└── backend/Dockerfile ✓

WEEK 6-7: FRONTEND
├── frontend/package.json ✓
├── frontend/src/App.js ✓
├── frontend/src/components/LiveFeed.js ✓
├── frontend/src/components/TrendChart.js ✓
├── frontend/src/components/AlertPanel.js ✓
└── frontend/src/components/MetricsPanel.js ✓

WEEK 7-8: EDGE & FINAL
├── edge/quantize_model.py ✓
├── edge/setup_pi.sh ✓
├── edge/edge_inference.py ✓
├── evaluation/comparative_analysis.py ✓
└── FINAL_REPORT.md ✓
```

---

**Status**: Ready to Code  
**Version**: 1.0  
**Scope**: COMPLETE (No Cuts)

