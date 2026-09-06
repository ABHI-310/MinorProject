# AI Smart Maintenance System for PCB Manufacturing
## Predictive Failure Detection, Defect-Aware Domain Adaptation, and Equipment RUL Estimation

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Hardware](https://img.shields.io/badge/Edge-Raspberry%20Pi%204-c51a4a.svg)](https://www.raspberrypi.com/)

---

## 📌 Project Overview

This repository houses the complete implementation of the **AI Smart Maintenance System for PCB Manufacturing**, designed to solve equipment calibration drift and reconfiguration latency in modern surface-mount and printed circuit board assembly lines.

- **Source Domain**: PKU-Market-PCB (1,386 images, 6 defect categories, 2-layer synthetic boards)
- **Target Domain**: DsPCBSD+ (10,259 images, 9 defect categories, multi-layer real manufacturing boards)
- **Domain Adaptation**: Custom Defect-Aware MMD (Maximum Mean Discrepancy) with class-importance weighting
- **Remaining Useful Life (RUL)**: Equipment degradation modeling linking temporal defect progression to failure thresholds
- **Backend & Database**: FastAPI real-time REST API with PostgreSQL logging
- **Frontend Dashboard**: React 18 monitoring interface with live defect feed, 24h trend analytics, and health status
- **Edge Deployment**: Quantized INT8 ONNX pipeline on Raspberry Pi 4 (4GB) with Camera Module 2

---

## 📁 Repository Structure

```
PCB-Smart-Maintenance/
├── README.md                                  # Repository overview and quickstart
├── PCB_Minor_Project_README.md                # Full project architecture and execution guide
├── PROJECT_MASTER_PROMPT.md                   # Master prompt reference with 8-week deliverables
├── FINAL_REPORT.md                            # Comprehensive final experimental report
│
├── data/                                      # Datasets & synthetic data directories
│   ├── pku_market/                            # Source dataset (PKU-Market-PCB)
│   ├── pku_market_processed/                  # Preprocessed & augmented source dataset
│   ├── dspcbsd/                               # Target dataset (DsPCBSD+)
│   ├── dspcbsd/labelled_few_shot/             # 50-100 target samples
│   ├── dspcbsd/unlabelled/                    # Remaining target images
│   └── rul_synthetic/                         # Synthetic degradation sequences
│
├── models/
│   ├── adapter.py                             # Lightweight adapter (2 FC layers: 2048->512->128)
│   └── mmd_loss.py                            # Defect-aware class-weighted MMD loss
│
├── preprocessing/
│   ├── preprocessing.py                       # OpenCV normalization & resize (640x640)
│   └── augmentation.py                        # Augmentation (rotation, flip, jitter, noise)
│
├── train/
│   ├── train_source_model.py                  # YOLOv8m source domain training on PKU
│   └── train_adapter.py                       # MMD domain adaptation training
│
├── evaluation/
│   ├── eval_source_model.py                   # Source model evaluation (mAP, confusion matrix)
│   ├── eval_adapted_model.py                  # Adapted model few-shot target evaluation
│   └── comparative_analysis.py                # Source-only vs Adapted benchmark & charts
│
├── scripts/
│   ├── 01_download_pku_dataset.py             # PKU dataset fetch & extraction
│   ├── 02_dataset_stats.py                    # Dataset statistics & class distribution
│   ├── 03_train_test_split.py                 # Stratified 70/15/15 split & data.yaml generator
│   ├── 04_download_dspcbsd_dataset.py         # DsPCBSD+ download script (Kaggle/Figshare)
│   ├── 05_class_mapping.py                    # 9 target classes -> 6 source classes alignment
│   └── 06_sample_target_images.py             # Stratified 50-100 few-shot image sampler
│
├── rul_synthesis/
│   ├── degradation_model.py                   # Exponential decay & defect correlation generator
│   └── export_rul_data.py                     # CSV/JSON exporter for DB and dashboard
│
├── database/
│   ├── schema.sql                             # PostgreSQL DDL tables & indexes
│   └── db.py                                  # SQLAlchemy models & session initialization
│
├── backend/
│   ├── main.py                                # FastAPI app (/detect, /metrics, /adapt, /health)
│   ├── requirements.txt                       # Backend Python dependencies
│   └── Dockerfile                             # Containerization definition
│
├── frontend/
│   ├── package.json                           # React app configuration & scripts
│   ├── public/                                # Static web assets
│   └── src/                                   # React UI components (LiveFeed, Alerts, Charts)
│
├── edge/
│   ├── quantize_model.py                      # ONNX export & INT8 quantization for Pi
│   ├── setup_pi.sh                            # Raspberry Pi 4 OS provisioning script
│   ├── edge_inference.py                      # Picamera2 / OpenCV inference loop & REST client
│   └── requirements_pi.txt                    # Pi lightweight edge dependencies
│
├── weights/                                   # Model weights (.pt, .onnx)
└── reports/                                   # Generated metrics, plots & analysis
```

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Create and activate virtual environment
uv venv --python 3.11 .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/macOS

# Install requirements
pip install -r backend/requirements.txt
```

### 2. Phase 1: Source Domain Training (PKU-Market-PCB)
```bash
# Download and verify dataset
python scripts/01_download_pku_dataset.py

# Compute dataset statistics
python scripts/02_dataset_stats.py

# Generate stratified train/val/test splits
python scripts/03_train_test_split.py

# Run preprocessing and augmentation
python preprocessing/preprocessing.py

# Train YOLOv8m source model
python train/train_source_model.py

# Evaluate source model metrics
python evaluation/eval_source_model.py
```

### 3. Phase 2: Domain Adaptation (DsPCBSD+)
```bash
# Download target dataset
python scripts/04_download_dspcbsd_dataset.py

# Map 9 DsPCBSD defect classes to PKU classes
python scripts/05_class_mapping.py

# Extract 50-100 few-shot labelled images
python scripts/06_sample_target_images.py

# Train domain adapter with defect-aware MMD
python train/train_adapter.py

# Evaluate target domain performance
python evaluation/eval_adapted_model.py
```

### 4. Phase 3: Synthetic RUL Estimation
```bash
# Simulate equipment lifecycles & degradation
python rul_synthesis/degradation_model.py

# Export CSV and JSON for DB & UI
python rul_synthesis/export_rul_data.py
```

### 5. Phase 4: Backend API
```bash
# Initialize DB tables & run FastAPI server
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Phase 5: React Dashboard
```bash
cd frontend
npm install
npm start
```

### 7. Phase 6: Edge Deployment (Raspberry Pi 4)
```bash
# Quantize model to ONNX INT8
python edge/quantize_model.py

# On Raspberry Pi:
bash edge/setup_pi.sh
python edge/edge_inference.py
```

### 8. Phase 7: Comparative Benchmark & Report
```bash
python evaluation/comparative_analysis.py
```

---

## 👥 Contributors & Documentation
- Refer to [`PCB_Minor_Project_README.md`](file:///d:/Programming/Minor%20Project/PCB_Minor_Project_README.md) for background theory and methodology.
- Refer to [`PROJECT_MASTER_PROMPT.md`](file:///d:/Programming/Minor%20Project/PROJECT_MASTER_PROMPT.md) for prompt instructions.
- Refer to [`FINAL_REPORT.md`](file:///d:/Programming/Minor%20Project/FINAL_REPORT.md) for the experimental summary.

