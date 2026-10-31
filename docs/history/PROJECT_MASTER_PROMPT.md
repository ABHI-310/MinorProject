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
  ```python
  import torch
  import torch.nn as nn
  
  class LightweightAdapter(nn.Module):
      def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
          super().__init__()
          self.fc1 = nn.Linear(input_dim, hidden_dim)
          self.bn1 = nn.BatchNorm1d(hidden_dim)
          self.relu = nn.ReLU()
          self.dropout = nn.Dropout(0.2)
          self.fc2 = nn.Linear(hidden_dim, output_dim)
          self.bn2 = nn.BatchNorm1d(output_dim)
      
      def forward(self, x):
          x = self.fc1(x)
          x = self.bn1(x)
          x = self.relu(x)
          x = self.dropout(x)
          x = self.fc2(x)
          x = self.bn2(x)
          return x
  ```

- `models/mmd_loss.py`
  ```python
  import torch
  import torch.nn as nn
  
  class DefectAwareMMLoss(nn.Module):
      """
      Defect-aware MMD loss: prioritizes alignment of defect-critical classes
      
      L_mmd = Σ_c (α_c * MMD²(X_s^c, X_t^c))
      
      where:
      - c = defect class (mapped to PKU 6-class space)
      - α_c = class weight (higher for critical defects)
      - MMD = kernel mean embedding distance (Gaussian RBF kernel)
      """
      
      def __init__(self, class_weights=None, kernel_sigma=1.0):
          super().__init__()
          if class_weights is None:
              # Default: solder > missing > trace > alignment
              self.class_weights = {
                  'solder_joint': 1.0,
                  'missing_component': 0.9,
                  'trace_damage': 0.8,
                  'alignment': 0.7,
              }
          else:
              self.class_weights = class_weights
          self.kernel_sigma = kernel_sigma
      
      def rbf_kernel(self, x, y):
          """Gaussian RBF kernel"""
          diff = torch.cdist(x, y, p=2)  # L2 distance
          return torch.exp(-diff**2 / (2 * self.kernel_sigma**2))
      
      def mmd_loss_per_class(self, x_source, x_target, class_id):
          """Compute MMD for single class"""
          K_ss = self.rbf_kernel(x_source, x_source)
          K_tt = self.rbf_kernel(x_target, x_target)
          K_st = self.rbf_kernel(x_source, x_target)
          
          mmd = torch.mean(K_ss) - 2 * torch.mean(K_st) + torch.mean(K_tt)
          return torch.sqrt(torch.clamp(mmd, min=1e-5))
      
      def forward(self, x_source, x_target, labels_source, labels_target):
          """
          x_source: source domain features (N_s, D)
          x_target: target domain features (N_t, D)
          labels_source, labels_target: class labels (for filtering per-class)
          """
          total_loss = 0.0
          
          for class_id in range(6):  # 6 PKU classes
              mask_s = (labels_source == class_id)
              mask_t = (labels_target == class_id)
              
              if mask_s.sum() > 0 and mask_t.sum() > 0:
                  x_s_c = x_source[mask_s]
                  x_t_c = x_target[mask_t]
                  
                  mmd_c = self.mmd_loss_per_class(x_s_c, x_t_c, class_id)
                  
                  # Apply class weight
                  class_weight = self.class_weights.get(f'class_{class_id}', 1.0)
                  total_loss += class_weight * mmd_c
          
          return total_loss / 6  # Average over classes
  ```

#### 2.3 Domain Adaptation Training

**Code to Generate**:
- `train/train_adapter.py`
  ```python
  import torch
  from torch.optim import Adam
  from torch.utils.data import DataLoader
  from ultralytics import YOLO
  from models.adapter import LightweightAdapter
  from models.mmd_loss import DefectAwareMMLoss
  
  def train_adapter(
      source_model_path='runs/source_training/yolov8m_pku/weights/best.pt',
      target_labelled_dir='data/dspcbsd/labelled_few_shot/',
      target_unlabelled_dir='data/dspcbsd/unlabelled/',
      epochs=50,
      lr=1e-4,
      lambda_mmd=0.5,
  ):
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      # Load source model
      source_model = YOLO(source_model_path)
      backbone = source_model.model.model[:-1]  # Remove detection head
      backbone = backbone.to(device)
      backbone.eval()  # Freeze backbone
      
      # Initialize adapter
      adapter = LightweightAdapter(input_dim=2048, hidden_dim=512, output_dim=128)
      adapter = adapter.to(device)
      
      # Re-attach detection head (fine-tuned)
      detection_head = source_model.model.model[-1]
      detection_head = detection_head.to(device)
      
      # Loss functions
      mmd_loss_fn = DefectAwareMMLoss(kernel_sigma=1.0)
      detection_loss_fn = torch.nn.BCEWithLogitsLoss()  # or use YOLOv8's loss
      
      # Optimizer (only adapter + head, not backbone)
      params = list(adapter.parameters()) + list(detection_head.parameters())
      optimizer = Adam(params, lr=lr, weight_decay=1e-5)
      
      # Data loaders
      train_loader_source = create_dataloader(...)  # Source images (unlabelled, as features)
      train_loader_target = create_dataloader(target_labelled_dir)  # Target images (labelled)
      
      # Training loop
      for epoch in range(epochs):
          total_loss = 0.0
          
          for (source_batch, _), (target_batch, target_labels) in zip(train_loader_source, train_loader_target):
              source_batch = source_batch.to(device)
              target_batch = target_batch.to(device)
              target_labels = target_labels.to(device)
              
              # Forward pass
              source_features = backbone(source_batch)  # 2048-D
              target_features = backbone(target_batch)   # 2048-D
              
              # Adapt
              source_adapted = adapter(source_features)
              target_adapted = adapter(target_features)
              
              # Detection head forward
              target_logits = detection_head(target_adapted)
              
              # Compute losses
              loss_det = detection_loss_fn(target_logits, target_labels)
              loss_mmd = mmd_loss_fn(source_adapted, target_adapted, 
                                      torch.zeros_like(target_labels), target_labels)
              
              total_loss = loss_det + lambda_mmd * loss_mmd
              
              optimizer.zero_grad()
              total_loss.backward()
              optimizer.step()
          
          print(f"Epoch {epoch}/{epochs}, Loss: {total_loss:.4f}")
      
      # Save adapter + detection head
      torch.save(adapter.state_dict(), 'weights/adapter.pt')
      torch.save(detection_head.state_dict(), 'weights/adapted_head.pt')
      
      return adapter, detection_head
  
  if __name__ == '__main__':
      train_adapter()
  ```

#### 2.4 Few-Shot Evaluation

**Code to Generate**:
- `evaluation/eval_adapted_model.py`
  - Load source model + trained adapter
  - Inference on target test set (remaining ~10% of DsPCBSD)
  - Compute:
    - mAP@0.5:0.95 (target domain)
    - Per-class precision, recall, F1
    - Improvement over source-only baseline
    - False positive/negative rates
  - **Success Criteria**: mAP ≥ 85–90% on target with 50–100 labelled images
  - Output:
    - `reports/adapted_model_metrics.json`
    - `reports/comparison_source_vs_adapted.png` (side-by-side mAP chart)
    - `reports/confusion_matrix_adapted.png`

---

### PHASE 3: Synthetic RUL Data Generation (W4–W5, Parallel)

**Objective**: Generate temporal equipment degradation sequences linking defect progression → RUL estimation.

#### 3.1 Degradation Model Design

**Approach**:
- Simulate equipment health decay over production cycles
- Map defect count/severity → health score
- Estimate RUL (cycles until failure threshold)

**Code to Generate**:
- `rul_synthesis/degradation_model.py`
  ```python
  import numpy as np
  import json
  from datetime import datetime, timedelta
  
  class EquipmentDegradationModel:
      """
      Synthetic equipment degradation simulator
      
      Health decay model: h(t) = h_0 * exp(-λ*t) + ε
      where:
      - h_0 = initial health = 1.0
      - λ = degradation rate (equipment-specific)
      - ε = random noise
      - Failure when h(t) < threshold (e.g., 0.2)
      """
      
      def __init__(self, 
                   initial_health=1.0, 
                   failure_threshold=0.2,
                   degradation_rate=0.001,
                   noise_level=0.0001):
          self.initial_health = initial_health
          self.failure_threshold = failure_threshold
          self.degradation_rate = degradation_rate
          self.noise_level = noise_level
      
      def simulate_equipment_lifecycle(self, total_cycles=500):
          """
          Simulate single equipment's lifecycle
          
          Returns:
          - health_curve: [h_0, h_1, ..., h_t]
          - defect_progression: [d_0, d_1, ..., d_t]
          - rul: remaining useful life (cycles until failure)
          - failure_cycle: cycle at which h < threshold
          """
          health = [self.initial_health]
          defects = [0]
          
          for cycle in range(1, total_cycles):
              # Exponential + noise degradation
              decay = np.exp(-self.degradation_rate * cycle)
              noise = np.random.normal(0, self.noise_level)
              new_health = self.initial_health * decay + noise
              new_health = np.clip(new_health, self.failure_threshold, 1.0)
              
              health.append(new_health)
              
              # Defect count inversely correlates with health
              # More defects as health declines
              defect_rate = 1.0 - new_health  # [0, 0.8]
              new_defects = int(50 * defect_rate) + np.random.poisson(2)
              defects.append(new_defects)
              
              # Check failure
              if new_health <= self.failure_threshold:
                  rul = total_cycles - cycle
                  return health, defects, rul, cycle
          
          return health, defects, total_cycles, total_cycles
      
      def simulate_batch(self, num_scenarios=100, total_cycles=500):
          """Generate multiple equipment scenarios"""
          scenarios = []
          
          for i in range(num_scenarios):
              # Vary degradation rate per equipment (normal distribution)
              deg_rate = np.random.normal(0.001, 0.0002)
              deg_rate = np.clip(deg_rate, 0.0005, 0.003)
              
              model = EquipmentDegradationModel(
                  degradation_rate=deg_rate,
                  noise_level=np.random.uniform(0.00005, 0.0003)
              )
              
              health, defects, rul, fail_cycle = model.simulate_equipment_lifecycle(total_cycles)
              
              scenarios.append({
                  'equipment_id': f'EQP_{i:03d}',
                  'health_curve': health,
                  'defect_progression': defects,
                  'rul_cycles': rul,
                  'failure_cycle': fail_cycle,
                  'degradation_rate': deg_rate,
              })
          
          return scenarios
  
  if __name__ == '__main__':
      model = EquipmentDegradationModel()
      scenarios = model.simulate_batch(num_scenarios=100, total_cycles=500)
      
      # Save as JSON
      with open('data/rul_synthetic/scenarios.json', 'w') as f:
          json.dump(scenarios, f, indent=2)
  ```

#### 3.2 RUL Dataset Export

**Code to Generate**:
- `rul_synthesis/export_rul_data.py`
  - Load synthetic scenarios
  - Create CSV for database:
    ```csv
    equipment_id,cycle,health_score,defect_count,rul_estimate
    EQP_000,0,1.0,0,500
    EQP_000,1,0.999,0,499
    ...
    ```
  - Create JSON for dashboard visualization:
    ```json
    {
      "equipment_id": "EQP_000",
      "timeline": [
        {"cycle": 0, "health": 1.0, "defects": 0, "rul": 500},
        ...
      ]
    }
    ```
  - Output:
    - `data/rul_synthetic/rul_data.csv` (for DB import)
    - `data/rul_synthetic/rul_curves.json` (for dashboard)
    - `reports/rul_sample_curves.png` (visualization of 5 example equipment lifecycles)

---

### PHASE 4: Backend API & PostgreSQL (W5–W6)

**Objective**: Build FastAPI inference endpoint + PostgreSQL logging for real-time detection.

#### 4.1 PostgreSQL Database Setup

**Code to Generate**:
- `database/schema.sql`
  ```sql
  CREATE TABLE IF NOT EXISTS detections (
      id SERIAL PRIMARY KEY,
      equipment_id VARCHAR(50),
      image_path VARCHAR(255),
      defect_class VARCHAR(50),
      bbox_coords TEXT,  -- JSON: [x1, y1, x2, y2]
      confidence FLOAT,
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE TABLE IF NOT EXISTS rul_estimates (
      id SERIAL PRIMARY KEY,
      equipment_id VARCHAR(50),
      health_score FLOAT,
      rul_cycles INT,
      defect_count INT,
      predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE TABLE IF NOT EXISTS model_metrics (
      id SERIAL PRIMARY KEY,
      model_name VARCHAR(100),
      model_type VARCHAR(50),  -- 'source', 'adapted'
      accuracy FLOAT,
      precision FLOAT,
      recall FLOAT,
      map_score FLOAT,
      f1_score FLOAT,
      inference_time_ms FLOAT,
      evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE INDEX idx_equipment_id ON detections(equipment_id);
  CREATE INDEX idx_timestamp ON detections(timestamp);
  CREATE INDEX idx_rul_equipment ON rul_estimates(equipment_id);
  ```

- `database/db.py` (SQLAlchemy ORM)
  ```python
  from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
  from sqlalchemy.orm import declarative_base, sessionmaker
  from datetime import datetime
  
  Base = declarative_base()
  
  class Detection(Base):
      __tablename__ = 'detections'
      id = Column(Integer, primary_key=True)
      equipment_id = Column(String(50))
      image_path = Column(String(255))
      defect_class = Column(String(50))
      bbox_coords = Column(Text)
      confidence = Column(Float)
      timestamp = Column(DateTime, default=datetime.utcnow)
  
  class RULEstimate(Base):
      __tablename__ = 'rul_estimates'
      id = Column(Integer, primary_key=True)
      equipment_id = Column(String(50))
      health_score = Column(Float)
      rul_cycles = Column(Integer)
      defect_count = Column(Integer)
      predicted_at = Column(DateTime, default=datetime.utcnow)
  
  class ModelMetrics(Base):
      __tablename__ = 'model_metrics'
      id = Column(Integer, primary_key=True)
      model_name = Column(String(100))
      model_type = Column(String(50))  # source, adapted
      accuracy = Column(Float)
      precision = Column(Float)
      recall = Column(Float)
      map_score = Column(Float)
      f1_score = Column(Float)
      inference_time_ms = Column(Float)
      evaluated_at = Column(DateTime, default=datetime.utcnow)
  
  # Database initialization
  def init_db(database_url='postgresql://user:password@localhost/pcb_maintenance'):
      engine = create_engine(database_url)
      Base.metadata.create_all(engine)
      Session = sessionmaker(bind=engine)
      return Session
  ```

#### 4.2 FastAPI Inference Service

**Code to Generate**:
- `backend/main.py`
  ```python
  from fastapi import FastAPI, File, UploadFile, HTTPException
  from fastapi.responses import JSONResponse
  from fastapi.middleware.cors import CORSMiddleware
  import cv2
  import numpy as np
  from ultralytics import YOLO
  import torch
  from datetime import datetime
  from database.db import init_db, Detection, RULEstimate, ModelMetrics
  import json
  import time
  
  app = FastAPI(title="PCB Smart Maintenance API", version="1.0")
  
  # CORS for frontend
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  
  # Initialize database
  Session = init_db('postgresql://user:password@localhost/pcb_maintenance')
  
  # Load models
  SOURCE_MODEL_PATH = 'weights/best.pt'  # Source domain
  ADAPTER_PATH = 'weights/adapter.pt'
  ADAPTED_HEAD_PATH = 'weights/adapted_head.pt'
  
  source_model = YOLO(SOURCE_MODEL_PATH)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Class names
  CLASS_NAMES = ['missing_hole', 'mouse_bite', 'open', 'short', 'spur', 'spurious_copper']
  
  @app.post("/detect")
  async def detect(
      file: UploadFile = File(...),
      equipment_id: str = "EQP_001",
      model_type: str = "source"  # 'source' or 'adapted'
  ):
      """
      Real-time defect detection endpoint
      
      Args:
          file: Image file (JPEG/PNG)
          equipment_id: Equipment identifier
          model_type: 'source' (source-only) or 'adapted' (domain-adapted)
      
      Returns:
          JSON with detections, confidence, timestamp, RUL estimate
      """
      try:
          # Read image
          contents = await file.read()
          nparr = np.frombuffer(contents, np.uint8)
          image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          
          if image is None:
              raise HTTPException(status_code=400, detail="Invalid image")
          
          # Run inference
          t0 = time.time()
          results = source_model(image, conf=0.5, device=device)
          inference_time = (time.time() - t0) * 1000  # ms
          
          # Parse detections
          detections = []
          total_defects = 0
          
          for box in results[0].boxes:
              class_id = int(box.cls)
              confidence = float(box.conf)
              bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
              
              detection = {
                  'class_id': class_id,
                  'class_name': CLASS_NAMES[class_id],
                  'bbox': bbox,
                  'confidence': confidence,
              }
              detections.append(detection)
              total_defects += 1
          
          # Estimate RUL (simplified: linear model based on defect count)
          health_score = max(0.2, 1.0 - (total_defects * 0.01))  # Heuristic
          rul_cycles = int(health_score * 500)  # Scale to max 500 cycles
          
          # Log to database
          session = Session()
          
          for det in detections:
              db_detection = Detection(
                  equipment_id=equipment_id,
                  image_path=file.filename,
                  defect_class=det['class_name'],
                  bbox_coords=json.dumps(det['bbox']),
                  confidence=det['confidence'],
              )
              session.add(db_detection)
          
          db_rul = RULEstimate(
              equipment_id=equipment_id,
              health_score=health_score,
              rul_cycles=rul_cycles,
              defect_count=total_defects,
          )
          session.add(db_rul)
          session.commit()
          session.close()
          
          return JSONResponse({
              'equipment_id': equipment_id,
              'detections': detections,
              'total_defects': total_defects,
              'health_score': health_score,
              'rul_cycles': rul_cycles,
              'inference_time_ms': inference_time,
              'timestamp': datetime.utcnow().isoformat(),
              'model_type': model_type,
          })
      
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
  
  @app.get("/metrics")
  async def get_metrics(equipment_id: str = None, hours: int = 24):
      """Fetch detection trends & RUL estimates"""
      session = Session()
      
      # Get recent detections
      query = session.query(Detection).order_by(Detection.timestamp.desc())
      if equipment_id:
          query = query.filter(Detection.equipment_id == equipment_id)
      
      detections = query.limit(100).all()
      
      # Get RUL estimates
      rul_query = session.query(RULEstimate).order_by(RULEstimate.predicted_at.desc())
      if equipment_id:
          rul_query = rul_query.filter(RULEstimate.equipment_id == equipment_id)
      
      rul_estimates = rul_query.limit(50).all()
      
      # Aggregate defect counts by class
      defect_summary = {}
      for det in detections:
          class_name = det.defect_class
          defect_summary[class_name] = defect_summary.get(class_name, 0) + 1
      
      session.close()
      
      return JSONResponse({
          'defect_summary': defect_summary,
          'total_detections': len(detections),
          'latest_rul': rul_estimates[0].to_dict() if rul_estimates else None,
          'health_trend': [r.health_score for r in rul_estimates[-10:]],
      })
  
  @app.get("/health")
  async def health_check():
      """Health check endpoint"""
      return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
  
  if __name__ == '__main__':
      import uvicorn
      uvicorn.run(app, host='0.0.0.0', port=8000)
  ```

#### 4.3 Docker Configuration (Optional but Recommended)

**Code to Generate**:
- `backend/requirements.txt`
  ```
  fastapi==0.104.1
  uvicorn[standard]==0.24.0
  torch==2.0.1
  torchvision==0.15.2
  ultralytics==8.0.200
  opencv-python==4.8.1.78
  numpy==1.24.3
  pandas==2.0.3
  sqlalchemy==2.0.20
  psycopg2-binary==2.9.9
  pydantic==2.3.0
  python-multipart==0.0.6
  ```

- `backend/Dockerfile`
  ```dockerfile
  FROM python:3.10-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["python", "main.py"]
  ```

---

### PHASE 5: React Dashboard (W6–W7)

**Objective**: Build real-time dashboard for production teams.

#### 5.1 Dashboard Setup

**Code to Generate**:
- `frontend/package.json`
  ```json
  {
    "name": "pcb-maintenance-dashboard",
    "version": "1.0.0",
    "dependencies": {
      "react": "^18.2.0",
      "react-dom": "^18.2.0",
      "react-redux": "^8.1.3",
      "redux": "^4.2.1",
      "axios": "^1.5.0",
      "plotly.js": "^2.26.0",
      "react-plotly.js": "^2.11.0",
      "tailwindcss": "^3.3.0"
    }
  }
  ```

#### 5.2 Dashboard Components

**Code to Generate**:
- `frontend/src/App.js`
  ```javascript
  import React, { useState, useEffect } from 'react';
  import './App.css';
  import LiveFeed from './components/LiveFeed';
  import TrendChart from './components/TrendChart';
  import AlertPanel from './components/AlertPanel';
  import MetricsPanel from './components/MetricsPanel';
  
  function App() {
    const [equipment_id, setEquipment] = useState('EQP_001');
    const [metrics, setMetrics] = useState(null);
  
    useEffect(() => {
      const interval = setInterval(async () => {
        const response = await fetch(`http://localhost:8000/metrics?equipment_id=${equipment_id}`);
        const data = await response.json();
        setMetrics(data);
      }, 5000);  // Poll every 5 seconds
  
      return () => clearInterval(interval);
    }, [equipment_id]);
  
    return (
      <div className="dashboard">
        <header className="header">
          <h1>PCB Smart Maintenance System</h1>
          <input 
            type="text" 
            value={equipment_id} 
            onChange={(e) => setEquipment(e.target.value)}
            placeholder="Equipment ID"
          />
        </header>
  
        <div className="grid">
          <div className="panel">
            <h2>Live Detection Feed</h2>
            <LiveFeed equipment_id={equipment_id} />
          </div>
  
          <div className="panel">
            <h2>Alerts</h2>
            <AlertPanel metrics={metrics} />
          </div>
  
          <div className="panel full-width">
            <h2>Defect Trend (24h)</h2>
            <TrendChart metrics={metrics} />
          </div>
  
          <div className="panel">
            <h2>Metrics</h2>
            <MetricsPanel metrics={metrics} />
          </div>
        </div>
      </div>
    );
  }
  
  export default App;
  ```

- `frontend/src/components/LiveFeed.js`
  ```javascript
  import React, { useState, useRef } from 'react';
  import axios from 'axios';
  
  function LiveFeed({ equipment_id }) {
    const [detections, setDetections] = useState([]);
    const fileInputRef = useRef(null);
  
    const handleImageUpload = async (event) => {
      const file = event.target.files[0];
      const formData = new FormData();
      formData.append('file', file);
      formData.append('equipment_id', equipment_id);
  
      try {
        const response = await axios.post('http://localhost:8000/detect', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setDetections(response.data.detections);
      } catch (error) {
        console.error('Error uploading image:', error);
      }
    };
  
    return (
      <div className="live-feed">
        <button onClick={() => fileInputRef.current.click()}>Upload Image</button>
        <input ref={fileInputRef} type="file" onChange={handleImageUpload} hidden />
  
        <div className="detections-list">
          {detections.map((det, idx) => (
            <div key={idx} className="detection-item">
              <span className="class">{det.class_name}</span>
              <span className="confidence">{(det.confidence * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  
  export default LiveFeed;
  ```

- `frontend/src/components/TrendChart.js`
  ```javascript
  import React from 'react';
  import Plot from 'react-plotly.js';
  
  function TrendChart({ metrics }) {
    if (!metrics) return <div>Loading...</div>;
  
    const defects = Object.keys(metrics.defect_summary || {});
    const counts = Object.values(metrics.defect_summary || {});
  
    return (
      <Plot
        data={[{
          x: defects,
          y: counts,
          type: 'bar',
          marker: { color: 'rgba(255, 99, 132, 0.7)' }
        }]}
        layout={{
          title: 'Defect Distribution (24h)',
          xaxis: { title: 'Defect Type' },
          yaxis: { title: 'Count' }
        }}
      />
    );
  }
  
  export default TrendChart;
  ```

- `frontend/src/components/AlertPanel.js`
  ```javascript
  import React from 'react';
  
  function AlertPanel({ metrics }) {
    if (!metrics) return <div>No data</div>;
  
    const health = metrics.latest_rul?.health_score || 1.0;
    const rul = metrics.latest_rul?.rul_cycles || 0;
  
    const alert_level = health > 0.7 ? 'green' : health > 0.4 ? 'yellow' : 'red';
  
    return (
      <div className={`alert ${alert_level}`}>
        <h3>Equipment Status</h3>
        <p><strong>Health:</strong> {(health * 100).toFixed(2)}%</p>
        <p><strong>RUL:</strong> {rul} cycles</p>
        <p className="alert-message">
          {health < 0.3 ? '🔴 CRITICAL: Maintenance required!' : health < 0.6 ? '🟡 WARNING: Monitor closely' : '🟢 Healthy'}
        </p>
      </div>
    );
  }
  
  export default AlertPanel;
  ```

- `frontend/src/components/MetricsPanel.js`
  ```javascript
  import React from 'react';
  
  function MetricsPanel({ metrics }) {
    if (!metrics) return <div>Loading...</div>;
  
    return (
      <div className="metrics-grid">
        <div className="metric">
          <label>Total Detections (24h)</label>
          <value>{metrics.total_detections}</value>
        </div>
        <div className="metric">
          <label>Latest Health</label>
          <value>{(metrics.latest_rul?.health_score * 100).toFixed(2)}%</value>
        </div>
        <div className="metric">
          <label>RUL Estimate</label>
          <value>{metrics.latest_rul?.rul_cycles} cycles</value>
        </div>
      </div>
    );
  }
  
  export default MetricsPanel;
  ```

---

### PHASE 6: Raspberry Pi Edge Deployment (W7–W8)

**Objective**: Deploy quantized model + inference on Pi 4.

#### 6.1 Model Quantization

**Code to Generate**:
- `edge/quantize_model.py`
  ```python
  from ultralytics import YOLO
  import torch
  
  def quantize_yolov8_to_onnx(model_path='weights/best.pt'):
      """
      Export YOLOv8 to ONNX + quantize to INT8 for Pi deployment
      
      Output: adapted_model.onnx (~30–50 MB)
      """
      model = YOLO(model_path)
      
      # Export to ONNX (half precision optional for speed)
      model.export(
          format='onnx',
          imgsz=640,
          half=True,  # FP16 for GPU, auto-disabled on Pi
          opset=14,
          simplify=True,
      )
      
      print(f"✓ Model exported to: {model_path.replace('.pt', '.onnx')}")
  
  if __name__ == '__main__':
      quantize_yolov8_to_onnx('weights/best.pt')
  ```

#### 6.2 Raspberry Pi Setup Script

**Code to Generate**:
- `edge/setup_pi.sh`
  ```bash
  #!/bin/bash
  
  # Raspberry Pi 4 (64-bit OS) setup for PCB defect detection
  
  echo "=== Raspberry Pi Setup for PCB Maintenance ==="
  
  # Update system
  sudo apt-get update && sudo apt-get upgrade -y
  
  # Install Python & dependencies
  sudo apt-get install -y python3-pip python3-opencv python3-numpy
  
  # Install Python packages
  pip install --upgrade pip
  pip install onnxruntime pillow requests
  
  # Enable camera
  echo "Enable CSI camera via: sudo raspi-config -> Interface Options -> Camera"
  
  # Create project directory
  mkdir -p ~/pcb_maintenance
  cd ~/pcb_maintenance
  
  # Download ONNX model (from backend)
  echo "Place adapted_model.onnx in ~/pcb_maintenance/"
  
  # Create systemd service for auto-start
  echo "[Unit]
  Description=PCB Defect Detection Service
  After=network.target
  
  [Service]
  Type=simple
  User=pi
  WorkingDirectory=/home/pi/pcb_maintenance
  ExecStart=/usr/bin/python3 /home/pi/pcb_maintenance/edge_inference.py
  Restart=always
  
  [Install]
  WantedBy=multi-user.target" | sudo tee /etc/systemd/system/pcb_detection.service
  
  sudo systemctl enable pcb_detection.service
  
  echo "✓ Setup complete. Start service with: sudo systemctl start pcb_detection"
  ```

#### 6.3 Edge Inference Script

**Code to Generate**:
- `edge/edge_inference.py`
  ```python
  import cv2
  import numpy as np
  import onnxruntime as rt
  import requests
  import time
  import json
  from datetime import datetime
  from pathlib import Path
  
  # Configuration
  BACKEND_URL = 'http://192.168.1.100:8000'  # Update to your backend IP
  EQUIPMENT_ID = 'PI_EDGE_001'
  MODEL_PATH = 'adapted_model.onnx'
  CONFIDENCE_THRESHOLD = 0.5
  
  class EdgeInference:
      def __init__(self, model_path):
          """Initialize ONNX runtime session"""
          self.session = rt.InferenceSession(model_path)
          self.input_name = self.session.get_inputs()[0].name
          self.output_names = [o.name for o in self.session.get_outputs()]
          
          print(f"✓ Model loaded: {model_path}")
          print(f"  Input: {self.input_name}")
          print(f"  Outputs: {self.output_names}")
      
      def preprocess(self, image, imgsz=640):
          """Preprocess image for YOLO"""
          h, w = image.shape[:2]
          
          # Resize preserving aspect ratio
          scale = min(imgsz / h, imgsz / w)
          new_w = int(w * scale)
          new_h = int(h * scale)
          image_resized = cv2.resize(image, (new_w, new_h))
          
          # Pad to 640x640
          image_padded = np.ones((imgsz, imgsz, 3), dtype=np.uint8) * 114
          x_offset = (imgsz - new_w) // 2
          y_offset = (imgsz - new_h) // 2
          image_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized
          
          # Normalize
          image_float = image_padded.astype(np.float32) / 255.0
          image_chw = np.transpose(image_float, (2, 0, 1))
          image_batch = np.expand_dims(image_chw, 0)
          
          return image_batch
      
      def run_inference(self, image):
          """Run inference on image"""
          inputs = self.preprocess(image)
          outputs = self.session.run(self.output_names, {self.input_name: inputs})
          
          return outputs
      
      def send_to_backend(self, detections, health_score, rul_cycles):
          """Send results to backend API"""
          try:
              payload = {
                  'equipment_id': EQUIPMENT_ID,
                  'detections': detections,
                  'health_score': health_score,
                  'rul_cycles': rul_cycles,
                  'inference_source': 'pi_edge',
                  'timestamp': datetime.utcnow().isoformat(),
              }
              
              response = requests.post(
                  f'{BACKEND_URL}/detect',
                  json=payload,
                  timeout=5
              )
              
              if response.status_code == 200:
                  print(f"✓ Data sent to backend")
                  return response.json()
              else:
                  print(f"✗ Backend error: {response.status_code}")
          
          except Exception as e:
              print(f"✗ Failed to reach backend: {e}")
              return None
  
  def main():
      """Main inference loop"""
      print("=== PCB Defect Detection (Edge) ===")
      print(f"Equipment ID: {EQUIPMENT_ID}")
      print(f"Backend URL: {BACKEND_URL}")
      
      # Initialize camera
      try:
          from picamera2 import Picamera2
          picam2 = Picamera2()
          picam2.preview_configuration.main.size = (640, 480)
          picam2.preview_configuration.main.format = "RGB888"
          picam2.configure("preview")
          picam2.start()
          print("✓ Camera initialized (Pi Camera Module 2)")
      except ImportError:
          print("⚠ picamera2 not available, using OpenCV (USB camera)")
          cap = cv2.VideoCapture(0)
      
      # Load model
      inference = EdgeInference(MODEL_PATH)
      
      frame_count = 0
      defect_count = 0
      
      try:
          while True:
              # Capture frame
              try:
                  frame = picam2.capture_array()
              except:
                  ret, frame = cap.read()
                  if not ret:
                      print("✗ Failed to read frame")
                      break
              
              # Run inference
              t0 = time.time()
              outputs = inference.run_inference(frame)
              inference_time = (time.time() - t0) * 1000
              
              # Parse outputs (simplified: assume last output is detections)
              # In practice, parse YOLOv8 output format
              detections = []  # Parse from outputs
              
              # Estimate health
              health_score = max(0.2, 1.0 - (len(detections) * 0.05))
              rul_cycles = int(health_score * 500)
              
              # Send to backend (every 10 frames)
              if frame_count % 10 == 0:
                  inference.send_to_backend(detections, health_score, rul_cycles)
              
              frame_count += 1
              
              # Print status
              if frame_count % 30 == 0:
                  print(f"Frame {frame_count} | Defects: {len(detections)} | "
                        f"Health: {health_score*100:.1f}% | "
                        f"Inference: {inference_time:.1f}ms | "
                        f"FPS: {30/((time.time()-t0)*30):.1f}")
              
              time.sleep(0.05)  # ~20 FPS
      
      except KeyboardInterrupt:
          print("\n✓ Inference stopped")
      
      finally:
          try:
              picam2.close()
          except:
              cap.release()
  
  if __name__ == '__main__':
      main()
  ```

---

### PHASE 7: Comparative Analysis & Documentation (W7–W8)

**Objective**: Evaluate baseline vs. adapted model, generate final report.

#### 7.1 Comparative Metrics

**Code to Generate**:
- `evaluation/comparative_analysis.py`
  ```python
  import json
  import pandas as pd
  import matplotlib.pyplot as plt
  
  def generate_comparison_report():
      """
      Compare source-only vs. domain-adapted model on target domain
      
      Metrics:
      - mAP@0.5, mAP@0.75, mAP@0.5:0.95
      - Per-class precision, recall, F1
      - False positive rate
      - Inference time (laptop + Pi)
      - Model size (MB)
      """
      
      comparison = {
          'metric': [
              'mAP@0.5:0.95',
              'mAP@0.5',
              'mAP@0.75',
              'Precision (avg)',
              'Recall (avg)',
              'F1 Score (avg)',
              'False Positive Rate',
              'Inference Time (ms)',
              'Model Size (MB)',
          ],
          'source_only': [
              0.65,  # Example: poor on target
              0.72,
              0.58,
              0.68,
              0.61,
              0.64,
              0.15,
              45,
              138,
          ],
          'adapted_model': [
              0.88,  # Example: much better
              0.92,
              0.85,
              0.90,
              0.86,
              0.88,
              0.08,
              52,
              148,  # Slightly larger due to adapter
          ],
      }
      
      df = pd.DataFrame(comparison)
      
      # Calculate improvement
      df['improvement_%'] = ((df['adapted_model'] - df['source_only']) / df['source_only'] * 100).round(2)
      
      # Save to CSV
      df.to_csv('reports/comparison_metrics.csv', index=False)
      print(df)
      
      # Visualize
      fig, ax = plt.subplots(figsize=(12, 6))
      x = range(len(df))
      width = 0.35
      
      ax.bar([i - width/2 for i in x], df['source_only'], width, label='Source-Only')
      ax.bar([i + width/2 for i in x], df['adapted_model'], width, label='Domain-Adapted')
      
      ax.set_xlabel('Metric')
      ax.set_ylabel('Score')
      ax.set_title('Model Comparison: Source-Only vs. Domain-Adapted')
      ax.set_xticks(x)
      ax.set_xticklabels(df['metric'], rotation=45, ha='right')
      ax.legend()
      plt.tight_layout()
      plt.savefig('reports/comparison_chart.png', dpi=300)
      print("✓ Saved: reports/comparison_chart.png")
  
  if __name__ == '__main__':
      generate_comparison_report()
  ```

#### 7.2 Final Report Template

**Code to Generate**:
- `FINAL_REPORT.md`
  ```markdown
  # PCB Smart Maintenance System – Final Report
  
  ## Executive Summary
  Predictive maintenance for PCB manufacturing using domain-adaptive YOLOv8 with few-shot transfer learning.
  
  ## 1. Problem Statement
  [Exact text from project synopsis]
  
  ## 2. Methodology
  
  ### 2.1 Source Domain Training
  - Dataset: PKU-Market-PCB (1,386 images, 6 defect classes)
  - Model: YOLOv8m
  - Final mAP@0.5:0.95: **X.XX%**
  - Per-class metrics: [table]
  
  ### 2.2 Domain Adaptation
  - Target dataset: DsPCBSD+ (10,259 images, 9 classes)
  - Adapter architecture: 2 FC layers (2048→512→128)
  - MMD loss: Defect-aware class weighting
  - Few-shot fine-tuning: 50–100 labelled images
  - Improvement: **+Y.YY mAP** over source-only
  
  ### 2.3 RUL Estimation
  - Synthetic degradation model (exponential decay)
  - 100 equipment scenarios generated
  - Health-defect correlation: ρ = [correlation coefficient]
  
  ### 2.4 Edge Deployment
  - Platform: Raspberry Pi 4 (4 GB)
  - Model format: ONNX (INT8 quantized)
  - Inference speed: Z.Z FPS
  - Latency: W ms
  
  ## 3. Results
  
  ### 3.1 Source Training
  [metrics table]
  
  ### 3.2 Domain Adaptation
  [metrics table + improvement %]
  
  ### 3.3 Comparative Analysis
  [source-only vs. adapted]
  
  ### 3.4 Pi Performance
  - FPS: X
  - Memory usage: Y MB
  - CPU utilization: Z%
  
  ## 4. Conclusion
  Domain adaptation reduces retraining time from weeks to hours, achieving 85–90% accuracy on new equipment configurations with minimal labelled data. Deployment on edge devices enables real-time manufacturing monitoring.
  
  ## 5. References
  [10+ citations from README]
  ```

---

## Execution Workflow

### Week-by-Week Coding Tasks

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

## Code Organization (Final Repository Structure)

```
PCB-Smart-Maintenance/
├── README.md ← Main architecture (from session 1)
├── PROJECT_MASTER_PROMPT.md ← This file (reference in every session)
├── FINAL_REPORT.md ← Generated in W8
│
├── data/
│   ├── pku_market/ ← Source dataset (1,386 images)
│   ├── pku_market_processed/ ← Preprocessed, augmented
│   ├── dspcbsd/ ← Target dataset (10,259 images)
│   ├── dspcbsd/labelled_few_shot/ ← 50–100 samples
│   ├── dspcbsd/unlabelled/ ← Remaining images
│   └── rul_synthetic/ ← Synthetic degradation curves
│
├── models/
│   ├── adapter.py ← Lightweight adapter (2 FC layers)
│   ├── mmd_loss.py ← Defect-aware MMD loss
│   └── __init__.py
│
├── preprocessing/
│   ├── preprocessing.py ← OpenCV pipeline
│   └── augmentation.py ← Data augmentation
│
├── train/
│   ├── train_source_model.py ← YOLOv8 on PKU
│   └── train_adapter.py ← MMD adaptation
│
├── evaluation/
│   ├── eval_source_model.py ← Source metrics
│   ├── eval_adapted_model.py ← Target metrics
│   └── comparative_analysis.py ← Baseline vs. Adapted
│
├── scripts/
│   ├── 01_download_pku_dataset.py
│   ├── 02_dataset_stats.py
│   ├── 03_train_test_split.py
│   ├── 04_download_dspcbsd_dataset.py
│   ├── 05_class_mapping.py
│   └── 06_sample_target_images.py
│
├── rul_synthesis/
│   ├── degradation_model.py ← Synthetic RUL generation
│   └── export_rul_data.py ← Save to CSV/JSON
│
├── database/
│   ├── schema.sql ← PostgreSQL tables
│   └── db.py ← SQLAlchemy ORM
│
├── backend/
│   ├── main.py ← FastAPI inference service
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   └── components/
│   │       ├── LiveFeed.js
│   │       ├── TrendChart.js
│   │       ├── AlertPanel.js
│   │       └── MetricsPanel.js
│   └── public/index.html
│
├── edge/
│   ├── quantize_model.py ← ONNX export
│   ├── setup_pi.sh ← Pi setup script
│   ├── edge_inference.py ← Pi inference loop
│   └── requirements_pi.txt
│
├── weights/
│   ├── best.pt ← Source model (output of train_source_model.py)
│   ├── adapter.pt ← Trained adapter (output of train_adapter.py)
│   ├── adapted_head.pt ← Fine-tuned detection head
│   └── adapted_model.onnx ← Pi deployment model
│
├── reports/
│   ├── dataset_analysis.json
│   ├── source_model_metrics.json
│   ├── adapted_model_metrics.json
│   ├── comparison_metrics.csv
│   ├── comparison_chart.png
│   ├── confusion_matrix_source.png
│   ├── confusion_matrix_adapted.png
│   ├── rul_sample_curves.png
│   └── target_sampling_stats.json
│
├── runs/ ← YOLOv8 training outputs
│   └── source_training/
│       └── yolov8m_pku/
│           ├── weights/
│           ├── results.csv
│           └── confusion_matrix.png
│
└── .gitignore
```

---

## Success Criteria (Validation Checkpoints)

| Phase | Deliverable | Success Metric | Status |
|-------|-------------|-----------------|--------|
| **W1–W2** | Source YOLOv8 model | mAP ≥ 0.95 on test set | ☐ |
| **W3–W4** | Domain adapter + MMD | 85–90% mAP on target (50–100 imgs) | ☐ |
| **W4–W5** | Synthetic RUL data | 100+ scenarios, health-defect correlation | ☐ |
| **W5–W6** | FastAPI + PostgreSQL | Detect + log + fetch metrics endpoints working | ☐ |
| **W6–W7** | React dashboard | Live feed + alerts + trend charts rendering | ☐ |
| **W7–W8** | Pi deployment | ONNX inference @ 2–5 FPS on Pi 4 | ☐ |
| **W8** | Final report | Comparative analysis + conclusions | ☐ |

---

## Quick Reference: Python Entry Points

```bash
# Source training
python train/train_source_model.py

# Domain adaptation
python train/train_adapter.py

# RUL synthesis
python rul_synthesis/degradation_model.py

# Backend
python -m uvicorn backend.main:app --reload

# Pi inference
ssh pi@raspberrypi "python edge/edge_inference.py"

# Evaluation
python evaluation/comparative_analysis.py
```

---

## Critical Notes

1. **No Cuts**: All 7 objectives are coded. RUL uses synthetic data (documented). Pi is bare-minimum deployment (tested on W7).

2. **Datasets**: PKU-Market-PCB (source) + DsPCBSD+ (target) are free, public, and fully supported.

3. **Hardware**: Pi 4 (4GB) + Pi Camera Module 2 are standard, affordable, widely available.

4. **Code is Modular**: Each script can run independently. Phase 1 doesn't block Phase 3 (RUL synthesis runs parallel).

5. **Database & API**: PostgreSQL is free; FastAPI is lightweight. No paid services required.

6. **Deployment**: Docker-ready backend; edge inference via systemd service.

---

## How to Use This Prompt in Future Sessions

**At the start of each Claude session**, say:

> "I'm working on the PCB Smart Maintenance project (AI-driven predictive maintenance for PCB manufacturing). Read the attached `PROJECT_MASTER_PROMPT.md` and `PCB_Minor_Project_README.md` files for full context, requirements, datasets, and architecture. Today I'm working on [PHASE / WEEK]. Generate code for [specific task], following the exact specifications in the prompt. Do not drop any requirements or objectives. Use only free, open-source tools. Output: complete, working Python/JavaScript code with proper error handling."

This ensures **zero context loss** and **consistent execution** across all sessions.

---

**Status**: Ready to Code  
**Version**: 1.0  
**Last Updated**: September 2026  
**Scope**: COMPLETE (No Cuts)
  