"""
train_adapter.py
Trains the lightweight domain adapter and fine-tunes the detection head using defect-aware MMD
alignment between source (PKU-Market-PCB) and target few-shot (DsPCBSD+) samples.
Saves weights to weights/adapter.pt and weights/adapted_head.pt.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from models.adapter import LightweightAdapter
from models.mmd_loss import DefectAwareMMLoss


class PCBPairDataset(Dataset):
    """Dataset for loading PCB images and labels for domain adaptation training"""

    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 640):
        self.imgsz = imgsz
        self.image_files = sorted(
            list(images_dir.glob("*.jpg"))
            + list(images_dir.glob("*.png"))
            + list(images_dir.glob("*.jpeg"))
        )
        self.labels_dir = labels_dir

    def __len__(self):
        return max(1, len(self.image_files))

    def __getitem__(self, idx):
        if not self.image_files:
            # Synthetic fallback if empty
            img = np.zeros((3, self.imgsz, self.imgsz), dtype=np.float32)
            label = 0
            return torch.tensor(img), torch.tensor(label, dtype=torch.long)

        img_path = self.image_files[idx % len(self.image_files)]
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.imgsz, self.imgsz))

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        # Read label
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        label = 0
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    label = int(parts[0])

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class DomainAdaptedYOLO(nn.Module):
    """Encapsulates Backbone + Adapter + Adapted Classifier Head"""

    def __init__(self, backbone: nn.Module, feature_dim: int = 2048, hidden_dim: int = 512, num_classes: int = 6):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze backbone

        self.adapter = LightweightAdapter(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.backbone(x)
            if isinstance(feat, (list, tuple)):
                feat = feat[-1]
            if feat.dim() > 2:
                feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = torch.flatten(feat, 1)
        return feat

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        adapted_feat = self.adapter(feat)
        logits = self.head(adapted_feat)
        return logits, adapted_feat


def train_adapter(
    source_model_path: str = "weights/best.pt",
    source_data_dir: str = "data/pku_market",
    target_labelled_dir: str = "data/dspcbsd/labelled_few_shot",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    lambda_mmd: float = 0.5,
    device_str: str = None,
):
    device = torch.device(
        device_str
        if device_str
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"=== Starting Domain Adaptation Training ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | LR: {lr} | Lambda MMD: {lambda_mmd}")

    # Load source model
    if not Path(source_model_path).exists():
        print(f"Warning: {source_model_path} not found. Utilizing yolov8n.pt as baseline backbone...")
        source_model_path = "yolov8n.pt"

    yolo_wrapper = YOLO(source_model_path)
    # Extract YOLO backbone (layers 0 through 9 up to SPPF)
    raw_backbone = yolo_wrapper.model.model[:10]
    raw_backbone = raw_backbone.to(device).eval()

    # Determine backbone output feature dimension dynamically
    dummy_input = torch.zeros((2, 3, 640, 640), device=device)
    with torch.no_grad():
        dummy_out = raw_backbone(dummy_input)
        if isinstance(dummy_out, (list, tuple)):
            dummy_out = dummy_out[-1]
        feat_dim = dummy_out.shape[1] if dummy_out.dim() > 1 else 512

    print(f"Detected Backbone Feature Dimension: {feat_dim}")

    model = DomainAdaptedYOLO(raw_backbone, feature_dim=feat_dim, hidden_dim=256, num_classes=6).to(device)

    # Datasets and Loaders
    src_path = Path(source_data_dir)
    tgt_path = Path(target_labelled_dir)

    src_dataset = PCBPairDataset(src_path / "images", src_path / "labels")
    tgt_dataset = PCBPairDataset(tgt_path / "images", tgt_path / "labels")

    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Loss Functions & Optimizer
    mmd_loss_fn = DefectAwareMMLoss(kernel_sigma=1.0)
    classification_loss_fn = nn.CrossEntropyLoss()

    trainable_params = list(model.adapter.parameters()) + list(model.head.parameters())
    optimizer = Adam(trainable_params, lr=lr, weight_decay=1e-5)

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("Beginning Adaptation Optimization...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_det_loss = 0.0
        total_mmd_loss = 0.0
        batches = 0

        # Zip source and target batches
        for (x_src, y_src), (x_tgt, y_tgt) in zip(src_loader, tgt_loader):
            x_src, y_src = x_src.to(device), y_src.to(device)
            x_tgt, y_tgt = x_tgt.to(device), y_tgt.to(device)

            optimizer.zero_grad()

            # Forward passes
            src_feat = model.extract_features(x_src)
            tgt_feat = model.extract_features(x_tgt)

            src_adapted = model.adapter(src_feat)
            tgt_adapted = model.adapter(tgt_feat)

            tgt_logits = model.head(tgt_adapted)

            # Losses
            loss_det = classification_loss_fn(tgt_logits, y_tgt)
            loss_mmd = mmd_loss_fn(src_adapted, tgt_adapted, y_src, y_tgt)
            loss_total = loss_det + lambda_mmd * loss_mmd

            loss_total.backward()
            optimizer.step()

            total_epoch_loss += loss_total.item()
            total_det_loss += loss_det.item()
            total_mmd_loss += loss_mmd.item()
            batches += 1

        avg_loss = total_epoch_loss / max(1, batches)
        avg_det = total_det_loss / max(1, batches)
        avg_mmd = total_mmd_loss / max(1, batches)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Total Loss: {avg_loss:.4f} "
                f"| Det Loss: {avg_det:.4f} | MMD Loss: {avg_mmd:.4f}"
            )

    # Save weights
    adapter_path = weights_dir / "adapter.pt"
    head_path = weights_dir / "adapted_head.pt"
    torch.save(model.adapter.state_dict(), adapter_path)
    torch.save(model.head.state_dict(), head_path)
    print(f"✓ Saved adapter weights: {adapter_path}")
    print(f"✓ Saved adapted head weights: {head_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train domain adapter with defect-aware MMD")
    parser.add_argument("--source-model", type=str, default="weights/best.pt")
    parser.add_argument("--source-data", type=str, default="data/pku_market")
    parser.add_argument("--target-data", type=str, default="data/dspcbsd/labelled_few_shot")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-mmd", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_adapter(
        source_model_path=args.source_model,
        source_data_dir=args.source_data,
        target_labelled_dir=args.target_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mmd=args.lambda_mmd,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

