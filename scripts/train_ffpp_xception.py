import os
import time
import random
import argparse
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from media_auth_forensics.models.xception_forensics import ForensicsBinaryClassifier


class FFPPFramesDataset(Dataset):
    """
    Minimal FF++ frames dataset loader.

    Expected folder layout (recommended):
      data_root/
        train/
          real/   (images)
          fake/   (images)
        val/
          real/
          fake/

    Labels:
      real=0, fake=1
    """

    def __init__(self, data_root: str, split: str, tfm: transforms.Compose, max_samples: Optional[int] = None):
        self.samples: List[Tuple[str, int]] = []
        self.tfm = tfm
        split_dir = os.path.join(data_root, split)
        real_dir = os.path.join(split_dir, "real")
        fake_dir = os.path.join(split_dir, "fake")

        if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
            raise FileNotFoundError("Expected train/val split with real/ and fake/ folders.")

        for root, _, files in os.walk(real_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(root, f), 0))

        for root, _, files in os.walk(fake_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append((os.path.join(root, f), 1))

        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.tfm(img)
        return x, torch.tensor(float(y)), path


def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, epoch: int, best_auc: float):
    """
    Save a checkpoint in a consistent schema used by infer pipeline.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_auc": best_auc,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
    }, path)


def train_ffpp(
    data_root: str,
    exp_dir: str,
    backbone: str = "efficientnet_b3",
    device: str = "cuda",
    image_size: int = 224,
    batch_size: int = 32,
    lr: float = 1e-4,
    epochs: int = 10,
    num_workers: int = 8,
    amp: bool = True,
    resume: Optional[str] = None,
):
    """
    Fine-tune a forensics classifier (Xception-like interface via EfficientNet by default).

    Notes:
      - For true Xception, plug a dedicated Xception implementation into ForensicsBinaryClassifier.
      - This script uses BCEWithLogitsLoss and validates via ROC-AUC.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FFPPFramesDataset(data_root, "train", train_tfm)
    val_ds = FFPPFramesDataset(data_root, "val", val_tfm)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ForensicsBinaryClassifier(backbone=backbone, pretrained=True).to(dev)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler(enabled=amp and dev.type == "cuda")

    start_epoch = 0
    best_auc = 0.0

    if resume:
        ckpt = torch.load(resume, map_location=dev)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_auc = float(ckpt.get("best_auc", 0.0))

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_n = 0

        for x, y, _ in train_ld:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += float(loss.item()) * x.size(0)
            total_n += x.size(0)

        train_loss = total_loss / max(1, total_n)

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y, _ in val_ld:
                x = x.to(dev)
                y = y.to(dev)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(x)
                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())

        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np))

        auc = float(roc_auc_score(labels_np, probs))
        preds = (probs >= 0.5).astype(int)
        acc = float(accuracy_score(labels_np, preds))

        sched.step()

        os.makedirs(exp_dir, exist_ok=True)
        save_ckpt(os.path.join(exp_dir, f"checkpoint_epoch{epoch+1}.pt"), model, optim, epoch+1, best_auc)

        if auc > best_auc:
            best_auc = auc
            save_ckpt(os.path.join(exp_dir, "checkpoint_best.pt"), model, optim, epoch+1, best_auc)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} | val_auc={auc:.4f} | val_acc={acc:.4f} | time={time.time()-t0:.1f}s")

    print(f"Training complete. Best val AUC={best_auc:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--backbone", default="efficientnet_b3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    train_ffpp(
        data_root=args.data_root,
        exp_dir=args.exp_dir,
        backbone=args.backbone,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
