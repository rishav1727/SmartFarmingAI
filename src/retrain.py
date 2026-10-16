"""
Retraining pipeline — fine-tunes the existing ViT model on
community-contributed images pulled from the database.

Workflow
--------
1. Receive list of DB contribution records (verified, unused)
2. Build a PyTorch Dataset from those images (download from Supabase / local)
3. Fine-tune ONLY the last 2 encoder blocks + classification head
4. Validate, save best checkpoint, back up old weights
5. Optionally push to HuggingFace Hub
6. Return detailed metrics dict
"""

import os
import json
import shutil
from datetime import datetime
from io import BytesIO
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.consts import MODEL_DIR, IMAGE_SIZE, DEVICE
from src.model import get_model
from src import storage as store


# ── Dataset ───────────────────────────────────────────────────────────────────

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ContributionDataset(Dataset):
    """PyTorch dataset built from community-contributed DB records."""

    def __init__(self, records: list[dict], class_names: list[str],
                 transform=None):
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.transform    = transform
        self.samples: list[dict] = []

        for rec in records:
            label = rec.get("label", "")
            if label in self.class_to_idx:
                self.samples.append({
                    "storage_path": rec.get("storage_path", ""),
                    "image_url":    rec.get("image_url", ""),
                    "label_idx":    self.class_to_idx[label],
                })
            else:
                print(f"[retrain] Skipping unknown label: '{label}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = None

        # 1. Local path
        raw = store.download_image(s["storage_path"])
        if raw:
            try:
                img = Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                img = None

        # 2. Public URL fallback
        if img is None and s["image_url"]:
            try:
                import requests as _req
                r = _req.get(s["image_url"], timeout=10)
                img = Image.open(BytesIO(r.content)).convert("RGB")
            except Exception:
                img = None

        # 3. Grey placeholder so the batch doesn't crash
        if img is None:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))

        if self.transform:
            img = self.transform(img)
        return img, s["label_idx"]


# ── Main retraining function ───────────────────────────────────────────────────

def retrain_with_contributions(
    db_records: list[dict],
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    Fine-tune the production model on community contributions.

    Parameters
    ----------
    db_records        : list of dicts with keys: storage_path, image_url, label
    epochs            : number of fine-tuning epochs
    batch_size        : DataLoader batch size
    learning_rate     : AdamW learning rate (keep low — 1e-5 recommended)
    progress_callback : called after each epoch with the epoch metrics dict

    Returns
    -------
    dict with keys: status, samples_used, epochs, best_val_accuracy,
                    history, weights_path, hf_url (or error)
    """
    # ── Load class names ──────────────────────────────────────────────────────
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    if not os.path.exists(class_names_path):
        return {"error": "class_names.json not found in saved_models/."}
    with open(class_names_path) as f:
        class_names = json.load(f)

    # ── Build datasets ────────────────────────────────────────────────────────
    full_dataset = ContributionDataset(db_records, class_names,
                                       transform=_TRAIN_TRANSFORM)
    if len(full_dataset) == 0:
        return {"error": "No valid labelled samples found — check labels match class_names.json."}

    val_size   = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Val set uses non-augmenting transform
    val_ds.dataset.transform = _VAL_TRANSFORM  # type: ignore[attr-defined]

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Load current production model ─────────────────────────────────────────
    model = get_model(len(class_names), DEVICE)
    weights_path = os.path.join(MODEL_DIR, "best_model.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        print(f"[retrain] Loaded base weights from {weights_path}")
    else:
        print("[retrain] No base weights found — fine-tuning from ImageNet ViT.")

    # ── Freeze everything → unfreeze last 2 blocks + head ────────────────────
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.vit.encoder.layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.vit.heads.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=0.01
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc   = 0.0
    tmp_weights    = os.path.join(MODEL_DIR, "_retrain_tmp.pth")
    history        = []

    for epoch in range(epochs):
        # -- Train --
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (out.argmax(1) == labels).sum().item()

        # -- Validate --
        model.eval()
        vl_loss, vl_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out       = model(imgs)
                vl_loss  += criterion(out, labels).item()
                vl_correct += (out.argmax(1) == labels).sum().item()

        tr_acc = tr_correct / train_size * 100
        vl_acc = vl_correct / val_size   * 100
        entry  = {
            "epoch":      epoch + 1,
            "train_loss": round(tr_loss / len(train_loader), 4),
            "val_loss":   round(vl_loss / len(val_loader),   4),
            "train_acc":  round(tr_acc, 2),
            "val_acc":    round(vl_acc, 2),
        }
        history.append(entry)
        print(f"[retrain] Epoch {epoch+1}/{epochs} — "
              f"train_loss={entry['train_loss']} val_acc={entry['val_acc']}%")

        if progress_callback:
            try:
                progress_callback(entry)
            except Exception:
                pass

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), tmp_weights)

        scheduler.step()

    # ── Promote best checkpoint to production ─────────────────────────────────
    if os.path.exists(tmp_weights):
        ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(MODEL_DIR, f"backup_{ts}.pth")
        if os.path.exists(weights_path):
            shutil.copy(weights_path, backup_path)
            print(f"[retrain] Old weights backed up → {backup_path}")
        shutil.move(tmp_weights, weights_path)
        print(f"[retrain] New weights saved → {weights_path}")

    # ── Optional HuggingFace Hub push ─────────────────────────────────────────
    hf_url   = None
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_repo  = os.environ.get("HF_MODEL_REPO", "")
    if hf_token and hf_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=weights_path,
                path_in_repo="best_model.pth",
                repo_id=hf_repo,
                token=hf_token,
            )
            hf_url = f"https://huggingface.co/{hf_repo}"
            print(f"[retrain] Model pushed to HuggingFace Hub: {hf_url}")
        except Exception as e:
            print(f"[retrain] HuggingFace push failed: {e}")

    return {
        "status":           "success",
        "samples_used":     len(full_dataset),
        "train_size":       train_size,
        "val_size":         val_size,
        "epochs":           epochs,
        "best_val_accuracy": round(best_val_acc, 2),
        "history":          history,
        "weights_path":     weights_path,
        "hf_url":           hf_url,
    }
