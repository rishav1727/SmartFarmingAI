# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

"""
TABLE I — CNN vs ViT vs Hybrid Baseline Comparison
====================================================
Runs 3 architectures on the same labeled_test_set/ and reports:
  - Top-1 Accuracy
  - Macro F1
  - Inference time per image (ms)

Architectures:
  1. CNN     : ResNet-50 (pretrained ImageNet, no fine-tuning) — zero-shot baseline
  2. ViT     : ViT-B/16  (all encoder layers FROZEN, only head trained)
               -> Uses saved best_model.pth with encoder frozen for eval
  3. Hybrid  : ViT-B/16  (last 6 layers unfrozen — current best_model.pth)

Run:  venv\\Scripts\\python.exe run_baseline_comparison.py
"""

import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
sys.path.insert(0, BASE_DIR)

from src.consts import DEVICE, IMAGE_SIZE
from src.model  import get_model

SEP = "=" * 70
def banner(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# ── load class names ──────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Classes : {num_classes} | Device : {DEVICE}")

# ── transforms ───────────────────────────────────────────────────────────────
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── labeled test set ─────────────────────────────────────────────────────────
LABELED_DIR = os.path.join(BASE_DIR, "labeled_test_set")

def load_labeled_dataset(labeled_dir):
    X, y = [], []
    if not os.path.isdir(labeled_dir):
        return X, y
    for cls in sorted(os.listdir(labeled_dir)):
        cls_path = os.path.join(labeled_dir, cls)
        if not os.path.isdir(cls_path) or cls not in class_names:
            continue
        gt_idx = class_names.index(cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                X.append(os.path.join(cls_path, fname))
                y.append(gt_idx)
    return X, y

X_test, y_true = load_labeled_dataset(LABELED_DIR)

if not X_test:
    print("\n[ERROR] labeled_test_set/ is empty or missing.")
    print("        Run evaluate_metrics.py first — it will auto-download the test set.")
    sys.exit(1)

print(f"Test set : {len(X_test)} images, {len(set(y_true))} classes")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate any model on the test set
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, device, name="model"):
    model.eval()
    y_pred = []
    times  = []
    with torch.no_grad():
        for img_path in X_test:
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img).convert("RGB")
            tensor = eval_tf(img).unsqueeze(0).to(device)
            t0  = time.perf_counter()
            out = model(tensor)
            times.append((time.perf_counter() - t0) * 1000)
            pred = torch.argmax(out, dim=1).item()
            y_pred.append(pred)
    acc  = accuracy_score(y_true, y_pred) * 100
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    ms   = float(np.mean(times))
    print(f"  [{name}]  Accuracy={acc:.2f}%  Macro-F1={f1:.4f}  Latency={ms:.1f}ms/img")
    return acc, f1, ms, y_pred

results_table = {}

# ─────────────────────────────────────────────────────────────────────────────
# Model A — CNN baseline: ResNet-50 pretrained (ImageNet), zero-shot
#   Head replaced with Linear(2048 → num_classes), randomly initialized.
#   NOT fine-tuned on PlantVillage → honest "untrained classifier" baseline.
# ─────────────────────────────────────────────────────────────────────────────
banner("Model A — CNN Baseline  (ResNet-50, ImageNet pretrained, zero-shot head)")

cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# Freeze all layers
for p in cnn.parameters():
    p.requires_grad = False
# Replace head — random init, not trained
cnn.fc = nn.Linear(cnn.fc.in_features, num_classes)
cnn = cnn.to(DEVICE)

print(f"  Architecture : ResNet-50 (ImageNet pretrained backbone, random head)")
print(f"  Fine-tuned   : NO — zero-shot on PlantVillage classes")
cnn_acc, cnn_f1, cnn_ms, _ = evaluate_model(cnn, DEVICE, "CNN-ResNet50-ZeroShot")
results_table["CNN (ResNet-50, zero-shot)"] = {
    "accuracy_pct": round(cnn_acc, 2), "macro_f1": round(cnn_f1, 4),
    "latency_ms_per_img": round(cnn_ms, 1),
    "note": "ImageNet pretrained backbone, untrained PlantVillage head"
}
del cnn; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ─────────────────────────────────────────────────────────────────────────────
# Model B — ViT-only baseline: ALL encoder layers frozen, only head fine-tuned
#   Load best_model.pth weights, then freeze the entire encoder.
#   This tests if ViT features alone (without unfreezing) are sufficient.
# ─────────────────────────────────────────────────────────────────────────────
banner("Model B — ViT-only Baseline  (ViT-B/16, all encoder frozen, head only)")

vit_frozen = get_model(num_classes, DEVICE)
vit_frozen.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE
))
# Freeze the entire encoder
for p in vit_frozen.vit.encoder.parameters():
    p.requires_grad = False
vit_frozen.eval()

print(f"  Architecture : ViT-B/16 (all encoder layers FROZEN, head fine-tuned)")
print(f"  Weights from : best_model.pth (same weights, encoder locked for comparison)")
vit_acc, vit_f1, vit_ms, _ = evaluate_model(vit_frozen, DEVICE, "ViT-FrozenEncoder")
results_table["ViT-B/16 (frozen encoder)"] = {
    "accuracy_pct": round(vit_acc, 2), "macro_f1": round(vit_f1, 4),
    "latency_ms_per_img": round(vit_ms, 1),
    "note": "ViT-B/16 with encoder fully frozen; only classification head active"
}
del vit_frozen; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ─────────────────────────────────────────────────────────────────────────────
# Model C — Hybrid (current system): ViT-B/16, last-6-layers unfrozen
#   This is the actual fine-tuned model — best_model.pth as trained.
# ─────────────────────────────────────────────────────────────────────────────
banner("Model C — Hybrid  (ViT-B/16, last-6 encoder layers unfrozen — current model)")

hybrid = get_model(num_classes, DEVICE)
hybrid.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE
))
hybrid.eval()

print(f"  Architecture : ViT-B/16 (last 6 encoder layers fine-tuned)")
print(f"  Weights from : best_model.pth (fully trained)")
hyb_acc, hyb_f1, hyb_ms, _ = evaluate_model(hybrid, DEVICE, "ViT-Hybrid(last6)")
results_table["ViT-B/16 Hybrid (last-6 unfrozen)"] = {
    "accuracy_pct": round(hyb_acc, 2), "macro_f1": round(hyb_f1, 4),
    "latency_ms_per_img": round(hyb_ms, 1),
    "note": "ViT-B/16 with last 6 encoder layers fine-tuned on PlantVillage"
}
del hybrid; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ─────────────────────────────────────────────────────────────────────────────
# TABLE I OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
banner("TABLE I — CNN vs ViT vs Hybrid Comparison")

print(f"\n  Test set: {len(X_test)} images | {len(set(y_true))} classes | Device: {DEVICE}\n")
print(f"  {'Model':<38} {'Accuracy':>10} {'Macro F1':>10} {'Latency(ms)':>12}  Notes")
print("  " + "-" * 90)

for model_name, res in results_table.items():
    print(f"  {model_name:<38} {res['accuracy_pct']:>9.2f}% {res['macro_f1']:>10.4f} "
          f"{res['latency_ms_per_img']:>11.1f}  {res['note'][:40]}")

# Save
out_path = os.path.join(BASE_DIR, "table1_baseline_results.json")
with open(out_path, "w") as f:
    json.dump({
        "test_set_size": len(X_test),
        "num_classes": len(set(y_true)),
        "device": str(DEVICE),
        "models": results_table
    }, f, indent=2)

print(f"\n  Results saved → table1_baseline_results.json")
print(f"\n  KEY FINDINGS:")
best = max(results_table.items(), key=lambda x: x[1]["accuracy_pct"])
worst = min(results_table.items(), key=lambda x: x[1]["accuracy_pct"])
print(f"  Best  model: {best[0]}  →  {best[1]['accuracy_pct']:.2f}%")
print(f"  Worst model: {worst[0]}  →  {worst[1]['accuracy_pct']:.2f}%")
gain = best[1]["accuracy_pct"] - worst[1]["accuracy_pct"]
print(f"  Gain over baseline: +{gain:.2f}% accuracy\n")
