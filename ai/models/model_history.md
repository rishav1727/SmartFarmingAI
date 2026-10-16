# 📊 Model Version History

All ML model checkpoints, retraining events, and performance metrics are logged here.

---

## Version Table

| Version | Date | Architecture | Dataset | Key Metrics | Notes |
|---|---|---|---|---|---|
| v0.1 | 2026-10-16 | Initial build | — | — | Baseline model created in `src/model.py` |

---

## How to Add an Entry

When you retrain or export a model, add a row to the table above **and** paste the key
training output (loss, accuracy, mAP, etc.) in a new section below.

### Template

```
## v<X.Y> — <YYYY-MM-DD>

**Architecture:** <e.g. ResNet18 / Custom CNN>
**Task:** <e.g. crop disease detection>
**Dataset:** <name + size>
**Training:** epochs=X, lr=X, batch=X
**Results:**
  - Train loss: X
  - Val loss:   X
  - Accuracy / mAP: X

**Exported:** ONNX / TorchScript / None
**Notes:** <anything unusual or important>
```
