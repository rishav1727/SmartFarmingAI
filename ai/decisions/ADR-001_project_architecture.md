# ADR-001 — SmartFarmingAI Project Architecture

**Date:** 2026-10-16
**Status:** Accepted
**Deciders:** AI (Antigravity) + User

---

## Context

We needed a clean module structure for an AI-powered smart farming application that:
- Supports crop disease detection via image classification
- Provides an AI farming advisor interface
- Exposes predictions via a FastAPI REST API
- Can export models to ONNX for deployment

## Decision

Adopted a flat `src/` module layout with single-responsibility files:

| File | Responsibility |
|---|---|
| `model.py` | Model architecture definition |
| `train.py` | Training loop and checkpointing |
| `inference.py` | Inference pipeline (load model → predict) |
| `data_loader.py` | Dataset loading, transforms, augmentation |
| `visualize.py` | Plotting utilities (loss curves, confusion matrix, etc.) |
| `advisor.py` | AI farming advisor logic (rules / LLM integration) |
| `consts.py` | Global constants (class names, paths, hyperparams) |
| `app.py` | FastAPI application with routes |
| `export_onnx.py` | ONNX export script |

## Consequences

✅ Easy to navigate — one concern per file  
✅ Simple imports (`from src.model import ...`)  
⚠️ Will need sub-packages as complexity grows (e.g. `src/api/`, `src/ml/`)

---

## How to Add a New ADR

Copy this template, increment the number, and link it from `CHANGELOG.md`.
