---
title: SmartFarming AI
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# SmartFarming AI 🌿

**Multimodal Generative AI for Smart Farming** — crop disease detection via
Vision Transformer (ViT-B/16) with Grad-CAM explainability, multilingual
AI advisor, and a community contribution system that lets anyone submit
labeled images to continuously improve model accuracy.

## Features
- 📸 Upload leaf images → get disease diagnosis + confidence score
- 🔥 Grad-CAM heatmap showing *why* the model made its decision
- 💬 Multilingual AI farming advisor (10+ Indian languages + English)
- 📤 Contribute labeled images to help retrain the model
- 📊 Admin retraining API — fine-tune from community data with one API call
- 🔗 WhatsApp integration via Twilio

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Diagnose a leaf image |
| POST | `/api/advice` | Get treatment advice |
| POST | `/api/chat` | Chat with AI agronomist |
| POST | `/api/contribute` | Submit a labeled training image |
| GET | `/api/contributions` | List community contributions |
| GET | `/api/model/info` | Current model version & stats |
| POST | `/api/admin/retrain` | Trigger retraining (admin only) |
