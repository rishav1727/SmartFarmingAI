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
**Multimodal Generative AI for Smart Farming with Explainable Vision Transformers & Continuous Learning**

*Final Year B.Tech IT Project (BIT-753) — ABESEC Ghaziabad*

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/rishav1727/SmartFarmingAI)
[![GitHub Actions CI](https://github.com/rishav1727/SmartFarmingAI/actions/workflows/deploy.yml/badge.svg)](https://github.com/rishav1727/SmartFarmingAI/actions)

---

## 🚀 One-Click Cloud Deployment

You can deploy the entire stack (FastAPI Backend + 3D WebGL Digital Twin + Free PostgreSQL Database) with one click on Render:

1. Click the **Deploy to Render** button above.
2. Render automatically provisions:
   - **Web Service:** Docker container running FastAPI & Uvicorn.
   - **Managed Database:** Free PostgreSQL instance with auto-injected `DATABASE_URL`.
3. Set optional environment variables in Render Dashboard:
   - `WEIGHTS_URL` / `HF_MODEL_REPO`: To pull custom fine-tuned weights on boot.
   - `SUPABASE_URL` & `SUPABASE_KEY`: If using Supabase Object Storage for training images.
   - `OPENAI_API_KEY`: For generative treatment advisor (hybrid local fallback RAG is active by default).

---

## 🌟 Key Features

- **Multimodal Disease Diagnosis:** Vision Transformer (ViT-B/16) pre-trained and fine-tuned across 38 crop disease classes (PlantVillage dataset).
- **Explainable AI (Grad-CAM):** Real-time attention heatmaps highlighting diseased leaf lesions to build farmer trust.
- **Safety Pre-screener (OOD):** Dual MobileNet-V3 pre-filter reject non-plant objects (e.g. phones, shoes, faces).
- **Hybrid Agricultural RAG:** Sublinear TF-IDF + dense vector similarity with crop-specific metadata filtering preventing cross-crop hallucinations.
- **Continuous Learning Loop:** Community farmers upload field images $\rightarrow$ admin verifies $\rightarrow$ automated background fine-tuning pipeline (`src/retrain.py`) upgrades `best_model.pth`.
- **Interactive 3D WebGL Digital Twin:** Procedural Three.js leaf canvas with gesture controls, ViT attention grid projection, and bio-particle flow.
- **Universal Mobile Adaptation:** Responsive design tested across desktop, tablet, and all smartphone viewport widths ($320\text{px} - 430\text{px}$) with zero horizontal overflow.

---

## 🗄️ Database Architecture & Setup

The system supports both local SQLite development and cloud PostgreSQL:

```bash
# 1. Initialize and test the database connection
python -m src.setup_db
```

### Environment Configuration (`.env`)
```env
# Database (Auto-fallback to SQLite if unset)
DATABASE_URL=postgresql://user:password@db.supabase.co:5432/postgres

# Cloud Storage (Optional - falls back to public/contributions/)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-or-anon-key
SUPABASE_BUCKET=training-contributions

# Model Checkpoints (Optional)
WEIGHTS_URL=https://your-storage-host.com/best_model.pth
HF_MODEL_REPO=rishav1727/smartfarming-vit

# Admin Security
ADMIN_SECRET=your_secure_secret_key
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Multimodal leaf diagnosis with Grad-CAM heatmap & confidence |
| `POST` | `/api/advice` | Hybrid RAG agronomic treatment plan (10+ languages) |
| `POST` | `/api/chat` | Conversational agronomist assistant |
| `POST` | `/api/contribute` | Community submission of field-collected leaf photos |
| `GET` | `/api/contributions` | List pending & verified community submissions |
| `GET` | `/api/model/info` | Live model checkpoint metadata, parameters & stats |
| `POST` | `/api/admin/verify` | Admin endpoint to verify community contributions |
| `POST` | `/api/admin/retrain` | Trigger background fine-tuning on new field data |
| `POST` | `/api/index-docs` | Ingest agricultural PDF research documents into RAG index |
| `POST` | `/api/whatsapp` | Twilio WhatsApp MMS webhook for field diagnostics |

---

## 🧪 Local Development

```bash
# Clone repository
git clone https://github.com/rishav1727/SmartFarmingAI.git
cd SmartFarmingAI

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database setup
python -m src.setup_db

# Start server
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📜 Academic Attribution
- **Course:** Final Year B.Tech Project BIT-753
- **Department:** Information Technology, ABES Engineering College, Ghaziabad
- **Affiliation:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
- **License:** MIT Open Source
