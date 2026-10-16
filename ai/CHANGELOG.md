# 📋 Master Changelog — SmartFarmingAI

All changes are listed in **reverse chronological order** (newest first).
Each entry links to the detailed session file.

---

## [2026-10-17] — Cloud Database Initializer, CI/CD Pipeline & Remote Weights Engine

**Type:** `[ADD]` `[FEAT]` `[DEVOPS]` `[DB]`

### Summary
- **Database & Storage Setup Tool (`src/setup_db.py`):**
  - Automated diagnostic and setup tool supporting PostgreSQL (Supabase / Render) and local SQLite.
  - Automatically provisions tables (`contributions`, `model_versions`), tests latency, seeds baseline `v1.0-base` ViT-B/16 checkpoint, and checks Supabase bucket connectivity.
- **Dynamic Remote Weights Downloader (`src/inference.py`):**
  - Added support for `WEIGHTS_URL` and `HF_MODEL_REPO` (`huggingface_hub`) to automatically fetch checkpoints if missing locally on container boot.
- **GitHub Actions CI/CD (`.github/workflows/deploy.yml`):**
  - Automated workflow verifying dependencies, database integrity, and module imports on push, with support for Render deploy hook.
- **One-Click Render Cloud Deployment & Docker Optimization:**
  - Configured `render.yaml` with free managed PostgreSQL database and web service.
  - Added "Deploy to Render" one-click button and complete architecture documentation to `README.md`.
  - Optimized `Dockerfile` for Debian Bookworm, PyTorch CPU (`--index-url https://download.pytorch.org/whl/cpu`), and `opencv-python-headless` to eliminate 2 GB CUDA bloat and fix cloud build failures.
  - Updated Project Guide attribution across footer (`public/index.html`) and documentation (`README.md`) to **Mr. Deepak Aneja**.
  - Engineered **Memory-Safe Cloud Inference Engine**: eliminated runtime 330 MB default weight download (`weights=None`), guarded secondary MobileNet network, and wrapped Grad-CAM in robust error recovery so diagnosis runs smoothly under 512 MB RAM without OOM 502 errors.

---

## [2026-10-17] — Precision Agritech Theme Overhaul & Multi-Phone Zero-Overflow Guarantee

**Type:** `[UI/UX]` `[THEME]` `[MOBILE]`

### Summary
- **Modern Agritech Color Palette & Ambience:**
  - Upgraded base palette to **Midnight Nature-Tech Slate** (`#070d12` with emerald/cyan radial gradients and micro-mesh dot grid).
  - Replaced harsh dark glass with luminous frosted mint glass (`rgba(13, 23, 33, 0.78)` with `rgba(52, 211, 153, 0.15)` borders).
  - Modernized typography with silver-sage high-legibility hierarchy (`#f8fafc`, `#cbd5e1`, `#8193a8`).
- **Bulletproof Multi-Phone Adaptation (Zero Horizontal Overflow):**
  - Eliminated `nowrap` on `.steps-banner` on mobile; converts to a fluid vertical list on phones so it cannot push past screen boundaries.
  - Converted `.stats-strip` to a clean 2x2 grid on mobile phones.
  - Set `header`, `hero-3d-container`, `.crop-pills-container`, and `.site-footer` to strict `calc(100% - 16px)` width with zero horizontal margin bleed.
  - Tested across screen sizes: 320px (iPhone SE/Galaxy Fold), 375px–390px (iPhone 12–16), 412px–440px (Galaxy Ultra/Pro Max), and 768px+ tablets.

---

## [2026-10-16] — Universal Mobile Adaptation & Comprehensive Agritech Footer

**Type:** `[ADD]` `[FEAT]` `[UI/UX]` `[RESPONSIVE]`

### Summary
- **Universal Mobile Adaptation (Every Smartphone):**
  - Added global horizontal overflow immunity (`overflow-x: hidden !important`, `max-width: 100vw !important`).
  - Added multi-tier breakpoint scaling:
    - Extra-compact screens (≤380px: iPhone SE, Galaxy A-series).
    - Standard smartphones (≤768px: iPhone 12/13/14/15/16, Galaxy S/Ultra, Pixel, OnePlus).
    - Tablets (≤992px: iPad, Android tablets).
  - 3D Hero canvas height dynamically scales to `240px` (or `200px` on extra-small screens) with `touch-action: pan-y` so vertical scrolling is fluid.
  - Native mobile bottom navigation bar safe-area insets supported (`padding-bottom: 90px`).
  - Crop pills horizontal touch scroll enabled with momentum snapping.
- **Comprehensive Agritech Footer:**
  - Full academic attribution: B.Tech Final Year Project (BIT-753), IT Department, ABES Engineering College, Ghaziabad.
  - Student team and Project Guide credentials.
  - Architectural tech stack badges (`ViT-B/16`, `Grad-CAM`, `Three.js`, `Hybrid RAG`, `FastAPI`, `Twilio`).
  - Quick action shortcuts, legal agricultural safety disclaimer, and neural engine status indicators.

---

## [2026-10-16] — 3D Botanical Digital Twin WebGL Hero & Modern Glassmorphism UI

**Type:** `[ADD]` `[FEAT]` `[UI/UX]`

### Summary
- **Three.js 3D WebGL Digital Twin:**
  - Integrated a real-time procedural 3D Botanical Leaf with central vascular bundle and organic bezier curvature.
  - Added 180 floating bio-energy particles representing cellular respiration and vitality.
  - Interactive mouse and touch drag-to-rotate with velocity damping and continuous organic floating bounce.
  - Interactive Mode Switchers:
    - 🌿 **Organic Leaf**: standard photorealistic glossy shader.
    - 🧠 **ViT Attention Grid**: wireframe representing Vision Transformer 14x14 attention patches.
    - ✨ **Bio-Particles**: particle cloud simulation.
  - Telemetry HUD displaying ViT multi-head architecture, Grad-CAM layer 11 hook, and 60 FPS WebGL counter.
- **Micro-Interactions & 3D Glass Depth:**
  - Added physical elevation and depth hover animations on all glassmorphic cards.
  - Added quick CTA buttons directly inside the 3D hero.

---

## [2026-10-16] — Advanced Hybrid Agricultural RAG & Community Contribution UI

**Type:** `[ADD]` `[MOD]` `[FEAT]`

### Summary
- **Advanced Hybrid RAG Engine (`src/advisor.py`):**
  - Upgraded from simple keyword matching to high-dimensional **Sublinear TF-IDF N-Gram Vector Engine** combined with dense vector embeddings.
  - Implemented **Agricultural Metadata Filtering** (guards retrieval to crop specific pathology: Apple, Tomato, Potato, Corn, etc.).
  - Added semantic chunking with automated category tagging (chemical, biological, preventative, symptoms).
  - Handles offline/exhausted quota gracefully with zero hallucination.
- **Community Contribution Web UI (`public/index.html`, `public/app.js`):**
  - Added **"🌱 Contribute Data"** button to the main dashboard.
  - Interactive submission modal for farmers and researchers to upload field photos, select from 29+ known disease categories, and submit directly to `/api/contribute`.
- **Public Cloudflare Live Tunnel Active:**
  - Accessible worldwide via trycloudflare.com.

---

## [2026-10-16] — Cloud Architecture, Database & Community Retraining System

**Type:** `[ADD]` `[MOD]` `[INFRA]`

### Summary
- `src/database.py` — SQLAlchemy ORM (SQLite local / PostgreSQL production)
  - `Contribution` table: stores community-submitted labeled images
  - `ModelVersion` table: tracks every training run and its metrics
- `src/storage.py` — cloud storage abstraction (Supabase → local fallback)
- `src/retrain.py` — full fine-tuning pipeline using community contributions
- `src/app.py` — 7 new API endpoints added:
  - `POST /api/contribute` — submit a labeled leaf image
  - `GET  /api/contributions` — browse all submissions
  - `GET  /api/model/info` — model version + community stats
  - `POST /api/admin/verify/{id}` — admin approves a label
  - `POST /api/admin/retrain` — triggers background fine-tuning
  - `GET  /api/admin/retrain/status` — poll retraining progress
- `requirements.txt` — added `sqlalchemy`, `psycopg2-binary`, `supabase`, `huggingface_hub`, `python-dotenv`
- `Dockerfile` — containerized deployment (HuggingFace Spaces / Render / Railway)
- `render.yaml` — Render.com IaC (PostgreSQL + 2GB persistent disk for weights)
- `README.md` — HuggingFace Spaces YAML front-matter + API table
- `.env.example` — full environment variables template
- `.gitignore` — protects secrets and large files

### Files Changed
| Action | File |
|---|---|
| `[ADD]` | `src/database.py` |
| `[ADD]` | `src/storage.py` |
| `[ADD]` | `src/retrain.py` |
| `[MOD]` | `src/app.py` |
| `[MOD]` | `requirements.txt` |
| `[ADD]` | `Dockerfile` |
| `[ADD]` | `render.yaml` |
| `[ADD]` | `README.md` |
| `[ADD]` | `.env.example` |

---

## [2026-10-16] — Project Initialization & AI Record System

**Session:** [2026-10-16_project_init](sessions/2026-10-16_project_init.md)
**Author:** AI (Antigravity)
**Type:** `[INFRA]` `[ADD]` `[DOCS]`

### Summary
- Initial SmartFarmingAI project scaffolded with `src/` module structure
- Core modules created: `model.py`, `train.py`, `inference.py`, `data_loader.py`,
  `visualize.py`, `advisor.py`, `consts.py`, `app.py`, `export_onnx.py`
- FastAPI app skeleton set up in `src/app.py`
- ONNX export pipeline added in `src/export_onnx.py`
- `ai/` folder created to track all future AI-assisted changes
- Established changelog, session log, ADR, model history, and prompt archive structure

### Files Changed
| Action | File |
|---|---|
| `[ADD]` | `src/model.py` |
| `[ADD]` | `src/train.py` |
| `[ADD]` | `src/inference.py` |
| `[ADD]` | `src/data_loader.py` |
| `[ADD]` | `src/visualize.py` |
| `[ADD]` | `src/advisor.py` |
| `[ADD]` | `src/consts.py` |
| `[ADD]` | `src/app.py` |
| `[ADD]` | `src/export_onnx.py` |
| `[ADD]` | `ai/README.md` |
| `[ADD]` | `ai/CHANGELOG.md` |
| `[ADD]` | `ai/models/model_history.md` |
| `[ADD]` | `ai/decisions/ADR-001_project_architecture.md` |
| `[ADD]` | `ai/sessions/2026-10-16_project_init.md` |
| `[ADD]` | `ai/prompts/2026-10-16_ai_folder_setup.md` |

---

## [2026-10-16] — Deployment Configuration

**Session:** [2026-10-16_deployment_setup](sessions/2026-10-16_deployment_setup.md)
**Author:** AI (Antigravity)
**Type:** `[MOD]` `[INFRA]` `[ADD]`

### Summary
- Changed `src/app.py` host binding from `127.0.0.1` → `0.0.0.0` for LAN access
- Disabled `reload=True` → `reload=False` in uvicorn for stable production serving
- Created `start_server.bat` one-click startup script with IP display and weight check
- Created full deployment guide artifact covering: network access, firewall, model loading,
  auto-start (Task Scheduler), internet access (ngrok / port forwarding), and troubleshooting

### Files Changed
| Action | File |
|---|---|
| `[MOD]` | `src/app.py` — host `0.0.0.0`, `reload=False` |
| `[ADD]` | `start_server.bat` — one-click startup |

---

<!-- Add new entries above this line in the format shown above -->
