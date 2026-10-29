# ─────────────────────────────────────────────────────────────────────────────
# SmartFarmingAI — Cloud Optimized Dockerfile
# Targets Fly.io (port 8080), Render, Railway, and HuggingFace Spaces.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System libraries needed by image processing (Debian Bookworm compatible)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install PyTorch CPU wheels first (saves ~2 GB and avoids CUDA bloat on cloud)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 3: Copy source code
COPY . .

# Step 4: Create runtime directories
RUN mkdir -p public/uploads public/contributions public/samples saved_models data/documents

# Step 5: Bake official ViT-B/16 research checkpoint into the image
RUN python -c "import urllib.request; print('Downloading ViT-B/16 weights...'); urllib.request.urlretrieve('https://github.com/rishav1727/SmartFarmingAI/releases/download/v1.0-vit/best_model.pth', 'saved_models/best_model.pth'); print('ViT-B/16 weights ready.')"

# Non-root user support (compatible with HuggingFace Spaces & secure Linux containers)
RUN chown -R 1000:1000 /app
USER 1000

# Fly.io uses 8080, HuggingFace Spaces uses 7860, Render injects $PORT
EXPOSE 8080 7860 10000 8000

# PORT env var: Fly.io injects 8080, HF Spaces uses 7860, fallback to 8000
CMD ["sh", "-c", "python -m uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
