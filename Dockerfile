# ─────────────────────────────────────────────────────────────────────────────
# SmartFarmingAI — Cloud Optimized Dockerfile
# Targets Render (Free Tier), Railway, and HuggingFace Spaces.
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

# Non-root user support (compatible with HuggingFace Spaces & secure Linux containers)
RUN chown -R 1000:1000 /app
USER 1000

# Render uses port $PORT (defaults to 10000 on Render, 7860 on HuggingFace Spaces, 8000 locally)
EXPOSE 10000 7860 8000

CMD ["sh", "-c", "python -m uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
