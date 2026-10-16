# ─────────────────────────────────────────────────────────────────────────────
# SmartFarmingAI — Dockerfile
# Targets HuggingFace Spaces (16 GB CPU) and Render / Railway.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System libs needed by OpenCV and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create runtime directories
RUN mkdir -p public/uploads public/contributions public/samples saved_models data/documents

# HuggingFace Spaces runs as non-root user 1000
RUN chown -R 1000:1000 /app
USER 1000

EXPOSE 7860

# HuggingFace Spaces uses port 7860; Render/Railway use $PORT
CMD ["sh", "-c", "python -m uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
