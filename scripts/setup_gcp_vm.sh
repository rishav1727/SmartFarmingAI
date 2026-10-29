#!/bin/bash
set -e

echo "=========================================================="
echo "🌱 SmartFarmingAI — GCP E2-Micro Automated Setup Script"
echo "=========================================================="

# 1. Enable 4GB Swap Space (expands 1GB RAM -> 5GB memory space)
if [ ! -f /swapfile ]; then
    echo "[1/5] Configuring 4GB Swapfile to eliminate PyTorch OOM..."
    sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
    sudo sysctl -p
    echo "✓ Swap space active: $(free -h | grep -i swap)"
else
    echo "✓ Swapfile already exists."
fi

# 2. Update packages and install Docker
echo "[2/5] Installing Docker and dependencies..."
sudo apt-get update -y
sudo apt-get install -y git curl docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# 3. Clone Repository
echo "[3/5] Cloning SmartFarmingAI codebase..."
cd /opt
if [ -d "SmartFarmingAI" ]; then
    cd SmartFarmingAI
    git pull origin main
else
    sudo git clone https://github.com/rishav1727/SmartFarmingAI.git
    cd SmartFarmingAI
fi
sudo chown -R $USER:$USER /opt/SmartFarmingAI

# 4. Build and Run Container on Port 80
echo "[4/5] Building Docker container (this takes ~3-4 minutes)..."
sudo docker build -t smartfarming-backend .

# Stop any previous instance
sudo docker stop smartfarming-backend 2>/dev/null || true
sudo docker rm smartfarming-backend 2>/dev/null || true

echo "[5/5] Launching container with auto-restart on port 80..."
sudo docker run -d \
    --name smartfarming-backend \
    --restart unless-stopped \
    -p 80:8000 \
    -e PORT=8000 \
    -e RENDER=false \
    -e LOW_MEMORY_MODE=0 \
    -e DATABASE_URL="postgresql://neondb_owner:npg_u1v4d0WqXQeL@ep-rough-mouse-b5q658wa.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require" \
    smartfarming-backend

echo "=========================================================="
echo "🎉 SUCCESS: SmartFarmingAI Backend is LIVE on your VM!"
EXTERNAL_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "YOUR_VM_EXTERNAL_IP")
echo "Test endpoint: http://${EXTERNAL_IP}/health"
echo "Test model:    http://${EXTERNAL_IP}/api/model/info"
echo "=========================================================="
