import os
import torch

# Directory settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5  # Lowered for stable fine-tuning
NUM_EPOCHS = 1
IMAGE_SIZE = 224  # Standard for ViT/ResNet

# Hardware configuration
# This automatically checks for CUDA (NVIDIA GPU). IdeaPad Gaming laptops generally support this.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_DATASET_NAME = "BrandonFors/Plant-Diseases-PlantVillage-Dataset" 
