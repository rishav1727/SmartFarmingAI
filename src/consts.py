import os
import torch

# Directory settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# Training parameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5  # Lowered for stable fine-tuning
NUM_EPOCHS = 7       # Task specific: 3 epochs as requested
IMAGE_SIZE = 224      # Standard for ViT/ResNet

# Hardware configuration
# Safe hardware check: verify CUDA can actually allocate tensors before setting DEVICE = "cuda"
def get_safe_device():
    try:
        if torch.cuda.is_available():
            # Test tensor allocation to catch asynchronous CUDA driver errors
            test_t = torch.zeros(1, device="cuda")
            del test_t
            return "cuda"
    except Exception as e:
        print(f"Notice: CUDA check failed ({e}). Defaulting inference hardware to CPU.")
    return "cpu"

DEVICE = get_safe_device()

HF_DATASET_NAME = "BrandonFors/Plant-Diseases-PlantVillage-Dataset" 

