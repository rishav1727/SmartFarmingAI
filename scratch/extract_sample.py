import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data_loader import get_dataloaders
from PIL import Image

def extract_sample():
    # Use the dataloader to get one sample
    train_loader, val_loader, class_names = get_dataloaders()
    
    if not val_loader:
        print("Could not load dataset.")
        return

    # Get one batch from val_loader
    images, labels = next(iter(val_loader))
    
    # We need the original image. But the loader returns normalized tensors.
    # Let's grab the raw dataset instead.
    from datasets import load_dataset
    from src.consts import HF_DATASET_NAME
    
    dataset = load_dataset(HF_DATASET_NAME, "default")
    val_split = "validation" if "validation" in dataset else "test"
    if val_split not in dataset:
        # If no validation set natively exists, split the train set
        split = dataset["train"].train_test_split(test_size=0.2)
        val_data = split["test"]
    else:
        val_data = dataset[val_split]

    sample = val_data[0]
    image = sample["image"]
    label_idx = sample["labels"] if "labels" in sample else sample.get("label", 0)
    label_name = class_names[label_idx] if class_names else str(label_idx)

    sample_path = "test_sample.jpg"
    image.save(sample_path)
    print(f"Saved sample image to {sample_path}")
    print(f"True Label: {label_name}")

if __name__ == "__main__":
    extract_sample()
