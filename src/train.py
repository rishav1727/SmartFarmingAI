import os
import sys

# Add the project root directory to Python's system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

from src.consts import LEARNING_RATE, NUM_EPOCHS, DEVICE, MODEL_DIR
from src.data_loader import get_dataloaders
from src.model import get_model

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders()
    if not train_loader:
        print("Failed to initialize dataloaders.")
        return

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    # Save class names for inference later
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # Initialize model
    model = get_model(num_classes=num_classes, device=DEVICE)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Differential Learning Rates
    # 1. Base ViT unfrozen layers (slow learning rate to preserve pre-trained knowledge)
    base_params = [p for name, p in model.named_parameters() if 'heads' not in name and p.requires_grad]
    # 2. Newly initialized classification head (fast learning rate to learn the new task)
    head_params = [p for name, p in model.named_parameters() if 'heads' in name and p.requires_grad]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': LEARNING_RATE},          # 2e-5
        {'params': head_params, 'lr': 1e-3}                    # 1e-3 for the new head
    ])
    # Resume from checkpoint if exists
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")
    best_val_acc = 0.0
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path} to continue training...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            print("Successfully loaded model weights.")
        except Exception as e:
            print(f"Failed to load existing weights: {e}")

    # Training Loop

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print("--> Best model weights saved!")

if __name__ == "__main__":
    train()
