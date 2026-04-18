import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        
        # Load pre-trained Vision Transformer
        print("Loading pre-trained ViT Model...")
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Freeze early layers to speed up fine-tuning and prevent overfitting
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Unfreeze the last 6 blocks (half the encoder) for better real-world adaptation
        for layer in self.vit.encoder.layers[-6:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Replace the heavily task-specific classification head
        # Note: ViT_B_16 hidden size is 768
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

def get_model(num_classes, device):
    model = PlantDiseaseModel(num_classes=num_classes)
    model.to(device)
    return model
