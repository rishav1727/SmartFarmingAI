import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PlantDiseaseModel(nn.Module):
    """
    Vision Transformer (ViT-B/16) for Plant Disease Diagnosis.
    Research Paper Architecture: 12 Transformer Encoder Blocks, 768 hidden dimension,
    16x16 patch projection, fine-tuned classification head for 38 disease categories.
    """
    def __init__(self, num_classes, pretrained=False):
        super(PlantDiseaseModel, self).__init__()
        
        if pretrained:
            print("[ViT] Loading pre-trained ViT-B/16 Model from PyTorch Hub...")
            self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        else:
            print("[ViT] Initializing ViT-B/16 architecture...")
            self.vit = vit_b_16(weights=None)
        
        # Freeze early layers
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Unfreeze last 6 blocks for fine-tuning
        for layer in self.vit.encoder.layers[-6:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Custom classification head (768 -> 512 -> num_classes)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

def get_model(num_classes, device, pretrained=False):
    model = PlantDiseaseModel(num_classes=num_classes, pretrained=pretrained)
    model.to(device)
    return model
