import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook to capture activations and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # For ViT, the output is often (Batch, SeqLen, HiddenSize)
        # We need to handle the specific shape of ViT blocks
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, category_index=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if category_index is None:
            category_index = torch.argmax(output)
            
        self.model.zero_grad()
        output[0, category_index].backward()

        # For Vision Transformers (ViT), the target layer output is (Batch, Seq_len, Hidden_dim)
        # Sequence length for ViT_B_16 is 1 + (224/16)^2 = 197
        # We ignore the first token (CLS token) and reshape the rest to 14x14
        
        gradients = self.gradients[0, 1:, :] # (196, 768)
        activations = self.activations[0, 1:, :] # (196, 768)
        
        # Pool the gradients across the sequence
        weights = torch.mean(gradients, dim=0) # (768,)
        
        # Weighted combination of activation maps
        cam = torch.matmul(activations, weights) # (196,)
        
        # Reshape to spatial dimensions (14x14 for ViT_B_16)
        cam = cam.reshape(14, 14).detach().cpu().numpy()
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def overlay_heatmap(img_path, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    overlayed_img = heatmap * alpha + img * (1 - alpha)
    overlayed_img = np.uint8(overlayed_img)
    
    return cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB)
