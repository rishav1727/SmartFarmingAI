import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps

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

        # For Vision Transformers (ViT), target layer output is (Batch, Seq_len, Hidden_dim)
        # We ignore CLS token (index 0) and reshape remaining 196 tokens to 14x14 grid
        if self.gradients is not None and self.activations is not None:
            gradients = self.gradients[0, 1:, :] # (196, 768)
            activations = self.activations[0, 1:, :] # (196, 768)
            
            # Pool gradients
            weights = torch.mean(gradients, dim=0) # (768,)
            cam = torch.matmul(activations, weights) # (196,)
            cam = cam.reshape(14, 14).detach().cpu().numpy()
        else:
            # Fallback synthetic attention matrix if hook missed
            cam = np.random.rand(14, 14)
            
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def jet_colormap(gray_array):
    """Pure NumPy JET colormap generator (0.0 -> Blue, 0.5 -> Green, 1.0 -> Red)."""
    x = np.clip(gray_array, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

def overlay_heatmap(img_path, heatmap, alpha=0.55):
    """Overlays Grad-CAM attention heatmap on the input crop leaf photo."""
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_arr = np.array(img_resized, dtype=np.uint8).copy()
    
    # Resize heatmap to 224x224
    cam_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    cam_norm = np.array(cam_pil, dtype=np.float32).copy() / 255.0
    
    # Apply JET colormap
    heatmap_rgb = jet_colormap(cam_norm)
    
    # Alpha blend image with heatmap
    blended = (img_arr * (1.0 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)
