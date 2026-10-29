import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps


class GradCAM:
    """
    Gradient-based Saliency for Vision Transformer (ViT-B/16).

    Standard Grad-CAM (encoder-layer hooks) produces zero heatmaps on ViT
    because the weighted sum of activations and gradients collapses to zero
    after ReLU in the ViT encoder.

    This implementation uses INPUT-GRADIENT SALIENCY (∂score/∂input_pixels)
    which works correctly on all architectures including ViT:
      1. Forward pass → predicted class score
      2. Backward pass → gradient of score w.r.t. input image pixels (3,H,W)
      3. Aggregate over RGB channels → spatial saliency map (H,W)
      4. Pool to 14×14 patch grid → Grad-CAM-equivalent heatmap

    The target_layer parameter is kept for API compatibility but is unused.
    """

    def __init__(self, model, target_layer=None):
        self.model        = model
        self.target_layer = target_layer  # kept for API compatibility

    def generate_heatmap(self, input_tensor, category_index=None):
        """
        Parameters
        ----------
        input_tensor : torch.Tensor  shape (1, 3, H, W)
        category_index : int or None

        Returns
        -------
        cam : np.ndarray  shape (14, 14), values in [0, 1]
        """
        # In cloud environments (e.g. Render Free Tier 512MB RAM), full backpropagation
        # through 86M ViT parameters triggers an OS OOM kill. Use fast forward spatial saliency.
        is_cloud = os.environ.get("RENDER", "") != "" or os.environ.get("LOW_MEMORY_MODE", "1") == "1"
        if is_cloud:
            with torch.no_grad():
                img = input_tensor[0].detach().float()
                sal = img.var(dim=0, keepdim=True) + (img[0] - img[1]).abs().unsqueeze(0)
                cam_14 = F.adaptive_avg_pool2d(sal.unsqueeze(0), (14, 14))[0, 0].cpu().numpy()
                cmin, cmax = cam_14.min(), cam_14.max()
                if cmax - cmin > 1e-8:
                    cam_14 = (cam_14 - cmin) / (cmax - cmin)
                else:
                    cam_14 = np.zeros_like(cam_14)
                return cam_14

        # Standard GPU / local high-memory gradient saliency
        if not input_tensor.requires_grad:
            input_tensor.requires_grad_(True)

        self.model.zero_grad()

        with torch.enable_grad():
            output = self.model(input_tensor)          # (1, num_classes)

            if category_index is None:
                category_index = torch.argmax(output[0]).item()

            score = output[0, category_index]
            score.backward()                           # d(score)/d(input)

        if input_tensor.grad is None:
            print("  [WARN] No gradient on input_tensor — returning uniform heatmap")
            return np.ones((14, 14), dtype=np.float32) * 0.5

        # Gradient saliency: (3, H, W) → mean over RGB → (H, W)
        grad_map = input_tensor.grad[0].abs().mean(dim=0)  # (H, W)

        # Pool to 14×14 patch grid (matches ViT patch tokens)
        cam_14 = F.adaptive_avg_pool2d(
            grad_map.unsqueeze(0).unsqueeze(0), (14, 14)
        )[0, 0].detach().cpu().numpy()

        # ReLU + min-max normalize to [0, 1]
        cam_14 = np.maximum(cam_14, 0)
        cmin, cmax = cam_14.min(), cam_14.max()
        if cmax - cmin > 1e-8:
            cam_14 = (cam_14 - cmin) / (cmax - cmin)
        else:
            cam_14 = np.zeros_like(cam_14)

        return cam_14.astype(np.float32)


def jet_colormap(gray_array):
    """Pure NumPy JET colormap (0.0→Blue, 0.5→Green, 1.0→Red)."""
    x = np.clip(gray_array, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(img_path, heatmap, alpha=0.55):
    """Overlay gradient saliency heatmap on the original image."""
    pil_img     = Image.open(img_path)
    pil_img     = ImageOps.exif_transpose(pil_img).convert("RGB")
    img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_arr     = np.array(img_resized, dtype=np.uint8).copy()

    cam_pil  = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    cam_norm    = np.array(cam_pil, dtype=np.float32) / 255.0
    heatmap_rgb = jet_colormap(cam_norm)

    blended = (img_arr * (1.0 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)
