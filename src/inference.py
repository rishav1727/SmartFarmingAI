import os
import torch
import json
from PIL import Image
from torchvision import transforms

from src.consts import DEVICE, MODEL_DIR, IMAGE_SIZE
from src.model import get_model

class DiseasePredictor:
    def __init__(self, model_weights_path="best_model.pth"):
        self.device = DEVICE
        self.class_names = self._load_class_names()
        
        # Load model using the saved class size
        num_classes = len(self.class_names)
        self.model = get_model(num_classes, self.device)
        
        # Load weights
        full_weights_path = os.path.join(MODEL_DIR, model_weights_path)
        if os.path.exists(full_weights_path):
            self.model.load_state_dict(torch.load(full_weights_path, map_location=self.device))
            print(f"Successfully loaded model weights from {full_weights_path}")
        else:
            print(f"Warning: Model weights not found at {full_weights_path}. Predictions will be random.")
            
        self.model.eval()
        
        # Enhanced transformations for real-world robustness
        self.transform = transforms.Compose([
            transforms.Resize(256),              # Resize smaller edge to 256
            transforms.CenterCrop(IMAGE_SIZE),  # Take 224x224 square from center
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_class_names(self):
        class_names_path = os.path.join(MODEL_DIR, "class_names.json")
        if os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                return json.load(f)
        return []

    def predict(self, image_path, save_heatmap=False):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path).convert("RGB")
        
        # Define a comprehensive set of views (Multi-Scale & Localization)
        width, height = image.size
        
        views = {
            "Standard (Full Image)": transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "Center-Zoom": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            "Deep-Center-Zoom": transforms.Compose([
                transforms.Resize(450),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
             "Macro-Center-Zoom": transforms.Compose([
                transforms.Resize(650),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        # Add 5-crop localization for a scaled-up image (captures leaves in corners)
        scale_transform = transforms.Resize(500)
        scaled_img = scale_transform(image)
        five_crop = transforms.FiveCrop(IMAGE_SIZE)(scaled_img)
        
        crop_names = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]
        to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for crop_name, crop_img in zip(crop_names, five_crop):
            views[f"Localized Patch ({crop_name})"] = transforms.Lambda(lambda img, c=crop_img: to_tensor_norm(c))
        
        best_result = None
        highest_conf = -1.0
        winning_tensor = None
        best_top3 = []

        for view_name, transform in views.items():
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
            
            outputs = self.model(img_tensor)
            logits = outputs[0]
            probabilities = torch.nn.functional.softmax(logits, dim=0)
            
            # Mathematical Protection: Energy-Based Out-Of-Distribution (OOD) Detection
            # Softmax forces everything to 100%, causing the model to guess "Tomato" for a Cats.
            # To fix the "Cat hallucination" issue, we calculate the Helmholtz Free Energy of the logits.
            temperature = 1.0
            energy_score = -temperature * torch.logsumexp(logits / temperature, dim=0).item()
            
            # Fetch top 3 instead of just top 1
            top3_probs, top3_catids = torch.topk(probabilities, 3)
            conf = top3_probs[0].item() * 100
            
            if conf > highest_conf:
                highest_conf = conf
                winning_tensor = img_tensor
                
                # Format the top 3 predictions
                best_top3 = []
                for p, cid in zip(top3_probs, top3_catids):
                    label = self.class_names[cid.item()] if self.class_names else f"ID_{cid.item()}"
                    best_top3.append({"disease": label, "confidence": round(p.item() * 100, 2)})
                
                prediction_label = best_top3[0]["disease"]
                
                # Dual-Layer Out-of-Distribution Rejection Threshold
                # 1) Softmax Confidence < 50%
                # 2) Energy Score > 0 (Overconfident Hallucination)
                is_ood = highest_conf < 50.0 or energy_score > 0.0
                detected_object = None
                
                if is_ood:
                    # Initialize ImageNet fallback dynamically to save memory mostly
                    import torchvision.models as models
                    import urllib.request
                    
                    try:
                        # Lightweight General Image Identifier
                        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(winning_tensor.device)
                        mobilenet.eval()
                        
                        # Use the same pre-processed target window but resize it to 224
                        import torchvision.transforms.functional as TF
                        imagenet_tensor = TF.resize(winning_tensor, [224, 224])
                        
                        with torch.no_grad():
                            mn_out = mobilenet(imagenet_tensor)
                            mn_prob = torch.nn.functional.softmax(mn_out[0], dim=0)
                            mn_top_prob, mn_top_catid = torch.topk(mn_prob, 1)
                            
                        # Load labels
                        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                        labels = urllib.request.urlopen(url).read().decode("utf-8").split("\n")
                        
                        detected_object = labels[mn_top_catid[0]].strip()
                        prediction_label = f"Not A Plant (Detected: {detected_object})"
                    except Exception as e:
                        print("Fallback ImageNet Error:", e)
                        prediction_label = "Unknown / Not A Recognizable Plant"
                
                best_result = {
                    "disease": prediction_label,
                    "confidence": round(conf, 2),
                    "view_used": view_name,
                    "catid": top3_catids[0],
                    "top3": best_top3,
                    "energy_score": round(energy_score, 2),
                    "is_ood": is_ood,
                    "detected_object": detected_object
                }

        if save_heatmap:
            from src.visualize import GradCAM, overlay_heatmap
            import cv2
            
            target_layer = self.model.vit.encoder.layers[-1]
            cam_engine = GradCAM(self.model, target_layer)
            
            heatmap = cam_engine.generate_heatmap(winning_tensor, category_index=best_result["catid"])
            vis_img = overlay_heatmap(image_path, heatmap)
            
            output_path = f"heatmap_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            best_result["heatmap_path"] = output_path
            
        return best_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a single plant image")
    parser.add_argument("image_path", help="Path to the image to classify")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmap")
    args = parser.parse_args()
    
    predictor = DiseasePredictor()
    result = predictor.predict(args.image_path, save_heatmap=args.gradcam)
    print(f"\nPrediction Results:")
    if result["is_ood"]:
        print(f"[WARNING]: Out-of-Distribution Safety Triggered!")
        print(f"    Confidence: <50% OR Energy Score: >0.0 (Actual Energy: {result.get('energy_score', 'N/A')})")
        if result.get("detected_object"):
             print(f"    Fallback Agent Identified Object As: '{result['detected_object']}'")
        else:
             print(f"    This image is highly anomalous. It is likely NOT a recognizable plant (e.g. a Cat, Dog, or blank space)!")
    print(f"Detected: {result['disease']}")
    
    print("\n--- Top 3 Match Probabilities ---")
    for i, match in enumerate(result['top3'], 1):
        print(f"  {i}. {match['disease']}: {match['confidence']}%")
        
    print(f"\nOOD Energy Score: {result.get('energy_score', 'N/A')} (Lower is better, >0.0 means Hallucination blocked)")
    print(f"Analysis Method: {result['view_used']}")
    if "heatmap_path" in result:
        print(f"Explanation saved to: {result['heatmap_path']}")
