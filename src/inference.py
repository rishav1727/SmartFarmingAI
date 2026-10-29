import os
import torch
import json
from PIL import Image
from torchvision import transforms

from src.consts import DEVICE, MODEL_DIR, IMAGE_SIZE
from src.model import get_model

class DiseasePredictor:
    def __init__(self, model_weights_path=None):
        import gc
        if os.environ.get("RENDER") or os.environ.get("LOW_MEMORY_MODE", "1") == "1":
            torch.set_num_threads(1)
            
        self.device = DEVICE
        self.class_names = self._load_class_names()
        
        # Load model using the saved class size
        num_classes = len(self.class_names)
        if num_classes == 0:
            print("Warning: No class names found. Model will use default 38 classes.")
            num_classes = 38 # Default for PlantVillage if classes missing
            
        # Vision Transformer (ViT-B/16) - Core Research Architecture
        self.model = get_model(num_classes, self.device)
        self.model_type = "vit"
        
        # Determine weights path dynamically across candidates
        candidate_weights = []
        if model_weights_path:
            candidate_weights.append(model_weights_path)
        else:
            default_weight = os.path.join(MODEL_DIR, "best_model.pth")
            
            # Check for remote weights URL (e.g., GitHub Release assets, S3, custom CDN)
            github_release_url = "https://github.com/rishav1727/SmartFarmingAI/releases/download/v1.0-vit/best_model.pth"
            remote_url = os.environ.get("WEIGHTS_URL", github_release_url)
            if not os.path.exists(default_weight):
                try:
                    import urllib.request
                    print(f"[inference] Pulling ViT-B/16 checkpoint from: {remote_url}...")
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    urllib.request.urlretrieve(remote_url, default_weight)
                    print(f"[inference] Successfully downloaded weights to {default_weight}")
                except Exception as e:
                    print(f"[inference] Remote weights download failed: {e}")

            # Check for Hugging Face model repository
            hf_repo = os.environ.get("HF_MODEL_REPO")
            if hf_repo and not os.path.exists(default_weight):
                try:
                    from huggingface_hub import hf_hub_download
                    token = os.environ.get("HF_TOKEN")
                    print(f"[inference] Pulling model checkpoint from Hugging Face: {hf_repo}...")
                    downloaded_path = hf_hub_download(repo_id=hf_repo, filename="best_model.pth", token=token)
                    candidate_weights.append(downloaded_path)
                except Exception as e:
                    print(f"[inference] Hugging Face Hub download failed: {e}")

            candidate_weights.extend([
                default_weight,
                os.path.join(MODEL_DIR, "best_model.pt"),
                os.path.join(MODEL_DIR, "crop_disease_model.pth"),
                os.path.join(MODEL_DIR, "model.pth")
            ])
            
        is_cloud = os.environ.get("RENDER", "") != "" or os.environ.get("LOW_MEMORY_MODE", "1") == "1"
        if is_cloud:
            self.model = self.model.half()

        found_weights = False
        for weight_file in candidate_weights:
            if os.path.exists(weight_file):
                try:
                    state_dict = torch.load(weight_file, map_location=self.device)
                    if is_cloud:
                        state_dict = {k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict)
                    del state_dict
                    gc.collect()
                    print(f"[inference] Successfully loaded ViT-B/16 weights from {weight_file} (fp16 mode: {is_cloud})")
                    self.weights_loaded = True
                    found_weights = True
                    break
                except Exception as e:
                    print(f"Error loading weights from {weight_file}: {e}")
                    
        if not found_weights:
            print("Notice: Running inference with base ViT-B/16 architecture.")
            self.weights_loaded = False

        self.model.eval()
        
        # Standard transformations for Vision Transformer (ViT-B/16)
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

    def predict(self, image_path, save_heatmap=False, target_crop=None):
        try:
            return self._predict_internal(image_path, save_heatmap=save_heatmap, target_crop=target_crop)
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                print(f"CUDA Exception caught during inference ({e}). Falling back to CPU execution.")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                if hasattr(self, "mobilenet_classifier") and self.mobilenet_classifier:
                    self.mobilenet_classifier = self.mobilenet_classifier.to(self.device)
                return self._predict_internal(image_path, save_heatmap=save_heatmap, target_crop=target_crop)
            raise e

    def _predict_internal(self, image_path, save_heatmap=False, target_crop=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = Image.open(image_path)
        from PIL import ImageOps
        image = ImageOps.exif_transpose(image).convert("RGB")

        
        # Target crop index filtering for high-precision targeted inference
        crop_indices = []
        if target_crop and target_crop.lower() not in ["all", "none", ""]:
            clean_crop = target_crop.lower().strip()
            aliases = {
                "corn": ["corn"],
                "pepper": ["pepper"],
                "cherry": ["cherry"],
                "orange": ["orange"],
                "citrus": ["orange"],
                "squash": ["squash"],
                "soybean": ["soybean"]
            }
            search_terms = aliases.get(clean_crop, [clean_crop])
            
            for idx, cname in enumerate(self.class_names):
                cname_lower = cname.lower().replace("_", " ")
                if any(term in cname_lower for term in search_terms):
                    crop_indices.append(idx)

        # Adaptive views for Vision Transformer
        is_cloud = os.environ.get("RENDER", "") != "" or os.environ.get("LOW_MEMORY_MODE", "1") == "1"
        
        if is_cloud:
            views = {
                "Standard": transforms.Compose([
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }
        else:
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

        with torch.no_grad():
            for view_name, transform in views.items():
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                if is_cloud:
                    img_tensor = img_tensor.half()
                outputs = self.model(img_tensor)
                logits = outputs[0].float()
                
                if crop_indices:
                    masked_logits = logits.clone()
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    mask[crop_indices] = False
                    masked_logits[mask] = -1e9
                    probabilities = torch.nn.functional.softmax(masked_logits, dim=0)
                else:
                    probabilities = torch.nn.functional.softmax(logits, dim=0)
                
                # Mathematical Protection: Energy-Based Out-Of-Distribution (OOD) Detection
                temperature = 1.0
                energy_score = -temperature * torch.logsumexp(logits / temperature, dim=0).item()
                
                # Fetch top 3 predictions
                k = min(3, len(crop_indices)) if crop_indices else 3
                top3_probs, top3_catids = torch.topk(probabilities, k)
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
                    
                    # Dual-Layer Out-of-Distribution & Non-Plant Rejection System
                    detected_object = None
                    is_non_plant = False
                    
                    ENABLE_OOD = os.environ.get("ENABLE_OOD_FILTER", "0") == "1"
                    if ENABLE_OOD:
                        try:
                            import torchvision.models as models
                            import urllib.request
                            import torchvision.transforms.functional as TF
                            
                            if not hasattr(self, "mobilenet_classifier") or self.mobilenet_classifier is None:
                                self.mobilenet_classifier = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1).to(winning_tensor.device)
                                self.mobilenet_classifier.eval()
                            
                            if not hasattr(self, "imagenet_classes") or not self.imagenet_classes:
                                url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                                self.imagenet_classes = urllib.request.urlopen(url, timeout=4).read().decode("utf-8").split("\n")
                            
                            # Get TOP-5 MobileNet predictions for better coverage
                            imagenet_tensor = TF.resize(winning_tensor, [224, 224])
                            with torch.no_grad():
                                mn_out = self.mobilenet_classifier(imagenet_tensor)
                                mn_prob = torch.nn.functional.softmax(mn_out[0], dim=0)
                                mn_top_prob, mn_top_catids = torch.topk(mn_prob, 5)

                            # Collect all top-5 labels
                            top5_labels = []
                            for cid in mn_top_catids:
                                lbl = self.imagenet_classes[cid].strip().lower() if cid < len(self.imagenet_classes) else ""
                                top5_labels.append(lbl)
                            
                            detected_object = self.imagenet_classes[mn_top_catids[0]].strip()
                            
                            # ===== PLANT WHITELIST (Top-5 scan) =====
                            plant_safelist_keywords = [
                                "cabbage", "broccoli", "cauliflower", "artichoke", "spinach", "lettuce",
                                "cucumber", "zucchini", "pumpkin", "squash", "acorn", "butternut",
                                "mushroom", "fig", "pineapple", "banana", "strawberry", "orange", "lemon",
                                "lime", "apple", "grape", "cherry", "peach", "pear", "mango", "papaya",
                                "corn", "maize", "wheat", "rice", "potato", "yam",
                                "tomato", "pepper", "chili", "eggplant", "avocado", "coconut",
                                "plant", "leaf", "flower", "blossom", "herb", "fern", "moss",
                                "tree", "shrub", "vine", "grass", "seed", "root", "stem", "petal",
                                "vegetable", "fruit", "crop", "foliage", "daisy", "tulip", "rose",
                                "sunflower", "clover", "algae", "seaweed", "bamboo", "cactus", "succulent",
                                "kale", "celery", "carrot", "radish", "beet", "turnip", "onion", "garlic",
                                "soybean", "bean", "pea", "lentil", "chickpea", "rapeseed", "hay"
                            ]
                            
                            # Check if ANY top-5 label contains a plant keyword
                            is_plant_confirmed = any(
                                any(p in lbl for p in plant_safelist_keywords)
                                for lbl in top5_labels
                            )
                            
                            # If NO top-5 prediction matched a plant keyword → non-plant
                            if not is_plant_confirmed:
                                is_non_plant = True
                                
                        except Exception as e:
                            print("General object pre-screener error:", e)

                    # Determine overall Out-of-Distribution status
                    is_ood = is_non_plant or energy_score > 23.0 or highest_conf < 52.0

                    if is_ood:
                        if detected_object:
                            prediction_label = f"Not A Specialized Plant (Safety Fallback: {detected_object})"
                        else:
                            prediction_label = "Not A Specialized Plant (Non-Plant Image Detected)"

                    best_result = {
                        "disease": prediction_label,
                        "confidence": round(conf, 2),
                        "view_used": view_name,
                        "catid": int(top3_catids[0].item() if hasattr(top3_catids[0], "item") else top3_catids[0]),
                        "top3": best_top3,
                        "energy_score": round(energy_score, 2),
                        "is_ood": is_ood,
                        "detected_object": detected_object
                    }

        if save_heatmap:
            try:
                from src.visualize import GradCAM, overlay_heatmap
                
                target_layer = self.model.vit.encoder.layers[-1]
                cam_engine = GradCAM(self.model, target_layer)
                
                cat_id = int(best_result["catid"]) if best_result and "catid" in best_result else None
                heatmap = cam_engine.generate_heatmap(winning_tensor, category_index=cat_id)
                vis_img = overlay_heatmap(image_path, heatmap)
                
                output_path = f"heatmap_{os.path.basename(image_path)}"
                vis_img.save(output_path, "JPEG")
                best_result["heatmap_path"] = output_path
            except Exception as cam_err:
                print(f"[inference] Grad-CAM generation skipped or failed ({cam_err}).")
                best_result["heatmap_path"] = None
            
        import gc
        gc.collect()
        return best_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a single plant image")
    parser.add_argument("image_path", help="Path to the image to classify")
    parser.add_argument("--weights", help="Path to specific model weights file")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM heatmap")
    args = parser.parse_args()
    
    predictor = DiseasePredictor(model_weights_path=args.weights)
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
