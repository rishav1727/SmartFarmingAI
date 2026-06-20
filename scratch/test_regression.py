import os
import sys
import torch
import numpy as np
from PIL import Image
import json

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.consts import MODEL_DIR
from src.inference import DiseasePredictor
from src.visualize import GradCAM

def safe_print(text):
    """Safely print text to standard out, fallback for Windows encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))

def test_model_loading():
    safe_print("\n[REGRESSION 1] Testing PyTorch Model Architecture...")
    try:
        # Load class names size
        class_names_path = os.path.join(MODEL_DIR, "class_names.json")
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        
        model = get_model(num_classes, 'cpu')
        
        # Verify model shape
        head_in = model.vit.heads.head[0].in_features
        head_out = model.vit.heads.head[3].out_features
        
        assert head_in == 768, f"Expected ViT hidden size of 768, got {head_in}"
        assert head_out == num_classes, f"Expected output features to match class size {num_classes}, got {head_out}"
        
        safe_print(f"--> SUCCESS: Model architecture loaded correctly. Classes: {num_classes}, Hidden Dim: {head_in}")
        return True
    except Exception as e:
        safe_print(f"--> FAILED Model Loading: {e}")
        return False

def test_inference_pipeline():
    safe_print("\n[REGRESSION 2] Testing Inference Pipeline & OOD Safety Fallbacks...")
    try:
        predictor = DiseasePredictor()
        
        # Test 1: Real leaf sample
        test_image = "test_sample.jpg"
        if os.path.exists(test_image):
            result = predictor.predict(test_image, save_heatmap=False)
            safe_print(f"--> Real Leaf Diagnosis: '{result['disease']}' (OOD: {result['is_ood']}, Energy: {result['energy_score']})")
            assert not result['is_ood'], "Expected real leaf sample to be IN-DISTRIBUTION."
        else:
            safe_print(f"--> Warning: {test_image} not found, skipping real-leaf test.")
            
        # Test 2: Anomaly / Random Noise (Out-of-Distribution testing)
        noise_image_path = "scratch/temp_noise.jpg"
        os.makedirs("scratch", exist_ok=True)
        # Create a random noise image
        noise_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise_arr)
        noise_img.save(noise_image_path)
        
        result_ood = predictor.predict(noise_image_path, save_heatmap=False)
        safe_print(f"--> Noise Input Diagnosis: '{result_ood['disease']}' (OOD: {result_ood['is_ood']}, Energy: {result_ood['energy_score']})")
        
        # Clean up temp file
        if os.path.exists(noise_image_path):
            os.remove(noise_image_path)
            
        assert result_ood['is_ood'], "Expected random noise to trigger OUT-OF-DISTRIBUTION safety fallback."
        safe_print("--> SUCCESS: OOD rejection logic and ImageNet safety fallbacks successfully validated.")
        return True
    except Exception as e:
        safe_print(f"--> FAILED Inference Pipeline: {e}")
        return False

def test_gradcam():
    safe_print("\n[REGRESSION 3] Testing Grad-CAM Activation Mapping...")
    try:
        class_names_path = os.path.join(MODEL_DIR, "class_names.json")
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
        num_classes = len(class_names)
        
        model = get_model(num_classes, 'cpu')
        
        # Setup GradCAM
        target_layer = model.vit.encoder.layers[-1]
        cam_engine = GradCAM(model, target_layer)
        
        # Input tensor
        dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
        
        heatmap = cam_engine.generate_heatmap(dummy_tensor, category_index=0)
        
        assert heatmap.shape == (224, 224), f"Expected heatmap shape (224, 224), got {heatmap.shape}"
        assert np.max(heatmap) <= 1.0 and np.min(heatmap) >= 0.0, "Expected heatmap values normalized [0, 1]"
        
        safe_print("--> SUCCESS: Grad-CAM activations successfully generated and normalized.")
        return True
    except Exception as e:
        safe_print(f"--> FAILED Grad-CAM Test: {e}")
        return False

def test_jit_model():
    safe_print("\n[REGRESSION 4] Testing TorchScript JIT Model Loading...")
    jit_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(jit_path):
        safe_print(f"--> Warning: JIT model {jit_path} not found. Skipping JIT regression.")
        return True
        
    try:
        # Load JIT model
        model = torch.jit.load(jit_path)
        model.eval()
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            
        assert output.shape[0] == 1, f"Expected output batch size 1, got {output.shape[0]}"
        safe_print(f"--> SUCCESS: TorchScript JIT model parsed and compiled successfully. Outputs shape: {list(output.shape)}")
        return True
    except Exception as e:
        safe_print(f"--> FAILED JIT Model loading: {e}")
        return False

def run_all_tests():
    safe_print("==================================================")
    safe_print("Running SmartFarmingAI Regression Test Suite")
    safe_print("==================================================")
    
    r1 = test_model_loading()
    r2 = test_inference_pipeline()
    r3 = test_gradcam()
    r4 = test_jit_model()
    
    safe_print("\n==================================================")
    safe_print("Regression Test Summary:")
    safe_print(f"  1. Model Architecture Check: {'PASSED' if r1 else 'FAILED'}")
    safe_print(f"  2. Inference & OOD Safety Check: {'PASSED' if r2 else 'FAILED'}")
    safe_print(f"  3. Grad-CAM Hooks Check: {'PASSED' if r3 else 'FAILED'}")
    safe_print(f"  4. JIT Compiler Check: {'PASSED' if r4 else 'FAILED'}")
    safe_print("==================================================")
    
    if all([r1, r2, r3, r4]):
        safe_print("All Regression Tests Passed Successfully!")
        sys.exit(0)
    else:
        safe_print("One or more regression tests failed. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
