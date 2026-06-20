import os
import sys
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import json

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DiseasePredictor

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))

def apply_blur(image, radius=5):
    return image.filter(ImageFilter.GaussianBlur(radius))

def apply_noise(image, amount=0.15):
    # Add salt and pepper noise
    arr = np.array(image).astype(float)
    h, w, c = arr.shape
    num_noise = int(amount * h * w)
    
    # Salt
    y_coords = np.random.randint(0, h, num_noise)
    x_coords = np.random.randint(0, w, num_noise)
    arr[y_coords, x_coords, :] = 255.0
    
    # Pepper
    y_coords = np.random.randint(0, h, num_noise)
    x_coords = np.random.randint(0, w, num_noise)
    arr[y_coords, x_coords, :] = 0.0
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def apply_brightness(image, factor=0.2):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_contrast(image, factor=0.2):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)

def run_harsh_tests():
    safe_print("==================================================")
    safe_print("STARTING HARSH STRESS & ROBUSTNESS TEST ON VIT MODEL")
    safe_print("==================================================")
    
    test_image_path = "test_sample.jpg"
    if not os.path.exists(test_image_path):
        safe_print(f"Error: {test_image_path} not found. Cannot run stress test.")
        return
        
    predictor = DiseasePredictor()
    base_img = Image.open(test_image_path).convert("RGB")
    
    # Define test conditions
    tests = {
        "1. Baseline (Normal Leaf)": lambda img: img,
        "2. Extreme Blur (Unfocused Camera)": lambda img: apply_blur(img, radius=8),
        "3. High Sensor Noise (Salt & Pepper)": lambda img: apply_noise(img, amount=0.2),
        "4. Extreme Underexposure (Dark/Night)": lambda img: apply_brightness(img, factor=0.15),
        "5. Extreme Overexposure (Bright Sunlight)": lambda img: apply_brightness(img, factor=3.5),
        "6. Low Contrast (Faded/Gray Leaf)": lambda img: apply_contrast(img, factor=0.1),
        "7. Empty/Flat Black (Sensor Cover Closed)": lambda img: Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    }
    
    os.makedirs("scratch/harsh_samples", exist_ok=True)
    report = []
    
    for name, transform_func in tests.items():
        safe_print(f"\nRunning stress case: {name}...")
        try:
            # Transform image
            stressed_img = transform_func(base_img)
            
            # Save sample to disk to inspect visually
            filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("&", "and").replace("/", "_") + ".jpg"
            sample_path = os.path.join("scratch", "harsh_samples", filename)
            stressed_img.save(sample_path)
            
            # Run prediction
            result = predictor.predict(sample_path, save_heatmap=False)
            
            # Format output
            safe_print(f"  -> Prediction: {result['disease']}")
            safe_print(f"  -> Confidence: {result['confidence']}%")
            safe_print(f"  -> Energy Score: {result['energy_score']}")
            safe_print(f"  -> OOD Triggered: {result['is_ood']}")
            
            report.append({
                "test_name": name,
                "disease": result['disease'],
                "confidence": result['confidence'],
                "energy": result['energy_score'],
                "is_ood": result['is_ood'],
                "sample_saved": sample_path
            })
            
        except Exception as e:
            safe_print(f"  -> Test crashed with error: {e}")
            report.append({
                "test_name": name,
                "error": str(e)
            })

    # Summary Report
    safe_print("\n==================================================")
    safe_print("HARSH TEST COMPLETED. STABILITY REPORT SUMMARY:")
    safe_print("==================================================")
    
    for entry in report:
        if "error" in entry:
            safe_print(f"[{entry['test_name']}] - CRASHED: {entry['error']}")
        else:
            status = "STABLE"
            if entry['is_ood']:
                status = "SAFE (OOD REJECTED)"
            elif entry['confidence'] < 60.0:
                status = "CONFUSED"
            
            safe_print(f"[{entry['test_name']}]")
            safe_print(f"  Status: {status} | Prediction: {entry['disease']} | Conf: {entry['confidence']}% | Energy: {entry['energy']}")
            
    safe_print("\nStress samples have been saved in: scratch/harsh_samples/")
    safe_print("==================================================")

if __name__ == "__main__":
    run_harsh_tests()
