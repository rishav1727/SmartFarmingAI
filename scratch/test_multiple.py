import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DiseasePredictor

images = [
    r"C:\Users\RISHAV\Downloads\download.htm",
    r"C:\Users\RISHAV\Downloads\images (2).jpg",
    r"C:\Users\RISHAV\Downloads\disease-on-pear-tree-leaves-260nw-2686391131.webp"
]

predictor = DiseasePredictor()

for img in images:
    print(f"\nEvaluating: {img}")
    if not os.path.exists(img):
        print("  -> File not found!")
        continue
    
    try:
        result = predictor.predict(img, save_heatmap=True)
        print(f"  -> Detected Disease: {result['disease']}")
        print(f"  -> Confidence: {result['confidence']}%")
        print(f"  -> Analysis Method used: {result['view_used']}")
    except Exception as e:
        print(f"  -> Failed to analyze: {e}")
