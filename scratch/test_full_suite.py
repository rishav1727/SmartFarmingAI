import os
import sys
import json
from fastapi.testclient import TestClient

# Ensure root dir in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app

def run_tests():
    print("=== STARTING FULL SYSTEM INTEGRATION CHECK ===")
    flags = []
    
    client = TestClient(app)
    
    # 1. Test Root / Index Route
    res = client.get("/")
    if res.status_code == 200:
        print("[OK] Root Route (GET /)")
        if '<button id="print-btn"' not in res.text:
            flags.append("Warning: print-btn button missing in index.html response")
        if 'data-i18n="printBtn"' not in res.text:
            flags.append("Warning: data-i18n printBtn missing in index.html response")
    else:
        flags.append(f"CRITICAL: Root route failed with status {res.status_code}")
        
    # 2. Test Samples Endpoint
    res = client.get("/api/samples")
    if res.status_code == 200 and "samples" in res.json():
        print("[OK] Samples Endpoint (GET /api/samples)")
    else:
        flags.append(f"ERROR: /api/samples failed with status {res.status_code}")
        
    # 3. Test Index Docs Endpoint
    res = client.post("/api/index-docs")
    if res.status_code == 200:
        print("[OK] RAG Index Docs Endpoint (POST /api/index-docs)")
    else:
        flags.append(f"WARNING: /api/index-docs failed with status {res.status_code}")
        
    # 4. Test Predict Endpoint with test_sample.jpg & Crop-Specific Filter Mode
    test_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_sample.jpg")
    if os.path.exists(test_img_path):
        with open(test_img_path, "rb") as f:
            res = client.post("/api/predict", files={"file": ("test_sample.jpg", f, "image/jpeg")}, data={"crop": "tomato"})
        if res.status_code == 200:
            data = res.json()
            print(f"[OK] AI Prediction Endpoint with Crop Filter Mode (POST /api/predict [crop=tomato]) -> Disease: {data.get('disease')}, Conf: {data.get('confidence')}%")
            if not data.get("heatmap_url"):
                flags.append("WARNING: Grad-CAM heatmap_url was not returned in prediction")
        else:
            flags.append(f"CRITICAL: /api/predict failed with status {res.status_code}: {res.text}")
    else:
        flags.append("NOTICE: test_sample.jpg not found for prediction test")

    # 5. Test Advisory Endpoint
    res = client.post("/api/advice", json={
        "disease": "Tomato___Bacterial_spot",
        "confidence": 98.5,
        "language": "English"
    })
    if res.status_code == 200:
        data = res.json()
        print("[OK] Smart Advisory Endpoint (POST /api/advice)")
        if not data.get("overview") or not data.get("chemical"):
            flags.append("WARNING: Treatment advisory response incomplete")
    else:
        flags.append(f"CRITICAL: /api/advice failed with status {res.status_code}: {res.text}")
        
    # 6. Test Chat Endpoint
    res = client.post("/api/chat", json={
        "disease": "Tomato___Bacterial_spot",
        "history": [],
        "message": "What is the best pesticide spray for this condition?",
        "language": "English"
    })
    if res.status_code == 200:
        data = res.json()
        print("[OK] Kisan AI Chat Endpoint (POST /api/chat)")
        if not data.get("reply"):
            flags.append("WARNING: Chat response reply text empty")
    else:
        flags.append(f"CRITICAL: /api/chat failed with status {res.status_code}: {res.text}")

    print("\n=== TEST RESULTS SUMMARY ===")
    if not flags:
        print("[SUCCESS] ALL TESTS PASSED! NO FLAGS DETECTED.")
    else:
        print(f"[FLAG] {len(flags)} FLAG(S) DETECTED:")
        for f in flags:
            print("  -", f)

if __name__ == "__main__":
    run_tests()
