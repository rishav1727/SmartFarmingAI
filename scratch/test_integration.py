import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

print("==================================================")
print("SMARTFARMING AI - FULL INTEGRATION TEST SUITE")
print("==================================================")

# 1. Test APK Download Endpoint
print("\n[Test 1] Testing APK Download Endpoint (/download/apk)...")
try:
    r = requests.get(f"{BASE_URL}/download/apk", stream=True)
    if r.status_code == 200 and r.headers.get("content-type") == "application/vnd.android.package-archive":
        print(f"  [PASS] SUCCESS: APK Download Endpoint active ({len(r.content)} bytes)")
    else:
        print(f"  [FAIL] FAILED: HTTP {r.status_code}, Content-Type: {r.headers.get('content-type')}")
except Exception as e:
    print(f"  [ERROR] ERROR: {e}")

# 2. Test Health Diagnostic & Grad-CAM Heatmap (/api/predict)
print("\n[Test 2] Testing Image Diagnostic & Grad-CAM Engine (/api/predict)...")
test_img_path = "test_sample.jpg"
if os.path.exists(test_img_path):
    try:
        with open(test_img_path, "rb") as f:
            files = {"file": (test_img_path, f, "image/jpeg")}
            r = requests.post(f"{BASE_URL}/api/predict", files=files)
            
        if r.status_code == 200:
            res = r.json()
            print(f"  [PASS] SUCCESS: Diagnosed: {res.get('disease')}")
            print(f"     Confidence: {res.get('confidence')}%")
            print(f"     View Used: {res.get('view_used')}")
            print(f"     Energy Score: {res.get('energy_score')}")
            print(f"     Image URL: {res.get('image_url')}")
            print(f"     Heatmap URL: {res.get('heatmap_url')}")
            print(f"     Top-3 Matches: {json.dumps(res.get('top3'))}")
        else:
            print(f"  [FAIL] FAILED: HTTP {r.status_code} - {r.text}")
    except Exception as e:
        print(f"  [ERROR] ERROR: {e}")
else:
    print("  [SKIP] Skipped: test_sample.jpg not found")

# 3. Test Treatment Advice Generation (/api/advice) in English
print("\n[Test 3] Testing Treatment Advice API (/api/advice) [English]...")
try:
    payload = {
        "disease": "Apple___Apple_scab",
        "confidence": 98.5,
        "language": "English"
    }
    r = requests.post(f"{BASE_URL}/api/advice", json=payload)
    if r.status_code == 200:
        res = r.json()
        print(f"  [PASS] SUCCESS: Overview: {res.get('overview')[:100]}...")
        print(f"     Chemical: {res.get('chemical')[:100]}...")
        print(f"     Biological: {res.get('biological')[:100]}...")
    else:
        print(f"  [FAIL] FAILED: HTTP {r.status_code} - {r.text}")
except Exception as e:
    print(f"  [ERROR] ERROR: {e}")

# 4. Test Treatment Advice Generation (/api/advice) in Hindi
print("\n[Test 4] Testing Treatment Advice API (/api/advice) [Hindi]...")
try:
    payload = {
        "disease": "Apple___Apple_scab",
        "confidence": 98.5,
        "language": "Hindi"
    }
    r = requests.post(f"{BASE_URL}/api/advice", json=payload)
    if r.status_code == 200:
        res = r.json()
        print(f"  [PASS] SUCCESS: Hindi Overview: {res.get('overview')[:100]}...")
    else:
        print(f"  [FAIL] FAILED: HTTP {r.status_code} - {r.text}")
except Exception as e:
    print(f"  [ERROR] ERROR: {e}")

# 5. Test AI Agronomist Chatbot (/api/chat)
print("\n[Test 5] Testing AI Chatbot Q&A (/api/chat)...")
try:
    payload = {
        "disease": "Apple___Apple_scab",
        "history": [{"sender": "user", "text": "Diagnose image"}],
        "message": "How often should I spray Neem oil?",
        "language": "English"
    }
    r = requests.post(f"{BASE_URL}/api/chat", json=payload)
    if r.status_code == 200:
        res = r.json()
        print(f"  [PASS] SUCCESS: Chatbot Reply: {res.get('reply')[:120]}...")
    else:
        print(f"  [FAIL] FAILED: HTTP {r.status_code} - {r.text}")
except Exception as e:
    print(f"  [ERROR] ERROR: {e}")

print("\n==================================================")
print("TEST SUITE COMPLETE")
print("==================================================")
