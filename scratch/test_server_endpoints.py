import os
import requests
import json
import sys

def main():
    print("==================================================")
    print("Programmatic Verification of FastAPI Server")
    print("==================================================")
    
    server_url = "http://127.0.0.1:8000"
    
    # 1. Test server connection (GET /)
    print("\n[TEST 1] Testing Connection to server root...")
    try:
        r = requests.get(server_url)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print("--> Success: Root page served successfully.")
        else:
            print("--> Failed: Root returned unexpected status.")
            return
    except Exception as e:
        print(f"--> Failed to connect to server: {e}")
        print("Please check if the FastAPI server is running on http://127.0.0.1:8000")
        return

    # 2. Test static assets (GET /style.css)
    print("\n[TEST 2] Testing Static File Delivery (/style.css)...")
    try:
        r = requests.get(f"{server_url}/style.css")
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            print(f"--> Success: style.css is active ({len(r.text)} bytes).")
        else:
            print("--> Failed to serve static files.")
    except Exception as e:
        print(f"--> Error: {e}")

    # 3. Test prediction endpoint (POST /api/predict)
    print("\n[TEST 3] Testing File Upload & Inference (/api/predict)...")
    test_image = "test_sample.jpg"
    if not os.path.exists(test_image):
        print(f"--> Skipped: {test_image} not found in current directory.")
    else:
        try:
            with open(test_image, "rb") as f:
                files = {"file": (test_image, f, "image/jpeg")}
                r = requests.post(f"{server_url}/api/predict", files=files)
            print(f"Status Code: {r.status_code}")
            if r.status_code == 200:
                result = r.json()
                print("--> Success! Prediction output:")
                print(f"    - Disease: {result.get('disease')}")
                print(f"    - Confidence: {result.get('confidence')}%")
                print(f"    - Image URL: {result.get('image_url')}")
                print(f"    - Heatmap URL: {result.get('heatmap_url')}")
                print(f"    - OOD safety triggered: {result.get('is_ood')}")
                
                # Store prediction for next tests
                disease_name = result.get('disease')
                confidence = result.get('confidence')
                
                # Test advice endpoint (POST /api/advice)
                print("\n[TEST 4] Testing LLM Treatment Advice (/api/advice) in English...")
                advice_payload = {
                    "disease": disease_name,
                    "confidence": confidence,
                    "language": "English"
                }
                r_adv = requests.post(f"{server_url}/api/advice", json=advice_payload)
                print(f"Status Code: {r_adv.status_code}")
                if r_adv.status_code == 200:
                    advice = r_adv.json()
                    print("--> Success! Overview:")
                    print(f"    {advice.get('overview')[:100]}...")
                else:
                    print("--> Failed: Advice endpoint error.")
                    
                # Test chat endpoint (POST /api/chat)
                print("\n[TEST 5] Testing AI Agronomist Chatbot (/api/chat)...")
                chat_payload = {
                    "disease": disease_name,
                    "history": [
                        {"sender": "user", "text": "Hi, what disease is this?"},
                        {"sender": "bot", "text": f"This looks like {disease_name}."}
                    ],
                    "message": "Is it safe to compost the leaves?",
                    "language": "English"
                }
                r_chat = requests.post(f"{server_url}/api/chat", json=chat_payload)
                print(f"Status Code: {r_chat.status_code}")
                if r_chat.status_code == 200:
                    chat_res = r_chat.json()
                    print("--> Success! Chatbot reply:")
                    print(f"    {chat_res.get('reply')[:150]}...")
                else:
                    print("--> Failed: Chatbot endpoint error.")
                    
            else:
                print(f"--> Failed: {r.text}")
        except Exception as e:
            print(f"--> Error: {e}")

    print("\n==================================================")
    print("Verification Completed.")
    print("==================================================")

if __name__ == "__main__":
    main()
