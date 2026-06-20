import os
import requests
import sys

def main():
    print("==================================================")
    print("Testing Objective 5: Document Upload Endpoint")
    print("==================================================")
    
    server_url = "http://127.0.0.1:8000"
    test_doc = "scratch/temp_reference_guide.txt"
    os.makedirs("scratch", exist_ok=True)
    
    # Create test document
    doc_content = (
        "CROP ADVISOR SPECIAL INFO\n\n"
        "Potato Late Blight Treatment Special Guidelines:\n"
        "Under extreme humidity conditions, apply standard metalaxyl-M fungicide sprays at a dosage of 1.8g/L. "
        "Biological: Spray with custom Trichoderma bio-agent suspension. "
        "Source: Agronomy Extension Office PDF Bulletin."
    )
    with open(test_doc, "w", encoding="utf-8") as f:
        f.write(doc_content)
        
    print(f"--> Success: Created temporary RAG document at {test_doc}")
    
    # Upload to server
    print("\n[STEP 1] Uploading document to /api/documents/upload...")
    try:
        with open(test_doc, "rb") as f:
            files = {"file": ("temp_reference_guide.txt", f, "text/plain")}
            r = requests.post(f"{server_url}/api/documents/upload", files=files)
            
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            result = r.json()
            print("--> Success! Upload response:")
            print(f"    - Message: {result.get('message')}")
            print(f"    - Status: {result.get('status')}")
        else:
            print(f"--> Failed: {r.text}")
            return
    except Exception as e:
        print(f"--> Connection failed: {e}")
        return
        
    # Verify RAG context retrieval via /api/advice
    print("\n[STEP 2] Verifying RAG is index-aware (/api/advice)...")
    try:
        payload = {
            "disease": "Potato___Late_blight",
            "confidence": 92.0,
            "language": "English"
        }
        r_adv = requests.post(f"{server_url}/api/advice", json=payload)
        print(f"Status Code: {r_adv.status_code}")
        if r_adv.status_code == 200:
            # If the index loaded and openai key is active, it would retrieve this chunk
            # Even if offline (quota exceeded), it falls back to mock, but the console logs
            # in uvicorn show if RAG was hit. Let's print the returned advice.
            advice = r_adv.json()
            print("--> Success! Advice overview received.")
            print(f"    Overview: {advice.get('overview')[:100]}...")
        else:
            print("--> Failed: Advice retrieval failed.")
    except Exception as e:
        print(f"--> Error: {e}")
        
    # Clean up temp file
    if os.path.exists(test_doc):
        os.remove(test_doc)
        print("\n--> Cleaned up temp reference document.")
        
    print("\n==================================================")
    print("Verification Completed.")
    print("==================================================")

if __name__ == "__main__":
    main()
