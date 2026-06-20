import os
import shutil
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DiseasePredictor
from src.advisor import SmartAdvisor

def safe_print(text):
    """Safely print Unicode strings to console, especially on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to encoding with replacement
        encoded = text.encode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(encoded.decode(sys.stdout.encoding or 'utf-8'))

def main():
    safe_print("==================================================")
    safe_print("Testing SmartFarming.AI Integration Pipeline")
    safe_print("==================================================")
    
    # Setup test file and uploads folder
    test_image = "test_sample.jpg"
    if not os.path.exists(test_image):
        safe_print(f"Error: {test_image} not found in root directory.")
        return
        
    uploads_dir = os.path.join("public", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    # 1. Test Prediction & Grad-CAM
    safe_print("\n[STEP 1] Testing Image Prediction & Grad-CAM...")
    try:
        predictor = DiseasePredictor()
        result = predictor.predict(test_image, save_heatmap=True)
        safe_print(f"--> Predicted Disease: {result['disease']}")
        safe_print(f"--> Confidence: {result['confidence']}%")
        safe_print(f"--> View Used: {result['view_used']}")
        safe_print(f"--> Energy Score: {result['energy_score']}")
        safe_print(f"--> Out-Of-Distribution (OOD): {result['is_ood']}")
        
        # Shift heatmap
        raw_heatmap = f"heatmap_{test_image}"
        if os.path.exists(raw_heatmap):
            dest_heatmap = os.path.join(uploads_dir, raw_heatmap)
            shutil.move(raw_heatmap, dest_heatmap)
            safe_print(f"--> Success: Heatmap saved to {dest_heatmap}")
        else:
            safe_print("--> Warning: Heatmap file was not generated.")
    except Exception as e:
        safe_print(f"--> Error in Step 1: {e}")
        return

    # 2. Test LLM Advice (English & Hindi)
    safe_print("\n[STEP 2] Testing LLM Treatment Advice...")
    try:
        advisor = SmartAdvisor()
        
        # Test English
        safe_print("--> Generating advice in English...")
        advice_eng = advisor.get_treatment_advice(result['disease'], result['confidence'], language="English")
        safe_print(f"--- English Overview ---\n{advice_eng.get('overview')}\n")
        safe_print(f"--- English Chemical Control ---\n{advice_eng.get('chemical')}\n")
        
        # Test Hindi
        safe_print("--> Generating advice in Hindi...")
        advice_hin = advisor.get_treatment_advice(result['disease'], result['confidence'], language="Hindi")
        safe_print("--- Hindi Overview ---")
        safe_print(advice_hin.get('overview'))
        safe_print("\n--- Hindi Biological Control ---")
        safe_print(advice_hin.get('biological'))
        safe_print("")
    except Exception as e:
        safe_print(f"--> Error in Step 2: {e}")

    # 3. Test Chatbot
    safe_print("\n[STEP 3] Testing AI Agronomist Chatbot...")
    try:
        history = [
            {"sender": "user", "text": "What is this disease?"},
            {"sender": "bot", "text": "It looks like Apple Scab."}
        ]
        query = "Can you recommend an organic treatment for it?"
        safe_print(f"--> Sending message: '{query}'")
        reply = advisor.chat_about_disease(result['disease'], history, query, language="English")
        safe_print(f"--- Chatbot Response ---\n{reply}\n")
    except Exception as e:
        safe_print(f"--> Error in Step 3: {e}")
        
    safe_print("==================================================")
    safe_print("Pipeline Testing Finished.")
    safe_print("==================================================")

if __name__ == "__main__":
    main()
