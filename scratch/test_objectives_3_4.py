import os
import sys

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import DiseasePredictor
from src.advisor import SmartAdvisor

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))

def test_objective_3():
    safe_print("\n" + "="*50)
    safe_print("TESTING OBJECTIVE 3: LLM ADVICE IN LOCAL LANGUAGES")
    safe_print("="*50)
    
    advisor = SmartAdvisor()
    disease = "Apple___Apple_scab"
    confidence = 98.5
    
    # 1. Test English Advice
    safe_print("\n[Objective 3.1] Generating Advice in English...")
    advice_eng = advisor.get_treatment_advice(disease, confidence, language="English")
    
    # Verify structure
    keys = ["overview", "chemical", "biological", "preventative", "advice"]
    missing_eng = [k for k in keys if k not in advice_eng]
    if not missing_eng:
        safe_print("--> SUCCESS: English advice generated with all required sections.")
        safe_print(f"    * Overview snippet: {advice_eng['overview'][:100]}...")
    else:
        safe_print(f"--> FAILED: English advice missing keys: {missing_eng}")
        return False
        
    # 2. Test Hindi Advice
    safe_print("\n[Objective 3.2] Generating Advice in Hindi...")
    advice_hin = advisor.get_treatment_advice(disease, confidence, language="Hindi")
    missing_hin = [k for k in keys if k not in advice_hin]
    if not missing_hin:
        safe_print("--> SUCCESS: Hindi advice generated with all required sections.")
        # Print safely replacing Hindi characters on Windows terminal
        safe_print("    * Hindi Overview snippet:")
        safe_print(advice_hin['overview'][:100] + "...")
    else:
        safe_print(f"--> FAILED: Hindi advice missing keys: {missing_hin}")
        return False
        
    # 3. Test Anomaly Rejection Advice
    safe_print("\n[Objective 3.3] Generating Advice for Anomaly (OOD Rejection)...")
    advice_anom = advisor.get_treatment_advice("Not A Specialized Plant (Safety Fallback: cat)", 15.0, language="English")
    if advice_anom and "Anomaly Detected" in advice_anom.get("overview", ""):
        safe_print("--> SUCCESS: Advisor handled non-plant anomaly rejection safely.")
        safe_print(f"    * Anomaly Overview: {advice_anom['overview']}")
    else:
        safe_print("--> FAILED: Advisor did not output correct safety response for anomaly.")
        return False
        
    safe_print("\n--> OBJECTIVE 3 TEST PASSED SUCCESSFULLY.")
    return True

def test_objective_4():
    safe_print("\n" + "="*50)
    safe_print("TESTING OBJECTIVE 4: 'DETECT + EXPLAIN + CURE' MULTIMODAL WORKFLOW")
    safe_print("="*50)
    
    test_image = "test_sample.jpg"
    if not os.path.exists(test_image):
        safe_print(f"--> FAILED: Test image '{test_image}' not found.")
        return False
        
    predictor = DiseasePredictor()
    advisor = SmartAdvisor()
    
    # STEP 1: DETECT
    safe_print("\n[Step A - DETECT] Running Vision Transformer classification...")
    result = predictor.predict(test_image, save_heatmap=False)
    disease = result['disease']
    confidence = result['confidence']
    safe_print(f"--> Class identified: {disease} (Confidence: {confidence}%)")
    
    # STEP 2: EXPLAIN
    safe_print("\n[Step B - EXPLAIN] Generating neural activation heatmap (Grad-CAM)...")
    result_with_cam = predictor.predict(test_image, save_heatmap=True)
    heatmap_path = result_with_cam.get('heatmap_path')
    if heatmap_path and os.path.exists(heatmap_path):
        safe_print(f"--> Attention overlay successfully generated at: {heatmap_path}")
    else:
        safe_print("--> FAILED: Grad-CAM attention overlay file not generated.")
        return False
        
    # STEP 3: CURE
    safe_print("\n[Step C - CURE] Generating customized chemical & biological remedies...")
    advice = advisor.get_treatment_advice(disease, confidence, language="English")
    if advice:
        safe_print("--> Success! Cure protocol generated:")
        safe_print(f"    * Chemical: {advice.get('chemical')[:80]}...")
        safe_print(f"    * Biological: {advice.get('biological')[:80]}...")
    else:
        safe_print("--> FAILED: Could not generate cure advice.")
        return False
        
    safe_print("\n--> OBJECTIVE 4 TEST PASSED SUCCESSFULLY.")
    safe_print("Unified workflow coordinates Image (Visual input) -> Heatmap (Visual explanation) -> Advice (Textual cure).")
    return True

if __name__ == "__main__":
    r3 = test_objective_3()
    r4 = False
    if r3:
        r4 = test_objective_4()
        
    if r3 and r4:
        safe_print("\n" + "="*50)
        safe_print("ALL TESTS COMPLETED SUCCESSFULLY!")
        safe_print("="*50)
        sys.exit(0)
    else:
        safe_print("\n" + "="*50)
        safe_print("SOME TESTS FAILED. CHECK LOGS.")
        safe_print("="*50)
        sys.exit(1)
