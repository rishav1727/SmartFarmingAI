import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from src.inference import DiseasePredictor
from src.advisor import SmartAdvisor

app = FastAPI(title="SmartFarmingAI", description="Multimodal Generative AI for Smart Farming")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")
UPLOAD_DIR = os.path.join(PUBLIC_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize models and services
print("Initializing AI Predictor and Smart Advisor...")
predictor = DiseasePredictor()
advisor = SmartAdvisor()

# Pre-index documents if any exist
advisor.index_documents()

# Pydantic schemas for requests
class AdviceRequest(BaseModel):
    disease: str
    confidence: float
    language: str = "English"

class ChatMessage(BaseModel):
    sender: str
    text: str

class ChatRequest(BaseModel):
    disease: str
    history: List[ChatMessage]
    message: str
    language: str = "English"

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Generate unique filename to avoid conflict
    file_ext = os.path.splitext(file.filename)[1]
    unique_id = str(uuid.uuid4())
    temp_filename = f"{unique_id}{file_ext}"
    temp_filepath = os.path.join(UPLOAD_DIR, temp_filename)
    
    # Save uploaded file
    try:
        with open(temp_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save uploaded file: {e}")
        
    # Run prediction and Grad-CAM
    try:
        # DiseasePredictor will save the heatmap to f"heatmap_{os.path.basename(temp_filepath)}" in the CWD
        result = predictor.predict(temp_filepath, save_heatmap=True)
        
        # Move generated heatmap to upload directory
        raw_heatmap_name = f"heatmap_{temp_filename}"
        raw_heatmap_path = os.path.join(BASE_DIR, raw_heatmap_name)
        heatmap_dest_path = os.path.join(UPLOAD_DIR, f"heatmap_{temp_filename}")
        
        if os.path.exists(raw_heatmap_path):
            shutil.move(raw_heatmap_path, heatmap_dest_path)
            # Update path in result relative to the web root
            result["heatmap_url"] = f"/uploads/heatmap_{temp_filename}"
        else:
            result["heatmap_url"] = None
            
        result["image_url"] = f"/uploads/{temp_filename}"
        
        return result
    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.post("/api/advice")
async def get_advice(req: AdviceRequest):
    try:
        advice = advisor.get_treatment_advice(req.disease, req.confidence, req.language)
        return advice
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate advice: {e}")

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        history_list = [{"sender": msg.sender, "text": msg.text} for msg in req.history]
        reply = advisor.chat_about_disease(req.disease, history_list, req.message, req.language)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")

# Simple in-memory session store for WhatsApp users
# Key: user phone number (e.g. "whatsapp:+14155238886")
# Value: {"disease": str, "history": list}
whatsapp_sessions = {}

@app.post("/api/whatsapp")
async def whatsapp_webhook(
    Body: Optional[str] = Form(None),
    NumMedia: int = Form(0),
    MediaUrl0: Optional[str] = Form(None),
    From: str = Form(...),
    To: str = Form(...)
):
    import requests
    from fastapi.responses import Response
    
    xml_response = ""
    user_id = From
    
    # 1. Image diagnostic upload
    if NumMedia > 0 and MediaUrl0:
        try:
            # Download image
            r = requests.get(MediaUrl0, stream=True)
            if r.status_code == 200:
                unique_id = str(uuid.uuid4())
                filename = f"whatsapp_{unique_id}.jpg"
                temp_filepath = os.path.join(UPLOAD_DIR, filename)
                
                with open(temp_filepath, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
                    
                # Run prediction
                result = predictor.predict(temp_filepath, save_heatmap=True)
                
                # Move heatmap
                raw_heatmap_name = f"heatmap_{filename}"
                raw_heatmap_path = os.path.join(BASE_DIR, raw_heatmap_name)
                heatmap_dest_path = os.path.join(UPLOAD_DIR, f"heatmap_{filename}")
                if os.path.exists(raw_heatmap_path):
                    shutil.move(raw_heatmap_path, heatmap_dest_path)
                    
                # Get advice
                disease = result["disease"]
                confidence = result["confidence"]
                
                advice = advisor.get_treatment_advice(disease, confidence, language="English")
                
                # Update session
                whatsapp_sessions[user_id] = {
                    "disease": disease,
                    "history": [
                        {"sender": "user", "text": "Diagnose image"},
                        {"sender": "bot", "text": f"Diagnosed: {disease} ({confidence}%)"}
                    ]
                }
                
                # Format response
                msg = (
                    f"*Crop Diagnosis Report*\n"
                    f"====================\n"
                    f"*Detected:* {disease.replace('___', ': ').replace('_', ' ')}\n"
                    f"*Confidence:* {confidence:.1f}%\n\n"
                    f"*Overview:* {advice.get('overview', '')[:200]}...\n\n"
                    f"*Chemical:* {advice.get('chemical', '')[:200]}...\n\n"
                    f"*Biological:* {advice.get('biological', '')[:200]}...\n\n"
                    f"View Explainable Heatmap overlay:\n"
                    f"http://your-server-ip/uploads/heatmap_{filename}\n\n"
                    f"Reply to ask follow-up questions!"
                )
                
                xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Message><Body>{msg}</Body></Message>
                </Response>"""
            else:
                xml_response = """<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Message><Body>Error downloading image from Twilio servers.</Body></Message>
                </Response>"""
        except Exception as e:
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Message><Body>Diagnostic error: {str(e)}</Body></Message>
            </Response>"""
            
    # 2. Conversational follow-ups
    else:
        user_message = Body.strip() if Body else ""
        if user_id in whatsapp_sessions:
            session = whatsapp_sessions[user_id]
            try:
                reply = advisor.chat_about_disease(session["disease"], session["history"], user_message, language="English")
                # Update session history
                session["history"].append({"sender": "user", "text": user_message})
                session["history"].append({"sender": "bot", "text": reply})
                
                xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Message><Body>{reply}</Body></Message>
                </Response>"""
            except Exception as e:
                xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
                <Response>
                    <Message><Body>Error generating response: {str(e)}</Body></Message>
                </Response>"""
        else:
            # Welcome message
            msg = (
                "Welcome to *SmartFarming.AI* WhatsApp Assistant!\n\n"
                "Please upload a clear close-up picture of a plant leaf (supported crops: Apple, Potato, Tomato, Grape, Corn) to run the diagnostic health check and receive explainable heatmaps and treatment advice."
            )
            xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
            <Response>
                <Message><Body>{msg}</Body></Message>
            </Response>"""
            
    return Response(content=xml_response, media_type="application/xml")

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    # Validate extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".txt", ".md", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only .txt, .md, and .pdf documents are supported.")
    
    dest_path = os.path.join(advisor.doc_dir, filename)
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Re-index documents
        advisor.index_documents()
        return {"status": "success", "message": f"Successfully uploaded and indexed '{filename}'. Now has {len(advisor.chunks)} active RAG chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {e}")

@app.post("/api/index-docs")
async def index_documents():
    success = advisor.index_documents()
    if success:
        return {"status": "success", "message": f"Successfully indexed documents. System now has {len(advisor.chunks)} active chunks."}
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No documents found to index or indexing failed."})

# Serve Frontend
@app.get("/")
async def get_index():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to SmartFarmingAI API. Frontend dashboard files are missing."}

# Mount static files (uploads and other assets)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
