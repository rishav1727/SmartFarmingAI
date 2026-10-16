import os
import sys
import shutil
import uuid
import json
from datetime import datetime

# Load .env file if present (local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session

from src.inference import DiseasePredictor
from src.advisor import SmartAdvisor
from src.database import init_db, get_db, Contribution, ModelVersion
from src import storage as store

app = FastAPI(
    title="SmartFarmingAI",
    description="Multimodal Generative AI for Smart Farming — with community retraining",
    version="2.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC_DIR   = os.path.join(BASE_DIR, "public")
UPLOAD_DIR   = os.path.join(PUBLIC_DIR, "uploads")
CONTRIB_DIR  = os.path.join(PUBLIC_DIR, "contributions")
os.makedirs(UPLOAD_DIR,  exist_ok=True)
os.makedirs(CONTRIB_DIR, exist_ok=True)

# Admin secret for protected endpoints
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "change_me")

# Initialize database tables
print("Initializing database...")
init_db()

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
async def predict(file: UploadFile = File(...), crop: Optional[str] = Form(None)):
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
        result = predictor.predict(temp_filepath, save_heatmap=True, target_crop=crop)
        
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

@app.get("/api/samples")
async def get_samples():
    samples_dir = os.path.join(PUBLIC_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_files = [
        {"id": "apple", "name": "Apple Scab Sample", "crop": "Apple", "url": "/samples/apple_scab.jpg"},
        {"id": "potato", "name": "Potato Late Blight Sample", "crop": "Potato", "url": "/samples/potato_blight.jpg"},
        {"id": "tomato", "name": "Tomato Leaf Mold Sample", "crop": "Tomato", "url": "/samples/tomato_mold.jpg"},
        {"id": "corn", "name": "Corn Common Rust Sample", "crop": "Corn", "url": "/samples/corn_rust.jpg"},
    ]
    return {"samples": sample_files}

# Download Android APK Endpoint
@app.get("/download/apk")
async def download_apk():
    apk_path = os.path.join(PUBLIC_DIR, "SmartFarmingAI.apk")
    if os.path.exists(apk_path):
        return FileResponse(
            apk_path,
            filename="SmartFarmingAI.apk",
            media_type="application/vnd.android.package-archive"
        )
    raise HTTPException(status_code=404, detail="APK file not found on server.")

# Serve Frontend
@app.get("/")
async def get_index():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to SmartFarmingAI API. Frontend dashboard files are missing."}


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNITY CONTRIBUTION & RETRAINING ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

class ContributeRequest(BaseModel):
    label: str
    crop_type: Optional[str] = None
    contributor_name: Optional[str] = None
    contributor_email: Optional[str] = None
    notes: Optional[str] = None

@app.post("/api/contribute", summary="Submit a labeled image for retraining")
async def contribute_image(
    file: UploadFile = File(...),
    label: str = Form(...),
    crop_type: Optional[str] = Form(None),
    contributor_name: Optional[str] = Form(None),
    contributor_email: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload a labeled plant image to help improve the model.

    - **file** : image file (jpg / png)
    - **label**: disease class name, e.g. `Tomato___Late_blight`
                 (must match one of the class names in class_names.json)
    - **crop_type**, **contributor_name**, **contributor_email** : optional
    """
    # Validate file type
    ext = os.path.splitext(file.filename or "image.jpg")[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=400, detail="Only JPG / PNG / WebP images are accepted.")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="Image must be smaller than 20 MB.")

    # Upload to cloud / local storage
    try:
        public_url, storage_path = store.upload_image(file_bytes, file.filename or "contribution.jpg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {e}")

    # Save record in DB
    record = Contribution(
        image_url        = public_url,
        storage_path     = storage_path,
        label            = label,
        crop_type        = crop_type,
        contributor_name = contributor_name,
        contributor_email= contributor_email,
        notes            = notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "status":      "success",
        "id":          record.id,
        "image_url":   public_url,
        "label":       label,
        "message":     "Thank you! Your image has been submitted. After admin verification it will be used to improve the model.",
        "submitted_at": record.submitted_at.isoformat(),
    }


@app.get("/api/contributions", summary="List community-contributed training images")
async def list_contributions(
    verified_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Returns community-submitted images. Use `verified_only=true` to see only approved ones."""
    query = db.query(Contribution)
    if verified_only:
        query = query.filter(Contribution.verified == True)  # noqa: E712
    total  = query.count()
    items  = query.order_by(Contribution.submitted_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "contributions": [
            {
                "id":              c.id,
                "image_url":       c.image_url,
                "label":           c.label,
                "crop_type":       c.crop_type,
                "contributor_name":c.contributor_name,
                "verified":        c.verified,
                "used_in_training":c.used_in_training,
                "submitted_at":    c.submitted_at.isoformat() if c.submitted_at else None,
            }
            for c in items
        ],
    }


@app.get("/api/model/info", summary="Current model version and statistics")
async def model_info(db: Session = Depends(get_db)):
    """Returns info about the currently active model and contribution stats."""
    import json as _json
    class_names_path = os.path.join(BASE_DIR, "saved_models", "class_names.json")
    num_classes = 0
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            num_classes = len(_json.load(f))

    weights_path = os.path.join(BASE_DIR, "saved_models", "best_model.pth")
    weights_size_mb = round(os.path.getsize(weights_path) / 1e6, 1) if os.path.exists(weights_path) else 0

    active_version = db.query(ModelVersion).filter(ModelVersion.is_active == True).first()  # noqa: E712
    total_contributions  = db.query(Contribution).count()
    pending_verification = db.query(Contribution).filter(Contribution.verified == False).count()  # noqa: E712
    verified_unused      = db.query(Contribution).filter(
        Contribution.verified == True, Contribution.used_in_training == False
    ).count()

    return {
        "model": {
            "architecture":     "Vision Transformer (ViT-B/16)",
            "num_classes":      num_classes,
            "weights_size_mb":  weights_size_mb,
            "weights_loaded":   os.path.exists(weights_path),
            "version":          active_version.version if active_version else "v1.0-base",
            "accuracy":         active_version.accuracy if active_version else None,
            "hf_repo":          active_version.hf_repo if active_version else None,
        },
        "community": {
            "total_contributions":  total_contributions,
            "pending_verification": pending_verification,
            "verified_ready_to_train": verified_unused,
        },
    }


# ── Admin: verify a contribution ──────────────────────────────────────────────
@app.post("/api/admin/verify/{contribution_id}", summary="[Admin] Approve a contribution label")
async def verify_contribution(
    contribution_id: str,
    admin_secret: str = Form(...),
    db: Session = Depends(get_db),
):
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret.")
    record = db.query(Contribution).filter(Contribution.id == contribution_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Contribution not found.")
    record.verified = True
    db.commit()
    return {"status": "verified", "id": contribution_id}


# ── Admin: trigger retraining ─────────────────────────────────────────────────
_retrain_status: dict = {"running": False, "last_result": None}

def _run_retrain(db_records: list):
    """Background retraining task."""
    global _retrain_status
    _retrain_status["running"] = True
    try:
        from src.retrain import retrain_with_contributions
        result = retrain_with_contributions(db_records, epochs=3, batch_size=8)
        _retrain_status["last_result"] = result
        _retrain_status["finished_at"] = datetime.utcnow().isoformat()
    except Exception as e:
        _retrain_status["last_result"] = {"error": str(e)}
    finally:
        _retrain_status["running"] = False


@app.post("/api/admin/retrain", summary="[Admin] Fine-tune model on verified contributions")
async def admin_retrain(
    background_tasks: BackgroundTasks,
    admin_secret: str = Form(...),
    epochs: int = Form(3),
    db: Session = Depends(get_db),
):
    """
    Triggers fine-tuning on all verified, unused community contributions.
    Runs in the background — poll `/api/admin/retrain/status` to track progress.
    """
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret.")
    if _retrain_status["running"]:
        raise HTTPException(status_code=409, detail="Retraining is already in progress.")

    records = db.query(Contribution).filter(
        Contribution.verified == True,        # noqa: E712
        Contribution.used_in_training == False
    ).all()

    if not records:
        raise HTTPException(status_code=400, detail="No verified, unused contributions found.")

    db_records = [
        {"storage_path": r.storage_path, "image_url": r.image_url, "label": r.label}
        for r in records
    ]

    # Mark all as used so they aren't double-counted
    for r in records:
        r.used_in_training = True
    db.commit()

    background_tasks.add_task(_run_retrain, db_records)
    return {
        "status":  "started",
        "samples": len(db_records),
        "epochs":  epochs,
        "message": "Retraining started in background. Poll /api/admin/retrain/status for updates.",
    }


@app.get("/api/admin/retrain/status", summary="[Admin] Check retraining progress")
async def retrain_status(admin_secret: str):
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret.")
    return _retrain_status


# Mount contributions folder so images are publicly accessible
app.mount("/contributions", StaticFiles(directory=CONTRIB_DIR), name="contributions")

# Mount static files (uploads and other assets)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
