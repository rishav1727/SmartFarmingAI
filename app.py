import os
import shutil
import urllib.request
import gradio as gr

# Ensure runtime directories exist
for folder in ["public/uploads", "public/contributions", "public/samples", "saved_models", "data/documents"]:
    os.makedirs(folder, exist_ok=True)

# Ensure weights exist (download if not present)
weights_path = os.path.join("saved_models", "best_model.pth")
if not os.path.exists(weights_path):
    print("[startup] Downloading ViT-B/16 checkpoint from GitHub Releases...")
    url = os.environ.get("WEIGHTS_URL", "https://github.com/rishav1727/SmartFarmingAI/releases/download/v1.0-vit/best_model.pth")
    try:
        urllib.request.urlretrieve(url, weights_path)
        print(f"[startup] Weights saved ({round(os.path.getsize(weights_path)/1e6, 1)} MB).")
    except Exception as e:
        print(f"[startup] Weights download notice: {e}")

# Import our complete FastAPI application
from src.app import app as fastapi_app

# Define a clean Gradio dashboard interface
with gr.Blocks(title="SmartFarming AI Backend") as demo:
    gr.Markdown("# 🌱 SmartFarming AI — Cloud Backend & Intelligence Engine")
    gr.Markdown(
        "Serving **Vision Transformer (ViT-B/16)** crop disease diagnostics, **Explainable Grad-CAM**, "
        "and **Agro-RAG Multi-Modal Treatment Advice**."
    )
    with gr.Row():
        gr.Markdown("🌐 **Live Web Dashboard:** [https://smart-farming-ai-2rht.vercel.app](https://smart-farming-ai-2rht.vercel.app)")
        gr.Markdown("📋 **Interactive API Docs:** [/docs](/docs)")
        gr.Markdown("❤️ **Health Status:** [/health](/health)")

# Mount Gradio onto the existing FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
