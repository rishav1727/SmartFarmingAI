import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import time

# Paths
MODEL_PATH = "saved_models/best_model.pt"
CLASS_NAMES_PATH = "saved_models/class_names.json"

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Load JIT traced model
print(f"Loading TorchScript (JIT) model from {MODEL_PATH}...")
start_load = time.time()
model = torch.jit.load(MODEL_PATH)
model.eval()
print(f"Model loaded in {time.time() - start_load:.4f} seconds!")

# Define standard ViT preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    print(f"\nEvaluating: {image_path}")
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension

    start_infer = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        logits = output[0]
        probabilities = torch.nn.functional.softmax(logits, dim=0)
        
        # Energy-based OOD Score (Lower energy = In distribution)
        temperature = 1.0
        energy_score = -temperature * torch.logsumexp(logits / temperature, dim=0)
        
    infer_time = time.time() - start_infer

    # Get Top-3
    top3_prob, top3_catid = torch.topk(probabilities, 3)
    
    print(f"Inference Time: {infer_time:.4f} seconds")
    print(f"Energy Score (Lower is better): {energy_score.item():.2f}")
    print("Top-3 Predictions:")
    for prob, cls_id in zip(top3_prob, top3_catid):
        print(f" - {class_names[cls_id]}: {prob.item() * 100:.2f}%")

if __name__ == "__main__":
    predict_image(r"C:\Users\RISHAV\Downloads\images (3).jpg")
    predict_image(r"C:\Users\RISHAV\Downloads\Powdery-Mildew-GettyImages-1090508010.jpg")
    predict_image(r"C:\Users\RISHAV\Downloads\images (4).jpg")
