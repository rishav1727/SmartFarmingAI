import os
import torch
import json
from src.model import get_model
from src.consts import MODEL_DIR, DEVICE

def export_to_onnx():
    print("Starting ONNX export process...")
    
    # Check if best model exists
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    # Load class names to get number of classes
    class_names_path = os.path.join(MODEL_DIR, "class_names.json")
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # Initialize PyTorch model on CPU for clean ONNX tracing!
    print(f"Loading PyTorch Model (Classes: {num_classes}) on CPU for tracing...")
    model = get_model(num_classes, 'cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Define dummy input tensor (Batch Size=1, Channels=3, H=224, W=224)
    dummy_input = torch.randn(1, 3, 224, 224, device='cpu')

    # Export Path
    jit_path = os.path.join(MODEL_DIR, "best_model.pt")

    print("Exporting model to TorchScript (JIT) format...")
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False, check_trace=False)
        traced_model.save(jit_path)
        print(f"\n[SUCCESS]: Model successfully exported to: {jit_path}")
        print("TorchScript (JIT) gives roughly ~5x speedup and avoids ONNX SDPA limitations!")
    except Exception as e:
        import traceback
        with open("error.txt", "w") as f:
            f.write(traceback.format_exc())
        print("\n\n=== JIT EXPORT FAILED ===")
        print("See error.txt for details")

if __name__ == "__main__":
    export_to_onnx()
