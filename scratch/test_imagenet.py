import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import json
import ast

# 1. Load Pretrained MobileNetV3
print("Loading MobileNetV3 for General Object Detection...")
mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
mobilenet.eval()

# 2. Get ImageNet Labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

# 3. Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def identify_object(image_path):
    print(f"\nIdentifying: {image_path}")
    img = Image.open(image_path).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)

    with torch.no_grad():
        out = mobilenet(batch_t)
    
    prob = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(prob, 1)
    
    print(f"I see a: {labels[top_catid[0]]} (Confidence: {top_prob[0].item()*100:.2f}%)")

identify_object(r"C:\Users\RISHAV\Downloads\images (4).jpg")
