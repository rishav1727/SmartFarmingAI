import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.consts import HF_DATASET_NAME, IMAGE_SIZE, BATCH_SIZE

class PlantDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["labels"] if "labels" in item else item.get("label", 0)

        # Ensure image is in RGB mode (some images might be grayscale or RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders():
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),      # Plants can be photographed from any angle
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading dataset: {HF_DATASET_NAME}")
    try:
        dataset = load_dataset(HF_DATASET_NAME, "default")
        # Attempt to find the split names
        train_split = "train" if "train" in dataset else list(dataset.keys())[0]
        val_split = "validation" if "validation" in dataset else "test"
        
        if val_split not in dataset:
            # If no validation set natively exists, split the train set
            print("No natural validation split found. Splitting training data 80/20...")
            split = dataset[train_split].train_test_split(test_size=0.2)
            train_data = split["train"]
            val_data = split["test"]
        else:
            train_data = dataset[train_split]
            val_data = dataset[val_split]

        # Extract class names to map labels to actual string predictions
        features = train_data.features
        label_col = "labels" if "labels" in features else "label"
        class_names = features[label_col].names if hasattr(features[label_col], "names") else []

        train_dataset = PlantDataset(train_data, transform=transform)
        val_dataset = PlantDataset(val_data, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        return train_loader, val_loader, class_names

    except Exception as e:
        print(f"Error loading dataset. Make sure you are connected to the internet and the Hugging Face dataset name is correct. Error: {e}")
        return None, None, []
