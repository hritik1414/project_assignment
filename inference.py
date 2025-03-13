import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Configuration
img_size = (224, 224)
num_classes = 10
cifar10_classes = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Use CUDA if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path: str):
    """
    Load the VGG16 model with custom classifier weights.
    """
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

# Load the model once and reuse it
model_path = os.path.join(os.path.dirname(__file__), "myModel.pth")
model = load_model(model_path)

def inference(image_path: str) -> str:
    """
    Loads an image, processes it, and returns the predicted class.
    """
    if model is None:
        raise RuntimeError("Model is not loaded. Please check your model file.")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Open and process image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)

    return cifar10_classes[predicted.item()]
