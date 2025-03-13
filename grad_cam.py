import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms, datasets
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image


img_size = (224, 224)
num_classes = 10
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_path", type=str, required=True, help="model path")
parser.add_argument("--image_path", type=str, required=True, help="image path")

args = parser.parse_args()
model_path = args.model_path
image_path = args.image_path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))


def inference(imagePath):
    image = Image.open(imagePath).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    model.eval()
    output = model(tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

image = Image.open(image_path).convert("RGB")
tensor = transform(image)
target_layer = model.features[-3]
cam = GradCAM(model=model, target_layers=[target_layer])


pred_class = inference(image_path)
targets = [ClassifierOutputTarget(pred_class)]
grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=targets)
grayscale_cam = grayscale_cam[0, :]
tensor_np = tensor.permute(1, 2, 0).cpu().numpy()
tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min()) 
visualization = show_cam_on_image(tensor_np, grayscale_cam, use_rgb=True)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(tensor_np)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.axis('off')
plt.show()
