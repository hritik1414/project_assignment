import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gradio as gr
import os
import numpy as np
from PIL import Image
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Model setup
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
model_path = os.path.join(os.path.dirname(__file__), "myModel.pth")
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict_and_visualize(image):
    image = Image.fromarray(image).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    output = model(tensor)
    _, predicted = torch.max(output, 1)
    pred_class = predicted.item()
    target_layer = model.features[-3]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    tensor_np = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    tensor_np = (tensor_np - tensor_np.min()) / (tensor_np.max() - tensor_np.min())
    visualization = show_cam_on_image(tensor_np, grayscale_cam, use_rgb=True)

    return cifar10_classes[pred_class], visualization

demo = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Label(label = 'Predicted Class'), gr.Image(type="numpy", label = 'Grad-CAM Visualisation')],
    title="Image Visualisation with Grad-CAM",
    description="Drag and drop an image to get the predicted class and Grad-CAM visualization."
)

demo.launch(share = True)
