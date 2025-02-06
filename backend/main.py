from fastapi import FastAPI, UploadFile
import io
import base64
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Import model & GradCAM
from backend.model import model  # ✅ Explicitly mention 'backend'
from backend.gradcam import GradCAM

app = FastAPI()

# Define Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Grad-CAM with the last convolutional layer
grad_cam = GradCAM(model, model.layer4[2].conv3)

@app.post("/predict")
async def predict(file: UploadFile):
    """Predicts Diabetic Retinopathy and generates a Grad-CAM heatmap."""
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    transformed_image = transform(image)

    cam, prediction = grad_cam.generate_cam(transformed_image)

    # Convert Grad-CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(image.resize((224, 224))), 0.5, heatmap, 0.5, 0)

    # Convert to Base64 for API response
    _, buffer = cv2.imencode(".png", overlay)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": "Diabetic Retinopathy" if prediction == 1 else "No DR",
        "gradcam": heatmap_base64
    }

# Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
