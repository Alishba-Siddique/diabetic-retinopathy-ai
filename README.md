# Diabetic Retinopathy Detection API with Grad-CAM
This FastAPI service classifies retinal images and generates Grad-CAM heatmaps for explainability.

You can test: [https://huggingface.co/spaces/Alishba404/detection-of-diabetic-retinopathy](https://huggingface.co/spaces/Alishba404/detection-of-diabetic-retinopathy)

https://github.com/user-attachments/assets/28b58233-a759-42f7-aaa9-105ea4f7fc12

## 🔹 Features
✅ **Predicts Diabetic Retinopathy (DR) from retinal images**  
✅ **Uses Grad-CAM for explainability**  
✅ **Returns both prediction and heatmap visualization**  

## 🔹 Setup Instructions
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Start the FastAPI Server**
``` bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

```
### **3️⃣ Test the API**
Use cURL:
```
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```
Or Python requests:

```
import requests

url = "http://localhost:8000/predict"
files = {"file": open("retinal_image.jpg", "rb")}
response = requests.post(url, files=files)

print(response.json())

```
## 🔹 Deployment
To deploy this on Render, push the repo to GitHub and create a new Render Web Service.
