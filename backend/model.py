import torch
import torch.nn as nn
import torchvision.models as models

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Loads the trained ResNet50 model securely."""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes (No DR, DR)

    # Securely load weights
    model.load_state_dict(torch.load("./models/best_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model

# Load model globally
model = load_model()
