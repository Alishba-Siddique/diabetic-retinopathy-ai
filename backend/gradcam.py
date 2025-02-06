import torch
import numpy as np
import cv2

class GradCAM:
    """Grad-CAM implementation for visualizing CNN decisions."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """Hooks the gradients and activations of the target layer."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, image_tensor):
        """Generates a Grad-CAM heatmap for an input image."""
        image_tensor = image_tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        output = self.model(image_tensor)
        class_idx = output.argmax().item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx
