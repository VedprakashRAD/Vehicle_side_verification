import os
import argparse
import h5py
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from utils import get_device
from models import get_model

def load_h5_model(model_path, model_name='efficientnet_b7'):
    """
    Load a model from H5 format.
    
    Args:
        model_path: Path to the H5 model
        model_name: Name of the model architecture
    
    Returns:
        model: Loaded PyTorch model
        idx_to_class: Class mapping
    """
    # Get device
    device = get_device()
    
    # Load class mapping from H5 file
    with h5py.File(model_path, 'r') as f:
        class_group = f['class_mapping']
        class_to_idx = {name: int(idx) for name, idx in class_group.attrs.items()}
        idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create model
    model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    
    # Load weights from H5 file
    with h5py.File(model_path, 'r') as f:
        weights_group = f['weights']
        state_dict = {}
        for name in weights_group.keys():
            state_dict[name] = torch.tensor(weights_group[name][()])
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, idx_to_class

def preprocess_image(image_path, image_size=320):
    """
    Preprocess an image for inference.
    
    Args:
        image_path: Path to the image
        image_size: Size to resize the image to
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    
    return tensor, image

def predict(model, image_tensor, idx_to_class, device=None):
    """
    Make a prediction on an image.
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        idx_to_class: Class mapping
        device: Device to use for inference
    
    Returns:
        pred_class: Predicted class
        confidence: Confidence score
    """
    if device is None:
        device = get_device()
    
    # Move tensor to device
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
    
    # Get class name
    pred_class = idx_to_class[pred_idx.item()]
    
    return pred_class, confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Predict vehicle orientation using H5 model')
    parser.add_argument('--model', type=str, default='models/vehicle_orientation_model.h5', help='Path to H5 model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--size', type=int, default=320, help='Image size')
    parser.add_argument('--architecture', type=str, default='efficientnet_b7', help='Model architecture')
    parser.add_argument('--show', action='store_true', help='Show image with prediction')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, idx_to_class = load_h5_model(args.model, args.architecture)
    print(f"Model loaded from {args.model}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(args.image, args.size)
    
    # Make prediction
    pred_class, confidence = predict(model, image_tensor, idx_to_class, device)
    
    # Print results
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show image with prediction
    if args.show:
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main() 