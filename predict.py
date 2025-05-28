import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from glob import glob

from utils import get_device
from models import get_model, EnsembleModel

def load_single_model(model_path, device):
    """
    Load a single trained model.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded model
        class_to_idx: Class to index mapping
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    class_to_idx = checkpoint['class_to_idx']
    args = checkpoint['args']
    
    model = get_model(args['model'], num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_to_idx

def load_ensemble_model(model_path, device):
    """
    Load an ensemble model.
    
    Args:
        model_path: Path to the ensemble model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded ensemble model
        class_to_idx: Class to index mapping
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    class_to_idx = checkpoint['class_to_idx']
    model_names = checkpoint['model_names']
    weights = checkpoint['weights']
    
    # Load individual models
    models = []
    for model_name in model_names:
        model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
        model.load_state_dict(checkpoint['model_states'][model_name])
        model = model.to(device)
        model.eval()
        models.append(model)
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights=weights)
    ensemble.eval()
    
    return ensemble, class_to_idx

def predict_image(model, image_path, class_to_idx, device, img_size=224):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image
        class_to_idx: Class to index mapping
        device: Device to run inference on
        img_size: Image size for preprocessing
        
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    
    # Convert index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[prediction.item()]
    
    return predicted_class, confidence.item()

def visualize_prediction(image_path, predicted_class, confidence):
    """
    Visualize the prediction result.
    
    Args:
        image_path: Path to the image
        predicted_class: Predicted class name
        confidence: Confidence score
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.tight_layout()
    
    # Save visualization
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"pred_{base_name}")
    plt.savefig(output_path)
    
    print(f"Visualization saved to {output_path}")

def load_model(model_path, model_name='efficientnet_b7'):
    """
    Load a trained PyTorch model.
    
    Args:
        model_path: Path to the model weights
        model_name: Name of the model architecture
    
    Returns:
        model: Loaded PyTorch model
        class_to_idx: Class mapping
    """
    # Get device
    device = get_device()
    
    # Define class mapping
    class_to_idx = {
        'front': 0,
        'rear': 1,
        'left': 2,
        'right': 3
    }
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Create model
    model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    pred_idx_value = pred_idx.item()
    
    # Debug output
    print(f"Prediction index: {pred_idx_value}")
    print(f"Available class indices: {list(idx_to_class.keys())}")
    
    # Convert index to class name
    if isinstance(idx_to_class, dict):
        # If idx_to_class is a dictionary mapping indices to class names
        if pred_idx_value in idx_to_class:
            pred_class = idx_to_class[pred_idx_value]
        else:
            # Handle case where idx_to_class might be in reverse format (class name -> index)
            # This is for backward compatibility
            orientation_map = {0: 'front', 1: 'left', 2: 'rear', 3: 'right'}
            pred_class = orientation_map.get(pred_idx_value, 'unknown')
    else:
        # If idx_to_class is a list
        if 0 <= pred_idx_value < len(idx_to_class):
            pred_class = idx_to_class[pred_idx_value]
        else:
            orientation_map = {0: 'front', 1: 'left', 2: 'rear', 3: 'right'}
            pred_class = orientation_map.get(pred_idx_value, 'unknown')
    
    print(f"Predicted class: {pred_class}, Confidence: {confidence.item()}")
    
    return pred_class, confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Predict vehicle orientation')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--size', type=int, default=320, help='Image size')
    parser.add_argument('--architecture', type=str, default='efficientnet_b7', help='Model architecture')
    parser.add_argument('--show', action='store_true', help='Show image with prediction')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model, idx_to_class = load_model(args.model, args.architecture)
    
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