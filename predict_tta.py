import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import get_device
from models import get_model, EnsembleModel

def load_model(model_path, device):
    """
    Load a model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        model: Loaded model
        class_to_idx: Class to index mapping
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it's an ensemble model
    if 'model_names' in checkpoint:
        # Load ensemble model
        model_names = checkpoint['model_names']
        model_states = checkpoint['model_states']
        weights = checkpoint['weights']
        class_to_idx = checkpoint['class_to_idx']
        
        # Load individual models
        models = []
        for model_name in model_names:
            model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
            model.load_state_dict(model_states[model_name])
            model = model.to(device)
            model.eval()
            models.append(model)
        
        # Create ensemble
        ensemble = EnsembleModel(models, weights=weights)
        ensemble.eval()
        
        return ensemble, class_to_idx
    else:
        # Load single model
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
        else:
            class_to_idx = {
                'front': 0,
                'rear': 1,
                'left': 2,
                'right': 3
            }
            
        if 'args' in checkpoint and 'model' in checkpoint['args']:
            model_name = checkpoint['args']['model']
        else:
            model_name = 'efficientnet_b7'
            
        model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model, class_to_idx

def create_tta_transforms(img_size=320):
    """
    Create a list of test-time augmentation transforms.
    
    Args:
        img_size: Image size
    
    Returns:
        transforms_list: List of transforms
    """
    # Define normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create a list of transforms
    transforms_list = [
        # Original image
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        
        # Center crop
        transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        
        # Color jitter
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        
        # Rotation 10 degrees
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    ]
    
    return transforms_list

def predict_with_tta(model, image_path, class_to_idx, device, img_size=320):
    """
    Predict with test-time augmentation.
    
    Args:
        model: Trained model
        image_path: Path to the image
        class_to_idx: Class to index mapping
        device: Device to run inference on
        img_size: Image size
    
    Returns:
        predicted_class: Predicted class name
        confidence: Confidence score
        all_probs: All class probabilities
    """
    # Create TTA transforms
    tta_transforms = create_tta_transforms(img_size)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms and make predictions
    all_outputs = []
    
    for transform in tta_transforms:
        # Apply transform
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            all_outputs.append(outputs)
    
    # Average predictions
    avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    probabilities = torch.nn.functional.softmax(avg_outputs, dim=1)
    confidence, prediction = torch.max(probabilities, 1)
    
    # Convert index to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[prediction.item()]
    
    # Get all probabilities
    all_probs = {idx_to_class[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    
    return predicted_class, confidence.item(), all_probs

def visualize_prediction(image_path, predicted_class, confidence, all_probs):
    """
    Visualize the prediction result with all class probabilities.
    
    Args:
        image_path: Path to the image
        predicted_class: Predicted class name
        confidence: Confidence score
        all_probs: All class probabilities
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(12, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
    plt.axis('off')
    
    # Display probabilities
    plt.subplot(1, 2, 2)
    classes = list(all_probs.keys())
    probs = list(all_probs.values())
    y_pos = np.arange(len(classes))
    
    bars = plt.barh(y_pos, probs, align='center')
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    # Color the predicted class differently
    for i, cls in enumerate(classes):
        if cls == predicted_class:
            bars[i].set_color('green')
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"pred_tta_{base_name}")
    plt.savefig(output_path)
    
    print(f"Visualization saved to {output_path}")
    
    # Show plot if requested
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Predict vehicle orientation with test-time augmentation')
    parser.add_argument('--model', type=str, default='models/ensemble_model.pth', help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    parser.add_argument('--visualize', action='store_true', help='Visualize prediction')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, class_to_idx = load_model(args.model, device)
    
    # Predict with TTA
    predicted_class, confidence, all_probs = predict_with_tta(model, args.image, class_to_idx, device, args.img_size)
    
    # Print results
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Print all probabilities
    print("Class probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.4f}")
    
    # Visualize prediction
    if args.visualize:
        visualize_prediction(args.image, predicted_class, confidence, all_probs)

if __name__ == "__main__":
    main() 