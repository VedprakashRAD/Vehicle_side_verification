import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from utils import get_device
from models import get_model, EnsembleModel
from predict_tta import predict_with_tta, load_model, visualize_prediction

def compare_models(image_path, base_model_path, improved_model_path=None, ensemble_model_path=None, img_size=320):
    """
    Compare different models on a single image.
    
    Args:
        image_path: Path to the image
        base_model_path: Path to the base model
        improved_model_path: Path to the improved model (continued training)
        ensemble_model_path: Path to the ensemble model
        img_size: Image size
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # 1. Base model prediction
    print("\n1. Base Model Prediction:")
    base_model, class_to_idx = load_model(base_model_path, device)
    
    with torch.no_grad():
        outputs = base_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    base_pred = idx_to_class[pred_idx.item()]
    base_conf = confidence.item()
    
    print(f"Prediction: {base_pred}")
    print(f"Confidence: {base_conf:.4f}")
    
    # Plot base model prediction
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f"Base Model: {base_pred} ({base_conf:.2f})")
    plt.axis('off')
    
    # 2. Base model with TTA
    print("\n2. Base Model with Test-Time Augmentation:")
    base_tta_pred, base_tta_conf, base_tta_probs = predict_with_tta(base_model, image_path, class_to_idx, device, img_size)
    
    print(f"Prediction: {base_tta_pred}")
    print(f"Confidence: {base_tta_conf:.4f}")
    
    # Plot base model with TTA
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    plt.title(f"Base Model + TTA: {base_tta_pred} ({base_tta_conf:.2f})")
    plt.axis('off')
    
    # 3. Improved model (if provided)
    if improved_model_path:
        print("\n3. Improved Model (Continued Training):")
        improved_model, _ = load_model(improved_model_path, device)
        
        with torch.no_grad():
            outputs = improved_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)
        
        improved_pred = idx_to_class[pred_idx.item()]
        improved_conf = confidence.item()
        
        print(f"Prediction: {improved_pred}")
        print(f"Confidence: {improved_conf:.4f}")
        
        # Plot improved model prediction
        plt.subplot(2, 2, 3)
        plt.imshow(image)
        plt.title(f"Improved Model: {improved_pred} ({improved_conf:.2f})")
        plt.axis('off')
    
    # 4. Ensemble model (if provided)
    if ensemble_model_path:
        print("\n4. Ensemble Model with Test-Time Augmentation:")
        ensemble_model, _ = load_model(ensemble_model_path, device)
        ensemble_pred, ensemble_conf, ensemble_probs = predict_with_tta(ensemble_model, image_path, class_to_idx, device, img_size)
        
        print(f"Prediction: {ensemble_pred}")
        print(f"Confidence: {ensemble_conf:.4f}")
        
        # Plot ensemble model prediction
        plt.subplot(2, 2, 4)
        plt.imshow(image)
        plt.title(f"Ensemble + TTA: {ensemble_pred} ({ensemble_conf:.2f})")
        plt.axis('off')
    
    # Save and show plot
    plt.tight_layout()
    
    # Create output directory
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"comparison_{base_name}")
    plt.savefig(output_path)
    print(f"\nComparison visualization saved to {output_path}")
    
    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare different model improvements')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--base_model', type=str, default='best_model.pth', help='Path to base model')
    parser.add_argument('--improved_model', type=str, default=None, help='Path to improved model')
    parser.add_argument('--ensemble_model', type=str, default=None, help='Path to ensemble model')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    
    args = parser.parse_args()
    
    compare_models(args.image, args.base_model, args.improved_model, args.ensemble_model, args.img_size)

if __name__ == "__main__":
    main() 