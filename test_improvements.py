import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import get_device
from models import get_model
from predict_tta import predict_with_tta, load_model

def test_tta_improvement(image_path, model_path, img_size=320):
    """
    Test the improvement from test-time augmentation.
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
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
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_to_idx = load_model(model_path, device)
    
    # Standard prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    std_pred = idx_to_class[pred_idx.item()]
    std_conf = confidence.item()
    
    print("\nStandard Prediction:")
    print(f"Prediction: {std_pred}")
    print(f"Confidence: {std_conf:.4f}")
    
    # TTA prediction
    print("\nTest-Time Augmentation Prediction:")
    tta_pred, tta_conf, tta_probs = predict_with_tta(model, image_path, class_to_idx, device, img_size)
    
    print(f"Prediction: {tta_pred}")
    print(f"Confidence: {tta_conf:.4f}")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot standard prediction
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Standard: {std_pred} ({std_conf:.2f})")
    plt.axis('off')
    
    # Plot TTA prediction
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title(f"With TTA: {tta_pred} ({tta_conf:.2f})")
    plt.axis('off')
    
    # Save and show plot
    plt.tight_layout()
    
    # Create output directory
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"tta_comparison_{base_name}")
    plt.savefig(output_path)
    print(f"\nComparison visualization saved to {output_path}")
    
    # Show plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test TTA improvement')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    
    args = parser.parse_args()
    
    test_tta_improvement(args.image, args.model, args.img_size)

if __name__ == "__main__":
    main() 