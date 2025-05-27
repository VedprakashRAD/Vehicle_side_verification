import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import warnings
import time
import cv2

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import get_device
from predict_tta import predict_with_tta, load_model

def predict_image(model, class_to_idx, image_path, device, img_size=320, use_tta=True):
    """
    Predict the class of an image.
    
    Args:
        model: PyTorch model
        class_to_idx: Class to index mapping
        image_path: Path to the image
        device: Device to run inference on
        img_size: Image size
        use_tta: Whether to use test-time augmentation
    
    Returns:
        pred_class: Predicted class name
        confidence: Confidence score
    """
    if use_tta:
        # Use test-time augmentation
        start_time = time.time()
        pred_class, confidence, _ = predict_with_tta(model, image_path, class_to_idx, device, img_size)
        inference_time = time.time() - start_time
    else:
        # Standard prediction
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)
        inference_time = time.time() - start_time
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        pred_class = idx_to_class[pred_idx.item()]
        confidence = confidence.item()
    
    return pred_class, confidence, inference_time

def visualize_result(image_path, pred_class, confidence, output_path=None):
    """
    Visualize the prediction result.
    
    Args:
        image_path: Path to the image
        pred_class: Predicted class name
        confidence: Confidence score
        output_path: Path to save the visualization
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    # Add prediction text
    plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.2f})", fontsize=16)
    
    # Add colored border based on class
    if pred_class == 'front':
        border_color = 'green'
    elif pred_class == 'rear':
        border_color = 'red'
    elif pred_class == 'left':
        border_color = 'blue'
    elif pred_class == 'right':
        border_color = 'orange'
    else:
        border_color = 'gray'
    
    # Add border
    plt.gca().spines['top'].set_color(border_color)
    plt.gca().spines['bottom'].set_color(border_color)
    plt.gca().spines['left'].set_color(border_color)
    plt.gca().spines['right'].set_color(border_color)
    plt.gca().spines['top'].set_linewidth(5)
    plt.gca().spines['bottom'].set_linewidth(5)
    plt.gca().spines['left'].set_linewidth(5)
    plt.gca().spines['right'].set_linewidth(5)
    
    plt.axis('off')
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()

def process_webcam(model, class_to_idx, device, img_size=320):
    """
    Process webcam feed for real-time inference.
    
    Args:
        model: PyTorch model
        class_to_idx: Class to index mapping
        device: Device to run inference on
        img_size: Image size
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Transform image
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        pred_class = idx_to_class[pred_idx.item()]
        confidence = confidence.item()
        
        # Add prediction text
        text = f"{pred_class}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Vehicle Orientation Detection", frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Vehicle Orientation Detection')
    parser.add_argument('--model', type=str, default='models/ensemble_model.pth', help='Path to model')
    parser.add_argument('--image', type=str, help='Path to image (optional)')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time inference')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    parser.add_argument('--output', type=str, help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    if not args.image and not args.webcam:
        parser.error("Either --image or --webcam must be specified")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, class_to_idx = load_model(args.model, device)
    
    if args.webcam:
        # Process webcam feed
        process_webcam(model, class_to_idx, device, args.img_size)
    else:
        # Process single image
        pred_class, confidence, inference_time = predict_image(
            model, class_to_idx, args.image, device, args.img_size, args.tta
        )
        
        print(f"Prediction: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference time: {inference_time:.4f} seconds")
        
        # Visualize result
        visualize_result(args.image, pred_class, confidence, args.output)

if __name__ == "__main__":
    main() 