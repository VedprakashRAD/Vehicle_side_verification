import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms

from predict import load_ensemble_model, preprocess_image, predict
from utils import get_device

def test_front_image(image_path):
    """Test prediction on a front image and print detailed diagnostics"""
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    MODEL_PATH = 'models/ensemble_model.pth'
    try:
        print("Loading model...")
        model, class_to_idx = load_ensemble_model(MODEL_PATH, device)
        print("Model loaded successfully")
        
        # Print class mapping
        print(f"Class to index mapping: {class_to_idx}")
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        print(f"Index to class mapping: {idx_to_class}")
        
        # Preprocess image
        print(f"\nPreprocessing image: {image_path}")
        image_tensor, _ = preprocess_image(image_path, image_size=320)
        
        # Make prediction with raw model output
        print("\nMaking prediction with model...")
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Print raw outputs and probabilities
            print(f"Raw model output: {outputs}")
            print(f"Probabilities: {probabilities}")
            
            # Get top predictions
            values, indices = torch.topk(probabilities, k=4)
            print("\nTop predictions (raw):")
            for i, (value, index) in enumerate(zip(values, indices)):
                print(f"{i+1}. Index {index.item()}: {value.item():.4f}")
            
            # Get top class predictions
            print("\nTop predictions (with class names):")
            for i, (value, index) in enumerate(zip(values, indices)):
                idx = index.item()
                class_name = idx_to_class.get(idx, f"Unknown (idx: {idx})")
                print(f"{i+1}. {class_name}: {value.item():.4f}")
        
        # Use the predict function
        print("\nUsing predict function:")
        pred_class, confidence = predict(model, image_tensor, idx_to_class, device)
        print(f"Predicted class: {pred_class}, Confidence: {confidence}")
        
        # Check if prediction matches expected 'front'
        print(f"\nDoes prediction match 'front'? {pred_class == 'front'}")
        print(f"Case insensitive match? {pred_class.lower() == 'front'.lower()}")
        
        return True
    except Exception as e:
        print(f"Error testing image: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Find a front image to test
    test_image = None
    
    # Try to find a front image in uploads
    search_paths = ['uploads/vehicle', 'uploads/temp']
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.startswith('front_') or '1.png' in file:
                    test_image = os.path.join(path, file)
                    break
    
    # If no image found, use a default
    if not test_image:
        test_image = input("Enter path to a front image: ")
    
    print(f"Testing with image: {test_image}")
    test_front_image(test_image) 