import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

from utils import get_device
from models import get_model, EnsembleModel
from predict import load_single_model, load_ensemble_model, predict_image

def display_sample_predictions(model, class_to_idx, device, data_dir, num_samples=3):
    """
    Display sample predictions from each class.
    
    Args:
        model: Trained model
        class_to_idx: Class to index mapping
        device: Device to run inference on
        data_dir: Path to the dataset directory
        num_samples: Number of samples to display per class
    """
    class_dirs = list(class_to_idx.keys())
    
    plt.figure(figsize=(15, 12))
    
    for i, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
            
        # Get image files
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Select random samples
        if len(image_files) > num_samples:
            samples = np.random.choice(image_files, num_samples, replace=False)
        else:
            samples = image_files
        
        for j, img_file in enumerate(samples):
            img_path = os.path.join(class_dir, img_file)
            
            # Make prediction
            predicted_class, confidence = predict_image(model, img_path, class_to_idx, device)
            
            # Display image with prediction
            plt.subplot(len(class_dirs), num_samples, i * num_samples + j + 1)
            img = Image.open(img_path).convert('RGB')
            plt.imshow(img)
            
            # Set title color based on prediction correctness
            title_color = 'green' if predicted_class == class_name else 'red'
            plt.title(f"True: {class_name}\nPred: {predicted_class}\nConf: {confidence:.2f}", 
                     color=title_color, fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.close()
    print("Sample predictions saved as 'sample_predictions.png'")

def main():
    parser = argparse.ArgumentParser(description='Demo for vehicle orientation detection')
    parser.add_argument('--model', type=str, default='ensemble_model.pth', 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', 
                       help='Path to the dataset directory')
    parser.add_argument('--samples', type=int, default=3, 
                       help='Number of samples to display per class')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check if model exists, if not, notify user
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
        print("You need to train a model first using train.py or train_ensemble.py.")
        return
    
    # Load model
    print(f"Loading model from {args.model}...")
    if 'ensemble' in args.model:
        model, class_to_idx = load_ensemble_model(args.model, device)
        print("Loaded ensemble model")
    else:
        model, class_to_idx = load_single_model(args.model, device)
        print(f"Loaded single model")
    
    # Display sample predictions
    display_sample_predictions(model, class_to_idx, device, args.data_dir, args.samples)
    
    print("\nTo use this model for predictions on new images, run:")
    print(f"python predict.py --model {args.model} --image path/to/your/image.jpg --visualize")
    print("or")
    print(f"python predict.py --model {args.model} --dir path/to/your/images/ --visualize")

if __name__ == "__main__":
    main() 