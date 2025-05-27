import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
import glob

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import get_device
from models import get_model
from predict_tta import predict_with_tta, load_model

def evaluate_on_folder(model_path, data_dir, use_tta=True, img_size=320, max_images_per_class=50):
    """
    Evaluate model on a folder of images organized by class.
    
    Args:
        model_path: Path to the model
        data_dir: Path to the data directory with class subdirectories
        use_tta: Whether to use test-time augmentation
        img_size: Image size
        max_images_per_class: Maximum number of images to evaluate per class
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_to_idx = load_model(model_path, device)
    
    # Get class names
    class_names = list(class_to_idx.keys())
    print(f"Classes: {class_names}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare for evaluation
    all_preds = []
    all_labels = []
    all_confs = []
    
    # Process each class
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory {class_dir} not found")
            continue
        
        # Get image paths
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg")) + \
                     glob.glob(os.path.join(class_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(class_dir, "*.png"))
        
        # Limit number of images
        if max_images_per_class > 0:
            image_paths = image_paths[:max_images_per_class]
        
        print(f"Evaluating {len(image_paths)} images for class '{class_name}'")
        
        # Process each image
        for image_path in image_paths:
            # Get true label
            true_label = class_to_idx[class_name]
            
            # Make prediction
            if use_tta:
                pred_class, confidence, _ = predict_with_tta(model, image_path, class_to_idx, device, img_size)
                pred_label = class_to_idx[pred_class]
            else:
                # Standard prediction
                image = Image.open(image_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    confidence, pred_idx = torch.max(probabilities, dim=0)
                
                pred_label = pred_idx.item()
                confidence = confidence.item()
            
            # Store results
            all_preds.append(pred_label)
            all_labels.append(true_label)
            all_confs.append(confidence)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)
    
    # Calculate accuracy
    accuracy = np.mean(all_preds == all_labels)
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Calculate average confidence
    avg_conf = np.mean(all_confs)
    print(f"Average confidence: {avg_conf:.4f}")
    
    # Generate classification report
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy:.4f})")
    
    # Save confusion matrix
    output_dir = "evaluations"
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = os.path.basename(model_path).replace(".pth", "").replace(".h5", "")
    output_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_{'tta' if use_tta else 'std'}.png")
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    
    # Return results
    return {
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "classification_report": classification_report(all_labels, all_preds, target_names=target_names, output_dict=True),
        "confusion_matrix": cm
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', type=str, default='models/ensemble_model.pth', help='Path to model')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to data directory')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    parser.add_argument('--max_images', type=int, default=50, help='Maximum images per class (0 for all)')
    
    args = parser.parse_args()
    
    evaluate_on_folder(args.model, args.data_dir, args.tta, args.img_size, args.max_images)

if __name__ == "__main__":
    main() 