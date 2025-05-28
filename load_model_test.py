import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms

from predict import load_ensemble_model
from utils import get_device

def inspect_model():
    """Inspect the model structure and class mappings"""
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    MODEL_PATH = 'models/ensemble_model.pth'
    try:
        model, class_to_idx = load_ensemble_model(MODEL_PATH, device)
        print("Model loaded successfully")
        
        # Print class mapping
        print(f"Class to index mapping: {class_to_idx}")
        print(f"Index to class mapping: {dict((v, k) for k, v in class_to_idx.items())}")
        
        # Check model structure
        print("\nModel structure:")
        print(f"Type: {type(model)}")
        print(f"Number of sub-models: {len(model.models)}")
        
        for i, submodel in enumerate(model.models):
            print(f"\nSubmodel {i+1}:")
            print(f"Type: {type(submodel)}")
            
            # Get the last layer to check output dimensions
            if hasattr(submodel, 'classifier'):
                last_layer = submodel.classifier
                print(f"Last layer: {last_layer}")
                print(f"Output dimensions: {last_layer.out_features}")
            elif hasattr(submodel, 'fc'):
                last_layer = submodel.fc
                print(f"Last layer: {last_layer}")
                print(f"Output dimensions: {last_layer.out_features}")
            elif hasattr(submodel, 'head') and hasattr(submodel.head, 'fc'):
                last_layer = submodel.head.fc
                print(f"Last layer: {last_layer}")
                print(f"Output dimensions: {last_layer.out_features}")
        
        # Test with a sample image if available
        sample_images = ['uploads/vehicle', 'uploads/temp', 'balanced_dataset/front']
        found_image = None
        
        for path in sample_images:
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.jpg', '.jpeg', '.png')):
                            found_image = os.path.join(root, file)
                            break
                    if found_image:
                        break
            if found_image:
                break
        
        if found_image:
            print(f"\nTesting with sample image: {found_image}")
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(found_image).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                values, indices = torch.topk(probabilities, k=4)
                
                print("\nPredictions:")
                for i, (value, index) in enumerate(zip(values, indices)):
                    idx = index.item()
                    # Try to map index to class name
                    if idx in class_to_idx.values():
                        # Find key by value
                        class_name = [k for k, v in class_to_idx.items() if v == idx][0]
                    else:
                        class_name = f"Unknown (idx: {idx})"
                    
                    print(f"{i+1}. {class_name}: {value.item():.4f}")
        
        return True
    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    inspect_model() 