import os
import argparse
import torch
import h5py

from utils import get_device
from models import get_model

def convert_to_h5(input_path, output_path, model_name='efficientnet_b7'):
    """
    Convert a PyTorch model to H5 format.
    
    Args:
        input_path: Path to the PyTorch model weights
        output_path: Path to save the H5 model
        model_name: Name of the model architecture
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Define class mapping
    class_to_idx = {
        'front': 0,
        'rear': 1,
        'left': 2,
        'right': 3
    }
    
    # Create model
    model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(input_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {input_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PyTorch format first
    torch_path = output_path.replace('.h5', '.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'args': {'model': model_name}
    }, torch_path)
    
    # Convert to H5 format
    with h5py.File(output_path, 'w') as f:
        # Store model architecture as attributes
        f.attrs['model_type'] = 'single'
        
        # Store class mapping
        class_group = f.create_group('class_mapping')
        for class_name, idx in class_to_idx.items():
            class_group.attrs[class_name] = idx
        
        # Store model weights
        weights_group = f.create_group('weights')
        for name, param in model.state_dict().items():
            weights_group.create_dataset(name, data=param.cpu().numpy())
    
    print(f"Model saved in H5 format to {output_path}")
    print(f"Model saved in PyTorch format to {torch_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to H5 format')
    parser.add_argument('--input', type=str, default='best_model.pth', help='Path to input PyTorch model')
    parser.add_argument('--output', type=str, default='models/vehicle_orientation_model.h5', help='Path to output H5 model')
    parser.add_argument('--model', type=str, default='efficientnet_b7', help='Model architecture')
    
    args = parser.parse_args()
    
    convert_to_h5(args.input, args.output, args.model)

if __name__ == "__main__":
    main() 