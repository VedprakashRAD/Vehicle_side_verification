import os
import argparse
import torch
import torch.nn as nn
import h5py
import json
import onnx
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

from utils import get_device
from predict_tta import load_model

def export_to_onnx(model_path, output_path, img_size=320):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the ONNX model
        img_size: Image size for the model input
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_to_idx = load_model(model_path, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model exported to ONNX format: {output_path}")
    
    # Verify the ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")
    
    # Save class mapping
    class_mapping_path = output_path.replace('.onnx', '_classes.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    
    print(f"Class mapping saved to: {class_mapping_path}")
    
    return output_path

def export_to_torchscript(model_path, output_path, img_size=320):
    """
    Export PyTorch model to TorchScript format.
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the TorchScript model
        img_size: Image size for the model input
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_to_idx = load_model(model_path, device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    print(f"Model exported to TorchScript format: {output_path}")
    
    # Save class mapping
    class_mapping_path = output_path.replace('.pt', '_classes.json')
    with open(class_mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    
    print(f"Class mapping saved to: {class_mapping_path}")
    
    return output_path

def export_to_h5(model_path, output_path):
    """
    Export PyTorch model to H5 format.
    
    Args:
        model_path: Path to the PyTorch model
        output_path: Path to save the H5 model
    """
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, class_to_idx = load_model(model_path, device)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as H5 format
    with h5py.File(output_path, 'w') as f:
        # Store model type as attribute
        if hasattr(model, 'models'):
            f.attrs['model_type'] = 'ensemble'
        else:
            f.attrs['model_type'] = 'single'
        
        # Store class mapping
        class_group = f.create_group('class_mapping')
        for class_name, idx in class_to_idx.items():
            class_group.attrs[class_name] = idx
        
        # Store weights
        weights_group = f.create_group('weights')
        
        if hasattr(model, 'models'):
            # For ensemble model
            weights_group.create_dataset('ensemble_weights', data=model.weights.cpu().numpy())
            
            for i, sub_model in enumerate(model.models):
                model_group = weights_group.create_group(f'model_{i}')
                for name, param in sub_model.state_dict().items():
                    model_group.create_dataset(name, data=param.cpu().numpy())
        else:
            # For single model
            for name, param in model.state_dict().items():
                weights_group.create_dataset(name, data=param.cpu().numpy())
    
    print(f"Model exported to H5 format: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Export model for deployment')
    parser.add_argument('--model', type=str, default='models/ensemble_model.pth', help='Path to PyTorch model')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'h5'], default='onnx', help='Export format')
    parser.add_argument('--output', type=str, help='Output path (optional)')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        model_name = os.path.basename(args.model).replace('.pth', '')
        if args.format == 'onnx':
            args.output = f"models/{model_name}.onnx"
        elif args.format == 'torchscript':
            args.output = f"models/{model_name}.pt"
        elif args.format == 'h5':
            args.output = f"models/{model_name}.h5"
    
    # Export model
    if args.format == 'onnx':
        export_to_onnx(args.model, args.output, args.img_size)
    elif args.format == 'torchscript':
        export_to_torchscript(args.model, args.output, args.img_size)
    elif args.format == 'h5':
        export_to_h5(args.model, args.output)

if __name__ == "__main__":
    main() 