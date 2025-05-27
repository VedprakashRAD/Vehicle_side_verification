import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import set_seed, get_device, prepare_data, evaluate_model, plot_confusion_matrix, plot_training_history
from models import get_model, LabelSmoothingCrossEntropy, EnsembleModel

def train_individual_models(model_names, train_loader, val_loader, criterion, device, args):
    """
    Train individual models for ensemble.
    
    Args:
        model_names: List of model names to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to train on
        args: Training arguments
        
    Returns:
        trained_models: List of trained models
        val_accuracies: List of validation accuracies
    """
    trained_models = []
    val_accuracies = []
    
    for model_name in model_names:
        print(f"\nTraining {model_name}...")
        
        # Create model
        model = get_model(model_name, num_classes=len(args.class_to_idx), pretrained=True)
        model = model.to(device)
        
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        
        # Define scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Train model
        best_val_acc = 0.0
        
        for epoch in range(args.epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}/{args.epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                progress_bar.set_postfix(loss=loss.item())
            
            if scheduler is not None:
                scheduler.step()
                
            # Validation phase
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{model_name}_best.pth')
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'{model_name}_best.pth'))
        
        # Final evaluation
        _, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        trained_models.append(model)
        val_accuracies.append(val_acc)
        
        print(f"{model_name} final validation accuracy: {val_acc:.4f}")
    
    return trained_models, val_accuracies

def train_single_model(model_name, train_loader, val_loader, num_classes, device, num_epochs=20, img_size=320, lr=0.0001, save_dir='models'):
    """
    Train a single model.
    
    Args:
        model_name: Name of the model architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
        device: Device to train on
        num_epochs: Number of epochs to train for
        img_size: Image size
        lr: Learning rate
        save_dir: Directory to save the model
    
    Returns:
        model: Trained model
        best_val_acc: Best validation accuracy
    """
    print(f"Training {model_name}...")
    
    # Create model
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Train the model
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best model with validation accuracy: {val_acc:.2f}%')
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, best_val_acc

def train_ensemble(model_names, train_loader, val_loader, class_to_idx, device, num_epochs=20, img_size=320, lr=0.0001, save_dir='models'):
    """
    Train an ensemble of models.
    
    Args:
        model_names: List of model architectures
        train_loader: Training data loader
        val_loader: Validation data loader
        class_to_idx: Class to index mapping
        device: Device to train on
        num_epochs: Number of epochs to train for
        img_size: Image size
        lr: Learning rate
        save_dir: Directory to save the models
    
    Returns:
        ensemble: Trained ensemble model
        best_val_acc: Best validation accuracy
    """
    # Train individual models
    models = []
    accuracies = []
    
    for model_name in model_names:
        model, val_acc = train_single_model(model_name, train_loader, val_loader, len(class_to_idx), device, num_epochs, img_size, lr, save_dir)
        models.append(model)
        accuracies.append(val_acc)
    
    # Calculate weights based on validation accuracies
    weights = np.array(accuracies) / sum(accuracies)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights=weights)
    ensemble.eval()
    
    # Evaluate ensemble
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = ensemble(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    ensemble_acc = 100. * correct / total
    print(f'Ensemble Validation Accuracy: {ensemble_acc:.2f}%')
    
    # Save ensemble model
    os.makedirs(save_dir, exist_ok=True)
    ensemble_path = os.path.join(save_dir, 'ensemble_model.pth')
    
    # Save model states and weights
    model_states = {model_name: models[i].state_dict() for i, model_name in enumerate(model_names)}
    
    torch.save({
        'model_names': model_names,
        'model_states': model_states,
        'weights': weights,
        'class_to_idx': class_to_idx
    }, ensemble_path)
    
    print(f"Ensemble model saved to {ensemble_path}")
    
    # Save as H5 format
    try:
        import h5py
        h5_path = os.path.join(save_dir, 'ensemble_model.h5')
        
        with h5py.File(h5_path, 'w') as f:
            # Store model architecture as attributes
            f.attrs['model_type'] = 'ensemble'
            
            # Store class mapping
            class_group = f.create_group('class_mapping')
            for class_name, idx in class_to_idx.items():
                class_group.attrs[class_name] = idx
            
            # Store weights
            weights_group = f.create_group('weights')
            weights_group.create_dataset('ensemble_weights', data=weights.cpu().numpy())
            
            # Store model weights
            for i, model_name in enumerate(model_names):
                model_group = weights_group.create_group(f'model_{i}')
                for name, param in models[i].state_dict().items():
                    model_group.create_dataset(name, data=param.cpu().numpy())
        
        print(f"Ensemble model saved in H5 format to {h5_path}")
    except ImportError:
        print("h5py not found. Skipping H5 format save.")
    
    return ensemble, ensemble_acc

def main():
    parser = argparse.ArgumentParser(description='Train an ensemble of models for vehicle orientation detection')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to the dataset directory')
    parser.add_argument('--models', nargs='+', default=['efficientnet_b7', 'convnext_small', 'densenet201'], 
                        help='List of models to include in the ensemble')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save the models')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Define data transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=train_transform)
    
    # Split dataset into train and validation
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Apply different transforms to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Train ensemble
    class_to_idx = train_dataset.dataset.class_to_idx
    ensemble, ensemble_acc = train_ensemble(args.models, train_loader, val_loader, class_to_idx, device, 
                                           args.epochs, args.img_size, args.lr, args.save_dir)
    
    print(f"Training completed. Ensemble accuracy: {ensemble_acc:.2f}%")

if __name__ == "__main__":
    main() 