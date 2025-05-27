import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import h5py

from utils import set_seed, get_device, prepare_data, evaluate_model, plot_confusion_matrix, plot_training_history, mixup_data, mixup_criterion
from models import get_model, LabelSmoothingCrossEntropy, EnsembleModel

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50, 
                use_mixup=True, mixup_alpha=0.3, early_stopping_patience=10):
    """
    Train the model with advanced techniques for maximum accuracy.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Alpha parameter for mixup
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        model: Trained model
        history: Training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Apply mixup if enabled
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # For accuracy calculation (only meaningful without mixup)
            if not use_mixup:
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Only calculate training accuracy if not using mixup
        if not use_mixup:
            epoch_acc = correct / total
        else:
            # Run a separate evaluation pass without mixup to get training accuracy
            train_loss, epoch_acc, _, _ = evaluate_model(model, train_loader, criterion, device)
        
        # Validation phase
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # If we've reached 100% accuracy, we can stop
        if val_acc >= 0.999:
            print("Reached 100% validation accuracy!")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, history

def create_ensemble(models, val_loader, criterion, device):
    """Create an ensemble of models with weights based on validation accuracy."""
    # Evaluate each model
    val_accuracies = []
    for model in models:
        _, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)
        print(f"Model validation accuracy: {val_acc:.4f}")
    
    # Create ensemble
    ensemble = EnsembleModel(models, weights=val_accuracies)
    
    return ensemble, val_accuracies

def save_model_h5(model, class_to_idx, model_path):
    """Save PyTorch model in H5 format."""
    # First save as PyTorch format
    torch_path = model_path.replace('.h5', '.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }, torch_path)
    
    # Convert to H5 format
    with h5py.File(model_path, 'w') as f:
        # Store model architecture as attributes
        f.attrs['model_type'] = 'ensemble' if isinstance(model, EnsembleModel) else 'single'
        
        # Store class mapping
        class_group = f.create_group('class_mapping')
        for class_name, idx in class_to_idx.items():
            class_group.attrs[class_name] = idx
        
        # Store model weights
        weights_group = f.create_group('weights')
        
        if isinstance(model, EnsembleModel):
            # For ensemble model
            for i, sub_model in enumerate(model.models):
                model_group = weights_group.create_group(f'model_{i}')
                for name, param in sub_model.state_dict().items():
                    model_group.create_dataset(name, data=param.cpu().numpy())
                
            # Store ensemble weights
            weights_group.create_dataset('ensemble_weights', data=model.weights.cpu().numpy())
        else:
            # For single model
            for name, param in model.state_dict().items():
                weights_group.create_dataset(name, data=param.cpu().numpy())
    
    print(f"Model saved in H5 format to {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser(description='Train vehicle orientation detection model with full dataset')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ensemble', action='store_true', help='Train an ensemble of models')
    parser.add_argument('--output', type=str, default='vehicle_orientation_model.h5', help='Output model path')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Prepare data with a smaller validation split to use more data for training
    train_loader, val_loader, class_to_idx = prepare_data(
        args.data_dir, 
        img_size=args.img_size, 
        batch_size=args.batch_size,
        val_split=0.1  # Use 90% of data for training
    )
    
    if args.ensemble:
        # Train multiple models for ensemble
        model_names = ['efficientnet_b7', 'convnext_small', 'densenet201']
        trained_models = []
        
        for model_name in model_names:
            print(f"\n=== Training {model_name} ===")
            
            # Create model
            model = get_model(model_name, num_classes=len(class_to_idx), pretrained=True)
            model = model.to(device)
            
            # Define loss function with label smoothing
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            
            # Define optimizer
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
            
            # Define scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            
            # Train model
            model, history = train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                device, 
                num_epochs=args.epochs,
                use_mixup=True,
                mixup_alpha=0.3,
                early_stopping_patience=10
            )
            
            # Plot training history
            plot_training_history(history)
            
            trained_models.append(model)
        
        # Create and evaluate ensemble
        ensemble_model, val_accuracies = create_ensemble(trained_models, val_loader, criterion, device)
        
        # Evaluate ensemble model
        _, ensemble_acc, y_true, y_pred = evaluate_model(ensemble_model, val_loader, criterion, device)
        
        print(f"\nEnsemble model validation accuracy: {ensemble_acc:.4f}")
        
        # Plot confusion matrix
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
        plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Save ensemble model in H5 format
        save_model_h5(ensemble_model, class_to_idx, args.output)
        
    else:
        # Train a single model (EfficientNet B7 for best performance)
        model = get_model('efficientnet_b7', num_classes=len(class_to_idx), pretrained=True)
        model = model.to(device)
        
        # Define loss function with label smoothing
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        
        # Define scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Train model
        model, history = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            device, 
            num_epochs=args.epochs,
            use_mixup=True,
            mixup_alpha=0.3,
            early_stopping_patience=10
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on validation set
        _, _, y_true, y_pred = evaluate_model(model, val_loader, criterion, device)
        
        # Plot confusion matrix
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
        plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Save model in H5 format
        save_model_h5(model, class_to_idx, args.output)

if __name__ == "__main__":
    main() 