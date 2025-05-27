import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from utils import set_seed, get_device, prepare_data, evaluate_model, plot_confusion_matrix, plot_training_history, mixup_data, mixup_criterion
from models import get_model, LabelSmoothingCrossEntropy, ProgressiveResizeTrainer

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, 
                use_mixup=True, mixup_alpha=0.2, use_progressive_resize=False, progressive_trainer=None):
    """
    Train the model.
    
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
        use_progressive_resize: Whether to use progressive resizing
        progressive_trainer: Progressive resize trainer instance
        
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
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        if use_progressive_resize and progressive_trainer is not None:
            current_size = progressive_trainer.get_current_size(epoch)
            print(f"Epoch {epoch+1}/{num_epochs}, Image size: {current_size}")
        else:
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
        
        if scheduler is not None:
            scheduler.step()
            
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train a vehicle orientation detection model')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to dataset')
    parser.add_argument('--model', type=str, default='efficientnet_b7', 
                        choices=['efficientnet_b7', 'convnext_small', 'densenet201', 'resnet50', 'vgg19_bn'],
                        help='Model architecture to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_mixup', action='store_true', help='Use mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Alpha parameter for mixup')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--progressive_resize', action='store_true', help='Use progressive resizing')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader, class_to_idx = prepare_data(
        args.data_dir, 
        img_size=args.img_size, 
        batch_size=args.batch_size
    )
    
    # Create model
    model = get_model(args.model, num_classes=len(class_to_idx), pretrained=True)
    model = model.to(device)
    
    # Define loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Define scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Progressive resize trainer
    if args.progressive_resize:
        sizes = [128, 160, 224]
        epochs_per_size = [args.epochs // 3, args.epochs // 3, args.epochs - 2 * (args.epochs // 3)]
        progressive_trainer = ProgressiveResizeTrainer(model, sizes=sizes, epochs_per_size=epochs_per_size)
    else:
        progressive_trainer = None
    
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
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_progressive_resize=args.progressive_resize,
        progressive_trainer=progressive_trainer
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on validation set
    _, _, y_true, y_pred = evaluate_model(model, val_loader, criterion, device)
    
    # Plot confusion matrix
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Save final model
    model_save_path = f"{args.model}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'args': vars(args)
    }, model_save_path)
    
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main() 