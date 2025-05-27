import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import warnings

# Suppress PIL warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from utils import get_device
from models import get_model

def load_checkpoint(model_path, device):
    """
    Load a checkpoint to continue training.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
    
    Returns:
        model: Loaded model
        class_to_idx: Class to index mapping
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # If it's a state_dict only
    if all(k.startswith('blocks.') or k.startswith('conv_stem.') or k.startswith('bn') or k.startswith('classifier.') for k in checkpoint.keys()):
        class_to_idx = {
            'front': 0,
            'rear': 1,
            'left': 2,
            'right': 3
        }
        model_name = 'efficientnet_b7'
        model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
        model.load_state_dict(checkpoint)
    else:
        # If it's a full checkpoint
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
        else:
            class_to_idx = {
                'front': 0,
                'rear': 1,
                'left': 2,
                'right': 3
            }
            
        if 'args' in checkpoint and 'model' in checkpoint['args']:
            model_name = checkpoint['args']['model']
        else:
            model_name = 'efficientnet_b7'
            
        model = get_model(model_name, num_classes=len(class_to_idx), pretrained=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    return model, class_to_idx, model_name

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, start_epoch=0, save_path='best_model.pth'):
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train for
        device: Device to train on
        start_epoch: Starting epoch number
        save_path: Path to save the best model
    """
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch + num_epochs}')
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
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{start_epoch + num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Continue training the vehicle orientation model')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--img_size', type=int, default=320, help='Image size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output', type=str, default='improved_model.pth', help='Path to save the improved model')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, class_to_idx, model_name = load_checkpoint(args.model, device)
    print(f"Model loaded successfully. Architecture: {model_name}")
    
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
    
    # Check if class_to_idx matches the dataset
    if train_dataset.class_to_idx != class_to_idx:
        print("Warning: Class mapping in model doesn't match dataset. Adjusting model...")
        # Create a new model with the correct number of classes
        model = get_model(model_name, num_classes=len(train_dataset.class_to_idx), pretrained=False)
        model = model.to(device)
    
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
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device, save_path=args.output)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.dataset.class_to_idx,
        'args': {'model': model_name}
    }, args.output.replace('.pth', '_final.pth'))
    
    print(f"Training completed. Final model saved to {args.output.replace('.pth', '_final.pth')}")

if __name__ == "__main__":
    main() 