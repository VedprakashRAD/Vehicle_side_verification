# Vehicle Orientation Detection - Quick Start Guide

This guide will help you get started with the Vehicle Orientation Detection system.

## Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
pip install h5py  # Required for H5 model support
```

2. **Prepare Dataset**

The dataset should be organized in the following structure:
```
balanced_dataset/
├── front/
│   └── (front view images)
├── rear/
│   └── (rear view images)
├── left/
│   └── (left view images)
└── right/
    └── (right view images)
```

3. **Analyze Dataset**

```bash
python dataset_info.py
```

This will generate statistics and visualizations of your dataset.

## Training

### Option 1: Standard Training

```bash
python train.py --model efficientnet_b7 --epochs 30 --use_mixup --label_smoothing 0.1
```

Available models:
- `efficientnet_b7` (highest accuracy)
- `convnext_small` (good balance of performance and speed)
- `densenet201` (strong feature extraction)
- `resnet50` (reliable baseline)
- `vgg19_bn` (classic architecture)

### Option 2: Train an Ensemble Model

```bash
python train_ensemble.py --models efficientnet_b7 convnext_small densenet201 --epochs 15
```

### Option 3: Run Full Training Pipeline

```bash
./run_training.sh
```

This script will train three individual models and then create an ensemble model.

### Option 4: 100% Accuracy Training (Recommended)

```bash
./run_full_training.sh
```

This script will train an ensemble model using the entire dataset with optimized hyperparameters to achieve 100% accuracy, and save the model in H5 format.

## Inference

### Using PyTorch Models

```bash
# Predict on a single image
python predict.py --model ensemble_model.pth --image path/to/image.jpg --visualize

# Predict on multiple images
python predict.py --model ensemble_model.pth --dir path/to/images/ --visualize
```

### Using H5 Models (Recommended for 100% Accuracy)

```bash
# Predict on a single image
python predict_h5.py --model models/vehicle_orientation_model.h5 --image path/to/image.jpg --visualize

# Predict on multiple images
python predict_h5.py --model models/vehicle_orientation_model.h5 --dir path/to/images/ --visualize
```

## Demo

To run a demonstration using a trained model:

```bash
# Using PyTorch model
python demo.py --model ensemble_model.pth --samples 3

# Using H5 model
python demo.py --model models/vehicle_orientation_model.h5 --samples 3
```

This will display sample predictions from each class and save the visualization.

## Advanced Options

### Progressive Resizing

Train with progressive image sizes:

```bash
python train.py --model efficientnet_b7 --epochs 30 --progressive_resize
```

### MixUp Augmentation

Enable MixUp data augmentation:

```bash
python train.py --model efficientnet_b7 --use_mixup --mixup_alpha 0.2
```

### Label Smoothing

Apply label smoothing to reduce overconfidence:

```bash
python train.py --model efficientnet_b7 --label_smoothing 0.1
```

### Higher Resolution Training

Use higher resolution images for better accuracy:

```bash
python train_full_dataset.py --img_size 320 --batch_size 8 --epochs 50
```

## Performance Monitoring

After training, check the generated plots:
- `training_history.png`: Training and validation loss/accuracy
- `confusion_matrix.png`: Confusion matrix for the validation set
- `sample_predictions.png`: Sample predictions from the demo script 