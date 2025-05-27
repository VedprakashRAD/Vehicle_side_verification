#!/bin/bash

# Script to train vehicle orientation detection models

# Check if the dataset exists
if [ ! -d "balanced_dataset" ]; then
    echo "Error: balanced_dataset directory not found."
    exit 1
fi

# Create a directory for model outputs
mkdir -p models

# Set common parameters
BATCH_SIZE=16
EPOCHS=30
IMG_SIZE=224
LR=0.0001

# 1. Train a single model (EfficientNet B7)
echo "=== Training EfficientNet B7 ==="
python train.py \
    --model efficientnet_b7 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --lr $LR \
    --use_mixup \
    --label_smoothing 0.1

# Move the model to the models directory
mv efficientnet_b7_final.pth models/
mv best_model.pth models/efficientnet_b7_best.pth

# 2. Train a single model (ConvNext Small)
echo "=== Training ConvNext Small ==="
python train.py \
    --model convnext_small \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --lr $LR \
    --use_mixup \
    --label_smoothing 0.1

# Move the model to the models directory
mv convnext_small_final.pth models/
mv best_model.pth models/convnext_small_best.pth

# 3. Train a single model (DenseNet201)
echo "=== Training DenseNet201 ==="
python train.py \
    --model densenet201 \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --lr $LR \
    --use_mixup \
    --label_smoothing 0.1

# Move the model to the models directory
mv densenet201_final.pth models/
mv best_model.pth models/densenet201_best.pth

# 4. Train the ensemble model
echo "=== Training Ensemble Model ==="
python train_ensemble.py \
    --models efficientnet_b7 convnext_small densenet201 \
    --batch_size $BATCH_SIZE \
    --epochs 15 \
    --img_size $IMG_SIZE \
    --lr $LR \
    --label_smoothing 0.1

# Move the ensemble model to the models directory
mv ensemble_model.pth models/

echo "=== Training Complete ==="
echo "All models saved in the 'models' directory."
echo "Run 'python demo.py --model models/ensemble_model.pth' to test the ensemble model." 