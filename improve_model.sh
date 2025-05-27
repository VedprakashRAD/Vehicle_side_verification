#!/bin/bash

# Create models directory if it doesn't exist
mkdir -p models

# Step 1: Continue training the existing model for more epochs
echo "Step 1: Continuing training for more epochs..."
python continue_training.py --model best_model.pth --epochs 20 --batch_size 8 --img_size 320 --output models/improved_model.pth
if [ $? -ne 0 ]; then
    echo "Error in continued training. Exiting."
    exit 1
fi

# Step 2: Train an ensemble of models
echo "Step 2: Training ensemble models..."
python train_ensemble.py --models efficientnet_b7 convnext_small densenet201 --epochs 10 --batch_size 8 --img_size 320 --save_dir models
if [ $? -ne 0 ]; then
    echo "Error in ensemble training. Exiting."
    exit 1
fi

# Step 3: Test the ensemble model with TTA on a sample image
echo "Step 3: Testing ensemble model with test-time augmentation..."
if [ -f "models/ensemble_model.pth" ]; then
    python predict_tta.py --image ./balanced_dataset/front/Biketrain190.jpg --model models/ensemble_model.pth --visualize
else
    echo "Ensemble model not found. Skipping TTA test."
fi

echo "Model improvement pipeline completed!"
echo "You can now use the improved models:"
if [ -f "models/improved_model.pth" ]; then
    echo "- Single model: models/improved_model.pth"
fi
if [ -f "models/ensemble_model.pth" ]; then
    echo "- Ensemble model: models/ensemble_model.pth"
fi
if [ -f "models/ensemble_model.h5" ]; then
    echo "- H5 ensemble model: models/ensemble_model.h5"
fi 