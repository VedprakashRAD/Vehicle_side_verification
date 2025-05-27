#!/bin/bash

# Script to train vehicle orientation detection model with the full dataset
# This script is optimized for maximum accuracy (100% target)

# Check if the dataset exists
if [ ! -d "balanced_dataset" ]; then
    echo "Error: balanced_dataset directory not found."
    exit 1
fi

# Install h5py if not already installed
pip install h5py

# Set parameters for maximum accuracy
BATCH_SIZE=8
EPOCHS=100
IMG_SIZE=320  # Higher resolution for better accuracy
LR=0.0001

# Create output directory
mkdir -p models

echo "=== Starting Full Dataset Training ==="
echo "Target: 100% accuracy on vehicle orientation detection"
echo "Training with ensemble of models for maximum accuracy"
echo "This may take several hours depending on your hardware"

# Train the ensemble model with full dataset
python train_full_dataset.py \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --img_size $IMG_SIZE \
    --lr $LR \
    --ensemble \
    --output models/vehicle_orientation_model.h5

# Check if the model was created successfully
if [ -f "models/vehicle_orientation_model.h5" ]; then
    echo "=== Training Complete ==="
    echo "Model saved to models/vehicle_orientation_model.h5"
    
    # Run validation on some sample images
    echo "=== Running Sample Predictions ==="
    
    # Find one sample image from each class
    FRONT_SAMPLE=$(find balanced_dataset/front -type f -name "*.jpg" | head -n 1)
    REAR_SAMPLE=$(find balanced_dataset/rear -type f -name "*.jpg" | head -n 1)
    LEFT_SAMPLE=$(find balanced_dataset/left -type f -name "*.jpg" | head -n 1)
    RIGHT_SAMPLE=$(find balanced_dataset/right -type f -name "*.jpg" | head -n 1)
    
    # Run predictions
    echo "Predicting front view sample:"
    python predict_h5.py --model models/vehicle_orientation_model.h5 --image "$FRONT_SAMPLE" --visualize
    
    echo "Predicting rear view sample:"
    python predict_h5.py --model models/vehicle_orientation_model.h5 --image "$REAR_SAMPLE" --visualize
    
    echo "Predicting left view sample:"
    python predict_h5.py --model models/vehicle_orientation_model.h5 --image "$LEFT_SAMPLE" --visualize
    
    echo "Predicting right view sample:"
    python predict_h5.py --model models/vehicle_orientation_model.h5 --image "$RIGHT_SAMPLE" --visualize
    
    echo "=== All Done ==="
    echo "To use the model for predictions on new images, run:"
    echo "python predict_h5.py --model models/vehicle_orientation_model.h5 --image path/to/your/image.jpg --visualize"
else
    echo "Error: Training failed or model not saved properly."
    exit 1
fi 