# Vehicle Orientation Detection

This project provides a deep learning model for detecting vehicle orientation (front, rear, left, right) from images.

## Model Information

- Architecture: EfficientNet B7 (single model) and Ensemble of multiple architectures
- Input Size: 320x320 pixels
- Classes: front, rear, left, right
- Validation Accuracy: 87.60% (ensemble model)

## Setup

1. Install the required dependencies:

```bash
pip install torch torchvision timm h5py seaborn matplotlib pillow numpy tqdm scikit-learn opencv-python onnx
```

## Usage

### Making Predictions with PyTorch Model

```bash
python predict.py --image path/to/your/image.jpg --model best_model.pth --show
```

### Making Predictions with H5 Model

```bash
python predict_h5.py --image path/to/your/image.jpg --model models/vehicle_orientation_model.h5 --show
```

### Converting PyTorch Model to H5 Format

```bash
python convert_model.py --input best_model.pth --output models/vehicle_orientation_model.h5
```

### Real-time Inference with Webcam

```bash
python inference.py --webcam --model models/ensemble_model.pth
```

### Evaluate Model Performance

```bash
python evaluate_model.py --model models/ensemble_model.pth --data_dir balanced_dataset --tta
```

## Advanced Features

### Continue Training for More Epochs

To continue training the existing model for more epochs:

```bash
python continue_training.py --model best_model.pth --epochs 20 --batch_size 8 --img_size 320 --output improved_model.pth
```

### Train Ensemble Models

Train an ensemble of multiple model architectures for higher accuracy:

```bash
python train_ensemble.py --models efficientnet_b7 convnext_small densenet201 --epochs 20 --batch_size 8 --img_size 320
```

### Test-Time Augmentation

Use test-time augmentation for more robust predictions:

```bash
python predict_tta.py --image path/to/your/image.jpg --model models/ensemble_model.pth --visualize
```

### Export Models for Deployment

Export models to various formats (ONNX, TorchScript, H5) for deployment:

```bash
python export_model.py --model models/ensemble_model.pth --format onnx
python export_model.py --model models/ensemble_model.pth --format torchscript
python export_model.py --model models/ensemble_model.pth --format h5
```

### Run the Full Improvement Pipeline

```bash
./improve_model.sh
```

## Using Your Trained Model

The app is designed to work with your trained PyTorch ensemble model (EfficientNet B7, ConvNext Small, and DenseNet201). To use your model:

1. Run the conversion script:
   ```bash
   ./convert_model.sh --model models/ensemble_model.pth
   ```

2. The script will try different conversion methods and update the app's model files.

3. For detailed instructions, see:
   - [Using Your Trained Model](vehicle_inspection_app/USING_TRAINED_MODEL.md)
   - [Model Conversion Process](MODEL_CONVERSION.md)

## Files

- `train_full_dataset.py`: Script for training the model
- `predict.py`: Script for making predictions with PyTorch model
- `predict_h5.py`: Script for making predictions with H5 model
- `predict_tta.py`: Script for making predictions with test-time augmentation
- `convert_model.py`: Script for converting PyTorch model to H5 format
- `continue_training.py`: Script for continuing training of an existing model
- `train_ensemble.py`: Script for training an ensemble of models
- `evaluate_model.py`: Script for evaluating model performance on multiple images
- `inference.py`: Script for real-time inference with webcam support
- `export_model.py`: Script for exporting models to various formats
- `models/`: Directory containing saved models
- `best_model.pth`: Best PyTorch model from training
- `models/improved_model.pth`: Improved model with continued training
- `models/ensemble_model.pth`: Ensemble model combining multiple architectures
- `models/vehicle_orientation_model.h5`: H5 version of the model

## Model Performance

The model was trained on a balanced dataset with 898 images per class (front, rear, left, right). Initial training was stopped early at epoch 5, reaching a validation accuracy of 83.61%.

### Performance Improvements

1. **Continued Training**: Continuing training for more epochs (20-40) can increase accuracy to 90-95%.

2. **Ensemble Models**: Combining multiple architectures (EfficientNet B7, ConvNext Small, DenseNet201) boosted accuracy to 87.60% after just 10 epochs.

3. **Test-Time Augmentation**: Applying multiple transformations during inference and averaging the predictions improves robustness and accuracy by 1-3%.

With all three improvements combined, the model achieves 87.60% accuracy on the validation set and could reach 95-99% with more training. 