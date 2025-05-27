import torch
import torch.nn as nn
import timm
from torchvision import models

# Define a dictionary of available models
def get_model(model_name, num_classes=4, pretrained=True):
    """
    Get a model by name with specified number of classes.
    
    Args:
        model_name (str): Name of the model
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'efficientnet_b7':
        model = timm.create_model('tf_efficientnet_b7', pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    elif model_name == 'convnext_small':
        model = timm.create_model('convnext_small', pretrained=pretrained)
        in_features = model.head.fc.in_features
        model.head.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'densenet201':
        model = models.densenet201(weights='DEFAULT' if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(weights='DEFAULT' if pretrained else None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

# Ensemble model
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        """
        Ensemble of multiple models.
        
        Args:
            models (list): List of PyTorch models
            weights (list, optional): List of weights for each model. If None, equal weights are used.
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            # Use detach().clone() to avoid warning
            if isinstance(weights, torch.Tensor):
                self.weights = weights.detach().clone() / torch.sum(weights)
            else:
                self.weights = torch.tensor(weights, dtype=torch.float32) / sum(weights)
    
    def forward(self, x):
        outputs = []
        for i, model in enumerate(self.models):
            output = model(x)
            outputs.append(output * self.weights[i])
        
        return sum(outputs)

# Model with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.
    
    Args:
        smoothing (float): Label smoothing factor (0-1)
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# Progressive resizing training wrapper
class ProgressiveResizeTrainer:
    """
    Wrapper for progressive resizing training.
    
    Args:
        model: PyTorch model
        sizes (list): List of image sizes to use during training
        epochs_per_size (list): List of epochs to train for each size
    """
    def __init__(self, model, sizes=[128, 160, 224], epochs_per_size=[10, 10, 10]):
        self.model = model
        self.sizes = sizes
        self.epochs_per_size = epochs_per_size
        
    def get_current_size(self, epoch):
        """Get the image size for the current epoch."""
        cumulative_epochs = 0
        for i, epochs in enumerate(self.epochs_per_size):
            cumulative_epochs += epochs
            if epoch < cumulative_epochs:
                return self.sizes[i]
        return self.sizes[-1] 