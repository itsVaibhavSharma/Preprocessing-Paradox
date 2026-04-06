import torch.nn as nn
import torchvision.models as models

def build_optimized_model(model_type, n_classes):
    """Build and optimize model"""
    try:
        if model_type == 'mobilenetv2':
            model = models.mobilenet_v2(pretrained=True)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, n_classes)
            )
        elif model_type == 'squeezenet':
            model = models.squeezenet1_1(pretrained=True)
            # SqueezeNet has a different classifier structure
            model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=1)
            model.num_classes = n_classes
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise
