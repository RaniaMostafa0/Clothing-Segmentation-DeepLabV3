from dataset import NUM_CLASSES
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101

def get_model(device='cuda'):
    model = deeplabv3_resnet101(weights='DEFAULT')
    
    # replace final layer for 11 classes instead of 21 COCO classes
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    
    return model.to(device)