from backbones.resnets import *
from backbones.alexnet import *
from backbones.convnext import *

backbones = {'ResNet18Encoder': ResNet18Encoder,
             'ResNet34Encoder': ResNet34Encoder,
             'ResNet50Encoder': ResNet50Encoder,
             'ResNet152Encoder': ResNet152Encoder,
             'convnext_tiny': convnext_tiny,
             'convnext_small': convnext_small,
             }

backbones_with_clf = {'AlexNet': AlexNet,
                      'ResNet18': ResNet18,
                      'ResNet34': ResNet34,
                      }
