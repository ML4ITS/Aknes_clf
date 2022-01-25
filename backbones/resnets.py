from torchvision import models
import torch
import torch.nn as nn
from utils.types_ import *


# ================================================
# Encoder


class ResNet18Encoder(nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

        # add dropout
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for l in layer:
                l.relu = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout_rate))

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


class ResNet34Encoder(nn.Module):
    def __init__(self, dropout_rate: float = 0.0):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

        # add dropout
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for l in layer:
                l.relu = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout_rate))

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


class ResNet152Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet152(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


# ================================================
# Encoder + Classifier

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.model.fc = nn.Linear(in_features=512, out_features=8, bias=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


if __name__ == '__main__':
    # toy dataset
    B = 32
    C = 1
    H, W = 128, 71
    X = torch.rand(B, C, H, W)

    # modify model
    encoder = ResNet34Encoder()

    # forward
    out = encoder(X)
    print('out.shape:', out.shape)
