from torchvision import models
import torch
import torch.nn as nn
from utils.types_ import *


# ================================================
# Encoder


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.fc = nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        out = self.model(input).squeeze()
        return out


# class ResNet34Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.resnet34(pretrained=False)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.model.fc = nn.Identity()
#
#         self.conv1x1_u1 = nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False)
#         self.bn_u1 = nn.BatchNorm2d(512)
#         self.repr_comp_linear = nn.Linear(512*2, 512)
#
#     def forward(self, input: Tensor) -> Tensor:
#         out = self.model.conv1(input)
#         out = self.model.bn1(out)
#         out = self.model.relu(out)
#         u1 = self.model.maxpool(out)
#
#         u1_c = self.model.avgpool(self.bn_u1(self.conv1x1_u1(u1)))  # c: compressed; (B, 512)
#
#         out = self.model.layer1(u1)
#         out = self.model.layer2(out)
#         out = self.model.layer3(out)
#         out = self.model.layer4(out)
#
#         out = self.model.avgpool(out)
#
#         out = torch.cat((out, u1_c), dim=1).squeeze()
#         out = self.repr_comp_linear(out)  # (B, 512)
#         return out


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
    H, W = 128, 711
    X = torch.rand(B, C, H, W)

    # modify model
    encoder = ResNet34Encoder()
    print(encoder)

    # forward
    out = encoder(X)
    print('out.shape:', out.shape)
