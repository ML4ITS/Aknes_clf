from torchvision import models
import torch.nn as nn
from utils.types_ import *


class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.model = models.alexnet(pretrained=False)

        # modify the first conv layer
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

        # modify the classification layer (last FC layer)
        self.model.classifier[-1] = nn.Linear(4096, n_classes)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


if __name__ == '__main__':
    import torch

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # toy dataset
    B = 32
    C = 1
    W, H = 224, 500
    X = torch.rand(B, C, W, H)

    # modify model
    encoder = AlexNet()
    print(encoder, '\n')
    print(count_parameters(encoder))

    # forward
    out = encoder(X)
    print('out.shape:', out.shape, '\n')
