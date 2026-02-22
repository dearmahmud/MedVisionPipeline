import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

class ResNetPneumonia(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            self.model = models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        else:
            self.model = models.resnet18(weights=None)

        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)