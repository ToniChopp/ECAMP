import ipdb
import torch
import torch.nn as nn
from torchvision import models as models_2d
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


class DetResNet50(ResNet):
    def __init__(self, pretrained, in_channels):
        super(DetResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


def resnet_50(pretrained=True, in_channels=3):
    model = DetResNet50(pretrained=pretrained, in_channels=in_channels)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


class ResNetDetector(nn.Module):
    def __init__(self, model_name, pretrained=True, in_channels=3):
        super().__init__()

        model_function = resnet_50
        self.model, self.feature_dim, self.interm_feature_dim = model_function(
            pretrained=pretrained,
            in_channels=in_channels
        )

        if model_name == "resnet_50":
            self.filters = [512, 1024, 2048]

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        out3 = self.model.layer2(x)   # bz, 512, 28
        out4 = self.model.layer3(out3)
        out5 = self.model.layer4(out4)

        return out3, out4, out5


