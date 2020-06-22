# For image classifier, we can use the implemented one from pytorch lib? VGG-16
import torch
from torch import nn


class VGG16_IMG_CLASSIFIER(nn.Module):
    def __init__(self):
        super(VGG16_IMG_CLASSIFIER, self).__init__()
        self.conv2d_1 = nn.Sequential( #Todo: Padding same
            nn.Conv2d(in_channels=3, out_channels=32)
        )