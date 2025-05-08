from conv2d import Conv2D
from batchnorm import BatchNorm2D
from relu import ReLU
import numpy as np

class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        self.downsample = downsample
        self.stride = stride
        self.equal_in_out = (in_channels == out_channels)

        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()

        if self.downsample or not self.equal_in_out:
            self.shortcut = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.bn_shortcut = BatchNorm2D(out_channels)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.shortcut:
            identity = self.shortcut.forward(x)
            identity = self.bn_shortcut.forward(identity)

        out += identity
        out = self.relu2.forward(out)

        return out
