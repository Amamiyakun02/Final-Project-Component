
from Models.layers.batchnorm import BatchNorm2D
from Models.layers.conv2d import Conv2D
from Models.layers.fc import FullyConnected
from Models.layers.flatten import Flatten
from Models.layers.loss import SoftmaxCrossEntropyLoss
from Models.layers.maxpool import MaxPool2D
from Models.layers.relu import ReLU
from Models.layers.residual_block import ResidualBlock
from Models.layers.training_loop import train


import numpy as np

class ResNet16:
    def __init__(self, num_classes=3):
        self.layers = []

        # Initial Conv + BN + ReLU
        self.conv1 = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2D(64)
        self.relu = ReLU()
        self.pool = MaxPool2D(kernel_size=2, stride=2)

        # Layer 1: 2 blocks with 64 channels
        self.layer1 = [
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        ]

        # Layer 2: 2 blocks with 128 channels, first block downsamples
        self.layer2 = [
            ResidualBlock(64, 128, stride=2, downsample=True),
            ResidualBlock(128, 128)
        ]

        # Layer 3: 2 blocks with 256 channels
        self.layer3 = [
            ResidualBlock(128, 256, stride=2, downsample=True),
            ResidualBlock(256, 256)
        ]

        # Layer 4: 2 blocks with 512 channels
        self.layer4 = [
            ResidualBlock(256, 512, stride=2, downsample=True),
            ResidualBlock(512, 512)
        ]

        self.flatten = Flatten()
        self.fc = FullyConnected(512 * 4 * 4, num_classes)  # Assuming input image is 64x64

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        out = self.pool.forward(out)

        for block in self.layer1:
            out = block.forward(out)
        for block in self.layer2:
            out = block.forward(out)
        for block in self.layer3:
            out = block.forward(out)
        for block in self.layer4:
            out = block.forward(out)

        out = self.flatten.forward(out)
        out = self.fc.forward(out)

        return out