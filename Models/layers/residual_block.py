# layers/residual_block.py
import numpy as np
from Models.layers.conv2d import Conv2D
from Models.layers.batchnorm import BatchNorm2D
from Models.layers.relu import ReLU

class ResidualBlock:
    def __init__(self, in_channels, out_channels, downsample=False):
        stride = 2 if downsample else 1

        # Main path
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)

        # Shortcut path
        self.downsample = downsample
        if downsample or in_channels != out_channels:
            self.shortcut_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.shortcut_bn = BatchNorm2D(out_channels)
        else:
            self.shortcut_conv = None

        self.relu_out = ReLU()

    def forward(self, x):
        self.input = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.downsample or self.shortcut_conv:
            shortcut = self.shortcut_conv.forward(x)
            shortcut = self.shortcut_bn.forward(shortcut)
        else:
            shortcut = x

        self.output = self.relu_out.forward(out + shortcut)
        return self.output

    def backward(self, grad_output, learning_rate):
        grad = self.relu_out.backward(grad_output)

        if self.downsample or self.shortcut_conv:
            grad_shortcut = self.shortcut_bn.backward(grad, learning_rate)
            grad_shortcut = self.shortcut_conv.backward(grad_shortcut, learning_rate)
        else:
            grad_shortcut = grad

        grad = self.bn2.backward(grad, learning_rate)
        grad = self.conv2.backward(grad, learning_rate)
        grad = self.relu1.backward(grad)
        grad = self.bn1.backward(grad, learning_rate)
        grad = self.conv1.backward(grad, learning_rate)

        grad_input = grad + grad_shortcut
        return grad_input
