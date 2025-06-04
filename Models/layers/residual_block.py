# layers/residual_block.py
from Models.layers.conv2d import Conv2D
from Models.layers.batchnorm2d import BatchNorm2D
from Models.layers.relu import ReLU

class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu = ReLU()

        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)

        self.use_projection = (stride != 1 or in_channels != out_channels)
        if self.use_projection:
            self.shortcut_conv = Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)
            self.shortcut_bn = BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.use_projection:
            shortcut = self.shortcut_conv.forward(x)
            shortcut = self.shortcut_bn.forward(shortcut)
        else:
            shortcut = x  # asumsi ukuran sama

        out += shortcut
        out = self.relu.forward(out)
        return out

    def backward(self, grad_output):
        grad_output = self.relu.backward(grad_output)

        grad_shortcut = grad_output.copy()
        grad_main = grad_output.copy()

        # Backward main path
        grad_main = self.bn2.backward(grad_main)
        grad_main = self.conv2.backward(grad_main)
        grad_main = self.relu.backward(grad_main)
        grad_main = self.bn1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)

        # Backward shortcut
        if self.use_projection:
            grad_shortcut = self.shortcut_bn.backward(grad_shortcut)
            grad_shortcut = self.shortcut_conv.backward(grad_shortcut)

        # Combine grads
        grad_input = grad_main + grad_shortcut
        return grad_input
