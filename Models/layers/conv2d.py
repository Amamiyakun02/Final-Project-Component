import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        # Simpan input untuk backward pass
        self.x = x

        batch_size, in_c, in_h, in_w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        # Padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        out_h = (in_h + 2 * p - k) // s + 1
        out_w = (in_w + 2 * p - k) // s + 1
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(in_c):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = x_padded[b, ic, i * s:i * s + k, j * s:j * s + k]
                            out[b, oc, i, j] += np.sum(region * self.weights[oc, ic])
                out[b, oc] += self.bias[oc]

        return out
