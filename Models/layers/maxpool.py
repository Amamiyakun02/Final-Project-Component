import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        batch_size, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1

        out = np.zeros((batch_size, c, out_h, out_w))

        for b in range(batch_size):
            for ch in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        region = x[b, ch, i*s:i*s+k, j*s:j*s+k]
                        out[b, ch, i, j] = np.max(region)
        return out
